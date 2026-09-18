"""cuRobo 로 2계층 ESDF 를 만들어 npz 로 내보낸다. **curobo venv 에서** 돌린다.

    PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-curobo/bin/python \\
        -m benchmark.ag3s.experiments.curobo.build_field

왜 npz 를 거치는가: `.venv-curobo` 에는 `mujoco`/`casadi`/`osqp` 가 없고 `.venv-ag3s` 에는
cuRobo 가 없다 (numpy 버전이 충돌해 환경을 둘로 나눴다 — `docs/AG3S_REVIEW_LOG.md` 의 D3).
그런데 **질의에는 cuRobo 가 필요 없다** — 거리 격자는 `(값, origin, voxel_size)` 로 완전히
기술되므로, 생산은 이쪽에서 하고 소비는 검증된 `.venv-ag3s` 에서 한다. 회귀 기준선을 그대로
비교할 수 있다는 것이 이 분리의 값어치다.

**마스킹된 depth 를 쓴다.** `export_frame.py` 가 AG3S 의 `_robot_mask_for` 로 만들어 둔
`depth_masked_*` 다. raw depth 를 넣으면 로봇이 자기 몸을 장애물로 보고 구 120 개 중 105 개가
자기 자신과 충돌한다고 나온다. `--raw` 로 그 대조군을 만들 수 있다.
"""

import argparse

import numpy as np
import torch

from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
from curobo.perception import Mapper, MapperCfg

IDS = ("head", "left_wrist", "right_wrist")


def _eikonal_check(values: np.ndarray, voxel_size: float, label: str) -> bool:
    """자유공간에서 `|∇d| ≈ 1` 인가. 아니면 해상도를 잘못 짝지은 것이다 (발견 C4).

    `0.25` 는 5 mm 간격 데이터를 20 mm 격자로 읽은 것 — `compute_esdf()` 의 재사용 버퍼를
    복사하지 않고 두 계층을 만든 뒤에 읽었을 때 나온다.
    """
    v = np.asarray(values, np.float64)
    g = np.gradient(v, float(voxel_size))
    n = np.sqrt(sum(x ** 2 for x in g))
    free = v > 0.03
    if not free.any():
        print(f"    [eikonal] {label}: 자유공간 표본 없음 — 검사 건너뜀")
        return True
    med = float(np.median(n[free]))
    ok = 0.9 <= med <= 1.1
    print(f"    [eikonal] {label}: |∇d| 중앙 {med:.3f}  {'OK' if ok else '<<< 이상! 해상도 짝이 틀렸다'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frame", default="/tmp/rby1_frame.npz")
    ap.add_argument("--out", default="/tmp/rby1_field.npz")
    ap.add_argument("--coarse", type=float, default=0.020, help="거친 계층 복셀 [m]")
    ap.add_argument("--fine", type=float, default=0.005, help="미세 계층 복셀 [m]")
    ap.add_argument("--tsdf-voxel", type=float, default=0.005)
    ap.add_argument("--raw", action="store_true",
                    help="마스킹 안 된 depth 를 쓴다 (대조군 — 로봇이 자기 몸을 본다)")
    ap.add_argument("--no-fine", action="store_true", help="거친 계층만 만든다")
    args = ap.parse_args()

    d = np.load(args.frame)
    key = "depth_{}" if args.raw else "depth_masked_{}"
    if key.format(IDS[0]) not in d:
        raise SystemExit(
            f"{args.frame} 에 {key.format(IDS[0])} 가 없습니다 — export_frame.py 를 다시 돌리세요")

    dev = "cuda:0"
    tc = np.asarray(d["target_centroid"], np.float32)
    cfg = MapperCfg(
        extent_meters_xyz=(1.5, 1.8, 1.6), voxel_size=args.tsdf_voxel,
        esdf_voxel_size=args.coarse,
        grid_center=torch.tensor([0.45, 0.0, 0.8], device=dev, dtype=torch.float32),
        truncation_distance=0.04, depth_minimum_distance=0.05, depth_maximum_distance=3.0,
        image_height=480, image_width=640, device=dev)
    mapper = Mapper(cfg)

    for cid in IDS:
        depth = np.asarray(d[key.format(cid)], np.float32)
        mapper.integrate(CameraObservation(
            name=cid,
            depth_image=torch.as_tensor(depth, device=dev, dtype=torch.float32)[None],
            rgb_image=torch.zeros((1, 480, 640, 3), device=dev, dtype=torch.uint8),
            intrinsics=torch.as_tensor(np.asarray(d[f"K_{cid}"], np.float32),
                                       device=dev, dtype=torch.float32)[None],
            pose=Pose.from_matrix(torch.as_tensor(np.asarray(d[f"T_{cid}"], np.float32),
                                                  device=dev, dtype=torch.float32)),
            depth_to_meter=1.0))

    out = {"target_centroid": np.asarray(tc, np.float64),
           "sphere_centers": np.asarray(d["sphere_centers"], np.float64),
           "sphere_radii": np.asarray(d["sphere_radii"], np.float64),
           "masked": np.asarray(not args.raw)}

    ok_all = True
    layers = [("coarse", dict(esdf_voxel_size=args.coarse))]
    if not args.no_fine:
        layers.append(("fine", dict(
            esdf_origin=torch.as_tensor(tc, device=dev, dtype=torch.float32),
            esdf_voxel_size=args.fine)))

    for name, kw in layers:
        vg = mapper.compute_esdf(**kw)
        vs = float(vg.voxel_size)
        dims = np.asarray(vg.dims, np.float64).reshape(3)
        pose = np.asarray(vg.pose, np.float64).reshape(-1)[:3]
        # `esdf_origin` 은 격자 코너가 아니라 **중심**이다. 여기서 반 칸을 더해 첫 복셀
        # *중심*으로 바꾼다 — 우리 `VoxelGrid.origin` 규약과 같아진다.
        origin = pose - 0.5 * dims + 0.5 * vs
        field = vg.feature_tensor.detach().float().cpu().numpy()
        out[f"{name}_values"] = field.astype(np.float32)
        out[f"{name}_origin"] = origin
        out[f"{name}_voxel_size"] = np.asarray(vs)
        print(f"{name:>7}  {vs*1000:5.1f} mm  {field.shape}  "
              f"origin {np.round(origin, 3)}  "
              f"값 [{field.min():+.3f}, {field.max():+.3f}]  "
              f"음수 {100.0*(field < 0).mean():.3f} %")
        ok_all &= _eikonal_check(field, vs, name)

    if not ok_all:
        raise SystemExit("eikonal 검사 실패 — 필드를 저장하지 않았다. 발견 C4 를 볼 것")
    np.savez(args.out, **out)
    print(f"\nwrote {args.out}  (masked={not args.raw})")


if __name__ == "__main__":
    main()
