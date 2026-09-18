"""롤아웃 프레임마다 cuRobo 2계층 ESDF 를 만들어 npz 로 내보낸다. **curobo venv 에서** 돌린다.

    # (1) ag3s venv — 기준선을 돌리면서 프레임을 덤프
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
      benchmark.trajopt.experiments.esdf_rollout --records run_0004 \\
      --attention attention_step1_run0004.npz --frames 15 --voxel 0.020 \\
      --esdf-margin 0.05 --support-surfaces field --constraint-links arms \\
      --dump-frames /tmp/rollout_frames.npz --out-json /tmp/base.json

    # (2) curobo venv — 필드 생성
    PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-curobo/bin/python -m \\
      benchmark.ag3s.experiments.curobo.build_rollout_fields \\
      --frames /tmp/rollout_frames.npz --out /tmp/rollout_fields.npz

    # (3) ag3s venv — 같은 롤아웃을 필드만 바꿔 다시
    ... --curobo-fields /tmp/rollout_fields.npz --out-json /tmp/curobo.json

**프레임 간 TSDF 를 누적한다.** AG3S 기준선의 `EsdfBuilder` 가 프레임마다 국소 갱신으로
누적하므로 (`esdf.py:EsdfBuilder.update`), 여기서도 `Mapper` 하나를 계속 쓴다. 프레임마다
새로 만들면 비교가 성립하지 않는다.

**마스킹된 depth 를 쓴다.** 덤프의 `mask_i` 는 AG3S 가 자기 ESDF 에 쓰는 것과 같은
`_robot_mask_for` 출력이다. 마스크 픽셀은 `0.0` 으로 두어 cuRobo 가 그 광선을 건너뛰게 한다
(= 자유가 아니라 미관측).
"""

import argparse
import time

import numpy as np
import torch

from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
from curobo.perception import Mapper, MapperCfg


def _eikonal_check(values: np.ndarray, voxel_size: float, label: str,
                   quiet: bool = False) -> bool:
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
    if not quiet or not ok:
        print(f"    [eikonal] {label}: |∇d| 중앙 {med:.3f}  "
              f"{'OK' if ok else '<<< 이상! 해상도 짝이 틀렸다'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", default="/tmp/rollout_frames.npz")
    ap.add_argument("--out", default="/tmp/rollout_fields.npz")
    ap.add_argument("--coarse", type=float, default=0.020)
    ap.add_argument("--fine", type=float, default=0.005)
    ap.add_argument("--tsdf-voxel", type=float, default=0.005)
    ap.add_argument("--raw", action="store_true", help="마스크를 무시한다 (대조군)")
    ap.add_argument("--no-fine", action="store_true")
    ap.add_argument("--check-all", action="store_true",
                    help="eikonal 검사를 모든 프레임에 (기본은 첫·끝만 — 시간 측정 보호)")
    args = ap.parse_args()

    blob = np.load(args.frames)
    n = int(blob["n_frames"])
    h, w = np.asarray(blob["depth_0"]).shape
    dev = "cuda:0"

    cfg = MapperCfg(
        extent_meters_xyz=(1.5, 1.8, 1.6), voxel_size=args.tsdf_voxel,
        esdf_voxel_size=args.coarse,
        grid_center=torch.tensor([0.45, 0.0, 0.8], device=dev, dtype=torch.float32),
        truncation_distance=0.04, depth_minimum_distance=0.05, depth_maximum_distance=3.0,
        image_height=h, image_width=w, device=dev)
    mapper = Mapper(cfg)

    out = {"n_frames": np.asarray(n)}
    t_int, t_esdf, t_xfer = [], [], []
    ok_all = True
    print(f"{n} 프레임  {h}x{w}  masked={not args.raw}")
    for i in range(n):
        depth = np.asarray(blob[f"depth_{i}"], np.float32)
        if not args.raw:
            depth = np.where(np.asarray(blob[f"mask_{i}"], bool), np.float32(0.0), depth)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        mapper.integrate(CameraObservation(
            name="head",
            depth_image=torch.as_tensor(depth, device=dev, dtype=torch.float32)[None],
            rgb_image=torch.zeros((1, h, w, 3), device=dev, dtype=torch.uint8),
            intrinsics=torch.as_tensor(np.asarray(blob[f"K_{i}"], np.float32),
                                       device=dev, dtype=torch.float32)[None],
            pose=Pose.from_matrix(torch.as_tensor(np.asarray(blob[f"T_{i}"], np.float32),
                                                  device=dev, dtype=torch.float32)),
            depth_to_meter=1.0))
        torch.cuda.synchronize(); t_int.append(time.perf_counter() - t0)

        centroid = np.asarray(blob[f"centroid_{i}"], np.float64)
        layers = [("coarse", dict(esdf_voxel_size=args.coarse))]
        if not args.no_fine and np.all(np.isfinite(centroid)):
            layers.append(("fine", dict(
                esdf_origin=torch.as_tensor(centroid.astype(np.float32),
                                            device=dev, dtype=torch.float32),
                esdf_voxel_size=args.fine)))

        # **`compute_esdf()` 의 `feature_tensor` 는 재사용되는 버퍼다.** 반환되는 `VoxelGrid`
        # 래퍼는 호출마다 새 객체이고 `voxel_size` 도 제 값을 유지하지만, 그 안의 텐서는 다음
        # 호출이 **덮어쓴다**. 두 계층을 다 만든 뒤에 읽으면 거친 계층 자리에 미세 계층 값이
        # 들어가고, 메타데이터는 거친 것이라 알아채기 어렵다 — 증상은 거친 계층의
        # `|∇d| = 0.25` (= 5 mm / 20 mm) 다. 그래서 **호출 직후 device 에서 복사**한다.
        grids = []
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for name, kw in layers:
            vg = mapper.compute_esdf(**kw)
            grids.append((name, vg.feature_tensor.detach().clone(),
                          float(vg.voxel_size),
                          np.asarray(vg.dims, np.float64).reshape(3),
                          np.asarray(vg.pose, np.float64).reshape(-1)[:3]))
        torch.cuda.synchronize(); t_esdf.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for name, feat, vs, dims, pose in grids:
            out[f"f{i}_{name}_values"] = feat.float().cpu().numpy()
            out[f"f{i}_{name}_origin"] = pose - 0.5 * dims + 0.5 * vs
            out[f"f{i}_{name}_voxel_size"] = np.asarray(vs)
        t_xfer.append(time.perf_counter() - t0)

        # C4 재발 방지. 128³ 에 `np.gradient` 라 CPU 로 100 ms 쯤 들고 위 시간 측정에
        # 잡음을 섞으므로, 기본은 첫·끝 프레임만 본다 (해상도 짝은 프레임마다 바뀌지 않는다).
        if args.check_all or i in (0, n - 1):
            for name, *_rest in grids:
                ok_all &= _eikonal_check(out[f"f{i}_{name}_values"],
                                         float(out[f"f{i}_{name}_voxel_size"]),
                                         f"frame {i} {name}")

        occ = mapper.extract_occupied_voxels()
        n_occ = int(occ.centers.shape[0]) if getattr(occ, "centers", None) is not None else -1
        out[f"f{i}_n_occupied"] = np.asarray(n_occ)
        # cuRobo 는 우리 `UNKNOWN` 에 해당하는 세 번째 상태를 내주지 않는다. 지어내지 않고
        # -1 로 둔다 — `esdf_rollout` 의 `esdf_unknown` 열은 이 실행에서 의미가 없다.
        out[f"f{i}_unknown_fraction"] = np.asarray(-1.0)
        if i % 5 == 0 or i == n - 1:
            print(f"  [{i:3d}] 적분 {t_int[-1]*1000:6.2f} ms  ESDF {t_esdf[-1]*1000:6.2f} ms  "
                  f"전송 {t_xfer[-1]*1000:5.2f} ms  "
                  f"점유 {n_occ:,}  계층 {len(layers)}"
                  f"{'  (target 없음 — 거친 계층만)' if len(layers) == 1 else ''}")

    if not ok_all:
        raise SystemExit("eikonal 검사 실패 — 저장하지 않았다. 발견 C4 를 볼 것")
    np.savez_compressed(args.out, **out)
    # 첫 프레임은 JIT 컴파일 + CUDA graph capture 라 버린다 (§6 함정 3).
    print(f"\nwrote {args.out}")
    if n > 1:
        print(f"  정상 상태(첫 프레임 제외)  적분 {np.mean(t_int[1:])*1000:.2f} ms  "
              f"ESDF {np.mean(t_esdf[1:])*1000:.2f} ms  "
              f"= 지각 합계 {(np.mean(t_int[1:])+np.mean(t_esdf[1:]))*1000:.2f} ms")
        print(f"  (npz 핸드오프 전용 비용  host 전송 {np.mean(t_xfer[1:])*1000:.2f} ms — "
              f"한 프로세스로 합치면 사라진다)")
        print(f"  첫 프레임(워밍업)          적분 {t_int[0]*1000:.1f} ms  ESDF {t_esdf[0]*1000:.1f} ms")


if __name__ == "__main__":
    main()
