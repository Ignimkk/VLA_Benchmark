"""curobo 자신의 `create_xyzr_tensor` 로 좌표를 얻어 2계층 ESDF 를 검증한다 (C3 — 미세 창
경계에서 거리가 얼마나 튀나).

    PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-openpi-live/bin/python -m \\
        benchmark.ag3s.experiments.curobo.verify_two_tier \\
        --frame outputs/verify/R/rby1_frame_16d.npz

입력 프레임은 `curobo/export_frame.py` 가 만든다. **`--frame` 에 기본값을 두지 않는다** —
예전에는 `/tmp/rby1_frame.npz` 가 코드에 박혀 있었고 argparse 자체가 없었다. 그 파일은
2026-09-11 에 `run_0004`(14D)에서 구운 것인데, 2026-09-25 의 16D 재측정이 그것을 말없이
읽어 "16D 로 다시 쟀다" 는 결과를 냈다. 인자로 빼는 것만으로는 부족해서 기본값도 없앴다:
경로를 손으로 적게 하면 무엇을 먹이는지 적는 사람이 안다.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def grid_points_and_vals(vg):
    """curobo 의 생성기로 `(N,3)` 월드좌표와 `(N,)` 거리.

    인덱싱을 손으로 쓰지 않는 이유는 함정 2 다 — `esdf_origin` 은 격자 코너가 아니라
    **중심**이라 직접 계산하면 반 격자만큼 밀린다.
    """
    xyzr = vg.create_xyzr_tensor(transform_to_origin=True)
    pts = xyzr[:, :3].detach().cpu().numpy()
    vals = vg.feature_tensor.detach().float().reshape(-1).cpu().numpy()
    return pts, vals


def main() -> None:
    import torch
    from curobo._src.types.camera import CameraObservation
    from curobo._src.types.pose import Pose
    from curobo.perception import Mapper, MapperCfg

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frame", required=True,
                    help="`curobo/export_frame.py` 가 만든 npz. 기본값 없음 (위 머리말 참고)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--coarse", type=float, default=0.020)
    ap.add_argument("--fine", type=float, default=0.005)
    ap.add_argument("--tsdf-voxel", type=float, default=0.005)
    ap.add_argument("--table-top", type=float, default=0.823,
                    help="테이블 상판 높이 [m]. 근접 띠의 z 중앙값을 견줄 기준")
    ap.add_argument("--out-json", default=None,
                    help="찍은 수치를 JSON 으로도 남긴다 (규칙 A 의 sidecar)")
    args = ap.parse_args()

    frame_path = pathlib.Path(args.frame)
    if not frame_path.exists():
        raise SystemExit(f"{frame_path} 가 없다. curobo/export_frame.py 로 먼저 구워라:\n"
                         f"  MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work "
                         f".venv-openpi-live/bin/python -m "
                         f"benchmark.ag3s.experiments.curobo.export_frame "
                         f"--records <기록> --step <n> --out {frame_path}")
    d = np.load(frame_path)
    dev = args.device

    # 출처를 찍는다. 이 스크립트가 어느 기록의 프레임을 봤는지 출력만 보고 알 수 있어야 한다.
    from benchmark.ag3s.experiments.sources.policy_record import read_provenance
    stamp = read_provenance(d, where=str(frame_path))
    if stamp["stamped"]:
        print(f"프레임 출처: {stamp['record']}  (도장 {stamp['fingerprint']})")
    else:
        print(f"[경고] {frame_path} 에 출처 도장이 없다 (옛 형식) — 어느 기록에서 나온 "
              f"프레임인지 이 파일만으로는 알 수 없다. export_frame.py 로 다시 구우면 붙는다")

    tc = np.asarray(d["target_centroid"], np.float32)
    if not np.all(np.isfinite(tc)):
        raise SystemExit(
            f"{frame_path} 의 target_centroid 가 유한하지 않다 {tc.tolist()} — 이 프레임에서 "
            "grounding 이 서지 않았다는 뜻이다. 미세 계층은 target 중심에 놓이므로 "
            "2계층을 만들 수 없다. **코드 결함이 아니라 프레임의 성질이다** — "
            "target 이 서는 다른 --step 을 export_frame.py 로 구워라")

    h, w = np.asarray(d["depth_head"]).shape
    cfg = MapperCfg(extent_meters_xyz=(1.5, 1.8, 1.6), voxel_size=args.tsdf_voxel,
                    esdf_voxel_size=args.coarse,
                    grid_center=torch.tensor([0.45, 0.0, 0.8], device=dev,
                                             dtype=torch.float32),
                    truncation_distance=0.04, depth_minimum_distance=0.05,
                    depth_maximum_distance=3.0, image_height=h, image_width=w, device=dev)
    m = Mapper(cfg)
    for cid in ("head", "left_wrist", "right_wrist"):
        T = np.asarray(d[f"T_{cid}"], np.float32)
        m.integrate(CameraObservation(
            name=cid,
            depth_image=torch.as_tensor(d[f"depth_{cid}"], device=dev,
                                        dtype=torch.float32)[None],
            rgb_image=torch.zeros((1, h, w, 3), device=dev, dtype=torch.uint8),
            intrinsics=torch.as_tensor(d[f"K_{cid}"], device=dev, dtype=torch.float32)[None],
            pose=Pose.from_matrix(torch.as_tensor(T, device=dev, dtype=torch.float32)),
            depth_to_meter=1.0))

    report: dict = {"frame": str(frame_path), "provenance": stamp,
                    "table_top_m": args.table_top, "layers": {}}
    for label, kw, vs in ((f"거친 {args.coarse*1000:.0f} mm",
                           dict(esdf_voxel_size=args.coarse), args.coarse),
                          (f"미세 {args.fine*1000:.0f} mm (target 중심)",
                           dict(esdf_origin=torch.as_tensor(tc, device=dev,
                                                            dtype=torch.float32),
                                esdf_voxel_size=args.fine), args.fine)):
        vg = m.compute_esdf(**kw)
        pts, vals = grid_points_and_vals(vg)
        print(f"\n=== {label} ===")
        print(f"  pose(=중심?) {np.round(vg.pose[:3],3)}   dims {np.round(vg.dims,2)}")
        print(f"  좌표 범위  x[{pts[:,0].min():+.2f},{pts[:,0].max():+.2f}] "
              f"y[{pts[:,1].min():+.2f},{pts[:,1].max():+.2f}] "
              f"z[{pts[:,2].min():+.2f},{pts[:,2].max():+.2f}]")
        entry = {"voxel_size_m": float(vs),
                 "pose_xyz": np.asarray(vg.pose, np.float64).reshape(-1)[:3].tolist(),
                 "dims": np.asarray(vg.dims, np.float64).reshape(-1).tolist()}
        # 테이블 상판 근처에서 거리가 0 에 가까운 복셀의 z 분포
        sel = ((pts[:, 0] > 0.45) & (pts[:, 0] < 0.85)
               & (np.abs(pts[:, 1]) < 0.4) & (np.abs(vals) < vs))
        if sel.sum():
            z_med = float(np.median(pts[sel, 2]))
            print(f"  |d|<{vs*1000:.0f}mm 인 복셀의 z 중앙값 : {z_med:.3f} m "
                  f"(n={sel.sum():,})   <- 테이블 {args.table_top} 과 비교")
            entry["near_zero_band"] = {"z_median_m": z_med, "n": int(sel.sum())}
        k = int(np.argmin(np.linalg.norm(pts - tc, axis=1)))
        print(f"  target centroid 최근접 복셀 d = {vals[k]:+.3f} m  "
              f"(복셀 위치 {np.round(pts[k],3)})")
        entry["target_centroid_nearest"] = {
            "d_m": float(vals[k]), "voxel_xyz": pts[k].astype(float).tolist()}
        report["layers"][label] = entry

    if args.out_json:
        out = pathlib.Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
