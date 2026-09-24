"""부호 교정 문턱을 쓸어 정한다 — 추측하지 않는다 (F14 의 교훈).

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work XLA_PYTHON_CLIENT_PREALLOCATE=false \\
      /mnt/dev/work/.venv-openpi-live/bin/python -m \\
      benchmark.ag3s.experiments.live.sweep_attached_threshold \\
      --out outputs/live_test/20260922_attached/threshold_sweep.json

## 무엇을 가르나

문턱이 크면 **공유로 판정하는 복셀이 많아진다** — 부호를 안 고치므로 관통을 숨기지 않지만
쥔 물체가 자기 자신에게 남는 **잔여 행**이 늘어난다.
문턱이 작으면 그 반대다 — 잔여 행이 줄지만 실제 관통을 숨길 수 있다.

그래서 두 수치를 문턱마다 함께 본다.

| 수치 | 무엇 | 어느 방향이 위험한가 |
|---|---|---|
| **운반 구간의 잔여 음수** | 들어올림 100·200 mm 에서 남은 음수 질의점 수 | 많으면 **못 푸는 행**이 생겨 영구 hold |
| **공유 판정 복셀 수** | 부호를 안 고친 복셀 | 0 이면 판정이 무력해진 것 — 숨김 위험 |

판정 기준: **운반 구간 잔여 음수 0** 을 만족하는 문턱 중 **가장 큰 것**을 고른다. 크면 클수록
공유를 더 많이 잡아 숨김 위험이 작아지므로, 운용을 깨지 않는 한 큰 쪽이 안전하다.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0))
    args = ap.parse_args()

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import (
        TRANSPORT_MODEL, TransportScene, gaussian_attention)
    from benchmark.ag3s.fields.curobo_builder import CuroboFieldBuilder
    from benchmark.ag3s.fields.esdf import CameraDepth
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.ag3s.types import CameraID, CameraObservation

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"stage": "attached_threshold_sweep", "seed": args.seed,
                    "thresholds_voxels": list(args.thresholds),
                    "source": "fresh_mujoco_scene", "used_saved_run": False}

    scene = TransportScene(TRANSPORT_MODEL, settle_steps=400, height=480, width=640)
    try:
        rng = np.random.default_rng(args.seed)
        qadr = scene._qadr[f"{args.target}_free"]
        scene.data.qpos[qadr:qadr + 2] += rng.uniform(-0.012, 0.012, size=2)
        scene.mujoco.mj_forward(scene.model, scene.data)
        truth = scene.body_position_in_base(args.target)
        filter_robot = build_robot_model(scene)
        constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
        frames = {c: scene.capture(c) for c in CAMS}
        ids = {"zed_left": CameraID.HEAD, "wrist_cam_l": CameraID.LEFT_WRIST,
               "wrist_cam_r": CameraID.RIGHT_WRIST}
        obs = [CameraObservation(
            camera_id=ids[c], depth=frames[c].depth,
            camera_intrinsics=frames[c].camera_intrinsics,
            T_base_cam=frames[c].T_base_cam, robot_state=frames[c].robot_state,
            timestamp=0.0, state_timestamp=0.0,
            attention_map=gaussian_attention(frames[c], truth), image_hw=frames[c].hw)
            for c in CAMS]
        pipe = AG3S(AG3SConfig.from_dict({"collision_backend": "primitive",
                                          "pointcloud": {"range_max": 2.0}}),
                    robot_model=filter_robot, constraint_robot_model=constraint_robot)
        cs, _ = pipe.process_multi_debug(obs, phase="approach",
                                         active_manipulators=["left"])
        tgt = np.asarray(cs.target.points, np.float64).reshape(-1, 3)
        keys = np.floor(tgt / 0.010).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        attached = tgt[np.sort(keep)]

        cams = []
        for c in CAMS:
            f = frames[c]
            depth = np.asarray(f.depth, np.float64)
            K = np.asarray(f.camera_intrinsics, np.float64)
            T = np.asarray(f.T_base_cam, np.float64)
            cams.append(CameraDepth(c, depth, K, T,
                                    robot_mask=pipe._robot_mask_for(depth, K, T,
                                                                    f.robot_state)))

        placements = {"on_table": attached}
        for lift in (0.05, 0.10, 0.20):
            placements[f"lifted_{int(lift*1000)}mm"] = attached + np.array([0, 0, lift])

        rows = []
        for thr in args.thresholds:
            cfg = AG3SConfig.from_dict({
                "collision_backend": "esdf", "pointcloud": {"range_max": 2.0},
                "esdf": {"voxel_size": 0.020, "max_distance": 0.4,
                         "exclude_support_surfaces": False, "backend": "curobo",
                         "fine_voxel_size": 0.005, "tsdf_voxel_size": 0.005,
                         "attached_sign_threshold_voxels": thr}}).esdf
            per = {}
            for pname, apts in placements.items():
                b = CuroboFieldBuilder(cfg)
                field = b.update(cams, target_points=tgt, exclude_target=False,
                                 attached_points=apts, observed_at=0.0)
                d = np.asarray(field.distance(apts), np.float64)
                sc = field.stats.get("attached_sign_correction") or []
                per[pname] = {
                    "n_negative": int((d < 0).sum()),
                    "min_mm": float(d.min() * 1000),
                    "median_mm": float(np.median(d) * 1000),
                    "n_sign_forced": sum(t.get("n_sign_forced", 0) for t in sc),
                    "n_shared_left_alone": sum(t.get("n_shared_left_alone", 0) for t in sc),
                    "per_tier": sc}
            transport = ["lifted_100mm", "lifted_200mm"]
            rows.append({
                "threshold_voxels": thr,
                "placements": per,
                "transport_residual_negatives": sum(per[k]["n_negative"] for k in transport),
                "shared_voxels_on_table": per["on_table"]["n_shared_left_alone"],
                "shared_voxels_lifted_50mm": per["lifted_50mm"]["n_shared_left_alone"]})
        report["sweep"] = rows

        ok = [r for r in rows if r["transport_residual_negatives"] == 0]
        chosen = max(ok, key=lambda r: r["threshold_voxels"]) if ok else None
        report["choice"] = {
            "rule": "운반 구간(들어올림 100·200 mm) 잔여 음수가 0 인 문턱 중 가장 큰 값. "
                    "크면 공유를 더 많이 잡아 숨김 위험이 작아진다",
            "candidates_passing": [r["threshold_voxels"] for r in ok],
            "chosen_threshold_voxels": (None if chosen is None
                                        else chosen["threshold_voxels"])}
    finally:
        scene.close()

    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    print(f"  {'문턱':>6} | {'운반 잔여음수':>12} | "
          f"{'on_table 공유':>13} {'lifted50 공유':>13} | "
          f"{'on_table 최소':>13} {'lifted50 최소':>13} {'lifted100 최소':>14}")
    for r in report["sweep"]:
        p_ = r["placements"]
        print(f"  {r['threshold_voxels']:>6.1f} | {r['transport_residual_negatives']:>12} | "
              f"{r['shared_voxels_on_table']:>13} {r['shared_voxels_lifted_50mm']:>13} | "
              f"{p_['on_table']['min_mm']:>13.2f} {p_['lifted_50mm']['min_mm']:>13.2f} "
              f"{p_['lifted_100mm']['min_mm']:>14.2f}")
    print(f"  고른 값: {report['choice']['chosen_threshold_voxels']} 복셀 "
          f"(통과 후보 {report['choice']['candidates_passing']})")


if __name__ == "__main__":
    main()
