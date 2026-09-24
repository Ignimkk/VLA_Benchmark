"""부호 교정이 숨길 수 있는 관통의 양을 잰다.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.live.measure_attached_overlap \\
        --out outputs/live_test/20260922_attached/overlap.json

## 왜 이것을 재나

cuRobo 경로에서 쥔 물체를 빼려면 부호를 양수로 강제해야 한다 (크기는 seed 제외가 맞게 주지만
부호가 질의 복셀의 TSDF 에서 오기 때문 — `builder_esdf.py:455-489`). 그 강제가 **정당한
경우**는 "그 자리에 있는 것이 쥔 물체뿐" 일 때다. 쥔 물체와 **실제 환경 표면이 같은 복셀을
공유**하면 그 관통을 숨긴다.

그래서 재는 것은 둘이다.

1. **복셀 공유** — 쥔 물체 복셀(팽창 포함) 중 환경 표면 점도 들어 있는 복셀 수. 계층마다.
2. **질의점에서 환경까지의 거리** — 각 질의점에서 **사과가 아닌** 관측 표면까지의 최근접 거리.
   이 값이 0 근처거나 음수인 점만 부호 강제가 숨길 수 있다.

참값은 **MuJoCo segmentation** 이다 — 관측 점마다 어느 body 에서 왔는지 안다. 필드는 익명이라
그 질문에 답할 수 없고, 한쪽만 보고 귀속하면 틀린다 (F13 이 그 사례).

## 두 배치를 본다

| 배치 | 무엇 | 왜 |
|---|---|---|
| `on_table` | 사과가 테이블에 놓인 채 | **상한**이다. 바닥면이 테이블과 붙어 있으니 공유가 최대다. 다만 파내기는 `attach()` 뒤에만 도므로 **운용 조건이 아니다** |
| `lifted_*` | 사과 점을 위로 옮긴 채 | 운용 조건에 가깝다. 쥔 동안 사과는 공중에 있고 손가락은 depth 에서 마스킹되어 필드에 없다 |
| `at_rim` | 사과 점을 바구니 테두리 옆으로 옮긴 채 | 담기 구간 — F18(목적지 마진)이 사는 자리이고 환경과 가장 가까워지는 실제 국면 |
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def _q(a, scale=1000.0) -> dict:
    a = np.asarray(a, float).reshape(-1)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "min_mm": float(a.min() * scale),
            "p05_mm": float(np.percentile(a, 5) * scale),
            "median_mm": float(np.median(a) * scale),
            "max_mm": float(a.max() * scale)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--dilate", type=int, default=1,
                    help="esdf.attached_dilate_voxels 와 같은 값")
    args = ap.parse_args()

    from scipy.spatial import cKDTree

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import (
        TRANSPORT_MODEL, TransportScene, gaussian_attention)
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.ag3s.types import CameraID, CameraObservation

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"stage": "attached_overlap", "seed": args.seed,
                    "dilate_voxels": args.dilate,
                    "source": "fresh_mujoco_scene", "used_saved_run": False,
                    "ground_truth": "MuJoCo body segmentation per camera pixel"}

    scene = TransportScene(TRANSPORT_MODEL, settle_steps=400,
                           height=args.height, width=args.width)
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
        if cs.target is None:
            raise RuntimeError(f"grounding 실패: {cs.status.value}")
        tgt = np.asarray(cs.target.points, np.float64).reshape(-1, 3)
        keys = np.floor(tgt / 0.010).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        attached = tgt[np.sort(keep)]

        # ── 관측 표면 점을 body 라벨과 함께 모은다 (MuJoCo 참값) ──
        body_names = frames[CAMS[0]].body_names
        apple_bid = scene.mujoco.mj_name2id(
            scene.model, scene.mujoco.mjtObj.mjOBJ_BODY, args.target)
        pts_all, lab_all = [], []
        for c in CAMS:
            f = frames[c]
            depth = np.asarray(f.depth, np.float64)
            K = np.asarray(f.camera_intrinsics, np.float64)
            T = np.asarray(f.T_base_cam, np.float64)
            mask = pipe._robot_mask_for(depth, K, T, f.robot_state)
            bid = np.asarray(f.body_ids, np.int64)
            v, u = np.nonzero((depth > 0.05) & (depth < 3.0)
                              & (~np.asarray(mask, bool) if mask is not None else True))
            if not v.size:
                continue
            z = depth[v, u]
            x = (u - K[0, 2]) / K[0, 0] * z
            y = (v - K[1, 2]) / K[1, 1] * z
            cam_pts = np.stack([x, y, z], axis=1)
            pts_all.append(cam_pts @ T[:3, :3].T + T[:3, 3])
            lab_all.append(bid[v, u])
        pts = np.concatenate(pts_all, axis=0)
        lab = np.concatenate(lab_all, axis=0)
        is_apple = lab == apple_bid
        env_pts = pts[~is_apple]
        report["observation"] = {
            "n_surface_points": int(len(pts)),
            "n_apple_points": int(is_apple.sum()),
            "n_env_points": int(len(env_pts)),
            "n_attached_query_points": int(len(attached)),
            "apple_body_id": int(apple_bid)}

        env_tree = cKDTree(env_pts)
        crate_bid = scene.mujoco.mj_name2id(
            scene.model, scene.mujoco.mjtObj.mjOBJ_BODY, "crate")
        crate_pts = pts[lab == crate_bid] if crate_bid >= 0 else np.zeros((0, 3))

        # ── 배치들 ──
        placements: dict[str, np.ndarray] = {"on_table": attached}
        for lift in (0.05, 0.10, 0.20):
            placements[f"lifted_{int(lift*1000)}mm"] = attached + np.array([0.0, 0.0, lift])
        if len(crate_pts):
            # 바구니 테두리 위 — 담기 국면. 테두리 최고점 위 30 mm 로 옮긴다.
            rim = crate_pts[np.argmax(crate_pts[:, 2])]
            shift = rim - attached.mean(axis=0) + np.array([0.0, 0.0, 0.030])
            placements["at_rim_30mm"] = attached + shift
            report["crate_rim_m"] = [float(v) for v in rim]

        cases = {}
        for name, apts in placements.items():
            # 1) 질의점에서 **환경** 표면까지의 최근접 거리
            d_env, _ = env_tree.query(apts, k=1)
            # 2) 복셀 공유 — 계층마다
            tiers = {}
            for vs in (0.020, 0.005):
                a_idx = np.floor(apts / vs).astype(np.int64)
                a_set = {tuple(v) for v in np.unique(a_idx, axis=0)}
                if args.dilate > 0:
                    grown = set()
                    offs = range(-args.dilate, args.dilate + 1)
                    for k in a_set:
                        for dx in offs:
                            for dy in offs:
                                for dz in offs:
                                    grown.add((k[0] + dx, k[1] + dy, k[2] + dz))
                    a_set = grown
                e_idx = np.floor(env_pts / vs).astype(np.int64)
                e_set = {tuple(v) for v in np.unique(e_idx, axis=0)}
                shared = a_set & e_set
                tiers[f"{int(vs*1000)}mm"] = {
                    "voxel_size_m": vs,
                    "n_attached_voxels": len(a_set),
                    "n_shared_with_env": len(shared),
                    "shared_fraction": (len(shared) / len(a_set)) if a_set else None}
            cases[name] = {
                "n_query_points": int(len(apts)),
                "centroid_m": [float(v) for v in apts.mean(axis=0)],
                "distance_to_env_surface": _q(d_env),
                "n_within_20mm_of_env": int((d_env < 0.020).sum()),
                "n_within_5mm_of_env": int((d_env < 0.005).sum()),
                "tiers": tiers}
        report["cases"] = cases

        # ── 판정에 쓸 요약 ──
        op = [k for k in cases if k.startswith("lifted") or k == "at_rim_30mm"]
        report["summary"] = {
            "upper_bound_on_table": {
                "shared_20mm": cases["on_table"]["tiers"]["20mm"]["n_shared_with_env"],
                "shared_20mm_fraction": cases["on_table"]["tiers"]["20mm"]["shared_fraction"],
                "min_dist_to_env_mm": cases["on_table"]["distance_to_env_surface"]["min_mm"]},
            "operating_cases": {k: {
                "shared_20mm": cases[k]["tiers"]["20mm"]["n_shared_with_env"],
                "shared_20mm_fraction": cases[k]["tiers"]["20mm"]["shared_fraction"],
                "shared_5mm": cases[k]["tiers"]["5mm"]["n_shared_with_env"],
                "min_dist_to_env_mm": cases[k]["distance_to_env_surface"]["min_mm"],
                "n_within_20mm": cases[k]["n_within_20mm_of_env"]} for k in op},
            "what_sign_forcing_can_hide":
                "부호 강제는 쥔 물체 복셀 안의 부호만 뒤집는다. 그 복셀에 환경 표면이 없으면 "
                "숨길 것이 없다. 위 shared_* 가 그 수이고, min_dist_to_env_mm 가 여유다"}
    finally:
        scene.close()

    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    o = report["observation"]
    print(f"  관측 표면 {o['n_surface_points']:,} 점 (사과 {o['n_apple_points']:,} · "
          f"환경 {o['n_env_points']:,})   쥔 물체 질의점 {o['n_attached_query_points']}")
    print(f"  {'배치':16s} {'환경까지 최소':>12} {'중앙':>9} | "
          f"{'20mm 복셀':>9} {'공유':>6} {'비율':>7} | {'5mm 공유':>9} | {'20mm 안 점':>10}")
    for name, c in report["cases"].items():
        t20 = c["tiers"]["20mm"]; t5 = c["tiers"]["5mm"]
        d = c["distance_to_env_surface"]
        print(f"  {name:16s} {d['min_mm']:>12.2f} {d['median_mm']:>9.2f} | "
              f"{t20['n_attached_voxels']:>9} {t20['n_shared_with_env']:>6} "
              f"{100*t20['shared_fraction']:>6.1f}% | {t5['n_shared_with_env']:>9} | "
              f"{c['n_within_20mm_of_env']:>10}")


if __name__ == "__main__":
    main()
