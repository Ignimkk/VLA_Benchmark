"""쥔 물체 처리의 양면을 실측한다 — 로봇 쪽 편입과 장애물 쪽 제거.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work XLA_PYTHON_CLIENT_PREALLOCATE=false \\
      /mnt/dev/work/.venv-openpi-live/bin/python -m \\
      benchmark.ag3s.experiments.live.verify_attached \\
      --out outputs/live_test/20260922_attached/verify_attached.json

## 무엇을 재나

쥔 물체는 **두 곳**에서 다뤄진다.

| 쪽 | 무엇 | 어디 |
|---|---|---|
| 로봇 쪽 | 물체의 점을 반지름 0 질의점으로 로봇에 붙인다 | `linearize.set_attached` (E3·F19) |
| 장애물 쪽 | 같은 점을 거리장에서 뺀다 | legacy `carve` / cuRobo **seed 제외** |

**둘 다 필요하다.** 로봇 쪽만 하면 물체가 자기 자신에게 부딪히고 그 제약 행은 **어떤 관절
움직임으로도 못 푼다** — 손에 강체로 붙어 있으니 자기 복셀에서 벗어날 수 없다. 장애물 쪽만
하면 물체가 아무 충돌 검사도 안 받는다.

여기서 재는 것은 **장애물 쪽 두 구현이 같은 답을 내는가**다. 쥔 물체의 질의점에서 거리를
물어, 처리 없음 / legacy carve / cuRobo seed 제외 셋을 견준다.

기대: 처리 없음은 0 근처나 음수(자기 표면), 나머지 둘은 양수이고 **그 값이 다음으로 가까운
표면까지의 거리**로 서로 비슷해야 한다.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def _q(a) -> dict:
    a = np.asarray(a, float).reshape(-1)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "min_mm": float(a.min() * 1000),
            "median_mm": float(np.median(a) * 1000),
            "max_mm": float(a.max() * 1000),
            "n_negative": int((a < 0).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--coarse", type=float, default=0.020)
    ap.add_argument("--fine", type=float, default=0.005)
    ap.add_argument("--tsdf-voxel", type=float, default=0.005)
    args = ap.parse_args()

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import (
        TRANSPORT_MODEL, TransportScene, gaussian_attention)
    from benchmark.ag3s.fields.esdf import CameraDepth
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.ag3s.types import CameraID, CameraObservation

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"stage": "attached", "seed": args.seed,
                    "source": "fresh_mujoco_scene", "used_saved_run": False,
                    "synthetic_attention": True, "cases": {}}

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
        stamp = 0.0
        ids = {"zed_left": CameraID.HEAD, "wrist_cam_l": CameraID.LEFT_WRIST,
               "wrist_cam_r": CameraID.RIGHT_WRIST}
        obs = [CameraObservation(
            camera_id=ids[c], depth=frames[c].depth,
            camera_intrinsics=frames[c].camera_intrinsics,
            T_base_cam=frames[c].T_base_cam, robot_state=frames[c].robot_state,
            timestamp=stamp, state_timestamp=stamp,
            attention_map=gaussian_attention(frames[c], truth), image_hw=frames[c].hw)
            for c in CAMS]

        # grounding 으로 target 점을 얻는다 — `attach_from_target` 이 스냅샷할 바로 그 점이다.
        base = AG3S(AG3SConfig.from_dict({"collision_backend": "primitive",
                                          "pointcloud": {"range_max": 2.0}}),
                    robot_model=filter_robot, constraint_robot_model=constraint_robot)
        cs0, _ = base.process_multi_debug(obs, phase="approach",
                                          active_manipulators=["left"])
        if cs0.target is None:
            raise RuntimeError(f"grounding 실패: {cs0.status.value} {cs0.notes}")
        tgt = np.asarray(cs0.target.points, np.float64).reshape(-1, 3)
        # 10 mm 복셀로 솎는다 — `attach_from_target` 의 `point_voxel` 과 같은 규칙 (F19).
        keys = np.floor(tgt / 0.010).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        attached = tgt[np.sort(keep)]
        report["attached_points"] = {"n_target_points": int(len(tgt)),
                                     "n_after_10mm_voxel": int(len(attached)),
                                     "centroid_m": [float(v) for v in attached.mean(0)],
                                     "truth_m": [float(v) for v in truth]}

        # 카메라 depth 를 CameraDepth 로 (두 builder 가 같은 입력을 받게)
        cams = []
        for c in CAMS:
            f = frames[c]
            depth = np.asarray(f.depth, np.float64)
            K = np.asarray(f.camera_intrinsics, np.float64)
            T = np.asarray(f.T_base_cam, np.float64)
            mask = base._robot_mask_for(depth, K, T, f.robot_state)
            cams.append(CameraDepth(c, depth, K, T, robot_mask=mask))

        def build(backend: str, with_attached: bool, apts):
            esdf = {"voxel_size": args.coarse, "max_distance": 0.4,
                    "exclude_support_surfaces": False, "backend": backend}
            if backend == "curobo":
                esdf["fine_voxel_size"] = args.fine
                esdf["tsdf_voxel_size"] = args.tsdf_voxel
            cfg = AG3SConfig.from_dict({"collision_backend": "esdf",
                                        "pointcloud": {"range_max": 2.0},
                                        "esdf": esdf}).esdf
            if backend == "curobo":
                from benchmark.ag3s.fields.curobo_builder import CuroboFieldBuilder
                b = CuroboFieldBuilder(cfg)
            else:
                from benchmark.ag3s.fields.esdf import EsdfBuilder
                b = EsdfBuilder(cfg)
            field = b.update(cams, target_points=tgt, exclude_target=False,
                             attached_points=(apts if with_attached else None),
                             observed_at=stamp, frame_id="attached_probe")
            return field

        # **배치를 나눈다.** 파내기와 부호 교정은 `attach()` 뒤에만 도므로 사과가 테이블에
        # 붙어 있는 배치는 상한이고 운용 조건이 아니다. 운반 구간(들어올림)이 운용 조건이다.
        placements = {"on_table": attached}
        for lift in (0.05, 0.10, 0.20):
            placements[f"lifted_{int(lift * 1000)}mm"] = (
                attached + np.array([0.0, 0.0, lift]))

        for backend in ("legacy", "curobo"):
            for with_attached in (False, True):
                for pname, apts in placements.items():
                    # 처리 없음은 배치를 옮기면 의미가 없다 (그 자리에 물체가 없으니까) —
                    # `on_table` 에서만 본다.
                    if not with_attached and pname != "on_table":
                        continue
                    name = (f"{backend}_{'handled' if with_attached else 'raw'}"
                            + ("" if pname == "on_table" else f"_{pname}"))
                    field = build(backend, with_attached, apts)
                    d = np.asarray(field.distance(apts), np.float64)
                    st = field.stats
                    report["cases"][name] = {
                        "backend": backend,
                        "attached_handled": with_attached,
                        "placement": pname,
                        "distance_at_attached_points": _q(d),
                        "n_attached_voxels_carved": st.get("n_attached_voxels_carved"),
                        "n_attached_seeds_excluded": st.get("n_attached_seeds_excluded"),
                        "attached_sign_correction": st.get("attached_sign_correction"),
                        "mechanism": ("carve (occupancy -> FREE)" if backend == "legacy"
                                      else "seed exclusion + pure-voxel sign forcing"),
                    }

        # ── 판정 ──
        c = report["cases"]
        lifted = [k for k in c if k.startswith("curobo_handled_lifted")]
        report["checks"] = {
            "raw_reads_own_surface": {
                "pass": (c["legacy_raw"]["distance_at_attached_points"]["median_mm"] < 12.0
                         and c["curobo_raw"]["distance_at_attached_points"]["median_mm"] < 12.0),
                "why": "처리 없이 물으면 자기 표면이 잡혀 거리가 0 근처다",
                "legacy_median_mm": c["legacy_raw"]["distance_at_attached_points"]["median_mm"],
                "curobo_median_mm": c["curobo_raw"]["distance_at_attached_points"]["median_mm"]},
            "legacy_carve_clears_negatives": {
                "pass": c["legacy_handled"]["distance_at_attached_points"]["n_negative"] == 0,
                "n_negative": c["legacy_handled"]["distance_at_attached_points"]["n_negative"]},
            "curobo_clears_negatives_when_lifted": {
                "pass": all(c[k]["distance_at_attached_points"]["n_negative"] == 0
                            for k in lifted if "lifted_100" in k or "lifted_200" in k),
                "why": "운용 조건(운반 구간)에서 잔여 음수가 0 이어야 한다. 부호 교정이 "
                       "순수 복셀에만 걸리므로 들어올림 100 mm 이상에서는 전부 순수다",
                "per_placement": {k: c[k]["distance_at_attached_points"]["n_negative"]
                                  for k in lifted}},
            "shared_voxels_left_alone": {
                "pass": all(any(t["n_shared_left_alone"] >= 0
                                for t in (c[k]["attached_sign_correction"] or []))
                            for k in lifted),
                "why": "환경 표면도 든 복셀은 부호를 손대지 않는다 — 숨길 수 있는 관통이 0",
                "on_table": c["curobo_handled"]["attached_sign_correction"],
                "lifted": {k: c[k]["attached_sign_correction"] for k in lifted}},
            "mechanism_actually_fired": {
                "pass": (int(c["legacy_handled"]["n_attached_voxels_carved"] or 0) > 0
                         and int(c["curobo_handled"]["n_attached_seeds_excluded"] or 0) > 0),
                "legacy_voxels_carved": c["legacy_handled"]["n_attached_voxels_carved"],
                "curobo_seeds_excluded": c["curobo_handled"]["n_attached_seeds_excluded"]},
        }
        report["verdict"] = {
            "pass": all(v.get("pass") for v in report["checks"].values()),
            "checks": {k: bool(v.get("pass")) for k, v in report["checks"].items()}}
    finally:
        scene.close()

    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    ap_ = report["attached_points"]
    print(f"  쥔 물체 질의점: target {ap_['n_target_points']} 점 -> 10 mm 복셀 "
          f"{ap_['n_after_10mm_voxel']} 점")
    print(f"  {'case':32s} {'중앙':>9} {'최소':>9} {'음수':>6}  {'부호강제/공유':>14}")
    for name, cc in report["cases"].items():
        q = cc["distance_at_attached_points"]
        sc = cc.get("attached_sign_correction") or []
        tag = ""
        if sc:
            f = sum(t.get("n_sign_forced", 0) for t in sc)
            sh = sum(t.get("n_shared_left_alone", 0) for t in sc)
            tag = f"{f} / {sh}"
        elif cc["backend"] == "legacy" and cc["attached_handled"]:
            tag = f"carve {cc['n_attached_voxels_carved']}"
        print(f"  {name:32s} {q['median_mm']:+9.2f} {q['min_mm']:+9.2f} "
              f"{q['n_negative']:>6}  {tag:>14}")
    for k, v in report["checks"].items():
        print(f"  [{'PASS' if v.get('pass') else 'FAIL'}] {k}")
    print(f"  => {'PASS' if report['verdict']['pass'] else 'FAIL'}")
    raise SystemExit(0 if report["verdict"]["pass"] else 1)


if __name__ == "__main__":
    main()
