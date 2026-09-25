"""부호 교정 문턱을 쓸어 정한다 — 추측하지 않는다 (F14 의 교훈).

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work XLA_PYTHON_CLIENT_PREALLOCATE=false \\
      /mnt/dev/work/.venv-openpi-live/bin/python -m \\
      benchmark.ag3s.experiments.live.sweep_attached_threshold \\
      --records outputs/live_test/20260924_long16d/run_0000 \\
      --out outputs/verify/R/R3_threshold_sweep_16d.json

## 2026-09-25 에 바뀐 것 — **기록의 실제 파지 구간을 먹는다**

예전에는 `TransportScene(settle_steps=400)` 으로 **새 씬을 만들고** 탁자 위 target 점을
합성으로 50·100·200 mm 들어올려 "운반" 을 흉내 냈다. 보고서에도
`used_saved_run: false` 라고 적혀 있었다. 그래서 그 수치는 어느 정책 모델로 재든 같았다 —
정책이 그 경로에 들어올 자리가 없었기 때문이다.

지금은 16D 기록의 **실제 파지 구간**을 읽는다. 운반은 합성 들어올림이 아니라 로봇이 정말로
사과를 들고 움직인 프레임들이다.

**표본이 작다.** 쓸 수 있는 파지 프레임은 이 기록에서 **여섯**뿐이다 — `wrist_cam_l` 만
사과를 보고(`zed_left`·`wrist_cam_r` 는 50 프레임 내내 사과 픽셀 0), 그중 호출 36~43 여덟
프레임은 그리퍼가 완전히 가린다. 그래서 32~35 와 44~45 가 남는다. **이 수가 산출물에
그대로 실린다** (`n_grasp_frames`, `frames`) — 여섯으로 고른 문턱임을 읽는 쪽이 알아야 한다.

## 무엇을 가르나

문턱이 크면 **공유로 판정하는 복셀이 많아진다** — 부호를 안 고치므로 관통을 숨기지 않지만
쥔 물체가 자기 자신에게 남는 **잔여 행**이 늘어난다.
문턱이 작으면 그 반대다 — 잔여 행이 줄지만 실제 관통을 숨길 수 있다.

| 수치 | 무엇 | 어느 방향이 위험한가 |
|---|---|---|
| **운반 구간의 잔여 음수** | 파지 프레임에서 남은 음수 질의점 수 | 많으면 **못 푸는 행**이 생겨 영구 hold |
| **공유 판정 복셀 수** | 부호를 안 고친 복셀 | 0 이면 판정이 무력해진 것 — 숨김 위험 |

판정 기준: **운반 구간 잔여 음수 0** 을 만족하는 문턱 중 **가장 큰 것**을 고른다. 크면 클수록
공유를 더 많이 잡아 숨김 위험이 작아지므로, 운용을 깨지 않는 한 큰 쪽이 안전하다.

**후보 목록의 상한이 답이 되면 그것은 측정이 아니다.** 1 차가 고른 3.0 이 후보 목록의 마지막
값이었다 — 잘린 것인지 정말 거기가 끝인지 알 수 없다. 그래서 기본 후보를 넓혔고, 고른 값이
상한이면 산출물에 `chosen_is_top_of_list: true` 로 **경고를 박는다**.

## 용어 (규칙 C)

*순수 복셀* = 쥔 물체 말고는 아무것도 없는 복셀. *공유 복셀* = 쥔 물체와 다른 표면이 같이
들어간 복셀 — 부호를 고치면 실제 관통을 숨기므로 그대로 둔다. *잔여 음수* = 부호 교정 뒤에도
음수로 남은 쥔 물체 질의점.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def main() -> None:
    import mujoco

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.reports.attention_report import target_from_prompt
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.policy_record import (
        load_run, phase_boundaries_for_path, phase_for, pose_scene, replay_scene)
    from benchmark.ag3s.experiments.studies.n1_self_collision_clearance import backproject
    from benchmark.ag3s.fields.curobo_builder import CuroboFieldBuilder
    from benchmark.ag3s.fields.esdf import CameraDepth
    from benchmark.ag3s.runtime.pipeline import AG3S

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", required=True,
                    help="16D 기록. 파지 구간을 여기서 읽는다 (예전에는 씬을 새로 만들었다)")
    ap.add_argument("--held", default=None,
                    help="쥔 물체의 MuJoCo body 이름. 기본은 prompt 에서 뽑는다")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0),
                    help="복셀 단위 후보. **상한을 넓혀 두었다** — 1 차가 고른 3.0 이 "
                         "후보 목록의 마지막 값이라 잘린 값인지 측정인지 알 수 없었다")
    ap.add_argument("--min-points", type=int, default=8,
                    help="이보다 점이 적은 프레임은 쓰지 않는다 (그리퍼가 물체를 가린 구간)")
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--fine-voxel", type=float, default=0.005)
    ap.add_argument("--tsdf-voxel", type=float, default=0.005)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    run = load_run(args.records)
    held = args.held or target_from_prompt(run.prompt)
    boundaries, phase_evidence = phase_boundaries_for_path(args.records)

    scene = replay_scene(run)
    report: dict = {
        "stage": "attached_threshold_sweep",
        "source": "recorded_grasp",          # 예전에는 "fresh_mujoco_scene"
        "used_saved_run": True,
        "records": str(args.records),
        "held": held,
        "policy_model": run.meta.get("policy_model"),
        "phase_boundaries": phase_evidence,
        "thresholds_voxels": list(args.thresholds),
    }
    try:
        filter_robot = build_robot_model(scene)
        constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
        pipe = AG3S(AG3SConfig.from_dict({"collision_backend": "primitive",
                                          "pointcloud": {"range_max": 2.0}}),
                    robot_model=filter_robot, constraint_robot_model=constraint_robot)
        held_body = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, held)
        if held_body < 0:
            raise SystemExit(f"body {held!r} 가 모델에 없다 — --held 로 이름을 직접 주라")

        # ---------- 쓸 수 있는 파지 프레임을 **세어서** 정한다 ----------
        # 범위를 손으로 박지 않는다. 그리퍼가 물체를 가리는 구간은 기록마다 다르고,
        # 박아 두면 다른 기록에서 조용히 틀린 구간을 잰다.
        frames_used, skipped = [], []
        for i, step in enumerate(run):
            if phase_for(step.t_step, boundaries) != "grasp":
                continue
            pose_scene(scene, step)
            caps = {c: scene.capture(c) for c in CAMS}
            per_cam = {c: int((np.asarray(f.body_ids) == held_body).sum())
                       for c, f in caps.items()}
            pts = np.vstack([
                backproject(np.asarray(f.depth, np.float64),
                            np.asarray(f.camera_intrinsics, np.float64),
                            np.asarray(f.T_base_cam, np.float64),
                            np.asarray(f.body_ids) == held_body)
                for f in caps.values()])
            if len(pts) < args.min_points:
                skipped.append({"i": i, "t_step": int(step.t_step),
                                "n_points": int(len(pts)), "per_camera_px": per_cam})
                continue
            cams = []
            for c, f in caps.items():
                depth = np.asarray(f.depth, np.float64)
                K = np.asarray(f.camera_intrinsics, np.float64)
                T = np.asarray(f.T_base_cam, np.float64)
                cams.append(CameraDepth(c, depth, K, T,
                                        robot_mask=pipe._robot_mask_for(depth, K, T,
                                                                        f.robot_state)))
            # 10 mm 복셀로 솎는다 — `attach_from_target` 의 `point_voxel` 과 같은 규칙 (F19).
            keys = np.floor(pts / 0.010).astype(np.int64)
            _, keep = np.unique(keys, axis=0, return_index=True)
            frames_used.append({"i": i, "t_step": int(step.t_step),
                                "n_points": int(len(pts)), "per_camera_px": per_cam,
                                "attached": pts[np.sort(keep)], "cams": cams})

        report["n_grasp_frames"] = len(frames_used)
        report["frames"] = [{k: v for k, v in f.items() if k not in ("attached", "cams")}
                            for f in frames_used]
        report["frames_skipped"] = skipped
        report["sample_note"] = (
            f"표본은 파지 프레임 {len(frames_used)} 개다. 가림({len(skipped)} 프레임)은 "
            f"그리퍼가 물체를 덮는 구간이고, zed_left·wrist_cam_r 은 이 기록에서 사과를 "
            f"한 번도 보지 못한다 — 점은 사실상 wrist_cam_l 에서만 온다.")
        print(f"파지 프레임 {len(frames_used)} 개 사용, {len(skipped)} 개 건너뜀")
        for f in frames_used:
            print(f"  [{f['i']:>2}] t={f['t_step']:>4}  점 {f['n_points']:>6}  {f['per_camera_px']}")
        for f in skipped:
            print(f"  [{f['i']:>2}] t={f['t_step']:>4}  점 {f['n_points']:>6}  건너뜀 "
                  f"{f['per_camera_px']}")
        if not frames_used:
            raise SystemExit(
                "쓸 수 있는 파지 프레임이 없다. 이 기록에 파지가 없거나 물체가 내내 가려졌다. "
                f"단계 경계 {tuple(boundaries)} 와 위 per_camera_px 를 볼 것.")

        # ---------- 문턱 쓸기 ----------
        rows = []
        for thr in args.thresholds:
            cfg = AG3SConfig.from_dict({
                "collision_backend": "esdf", "pointcloud": {"range_max": 2.0},
                "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                         "exclude_support_surfaces": False, "backend": "curobo",
                         "fine_voxel_size": args.fine_voxel,
                         "tsdf_voxel_size": args.tsdf_voxel,
                         "attached_sign_threshold_voxels": thr}}).esdf
            per_frame = []
            for f in frames_used:
                apts = f["attached"]
                b = CuroboFieldBuilder(cfg)
                field = b.update(f["cams"], target_points=apts, exclude_target=False,
                                 attached_points=apts, observed_at=0.0)
                d = np.asarray(field.distance(apts), np.float64)
                sc = field.stats.get("attached_sign_correction") or []
                per_frame.append({
                    "i": f["i"], "t_step": f["t_step"],
                    "n_negative": int((d < 0).sum()),
                    "min_mm": float(d.min() * 1000),
                    "median_mm": float(np.median(d) * 1000),
                    "n_sign_forced": sum(t.get("n_sign_forced", 0) for t in sc),
                    "n_shared_left_alone": sum(t.get("n_shared_left_alone", 0) for t in sc)})
            rows.append({
                "threshold_voxels": thr,
                "per_frame": per_frame,
                # 운반 = 파지 구간 전체다. 예전의 합성 들어올림을 대신한다.
                "transport_residual_negatives": sum(x["n_negative"] for x in per_frame),
                "shared_voxels_total": sum(x["n_shared_left_alone"] for x in per_frame),
                "sign_forced_total": sum(x["n_sign_forced"] for x in per_frame),
                "min_mm_worst": min(x["min_mm"] for x in per_frame)})
            r = rows[-1]
            print(f"  문턱 {thr:>4.1f} 복셀  잔여음수 {r['transport_residual_negatives']:>5}  "
                  f"공유 {r['shared_voxels_total']:>6}  교정 {r['sign_forced_total']:>6}  "
                  f"최악 {r['min_mm_worst']:>8.2f} mm")
        report["sweep"] = rows

        ok = [r for r in rows if r["transport_residual_negatives"] == 0]
        chosen = max(ok, key=lambda r: r["threshold_voxels"]) if ok else None
        top = max(args.thresholds)
        report["choice"] = {
            "rule": "운반 구간(기록의 파지 프레임) 잔여 음수가 0 인 문턱 중 가장 큰 값. "
                    "크면 공유를 더 많이 잡아 숨김 위험이 작아진다",
            "candidates_passing": [r["threshold_voxels"] for r in ok],
            "chosen_threshold_voxels": (None if chosen is None
                                        else chosen["threshold_voxels"]),
            "chosen_is_top_of_list": bool(chosen is not None
                                          and chosen["threshold_voxels"] >= top),
            "top_of_candidate_list": top,
            "n_grasp_frames": len(frames_used),
        }
    finally:
        scene.close()

    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print(f"\nwrote {out}")
    c = report["choice"]
    print(f"  고른 값: {c['chosen_threshold_voxels']} 복셀 "
          f"(통과 후보 {c['candidates_passing']}, 표본 {c['n_grasp_frames']} 프레임)")
    if c["chosen_is_top_of_list"]:
        print(f"  [경고] 고른 값이 후보 목록의 상한({c['top_of_candidate_list']})이다 — "
              f"잘린 값일 수 있다. --thresholds 를 더 넓혀 다시 돌려라")


if __name__ == "__main__":
    main()
