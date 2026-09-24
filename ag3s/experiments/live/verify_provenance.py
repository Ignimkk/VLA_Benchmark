"""I3 검증 — 거리장의 출처와 age 가 실제로 찍히고 상태 기계가 맞는가.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-openpi-live/bin/python -m \\
        benchmark.ag3s.experiments.live.verify_provenance \\
        --out outputs/live_test/20260922_i3_provenance/verify_provenance.json

재는 것 넷.

1. **도장** — 두 backend 가 같은 모양으로 `sequence`·`backend`·`observed_at`·`tiers` 를 찍는가
2. **일련번호** — 프레임을 이어 돌리면 1 부터 단조 증가하는가 (되돌아가거나 건너뛰면 배선 결함)
3. **age 기준** — `observed_at` 이 그 프레임 관측의 **가장 최신 촬영 시각**과 같은가
4. **상태 기계** — `new` / `carried` / `stale` / `unavailable` 이 한도 유무에 따라 옳게 나오나.
   특히 한도가 `None` 일 때 `stale` 로 올라가지 **않고** `staleness_checked: false` 가 실리나

새 MuJoCo 씬을 쓴다. 저장 기록을 읽지 않는다.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def build(seed: int, height: int, width: int, target: str):
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import (
        TRANSPORT_MODEL, TransportScene, gaussian_attention)
    from benchmark.ag3s.types import CameraID, CameraObservation

    scene = TransportScene(TRANSPORT_MODEL, settle_steps=400, height=height, width=width)
    rng = np.random.default_rng(seed)
    qadr = scene._qadr[f"{target}_free"]
    scene.data.qpos[qadr:qadr + 2] += rng.uniform(-0.012, 0.012, size=2)
    scene.mujoco.mj_forward(scene.model, scene.data)
    truth = scene.body_position_in_base(target)
    filter_robot = build_robot_model(scene)
    constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    ids = {"zed_left": CameraID.HEAD, "wrist_cam_l": CameraID.LEFT_WRIST,
           "wrist_cam_r": CameraID.RIGHT_WRIST}

    def observe(stamp_base: float):
        frames = {c: scene.capture(c) for c in CAMS}
        # 카메라마다 다른 촬영 시각을 준다 — `observed_at` 이 **가장 최신**을 고르는지 보려면
        # 셋이 같아서는 안 된다. 재생에서 실제 지연은 0 이므로(하나의 qpos) 이것은 검사용
        # 값이고, 파이프라인의 skew 한계 안에 둔다.
        offsets = {"zed_left": 0.000, "wrist_cam_l": 0.010, "wrist_cam_r": 0.020}
        obs = [CameraObservation(
            camera_id=ids[c], depth=frames[c].depth,
            camera_intrinsics=frames[c].camera_intrinsics,
            T_base_cam=frames[c].T_base_cam, robot_state=frames[c].robot_state,
            timestamp=stamp_base + offsets[c], state_timestamp=stamp_base + offsets[c],
            attention_map=gaussian_attention(frames[c], truth), image_hw=frames[c].hw)
            for c in CAMS]
        return obs, stamp_base + max(offsets.values())

    return scene, observe, filter_robot, constraint_robot


def run_backend(backend, observe, filter_robot, constraint_robot, *, n_frames,
                coarse, fine, tsdf):
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.runtime.pipeline import AG3S

    esdf = {"voxel_size": coarse, "max_distance": 0.4,
            "exclude_support_surfaces": False, "backend": backend}
    if backend == "curobo":
        esdf["fine_voxel_size"] = fine
        esdf["tsdf_voxel_size"] = tsdf
    ag3s = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf",
                                      "pointcloud": {"range_max": 2.0},
                                      "esdf": esdf}),
                robot_model=filter_robot, constraint_robot_model=constraint_robot)
    out = []
    for k in range(n_frames):
        stamp = 1000.0 + k
        obs, newest = observe(stamp)
        cs, _ = ag3s.process_multi_debug(obs, phase="approach",
                                         active_manipulators=["left"])
        prov = getattr(cs.esdf, "provenance", None)
        out.append({
            "frame": k,
            "newest_observation_stamp": newest,
            "has_field": cs.esdf is not None,
            "has_stamp": prov is not None,
            "provenance": None if prov is None else prov.to_dict(),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--coarse", type=float, default=0.020)
    ap.add_argument("--fine", type=float, default=0.005)
    ap.add_argument("--tsdf-voxel", type=float, default=0.005)
    args = ap.parse_args()

    from benchmark.ag3s.fields.provenance import FieldProvenance

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"stage": "I3", "seed": args.seed, "frames": args.frames,
                    "source": "fresh_mujoco_scene", "used_saved_run": False,
                    "checks": {}}

    scene, observe, filter_robot, constraint_robot = build(
        args.seed, args.height, args.width, args.target)
    try:
        per_backend = {}
        for backend in ("legacy", "curobo"):
            per_backend[backend] = run_backend(
                backend, observe, filter_robot, constraint_robot,
                n_frames=args.frames, coarse=args.coarse, fine=args.fine,
                tsdf=args.tsdf_voxel)
        report["per_backend"] = per_backend

        # ── 조건 1: 두 backend 가 같은 모양으로 찍는가 ──
        shapes = {}
        for b, rows in per_backend.items():
            shapes[b] = sorted((rows[0]["provenance"] or {}).keys())
        report["checks"]["same_stamp_shape"] = {
            "pass": all(r["has_stamp"] for rows in per_backend.values() for r in rows)
                    and shapes["legacy"] == shapes["curobo"],
            "keys": shapes["legacy"],
            "legacy_backend_field": per_backend["legacy"][0]["provenance"]["backend"],
            "curobo_backend_field": per_backend["curobo"][0]["provenance"]["backend"]}

        # ── 조건 2: 일련번호가 1 부터 단조 증가 ──
        seqs = {b: [r["provenance"]["sequence"] for r in rows]
                for b, rows in per_backend.items()}
        report["checks"]["sequence_monotonic_from_one"] = {
            "pass": all(s == list(range(1, len(s) + 1)) for s in seqs.values()),
            "sequences": seqs}

        # ── 조건 3: observed_at 이 가장 최신 촬영 시각 ──
        drift = {b: [abs(r["provenance"]["observed_at"] - r["newest_observation_stamp"])
                     for r in rows] for b, rows in per_backend.items()}
        report["checks"]["observed_at_is_newest_stamp"] = {
            "pass": all(max(v) < 1e-9 for v in drift.values()),
            "max_abs_error_sec": {b: max(v) for b, v in drift.items()}}

        # ── 조건 4: 상태 기계 ──
        base = FieldProvenance.from_dict(per_backend["curobo"][0]["provenance"])
        t0 = float(base.observed_at)
        cases = []
        for label, dt, carried, limit, want in (
            ("갱신 프레임, 한도 없음", 0.020, False, None, "new"),
            ("control frame 2번째, 한도 없음", 0.100, True, None, "carried"),
            ("한도 안 (0.6 s)", 0.300, True, 0.6, "carried"),
            ("한도 초과 (0.2 s)", 0.300, True, 0.2, "stale"),
            ("한도 없으면 stale 로 안 올린다", 5.000, True, None, "carried"),
        ):
            a = base.applied_by_client(t0 + dt, age_limit_sec=limit, carried=carried)
            cases.append({"case": label, "age_ms": a.age_ms, "limit_sec": limit,
                          "state": a.state, "expected": want,
                          "staleness_checked": a.staleness_checked,
                          "ok": a.state == want})
        miss = FieldProvenance.unavailable("no field").applied_by_client(t0, age_limit_sec=0.5)
        cases.append({"case": "필드 없음", "age_ms": miss.age_ms, "limit_sec": 0.5,
                      "state": miss.state, "expected": "unavailable",
                      "staleness_checked": miss.staleness_checked,
                      "ok": miss.state == "unavailable"})
        report["state_machine"] = cases
        report["checks"]["state_machine"] = {
            "pass": all(c["ok"] for c in cases),
            "n_cases": len(cases),
            "n_ok": sum(1 for c in cases if c["ok"])}

        # ── 조건 5: 한도 기본값이 추측이 아닌가 ──
        from benchmark.ag3s.config import AG3SConfig
        default_limit = AG3SConfig.from_dict({}).timing.max_field_age_sec
        report["checks"]["limit_default_is_unset"] = {
            "pass": default_limit is None,
            "default": default_limit,
            "why": "F14(상태 지연 한계 100 ms 가 피해 시작점보다 6 배 느슨했다)를 반복하지 "
                   "않기 위해 측정 전에는 정하지 않는다. T3 에서 정한다"}

        report["verdict"] = {
            "pass": all(c.get("pass") for c in report["checks"].values()),
            "checks": {k: bool(v.get("pass")) for k, v in report["checks"].items()}}
    finally:
        scene.close()

    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    for name, c in report["checks"].items():
        print(f"  [{'PASS' if c.get('pass') else 'FAIL'}] {name}")
        if not c.get("pass"):
            print("        " + json.dumps({k: v for k, v in c.items() if k != "pass"},
                                          ensure_ascii=False)[:280])
    print("  상태 기계:")
    for c in report["state_machine"]:
        age = "-" if c["age_ms"] is None else f"{c['age_ms']:7.1f} ms"
        print(f"    [{'OK' if c['ok'] else 'NG'}] {c['case']:32s} age {age}  "
              f"limit {str(c['limit_sec']):>5}  -> {c['state']:12s} "
              f"(검사됨 {c['staleness_checked']})")
    print(f"  => I3 {'PASS' if report['verdict']['pass'] else 'FAIL'}")
    raise SystemExit(0 if report["verdict"]["pass"] else 1)


if __name__ == "__main__":
    main()
