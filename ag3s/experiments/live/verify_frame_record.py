"""I4 검증 — 프레임 기록과 completeness 표가 T0 의 요구를 만족하는가.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.live.verify_frame_record \\
        --out outputs/live_test/20260922_i4_frames

**`pi05_infer` 를 돌리지 않는다.** 그 driver 는 `mink`(MuJoCo IK)를 요구하고 이 서버의 어느
venv 에도 없다 — RUNBOOK 이 "로컬" 이라고 적은 그대로다. 그래서 여기서는 driver 가 부르는
것과 **같은 호출 순서**를 재현해 기록기와 상태 기계를 검증한다: 청크 하나를 받고
8 개 control frame 을 돌리는 것을 프레임마다 반복한다.

재는 것 다섯.

1. **줄 수** — observation / planning / control 이 기대한 수만큼 나오는가
2. **`carried` 가 control 에서만 생기는가** — planning frame 은 매번 새 필드를 받는다
3. **age 가 스텝마다 늘어나는가** — 0 → 467 ms (`1/ctrl_hz` 씩)
4. **T0 의 즉시 실패 조건** — 중복 일련번호 · timestamp 역전 · 설명 안 되는 `carried`
5. **한도가 있을 때 `stale` 로 바뀌는가** — 그리고 없을 때 바뀌지 **않는가**
"""

from __future__ import annotations

import argparse
import json
import pathlib

CTRL_HZ = 15.0
HORIZON = 8


def run_session(out_dir: pathlib.Path, *, n_chunks: int, age_limit, label: str,
                inject: str = "none") -> dict:
    """driver 가 부르는 순서를 그대로 재현한다."""
    from benchmark.ag3s.fields.provenance import FieldProvenance
    from benchmark.ag3s.runtime.frame_record import FrameRecorder, collect_manifest

    manifest = collect_manifest(
        seed=101,
        scene={"model": "rby1_transport_14d", "fruit_layout_index": 6,
               "fruit_slot_order": ["banana", "apple", "pear", "orange"]},
        policy={"prompt": "put the apple in the basket", "safe_remote": True},
        esdf={"owner": "server (serve_safe)", "reported_per_frame_in": "field.backend"},
        timing={"ctrl_hz": CTRL_HZ, "open_loop_horizon": HORIZON,
                "chunk_period_ms": HORIZON / CTRL_HZ * 1000.0,
                "max_field_age_sec": age_limit},
        cameras=("zed_left", "wrist_cam_l", "wrist_cam_r"),
        render={"third_person_video": None},
        extra={"harness": "verify_frame_record", "injection": inject, "label": label})

    rec = FrameRecorder(str(out_dir), manifest=manifest,
                        expected_observation_frames=n_chunks)
    now = 1000.0
    for c in range(n_chunks):
        stamps = {"zed_left": now, "wrist_cam_l": now + 0.010,
                  "wrist_cam_r": now + 0.020}
        if inject == "timestamp_reversal" and c == 2:
            # T0 의 즉시 실패 조건 — 시각이 되돌아간다. 검사가 잡는지 본다.
            stamps = {k: v - 1.0 for k, v in stamps.items()}
        rec.observation(t_step=c, stamps=stamps, ag3s_status="ok",
                        grounding_status="ok", validity="valid", n_points=29882)

        seq = c + 1
        if inject == "duplicate_seq" and c == 3:
            seq = c   # 같은 일련번호가 두 번 — 검사가 잡는지 본다
        field = FieldProvenance(
            sequence=seq, backend="curobo",
            observed_at=max(stamps.values()), built_at=now + 0.05,
            frame_id="live", frame_index=c,
            cameras=("zed_left", "wrist_cam_l", "wrist_cam_r"),
            tiers=({"voxel_size_m": 0.02}, {"voxel_size_m": 0.005}))
        if inject == "no_field" and c == 4:
            field = FieldProvenance.unavailable("injected: server returned no field")
        rec.planning(seq=seq, t_step=c, field=field,
                     verdict={"safe": True, "geometry_certified": True},
                     timing_ms={"policy_infer": 120.0, "ag3s": 3500.0, "trajopt": 430.0},
                     ipc="ok")

        for k in range(HORIZON):
            rec.control(t_step=c, chunk_seq=seq, step_in_chunk=k,
                        now=max(stamps.values()) + k / CTRL_HZ,
                        field=field, executed=True)
        now += HORIZON / CTRL_HZ
    return rec.close(ipc_stats={"sent": n_chunks, "safe": n_chunks, "unsafe": 0,
                                "timeout": 0, "stale": 0, "error": 0})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=6)
    args = ap.parse_args()

    root = pathlib.Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    report: dict = {"stage": "I4", "chunks": args.chunks, "checks": {}, "sessions": {}}

    # ── 정상 세션, 한도 없음 ──
    clean = run_session(root / "clean_no_limit", n_chunks=args.chunks,
                        age_limit=None, label="정상 · 한도 없음")
    report["sessions"]["clean_no_limit"] = clean
    # ── 정상 세션, 한도 0.2 s ──
    limited = run_session(root / "clean_limit_200ms", n_chunks=args.chunks,
                          age_limit=0.2, label="정상 · 한도 0.2 s")
    report["sessions"]["clean_limit_200ms"] = limited
    # ── 주입 세션 셋 ──
    for inject in ("timestamp_reversal", "duplicate_seq", "no_field"):
        report["sessions"][inject] = run_session(
            root / inject, n_chunks=args.chunks, age_limit=0.2, label=inject,
            inject=inject)

    exp_ctrl = args.chunks * HORIZON

    # ── 조건 1: 줄 수 ──
    report["checks"]["record_counts"] = {
        "pass": (clean["captured_observation_frames"] == args.chunks
                 and clean["planning_records"] == args.chunks
                 and clean["control_records"] == exp_ctrl),
        "observation": clean["captured_observation_frames"],
        "planning": clean["planning_records"],
        "control": clean["control_records"],
        "expected_control": exp_ctrl}

    # ── 조건 2: carried 가 control 에서만 ──
    rows = [json.loads(l) for l in
            (root / "clean_no_limit" / "frames.jsonl").read_text().splitlines()]
    plan_states = {r["field"]["state"] for r in rows if r["kind"] == "planning"}
    ctrl_states = {r["field"]["state"] for r in rows if r["kind"] == "control"}
    report["checks"]["carried_only_in_control"] = {
        "pass": plan_states == {"new"} and ctrl_states == {"new", "carried"},
        "planning_states": sorted(plan_states),
        "control_states": sorted(ctrl_states)}

    # ── 조건 3: age 가 스텝마다 늘어난다 ──
    first = [r for r in rows if r["kind"] == "control" and r["chunk_seq"] == 1]
    ages = [r["field"]["age_ms"] for r in sorted(first, key=lambda r: r["step_in_chunk"])]
    # `1000/15 = 66.666…` 를 반올림해 두고 1e-6 으로 견주면 실패한다. 허용오차는 부동소수
    # 오차에 맞춘다 — 여기서 재려는 것은 "스텝마다 1/ctrl_hz 씩 늘어나는가" 이고 마이크로초
    # 자리의 표기 차이가 아니다.
    step_ms = 1000.0 / CTRL_HZ
    want = [k * step_ms for k in range(HORIZON)]
    err = [abs(a - w) for a, w in zip(ages, want)]
    report["checks"]["age_grows_per_step"] = {
        "pass": len(ages) == HORIZON and max(err) < 1e-6,
        "ages_ms": [round(a, 3) for a in ages],
        "expected_ms": [round(w, 3) for w in want],
        "max_abs_error_ms": max(err) if err else None,
        "step_ms": step_ms}

    # ── 조건 4: 즉시 실패 조건을 잡는가 ──
    report["checks"]["catches_t0_failures"] = {
        "pass": (clean["timestamp_reversals"] == 0
                 and clean["duplicate_sequence_ids"] == 0
                 and clean["unexplained_carried_or_stale"] == 0
                 and clean["pass"] is True
                 and report["sessions"]["timestamp_reversal"]["timestamp_reversals"] > 0
                 and report["sessions"]["duplicate_seq"]["duplicate_sequence_ids"] > 0
                 and report["sessions"]["timestamp_reversal"]["pass"] is False
                 and report["sessions"]["duplicate_seq"]["pass"] is False),
        "clean": {k: clean[k] for k in ("timestamp_reversals", "duplicate_sequence_ids",
                                        "unexplained_carried_or_stale", "pass")},
        "timestamp_reversal": {k: report["sessions"]["timestamp_reversal"][k]
                               for k in ("timestamp_reversals", "pass")},
        "duplicate_seq": {k: report["sessions"]["duplicate_seq"][k]
                          for k in ("duplicate_sequence_ids", "pass")}}

    # ── 조건 5: 한도가 있을 때만 stale ──
    report["checks"]["stale_only_with_limit"] = {
        "pass": ("stale" not in clean["field_state_counts"]
                 and clean["staleness_checked"] is False
                 and "stale" in limited["field_state_counts"]
                 and limited["staleness_checked"] is True),
        "no_limit_states": clean["field_state_counts"],
        "limit_200ms_states": limited["field_state_counts"]}

    # ── 조건 6: 필드 없음이 unavailable 로 남는가 ──
    nf = report["sessions"]["no_field"]
    report["checks"]["missing_field_is_unavailable"] = {
        "pass": nf["field_state_counts"].get("unavailable", 0) == 1 + HORIZON,
        "states": nf["field_state_counts"],
        "expected_unavailable": 1 + HORIZON}

    # ── 조건 7: manifest 가 실측 버전을 담는가 ──
    man = json.loads((root / "clean_no_limit" / "manifest.json").read_text())
    v = man["versions"]
    report["checks"]["manifest_has_measured_versions"] = {
        "pass": (v.get("python") and not str(v.get("numpy", "")).startswith("MISSING")
                 and "frames_of_reference" in man and man["seed"] == 101),
        "python": v.get("python"), "numpy": v.get("numpy"),
        "mujoco": v.get("mujoco"), "curobo": v.get("curobo"),
        "curobo_commit": (man.get("curobo_commit") or "")[:12],
        "gpu": v.get("gpu"), "seed": man.get("seed")}

    report["verdict"] = {
        "pass": all(c.get("pass") for c in report["checks"].values()),
        "checks": {k: bool(v.get("pass")) for k, v in report["checks"].items()}}
    (root / "verify_frame_record.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str))

    print(f"wrote {root / 'verify_frame_record.json'}")
    for name, c in report["checks"].items():
        print(f"  [{'PASS' if c.get('pass') else 'FAIL'}] {name}")
        if not c.get("pass"):
            print("        " + json.dumps({k: v for k, v in c.items() if k != "pass"},
                                          ensure_ascii=False)[:300])
    print("  세션별 completeness:")
    for name, t in report["sessions"].items():
        print(f"    {name:20s} pass={str(t['pass']):5s} obs={t['captured_observation_frames']:2d} "
              f"plan={t['planning_records']:2d} ctrl={t['control_records']:3d} "
              f"dup={t['duplicate_sequence_ids']} rev={t['timestamp_reversals']} "
              f"states={t['field_state_counts']}")
    print(f"  => I4 {'PASS' if report['verdict']['pass'] else 'FAIL'}")
    raise SystemExit(0 if report["verdict"]["pass"] else 1)


if __name__ == "__main__":
    main()
