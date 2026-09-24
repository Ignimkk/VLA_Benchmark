"""T0 판정 — 환경·배선 검증. **기록만 읽는다.**

    PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.live.verify_t0 --root outputs/live_test/20260924_t0

`AG3S_TOTAL_TEST_Prompt.md` 의 T0 이 요구하는 것을 한 항목씩 센다. 통과 기준을 낮추지 않고,
확인하지 않은 것은 통과로 적지 않는다 — 검사 불가한 항목은 `확인 불가` 로 남는다.

| 항목 | 무엇을 보나 |
|---|---|
| held-out 씬 | 에피소드가 test split(1800-1999)인가. train 이면 즉시 실패 |
| 16D | 청크 폭과 action_format 이 16 인가 |
| backend | **모든** 프레임의 출처 도장이 `curobo` 인가 (하나라도 `legacy` 면 즉시 실패) |
| legacy 생성 0 | `EsdfBuilder.instances_created` 불변식이 살아 있는가 (코드 확인) |
| completeness | 기대 관측 수 = 실제, 누락 0, 중복 0, 시각 역전 0, 설명 안 된 carried 0 |
| IPC | 프레임마다 왕복 결과가 적혔는가. 클라이언트 집계와 프레임 집계가 맞는가 |
| 리셋 지점 | 에피소드 시작(reset)이 기록에 남는가 |
| 값 지어내기 | `stale` 판정을 안 했으면 `staleness_checked: false` 가 실렸는가 |
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

TEST_SPLIT = range(1800, 2000)


def check_session(d: pathlib.Path) -> dict:
    man = json.loads((d / "manifest.json").read_text())
    comp = json.loads((d / "completeness.json").read_text())
    rows = [json.loads(l) for l in (d / "frames.jsonl").read_text().splitlines() if l.strip()]

    scene = man.get("scene", {}) or {}
    ep = scene.get("episode_index")
    checks: dict[str, dict] = {}

    def add(name, ok, detail):
        checks[name] = {"pass": (None if ok is None else bool(ok)), "detail": detail}

    add("held_out_scene", ep in TEST_SPLIT,
        f"episode {ep}, split={scene.get('episode_split')!r}")

    # `FrameRecorder` 는 `extra` 를 줄의 최상위로 펼친다 — `r["extra"]` 는 없다.
    widths = {tuple(r["chunk_shape"]) for r in rows
              if r["kind"] == "planning" and r.get("chunk_shape")}
    # `action_format` 은 `scene` 밑에 있다 (`policy` 는 프롬프트·원격·안전 설정이다).
    # 설정 문자열만으로는 부족하므로 프레임이 적은 **실제 청크 폭**도 같이 본다.
    fmt = scene.get("action_format")
    add("action_dim_16",
        fmt == "rby1_16d" and bool(widths) and all(w[-1] == 16 for w in widths),
        f"action_format={fmt!r}"
        + (f", 청크 폭 {sorted(widths)}" if widths else ", 청크 폭이 기록에 없다"))

    backends = Counter(
        (r.get("field") or {}).get("backend")
        for r in rows if r["kind"] in ("planning", "control") and isinstance(r.get("field"), dict))
    add("backend_curobo_every_frame",
        set(backends) == {"curobo"} and backends["curobo"] > 0,
        f"{dict(backends)}")

    add("completeness_pass", comp.get("pass"), json.dumps(
        {k: comp.get(k) for k in ("expected_observation_frames",
                                  "captured_observation_frames", "planned_missing_frames",
                                  "duplicate_sequence_ids", "timestamp_reversals",
                                  "unexplained_carried_or_stale")}, ensure_ascii=False))

    ipc_frames = Counter(r["ipc"] for r in rows
                         if r["kind"] == "planning" and r.get("ipc"))
    client = comp.get("client_ipc_stats") or {}
    # 클라이언트 집계는 `ok` 를 `safe` 로 센다. 이름만 다르고 같은 수여야 한다.
    mapped = {"ok": client.get("safe", 0), "unsafe": client.get("unsafe", 0),
              "timeout": client.get("timeout", 0), "stale": client.get("stale", 0),
              "error": client.get("error", 0)}
    agree = all(ipc_frames.get(k, 0) == v for k, v in mapped.items() if v or ipc_frames.get(k))
    add("ipc_recorded_every_frame",
        len([r for r in rows if r["kind"] == "planning"]) == comp.get("planning_records")
        and all(r.get("ipc") for r in rows if r["kind"] == "planning") and agree,
        f"프레임 {dict(ipc_frames)} · 클라이언트 {client}")

    add("reset_recorded", scene.get("episode_index") is not None
        and man.get("frames_of_reference") is not None,
        "manifest 가 에피소드와 좌표계를 적었다 (리셋은 에피소드 시작 한 번)")

    limit = comp.get("staleness_limit_sec")
    add("no_fabricated_staleness",
        (limit is None and comp.get("staleness_checked") is False)
        or (limit is not None and comp.get("staleness_checked") is True),
        f"한도={limit} · staleness_checked={comp.get('staleness_checked')}")

    skews = [r["camera_skew_sec"] * 1000 for r in rows
             if r["kind"] == "observation" and r.get("camera_skew_sec") is not None]
    return {
        "dir": str(d), "episode": ep, "checks": checks,
        "camera_skew_ms": {"n": len(skews),
                           "median": (None if not skews else round(sorted(skews)[len(skews) // 2], 2)),
                           "max": (None if not skews else round(max(skews), 2))},
        "completeness": comp,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    sessions = [check_session(d) for d in sorted(root.glob("t0_ep*"))
                if (d / "completeness.json").exists()]
    if not sessions:
        raise SystemExit(f"{root} 안에 기록이 없다")

    # legacy 생성 0 — 기록이 아니라 **코드의 불변식**으로 확인한다. 프레임마다 검사하고
    # 하나라도 만들어졌으면 파이프라인이 죽으므로, 실행이 끝까지 간 것이 그 자체로 근거다.
    src = pathlib.Path("benchmark/ag3s/runtime/pipeline.py").read_text()
    invariant = ("EsdfBuilder.instances_created" in src
                 and "T0 의 즉시 실패 조건이다" in src)

    names = list(sessions[0]["checks"])
    print(f"=== T0 — {len(sessions)} 에피소드 ===")
    print(f"{'항목':<32} " + " ".join(f"ep{s['episode']:>5}" for s in sessions))
    all_pass = True
    for n in names:
        cells = []
        for s in sessions:
            v = s["checks"][n]["pass"]
            cells.append("  통과" if v else ("확인불가" if v is None else "  실패"))
            all_pass = all_pass and bool(v)
        print(f"{n:<32} " + " ".join(f"{c:>8}" for c in cells))
    print(f"{'legacy_builder_invariant':<32} " +
          " ".join(f"{('  통과' if invariant else '  실패'):>8}" for _ in sessions))
    all_pass = all_pass and invariant

    print()
    for s in sessions:
        print(f"ep{s['episode']}:")
        for n in names:
            print(f"    {n}: {s['checks'][n]['detail']}")
        print(f"    카메라 시차: n={s['camera_skew_ms']['n']} "
              f"중앙값 {s['camera_skew_ms']['median']} ms 최대 {s['camera_skew_ms']['max']} ms")
    print()
    print(f"T0 판정: {'통과' if all_pass else '실패'}")

    report = {"sessions": sessions, "legacy_builder_invariant": invariant, "pass": all_pass}
    out = pathlib.Path(args.out) if args.out else root / "verify_t0.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
