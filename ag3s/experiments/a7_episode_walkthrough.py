"""한 에피소드 전체에서 **파이프라인의 모든 단계가 실제로 적용되는가** (2026-09-18).

규칙 B 의 전체 틀을 그대로 따라가며, 44 프레임 **전부**에 대해 단계마다 무엇이 일어났는지
모은다. 한두 프레임 그림이 아니라 **에피소드 전체의 추이**를 남기는 것이 목적이다.

    카메라 depth → 로봇 마스크 → attention lifting → target grounding
                → TSDF/ESDF (+해석적 채널 ·라벨 층 ·쥔 물체 파내기 ·감쇠)
                → 거리장 어댑터 → SQP 선형화 (+쥔 물체 질의점 ·목적지 마진) → QP 해 → 안전 게이트

**실제 통합 경로를 탄다** — `SafePolicy` 를 그대로 돌린다 (`serve_safe.py` 와 같은 것).
관측만 기록에서 꺼내 `wire` 요청으로 만들어 넣는다. 그래야 "실기에서도 이렇게 돈다" 가 된다.

두 단계로 나뉜다:

* `collect` — 에피소드를 돌리며 단계별 수치를 JSON 으로 남긴다 (느리다, 44 × 약 2 초)
* `plot`    — 그 JSON 으로 그림과 표를 만든다 (빠르다, 다시 안 돌린다)

실행:
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -m \
        benchmark.ag3s.experiments.a7_episode_walkthrough collect \
        --records run_0004 --attention attention_step1_run0004.npz
    ... plot
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures/a7-episode-walkthrough.png")
OUT2 = pathlib.Path("benchmark/ag3s/docs/figures/a7-episode-stages.png")
JSON = pathlib.Path("benchmark/ag3s/docs/figures/a7-episode-walkthrough.json")
CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments import figstyle
    figstyle.use_korean()
    return figstyle


# --------------------------------------------------------------------------- 수집

def collect(args) -> dict:
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s import static_scene
    from benchmark.ag3s.experiments.attention_report import target_from_prompt
    from benchmark.ag3s.experiments.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.mujoco_source import gaussian_attention
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene
    from benchmark.ag3s.pipeline import AG3S
    from benchmark.trajopt import wire
    from benchmark.trajopt.config import TrajOptConfig
    from benchmark.trajopt.experiments.esdf_rollout import phase_for
    from benchmark.trajopt.placed import destination_placement
    from benchmark.trajopt.safe_policy import SafePolicy, grasp_parent_links

    run = load_run(args.records, limit=args.frames or None)
    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)

    ag3s = AG3S(
        AG3SConfig.from_dict({
            "collision_backend": "esdf",
            "pointcloud": {"range_max": args.range_max},
            "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                     "exclude_support_surfaces": False},
        }),
        robot_model=filter_robot, constraint_robot_model=constraint_robot,
        attached_parent_links=grasp_parent_links())

    shapes, sg = static_scene.from_mujoco(scene.model, scene.data)
    print(f"[static] {sg.summary()}")

    # attention — 실측 npz 가 있으면 그것, 없으면 합성 블롭. 없으면 하류가 통째로 안 돈다.
    att_block = cell = aidx = None
    kind = "합성 블롭"
    if args.attention and pathlib.Path(args.attention).exists():
        blob = np.load(args.attention, allow_pickle=False)
        cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
        att_block = np.asarray(blob["attention"], np.float32)
        aidx = ([int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"])),
                [str(a) for a in blob["aggregations"]].index(str(cell["agg"])),
                [str(c) for c in blob["cameras"]].index("cam_high"))
        kind = "실측"
    target_name = target_from_prompt(run.prompt)
    here = {"i": 0}

    def attention_fn(_o, _r):
        i = here["i"]
        if att_block is not None and i < att_block.shape[0]:
            di, ai, ci = aidx
            return {"zed_left": att_block[i, di, ai, cell["layer"], cell["head"], ci]}
        head = scene.capture("zed_left")
        return {"zed_left": gaussian_attention(head, scene.body_position_in_base(target_name))}

    class _Recorded:
        def __init__(self): self.i = 0
        def infer(self, _o, **_k):
            a = np.asarray(run.steps[min(self.i, len(run.steps) - 1)].actions, np.float64)
            self.i += 1
            return {"actions": a}
        def reset(self): self.i = 0
        @property
        def metadata(self): return {"stub": "recorded chunks"}

    safe = SafePolicy(
        _Recorded(), ag3s=ag3s, attention_fn=attention_fn,
        static_geometry=shapes,
        placed_fn=destination_placement(robot_model=constraint_robot, below_rim=args.below_rim),
        to_config=TrajOptConfig.from_dict({
            "collision": {"backend": "esdf", "esdf_margin": args.esdf_margin,
                          "use_support_planes": False},
            "safety": {"require_certified_geometry": True}}))

    rows, events = [], []
    prev_latch = safe._latch.phase.name
    print(f"{'i':>3} {'단계':>9} {'마스크%':>7} {'att점':>6} {'target':>7} {'점유':>7} "
          f"{'미관측%':>7} {'라벨':>4} {'쥔점':>5} {'파냄':>5} {'목적지':>6} "
          f"{'잠금':>9} {'위반mm':>8} {'safe':>5} {'ms':>6}")

    for i, step in enumerate(run.steps):
        here["i"] = i
        pose_scene(scene, step)
        depth, K, T, rstate, stamps = {}, {}, {}, {}, {}
        mask_px = seen_px = 0
        # **카메라 셋은 같은 시뮬 순간이다.** 하나의 `qpos` 에서 렌더하므로 촬영 시각이 같다.
        # 카메라마다 `time.monotonic()` 을 찍으면 OSMesa 렌더 시간(카메라당 약 100 ms)이
        # 그대로 촬영 지연으로 읽혀 F14(자세 지연 — 관측이 늦으면 그 사이 로봇이 움직인다)가
        # 매 프레임 DEGRADED 를 낸다 (실측 296 ms, 한계 100 ms). 그것은 파이프라인 결함이
        # 아니라 **하네스가 만든 가짜 지연**이다. 한 번만 찍는다.
        stamp = time.monotonic()
        for c in CAMERAS:
            fr = scene.capture(c)
            d = np.asarray(fr.depth, np.float64)
            k = np.asarray(fr.camera_intrinsics, np.float64)
            t = np.asarray(fr.T_base_cam, np.float64)
            depth[c], K[c], T[c] = d, k, t
            rstate[c] = np.asarray(fr.robot_state, np.float64)
            stamps[c] = stamp
            m = ag3s._robot_mask_for(d, k, t, fr.robot_state)
            mask_px += 0 if m is None else int(np.asarray(m, bool).sum())
            seen_px += int(np.isfinite(d).sum())

        phase = phase_for(step.t_step, args.phase_boundaries)
        manipulators = ("left",) if safe._latch.holding else ()
        req = wire.pack_request(
            {"state": np.asarray(step.state, np.float64)},
            cameras=CAMERAS, depth=depth, intrinsics=K, extrinsics=T,
            robot_state=rstate, stamps=stamps, phase=phase,
            active_manipulators=manipulators, reset=(i == 0), seq=i + 1)

        res = safe.infer(req)
        cs = safe._last_constraint_set
        dbg = safe._last_debug or {}
        st = {} if cs is None or cs.esdf is None else cs.esdf.stats
        met = {} if cs is None else (cs.metrics or {})
        att = safe.ag3s.attached
        sol = safe.refiner.last_result

        latch = safe._latch.phase.name
        if latch != prev_latch:
            events.append({"i": i, "from": prev_latch, "to": latch})
            prev_latch = latch

        row = {
            "i": i, "t_step": int(step.t_step), "phase": phase,
            # ① 카메라 depth · ② 로봇 마스크
            "depth_px": seen_px, "mask_px": mask_px,
            "mask_frac": mask_px / max(seen_px, 1),
            "points_raw": int(met.get("n_points_raw", 0)),
            "points_self_filtered": int(met.get("n_points_self_filtered", 0)),
            "points_final": int(met.get("n_points_final", 0)),
            # ③ attention lifting · ④ grounding
            "attention_points": int(len(dbg.get("attention_cloud").points)
                                    if dbg.get("attention_cloud") is not None else 0),
            "cameras_with_attention": len(met.get("cameras_with_attention", ()) or ()),
            "grounding_status": str(met.get("target_grounding_status", "-")),
            "target_points": int(0 if cs is None or cs.target is None else len(cs.target.points)),
            "target_confidence": float(met.get("target_confidence", 0.0) or 0.0),
            # ⑤ TSDF/ESDF + 층들
            "occupied": int(st.get("n_occupied", 0)),
            "unknown_frac": float(st.get("unknown_fraction", 0.0)),
            "n_labels": int(st.get("n_labels", 0)),
            "n_static_shapes": int(st.get("n_static_shapes", 0)),
            "carved_attached": int(st.get("n_attached_voxels_carved", 0)),
            "decayed": bool((st.get("decay") or {}).get("decayed", False)),
            # ⑥ 쥔 물체 · 목적지
            "held_points": int(0 if att is None or att.points is None else len(att.points)),
            "destination": bool(getattr(cs, "destination_label", None)),
            "latch": latch,
            # ⑦ SQP · QP 해 · 안전 게이트
            "sqp_iterations": int(getattr(sol, "iterations", 0) or 0),
            "violation_before_mm": float(getattr(sol, "reference_violation", 0.0) or 0.0) * 1000.0,
            "violation_after_mm": (float(res["max_violation_m"]) * 1000.0
                                   if np.isfinite(res["max_violation_m"]) else None),
            "trajopt_status": res["trajopt_status"],
            "ag3s_status": res["ag3s_status"],
            "certified": bool(res["geometry_certified"]),
            "safe": bool(res["safe"]),
            "scene_failure": getattr(safe.refiner, "last_failure", None),
            "validity": str(cs.validity.name if cs is not None else "-"),
            "notes": list(cs.notes)[:6] if cs is not None else [],
            "ms_total": float(res["timing_ms"]["total"]),
            "ms_ag3s": float(res["timing_ms"].get("ag3s", 0.0)),
            "ms_trajopt": float(res["timing_ms"].get("trajopt", 0.0)),
        }
        rows.append(row)
        v = row["violation_after_mm"]
        print(f"{i:>3} {phase:>9} {row['mask_frac']*100:>6.1f}% {row['attention_points']:>6} "
              f"{row['target_points']:>7} {row['occupied']:>7} {row['unknown_frac']*100:>6.1f}% "
              f"{row['n_labels']:>4} {row['held_points']:>5} {row['carved_attached']:>5} "
              f"{str(row['destination']):>6} {latch:>9} "
              f"{(f'{v:8.1f}' if v is not None else '     inf')} "
              f"{str(row['safe']):>5} {row['ms_total']:>6.0f}")

    return {"records": args.records, "frames": len(rows), "attention": kind,
            "n_static_shapes": len(shapes), "esdf_margin_mm": args.esdf_margin * 1000,
            "latch_events": events, "rows": rows}


# --------------------------------------------------------------------------- 그림

#: 파이프라인 단계와, 한 프레임에서 그 단계가 "적용됐다" 를 어떻게 읽는가.
STAGES = [
    ("① 카메라 depth", lambda r: r["depth_px"] > 0),
    ("② 로봇 마스크 (자기 필터)", lambda r: r["mask_px"] > 0),
    ("③ attention lifting", lambda r: r["attention_points"] > 0),
    ("④ target grounding", lambda r: r["target_points"] > 0),
    ("⑤ TSDF/ESDF", lambda r: r["occupied"] > 0),
    ("⑥ 해석적 채널 (A1)", lambda r: r["n_static_shapes"] > 0),
    ("⑦ 라벨 층", lambda r: r["n_labels"] > 0),
    ("⑧ 감쇠 (F20, 기본 끔)", lambda r: r["decayed"]),
    ("⑨ 잠금 걸기·유지", lambda r: r["latch"] != "SEARCHING"),
    ("⑩ 쥔 물체 질의점 (E3·F19)", lambda r: r["held_points"] > 0),
    ("⑪ 쥔 물체 파내기 (A2)", lambda r: r["carved_attached"] > 0),
    ("⑫ 목적지 마진 (F18·A3)", lambda r: r["destination"]),
    ("⑬ SQP 선형화", lambda r: r["sqp_iterations"] > 0),
    ("⑭ 기하 인증", lambda r: r["certified"]),
    ("⑮ 안전 게이트 통과", lambda r: r["safe"]),
]


def plot(data, out, out2):
    fs = _style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rows = data["rows"]
    idx = [r["i"] for r in rows]
    n = len(rows)
    grid = np.array([[bool(fn(r)) for r in rows] for _, fn in STAGES])

    # ---------- 그림 1: 단계별 적용 지도 + 요약표 --------------------------------
    fig = plt.figure(figsize=(16.0, 9.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.55, 1.0],
                          wspace=0.20, hspace=0.30)

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(grid, aspect="auto", cmap=ListedColormap(["#fdeceb", "#cfe8da"]),
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(STAGES)))
    ax.set_yticklabels([s for s, _ in STAGES], fontsize=8.8)
    ax.set_xticks(range(0, n, 2))
    ax.set_xticklabels([str(k) for k in range(0, n, 2)], fontsize=8)
    ax.set_xlabel("프레임 (에피소드 전체)")
    ax.set_title("① 프레임마다 어느 단계가 적용됐나  (초록 = 적용)",
                 fontsize=11.0, color=fs.INK)
    for m in range(len(STAGES) + 1):
        ax.axhline(m - 0.5, color="#ffffff", lw=1.5)
    for e in data["latch_events"]:
        ax.axvline(e["i"] - 0.5, color=fs.CATEGORICAL[7], lw=1.4, ls="--")
        ax.text(e["i"] - 0.4, -0.9, e["to"], fontsize=7.2, color=fs.CATEGORICAL[7],
                rotation=90, va="bottom")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_title("② 단계별 요약 — 전체 프레임 기준", fontsize=11.0, color=fs.INK, pad=14)
    trows = [["단계", "적용 프레임", "대표 수치"]]
    reps = {
        "① 카메라 depth": f"{int(np.median([r['depth_px'] for r in rows])):,} px/프레임",
        "② 로봇 마스크 (자기 필터)":
            f"중앙 {np.median([r['mask_frac'] for r in rows])*100:.1f} %",
        "③ attention lifting": f"중앙 {int(np.median([r['attention_points'] for r in rows])):,} 점",
        "④ target grounding":
            f"중앙 {int(np.median([r['target_points'] for r in rows])):,} 점",
        "⑤ TSDF/ESDF": f"점유 {int(np.median([r['occupied'] for r in rows])):,} 복셀",
        "⑥ 해석적 채널 (A1)": f"{rows[0]['n_static_shapes']} 도형",
        "⑦ 라벨 층": f"최대 {max(r['n_labels'] for r in rows)} 라벨",
        "⑧ 감쇠 (F20, 기본 끔)": "끔 (잠복)",
        "⑨ 잠금 걸기·유지": " → ".join(e["to"] for e in data["latch_events"]) or "-",
        "⑩ 쥔 물체 질의점 (E3·F19)": f"{max(r['held_points'] for r in rows)} 점",
        "⑪ 쥔 물체 파내기 (A2)": f"최대 {max(r['carved_attached'] for r in rows)} 복셀",
        "⑫ 목적지 마진 (F18·A3)": f"{sum(r['destination'] for r in rows)} 프레임",
        "⑬ SQP 선형화": f"중앙 {int(np.median([r['sqp_iterations'] for r in rows]))} 반복",
        "⑭ 기하 인증": f"{sum(r['certified'] for r in rows)} / {n}",
        "⑮ 안전 게이트 통과": f"{sum(r['safe'] for r in rows)} / {n}",
    }
    for m, (name, _) in enumerate(STAGES):
        trows.append([name, f"{int(grid[m].sum())} / {n}", reps.get(name, "")])
    t = ax2.table(cellText=trows, colWidths=[0.42, 0.20, 0.38], loc="upper center",
                  cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(7.6); t.scale(1, 1.42)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if r == 0:
            cell.set_facecolor("#ecebe7"); cell.set_text_props(weight="bold")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    ax3.set_title("③ 에피소드 요약", fontsize=11.0, color=fs.INK, pad=14)
    ev = " · ".join(f"{e['i']}:{e['to']}" for e in data["latch_events"]) or "-"
    vio = [r["violation_after_mm"] for r in rows if r["violation_after_mm"] is not None]
    srows = [["무엇", "값"],
             ["기록 · 프레임", f"{data['records']} · {n}"],
             ["attention", data["attention"]],
             ["정적 기하", f"{data['n_static_shapes']} 도형"],
             ["잠금 사건", ev],
             ["씬 실패 (조용한 정지)", f"{sum(1 for r in rows if r['scene_failure'])} 프레임"],
             ["최악 위반 (최적화 후)", f"{max(vio):.1f} mm" if vio else "-"],
             ["safe 프레임", f"{sum(r['safe'] for r in rows)} / {n}"],
             ["프레임당 시간 (중앙)", f"{np.median([r['ms_total'] for r in rows]):.0f} ms"]]
    t2 = ax3.table(cellText=srows, colWidths=[0.44, 0.56], loc="upper center", cellLoc="left")
    t2.auto_set_font_size(False); t2.set_fontsize(8.2); t2.scale(1, 1.6)
    for (r, c), cell in t2.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if r == 0:
            cell.set_facecolor("#ecebe7"); cell.set_text_props(weight="bold")

    fig.suptitle("한 에피소드 전체에서 파이프라인 단계가 적용되는가"
                 f"   ({data['records']}, {n} 프레임, 3 카메라)",
                 fontsize=13.0, color=fs.INK)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {out}")

    # ---------- 그림 2: 단계별 수치 추이 (전 프레임) ------------------------------
    fig2 = plt.figure(figsize=(16.0, 9.4))
    g2 = fig2.add_gridspec(3, 2, wspace=0.22, hspace=0.42)

    def _marks(a):
        for e in data["latch_events"]:
            a.axvline(e["i"], color=fs.CATEGORICAL[7], lw=1.1, ls="--")

    a = fig2.add_subplot(g2[0, 0])
    a.plot(idx, [r["points_raw"] for r in rows], "-o", ms=3, color=fs.CATEGORICAL[0],
           label="원시 점")
    a.plot(idx, [r["points_self_filtered"] for r in rows], "-s", ms=3,
           color=fs.CATEGORICAL[7], label="자기 필터가 지운 점")
    a.plot(idx, [r["points_final"] for r in rows], "-^", ms=3, color=fs.CATEGORICAL[2],
           label="융합 후 남은 점")
    _marks(a); a.set_yscale("log"); a.set_xlabel("프레임"); a.set_ylabel("점 수 (log)")
    a.set_title("① 관측 → 로봇 마스크 → 융합", fontsize=10.4, color=fs.INK)
    a.legend(fontsize=7.8, frameon=False)

    b = fig2.add_subplot(g2[0, 1])
    b.plot(idx, [r["attention_points"] for r in rows], "-o", ms=3,
           color=fs.CATEGORICAL[0], label="attention 점")
    b.plot(idx, [r["target_points"] for r in rows], "-s", ms=3,
           color=fs.CATEGORICAL[2], label="target 점")
    _marks(b); b.set_xlabel("프레임"); b.set_ylabel("점 수")
    b.set_title("② attention lifting → target grounding", fontsize=10.4, color=fs.INK)
    b.legend(fontsize=7.8, frameon=False)

    c = fig2.add_subplot(g2[1, 0])
    c.plot(idx, [r["occupied"] for r in rows], "-o", ms=3, color=fs.CATEGORICAL[0],
           label="점유 복셀")
    c.set_xlabel("프레임"); c.set_ylabel("점유 복셀", color=fs.CATEGORICAL[0])
    c2 = c.twinx()
    c2.plot(idx, [r["unknown_frac"] * 100 for r in rows], "-s", ms=3,
            color=fs.CATEGORICAL[3], label="미관측 %")
    c2.set_ylabel("미관측 [%]", color=fs.CATEGORICAL[3])
    _marks(c)
    c.set_title("③ TSDF/ESDF — 점유와 미관측", fontsize=10.4, color=fs.INK)

    d = fig2.add_subplot(g2[1, 1])
    d.plot(idx, [r["held_points"] for r in rows], "-o", ms=3, color=fs.CATEGORICAL[0],
           label="쥔 물체 질의점 (E3·F19)")
    d.plot(idx, [r["carved_attached"] for r in rows], "-s", ms=3, color=fs.CATEGORICAL[7],
           label="파낸 복셀 (A2)")
    d.plot(idx, [r["n_labels"] * 10 for r in rows], "-^", ms=3, color=fs.CATEGORICAL[2],
           label="라벨 수 (x10)")
    _marks(d); d.set_xlabel("프레임"); d.set_ylabel("개수")
    d.set_title("④ 쥔 물체 · 파내기 · 라벨 층", fontsize=10.4, color=fs.INK)
    d.legend(fontsize=7.8, frameon=False)

    e = fig2.add_subplot(g2[2, 0])
    e.plot(idx, [r["violation_before_mm"] for r in rows], "-o", ms=3,
           color=fs.CATEGORICAL[7], label="최적화 전 (참조 청크)")
    e.plot(idx, [r["violation_after_mm"] if r["violation_after_mm"] is not None else np.nan
                 for r in rows], "-s", ms=3, color=fs.CATEGORICAL[2], label="최적화 후")
    e.axhline(0, color=fs.INK, lw=1.0)
    _marks(e); e.set_xlabel("프레임"); e.set_ylabel("최악 위반 [mm]")
    e.set_title("⑤ SQP — 위반을 얼마나 줄였나", fontsize=10.4, color=fs.INK)
    e.legend(fontsize=7.8, frameon=False)

    f = fig2.add_subplot(g2[2, 1])
    ag = np.array([r["ms_ag3s"] for r in rows])
    to = np.array([r["ms_trajopt"] for r in rows])
    f.bar(idx, ag, color=fs.CATEGORICAL[0], label="AG3S (지각)")
    f.bar(idx, to, bottom=ag, color=fs.CATEGORICAL[3], label="TO (최적화)")
    f.axhline(66.7, color=fs.CATEGORICAL[7], lw=1.4, ls=":", label="예산 66.7 ms")
    _marks(f); f.set_xlabel("프레임"); f.set_ylabel("[ms]")
    f.set_title("⑥ 프레임당 시간 — 예산 대비", fontsize=10.4, color=fs.INK)
    f.legend(fontsize=7.8, frameon=False)

    fs.style_axes(fig2, [a, b, c, c2, d, e, f])
    fig2.suptitle("에피소드 전체의 단계별 수치 추이"
                  f"   ({data['records']}, {n} 프레임; 세로 점선 = 잠금 전이)",
                  fontsize=13.0, color=fs.INK)
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {out2}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("collect", "plot", "both"))
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--frames", type=int, default=0, help="0 이면 에피소드 전부")
    ap.add_argument("--attention", default="attention_step1_run0004.npz")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/step-01-attention.json")
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--esdf-margin", type=float, default=0.05)
    ap.add_argument("--below-rim", type=float, default=0.060)
    ap.add_argument("--phase-boundaries", type=int, nargs=3, default=(24, 56, 72))
    ap.add_argument("--json", default=str(JSON))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--out2", default=str(OUT2))
    args = ap.parse_args()

    if args.mode in ("collect", "both"):
        data = collect(args)
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"wrote {args.json}")
    if args.mode in ("plot", "both"):
        data = json.loads(pathlib.Path(args.json).read_text())
        plot(data, args.out, args.out2)


if __name__ == "__main__":
    main()
