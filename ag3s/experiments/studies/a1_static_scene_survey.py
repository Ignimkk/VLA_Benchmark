"""A1 사전 조사 — **정적 기하를 누가 주는가**. 씬에 무엇이 고정으로 있고, 그것을
해석적 채널에 넣으면 제약 모델의 여유거리가 어떻게 되는가.

파이프라인을 건드리지 않는다. MJCF 에서 도형을 뽑고 FK 로 구 중심을 얻어
`analytic_distance` 만 부른다 — 여기서 나온 수치가 A1 의 선택지를 정한다.

실행:
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -m \
        benchmark.ag3s.experiments.studies.a1_static_scene_survey --records run_0004 --frames 15
"""

from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/archive/14d-era-20260923/archive/14d-era-20260923/figures/a1-static-scene.png")
MARGIN = 0.05


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments.common import figstyle
    figstyle.use_korean()
    return figstyle


def survey(model, data):
    """씬의 비-로봇 geom 을 네 갈래로 나눈다: 자유물체 · 충돌 정적 · 시각 전용 · 미지원 형상."""
    import mujoco

    from benchmark.ag3s.fields.esdf import StaticBox, StaticPlane
    from benchmark.ag3s.experiments.sources.mujoco_source import is_robot_body

    free_bodies = {int(model.jnt_bodyid[i]) for i in range(model.njnt)
                   if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE}
    buckets = defaultdict(list)
    shapes = []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        gn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
        if is_robot_body(bn):
            buckets["robot"].append(gn)
            continue
        if b in free_bodies:
            buckets["free"].append(gn)
            continue
        collidable = int(model.geom_contype[g]) or int(model.geom_conaffinity[g])
        t = model.geom_type[g]
        pos = np.array(data.geom_xpos[g], float)
        rot = np.array(data.geom_xmat[g], float).reshape(3, 3)
        if not collidable:
            buckets["visual"].append(gn)
            continue
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            shapes.append(StaticBox(center=pos, half_extents=np.array(model.geom_size[g], float),
                                    rotation=rot, label=f"{bn}/{gn}"))
            buckets["static"].append(gn)
        elif t == mujoco.mjtGeom.mjGEOM_PLANE:
            shapes.append(StaticPlane(point=pos, normal=rot[:, 2].copy(), label=f"{bn}/{gn}"))
            buckets["static"].append(gn)
        else:
            buckets["unsupported"].append(f"{gn}({t})")
    return shapes, buckets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from benchmark.ag3s.fields.esdf import analytic_distance
    from benchmark.ag3s.experiments.reports.grounding_report import (
        build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
    from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS

    rec = load_run(args.records, limit=args.frames)
    scene = replay_scene(rec)
    arms = build_constraint_robot_model(scene)
    full = build_robot_model(scene)

    pose_scene(scene, rec.steps[0])
    shapes0, buckets = survey(scene.model, scene.data)

    per_shape = defaultdict(lambda: np.inf)     # 도형 -> 최소 여유거리 (arms)
    frame_min = {"arms": [], "full": []}
    worst_link = {"arms": defaultdict(lambda: np.inf), "full": defaultdict(lambda: np.inf)}
    for st in rec.steps:
        pose_scene(scene, st)
        shapes, _ = survey(scene.model, scene.data)
        q = np.asarray([scene.data.qpos[scene._qadr[j]] for j in DEFAULT_RBY1_JOINTS], float)
        for tag, rm in (("arms", arms), ("full", full)):
            c, r = rm.sphere_centers_numeric(q)
            clr = analytic_distance(c, shapes) - r - MARGIN
            frame_min[tag].append(float(clr.min()))
            for name, v in zip(rm.sphere_link_names, clr):
                worst_link[tag][name] = min(worst_link[tag][name], float(v))
            if tag == "arms":
                each = np.stack([analytic_distance(c, [s]) for s in shapes]) - r[None, :] - MARGIN
                for s, row in zip(shapes, each):
                    per_shape[s.label] = min(per_shape[s.label], float(row.min()))

    fs = _style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.06, 0.94], hspace=0.34, wspace=0.26)

    # --- (1) 실제 씬 — 위에서 본 배치도 (z 는 아래 단면에서 본다) ---------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("① 씬 배치 — 위에서 본 것 (x–y, 로봇 base 원점)", fontsize=10.5, color=fs.INK)
    for s in shapes0:
        if hasattr(s, "half_extents"):
            h = s.half_extents
            ax.add_patch(Rectangle((s.center[0] - h[0], s.center[1] - h[1]), 2 * h[0], 2 * h[1],
                                   facecolor=fs.CATEGORICAL[0], alpha=0.30,
                                   edgecolor=fs.CATEGORICAL[0], lw=1.0))
    grid = dict(x=(-0.30, 1.20), y=(-0.90, 0.90))
    ax.add_patch(Rectangle((grid["x"][0], grid["y"][0]), grid["x"][1] - grid["x"][0],
                           grid["y"][1] - grid["y"][0], fill=False, lw=1.6, ls="--",
                           edgecolor=fs.CATEGORICAL[2]))
    ax.text(grid["x"][1], grid["y"][1] + 0.08, "ESDF 격자", color=fs.CATEGORICAL[2], fontsize=9,
            ha="right")
    ax.plot([0], [0], marker="o", ms=7, color=fs.CATEGORICAL[1])
    ax.annotate("robot base", (0, 0), xytext=(-1.35, 0.55), fontsize=9, color=fs.CATEGORICAL[1],
                arrowprops=dict(arrowstyle="->", color=fs.CATEGORICAL[1], lw=1.0))
    ax.annotate("테이블", (0.65, -0.40), xytext=(1.85, -0.95), fontsize=9.5, color=fs.INK,
                arrowprops=dict(arrowstyle="->", color=fs.INK_2, lw=1.0))
    ax.annotate("선반", (0.0, -1.62), xytext=(1.30, -2.35), fontsize=9.5, color=fs.INK,
                arrowprops=dict(arrowstyle="->", color=fs.INK_2, lw=1.0))
    ax.text(0.30, 3.30, "벽 4 장 (±3 m)", fontsize=9.5, color=fs.INK, ha="center")
    ax.set_xlim(-3.7, 3.9); ax.set_ylim(-4.0, 3.6); ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    # --- (2) 그래프 — 도형별 최소 여유거리 -------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1:])
    items = sorted(per_shape.items(), key=lambda kv: kv[1])
    names = [k.split("/")[-1] for k, _ in items]
    vals = [v * 1000 for _, v in items]
    cols = [fs.CATEGORICAL[7] if v < 0 else fs.CATEGORICAL[0] for v in vals]
    ax2.barh(range(len(vals)), vals, color=cols, height=0.62)
    ax2.set_yticks(range(len(vals))); ax2.set_yticklabels(names, fontsize=8.4)
    ax2.invert_yaxis()
    ax2.axvline(0, color=fs.INK, lw=1.0)
    ax2.set_xscale("symlog", linthresh=100)
    ax2.set_xlabel("제약 모델(양팔 120구)의 최소 여유거리  d − r − 50 mm  [mm, symlog]")
    ax2.set_title("② 도형별 — 음수인 것은 하나뿐이다 (table_top, 파지 하강)",
                  fontsize=10.5, color=fs.INK)
    for i, v in enumerate(vals):
        ax2.text(v, i, f" {v:,.0f}", va="center", fontsize=8,
                 ha="left" if v >= 0 else "right", color=fs.INK)

    # --- (3) 그래프 — 프레임별 추이, 두 모델 -----------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    n = len(frame_min["arms"])
    ax3.plot(range(n), np.array(frame_min["arms"]) * 1000, "-o", ms=4,
             color=fs.CATEGORICAL[0], label="제약 모델 (양팔 120구)")
    ax3.plot(range(n), np.array(frame_min["full"]) * 1000, "-s", ms=4,
             color=fs.CATEGORICAL[3], label="전신 194구 (참고)")
    ax3.axhline(0, color=fs.INK, lw=1.0)
    ax3.set_xlabel("프레임"); ax3.set_ylabel("최소 여유거리 [mm]")
    ax3.set_title("③ 프레임별 — 전신은 상수 위반, 양팔은 파지 구간만", fontsize=10.5, color=fs.INK)
    ax3.legend(fontsize=8.4, frameon=False)

    # --- (4) 표 — 갈래와 개수 --------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    rows = [["갈래", "개수 · 무엇"],
            ["자유물체 (free joint)", f"{len(buckets['free'])} — 사과·바나나·귤·배·상자"],
            ["**충돌 정적**", f"{len(buckets['static'])} — 바닥1 테이블5 선반6 벽4"],
            ["시각 전용 (con=0/0)", f"{len(buckets['visual'])} — vis 복제 · 창 · 문틀 · 마커"],
            ["미지원 형상 (구·메시)", f"{len(buckets['unsupported'])} — 상자·평면만 지원"],
            ["로봇", f"{len(buckets['robot'])} — 자기 필터가 본다"]]
    _table(ax4, rows, fs, "④ 비-로봇 geom 의 갈래")

    # --- (5) 표 — 링크별 최악 --------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 2])
    neg_a = sorted([(k, v) for k, v in worst_link["arms"].items() if v < 0], key=lambda kv: kv[1])
    neg_f = sorted([(k, v) for k, v in worst_link["full"].items() if v < 0], key=lambda kv: kv[1])
    rows = [["링크", "최소 여유거리"]]
    rows += [["**제약 모델(양팔)**", ""]]
    rows += [[k, f"{v*1000:,.1f} mm"] for k, v in neg_a] or [["(없음)", ""]]
    rows += [["**전신 — 참고**", ""]]
    rows += [[k, f"{v*1000:,.1f} mm"] for k, v in neg_f if k not in dict(neg_a)]
    _table(ax5, rows, fs, "⑤ 음수가 나오는 링크")

    fs.style_axes(fig, [ax, ax2, ax3])
    fig.suptitle("A1 — 정적 기하를 씬에서 뽑으면 무엇이 들어오고, 무엇이 제약이 되는가"
                 f"   ({args.records}, {n} 프레임, margin 50 mm)",
                 fontsize=12.5, color=fs.INK)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {out}")
    print(f"갈래: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))
    print("음수 도형:", [(k, round(v * 1000, 1)) for k, v in items if v < 0])


def _table(ax, rows, fs, title, fontsize=8.6):
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, color=fs.INK)
    body = [[c.replace("**", "") for c in r] for r in rows]
    t = ax.table(cellText=body, colWidths=[0.42, 0.58], loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(fontsize); t.scale(1, 1.42)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if r == 0 or (rows[r][1] == "" and rows[r][0]):
            cell.set_facecolor("#ecebe7"); cell.set_text_props(weight="bold")


if __name__ == "__main__":
    main()
