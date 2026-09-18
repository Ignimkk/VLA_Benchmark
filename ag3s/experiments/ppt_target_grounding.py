"""발표 슬라이드용 그림 — 모듈 3: Target Grounding.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.ppt_target_grounding \\
        --records run_0004 --attention attention_step1_run0004.npz --frames 21 --frame 3

| 파일 | 슬라이드 | 무엇 |
|---|---|---|
| `ppt-tg-01-method.png`  | 1 | 씨앗 → 영역 성장 → 점수 → target |
| `ppt-tg-02-io.png`      | 1 | 입력 2 / `ground_target()` / 출력 블록도 |
| `ppt-tg-03-support.png` | 2 | **전/후 ①** 지지면을 연결에서 빼지 않으면 씬 전체가 한 덩어리가 된다 |
| `ppt-tg-04-latch.png`   | 2 | **전/후 ②** 무상태 grounding 대 잠금 (F17) + attach/detach |

## 적용 전/후 둘 다 되돌리는 스위치가 있다

* **지지면 제외** — `ground_target(..., exclude_mask=None)` 이 그대로 "빼지 않은" 동작이다.
  빼지 않으면 영역 성장이 테이블을 타고 번져 씬 전체가 한 클러스터가 되고 target 을 못 찾는다.
* **F17**(조작 대상 식별에 에피소드 상태가 없다) — `GraspLatch` 를 태우지 않은 것이 곧 수정
  이전이다. 프레임마다 grounding 출력을 그대로 조작 대상으로 쓰던 때.

둘 다 코드를 고치지 않고 **호출만 바꿔** 재현한다.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from benchmark.ag3s.experiments.ppt_attention_projection import (
    AMBER, AMBER as _A, AMBER as _A2, BLUE, GREEN, INK, MUTED, PURPLE, RED,
    _arrow, _box, _box3d, _inbox, _name_of, _objects, _style, ATT_CAMS, CAMS, LABEL,
)

OUT = pathlib.Path("benchmark/ag3s/docs/figures/ppt")
GREY = "#c9c8c4"


# ------------------------------------------------------------------- 한 프레임 grounding
def run_frame(scene, cfg, filter_robot, grids, ts, adapter):
    """지지면을 **빼고** / **안 빼고** 두 갈래로 grounding 한다."""
    from benchmark.ag3s.experiments.mujoco_source import camera_observation
    from benchmark.ag3s.multiview import fuse, process_observation
    from benchmark.ag3s.support_surface import fit_support_surfaces
    from benchmark.ag3s.target_grounding import ground_target
    from benchmark.ag3s.types import AttentionPointCloud

    obs, frames = [], {}
    for cam, ac in zip(CAMS, ATT_CAMS):
        o, f = camera_observation(scene, cam, filter_robot, attention_map=grids[ac], timestamp=ts)
        obs.append(o)
        frames[cam] = f
    results = [process_observation(o, cfg, robot_model=filter_robot, attention_adapter=adapter)
               for o in obs]
    fused, att, raw = fuse(results, voxel_size=cfg.timing.fusion_voxel_size,
                           frame_id=cfg.frame_id, attention_config=cfg.attention)
    cloud = fused.as_pointcloud()
    surfaces, support = fit_support_surfaces(cloud, cfg.support_surface, timestamp=ts)
    apc = AttentionPointCloud(cloud, att, raw)

    def _g(mask):
        return ground_target(apc, cfg.clustering,
                             seed_percentile=cfg.attention.seed_percentile,
                             seed_threshold=cfg.attention.seed_threshold,
                             exclude_mask=mask, min_radius=cfg.geometry.min_radius,
                             timestamp=ts)

    return dict(cloud=cloud, attention=np.asarray(att, np.float64), support=support,
                surfaces=surfaces, frames=frames,
                fixed=_g(support), nomask=_g(None))


def _scores(gr):
    """1위 · 2위 점수. 잠금의 격차 문턱이 읽는 것."""
    sc = sorted((float(c.target_score) for c in gr.clusters), reverse=True)
    return (sc[0] if sc else 0.0), (sc[1] if len(sc) > 1 else 0.0)


# ------------------------------------------------------------------------ 01 방법론 3D
def fig_method(fs, r, objects, out):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.0, 8.6))
    fig.patch.set_facecolor(fs.SURFACE)
    gs = fig.add_gridspec(1, 3, wspace=0.10, left=0.015, right=0.985, top=0.84, bottom=0.10)
    rng = np.random.default_rng(0)

    p = r["cloud"].points
    keep = _inbox(p)
    gr = r["fixed"]
    seeds = np.zeros(len(p), bool)
    seeds[np.asarray(gr.seed_indices, np.int64)] = True
    support = np.asarray(r["support"], bool)

    # (a) 씨앗
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    rest = keep & ~seeds & ~support
    q = p[rest][rng.choice(int(rest.sum()), min(14000, int(rest.sum())), replace=False)]
    ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=0.8, c=GREY, linewidths=0, alpha=0.55,
               label="그 밖의 점")
    q = p[support & keep]
    q = q[rng.choice(len(q), min(9000, len(q)), replace=False)]
    ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=0.8, c="#9ec5f4", linewidths=0, alpha=0.55,
               label=f"지지면 {len(r['surfaces'])} 장 — 연결에서 뺀다")
    q = p[seeds & keep]
    ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=9, c=AMBER, linewidths=0,
               label=f"씨앗 {int(seeds.sum()):,} 개")
    pk = p[gr.attention_peak_index]
    ax.scatter(*pk, s=170, marker="*", c="#d62fd6", zorder=6, label="attention 최고점")
    _box3d(ax)
    ax.legend(loc="upper left", fontsize=9.5, markerscale=2.0, framealpha=0.9,
              borderpad=0.4, labelspacing=0.35)
    ax.set_title("① 씨앗 — attention 상위 5 % 에서 출발", fontsize=15, weight="bold", pad=6)

    # (b) 후보 덩어리
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    q = p[rest][rng.choice(int(rest.sum()), min(12000, int(rest.sum())), replace=False)]
    ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=0.7, c="#dedcd7", linewidths=0, alpha=0.45)
    top = sorted(gr.clusters, key=lambda c: -c.target_score)[:6]
    for i, cl in enumerate(top):
        s = p[np.asarray(cl.point_indices, np.int64)]
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=6, color=fs.CATEGORICAL[i % 8], linewidths=0,
                   label=f"#{cl.id} · {cl.point_count} 점")
    _box3d(ax)
    ax.legend(loc="upper left", fontsize=9.5, markerscale=2.0, framealpha=0.9,
              borderpad=0.4, labelspacing=0.35)
    ax.set_title(f"② 영역 성장 — 후보 {len(gr.clusters)} 덩어리\n"
                 "무엇이 한 덩어리인지는 기하가 정한다", fontsize=15, weight="bold", pad=6)

    # (c) 점수
    ax = fig.add_subplot(gs[0, 2])
    cl = sorted(gr.clusters, key=lambda c: -c.target_score)[:6]
    ys = np.arange(len(cl))[::-1]
    ax.barh(ys, [c.target_score for c in cl], height=0.6,
            color=[RED] + [GREY] * (len(cl) - 1))
    for y, c in zip(ys, cl):
        nm = _name_of(np.asarray(c.centroid, float), objects)
        ax.text(c.target_score + 0.006, y, f"{nm} · {c.point_count} 점", va="center", fontsize=12)
    ax.set_yticks(ys); ax.set_yticklabels([f"#{c.id}" for c in cl], fontsize=12)
    ax.set_xlabel("target_score", fontsize=13)
    ax.set_xlim(0, max(c.target_score for c in cl) * 1.75)
    ax.tick_params(labelsize=11, colors=MUTED)
    ax.set_facecolor(fs.SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    a, b = _scores(gr)
    ax.set_title(f"③ 점수 — attention 크기 · 조밀도 · 최고점까지 거리\n"
                 f"1위 / 2위 = {a/max(b,1e-9):.1f} 배", fontsize=15, weight="bold", pad=10)

    fig.suptitle("target grounding — attention 은 어디서 시작할지만 정하고, "
                 "물체 경계는 3D 기하가 정한다", fontsize=18, weight="bold", y=0.955)
    fig.text(0.5, 0.025,
             "후보 이름은 이 frame의 MuJoCo GT 중심에 가장 가까운 물체를 표시한 설명용 라벨이며, "
             "target_score 계산에는 사용하지 않는다.",
             ha="center", fontsize=11.5, color=MUTED)
    fig.savefig(out, dpi=130, facecolor=fs.SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------ 02 I/O
def fig_io(fs, stats, out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16.0, 9.0))
    fig.patch.set_facecolor(fs.SURFACE)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(8.0, 8.55, "ground_target() — 올린 점들에서 물체 하나를 고른다",
            ha="center", fontsize=20, weight="bold", color=INK)

    ins = [("AttentionPointCloud\n점마다 값 두 벌", "#9ec5f4"),
           (f"지지면 마스크\n평면 {stats['n_surfaces']} 장 · {stats['n_support']:,} 점", "#cde2fb")]
    y = 5.95
    for text, face in ins:
        _box(ax, 0.45, y, 4.3, 1.34, text, face, fontsize=14, weight="normal")
        _arrow(ax, (4.85, y + 0.67), (5.75, 4.55), MUTED, lw=1.5)
        y -= 1.95

    _box(ax, 5.85, 3.35, 4.3, 2.4,
         "ground_target()\n\n① 씨앗 (상위 5 %)\n② 영역 성장\n③ 점수 · 1위 선택",
         GREEN, fontsize=14.5, text_colour="#ffffff")

    outs = [("출력 — GroundingResult\ntarget 하나 + 후보 전부 + 실패 사유", "#f7d9a0"),
            ("후보를 전부 돌려주는 이유\nLOW_SCORE 는 튜닝 문제,\n후보 0 개는 지각 문제 — 구분돼야 한다",
             "#fdf0d5"),
            ("지지면은 연결에서만 뺀다\n점은 남아 SUPPORT_SURFACE 후보로 간다", "#fdf0d5")]
    y = 6.35
    for text, face in outs:
        _arrow(ax, (10.25, 4.55), (10.85, y + 0.62), MUTED, lw=1.5)
        _box(ax, 10.95, y, 4.75, 1.30, text, face, fontsize=12.5, weight="normal")
        y -= 1.78

    ax.add_patch(FancyBboxPatch((0.45, 0.35), 15.25, 2.15,
                                boxstyle="round,pad=0.10,rounding_size=0.18",
                                facecolor="#ecebe7", edgecolor="#d8d7d2", lw=1.2))
    ax.text(8.05, 1.85, "이 함수는 무상태다 — 프레임 사이에 아무것도 기억하지 않는다",
            ha="center", fontsize=15.5, weight="bold", color=INK)
    ax.text(8.05, 0.98,
            "그것이 옳다. 지각은 매 프레임 새로 판단해야 한다.\n"
            "다만 '지금 무엇을 쥐고 있는가' 는 지각이 아니라 에피소드 상태다 — 그래서 밖에서 잠근다.",
            ha="center", va="center", fontsize=13, color=MUTED, linespacing=1.7)

    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fs.SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- 03 전/후 ① 지지면 제외
def fig_support(fs, r, rows, objects, out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(16.0, 9.0))
    fig.patch.set_facecolor(fs.SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.20,
                          left=0.05, right=0.965, top=0.845, bottom=0.07)
    rng = np.random.default_rng(0)
    p = r["cloud"].points
    keep = _inbox(p)

    for col, (key, title) in enumerate((("nomask", "지지면을 빼지 않으면"),
                                        ("fixed", "지지면을 연결에서 빼면"))):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        gr = r[key]
        q = p[keep][rng.choice(int(keep.sum()), min(9000, int(keep.sum())), replace=False)]
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=0.7, c="#dedcd7", linewidths=0, alpha=0.40)
        top = sorted(gr.clusters, key=lambda c: -c.point_count)[:6]
        for i, cl in enumerate(top):
            s = p[np.asarray(cl.point_indices, np.int64)]
            s = s[rng.choice(len(s), min(9000, len(s)), replace=False)]
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=4, color=fs.CATEGORICAL[i % 8], linewidths=0)
        _box3d(ax)
        big = max((c.point_count for c in gr.clusters), default=0)
        ax.set_title(f"{title}\n후보 {len(gr.clusters)} 덩어리 · 최대 덩어리 {big:,} 점 "
                     f"({100.0*big/max(len(p),1):.0f} %)  ·  상태 {gr.status.name}",
                     fontsize=14.5, weight="bold",
                     color=RED if key == "nomask" else BLUE, pad=6)

    # (c) 프레임별 최대 덩어리 비율
    ax = fig.add_subplot(gs[1, 0])
    fr = np.array([x["frame"] for x in rows])
    ax.plot(fr, [100.0 * x["nomask_big"] for x in rows], lw=2.6, color=RED, marker="o", ms=4,
            label="빼지 않음")
    ax.plot(fr, [100.0 * x["fixed_big"] for x in rows], lw=2.6, color=BLUE, marker="s", ms=4,
            label="뺌 (현재)")
    ax.set_xlabel("프레임", fontsize=13)
    ax.set_ylabel("최대 덩어리가 차지하는 비율 [%]", fontsize=13)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=11, colors=MUTED)
    ax.set_facecolor(fs.SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("한 덩어리가 씬을 얼마나 삼키나", fontsize=14.5, weight="bold", pad=9)

    # (d) 숫자
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.add_patch(FancyBboxPatch((0.15, 0.3), 9.7, 9.35,
                                boxstyle="round,pad=0.12,rounding_size=0.25",
                                facecolor="#ecebe7", edgecolor="#d8d7d2", lw=1.3))
    n = len(rows)
    nm_fail = sum(1 for x in rows if not x["nomask_ok"])
    fx_fail = sum(1 for x in rows if not x["fixed_ok"])
    nm_wrong = sum(1 for x in rows if x["nomask_ok"] and x["nomask_name"] != x["fixed_name"])
    nm_big = float(np.median([x["nomask_big"] for x in rows])) * 100
    fx_big = float(np.median([x["fixed_big"] for x in rows])) * 100

    ax.text(5.0, 9.05, f"{n} 프레임", ha="center", fontsize=13, color=MUTED)
    ax.text(3.2, 8.05, "빼지 않음", ha="center", fontsize=16, weight="bold", color=RED)
    ax.text(7.0, 8.05, "뺌 (현재)", ha="center", fontsize=16, weight="bold", color=BLUE)
    ax.text(5.1, 6.95, "target 을 못 찾은 프레임", ha="center", fontsize=12.5, color=MUTED)
    ax.text(3.2, 5.80, f"{nm_fail} / {n}", ha="center", fontsize=30, weight="bold", color=RED)
    ax.text(7.0, 5.80, f"{fx_fail} / {n}", ha="center", fontsize=30, weight="bold", color=BLUE)
    ax.text(5.1, 4.55, "최대 덩어리 비율 (중앙)", ha="center", fontsize=12.5, color=MUTED)
    ax.text(3.2, 3.50, f"{nm_big:.0f} %", ha="center", fontsize=26, weight="bold", color=RED)
    ax.text(7.0, 3.50, f"{fx_big:.0f} %", ha="center", fontsize=26, weight="bold", color=BLUE)
    line = (f"한 프레임도 target 을 찾지 못한다 ({n} / {n})" if nm_fail == n
            else f"찾더라도 {nm_wrong} 프레임은 다른 물체를 가리킨다")
    ax.text(5.0, 2.05,
            f"{line}.\n지지면 점은 지워지지 않는다 — 연결에서만 빠진다",
            ha="center", va="center", fontsize=13, weight="bold", color=INK, linespacing=1.75)
    ax.text(5.0, 0.62, "되돌리는 스위치: ground_target(..., exclude_mask=None)",
            ha="center", fontsize=11.5, color=MUTED)

    fig.suptitle("적용 전 / 후 ① — 지지면을 연결에서 빼지 않으면 씬 전체가 한 덩어리가 된다",
                 fontsize=18, weight="bold", y=0.955)
    fig.savefig(out, dpi=130, facecolor=fs.SURFACE)
    plt.close(fig)
    return dict(nm_fail=nm_fail, fx_fail=fx_fail, nm_wrong=nm_wrong,
                nm_big=nm_big, fx_big=fx_big)


# -------------------------------------------------------------------- 04 전/후 ② 잠금
def fig_latch(fs, rows, out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(16.0, 9.0))
    fig.patch.set_facecolor(fs.SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 0.9, 1.6], hspace=0.42,
                          left=0.085, right=0.975, top=0.855, bottom=0.075)

    fr = np.array([x["frame"] for x in rows])
    names = sorted({x[k] for x in rows for k in ("fixed_name", "latch_name") if x[k] != "—"})
    idx = {nm: i for i, nm in enumerate(names)}

    ax = fig.add_subplot(gs[0, 0])
    gy = [idx.get(x["fixed_name"], np.nan) for x in rows]
    ly = [idx.get(x["latch_name"], np.nan) for x in rows]
    ax.plot(fr, gy, lw=2.8, color=RED, marker="o", ms=7,
            label="F17 이전 — 프레임마다 grounding 출력을 그대로 조작 대상으로")
    ax.plot(fr, ly, lw=3.2, color=BLUE, marker="s", ms=7, alpha=0.85,
            label="현재 — 잠금이 붙든 조작 대상")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=13)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xticks(fr[::2])
    ax.set_xlim(fr[0] - 0.8, fr[-1] + 1.2)
    ax.tick_params(labelsize=11, colors=MUTED)
    ax.set_facecolor(fs.SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=12, loc="upper left", framealpha=0.92)
    split = next((x["frame"] for x in rows if x["fixed_name"] != x["latch_name"]
                  and x["latch_name"] != "—"), None)
    if split is not None:
        ax.axvline(split, color=PURPLE, lw=1.6, ls=":")
        ax.annotate(f"프레임 {split} 부터 갈린다", xy=(split, len(names) - 0.75),
                    xytext=(split + 0.4, len(names) - 0.75), fontsize=12.5,
                    weight="bold", color=PURPLE, va="center")
    for x in rows:
        if x["attach"]:
            ax.annotate("attach", xy=(x["frame"], ly[x["frame"] - fr[0]]),
                        xytext=(x["frame"], -0.45), fontsize=12, weight="bold", color=GREEN,
                        ha="center", arrowprops=dict(arrowstyle="-|>", lw=1.8, color=GREEN))
        if x["detach"]:
            ax.annotate("detach", xy=(x["frame"], ly[x["frame"] - fr[0]]),
                        xytext=(x["frame"], -0.45), fontsize=12, weight="bold", color=GREEN,
                        ha="center", arrowprops=dict(arrowstyle="-|>", lw=1.8, color=GREEN))
    ax.set_title("grounding 이 가리키는 것과 로봇이 쥔 것은 파지 순간 갈라진다",
                 fontsize=15.5, weight="bold", pad=10)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(fr, [x["gripper"] for x in rows], lw=2.6, color=INK, marker="o", ms=5)
    ax.axhline(0.85, color=RED, ls="--", lw=1.8)
    ax.annotate("닫힘 문턱 0.85", xy=(fr[0], 0.85), xytext=(fr[0] + 0.2, 0.87),
                fontsize=11.5, color=RED)
    ax.set_ylabel("왼손 그리퍼", fontsize=12.5)
    ax.set_xticks(fr[::2])
    ax.tick_params(labelsize=11, colors=MUTED)
    ax.set_facecolor(fs.SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("잠금이 보는 유일한 로봇 신호 — 그리퍼 값 (씬은 보지 않는다)",
                 fontsize=13.5, weight="bold", pad=8)

    ax = fig.add_subplot(gs[2, 0]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.add_patch(FancyBboxPatch((0.05, 0.4), 9.9, 9.2,
                                boxstyle="round,pad=0.10,rounding_size=0.2",
                                facecolor="#ecebe7", edgecolor="#d8d7d2", lw=1.2))
    n_diff = sum(1 for x in rows if x["latch_name"] != "—" and x["fixed_name"] != x["latch_name"])
    att = [x["frame"] for x in rows if x["attach"]]
    det = [x["frame"] for x in rows if x["detach"]]
    dest = next((x["dest_name"] for x in reversed(rows) if x["dest_name"] != "—"), None)
    n_lost = sum(1 for x in rows if not x["fixed_ok"])
    ax.text(5.0, 8.55,
            f"잠금이 없으면 {n_diff} 프레임 동안 조작 대상이 엉뚱한 물체로 바뀐다",
            ha="center", fontsize=15, weight="bold", color=INK)
    ax.text(5.0, 6.35,
            f"조작 대상 잠금 {rows[-1]['latch_name']} · attach 프레임 {att if att else '—'} · "
            f"detach 프레임 {det if det else '—'}",
            ha="center", fontsize=13.5, color=INK)
    if dest is None:
        ax.text(5.0, 3.55,
                f"목적지 잠금은 이 경로에서 걸리지 않았다 — 파지 중 grounding 이 crate 를 내다가\n"
                f"한 프레임 튀고 {n_lost} 프레임은 target 을 아예 못 찾아, 연속 3 프레임이 끊긴다\n"
                "(조작 대상 잠금은 앞 3 프레임이 안정적이라 정상으로 걸린다)",
                ha="center", va="center", fontsize=12.5, color=RED, linespacing=1.75)
    else:
        ax.text(5.0, 3.55,
                f"목적지 잠금 {dest} — 잠금이 걸린 뒤 grounding 이 내는 이름은 조작 대상이 아니라\n"
                "목적지다. F11 이 결함으로 본 거동이 여기서는 신호가 된다",
                ha="center", va="center", fontsize=12.5, color=MUTED, linespacing=1.75)
    ax.text(5.0, 0.85, "되돌리는 스위치: GraspLatch 를 태우지 않는다",
            ha="center", fontsize=11.5, color=MUTED)

    fig.suptitle("적용 전 / 후 ② — 조작 대상 식별에 에피소드 상태가 없었다 (F17)",
                 fontsize=18, weight="bold", y=0.955)
    fig.savefig(out, dpi=130, facecolor=fs.SURFACE)
    plt.close(fig)
    return dict(n_diff=n_diff, attach=att, detach=det, dest=dest)


def main() -> None:
    fs = _style()

    from benchmark.ag3s.attention_lifting import GridAttentionAdapter
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.grounding_report import build_robot_model
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene
    from benchmark.trajopt.grasp_latch import CentroidIdentity, GraspLatch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--attention", default="attention_step1_run0004.npz")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/step-01-attention.json")
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--frame", type=int, default=3, help="방법론 그림에 쓸 프레임")
    ap.add_argument("--gripper-col", type=int, default=6, help="왼손 그리퍼 열 (state 기준)")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    blob = np.load(args.attention, allow_pickle=False)
    cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    cam_idx = {c: [str(x) for x in blob["cameras"]].index(c) for c in ATT_CAMS}
    A = np.asarray(blob["attention"], np.float32)

    run = load_run(args.records, limit=args.frames)
    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "pointcloud": {"range_max": 2.0}})
    adapter = GridAttentionAdapter.from_config(cfg.attention)

    latch, identity = GraspLatch(), CentroidIdentity()
    rows, vis, seen = [], None, {}
    print(f"{'i':>3}  {'grounding':<10} {'잠금':<10} {'목적지':<10} {'그리퍼':>6}  "
          f"{'후보':>4} {'최대%':>6} {'격차':>6}  {'상태':<12} 이벤트")
    for i in range(min(args.frames, len(run.steps))):
        step = run.steps[i]
        pose_scene(scene, step)
        grids = {c: A[i, di, ai, cell["layer"], cell["head"], cam_idx[c]] for c in ATT_CAMS}
        r = run_frame(scene, cfg, filter_robot, grids, float(i), adapter)
        objects = _objects(scene)
        n_pts = len(r["cloud"])

        def _info(key):
            g = r[key]
            c = None if g.target is None else np.asarray(g.target.centroid, float)
            big = max((cl.point_count for cl in g.clusters), default=0) / max(n_pts, 1)
            return c, _name_of(c, objects), g.target is not None, big

        fc, fn, fok, fbig = _info("fixed")
        nc, nn, nok, nbig = _info("nomask")

        # **명령(`actions`)이 아니라 상태(`state`)를 읽는다.** 명령은 한 프레임 앞서 닫히라고
        # 말하고(프레임 9 에 -0.007), 실제로 닫힌 것은 프레임 10 이다. 잠금이 알아야 하는 것은
        # "지금 쥐고 있는가" 이므로 상태가 맞다. 실측: 열림 1.00, 닫힘 0.72~0.74.
        st = np.asarray(step.state, float)
        gripper = float(st[args.gripper_col]) if st.size > args.gripper_col else 1.0
        a, b = _scores(r["fixed"])
        ev = latch.update(label=identity.label(fc), score=a, runner_up=b, gripper=gripper)

        # 잠긴 라벨(obj0 …)을 사람이 읽을 이름으로 되돌린다. **처음 본 이름으로 고정한다** —
        # `CentroidIdentity` 는 자리로 판정하므로 사과가 바구니 안에 들어간 뒤에는 바구니의
        # 라벨을 돌려준다. 덮어쓰면 프레임 19 에서 목적지 이름이 `crate` 에서 `apple` 로 뒤집힌다.
        lbl = identity.label(fc)
        if lbl is not None and lbl not in seen:
            seen[lbl] = fn
        rows.append(dict(frame=i, label=identity.label(fc),
                         fixed_name=fn, fixed_ok=fok, fixed_big=fbig,
                         nomask_name=nn, nomask_ok=nok, nomask_big=nbig,
                         latch_name=seen.get(ev.manipulated, "—") if ev.manipulated else "—",
                         dest_name=seen.get(ev.destination, "—") if ev.destination else "—",
                         gripper=gripper, attach=ev.attach, detach=ev.detach, phase=ev.phase.value,
                         status=r["fixed"].status.name, ratio=a / max(b, 1e-9),
                         dest_label=ev.destination))
        print(f"{i:3d}  {fn:<10} {rows[-1]['latch_name']:<10} {rows[-1]['dest_name']:<10} "
              f"{gripper:6.2f}  {len(r['fixed'].clusters):4d} {100*fbig:5.0f}%  "
              f"{a/max(b,1e-9):5.1f}배  {r['fixed'].status.name:<12} {ev.note}")

        if i == args.frame:
            vis = dict(r=r, objects=objects,
                       n_surfaces=len(r["surfaces"]), n_support=int(np.sum(r["support"])))
    scene.close()

    if vis is None:
        raise SystemExit(f"--frame {args.frame} 이 --frames {args.frames} 범위 밖이다")

    fig_method(fs, vis["r"], vis["objects"], out / "ppt-tg-01-method.png")
    fig_io(fs, vis, out / "ppt-tg-02-io.png")
    sup = fig_support(fs, vis["r"], rows, vis["objects"], out / "ppt-tg-03-support.png")
    lat = fig_latch(fs, rows, out / "ppt-tg-04-latch.png")

    print(f"\nwrote 4 figures -> {out}")
    print(f"  지지면 제외  target 못 찾음 {sup['nm_fail']} → {sup['fx_fail']} / {len(rows)}"
          f"  ·  최대 덩어리 중앙 {sup['nm_big']:.0f} % → {sup['fx_big']:.0f} %"
          f"  ·  찾더라도 다른 물체 {sup['nm_wrong']} 프레임")
    print(f"  잠금        조작 대상이 갈린 프레임 {lat['n_diff']}"
          f"  ·  attach {lat['attach']}  detach {lat['detach']}  목적지 {lat['dest']}")


if __name__ == "__main__":
    main()
