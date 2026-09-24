"""cuRobo 별도 기준선의 시각화 — legacy · cuRobo 단일 계층 · cuRobo 2계층 (규칙 A).

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.live.plot_baseline \\
        --legacy /tmp/base_legacy.json \\
        --curobo outputs/live_test/20260922_i2_baseline/curobo_baseline.json \\
        --curobo-coarse outputs/live_test/20260922_i2_baseline/curobo_baseline_coarse_only.json

`esdf_rollout` 이 낸 JSON 만 읽는다. 다시 측정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

VOXELS = {"legacy": 547_200, "curobo1": 2_097_152, "curobo2": 4_194_304}
#: 실측한 gradient 캐시 생성 시간 (`np.gradient` 3 성분). 프레임마다 필드가 새로 만들어지므로
#: 매 프레임 든다.
GRAD_CACHE_MS = {"legacy": 21.6, "curobo1": 55.9, "curobo2": 111.7}
LABEL = {"legacy": "legacy\n20 mm 단일\n(회귀 기준선)",
         "curobo1": "cuRobo\n20 mm 단일",
         "curobo2": "cuRobo\n20+5 mm 2계층\n(채택)"}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmark.ag3s.experiments.common import figstyle

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--legacy", required=True)
    ap.add_argument("--curobo", required=True)
    ap.add_argument("--curobo-coarse", required=True)
    ap.add_argument("--out",
                    default="benchmark/ag3s/docs/figures/live-test/i2-curobo-baseline.png")
    args = ap.parse_args()

    runs = {"legacy": json.loads(pathlib.Path(args.legacy).read_text()),
            "curobo1": json.loads(pathlib.Path(args.curobo_coarse).read_text()),
            "curobo2": json.loads(pathlib.Path(args.curobo).read_text())}
    fr = {k: v["frames"] for k, v in runs.items()}
    order = ("legacy", "curobo1", "curobo2")
    colour = {"legacy": figstyle.CATEGORICAL[3], "curobo1": figstyle.CATEGORICAL[4],
              "curobo2": figstyle.CATEGORICAL[0]}

    def col(k, key):
        return np.array([r[key] for r in fr[k]], float)

    figstyle.use_korean()
    fig = plt.figure(figsize=(16.4, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.94], hspace=0.42, wspace=0.26,
                          left=0.055, right=0.978, top=0.878, bottom=0.095)
    fig.patch.set_facecolor(figstyle.SURFACE)
    fig.text(0.5, 0.960, "cuRobo 별도 기준선 — 15 청크, run_0004, 실측 attention",
             ha="center", va="center", fontsize=15.5, color=figstyle.INK)
    fig.text(0.5, 0.928,
             "legacy 기준선은 고정하고 cuRobo 는 따로 뜬다 (2026-09-22 판정). "
             "단일 계층은 대조군 — 2계층이 값어치를 하는지 보는 축이다.",
             ha="center", va="center", fontsize=9.5, color=figstyle.INK_2)

    # ── ① 프레임별 최적화 후 여유거리 ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    ax.set_title("① 프레임별 최적화 후 최악 여유거리 — 0 은 이진 딱지일 뿐이다",
                 fontsize=10.8, color=figstyle.INK, loc="left", pad=8)
    x = col("legacy", "i")
    for k in order:
        ax.plot(x, col(k, "clearance_after_mm"), marker="o", markersize=5,
                linewidth=1.8, color=colour[k], label=LABEL[k].replace("\n", " "))
    ax.axhline(0.0, color=figstyle.CATEGORICAL[7], linewidth=1.3, linestyle="--")
    ax.axhspan(-4.5, 4.5, color=figstyle.CATEGORICAL[7], alpha=0.07)
    ax.text(0.15, 4.9, "±4.5 mm 띠 — 여기서 feasible/violated 딱지가 갈린다",
            fontsize=8.2, color=figstyle.INK_2)
    ax.set_xlabel("청크", fontsize=8.8)
    ax.set_ylabel("여유거리 after [mm]", fontsize=8.8)
    ax.set_xticks(x[::2])
    ax.legend(fontsize=8.4, frameon=False, ncol=3, labelcolor=figstyle.INK_2,
              loc="upper center", bbox_to_anchor=(0.5, -0.13))
    figstyle.style_axes(fig, ax)

    # ── ② 단일 계층의 낙관 ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("② 단일 계층 − 2계층\n양수 = 단일 계층이 더 멀다고 답함",
                  fontsize=10.8, color=figstyle.INK, loc="left", pad=8)
    diff = col("curobo1", "clearance_after_mm") - col("curobo2", "clearance_after_mm")
    bars = ax2.bar(x, diff, width=0.7,
                   color=[figstyle.CATEGORICAL[7] if v > 0 else figstyle.CATEGORICAL[2]
                          for v in diff])
    ax2.axhline(0.0, color=figstyle.INK_2, linewidth=1.0)
    ax2.axhline(float(np.median(diff)), color=figstyle.CATEGORICAL[7], linewidth=1.2,
                linestyle=":")
    ax2.annotate(f"중앙 {np.median(diff):+.2f} mm", xy=(7.5, float(np.median(diff))),
                 xytext=(0.52, 0.42), textcoords="axes fraction", fontsize=8.4,
                 color=figstyle.CATEGORICAL[7],
                 arrowprops=dict(arrowstyle="-", color=figstyle.CATEGORICAL[7],
                                 linewidth=0.8))
    ax2.set_xlabel("청크", fontsize=8.8)
    ax2.set_ylabel("차이 [mm]", fontsize=8.8)
    ax2.text(0.02, 0.97,
             f"{int((diff > 0).sum())}/15 프레임에서 단일 계층이 낙관적.\n"
             "해소 15/15 는 품질이 아니라 이것 때문이다.",
             transform=ax2.transAxes, fontsize=8.2, color=figstyle.INK_2, va="top",
             linespacing=1.6)
    figstyle.style_axes(fig, ax2)

    # ── ③ TO 시간 대 격자 복셀 수 ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("③ 최적화 시간은 격자 복셀 수를 따른다\n(계층 수가 아니다)",
                  fontsize=10.8, color=figstyle.INK, loc="left", pad=8)
    for k in order:
        to = np.median(col(k, "to_ms"))
        ax3.scatter([VOXELS[k] / 1e6], [to], s=110, color=colour[k], zorder=3)
        ax3.annotate(f"{to:.0f} ms", (VOXELS[k] / 1e6, to), textcoords="offset points",
                     xytext=(9, -3), fontsize=8.4, color=figstyle.INK)
    xs = np.array([VOXELS[k] / 1e6 for k in order])
    ys = np.array([np.median(col(k, "to_ms")) for k in order])
    ax3.plot(xs, ys, linewidth=1.2, color=figstyle.GRID_INK, zorder=1)
    ax3.bar(xs, [GRAD_CACHE_MS[k] for k in order], width=0.28,
            color=figstyle.GRID_INK, zorder=2,
            label="그중 gradient 캐시 생성 (실측)")
    ax3.set_xlabel("거리장 복셀 수 [백만]", fontsize=8.8)
    ax3.set_ylabel("TO 중앙 [ms]", fontsize=8.8)
    ax3.legend(fontsize=7.8, frameon=False, loc="upper left", labelcolor=figstyle.INK_2)
    figstyle.style_axes(fig, ax3)

    # ── ④ 표 ─────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.set_title("④ 표 — 세 기준선", fontsize=10.8, color=figstyle.INK,
                  loc="left", pad=10)
    ax4.axis("off")
    rows = [
        ("거리장 복셀 수", [f"{VOXELS[k]:,}" for k in order]),
        ("위반으로 시작", [f"{int((col(k,'clearance_before_mm')<0).sum())} / 15" for k in order]),
        ("해소", [str(int(((col(k,'clearance_before_mm')<0)
                           & (col(k,'clearance_after_mm')>=0)).sum())) for k in order]),
        ("개선", [str(int((col(k,'clearance_after_mm')
                          > col(k,'clearance_before_mm')+1e-6).sum())) for k in order]),
        ("SQP 상태 feasible / violated",
         [f"{sum(1 for r in fr[k] if r['status']=='feasible')} / "
          f"{sum(1 for r in fr[k] if r['status']=='violated')}" for k in order]),
        ("여유거리 after 중앙", [f"{np.median(col(k,'clearance_after_mm')):+.2f} mm" for k in order]),
        ("여유거리 after 최악", [f"{col(k,'clearance_after_mm').min():+.2f} mm" for k in order]),
        ("여유거리 before 중앙", [f"{np.median(col(k,'clearance_before_mm')):+.1f} mm" for k in order]),
        ("TO 중앙", [f"{np.median(col(k,'to_ms')):.0f} ms" for k in order]),
        ("AG3S 중앙 (프레임 0 제외)",
         [f"{np.median(col(k,'ag3s_ms')[1:]):.0f} ms" for k in order]),
        ("SQP 반복", [f"{int(np.median(col(k,'iterations')))}" for k in order]),
    ]
    xcol = (0.52, 0.70, 0.88)
    y = 1.0
    for j, k in enumerate(order):
        ax4.text(xcol[j], y + 0.075, LABEL[k], ha="center", va="bottom", fontsize=8.4,
                 color=colour[k], linespacing=1.4)
    ax4.plot([0.0, 1.0], [y + 0.038, y + 0.038], color=figstyle.GRID_INK, linewidth=1.0)
    for lab, vals in rows:
        ax4.text(0.0, y, lab, fontsize=8.7, va="center", color=figstyle.INK_2)
        for j, v in enumerate(vals):
            ax4.text(xcol[j], y, v, fontsize=8.7, va="center", ha="center",
                     color=figstyle.INK)
        y -= 0.082
    ax4.text(0.0, y - 0.02,
             "채택은 2계층이다. 단일 계층의 해소 15/15 는 중앙 +3.36 mm 낙관의 결과이고, "
             "그 낙관은 C5(거친 20 mm 가 판정 지점에서 여유거리를 +7.56 mm 낙관적으로 답한다)와\n"
             "같은 것이다. legacy 기준선(해소 14 · feasible 8)은 고정해 둔다 — 두 backend 는 "
             "다른 것을 재므로 한 숫자로 합치지 않는다.",
             fontsize=8.4, color=figstyle.INK_2, va="top", linespacing=1.7)
    ax4.set_xlim(-0.015, 1.015)
    ax4.set_ylim(y - 0.22, 1.20)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=figstyle.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
