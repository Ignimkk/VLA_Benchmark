"""쥔 물체 처리의 양면과 부호 교정의 위험을 한 장에 (규칙 A).

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.live.plot_attached \\
        --verify outputs/live_test/20260922_attached/verify_attached.json \\
        --overlap outputs/live_test/20260922_attached/overlap.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ORDER = ("on_table", "lifted_50mm", "lifted_100mm", "lifted_200mm", "at_rim_30mm")
LABEL = {"on_table": "테이블 위\n(상한 · 운용 아님)", "lifted_50mm": "들어올림\n50 mm",
         "lifted_100mm": "들어올림\n100 mm", "lifted_200mm": "들어올림\n200 mm",
         "at_rim_30mm": "바구니 테두리\n+30 mm (담기)"}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    from benchmark.ag3s.experiments.common import figstyle

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", required=True)
    ap.add_argument("--overlap", required=True)
    ap.add_argument("--out",
                    default="benchmark/ag3s/docs/figures/live-test/attached-two-sides.png")
    args = ap.parse_args()

    ver = json.loads(pathlib.Path(args.verify).read_text())
    ov = json.loads(pathlib.Path(args.overlap).read_text())
    cases = [k for k in ORDER if k in ov["cases"]]

    figstyle.use_korean()
    fig = plt.figure(figsize=(16.6, 10.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.92, 1.0], hspace=0.40, wspace=0.30,
                          left=0.055, right=0.978, top=0.885, bottom=0.065)
    fig.patch.set_facecolor(figstyle.SURFACE)
    fig.text(0.5, 0.962, "쥔 물체 처리 — 양면이고 둘 다 필요하다",
             ha="center", va="center", fontsize=15.5, color=figstyle.INK)
    fig.text(0.5, 0.930,
             "로봇 쪽은 물체를 질의점으로 붙이고(E3·F19), 장애물 쪽은 같은 점을 필드에서 뺀다"
             "(A2). 한쪽만 하면 각각 다른 방식으로 깨진다.",
             ha="center", va="center", fontsize=9.4, color=figstyle.INK_2)

    # ── ① 도식: 양면 ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("① 양면", fontsize=10.8, color=figstyle.INK, loc="left", pad=8)
    ax.axis("off")
    boxes = [
        (0.70, "로봇 쪽 — 편입", figstyle.CATEGORICAL[2],
         "물체의 관측 점을 반지름 0\n질의점으로 손 링크에 붙인다\n"
         "(primitive 아님 — F19)"),
        (0.36, "장애물 쪽 — 제거", figstyle.CATEGORICAL[0],
         "같은 점을 거리장에서 뺀다\nlegacy: carve\ncuRobo: seed 제외"),
    ]
    for y, title, colour, body in boxes:
        ax.add_patch(Rectangle((0.02, y), 0.96, 0.26, facecolor="#f4f7fb",
                               edgecolor=colour, linewidth=1.6))
        ax.text(0.06, y + 0.205, title, fontsize=9.6, color=colour, va="center")
        ax.text(0.06, y + 0.075, body, fontsize=8.4, color=figstyle.INK, va="center",
                linespacing=1.6)
    ax.add_patch(FancyArrowPatch((0.5, 0.70), (0.5, 0.625), arrowstyle="-|>",
                                 mutation_scale=12, color=figstyle.INK_2, linewidth=1.2))
    ax.text(0.02, 0.28,
            "로봇 쪽만 하면 — 물체가 자기 자신에게\n부딪히고 그 행은 어떤 관절 움직임으로도\n"
            "못 푼다 (손에 강체로 붙어 있다).\n\n"
            "장애물 쪽만 하면 — 물체가 아무 충돌\n검사도 안 받는다.",
            fontsize=8.4, color=figstyle.INK_2, va="top", linespacing=1.65)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ── ② 실측: 질의점의 거리 ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_title("② 쥔 물체 질의점에서 잰 거리 — legacy carve 는 옳고 cuRobo seed 제외는 더 나빠진다",
                  fontsize=10.8, color=figstyle.INK, loc="left", pad=8)
    names = ["legacy_raw", "legacy_handled", "curobo_raw", "curobo_handled"]
    short = {"legacy_raw": "legacy\n처리 없음", "legacy_handled": "legacy\ncarve",
             "curobo_raw": "cuRobo\n처리 없음", "curobo_handled": "cuRobo\nseed 제외"}
    for i, n in enumerate(names):
        q = ver["cases"][n]["distance_at_attached_points"]
        good = ver["cases"][n]["attached_handled"] and q["n_negative"] == 0
        colour = (figstyle.CATEGORICAL[2] if good
                  else (figstyle.CATEGORICAL[7] if ver["cases"][n]["attached_handled"]
                        else figstyle.CATEGORICAL[3]))
        ax2.plot([i, i], [q["min_mm"], q["max_mm"]], color=colour, linewidth=2.4,
                 solid_capstyle="round")
        ax2.plot([i], [q["median_mm"]], marker="o", markersize=9, color=colour)
        ax2.annotate(f"중앙 {q['median_mm']:+.2f}", (i, q["median_mm"]),
                     textcoords="offset points", xytext=(13, 4), fontsize=8.2,
                     color=figstyle.INK)
        ax2.annotate(f"최소 {q['min_mm']:+.2f}", (i, q["min_mm"]),
                     textcoords="offset points", xytext=(13, -4), fontsize=8.2,
                     color=figstyle.INK)
        ax2.text(i, -43, f"음수 {q['n_negative']}/{q['n']}", ha="center", fontsize=8.4,
                 color=colour)
    ax2.axhline(0.0, color=figstyle.INK_2, linewidth=1.1, linestyle="--")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([short[n] for n in names], fontsize=8.8)
    ax2.set_ylabel("질의점에서의 거리 [mm]", fontsize=8.8)
    ax2.set_ylim(-50, 90)
    ax2.text(0.015, 0.97,
             "부호가 seed 가 아니라 질의 복셀의 TSDF 에서 온다 (builder_esdf.py:455-489) —\n"
             "seed 를 지우면 크기만 다음 표면까지로 바뀌고 부호는 사과가 정한다.",
             transform=ax2.transAxes, ha="left", va="top", fontsize=8.4,
             color=figstyle.CATEGORICAL[7], linespacing=1.7)
    figstyle.style_axes(fig, ax2)

    # ── ③ 복셀 공유 ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_title("③ 부호 교정이 숨길 수 있는 양 — 쥔 물체 복셀 중 환경 표면도 든 복셀",
                  fontsize=10.8, color=figstyle.INK, loc="left", pad=8)
    xs = np.arange(len(cases))
    w = 0.36
    for j, (tier, colour) in enumerate((("20mm", figstyle.CATEGORICAL[0]),
                                        ("5mm", figstyle.CATEGORICAL[2]))):
        vals = [100.0 * ov["cases"][c]["tiers"][tier]["shared_fraction"] for c in cases]
        ax3.bar(xs + (j - 0.5) * w, vals, width=w, color=colour,
                label=f"{tier} 계층")
        for x, v, c in zip(xs + (j - 0.5) * w, vals, cases):
            n = ov["cases"][c]["tiers"][tier]["n_shared_with_env"]
            ax3.text(x, v + 0.6, f"{n}", ha="center", fontsize=8.0, color=colour)
    ax3.set_xticks(xs)
    ax3.set_xticklabels([LABEL[c] for c in cases], fontsize=8.4)
    ax3.set_ylabel("공유 복셀 비율 [%]  (막대 위 숫자 = 복셀 수)", fontsize=8.8)
    ax3.legend(fontsize=8.4, frameon=False, loc="upper right", labelcolor=figstyle.INK_2)
    ax3.text(0.34, 0.62,
             "파내기와 부호 교정은 attach() 뒤에만 돈다 —\n"
             "그래서 '테이블 위' 는 상한이고 운용 조건이 아니다.\n\n"
             "들어올림 100 · 200 mm 에서 두 계층 모두 0 이다.",
             transform=ax3.transAxes, fontsize=8.4, color=figstyle.INK_2,
             va="top", linespacing=1.7)
    figstyle.style_axes(fig, ax3)

    # ── ④ 표 ────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_title("④ 표 — 환경까지의 최근접 거리", fontsize=10.8, color=figstyle.INK,
                  loc="left", pad=10)
    ax4.axis("off")
    y = 1.0
    ax4.text(0.0, y, "배치", fontsize=8.4, color=figstyle.INK_2)
    ax4.text(0.62, y, "최소", fontsize=8.4, color=figstyle.INK_2, ha="right")
    ax4.text(1.0, y, "20 mm 안 점", fontsize=8.4, color=figstyle.INK_2, ha="right")
    y -= 0.055
    ax4.plot([0.0, 1.0], [y + 0.03, y + 0.03], color=figstyle.GRID_INK, linewidth=1.0)
    for c in cases:
        q = ov["cases"][c]["distance_to_env_surface"]
        n20 = ov["cases"][c]["n_within_20mm_of_env"]
        col = (figstyle.CATEGORICAL[7] if q["min_mm"] < 5.0
               else (figstyle.CATEGORICAL[3] if q["min_mm"] < 50.0
                     else figstyle.CATEGORICAL[2]))
        ax4.text(0.0, y, LABEL[c].replace("\n", " "), fontsize=8.2, color=figstyle.INK)
        ax4.text(0.62, y, f"{q['min_mm']:.2f} mm", fontsize=8.2, ha="right", color=col)
        ax4.text(1.0, y, f"{n20}", fontsize=8.2, ha="right", color=col)
        y -= 0.082
    y -= 0.03
    ax4.plot([0.0, 1.0], [y + 0.04, y + 0.04], color=figstyle.GRID_INK, linewidth=1.0)
    o = ov["observation"]
    ax4.text(0.0, y - 0.01,
             f"참값: MuJoCo body segmentation.\n"
             f"관측 표면 {o['n_surface_points']:,} 점 중\n"
             f"사과 {o['n_apple_points']:,} · 환경 {o['n_env_points']:,}.\n"
             f"쥔 물체 질의점 {o['n_attached_query_points']} 개\n(10 mm 복셀로 솎은 것 — F19).\n\n"
             "들어올림 100 mm 이상에서 두 계층 모두\n공유 0 — 부호 교정이 숨길 것이 없다.\n"
             "담기 국면만 남는다.",
             fontsize=8.1, color=figstyle.INK_2, va="top", linespacing=1.6)
    ax4.set_xlim(-0.02, 1.02)
    ax4.set_ylim(y - 0.62, 1.05)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=figstyle.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
