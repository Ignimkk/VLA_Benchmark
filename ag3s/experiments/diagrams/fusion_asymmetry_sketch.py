"""두 융합 방식이 왜 다른 답을 낼 수 있는가 — 개념 도식.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.diagrams.fusion_asymmetry_sketch

측정이 아니라 **설명 그림**이다. AG3S 는 카메라 셋을 한 씬으로 합치는 방법을 둘 갖고 있고,
둘은 "지운다" 는 능력에서 갈린다. 실제로 얼마나 갈리는지는 따로 재야 한다 (Step 7).
"""

import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments.common import figstyle
    figstyle.use_korean()


def main() -> None:
    _style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

    fig, axs = plt.subplots(1, 3, figsize=(19.5, 6.6))

    # 공통 배치 — 위에서 본 그림
    head = np.array([0.15, 0.85])       # 머리 카메라
    wrist = np.array([0.88, 0.72])      # 손목 카메라
    apple = np.array([0.50, 0.45])      # 사과
    table_y = 0.30

    def scene(a, title):
        a.set_xlim(0, 1); a.set_ylim(0.05, 1.0); a.set_aspect("equal"); a.axis("off")
        a.set_title(title, fontsize=12.5)
        a.add_patch(Rectangle((0.08, table_y - 0.06), 0.84, 0.06, fc="#d9c7a8", ec="0.4"))
        a.text(0.5, table_y - 0.10, "테이블", ha="center", fontsize=10, color="0.35")
        for p, name, c in ((head, "머리 카메라", "tab:blue"), (wrist, "손목 카메라", "tab:purple")):
            a.plot(*p, "^", ms=15, color=c, mec="k", zorder=5)
            a.text(p[0], p[1] + 0.05, name, ha="center", fontsize=10, color=c, fontweight="bold")

    # ---------- (a) 무슨 일이 일어나는가 ----------
    a = axs[0]
    scene(a, "(a) 같은 사과, 두 카메라가 다르게 본다")
    a.add_patch(Circle(apple, 0.055, fc="#d33", ec="k", lw=1.5, zorder=4))
    a.text(apple[0], apple[1] - 0.10, "사과 (실제로 있다)", ha="center", fontsize=10,
           color="#a11", fontweight="bold")

    # 머리 광선 — 사과에 맞는다
    a.add_patch(FancyArrowPatch(head, apple + np.array([-0.03, 0.045]), arrowstyle="-|>",
                                mutation_scale=15, color="tab:blue", lw=2.2, zorder=3))
    a.text(0.24, 0.62, "맞았다\n→ 표면", fontsize=9.5, color="tab:blue", fontweight="bold")

    # 손목 광선 — 사과 옆을 스쳐 테이블에 맞는다 (사과 자리를 통과)
    tail = np.array([0.38, table_y])
    a.add_patch(FancyArrowPatch(wrist, tail, arrowstyle="-|>", mutation_scale=15,
                                color="tab:purple", lw=2.2, zorder=3))
    a.plot([apple[0]], [apple[1]], "x", ms=13, color="tab:purple", mew=3, zorder=6)
    a.text(0.60, 0.55, "이 자리를\n지나갔다\n→ 빈 공간", fontsize=9.5, color="tab:purple",
           fontweight="bold")

    # ---------- (b) 방법 A ----------
    a = axs[1]
    scene(a, "(b) 방법 A — 점을 모은다 (multiview.fuse)")
    a.add_patch(Circle(apple, 0.055, fc="#d33", ec="k", lw=1.5, zorder=4))
    for dx, dy in ((-0.03, 0.03), (0.0, 0.045), (0.03, 0.03)):
        a.plot(apple[0] + dx, apple[1] + dy, "o", ms=6, color="tab:blue", mec="k", zorder=5)
    a.text(apple[0], apple[1] - 0.11, "사과가 남는다", ha="center", fontsize=11,
           color="#0a0", fontweight="bold")
    a.text(0.5, 0.93, "한 대라도 점을 찍으면 그 칸은 '있음'\n지우는 기능이 없다 — 합집합",
           ha="center", fontsize=10.5, color="0.25")
    a.text(0.5, 0.13, "소비자: 지지면 적합 · target grounding",
           ha="center", fontsize=10, color="0.35", style="italic")

    # ---------- (c) 방법 B ----------
    a = axs[2]
    scene(a, "(c) 방법 B — 광선을 붓는다 (TSDF 적분)")
    a.add_patch(Circle(apple, 0.055, fc="none", ec="0.6", lw=1.6, ls="--", zorder=4))
    a.text(apple[0], apple[1] + 0.005, "?", ha="center", va="center", fontsize=20,
           color="0.45", fontweight="bold")
    a.text(apple[0], apple[1] - 0.11, "사과가 지워질 수 있다", ha="center", fontsize=11,
           color="#c00", fontweight="bold")
    a.text(0.5, 0.93, "광선이 지나간 구간은 '빈 공간'\n한 대가 지나가면 다른 대의 표면을 깎는다",
           ha="center", fontsize=10.5, color="0.25")
    a.text(0.5, 0.13, "소비자: 충돌 필드 (ESDF)",
           ha="center", fontsize=10, color="0.35", style="italic")

    fig.suptitle("카메라 셋을 한 씬으로 합치는 방법이 둘이고, 둘은 '지운다' 는 능력에서 갈린다",
                 fontsize=13.5)
    fig.text(0.5, 0.015,
             "그래서 같은 물체를 두고 AG3S 의 두 절반이 다른 말을 할 수 있다 — "
             "grounding 은 '사과가 여기 있다', 충돌 필드는 '거기 비었다'. "
             "그것이 F13 이 지적한 조건 그대로다.",
             ha="center", fontsize=11.5, color="#a11")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    o = OUT / "step7-fusion-asymmetry.png"
    fig.savefig(o, dpi=110, bbox_inches="tight")
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
