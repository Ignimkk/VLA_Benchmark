"""16D 회귀 기준선 시각화 — 왜 head 한 대로는 기준선이 안 되는가 (규칙 A).

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.live.plot_baseline16d \\
        --root outputs/live_test/20260924_baseline16d

`esdf_rollout` 이 낸 JSON 만 읽는다. 세 판.

1. **프레임별 여유거리** — head 한 대 대 세 대. head 쪽은 시작부터 전부 양수라 **TO 가 고칠
   것이 없다**. 고칠 것이 없는 판은 회귀를 못 잡으므로 기준선이 될 수 없다.
2. **반복 안정성** — 같은 입력 4 회. 딱지(개수)는 고정, 프레임별 값은 흔들린다.
   `sqp.time_budget_ms` 가 벽시계 마감이라 반복 수가 부하를 따른다.
3. **표** — 옛 기준선(은퇴)과 새 기준선을 나란히.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmark.ag3s.experiments.common import figstyle

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out",
                    default="benchmark/ag3s/docs/figures/live-test/baseline-16d.png")
    args = ap.parse_args()
    figstyle.use_korean()
    root = pathlib.Path(args.root)

    head = json.loads((root / "baseline16d.json").read_text())
    allc = json.loads((root / "baseline16d_allcams.json").read_text())
    repeats = [allc] + [json.loads((root / f"repeat_{i}.json").read_text())
                        for i in (1, 2, 3)]
    if (root / "repeat_4_stablepath.json").exists():
        repeats.append(json.loads((root / "repeat_4_stablepath.json").read_text()))

    fig = plt.figure(figsize=(13.2, 12.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.78, 0.6], hspace=0.44)
    ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))
    figstyle.style_axes(fig, [ax0, ax1])

    # ── 1. 프레임별 여유거리 ──────────────────────────────────────────────────
    for run, label, slot in ((head, "카메라 head 한 대", 1), (allc, "카메라 세 대", 0)):
        i = [f["i"] for f in run["frames"]]
        before = [f["clearance_before_mm"] for f in run["frames"]]
        after = [f["clearance_after_mm"] for f in run["frames"]]
        c = figstyle.CATEGORICAL[slot]
        ax0.plot(i, before, lw=1.4, marker="o", ms=3.4, color=c, label=f"{label} — TO 앞")
        ax0.plot(i, after, lw=1.4, ls="--", marker="^", ms=3.4, color=c,
                 label=f"{label} — TO 뒤")
        n_target = sum(1 for f in run["frames"] if f["target"])
        ax0.text(0.985, 0.06 + 0.10 * slot,
                 f"{label}: grounding 성공 {n_target}/{len(i)} 프레임",
                 transform=ax0.transAxes, ha="right", fontsize=7.6, color=c)
    ax0.axhline(0, color=figstyle.INK, lw=1.0, zorder=1)
    ax0.text(0.008, 0.5, "0 = 마진 경계 (아래가 위반)", transform=ax0.transAxes,
             fontsize=7, color=figstyle.INK_2, va="bottom")
    ax0.set_xlabel("청크 i")
    ax0.set_ylabel("최소 여유거리 (mm)")
    ax0.set_title("head 한 대로는 시작부터 전부 양수 — TO 가 고칠 것이 없다 (기준선 구실을 못 한다)",
                  fontsize=9.6, color=figstyle.INK)
    ax0.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    ax0.grid(axis="y", color=figstyle.GRID_INK, lw=0.6, zorder=0)

    # ── 2. 반복 안정성 ────────────────────────────────────────────────────────
    idx = np.arange(len(repeats[0]["frames"]))
    for r, run in enumerate(repeats):
        ax1.scatter(idx, [f["clearance_after_mm"] for f in run["frames"]],
                    s=22, alpha=0.8, color=figstyle.SEQ_BLUE[3 + 2 * r],
                    label=f"실행 {r + 1}", zorder=3)
    spread = [max(run["frames"][k]["clearance_after_mm"] for run in repeats)
              - min(run["frames"][k]["clearance_after_mm"] for run in repeats)
              for k in idx]
    ax1b = ax1.twinx()
    ax1b.bar(idx, spread, width=0.5, color=figstyle.CATEGORICAL[1], alpha=0.30, zorder=1)
    ax1b.set_ylabel("실행 간 폭 (mm)", fontsize=8, color=figstyle.CATEGORICAL[1])
    ax1b.tick_params(colors=figstyle.CATEGORICAL[1], labelsize=7.4)
    for sp in ("top", "left"):
        ax1b.spines[sp].set_visible(False)
    ax1b.spines["right"].set_color(figstyle.GRID_INK)
    ax1b.set_facecolor("none")
    ax1.set_xlabel("청크 i")
    ax1.set_ylabel("TO 뒤 여유거리 (mm)")
    counts = {(tuple(sorted(
        (f["status"] for f in run["frames"]))) ) for run in repeats}
    ax1.set_title(
        f"같은 입력 {len(repeats)} 회 — 딱지는 고정(상태 조합 {len(counts)} 종), "
        "프레임별 값은 흔들린다 (sqp.time_budget_ms 가 벽시계 마감이다)",
        fontsize=9.6, color=figstyle.INK)
    ax1.legend(fontsize=7, frameon=False, ncol=len(repeats), loc="upper left")
    ax1.grid(axis="y", color=figstyle.GRID_INK, lw=0.6, zorder=0)

    # ── 3. 표 ────────────────────────────────────────────────────────────────
    ax2.set_axis_off()

    def summary(run):
        fr = run["frames"]
        started = sum(1 for f in fr if f["clearance_before_mm"] < 0)
        resolved = sum(1 for f in fr if f["clearance_before_mm"] < 0
                       and f["clearance_after_mm"] >= 0)
        improved = sum(1 for f in fr
                       if f["clearance_after_mm"] > f["clearance_before_mm"])
        feas = sum(1 for f in fr if f["status"] == "feasible")
        return (f"{started}/{len(fr)}", str(resolved), str(improved),
                f"{feas} / {len(fr) - feas}",
                f"{sum(1 for f in fr if f['target'])}/{len(fr)}")

    rows = [
        ["옛 기준선 (14D, run_0004, head)", "은퇴 — 실행 불가", "—", "—", "14 열 기록 대 16 열 layout", "—"],
        ["새 시도 (16D, head 한 대)", *summary(head)],
        ["새 기준선 (16D, 카메라 세 대)", *summary(allc)],
    ]
    header = ["기준선", "위반으로 시작", "해소", "개선", "feasible / violated", "grounding 성공"]
    tb = ax2.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.6)
    tb.scale(1, 1.75)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        cell.set_facecolor(figstyle.SURFACE)
        cell.set_text_props(color=figstyle.INK if r else figstyle.INK)
        if c == 0 and r:
            cell.set_text_props(color=figstyle.INK_2, ha="left")
        if r == 3:
            cell.set_facecolor("#f2f7fd")
    ax2.set_title("옛 기준선은 실행 불가, head 한 대는 고칠 것이 없다 — 세 대만 기준선이 된다",
                  fontsize=9.6, color=figstyle.INK)

    fig.suptitle("회귀 기준선을 16D 로 — held-out ep1800 · 15 청크", fontsize=12,
                 color=figstyle.INK, y=0.985)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=figstyle.SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
