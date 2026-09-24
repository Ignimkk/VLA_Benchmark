"""부호 교정의 실제 파지 검증과 문턱 결정 (규칙 A).

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.live.plot_attached_fix \\
        --dir outputs/live_test/20260922_attached

`safe_replay` 가 낸 JSON 만 읽는다. 다시 측정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

RUNS = (("legacy", "safe_replay_legacy.json", "legacy (carve)"),
        ("thr1.0", "safe_replay_curobo_thr1.0.json", "cuRobo 문턱 1.0"),
        ("thr1.5", "safe_replay_curobo_thr1.5.json", "cuRobo 문턱 1.5 (채택)"),
        ("thr3.0", "safe_replay_curobo_thr3.0.json", "cuRobo 문턱 3.0"))
PHASES = (("파지 직후", 11, 12), ("운반", 13, 17), ("담기", 18, 21))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmark.ag3s.experiments.common import figstyle

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out",
                    default="benchmark/ag3s/docs/figures/live-test/attached-sign-threshold.png")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    runs = {k: json.loads((d / f).read_text()) for k, f, _ in RUNS}
    lab = {k: l for k, _, l in RUNS}
    colour = {"legacy": figstyle.CATEGORICAL[3], "thr1.0": figstyle.CATEGORICAL[4],
              "thr1.5": figstyle.CATEGORICAL[0], "thr3.0": figstyle.CATEGORICAL[7]}

    def shared(run, lo, hi):
        tot = 0
        for r in runs[run]["frames"]:
            if not (lo <= r["seq"] <= hi):
                continue
            for t in (r["modules"].get("attached_sign_correction") or []):
                tot += t.get("n_shared_left_alone", 0)
        return tot

    figstyle.use_korean()
    fig = plt.figure(figsize=(16.4, 9.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.94], hspace=0.44, wspace=0.28,
                          left=0.055, right=0.978, top=0.868, bottom=0.075)
    fig.patch.set_facecolor(figstyle.SURFACE)
    fig.text(0.5, 0.958,
             "쥔 물체 부호 교정 — 문턱을 실제 파지에서 쓸어 정했다",
             ha="center", va="center", fontsize=15.5, color=figstyle.INK)
    fig.text(0.5, 0.925,
             "run_0004 22 청크. 잠금이 attach() 를 부르는 실제 파지 구간이다 — "
             "점만 옮긴 합성 배치는 TSDF 증거가 원래 자리에 남아 이 기전을 재지 못한다.",
             ha="center", va="center", fontsize=9.4, color=figstyle.INK_2)

    # ── ① 국면별 공유 복셀 ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    ax.set_title("① 국면별 공유 복셀 — 부호를 고치지 않고 남긴 복셀 수",
                 fontsize=10.8, color=figstyle.INK, loc="left", pad=8)
    keys = [k for k, _, _ in RUNS if k != "legacy"]
    xs = np.arange(len(PHASES))
    w = 0.26
    for j, k in enumerate(keys):
        vals = [shared(k, lo, hi) for _, lo, hi in PHASES]
        ax.bar(xs + (j - 1) * w, vals, width=w, color=colour[k], label=lab[k])
        for x, v in zip(xs + (j - 1) * w, vals):
            ax.text(x, v + 2, f"{v}", ha="center", fontsize=8.4, color=colour[k])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n(청크 {lo}~{hi})" for n, lo, hi in PHASES], fontsize=8.8)
    ax.set_ylabel("공유 복셀 수 (두 계층 합)", fontsize=8.8)
    ax.legend(fontsize=8.4, frameon=False, loc="upper left", labelcolor=figstyle.INK_2)
    ax.text(0.99, 0.95,
            "1.0 — 어느 국면에서도 공유를 못 잡는다. 규칙이 무력해져 전면 교정과 같아진다.\n"
            "1.5 — 파지 직후 157 · 운반 0 · 담기 26. 원하는 구분이다.\n"
            "3.0 — 운반에서도 26. 쥔 물체가 필드에 남는다.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.4,
            color=figstyle.INK_2, linespacing=1.7)
    figstyle.style_axes(fig, ax)

    # ── ② 프레임별 위반 ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("② 프레임별 최악 위반\n세 문턱이 완전히 같다",
                  fontsize=10.8, color=figstyle.INK, loc="left", pad=8)
    seqs = [r["seq"] for r in runs["legacy"]["frames"]]
    for k in ("legacy", "thr1.0", "thr1.5", "thr3.0"):
        v = [r["max_violation_mm"] for r in runs[k]["frames"]]
        ls = "-" if k == "legacy" else (":" if k != "thr1.5" else "-")
        lw = 1.8 if k in ("legacy", "thr1.5") else 1.1
        ax2.plot(seqs, v, linestyle=ls, linewidth=lw, marker="o", markersize=3.4,
                 color=colour[k], label=lab[k])
    ax2.set_xlabel("청크", fontsize=8.8)
    ax2.set_ylabel("최악 위반 [mm]", fontsize=8.8)
    ax2.legend(fontsize=7.4, frameon=False, loc="upper center",
               bbox_to_anchor=(0.62, 1.0), labelcolor=figstyle.INK_2)
    figstyle.style_axes(fig, ax2)

    # ── ③ 청크별 공유 복셀 (문턱 1.5) ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_title("③ 문턱 1.5 의 청크별 내역 — 파지 직후와 담기에만 공유가 있다",
                  fontsize=10.8, color=figstyle.INK, loc="left", pad=8)
    rows = {"coarse": {}, "fine": {}}
    for r in runs["thr1.5"]["frames"]:
        for t in (r["modules"].get("attached_sign_correction") or []):
            if t.get("n_attached_voxels", 0):
                rows[t["tier"]][r["seq"]] = (t["n_sign_forced"], t["n_shared_left_alone"])
    active = sorted(set(rows["coarse"]) | set(rows["fine"]))
    xs3 = np.arange(len(active))
    for j, (tier, cf) in enumerate((("coarse", figstyle.CATEGORICAL[0]),
                                    ("fine", figstyle.CATEGORICAL[2]))):
        forced = [rows[tier].get(s, (0, 0))[0] for s in active]
        sh = [rows[tier].get(s, (0, 0))[1] for s in active]
        off = (j - 0.5) * 0.4
        ax3.bar(xs3 + off, forced, width=0.38, color=cf, label=f"{tier} 부호 강제")
        ax3.bar(xs3 + off, sh, width=0.38, bottom=forced, color=cf, alpha=0.38,
                hatch="///", edgecolor="white", linewidth=0.0,
                label=f"{tier} 공유 (그대로 둠)")
    for n, lo, hi in PHASES:
        idx = [i for i, s in enumerate(active) if lo <= s <= hi]
        if idx:
            ax3.text(np.mean(idx), -11, n, ha="center", fontsize=8.4,
                     color=figstyle.INK_2)
    ax3.set_xticks(xs3)
    ax3.set_xticklabels(active, fontsize=8.2)
    ax3.set_xlabel("청크", fontsize=8.8, labelpad=22)
    ax3.set_ylabel("쥔 물체 복셀 수", fontsize=8.8)
    ax3.set_ylim(-14, 105)
    ax3.legend(fontsize=7.8, frameon=False, ncol=2, loc="upper right",
               labelcolor=figstyle.INK_2)
    figstyle.style_axes(fig, ax3)

    # ── ④ 표 ────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_title("④ 표 — 문턱별 결과", fontsize=10.8, color=figstyle.INK,
                  loc="left", pad=10)
    ax4.axis("off")
    hdr = ["run", "safe", "viol", "최악", "운반", "담기"]
    xcol = (0.0, 0.42, 0.56, 0.72, 0.87, 1.0)
    y = 1.0
    for x, h in zip(xcol, hdr):
        ax4.text(x, y, h, fontsize=8.3, color=figstyle.INK_2,
                 ha="left" if x == 0.0 else "right")
    y -= 0.07
    ax4.plot([0, 1], [y + 0.035, y + 0.035], color=figstyle.GRID_INK, linewidth=1.0)
    for k, _, l in RUNS:
        fr = runs[k]["frames"]
        vals = [l, str(sum(1 for r in fr if r["safe"])),
                str(sum(1 for r in fr if r["trajopt_status"] == "violated")),
                f"{max(r['max_violation_mm'] for r in fr):.1f}",
                str(shared(k, 13, 17)), str(shared(k, 18, 21))]
        col = figstyle.CATEGORICAL[0] if k == "thr1.5" else figstyle.INK
        for x, v, h in zip(xcol, vals, hdr):
            ax4.text(x, y, v, fontsize=8.2, color=col,
                     ha="left" if x == 0.0 else "right")
        y -= 0.082
    y -= 0.03
    ax4.plot([0, 1], [y + 0.04, y + 0.04], color=figstyle.GRID_INK, linewidth=1.0)
    ax4.text(0.0, y - 0.01,
             "세 문턱의 trajopt 판정이 프레임\n단위로 완전히 같다 — 문턱은\n"
             "최적화 결과가 아니라 숨김 위험만\n바꾼다. 그래서 운용을 깨지 않는\n"
             "가장 큰 값을 고른다.\n\n"
             "1.5 는 물리적으로도 의미가 있다 —\n이웃 복셀 중심의 표면이 우리 복셀\n"
             "안으로 반 칸까지 뻗을 수 있다.",
             fontsize=8.1, color=figstyle.INK_2, va="top", linespacing=1.6)
    ax4.set_xlim(-0.02, 1.02)
    ax4.set_ylim(y - 0.62, 1.05)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=figstyle.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
