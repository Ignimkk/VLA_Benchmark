"""I2 검증의 시각화 — 두 backend 짝 비교 (규칙 A: 실제 씬 · 그래프 · 표).

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.live.plot_i2 \\
        --dir outputs/live_test/20260922_i2_backend --seeds 101 202 303

`verify_backend.py` 가 낸 JSON 만 읽는다. 다시 측정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

STAGE_ORDER = ("scene_reconstruction", "target_grounding", "esdf",
               "constraint_generation", "support_surface")
STAGE_LABEL = {"scene_reconstruction": "점군 재구성", "target_grounding": "target grounding",
               "esdf": "거리장 (ESDF)", "constraint_generation": "제약 생성",
               "support_surface": "지지면"}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from benchmark.ag3s.experiments.common import figstyle

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=(101, 202, 303))
    ap.add_argument("--primary", type=int, default=101)
    ap.add_argument("--out", default="benchmark/ag3s/docs/figures/live-test/i2-backend-swap.png")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    rep = {s: json.loads((d / f"verify_backend_seed{s}.json").read_text())
           for s in args.seeds}
    r0 = rep[args.primary]

    figstyle.use_korean()
    fig = plt.figure(figsize=(16.6, 10.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.06], hspace=0.40, wspace=0.28,
                          left=0.055, right=0.978, top=0.885, bottom=0.075)
    fig.patch.set_facecolor(figstyle.SURFACE)
    fig.text(0.5, 0.962,
             "I2 — pipeline._build_esdf 가 cuRobo 를 부른다. 같은 씬에서 두 backend 짝 비교",
             ha="center", va="center", fontsize=15.5, color=figstyle.INK)
    fig.text(0.5, 0.932,
             "새 MuJoCo 씬 seed 3개. 같은 관측을 legacy numpy 필드와 cuRobo 2계층 필드에 "
             "각각 넣었다. 저장 기록 입력 0건.",
             ha="center", va="center", fontsize=9.5, color=figstyle.INK_2)

    # ── ① 그래프: delta 를 세 표본으로 ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("① cuRobo − legacy 거리 차이\n표본을 가르면 +171 mm 가 사라진다",
                 fontsize=10.6, color=figstyle.INK, loc="left", pad=8)
    keys = ("all_spheres", "unsaturated_only", "near_band_only")
    labs = ("전체\n120 구", "포화 제외\n104 구", "근접 띠\n50 구")
    colours = (figstyle.CATEGORICAL[7], figstyle.CATEGORICAL[0], figstyle.CATEGORICAL[2])
    for i, (k, colour) in enumerate(zip(keys, colours)):
        med = [rep[s]["backend_delta"][k]["median_mm"] for s in args.seeds]
        lo = [rep[s]["backend_delta"][k]["p05_mm"] for s in args.seeds]
        hi = [rep[s]["backend_delta"][k]["p95_mm"] for s in args.seeds]
        ax.errorbar([i] * len(args.seeds), med,
                    yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
                    fmt="o", markersize=8, capsize=5, color=colour, linestyle="none")
    ax.axhline(0.0, color=figstyle.INK_2, linewidth=1.0, linestyle=":")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs, fontsize=8.8)
    ax.set_ylabel("cuRobo − legacy [mm]  (막대 = p05~p95)", fontsize=8.8)
    ax.text(0.33, 0.97, "양수 = cuRobo 가 더 멀다고 답함", transform=ax.transAxes,
            fontsize=8.0, color=figstyle.INK_2, va="top")
    ax.text(0.33, 0.74,
            "legacy 는 max_distance 400 mm 에서\n자른다. 16 구가 거기 걸려 있고,\n"
            "그 차이는 낙관이 아니라 포화다.",
            transform=ax.transAxes, fontsize=8.0, color=figstyle.INK_2, va="top",
            linespacing=1.6)
    figstyle.style_axes(fig, ax)

    # ── ② 그래프: 여유거리 분포 ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("② 로봇 구 120 개의 여유거리\nd − r − margin(50 mm)",
                  fontsize=10.6, color=figstyle.INK, loc="left", pad=8)
    for i, (b, colour) in enumerate((("legacy", figstyle.CATEGORICAL[3]),
                                     ("curobo", figstyle.CATEGORICAL[0]))):
        c = r0["per_backend"][b]["clearance"]
        xs = i + np.array([-0.16, 0.0, 0.16])
        ax2.plot([i, i], [c["min_mm"], c["p95_mm"]], color=colour, linewidth=2.2,
                 solid_capstyle="round")
        ax2.plot([i], [c["median_mm"]], marker="o", markersize=9, color=colour)
        ax2.plot([i], [c["min_mm"]], marker="v", markersize=7, color=colour)
        ax2.annotate(f"중앙 {c['median_mm']:+.1f}", (i, c["median_mm"]),
                     textcoords="offset points", xytext=(12, -2), fontsize=8.2,
                     color=figstyle.INK)
        ax2.annotate(f"최악 {c['min_mm']:+.1f}", (i, c["min_mm"]),
                     textcoords="offset points", xytext=(12, -3), fontsize=8.2,
                     color=figstyle.INK)
    ax2.axhline(0.0, color=figstyle.CATEGORICAL[7], linewidth=1.2, linestyle="--")
    ax2.text(1.42, 3.0, "0 = 위반 경계", fontsize=8.0, color=figstyle.CATEGORICAL[7],
             ha="right")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["legacy\n20 mm 단일", "cuRobo\n20+5 mm 2계층"], fontsize=8.8)
    ax2.set_ylabel("여유거리 [mm]", fontsize=8.8)
    ax2.set_xlim(-0.45, 1.6)
    figstyle.style_axes(fig, ax2)

    # ── ③ 그래프: 단계별 시간 ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("③ 단계별 시간 — 거리장만 바뀐다\n(cuRobo 첫 프레임은 JIT 를 포함한다)",
                  fontsize=10.6, color=figstyle.INK, loc="left", pad=8)
    width = 0.38
    for i, (b, colour) in enumerate((("legacy", figstyle.CATEGORICAL[3]),
                                     ("curobo", figstyle.CATEGORICAL[0]))):
        prof = r0["per_backend"][b]["profile_ms"]
        vals = [prof.get(k, 0.0) for k in STAGE_ORDER]
        ax3.barh(np.arange(len(STAGE_ORDER)) + (i - 0.5) * width, vals, height=width,
                 color=colour, label=b)
    ax3.set_yticks(np.arange(len(STAGE_ORDER)))
    ax3.set_yticklabels([STAGE_LABEL[k] for k in STAGE_ORDER], fontsize=8.6)
    ax3.invert_yaxis()
    ax3.set_xscale("log")
    ax3.set_xlabel("ms (로그 눈금)", fontsize=8.8)
    ax3.axvline(533.0, color=figstyle.CATEGORICAL[7], linewidth=1.2, linestyle="--")
    ax3.text(560, len(STAGE_ORDER) - 0.4, "청크 예산\n533 ms", fontsize=7.8,
             color=figstyle.CATEGORICAL[7], va="bottom")
    ax3.legend(fontsize=8.2, frameon=False, loc="lower right",
               labelcolor=figstyle.INK_2)
    figstyle.style_axes(fig, ax3)

    # ── ④ 표: 통과 조건 + 귀속 ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_title("④ 표 — 통과 조건과 귀속 (seed 3개 모두 동일)", fontsize=10.6,
                  color=figstyle.INK, loc="left", pad=10)
    ax4.axis("off")
    names = list(r0["verdict"]["checks"].keys())
    pretty = {"curobo_builder_used": "backend: curobo 에서 CuroboFieldBuilder 가 돈다",
              "no_legacy_under_curobo": "legacy EsdfBuilder 생성 0 회 (T0 즉시 실패 조건)",
              "two_tiers": "2계층 — coarse 20 mm + fine 5 mm",
              "same_sign_convention": "부호 규약 동일 — 음수 = 물체 안쪽"}
    y = 1.0
    for n in names:
        ok = all(rep[s]["verdict"]["checks"][n] for s in args.seeds)
        ax4.text(0.0, y, "●", fontsize=11, va="center", family="DejaVu Sans",
                 color=figstyle.CATEGORICAL[2] if ok else figstyle.CATEGORICAL[7])
        ax4.text(0.04, y, pretty.get(n, n), fontsize=9.0, va="center", color=figstyle.INK)
        ax4.text(1.0, y, "PASS" if ok else "FAIL", fontsize=8.8, va="center", ha="right",
                 family="DejaVu Sans",
                 color=figstyle.CATEGORICAL[2] if ok else figstyle.CATEGORICAL[7])
        y -= 0.105
    y -= 0.04
    ax4.plot([0.0, 1.0], [y + 0.05, y + 0.05], color=figstyle.GRID_INK, linewidth=1.0)
    va = r0["violation_attribution"]
    rows = [
        ("delta 전체 p95 / 최대",
         f"{r0['backend_delta']['all_spheres']['p95_mm']:+.1f} / "
         f"{r0['backend_delta']['all_spheres']['max_mm']:+.1f} mm  <- 포화의 산물"),
        ("legacy 포화 구 (max_distance 400 mm)",
         f"{r0['backend_delta']['legacy_saturated']['n']} / 120"),
        ("delta 포화 제외 중앙 / p95",
         f"{r0['backend_delta']['unsaturated_only']['median_mm']:+.2f} / "
         f"{r0['backend_delta']['unsaturated_only']['p95_mm']:+.2f} mm"),
        ("delta 근접 띠 중앙 (판정이 갈리는 곳)",
         f"{r0['backend_delta']['near_band_only']['median_mm']:+.2f} mm"),
        ("cuRobo 에서만 위반인 구",
         ", ".join(va["only_curobo"]) or "없음"),
        ("그 구들의 delta",
         ", ".join(f"{v:+.1f}" for v in va["delta_on_curobo_only_mm"]) + " mm  <- 보수적"),
        ("legacy 에서만 위반인 구",
         ", ".join(va["only_legacy"]) or "없음  <- cuRobo 가 완화한 구 0"),
        ("기하 인증 (validity)",
         f"legacy {r0['per_backend']['legacy']['validity']}  ·  "
         f"cuRobo {r0['per_backend']['curobo']['validity']}"),
    ]
    for lab, val in rows:
        ax4.text(0.0, y, lab, fontsize=8.7, va="center", color=figstyle.INK_2)
        ax4.text(1.0, y, val, fontsize=8.7, va="center", ha="right", color=figstyle.INK)
        y -= 0.098
    ax4.set_xlim(-0.015, 1.015)
    ax4.set_ylim(y, 1.07)

    # ── ⑤ 도식: 무엇을 cuRobo 에 넘겼나 ───────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_title("⑤ 책임 분배", fontsize=10.6, color=figstyle.INK, loc="left", pad=10)
    ax5.axis("off")
    items = [
        ("depth 적분 · coarse/fine ESDF", "cuRobo", figstyle.CATEGORICAL[0]),
        ("잔상 감쇠 (F20, 기본 끔)", "cuRobo", figstyle.CATEGORICAL[0]),
        ("라벨 층 — site_index 조회", "우리", figstyle.CATEGORICAL[2]),
        ("쥔 물체 — seed 에서 제외 (A2)", "우리", figstyle.CATEGORICAL[2]),
        ("해석적 정적 기하 (N2)", "우리", figstyle.CATEGORICAL[2]),
        ("관측 판정 probe", "우리", figstyle.CATEGORICAL[2]),
        ("지지면 파내기", "안 함", figstyle.GRID_INK),
    ]
    yy = 1.0
    for lab, who, colour in items:
        ax5.add_patch(Rectangle((0.0, yy - 0.035), 0.055, 0.07, facecolor=colour,
                                edgecolor="none"))
        ax5.text(0.085, yy, lab, fontsize=8.6, va="center", color=figstyle.INK)
        ax5.text(1.0, yy, who, fontsize=8.4, va="center", ha="right",
                 color=figstyle.INK_2)
        yy -= 0.125
    ax5.text(0.0, yy - 0.02,
             "seed 제외가 파내기와 다른 점 둘 —\n"
             "TSDF 가 안 다치고, 쥔 물체 뒤에 있는\n"
             "다른 장애물의 거리가 유지된다.",
             fontsize=8.2, color=figstyle.INK_2, va="top", linespacing=1.6)
    ax5.set_xlim(-0.02, 1.02)
    ax5.set_ylim(yy - 0.38, 1.07)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=figstyle.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
