"""I3 검증의 시각화 — 출처 도장과 age 상태 기계 (규칙 A: 도식 · 그래프 · 표).

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.live.plot_i3 \\
        --json outputs/live_test/20260922_i3_provenance/verify_provenance.json

`verify_provenance.py` 가 낸 JSON 만 읽는다. 다시 측정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

#: 청크 하나가 덮는 control frame. `ctrl_hz 15` · `open_loop_horizon 8` -> 533 ms.
CTRL_HZ = 15.0
HORIZON = 8
CHUNK_MS = HORIZON / CTRL_HZ * 1000.0


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    from benchmark.ag3s.experiments.common import figstyle

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", required=True)
    ap.add_argument("--out",
                    default="benchmark/ag3s/docs/figures/live-test/i3-field-provenance.png")
    args = ap.parse_args()

    rep = json.loads(pathlib.Path(args.json).read_text())

    figstyle.use_korean()
    fig = plt.figure(figsize=(16.4, 10.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.86, 1.02], hspace=0.30,
                          wspace=0.40,
                          left=0.095, right=0.978, top=0.885, bottom=0.055)
    fig.patch.set_facecolor(figstyle.SURFACE)
    fig.text(0.5, 0.960,
             "I3 — 거리장의 출처와 age. 거리값만 보면 방금 만든 것과 낡은 것을 구별할 수 없다",
             ha="center", va="center", fontsize=15.5, color=figstyle.INK)
    fig.text(0.5, 0.929,
             "2026-09-18 의 통합에서 attach() 뒤 14 프레임 동안 지각이 한 번도 안 돌았는데 "
             "상태는 ok 로 나갔다. 그것이 안 보인 이유가 이 블록이 없었기 때문이다.",
             ha="center", va="center", fontsize=9.4, color=figstyle.INK_2)

    # ── ① 도식: 한 청크가 8 control frame 을 덮는다 ────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ax.set_title("① 상태가 두 곳에서 갈린다 — 서버는 new/unavailable, 클라이언트는 carried/stale",
                 fontsize=11.0, color=figstyle.INK, loc="left", pad=10)
    ax.axis("off")
    # planning frame 두 개와 그 안의 control frame 8 개씩
    w = 0.108
    for p in range(2):
        x0 = 0.03 + p * 0.50
        ax.add_patch(Rectangle((x0, 0.68), 8 * w * 0.5 + 0.02, 0.20,
                               facecolor="#eef4fc", edgecolor="#cde2fb", linewidth=1.2))
        ax.text(x0 + 0.11, 0.845, f"planning frame {p}  —  AG3S 가 돈다",
                fontsize=9.0, color=figstyle.INK, ha="center")
        ax.text(x0 + 0.11, 0.735,
                f"필드 sequence {p+1}  ·  state = new  ·  observed_at 도장",
                fontsize=8.4, color=figstyle.CATEGORICAL[2], ha="center")
        for k in range(HORIZON):
            xx = x0 + 0.012 + k * w * 0.5
            age = k / CTRL_HZ * 1000.0
            colour = figstyle.CATEGORICAL[2] if k == 0 else "#cde2fb"
            ax.add_patch(Rectangle((xx, 0.40), w * 0.5 - 0.006, 0.17,
                                   facecolor=colour, edgecolor="none"))
            ax.text(xx + w * 0.25 - 0.003, 0.485, f"{k}", fontsize=8.0, ha="center",
                    va="center", color="white" if k == 0 else figstyle.INK)
            ax.text(xx + w * 0.25 - 0.003, 0.375, f"{age:.0f}", fontsize=7.2, ha="center",
                    va="top", color=figstyle.INK_2)
        ax.text(x0 + 0.11, 0.285,
                "control frame  ·  위 숫자 = 스텝, 아래 숫자 = age [ms]\n"
                "0 은 new, 1~7 은 carried — 청크 하나를 나눠 쓴다",
                fontsize=8.2, color=figstyle.INK_2, ha="center", va="top",
                linespacing=1.5)
        if p == 0:
            ax.add_patch(FancyArrowPatch((x0 + 0.245, 0.78), (x0 + 0.48, 0.78),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color=figstyle.INK_2, linewidth=1.2))
            ax.text(x0 + 0.362, 0.805, f"{CHUNK_MS:.0f} ms", fontsize=8.2,
                    color=figstyle.INK_2, ha="center")
    ax.text(0.03, 0.155,
            "age 의 기준은 observed_at — 이 필드가 적분한 관측 중 가장 최신의 촬영 시각이다 "
            "(클라이언트 시계).\n"
            "서버의 built_at 은 monotonic 원점이 프로세스마다 달라 클라이언트가 자기 시계와 "
            "견줄 수 없으므로 단계 시간에만 쓴다.",
            fontsize=8.6, color=figstyle.INK, va="top", linespacing=1.7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.20, 0.95)

    # ── ② 그래프: 상태 기계 ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_title("② 상태 기계 — 한도가 없으면 stale 로 올리지 않는다",
                  fontsize=11.0, color=figstyle.INK, loc="left", pad=10)
    # JSON 은 서술적인 이름을 들고 있고, 그림에는 짧은 표기를 쓴다 — 긴 한글 라벨이
    # 축 왼쪽에서 잘리기 때문이다.
    SHORT = {
        "갱신 프레임, 한도 없음": "갱신 프레임\n한도 없음",
        "control frame 2번째, 한도 없음": "control frame 2번째\n한도 없음",
        "한도 안 (0.6 s)": "한도 안\n0.6 s",
        "한도 초과 (0.2 s)": "한도 초과\n0.2 s",
        "한도 없으면 stale 로 안 올린다": "age 5 s\n한도 없음",
    }
    cases = [c for c in rep["state_machine"] if c["age_ms"] is not None]
    colours = {"new": figstyle.CATEGORICAL[2], "carried": figstyle.CATEGORICAL[3],
               "stale": figstyle.CATEGORICAL[7], "unavailable": figstyle.INK_2}
    ys = np.arange(len(cases))[::-1]
    for y, c in zip(ys, cases):
        ax2.barh(y, c["age_ms"], height=0.56, color=colours[c["state"]])
        ax2.text(c["age_ms"] + 90, y, f"{c['state']}"
                 + ("" if c["staleness_checked"] else "  (한도 검사 안 함)"),
                 va="center", fontsize=8.6, color=colours[c["state"]])
        if c["limit_sec"] is not None:
            ax2.plot([c["limit_sec"] * 1000.0] * 2, [y - 0.33, y + 0.33],
                     color=figstyle.INK, linewidth=1.8)
            ax2.text(c["limit_sec"] * 1000.0, y + 0.40,
                     f"한도 {c['limit_sec']*1000:.0f}", fontsize=7.4, ha="center",
                     color=figstyle.INK)
    ax2.axvline(CHUNK_MS, color=figstyle.CATEGORICAL[0], linewidth=1.2, linestyle="--")
    ax2.text(CHUNK_MS + 20, len(cases) - 0.35, f"청크 {CHUNK_MS:.0f} ms",
             fontsize=7.8, color=figstyle.CATEGORICAL[0])
    ax2.set_yticks(ys)
    ax2.set_yticklabels([SHORT.get(c["case"], c["case"]) for c in cases], fontsize=8.0)
    ax2.tick_params(axis="y", pad=3)
    ax2.set_xlabel("age [ms]", fontsize=8.8)
    ax2.set_xlim(0, 6500)
    ax2.set_xscale("symlog", linthresh=600)
    figstyle.style_axes(fig, ax2)

    # ── ③ 표 ─────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_title("③ 표 — 통과 조건", fontsize=11.0, color=figstyle.INK,
                  loc="left", pad=10)
    ax3.axis("off")
    pretty = {
        "same_stamp_shape": "두 backend 가 같은 모양으로 찍는다",
        "sequence_monotonic_from_one": "일련번호 1 부터 단조 증가",
        "observed_at_is_newest_stamp": "observed_at = 가장 최신 촬영 시각",
        "state_machine": "new / carried / stale / unavailable",
        "limit_default_is_unset": "한도 기본값이 None (추측 아님)",
    }
    y = 1.0
    for k, v in rep["verdict"]["checks"].items():
        ax3.text(0.0, y, "●", fontsize=11, va="center", family="DejaVu Sans",
                 color=figstyle.CATEGORICAL[2] if v else figstyle.CATEGORICAL[7])
        ax3.text(0.07, y, pretty.get(k, k), fontsize=8.5, va="center",
                 color=figstyle.INK)
        ax3.text(1.0, y, "PASS" if v else "FAIL", fontsize=8.4, va="center", ha="right",
                 family="DejaVu Sans",
                 color=figstyle.CATEGORICAL[2] if v else figstyle.CATEGORICAL[7])
        y -= 0.115
    y -= 0.05
    ax3.plot([0.0, 1.0], [y + 0.055, y + 0.055], color=figstyle.GRID_INK, linewidth=1.0)
    pb = rep["per_backend"]
    rows = [
        ("legacy 계층", str([t["voxel_size_m"] for t in pb["legacy"][0]["provenance"]["tiers"]])),
        ("cuRobo 계층", str([t["voxel_size_m"] for t in pb["curobo"][0]["provenance"]["tiers"]])),
        ("일련번호 (3 프레임)",
         str([r["provenance"]["sequence"] for r in pb["curobo"]])),
        ("observed_at 오차", f"{rep['checks']['observed_at_is_newest_stamp']['max_abs_error_sec']['curobo']:.0e} s"),
        ("한도 기본값", str(rep["checks"]["limit_default_is_unset"]["default"])),
        ("도장 키 수", str(len(rep["checks"]["same_stamp_shape"]["keys"]))),
    ]
    for lab, val in rows:
        ax3.text(0.0, y, lab, fontsize=8.4, va="center", color=figstyle.INK_2)
        ax3.text(1.0, y, val, fontsize=8.4, va="center", ha="right", color=figstyle.INK)
        y -= 0.10
    ax3.text(0.0, y - 0.02,
             "한도는 T3 에서 프레임 간 필드\n변화량을 재고 정한다.\n"
             "F14(상태 지연 한계 100 ms 가\n피해 시작점보다 6 배 느슨했다)를\n"
             "반복하지 않는다.",
             fontsize=8.2, color=figstyle.INK_2, va="top", linespacing=1.6)
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(y - 0.42, 1.07)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=figstyle.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
