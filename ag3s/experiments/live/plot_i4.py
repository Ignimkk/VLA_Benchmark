"""I4 검증의 시각화 — 프레임 기록과 completeness (규칙 A: 도식 · 그래프 · 표).

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.live.plot_i4 \\
        --dir outputs/live_test/20260922_i4_frames

`verify_frame_record.py` 가 낸 JSON/JSONL 만 읽는다. 다시 측정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

STATE_ORDER = ("new", "carried", "stale", "unavailable")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from benchmark.ag3s.experiments.common import figstyle

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out",
                    default="benchmark/ag3s/docs/figures/live-test/i4-frame-records.png")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    rep = json.loads((d / "verify_frame_record.json").read_text())
    rows = {name: [json.loads(l) for l in
                   (d / name / "frames.jsonl").read_text().splitlines()]
            for name in rep["sessions"]}

    colour = {"new": figstyle.CATEGORICAL[2], "carried": figstyle.CATEGORICAL[3],
              "stale": figstyle.CATEGORICAL[7], "unavailable": figstyle.INK_2}

    figstyle.use_korean()
    fig = plt.figure(figsize=(16.6, 10.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.30,
                          left=0.055, right=0.978, top=0.848, bottom=0.065)
    fig.patch.set_facecolor(figstyle.SURFACE)
    fig.text(0.5, 0.958,
             "I4 — 프레임별 기록. 갱신되지 않은 프레임의 값을 새로 계산한 것처럼 적지 않는다",
             ha="center", va="center", fontsize=15.5, color=figstyle.INK)
    fig.text(0.5, 0.926,
             "driver 호출 순서를 그대로 재현했다 (청크 하나 + control frame 8 개 × 6 회). "
             "pi05_infer 자체는 mink 가 없어 이 서버에서 돌지 않는다.",
             ha="center", va="center", fontsize=9.4, color=figstyle.INK_2)

    # ── ① 한도 없음: 상태 띠 ─────────────────────────────────────────────
    for col, (name, title) in enumerate((
            ("clean_no_limit", "① 한도 없음 — stale 이 없다"),
            ("clean_limit_200ms", "② 한도 200 ms — 스텝 3 부터 stale"))):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(title, fontsize=10.8, color=figstyle.INK, loc="left", pad=8)
        ctrl = [r for r in rows[name] if r["kind"] == "control"]
        for r in ctrl:
            c = r["chunk_seq"] - 1
            k = r["step_in_chunk"]
            ax.add_patch(Rectangle((k, c), 0.92, 0.84,
                                   facecolor=colour[r["field"]["state"]],
                                   edgecolor="none"))
        plan = [r for r in rows[name] if r["kind"] == "planning"]
        for r in plan:
            ax.add_patch(Rectangle((-1.3, r["seq"] - 1), 0.92, 0.84,
                                   facecolor=colour[r["field"]["state"]],
                                   edgecolor="none"))
        ax.axvline(-0.2, color=figstyle.GRID_INK, linewidth=1.2)
        ax.text(-0.84, -0.9, "planning", fontsize=8.0, ha="center",
                color=figstyle.INK_2)
        ax.text(3.5, -0.9, "control frame (step in chunk)", fontsize=8.4, ha="center",
                color=figstyle.INK_2)
        ax.set_xticks(np.arange(8) + 0.46)
        ax.set_xticklabels(range(8), fontsize=8.2)
        ax.set_yticks(np.arange(len(plan)) + 0.42)
        ax.set_yticklabels([f"청크 {i+1}" for i in range(len(plan))], fontsize=8.2)
        ax.set_xlim(-1.5, 8.1)
        ax.set_ylim(-1.5, len(plan) + 0.1)
        ax.invert_yaxis()
        figstyle.style_axes(fig, ax)
        ax.tick_params(length=0)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)

    # ── ③ age 가 스텝마다 늘어난다 ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("③ age 는 스텝마다 1/ctrl_hz 씩 늘어난다", fontsize=10.8,
                  color=figstyle.INK, loc="left", pad=8)
    chk = rep["checks"]["age_grows_per_step"]
    xs = np.arange(len(chk["ages_ms"]))
    ax3.bar(xs, chk["ages_ms"], width=0.66,
            color=[colour["new"] if k == 0 else colour["carried"] for k in xs])
    ax3.plot(xs, chk["expected_ms"], marker="o", markersize=5, linestyle=":",
             color=figstyle.INK, linewidth=1.2, label="기대 = k / 15 Hz")
    ax3.axhline(200.0, color=figstyle.CATEGORICAL[7], linewidth=1.2, linestyle="--")
    ax3.text(7.4, 214, "한도 200 ms 예", fontsize=7.8, ha="right",
             color=figstyle.CATEGORICAL[7])
    ax3.set_xticks(xs)
    ax3.set_xlabel("step in chunk", fontsize=8.8)
    ax3.set_ylabel("age [ms]", fontsize=8.8)
    ax3.legend(fontsize=8.0, frameon=False, loc="upper left",
               labelcolor=figstyle.INK_2)
    ax3.text(0.03, 0.72, f"최대 오차 {chk['max_abs_error_ms']:.1e} ms",
             transform=ax3.transAxes, fontsize=8.0, color=figstyle.INK_2)
    figstyle.style_axes(fig, ax3)

    # ── ④ completeness 표 ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_title("④ 표 — 세션별 completeness (프롬프트 §5)", fontsize=10.8,
                  color=figstyle.INK, loc="left", pad=10)
    ax4.axis("off")
    names = list(rep["sessions"])
    short = {"clean_no_limit": "정상\n한도 없음", "clean_limit_200ms": "정상\n한도 200 ms",
             "timestamp_reversal": "주입\ntimestamp 역전",
             "duplicate_seq": "주입\n중복 일련번호", "no_field": "주입\n필드 없음"}
    keys = [("captured_observation_frames", "captured observation"),
            ("planning_records", "planning records"),
            ("control_records", "control records"),
            ("planned_missing_frames", "planned missing"),
            ("duplicate_sequence_ids", "duplicate sequence ids"),
            ("timestamp_reversals", "timestamp reversals"),
            ("unexplained_carried_or_stale", "unexplained carried/stale"),
            ("staleness_checked", "staleness checked"),
            ("pass", "completeness pass")]
    xcol = np.linspace(0.40, 0.98, len(names))
    y = 1.0
    for j, n in enumerate(names):
        ax4.text(xcol[j], y + 0.085, short[n], ha="center", va="bottom", fontsize=7.8,
                 color=figstyle.INK, linespacing=1.4)
    ax4.plot([0.0, 1.0], [y + 0.045, y + 0.045], color=figstyle.GRID_INK, linewidth=1.0)
    for key, lab in keys:
        ax4.text(0.0, y, lab, fontsize=8.5, va="center", color=figstyle.INK_2)
        for j, n in enumerate(names):
            v = rep["sessions"][n][key]
            txt = {True: "예", False: "아니오"}.get(v, str(v))
            col = figstyle.INK
            if key in ("duplicate_sequence_ids", "timestamp_reversals",
                       "unexplained_carried_or_stale", "planned_missing_frames"):
                col = figstyle.CATEGORICAL[7] if v else figstyle.CATEGORICAL[2]
            if key == "pass":
                col = figstyle.CATEGORICAL[2] if v else figstyle.CATEGORICAL[7]
            ax4.text(xcol[j], y, txt, fontsize=8.4, va="center", ha="center", color=col)
        y -= 0.093
    ax4.text(0.0, y - 0.02,
             "주입 세션 둘이 pass=아니오 다 — 검사가 실제로 잡는다는 증거다. "
             "'필드 없음' 은 pass=예 인데 옳다: completeness 는\n"
             "\"다 기록했는가\" 를 묻고 \"다 좋았는가\" 를 묻지 않는다. "
             "필드가 없던 9 프레임은 unavailable 로 표에 남는다.",
             fontsize=8.3, color=figstyle.INK_2, va="top", linespacing=1.7)
    ax4.set_xlim(-0.015, 1.02)
    ax4.set_ylim(y - 0.22, 1.20)

    # ── ⑤ manifest ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_title("⑤ manifest — 실측 버전과 좌표계", fontsize=10.8, color=figstyle.INK,
                  loc="left", pad=10)
    ax5.axis("off")
    man = json.loads((d / "clean_no_limit" / "manifest.json").read_text())
    v = man["versions"]
    items = [("seed", str(man.get("seed"))),
             ("python", v.get("python")),
             ("numpy", v.get("numpy")),
             ("mujoco", v.get("mujoco")),
             ("curobo", str(v.get("curobo"))[:18]),
             ("curobo commit", (man.get("curobo_commit") or "-")[:10]),
             ("GPU", str(v.get("gpu"))[:22]),
             ("ctrl_hz / horizon",
              f"{man['timing']['ctrl_hz']:.0f} / {man['timing']['open_loop_horizon']}"),
             ("chunk 주기", f"{man['timing']['chunk_period_ms']:.0f} ms"),
             ("max_field_age_sec", str(man["timing"]["max_field_age_sec"])),
             ("layout / slots", f"{man['scene']['fruit_layout_index']} / "
                                f"{len(man['scene']['fruit_slot_order'])}")]
    yy = 1.0
    for lab, val in items:
        ax5.text(0.0, yy, lab, fontsize=8.3, va="center", color=figstyle.INK_2)
        ax5.text(1.0, yy, str(val), fontsize=8.3, va="center", ha="right",
                 color=figstyle.INK)
        yy -= 0.078
    yy -= 0.03
    ax5.plot([0.0, 1.0], [yy + 0.04, yy + 0.04], color=figstyle.GRID_INK, linewidth=1.0)
    ax5.text(0.0, yy - 0.01,
             "이 하네스는 .venv-ag3s 에서 돌았다 —\n그래서 curobo 가 MISSING 이다. "
             "manifest 는\n그 프로세스가 실제로 import 한 것을\n적는다 (선언이 아니다).\n\n"
             "좌표계를 명시한다 — 이 검토에서\n좌표계 오해가 두 번 났다 (cuRobo 함정 1·2).\n\n"
             + "\n".join(f"· {k}: {str(val)[:34]}"
                         for k, val in list(man["frames_of_reference"].items())[:4]),
             fontsize=7.8, color=figstyle.INK_2, va="top", linespacing=1.6)
    ax5.set_xlim(-0.02, 1.02)
    ax5.set_ylim(yy - 0.60, 1.06)

    # 범례
    # 범례는 부제와 패널 제목 **사이**에 한 줄로 둔다 — 패널 안에 두면 첫 패널의 제목을 밟는다.
    lx = 0.5 - 0.5 * (len(STATE_ORDER) * 0.085)
    for i, st in enumerate(STATE_ORDER):
        fig.patches.append(plt.Rectangle((lx + i * 0.085, 0.884), 0.015, 0.013,
                                         facecolor=colour[st], edgecolor="none",
                                         transform=fig.transFigure, figure=fig))
        fig.text(lx + 0.019 + i * 0.085, 0.8905, st, fontsize=8.4, va="center",
                 color=figstyle.INK_2)
    fig.text(lx - 0.012, 0.8905, "필드 상태:", fontsize=8.4, va="center", ha="right",
             color=figstyle.INK)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=figstyle.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
