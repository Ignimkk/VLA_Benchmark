"""T0 시각화 — 환경·배선 검증 (규칙 A: 도식 · 그래프 · 표).

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.live.plot_t0 --root outputs/live_test/20260924_t0

`pi05_infer --record-frames` 가 낸 manifest / frames.jsonl / completeness.json 만 읽는다.
**다시 측정하지 않는다** — T0 의 통과 근거는 실행이 남긴 기록 그 자체다.

네 판을 낸다.

1. **배선 지도** (도식) — 어느 프로세스에 무엇이 있고 무엇이 어느 포트로 흐르는가. T0 이
   확인하려는 것이 바로 이 그림이 사실인지다.
2. **프레임 타임라인** (그래프) — observation / planning / control 세 줄. 거리장 상태
   (`new`/`carried`)를 색으로, 필드 나이를 선으로. `carried` 가 control 에서만 생기는지가 눈에 보인다.
3. **카메라 시차** (그래프) — 관측 프레임마다 세 대의 촬영 시각 차이. 순차 렌더의 비용이고,
   실제 로봇에서는 손목 클라우드의 번짐이 된다.
4. **completeness 표** — 에피소드별 기대 대 실제. 통과 조건은 프롬프트가 정한 그대로다.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

STATE_COLOR_SLOT = {"new": 2, "carried": 3, "stale": 7, "unavailable": 1}


def load(root: pathlib.Path) -> list[dict]:
    """에피소드 디렉터리마다 manifest · 프레임 · completeness 를 모은다."""
    out = []
    for d in sorted(root.glob("t0_ep*")):
        if not d.is_dir() or not (d / "completeness.json").exists():
            continue
        rows = [json.loads(l) for l in (d / "frames.jsonl").read_text().splitlines() if l.strip()]
        out.append({
            "dir": d,
            "name": d.name.replace("t0_ep", "ep "),
            "manifest": json.loads((d / "manifest.json").read_text()),
            "completeness": json.loads((d / "completeness.json").read_text()),
            "rows": rows,
        })
    if not out:
        raise SystemExit(f"{root} 안에 t0_ep*/completeness.json 이 없다")
    return out


def panel_wiring(ax, sessions) -> None:
    from matplotlib.patches import FancyArrow, FancyBboxPatch

    from benchmark.ag3s.experiments.common import figstyle

    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    man = sessions[0]["manifest"]
    versions = man.get("versions", {})
    # **ESDF 설정은 manifest 에 없다.** 클라이언트의 manifest 는 그것이 서버 소유이고
    # 프레임의 `field` 에 실린다고만 적는다 — 클라이언트는 서버의 설정을 모르므로 적으면
    # 지어내는 것이 된다. 그래서 여기서도 프레임의 출처 도장에서 읽는다.
    stamps = [r["field"] for r in sessions[0]["rows"]
              if r["kind"] == "planning" and isinstance(r.get("field"), dict)
              and r["field"].get("tiers")]
    stamp = stamps[0] if stamps else {}
    # **계층 구성은 프레임마다 다르다.** 미세 계층은 target grounding 이 성공한 프레임에만
    # 붙으므로 한 프레임만 보고 "계층 하나" 라 적으면 틀린다. 본 구성을 다 적는다.
    seen = sorted({tuple(round(t["voxel_size_m"] * 1000) for t in st["tiers"])
                   for st in stamps})
    tier_text = " / ".join(" + ".join(f"{v:.0f} mm" for v in cfg) for cfg in seen) or "?"

    def box(x, y, w, h, title, lines, slot):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.2",
                                    linewidth=1.2, edgecolor=figstyle.CATEGORICAL[slot],
                                    facecolor=figstyle.SURFACE, zorder=2))
        ax.text(x + w / 2, y + h - 5, title, ha="center", va="top", fontsize=9,
                color=figstyle.CATEGORICAL[slot], zorder=3)
        for i, line in enumerate(lines):
            ax.text(x + 3, y + h - 13 - i * 5.4, line, ha="left", va="top", fontsize=7.2,
                    color=figstyle.INK_2, zorder=3)

    box(2, 34, 42, 62, "클라이언트 — pi05_infer.py",
        [f"venv: {pathlib.Path(versions.get('prefix', '?')).name}",
         f"python {versions.get('python', '?')} · numpy {versions.get('numpy', '?')}",
         f"mujoco {versions.get('mujoco', '?')} (OSMesa 소프트 렌더)",
         "",
         "· 씬을 굴리고 카메라 3 대를 찍는다",
         "· 청크를 받아 실행 여부를 스스로 정한다",
         "· observation/planning/control 프레임을 남긴다"], 0)

    box(56, 34, 42, 62, "서버 — serve_safe.py (포트 8000)",
        [f"venv: {pathlib.Path(versions.get('prefix', '?')).name} (같은 프로세스에 전부)",
         f"jax {versions.get('jax', '?')} · torch {versions.get('torch', '?')}",
         f"curobo {versions.get('curobo', '?')} · warp {versions.get('warp', '?')}",
         "",
         "· 정책 π0.5 16D (LoRA, held-out 에피소드)",
         f"· AG3S backend = {stamp.get('backend', '?')}",
         f"· 본 계층 구성: {tier_text}",
         "   (미세 계층은 grounding 이 성공한 프레임에만 붙는다)",
         "· TO 가 청크를 검사하고 안전 판정을 붙인다"], 1)

    ax.add_patch(FancyArrow(44.5, 78, 10.5, 0, width=0.6, head_width=2.4, head_length=2.4,
                            color=figstyle.INK_2, zorder=4))
    ax.text(50, 81, "depth ×3\n+ 카메라별 자세\n+ 촬영 시각 + 위상", ha="center", va="bottom",
            fontsize=6.6, color=figstyle.INK_2)
    ax.add_patch(FancyArrow(55.5, 48, -10.5, 0, width=0.6, head_width=2.4, head_length=2.4,
                            color=figstyle.INK_2, zorder=4))
    ax.text(50, 45, "청크 (50, 16)\n+ 안전 판정\n+ 거리장 출처 도장", ha="center", va="top",
            fontsize=6.6, color=figstyle.INK_2)

    ax.text(50, 28, "T0 이 확인하는 것 — 이 그림이 프레임마다 사실인가", ha="center",
            fontsize=8.5, color=figstyle.INK)
    checks = [
        f"backend 가 모든 프레임에서 {stamp.get('backend', '?')} 인가 (legacy 면 즉시 실패)",
        "legacy EsdfBuilder 가 한 번도 안 만들어졌는가 — 생성 횟수를 프레임마다 본다",
        "일련번호 중복 0 · 촬영 시각 역전 0",
        "설명 안 되는 carried/stale 0 · IPC 결과가 프레임마다 기록됐는가",
        "청크 폭이 정말 16 인가 (설정 문자열이 아니라 온 것의 폭)",
    ]
    for i, c in enumerate(checks):
        ax.text(6, 22 - i * 5.0, f"· {c}", ha="left", va="top", fontsize=7.2,
                color=figstyle.INK_2)


def panel_timeline(ax, session) -> None:
    from benchmark.ag3s.experiments.common import figstyle

    rows = session["rows"]
    lanes = {"observation": 2.0, "planning": 1.0, "control": 0.0}
    for kind, y in lanes.items():
        xs = [r["t_step"] for r in rows if r["kind"] == kind]
        if not xs:
            continue
        colors = []
        for r in rows:
            if r["kind"] != kind:
                continue
            f = r.get("field")
            state = (f or {}).get("state", "unavailable") if isinstance(f, dict) else "unavailable"
            if kind == "observation":
                state = "new"
            colors.append(figstyle.CATEGORICAL[STATE_COLOR_SLOT.get(state, 1)])
        ax.scatter(xs, [y] * len(xs), s=44, c=colors, marker="s", zorder=3,
                   edgecolors=figstyle.SURFACE, linewidths=0.6)
    ax.set_yticks(list(lanes.values()))
    ax.set_yticklabels(["observation\n(정책 호출당 1)", "planning\n(청크당 1)",
                        "control\n(제어 스텝당 1)"], fontsize=7.4)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("제어 스텝 t")

    ages = [(r["t_step"], (r.get("field") or {}).get("age_ms"))
            for r in rows if r["kind"] == "control"]
    ages = [(t, a) for t, a in ages if a is not None]
    if ages:
        ax2 = ax.twinx()
        ax2.plot([t for t, _ in ages], [a for _, a in ages], lw=1.2,
                 color=figstyle.CATEGORICAL[0], marker="o", ms=2.4, zorder=2)
        ax2.set_ylabel("필드 나이 (ms)", fontsize=8, color=figstyle.CATEGORICAL[0])
        ax2.tick_params(colors=figstyle.CATEGORICAL[0], labelsize=7.4)
        for spine in ("top", "left"):
            ax2.spines[spine].set_visible(False)
        ax2.spines["right"].set_color(figstyle.GRID_INK)
        ax2.set_facecolor("none")
    ax.set_title(f"{session['name']} — 프레임 타임라인 (색 = 거리장 상태, 선 = 필드 나이)",
                 fontsize=9, color=figstyle.INK)
    handles = [
        ax.scatter([], [], s=44, marker="s", c=figstyle.CATEGORICAL[STATE_COLOR_SLOT["new"]],
                   label="new — 이 프레임에 새로 지었다"),
        ax.scatter([], [], s=44, marker="s", c=figstyle.CATEGORICAL[STATE_COLOR_SLOT["carried"]],
                   label="carried — 앞 프레임의 것을 그대로 쓴다"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.8, frameon=False,
              ncol=2, bbox_to_anchor=(0.0, -0.34))


def panel_skew(ax, sessions) -> None:
    from benchmark.ag3s.experiments.common import figstyle

    for i, s in enumerate(sessions):
        skews = [r["camera_skew_sec"] * 1000.0 for r in s["rows"]
                 if r["kind"] == "observation" and r.get("camera_skew_sec") is not None]
        if not skews:
            continue
        ax.scatter([i] * len(skews), skews, s=26, alpha=0.75,
                   color=figstyle.CATEGORICAL[i % len(figstyle.CATEGORICAL)], zorder=3)
        ax.plot([i - 0.24, i + 0.24], [np.median(skews)] * 2, lw=1.6,
                color=figstyle.INK, zorder=4)
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels([s["name"] for s in sessions], fontsize=7.4)
    ax.set_ylabel("세 카메라 촬영 시각의 최대 차이 (ms)")
    ax.set_title("카메라 시차 — 순차 렌더의 비용 (검은 선 = 중앙값)", fontsize=9,
                 color=figstyle.INK)
    ax.grid(axis="y", color=figstyle.GRID_INK, lw=0.6, zorder=0)
    ax.text(0.02, 0.04,
            "캡처 중 시뮬레이션은 멈춰 있어 이 씬에서는 기하 번짐이 없다.\n"
            "실제 로봇에서는 이 값이 그대로 손목 클라우드의 번짐이 된다.",
            transform=ax.transAxes, fontsize=6.8, color=figstyle.INK_2, va="bottom")


def panel_table(ax, sessions) -> None:
    from benchmark.ag3s.experiments.common import figstyle

    ax.set_axis_off()
    keys = [
        ("expected_observation_frames", "기대 관측"),
        ("captured_observation_frames", "실제 관측"),
        ("planning_records", "planning"),
        ("control_records", "control"),
        ("planned_missing_frames", "누락"),
        ("duplicate_sequence_ids", "중복 일련번호"),
        ("timestamp_reversals", "시각 역전"),
        ("unexplained_carried_or_stale", "설명 안 된 carried"),
        ("pass", "통과"),
    ]
    header = ["항목"] + [s["name"] for s in sessions]
    body = []
    for k, label in keys:
        row = [label]
        for s in sessions:
            v = s["completeness"].get(k)
            row.append("예" if v is True else "아니오" if v is False else str(v))
        body.append(row)
    # 거리장 상태·IPC 는 딕셔너리라 따로 한 줄씩.
    for k, label in (("field_state_counts", "거리장 상태"), ("ipc_outcomes", "IPC 결과")):
        row = [label]
        for s in sessions:
            d = s["completeness"].get(k) or {}
            row.append(" ".join(f"{kk} {vv}" for kk, vv in d.items()))
        body.append(row)

    tb = ax.table(cellText=body, colLabels=header, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.2)
    tb.scale(1, 1.42)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        cell.set_facecolor(figstyle.SURFACE)
        if r == 0:
            cell.set_text_props(color=figstyle.INK, fontsize=7.6)
        elif c == 0:
            cell.set_text_props(color=figstyle.INK_2, ha="left")
            cell.PAD = 0.04
        else:
            cell.set_text_props(color=figstyle.INK)
        if r == len(body) - 2 and r > 0:  # 통과 줄
            pass
    ax.set_title("completeness — 기대와 실제가 같은가 (통과 조건은 프롬프트가 정한 그대로)",
                 fontsize=9, color=figstyle.INK)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmark.ag3s.experiments.common import figstyle

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out",
                    default="benchmark/ag3s/docs/figures/live-test/t0-wiring-frames.png")
    args = ap.parse_args()

    figstyle.use_korean()
    sessions = load(pathlib.Path(args.root))

    fig = plt.figure(figsize=(13.6, 15.2))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.06, 0.78, 0.66, 1.0], hspace=0.42)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])
    figstyle.style_axes(fig, [ax1, ax2])

    panel_wiring(ax0, sessions)
    panel_timeline(ax1, sessions[0])
    panel_skew(ax2, sessions)
    panel_table(ax3, sessions)

    n_pass = sum(1 for s in sessions if s["completeness"].get("pass"))
    fig.suptitle(
        f"T0 — 환경·배선 검증 · held-out test split {len(sessions)} 에피소드 "
        f"(통과 {n_pass}/{len(sessions)})",
        fontsize=12, color=figstyle.INK, y=0.995)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=figstyle.SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
