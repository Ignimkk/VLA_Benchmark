"""두 서버는 다른 프로그램이다 — 8123 과 8000 (규칙 A: 도식 + 표).

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.live.plot_two_servers

수치를 재지 않는다. **와이어 계약(`benchmark/trajopt/wire.py`)과 실제 실행 기록에서 읽은
것만** 그린다. 세 판.

1. **도식** — 같은 체크포인트 위에 무엇이 더 얹혀 있는가.
2. **표** — 요청·응답 키 대조. T0 이 프레임마다 요구하는 것이 어느 쪽에만 있는가.
3. **표** — 이번 작업에서 두 서버를 각각 어디에 썼는가 (실측 시간 포함).
"""

from __future__ import annotations

import argparse
import pathlib


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow, FancyBboxPatch

    from benchmark.ag3s.experiments.common import figstyle

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out",
                    default="benchmark/ag3s/docs/figures/live-test/two-servers.png")
    args = ap.parse_args()
    figstyle.use_korean()

    fig = plt.figure(figsize=(13.4, 13.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.12, 1.0, 0.52], hspace=0.30)
    ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))
    for ax in (ax0, ax1, ax2):
        ax.set_axis_off()
    fig.patch.set_facecolor(figstyle.SURFACE)

    # ── 1. 도식 ──────────────────────────────────────────────────────────────
    ax0.set_xlim(0, 100)
    ax0.set_ylim(0, 100)

    def box(x, y, w, h, title, lines, slot, *, dashed=False):
        ax0.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=1.0", linewidth=1.3,
            edgecolor=figstyle.CATEGORICAL[slot], facecolor=figstyle.SURFACE,
            linestyle=("--" if dashed else "-"), zorder=2))
        ax0.text(x + w / 2, y + h - 4.5, title, ha="center", va="top", fontsize=9,
                 color=figstyle.CATEGORICAL[slot], zorder=3)
        for i, line in enumerate(lines):
            ax0.text(x + 2.6, y + h - 10 - i * 4.8, line, ha="left", va="top",
                     fontsize=7.4, color=figstyle.INK_2, zorder=3)

    # 공통 토대
    ax0.add_patch(FancyBboxPatch((6, 4), 88, 15, boxstyle="round,pad=1.0",
                                 linewidth=1.1, edgecolor=figstyle.GRID_INK,
                                 facecolor="#f4f4f2", zorder=1))
    ax0.text(50, 16.5, "같은 토대 — 두 서버가 같은 것을 로드한다", ha="center", fontsize=8.6,
             color=figstyle.INK)
    ax0.text(50, 10.5,
             "config pi05_rby1_randomized_pick_place_16d_lora  ·  "
             "checkpoint rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999",
             ha="center", fontsize=7.4, color=figstyle.INK_2)
    ax0.text(50, 5.6, "(실행 중인 두 프로세스의 명령줄에서 읽은 것이다 — 추정이 아니다)",
             ha="center", fontsize=6.8, color=figstyle.INK_2)

    box(6, 26, 41, 66, "8123 — serve_policy.py  (사용자가 띄운 것)",
        ["정책만 있다.",
         "",
         "요청: state · images · prompt",
         "응답: actions",
         "",
         "· AG3S 없음 (depth 를 받지도 않는다)",
         "· TO 없음",
         "· 안전 판정 없음",
         "· 거리장 출처 도장 없음",
         "· ag3s/seq 를 돌려주지 않는다",
         "",
         "실측: infer 0.09 s — 정책 자체는 거의 공짜다"], 2)

    box(53, 26, 41, 66, "8000 — serve_safe.py  (내가 띄운 것)",
        ["정책 + AG3S + TO 를 한 프로세스에.",
         "",
         "요청: 위 + ag3s/depth·K·T_base_cam·",
         "         robot_state·stamp·phase·seq",
         "응답: 위 + safe · ag3s_status ·",
         "         trajopt_status · field(출처 도장)",
         "",
         "· 체크포인트를 두 벌 로드한다",
         "   (한 벌은 attention 추출용)",
         "· AG3S backend = cuRobo 2계층",
         "",
         "실측: 왕복 2.5 s (AG3S 1.94 + TO 0.28) — T0 이 재려던 것이 여기 있다"], 1)

    ax0.add_patch(FancyArrow(26.5, 22, 0, -2.4, width=0.5, head_width=2.2,
                             head_length=2.0, color=figstyle.INK_2, zorder=4))
    ax0.add_patch(FancyArrow(73.5, 22, 0, -2.4, width=0.5, head_width=2.2,
                             head_length=2.0, color=figstyle.INK_2, zorder=4))
    ax0.set_title("두 서버는 같은 모델을 다른 계약으로 내놓는다", fontsize=10.4,
                  color=figstyle.INK, pad=8)

    # ── 2. 키 대조 표 ────────────────────────────────────────────────────────
    rows = [
        ["actions (청크)", "있다", "있다", "—"],
        ["ag3s/depth · K · T_base_cam (카메라 3대)", "받지 않는다", "받는다", "ESDF 를 지을 재료"],
        ["ag3s/robot_state — 카메라마다 따로", "받지 않는다", "받는다", "손목 클라우드 번짐·자기 필터 어긋남 방지"],
        ["ag3s/stamp — 촬영 시각", "받지 않는다", "받는다", "T0: 카메라 시차, 필드 나이"],
        ["ag3s/seq — 일련번호 왕복", "돌려주지 않는다", "돌려준다", "T0: 오래된 응답 버리기 · 중복 검사"],
        ["ag3s_status · geometry_certified", "없다", "있다", "T0: 프레임별 인증 여부"],
        ["trajopt_status · max_violation_m", "없다", "있다", "T0: 안전 게이트가 실제로 걸리는가"],
        ["safe — 실행 허가", "없다", "있다", "T0: IPC 결과 (ok/unsafe/timeout/stale)"],
        ["field — 거리장 출처 도장 14 키", "없다", "있다", "T0 의 핵심. backend·observed_at·계층·상태"],
    ]
    header = ["와이어 키", "8123 serve_policy", "8000 serve_safe", "T0 에서 무엇에 쓰이나"]
    tb = ax1.table(cellText=rows, colLabels=header, loc="center", cellLoc="center",
                   colWidths=[0.30, 0.16, 0.15, 0.39])
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.4)
    tb.scale(1, 1.62)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        cell.set_facecolor(figstyle.SURFACE)
        cell.set_text_props(color=figstyle.INK)
        if c in (0, 3) and r:
            cell.set_text_props(color=figstyle.INK_2, ha="left")
        if r == 0:
            cell.set_text_props(color=figstyle.INK, fontsize=7.6)
        if r and rows[r - 1][1] in ("없다", "받지 않는다", "돌려주지 않는다") and c == 1:
            cell.set_facecolor("#fdf0ea")
    ax1.set_title("T0 이 프레임마다 요구한 것은 전부 8000 쪽에만 있다  (정의: trajopt/wire.py)",
                  fontsize=10.4, color=figstyle.INK, pad=10)

    # ── 3. 어디에 무엇을 썼나 ─────────────────────────────────────────────────
    used = [
        ["T0 — 환경·배선 검증 (4 에피소드 × 48 스텝)", "8000 serve_safe",
         "프레임마다 출처 도장·안전 판정·IPC 가 필요하다"],
        ["16D 회귀 기준선의 기록 만들기 (ep1800, 120 스텝)", "8123 (사용자 서버)",
         "AG3S 는 오프라인에서 적용한다 — run_0004 를 만든 방식과 같다"],
        ["attention 추출 (체크포인트 직접 로드)", "서버 없음",
         "return_attn_probs 는 서버 프로토콜로 못 뽑는다"],
    ]
    tb2 = ax2.table(cellText=used,
                    colLabels=["이번 작업의 단계", "쓴 서버", "왜"],
                    loc="center", cellLoc="center", colWidths=[0.36, 0.17, 0.47])
    tb2.auto_set_font_size(False)
    tb2.set_fontsize(7.4)
    tb2.scale(1, 1.66)
    for (r, c), cell in tb2.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        cell.set_facecolor(figstyle.SURFACE)
        cell.set_text_props(color=figstyle.INK)
        if c != 1 and r:
            cell.set_text_props(color=figstyle.INK_2, ha="left")
        if r == 2:
            cell.set_facecolor("#f2f7fd")
    ax2.set_title("8123 도 실제로 썼다 — 기준선의 기록을 만드는 데 (그리고 거기서 0.09 s 가 나왔다)",
                  fontsize=10.4, color=figstyle.INK, pad=10)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=figstyle.SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
