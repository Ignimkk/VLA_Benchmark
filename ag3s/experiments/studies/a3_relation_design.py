"""작업 완료 판정 — 지금 방식과 **object relation** 방식의 비교 도식 (2026-09-17 사용자 제안).

코드를 바꾸지 않는다. A1~A3 에서 **이미 잰 수치**를 한자리에 놓고 두 설계를 비교하는 그림만
만든다. 수치의 출처는 전부 `AG3S_REVIEW_LOG.md` 의 해당 절이다.

실행:
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -m \
        benchmark.ag3s.experiments.studies.a3_relation_design
"""

from __future__ import annotations

import argparse
import pathlib

OUT = pathlib.Path("benchmark/ag3s/docs/archive/14d-era-20260923/archive/14d-era-20260923/figures/a3-relation-design.png")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments.common import figstyle
    figstyle.use_korean()
    return figstyle


def _table(ax, rows, fs, title, widths, fontsize=8.2, head=True):
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, color=fs.INK)
    t = ax.table(cellText=rows, colWidths=list(widths), loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(fontsize); t.scale(1, 2.05)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if head and r == 0:
            cell.set_facecolor("#ecebe7"); cell.set_text_props(weight="bold")
        elif rows[r][1] == "" and rows[r][0]:
            cell.set_facecolor("#f2f1ed"); cell.set_text_props(weight="bold")
    return t


def _flow(ax, fs, title, stages, colour, notes=None):
    """세로 파이프라인 도식 하나."""
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, color=fs.INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(stages)
    h = 0.86 / n
    for k, (text, kind) in enumerate(stages):
        y = 0.93 - (k + 1) * h
        face = {"ok": "#e8f1fc", "new": "#e6f6ee", "weak": "#fdeceb"}[kind]
        edge = {"ok": colour, "new": fs.CATEGORICAL[2], "weak": fs.CATEGORICAL[7]}[kind]
        ax.add_patch(plt_rect((0.04, y), 0.92, h * 0.78, face, edge))
        ax.text(0.50, y + h * 0.39, text, ha="center", va="center",
                fontsize=8.4, color=fs.INK, linespacing=1.35)
        if k < n - 1:
            ax.annotate("", xy=(0.50, y), xytext=(0.50, y - h * 0.22),
                        arrowprops=dict(arrowstyle="<-", color="#9a9a96", lw=1.1))
    if notes:
        ax.text(0.50, 0.02, notes, ha="center", va="bottom", fontsize=7.8,
                color=fs.INK_2, linespacing=1.4)


def plt_rect(xy, w, h, face, edge):
    from matplotlib.patches import FancyBboxPatch
    return FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                          facecolor=face, edgecolor=edge, linewidth=1.2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    fs = _style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.2, 9.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.92, 0.92, 1.25, 1.25],
                          height_ratios=[1.35, 1.0], wspace=0.22, hspace=0.30)

    # --- ① 지금 방식 ---------------------------------------------------------------
    now_stages = [
        ("attention → lifting → grounding\n(물체 이름 하나)", "ok"),
        ("잠금 걸기: 연속 3 프레임 + 격차\n유지: attention 무시", "ok"),
        ("그리퍼가 닫힘 → attach()\n쥔 물체 = 파지 순간 스냅샷", "weak"),
        ("스냅샷이 손에 강체로 따라간다\n(FK 한 번, 재관측 없음)", "weak"),
        ("Inside: 라벨 층 AABB + 테두리 아래", "ok"),
        ("Detached: 잴 수 없다 → 그리퍼가 대신", "weak"),
        ("TaskDone = Inside AND 그리퍼 열림", "ok"),
    ]
    _flow(fig.add_subplot(gs[:, 0]), fs, "① 지금 (A3 에서 구현)", now_stages,
          fs.CATEGORICAL[0],
          notes="붉은 칸 = 물리를 안 보고\n대리 신호를 쓰는 자리")

    # --- ② 제안 방식 ---------------------------------------------------------------
    new_stages = [
        ("Attention (intent)\n무엇을 다음에 할 것인가", "ok"),
        ("3D Lifting\ndepth + calibration → 점구름", "ok"),
        ("Object Point Cloud Tracking\n사과를 프레임마다 다시 잡는다", "new"),
        ("3D Relation Estimation\nAttached / Inside / Stable", "new"),
        ("Task Phase Transition\nattention 사과→바구니 = Pick→Place", "new"),
        ("TaskDone = Inside AND Detached\nAND Stable", "new"),
    ]
    _flow(fig.add_subplot(gs[:, 1]), fs, "② 제안 (object relation)", new_stages,
          fs.CATEGORICAL[2],
          notes="초록 칸 = 물리 상태를\n직접 재는 자리")

    # --- ③ 실패 모드 표 ------------------------------------------------------------
    rows = [["상황", "지금", "제안"],
            ["빈손으로 그리퍼가\n닫힌다", "attach 됨. 유령 구멍이\n손을 따라다닌다", "Attached 거짓\n→ attach 안 함"],
            ["옮기다 미끄러진다", "그리퍼가 열릴 때까지\n쥔 것으로 본다", "움직임 상관이\n깨진다 → Detached"],
            ["놓았는데 그리퍼가\n안 열린다", "영영 detach 안 됨\n팔 낙관 +19.7 mm", "Detached + Stable\n→ 완료"],
            ["그리퍼가 잠깐 튄다", "폴백 2 프레임이\n막는다", "Stable 이 막는다"],
            ["바구니가 돌아가\n있다", "AABB 가 헐거워진다", "내부 volume 이면\n정확"],
            ["이번 롤아웃", "프레임 19", "프레임 19 (동일)"]]
    _table(fig.add_subplot(gs[0, 2]), rows, fs, "③ 어느 쪽이 무엇을 잡는가",
           widths=(0.30, 0.35, 0.35), fontsize=7.4)

    # --- ④ 신호 분리 ---------------------------------------------------------------
    rows = [["", "intent signal", "geometric signal"],
            ["무엇", "attention", "점구름의 3D 관계"],
            ["답하는 것", "다음에 무엇을\n할 것인가", "지금 물리적으로\n무슨 상태인가"],
            ["쓰는 곳", "단계 전환\n(Pick → Place)", "완료 판정\n(Inside/Detached/Stable)"],
            ["틀릴 때", "엉뚱한 단계로 감\n— 되돌릴 수 있다", "없는 완료를 보고함\n— 되돌릴 수 없다"],
            ["지금 코드", "잠금이 무시한다\n(유지 규칙)", "일부만 — Inside 만 있고\nDetached·Stable 은 없다"]]
    _table(fig.add_subplot(gs[0, 3]), rows, fs, "④ 두 신호를 나누는 이유",
           widths=(0.20, 0.38, 0.42), fontsize=7.8)

    # --- ⑤ 측정이 말하는 것 --------------------------------------------------------
    rows = [["실측 (A1~A3)", "값", "제안에 대한 뜻"],
            ["쥔 동안 보인 사과 픽셀\n(3 카메라, F12 재현)", "143,942\n(프레임당 약 16,000)", "추적할 재료는 충분하다"],
            ["그중 자기 필터가 지운 것", "100.0 %", "지금은 하류에 사과가 없다\n— 마스크를 고쳐야 한다"],
            ["마스크를 걷었을 때\n점유 복셀 변화", "최대 4 복셀", "TSDF 로는 못 잡는다 — 움직여서\n가중치가 안 쌓인다. 추적이 옳은 도구"],
            ["강체 파지 가정 (E3)", "손 319 mm 이동 중\n표면 어긋남 중앙 0.8 mm", "움직임 상관으로\nAttached 판정 가능"],
            ["놓은 뒤 사과 재관측", "d@중심 −4.1 mm\n(프레임 20~23)", "Stable 판정 가능"],
            ["파내기 팽창 상한 (A2)", "1 복셀 (2 에서 E1 재발)", "마스크를 걷어도\n필드는 A2 가 지킨다"]]
    _table(fig.add_subplot(gs[1, 2:]), rows, fs,
           "⑤ 이미 잰 수치가 제안에 대해 말하는 것", widths=(0.28, 0.30, 0.42), fontsize=7.6)

    fig.suptitle("작업 완료 판정 — 지금 방식(스냅샷 + 그리퍼) 대 제안(object relation)."
                 "  근본 차이는 쥔 물체를 계속 보느냐다",
                 fontsize=13.0, color=fs.INK)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
