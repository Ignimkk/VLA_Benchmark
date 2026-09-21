"""문서용 시각화 — 규칙 B 의 파이프라인 지도, 2026-09-15 판.

    PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.diagrams.doc_pipeline_map

`figures/curobo-adapter-map.png` (2026-09-12) 의 갱신판이다. 그 사이에 닫힌 것
(E3 쥔 물체 배선 · 라벨 층 · F18 목적지 마진 · F17 잠금 · F15 지지면 기본값)과 새로 뜬 것
(N1 self-collision · N2 geometry 채널 · F20 잔상)을 반영한다.

상태는 `AG3S_REVIEW_LOG.md` 의 "진행 현황" · "누적 발견" 표에서 그대로 옮긴 것이고, 이 스크립트가
새로 판정하지 않는다.
"""

from __future__ import annotations

import pathlib

OUT = pathlib.Path("benchmark/ag3s/docs/figures")

DONE = "#1baf7a"      # 닫힘 — 수치로 검증
PART = "#eda100"      # 부분 — 돌지만 남은 항목이 있다
TODO = "#d8d7d2"      # 미검토

#: (제목, 부제, 소유자, 상태색, 아래 꼬리표)
STAGES = [
    ("카메라 depth\n+ 지시문", "3 대 · 480x640 · metric", "MuJoCo → AG3S", DONE,
     "G1 depth_scale 수정\nG3 카메라 기여 검증"),
    ("로봇 마스크\n(self-filter)", "FK 로 자기 몸 픽셀 삭제", "AG3S", PART,
     "C1 확정·수정\nF12 비로봇 픽셀도 삭제"),
    ("attention\nlifting", "2D 열지도 → 3D 점", "AG3S", DONE,
     "F2 융합 후 정규화\nF9 image_hw 기본값"),
    ("target\ngrounding\n+ 잠금", "점들 → 물체 하나", "AG3S", DONE,
     "F8 잠복 판정\nF11 조작/주목 분리\nF17 걸기·유지·풀기"),
    ("TSDF → ESDF\n(2계층)", "20 mm 전역 + 5 mm 국소", "cuRobo", PART,
     "E4·E6 소멸\nC4 버퍼 복사\nN2 geom 채널 없음\nF20 잔상"),
    ("거리장 어댑터\ndistance / gradient", "층 min() 합성, 순수 numpy", "우리", PART,
     "C3 창 경계 불연속\nC5 거친 층 +7.56 mm"),
    ("SQP 선형화\nd − r − margin ≥ 0", "쥔 물체 점 · 목적지 마진", "trajopt", PART,
     "E3 배선 완료\nF18 목적지 20 mm\nN1 self-collision 없음"),
    ("QP 해\n(OSQP)", "반복 1 회", "trajopt", DONE,
     "회귀 기준선\n해소 14 / 개선 15"),
]

LANES = [(0, 3, "지각 — 무엇이 어디에 있는가", "#2a78d6"),
         (4, 7, "기하 → 제약 → 해", "#eb6834")]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments.common import figstyle
    figstyle.use_korean()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

    fig, ax = plt.subplots(figsize=(21.0, 9.0))
    fig.patch.set_facecolor(figstyle.SURFACE)
    ax.set_facecolor(figstyle.SURFACE)
    ax.set_xlim(-0.4, len(STAGES) * 2.5 - 0.1)
    ax.set_ylim(-3.9, 3.2)
    ax.axis("off")

    w, h = 2.05, 1.5
    for i, (title, sub, owner, colour, notes) in enumerate(STAGES):
        x = i * 2.5
        ax.add_patch(FancyBboxPatch((x, -h / 2), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    facecolor=colour, edgecolor="#0b0b0b", lw=1.1))
        dark = colour is TODO
        ax.text(x + w / 2, 0.36, title, ha="center", va="center", fontsize=11.5,
                weight="bold", color="#0b0b0b" if dark else "#ffffff")
        ax.text(x + w / 2, -0.20, sub, ha="center", va="center", fontsize=8.6,
                color="#1b1b1b" if dark else "#f2f2f2")
        ax.text(x + w / 2, -0.55, owner, ha="center", va="center", fontsize=9,
                style="italic", color="#2b2b2b" if dark else "#e8e8e8")
        ax.text(x + w / 2, -1.15, notes, ha="center", va="top", fontsize=8.6,
                color=figstyle.INK_2, linespacing=1.5)
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, 0), (x + 2.5 - 0.05, 0),
                                         arrowstyle="-|>", mutation_scale=17,
                                         color="#0b0b0b", lw=1.4))

    for a, b, label, colour in LANES:
        x0, x1 = a * 2.5, b * 2.5 + w
        ax.add_patch(FancyArrowPatch((x0, 1.45), (x1, 1.45), arrowstyle="<|-|>",
                                     mutation_scale=13, color=colour, lw=1.4))
        ax.text((x0 + x1) / 2, 1.72, label, ha="center", fontsize=12, weight="bold",
                color=colour)

    # 되먹임 — 쥔 물체 · 목적지 · 잠금은 grounding 에서 선형화로 바로 간다
    ax.add_patch(FancyArrowPatch((3 * 2.5 + w / 2, -2.30), (6 * 2.5 + w / 2, -2.30),
                                 arrowstyle="-|>", mutation_scale=15, color="#4a3aa7",
                                 lw=1.5, linestyle="--",
                                 connectionstyle="arc3,rad=-0.10"))
    ax.text((3 * 2.5 + 6 * 2.5) / 2 + w / 2, -2.66,
            "조작 대상 · 목적지 · attach/detach — 필드를 거치지 않고 여유거리 정책으로 직행한다\n"
            "(E1 이 금지한 '필드에서 target 을 파내기' 를 대신하는 경로)",
            ha="center", fontsize=9.5, color="#4a3aa7", linespacing=1.5)

    handles = [Patch(facecolor=DONE, edgecolor="#0b0b0b", label="닫힘 — 실측으로 검증"),
               Patch(facecolor=PART, edgecolor="#0b0b0b", label="돌지만 남은 항목이 있다"),
               Patch(facecolor=TODO, edgecolor="#0b0b0b", label="미검토")]
    ax.legend(handles=handles, loc="lower left", fontsize=10.5, frameon=False,
              bbox_to_anchor=(0.0, -0.02), ncol=3)

    ax.text(len(STAGES) * 2.5 - 0.15, -3.60,
            "남은 것 셋 (구분표 순서):  N1 self-collision 이 아예 없다  →  "
            "N2 geometry 채널이 없다  →  F20 사라진 물체의 잔상이 8 프레임 남는다",
            ha="right", fontsize=10.5, color="#e34948", weight="bold")

    ax.set_title("AG3S / trajopt 파이프라인 — 전체 틀에서 지금 어디인가 (2026-09-15)",
                 fontsize=15.5, pad=18)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "doc-pipeline-map.png"
    fig.savefig(out, dpi=118, bbox_inches="tight", facecolor=figstyle.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
