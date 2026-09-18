"""작업 완료 판정 — 선택지 셋을 한자리에 놓는다 (2026-09-17).

코드를 바꾸지 않는다. A3 에서 **구현해 배포한 것**과 사용자가 제안한 둘을, **이미 잰 수치**와
함께 나란히 놓아 고를 수 있게 만드는 그림이다. 수치의 출처는 전부 `AG3S_REVIEW_LOG.md` 의
해당 절이고, 재현 스크립트는 `a3_release_signal.py` 와 `a4_task_tsdf.py` 다.

실행:
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -m \
        benchmark.ag3s.experiments.a5_completion_options
"""

from __future__ import annotations

import argparse
import pathlib

OUT = pathlib.Path("benchmark/ag3s/docs/figures/a5-completion-options.png")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments import figstyle
    figstyle.use_korean()
    return figstyle


def _table(ax, rows, fs, title, widths, fontsize=7.8, scale=1.9, pad=14):
    ax.axis("off")
    ax.set_title(title, fontsize=10.4, color=fs.INK, pad=pad)
    t = ax.table(cellText=rows, colWidths=list(widths), loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(fontsize); t.scale(1, scale)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if r == 0:
            cell.set_facecolor("#ecebe7"); cell.set_text_props(weight="bold")
        elif all(x == "" for x in rows[r][1:]) and rows[r][0]:
            cell.set_facecolor("#f2f1ed"); cell.set_text_props(weight="bold")
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    fs = _style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.4, 10.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0],
                          height_ratios=[1.0, 1.15], wspace=0.16, hspace=0.24)

    # --- ① 세 안의 판정식 ----------------------------------------------------------
    rows = [["", "A — 지금 (배포됨)", "B — object relation", "C — 작업용 TSDF"],
            ["판정식",
             "Inside AND 그리퍼 열림",
             "Inside AND Detached\nAND Stable",
             "attention 을 TSDF 에 적고\n위치 변동·안착으로 판정"],
            ["쥔 물체를\n무엇으로 보나",
             "파지 순간 스냅샷\n(손에 강체로 용접)",
             "프레임마다 다시 잡는\n점구름",
             "TSDF 의 점유 복셀"],
            ["목적지를\n무엇으로 보나",
             "거리장 라벨 층의 AABB",
             "attention 으로 올린\n점구름 + 내부 volume",
             "TSDF 에 적힌 attention"],
            ["코드",
             "trajopt/placed.py + grasp_latch\nsafe_replay --placed-fn",
             "없음 (신규)",
             "없음 (신규)"],
            ["상태", "구현·검증 완료", "설계만", "설계만"]]
    _table(fig.add_subplot(gs[0, :]), rows, fs, "① 세 안이 무엇을 하는가",
           widths=(0.15, 0.27, 0.28, 0.30), fontsize=8.0, scale=2.1)

    # --- ② 관계별로 누가 답하나 ----------------------------------------------------
    rows = [["관계", "A", "B", "C", "측정된 사실"],
            ["Attached\n(쥐었나)", "그리퍼\n(명령)", "움직임 상관\n(결과)", "안 됨",
             "이동 중 TSDF 0 복셀\n강체 파지 어긋남 0.8 mm"],
            ["Inside\n(안에 있나)", "AABB", "내부 volume", "attention 영역",
             "AABB 는 회전한 목적지에\n헐거워진다"],
            ["Detached\n(놓았나)", "잴 수 없음\n→ 그리퍼", "움직임 상관", "옛 자리가 빈다",
             "옛 자리 84 → 61\n(감쇠 켜도 27 % 만)"],
            ["Stable\n(멈췄나)", "없음", "위치 변화", "점유가 다시 쌓인다",
             "0 → 168 복셀\nTSDF 의 특기"]]
    _table(fig.add_subplot(gs[1, 0]), rows, fs, "② 네 관계를 누가 답하는가",
           widths=(0.16, 0.14, 0.17, 0.20, 0.33), fontsize=7.6, scale=2.3)

    # --- ③ 전제·비용·못 잡는 것 ----------------------------------------------------
    rows = [["", "값 / 내용"],
            ["A — 지금", ""],
            ["  검증", "참값과 0 프레임 차이, 깜빡임 0 / 25"],
            ["  못 잡는 것", "빈손 파지 · 미끄러짐 · 안 열리는 그리퍼"],
            ["  그때의 손해", "파내기가 손을 따라다닌다 (팔 낙관 +19.7 mm)"],
            ["B — object relation", ""],
            ["  전제", "로봇 마스크에서 쥔 물체 제외 (지금 100 % 삭제)"],
            ["  그 전제의 안전", "A2 의 파내기가 확보 (팽창 1, 2 에서 E1 재발)"],
            ["  재료", "사과 16,000 px/프레임 — 충분"],
            ["  새로 만들 것", "추적 1 개 + 관계 3 개"],
            ["C — 작업용 TSDF", ""],
            ["  비용", "적분 64 ms / 갱신 134 ms = 48 %  (ESDF 불필요)"],
            ["  분리의 진짜 근거", "감쇠와 마스킹이 장애물용과 정반대"],
            ["  안 되는 단계", "2 번 (움직임) — 이동 중 0 복셀"],
            ["  오염되는 단계", "4 번 — 바구니 점유가 관측 확대로 +130 (놓기 전)"]]
    _table(fig.add_subplot(gs[1, 1]), rows, fs, "③ 전제 · 비용 · 못 잡는 것",
           widths=(0.42, 0.58), fontsize=7.6, scale=1.52, pad=24)

    fig.suptitle("작업 완료 판정 — 선택지 셋.  A 는 배포돼 있고, B·C 는 서로 배타적이지 않다",
                 fontsize=13.0, color=fs.INK)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
