"""문서용 도식 — T0~T6 전체 통합 테스트의 **배선 현황과 결정된 프로세스 구조**.

    PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.diagrams.doc_total_test_plan

측정을 하지 않는다. `AG3S_TOTAL_TEST_Prompt.md` 의 T0~T6 를 실행하기 전에
(1) 지금 무엇이 배선되어 있고 무엇이 아닌지, (2) 세 venv 의 실제 내용과 이번에 고른
단일 프로세스 구조, (3) 단계별 게이트와 산출물을 한 장에 모은다. 숫자는 전부
이 세션에서 코드·환경을 직접 읽어 확인한 것이다.
"""

from __future__ import annotations

import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures/live-test/t0-t6-plan-map.png")

# ── 패널 1: 파이프라인 단계별 배선 현황 ──────────────────────────────────────
# (단계, live 경로에 배선됨?, 근거)
WIRING = [
    ("카메라 depth 3대",            "wired",   "wire.py: ag3s/depth·K·T·stamp"),
    ("로봇 마스크 (self-filter)",    "wired",   "pipeline._robot_mask_for"),
    ("attention lifting",           "wired",   "serve_safe.attention_extractor"),
    ("target grounding",            "wired",   "pipeline.process_multi"),
    ("잠금 (grasp latch)",           "wired",   "safe_policy._run_latch"),
    ("TSDF 적분",                   "legacy",  "pipeline._build_esdf → EsdfBuilder"),
    ("coarse ESDF 20 mm",           "legacy",  "같은 곳 — numpy scipy EDT"),
    ("fine ESDF 5 mm (2계층)",       "missing", "offline build_field.py 에만"),
    ("라벨 층",                     "wired",   "esdf.update(labelled_points)"),
    ("해석적 채널",                  "wired",   "serve_safe --static-geometry"),
    ("쥔 물체 질의점·파내기",         "wired",   "attached_points_in_base"),
    ("SQP 선형화 + OSQP",            "wired",   "refiner.refine"),
    ("안전 게이트 (verdict)",        "wired",   "wire.SafetyVerdict"),
    ("프레임별 provenance 기록",      "partial", "seq·stamp·timing_ms 만"),
    ("field age / stale 판정",       "missing", "T0 이 요구 — 없음"),
    ("격자 밖·미관측 fail-closed",    "partial", "outside_distance=None 기본"),
]

STATE_COLOR = {
    "wired":   "#1baf7a",
    "partial": "#eda100",
    "legacy":  "#eb6834",
    "missing": "#e34948",
}
STATE_LABEL = {
    "wired":   "배선됨 — live 경로가 실제로 부른다",
    "partial": "일부만 — T0 기준에 미달",
    "legacy":  "legacy backend — T0 즉시 실패 조건",
    "missing": "없음 — 이번에 만든다",
}

# ── 패널 2: venv 실측 ────────────────────────────────────────────────────────
VENVS = ["\n".join([".venv-ag3s", "py3.11 / np2.4.6"]),
         "\n".join([".venv-curobo", "py3.10 / np1.26.4"]),
         "\n".join(["openpi", "py3.11 / np1.26.4"]),
         "\n".join(["openpi-live (신설)", "py3.11 / np1.26.4"])]
PKGS = ["mujoco", "casadi", "osqp", "scipy", "torch", "jax+CUDA", "warp", "curobo"]
#  2 = 있음, 1 = 이번에 넣는다, 0 = 없음
HAVE = np.array([
    [2, 2, 2, 2, 0, 0, 0, 0],   # .venv-ag3s
    [0, 0, 0, 2, 2, 0, 2, 2],   # .venv-curobo
    [2, 2, 2, 2, 2, 2, 0, 0],   # openpi
    [2, 2, 2, 2, 2, 2, 1, 1],   # openpi-live
])

# ── 패널 3: 단계 게이트 ──────────────────────────────────────────────────────
GATES = [
    ("T0", "환경·배선 전 프레임", "manifest + 프레임별 provenance.\nlegacy EsdfBuilder 호출 0, stale 판정 동작"),
    ("T1", "연속 프레임 AG3S",    "새 seed 3개 이상, 실측 policy attention.\n모든 observation frame 의 diagnostic card"),
    ("T2", "pick-place 상태 전이", "9개 이벤트의 정확한 발생 프레임.\n조작 대상 ID 불변 · attach/detach 검증"),
    ("T3", "전 프레임 TSDF/ESDF",  "coarse+fine 매 update frame.\neikonal 0.9~1.1 · MuJoCo 참 거리 대비 낙관 오차"),
    ("T4", "fail-closed + 결합",   "격자 밖·미관측·stale·timestamp 역전이\n계획·제어를 실제로 차단하는가"),
    ("T5", "shadow closed loop",  "refined 여유거리 ≥ reference.\n지연 P50/P95/max 대 533 ms"),
    ("T6", "실제 통합 closed loop", "6 난도 × 새 seed 3개.\n금지 충돌 0 · 불확실할 때 hold"),
]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    from benchmark.ag3s.experiments.common import figstyle

    figstyle.use_korean()
    fig = plt.figure(figsize=(16.4, 11.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.52, 1.0], width_ratios=[1.02, 1.0],
                          hspace=0.26, wspace=0.14,
                          left=0.055, right=0.985, top=0.895, bottom=0.035)
    fig.patch.set_facecolor(figstyle.SURFACE)

    fig.text(0.5, 0.968, "AG3S-cuRobo 전체 통합 테스트 T0~T6 — 배선 현황과 실행 구조",
             ha="center", va="center", fontsize=16.5, color=figstyle.INK)
    fig.text(0.5, 0.936,
             "배선 상태와 패키지 유무는 2026-09-22 에 코드와 venv 를 직접 읽어 확인한 것이다. "
             "측정이 아니라 계획 도식이다.",
             ha="center", va="center", fontsize=9.8, color=figstyle.INK_2)

    # ───── 패널 1: 배선 현황 ─────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("① 지금 live 경로에 무엇이 배선되어 있는가", fontsize=12.5,
                 color=figstyle.INK, loc="left", pad=12)
    n = len(WIRING)
    for i, (stage, state, why) in enumerate(WIRING):
        y = n - 1 - i
        ax.add_patch(Rectangle((0.0, y - 0.36), 0.26, 0.72,
                               facecolor=STATE_COLOR[state], edgecolor="none", alpha=0.94))
        ax.text(-0.035, y, stage, ha="right", va="center", fontsize=9.6, color=figstyle.INK)
        ax.text(0.30, y, why, ha="left", va="center", fontsize=8.5, color=figstyle.INK_2)
    ax.set_xlim(-0.60, 1.30)
    ax.set_ylim(-2.95, n - 0.30)
    ax.axis("off")
    for j, k in enumerate(("wired", "partial", "legacy", "missing")):
        yy = -0.85 - j * 0.48
        ax.add_patch(Rectangle((-0.60, yy - 0.15), 0.19, 0.30,
                               facecolor=STATE_COLOR[k], edgecolor="none", alpha=0.94))
        ax.text(-0.36, yy, STATE_LABEL[k], ha="left", va="center",
                fontsize=8.7, color=figstyle.INK_2)

    # ───── 패널 2: venv 실측 + 고른 구조 ─────
    ax2 = fig.add_subplot(gs[0, 1])
    # 열 라벨이 격자 위에 눕기 때문에 제목은 축 안에서 직접 그린다 —
    # set_title 로 올리면 그림 전체의 부제와 겹친다.
    rows = HAVE.shape[0]
    for r in range(rows):
        for c in range(HAVE.shape[1]):
            v = HAVE[r, c]
            face = {2: "#2a78d6", 1: "#1baf7a", 0: "#eceae5"}[v]
            mark = {2: "●", 1: "+", 0: ""}[v]
            ax2.add_patch(Rectangle((c, rows - 1 - r), 0.92, 0.80,
                                    facecolor=face, edgecolor=figstyle.SURFACE,
                                    linewidth=1.6))
            if mark:
                ax2.text(c + 0.46, rows - 1 - r + 0.40, mark, ha="center", va="center",
                         fontsize=13, color="white", family="DejaVu Sans")
        ax2.text(-0.12, rows - 1 - r + 0.40, VENVS[r], ha="right", va="center",
                 fontsize=8.9, color=figstyle.INK, linespacing=1.4)
    for c, pkg in enumerate(PKGS):
        ax2.text(c + 0.46, rows + 0.06, pkg, ha="left", va="bottom", fontsize=8.8,
                 rotation=34, color=figstyle.INK_2, family="DejaVu Sans")
    ax2.text(-1.35, rows + 1.02, "② venv 실측 — 장벽은 문서보다 얕다",
             ha="left", va="bottom", fontsize=12.5, color=figstyle.INK)
    ax2.set_xlim(-1.35, len(PKGS) + 0.05)
    ax2.set_ylim(-2.62, rows + 1.52)
    ax2.axis("off")
    ax2.text(-1.35, -0.30,
             "● 이미 있음      + 이번에 넣는다 (warp-lang 1.17.0 · curobo 78fd485)"
             "      빈칸 없음",
             fontsize=8.8, color=figstyle.INK_2, va="top")
    ax2.add_patch(Rectangle((-1.35, -2.42), len(PKGS) + 1.38, 1.86,
                            facecolor="#eef4fc", edgecolor="#cde2fb", linewidth=1.2))
    ax2.text(-1.16, -0.80,
             "고른 구조 — openpi venv 를 cp -a 로 복제해 openpi-live 를 만들고\n"
             "거기에만 warp + curobo 를 넣는다. 정책을 서빙하는 원본은 손대지 않는다.\n"
             "결과: 정책 · MuJoCo · AG3S · cuRobo · SQP 가 한 프로세스 — IPC 가 없다.\n"
             "warp-lang 1.17.0 은 py3-none-manylinux 휠(requires_python ≥ 3.10)이고\n"
             "curobo 는 warp ≥ 0.10 + torch ≥ 2.5 만 요구한다 (CUDA 확장은 선택).",
             fontsize=8.8, color=figstyle.INK, va="top", linespacing=1.75)

    # ───── 패널 3: 단계 게이트 ─────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title("③ 실행 순서와 게이트 — 앞 단계가 통과해야 다음이 시작한다",
                  fontsize=12.5, color=figstyle.INK, loc="left", pad=12)
    w, gap = 1.0, 0.30
    for i, (tag, name, crit) in enumerate(GATES):
        x = i * (w + gap)
        strong = tag in ("T0", "T4")
        ax3.add_patch(Rectangle((x, 0.62), w, 0.30,
                                facecolor=figstyle.CATEGORICAL[0] if strong else "#cde2fb",
                                edgecolor="none"))
        ax3.text(x + w / 2, 0.825, tag, ha="center", va="center", fontsize=13.5,
                 color="white" if strong else figstyle.INK, family="DejaVu Sans")
        ax3.text(x + w / 2, 0.687, name, ha="center", va="center", fontsize=8.7,
                 color="white" if strong else figstyle.INK)
        ax3.text(x + w / 2, 0.555, crit, ha="center", va="top", fontsize=8.0,
                 color=figstyle.INK_2, linespacing=1.55)
        if i:
            ax3.add_patch(FancyArrowPatch((x - gap + 0.035, 0.77), (x - 0.035, 0.77),
                                          arrowstyle="-|>", mutation_scale=11,
                                          color=figstyle.INK_2, linewidth=1.2))
    ax3.text(0.0, 1.02,
             "T0 과 T4 가 게이트다 — T0 은 legacy backend 호출 0 과 stale 판정 동작을, "
             "T4 는 fail-closed 가 실제로 계획·제어를 막는지를 증명해야 통과다.",
             fontsize=9.3, color=figstyle.INK, va="bottom")
    ax3.text(0.0, 0.22,
             "실패가 나오면: 기록 → 원인 분류(구현 결함 / 하네스 결함 / 설정·환경 / 기준 자체) → "
             "직접 수정 → 영향받는 가장 앞 단계부터 새 run ID 로 재시험.\n"
             "같은 원인이 3회 수정 후에도 남거나 설계 결정(인터페이스 변경 · 알고리즘 교체 · "
             "합격 기준 변경)이 필요하면 임의로 진행하지 않고 보고한다.",
             fontsize=8.9, color=figstyle.INK_2, va="top", linespacing=1.7)
    ax3.set_xlim(-0.10, len(GATES) * (w + gap) - gap + 0.10)
    ax3.set_ylim(0.00, 1.22)
    ax3.axis("off")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
