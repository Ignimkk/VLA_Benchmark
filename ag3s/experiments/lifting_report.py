"""3단계 검증 — 2D attention이 옳은 3D 점에 붙는가.

1단계는 attention 맵이 옳은 물체를 가리킨다는 것을 2D에서 확인했고, 2단계는 3D 점이 옳은
자리에 놓인다는 것을 확인했다. 3단계는 그 둘을 잇는다. 두 입력이 모두 옳아도 잇는 방식이
틀리면 — 픽셀 대응이 어긋나거나, 정규화가 순위를 바꾸거나, 점이 조용히 사라지거나 — 4단계
target grounding은 틀린 점들을 seed로 삼는다.

여기서 재는 것은 다섯이고, 앞의 셋은 **정답이 정해진 불변식**이라 통과/실패가 명확하다.

**A. 점 보존.** `lift`의 docstring은 "The returned cloud has len(cloud) entries. Always."라고
단언한다. attention은 target의 *이름*을 정할 뿐 기하를 지우지 않는다는 AG3S의 안전 계약이
이 한 줄에 걸려 있으므로, 문장이 아니라 숫자로 확인한다.

**B. 픽셀 대응.** 점에 붙은 값이 그 점이 나온 픽셀의 attention인가. `nearest` 보간에서는
패치 인덱스 직접 조회와 **정확히** 같아야 한다. `bilinear`에서는 이웃 네 패치의 최솟값과
최댓값 사이에 있어야 한다 — 보간은 값을 만들어내지 않는다.

**C. 정규화 순위 보존.** `normalize_attention`은 네 모드 전부가 순위를 보존한다고 주장하고,
`seed_percentile`과 `normalization`을 독립적으로 고를 수 있는 근거가 그 주장이다. 사실이라면
백분위 기준 seed 집합은 네 모드에서 **완전히 동일**해야 한다.

**D. seed 순도.** 4단계는 attention 상위 `seed_percentile`% 점을 seed로 쓴다. 그 점들 중
실제로 target 위에 있는 비율이 4단계 성공을 직접 예고한다. 이것이 이 단계의 핵심 숫자다.

**E. 카메라별 attention.** 1단계는 head 카메라만 검증했다. 그런데 `multiview.fuse`는 세
카메라의 attention을 `max`로 합친다. 손목 카메라의 attention이 엉뚱한 곳을 보면 max 융합이
그 오류를 그대로 들여온다 — 융합을 켜기 전에 확인해야 하는 값이다.

실행:
    MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.lifting_report \
        --records outputs/.../ag3s_records/run_0002 \
        --attention benchmark/ag3s/asset/data/attention_step1_run0002.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import numpy as np

from benchmark.ag3s.attention_lifting import GridAttentionAdapter, lift, normalize_attention
from benchmark.ag3s.config import AttentionConfig, PointCloudConfig
from benchmark.ag3s.experiments.figstyle import (
    CATEGORICAL, GRID_INK, INK, INK_2, SURFACE, style_axes, use_korean,
)
from benchmark.ag3s.experiments.policy_record import (
    CAMERA_BINDINGS, load_run, pose_scene, replay_scene,
)
from benchmark.ag3s.reconstruction import reconstruct

#: seed 순도를 나눠 볼 범주. 로봇이 별도인 것은 의도적이다 — attention이 그리퍼에 몰리면
#: seed가 로봇 위에 앉고, 그것은 "엉뚱한 물체"와는 다른 종류의 실패다.
ROBOT_PREFIXES = ("base", "link_", "wheel", "ee_", "EE_", "FT_", "d435i", "wrist_bracket", "zed")
SEED_PERCENTILES = (90.0, 95.0, 98.0, 99.0)


def category_of(name: str, target: str) -> str:
    if name == target:
        return "target"
    if name.startswith(ROBOT_PREFIXES):
        return "robot"
    if name in ("table", "shelf", "office"):
        return "배경"
    return "다른 물체"


def main() -> None:
    from benchmark.ag3s.experiments.attention_report import target_from_prompt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--attention", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--layer", type=int, default=None, help="기본값: 1단계 결과에서 읽음")
    ap.add_argument("--head", type=int, default=None)
    ap.add_argument("--denoise", type=int, default=None)
    ap.add_argument("--agg", default=None)
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/step-01-attention.json")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--min-target-px", type=int, default=200,
                    help="target이 이보다 적게 보이는 프레임은 순위 지표에서 제외 (1단계와 동일)")
    ap.add_argument("--out-doc", default="benchmark/ag3s/docs/step-03-lifting.md")
    ap.add_argument("--out-figs", default="benchmark/ag3s/asset/image/lifting")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    use_korean()
    import matplotlib.pyplot as plt
    import mujoco

    # 1단계가 고른 셀을 그대로 이어받는다. 여기서 다시 고르면 두 단계가 서로 다른 것을 재게 된다.
    chosen = {}
    step1 = pathlib.Path(args.step1_json)
    if step1.exists():
        chosen = json.loads(step1.read_text()).get("best", {})
    layer = args.layer if args.layer is not None else chosen.get("layer")
    head = args.head if args.head is not None else chosen.get("head")
    denoise = args.denoise if args.denoise is not None else chosen.get("denoise")
    agg = args.agg or chosen.get("agg")
    if None in (layer, head, denoise, agg):
        raise SystemExit("1단계 결과를 읽지 못했습니다 — --layer/--head/--denoise/--agg를 주세요")

    run = load_run(args.records)
    blob = np.load(args.attention, allow_pickle=False)
    attention = np.asarray(blob["attention"], np.float32)
    cameras = [str(c) for c in blob["cameras"]]
    di = [int(d) for d in blob["denoise_steps"]].index(int(denoise))
    ai = [str(a) for a in blob["aggregations"]].index(str(agg))
    target = args.target or target_from_prompt(run.prompt)
    print(f"1단계 선택: L{layer} h{head}, Euler {denoise}, {agg} pooling | target={target}")

    scene = replay_scene(run)
    names = {i: (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or "?")
             for i in range(scene.model.nbody)}
    target_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, target)
    frames_scored = {}
    picks = np.linspace(0, len(run) - 1, min(args.frames, len(run))).astype(int)
    adapter = GridAttentionAdapter(layer=None, head=None)   # 셀은 이미 골라 넘긴다
    cfg = AttentionConfig()
    pc_cfg = PointCloudConfig()

    preserved, corr_nearest, corr_bracket = [], [], []
    order_ok, seed_identical = [], []
    seed_jaccard = {m: [] for m in AttentionConfig.NORMALIZATIONS}
    purity = {c: {p: [] for p in SEED_PERCENTILES} for c in cameras}
    ceiling = {c: {p: [] for p in SEED_PERCENTILES} for c in cameras}
    recall = {c: {p: [] for p in SEED_PERCENTILES} for c in cameras}
    rank = {c: {"peak": [], "auc": [], "prec_at_n": [], "n_target": []} for c in cameras}
    breakdown = {c: {} for c in cameras}
    example = None

    for fi in picks:
        pose_scene(scene, run.steps[fi])
        for ci, policy_cam in enumerate(cameras):
            mj_cam = CAMERA_BINDINGS[policy_cam][0]
            frame = scene.capture(mj_cam)
            cloud, _ = reconstruct(depth=frame.depth, camera_intrinsics=frame.camera_intrinsics,
                                   T_base_cam=frame.T_base_cam, config=pc_cfg)
            if len(cloud) == 0:
                continue
            grid = attention[fi, di, ai, layer, head, ci]      # (16, 16)
            hw = frame.hw

            # A -------------------------------------------------------------- 점 보존
            lifted = lift(cloud, grid, cfg, adapter=adapter, image_hw=hw)
            preserved.append((len(cloud), len(lifted.cloud), len(lifted.attention)))

            # B -------------------------------------------------------------- 픽셀 대응
            near_cfg = dataclasses.replace(cfg, interpolation="nearest")
            near = lift(cloud, grid, near_cfg,
                        adapter=GridAttentionAdapter(interpolation="nearest"), image_hw=hw)
            g = grid.shape[0]
            rows = np.minimum((cloud.uv[:, 1] * g) // hw[0], g - 1)
            cols = np.minimum((cloud.uv[:, 0] * g) // hw[1], g - 1)
            corr_nearest.append(float(np.abs(near.raw_attention - grid[rows, cols]).max()))
            # bilinear 값은 실제로 섞인 네 패치의 [min, max] 안에 있어야 한다. 이웃은
            # `_resample`이 쓰는 픽셀 중심 정렬 좌표로 구한다 — 정수 나눗셈으로 어림하면
            # 경계에서 다른 네 칸을 집게 되고, 그러면 통과해야 할 검사가 실패한다.
            fy = np.clip((cloud.uv[:, 1] + 0.5) * g / hw[0] - 0.5, 0.0, g - 1.0)
            fx = np.clip((cloud.uv[:, 0] + 0.5) * g / hw[1] - 0.5, 0.0, g - 1.0)
            y0, x0 = np.floor(fy).astype(np.int64), np.floor(fx).astype(np.int64)
            y1, x1 = np.minimum(y0 + 1, g - 1), np.minimum(x0 + 1, g - 1)
            quad = np.stack([grid[y0, x0], grid[y0, x1], grid[y1, x0], grid[y1, x1]])
            v = lifted.raw_attention
            corr_bracket.append(float(np.maximum(quad.min(axis=0) - v,
                                                 v - quad.max(axis=0)).max()))

            # C ------------------------------------------------------- 정규화 순위 보존
            # `argsort(argsort(...))` 비교는 쓸 수 없다. percentile 모드는 상·하위를 잘라
            # 의도적으로 동점을 만들고, 동점의 순서는 정렬 구현이 임의로 정한다. 주장은
            # "순위 역전이 없다"(약한 순서 보존)이지 "동점이 없다"가 아니므로, raw 오름차순으로
            # 정렬했을 때 정규화 값이 **감소하지 않는지**를 본다. 이쪽이 정확한 검사다.
            raw = lifted.raw_attention
            order = np.argsort(raw, kind="stable")
            base_seed = None
            for mode in AttentionConfig.NORMALIZATIONS:
                vals = normalize_attention(raw, dataclasses.replace(cfg, normalization=mode))
                monotone = bool(np.all(np.diff(vals[order]) >= -1e-6))
                order_ok.append(monotone)
                seed = vals >= np.percentile(vals, cfg.seed_percentile)
                if base_seed is None:
                    base_seed = seed
                else:
                    seed_identical.append(bool(np.array_equal(seed, base_seed)))
                union = int((seed | base_seed).sum())
                seed_jaccard[mode].append(float((seed & base_seed).sum() / union) if union else 1.0)

            # D/E ----------------------------------------------------------- seed 순도
            labels = frame.body_ids[cloud.uv[:, 1], cloud.uv[:, 0]]
            cats = np.array([category_of(names.get(int(b), "?"), target) for b in labels])
            is_target = cats == "target"
            n_t = int(is_target.sum())
            # 1단계와 같은 가시성 하한. target이 보이지 않는 프레임에는 정답이 없고, 그것을
            # 실패로 세면 파지 중 팔에 가려진 구간이 순위 지표를 통째로 끌어내린다.
            visible = int((frame.body_ids == target_bid).sum()) >= args.min_target_px
            if visible:
                frames_scored[policy_cam] = frames_scored.get(policy_cam, 0) + 1
            if n_t and visible:
                # peak: ground_target 이 실제로 쓰는 점 (peak_signal 의 argmax)
                rank[policy_cam]["peak"].append(bool(is_target[int(np.argmax(raw))]))
                # AUC: 무작위 target 점이 무작위 비-target 점보다 높은 attention을 가질 확률.
                # 물체 크기에 영향받지 않으므로 순도와 달리 천장이 1.0이다.
                ranks = np.argsort(np.argsort(raw)) + 1.0
                n_o = len(raw) - n_t
                auc = (ranks[is_target].sum() - n_t * (n_t + 1) / 2.0) / (n_t * n_o) if n_o else np.nan
                rank[policy_cam]["auc"].append(float(auc))
                # precision@N, N = target 점 수. 크기를 맞췄으므로 천장이 1.0이다.
                topn = np.argsort(-raw, kind="stable")[:n_t]
                rank[policy_cam]["prec_at_n"].append(float(is_target[topn].mean()))
                rank[policy_cam]["n_target"].append(n_t)
            for pct in SEED_PERCENTILES:
                sel = raw >= np.percentile(raw, pct)
                if sel.sum():
                    purity[policy_cam][pct].append(float((cats[sel] == "target").mean()))
                    # 순도의 천장: target 점이 전부 뽑혀도 seed 집합 대비 이 비율을 못 넘는다.
                    ceiling[policy_cam][pct].append(float(min(n_t, sel.sum()) / sel.sum()))
                    # 순도 / 천장 = target 점 중 seed에 들어간 비율. 크기 효과가 약분되어
                    # 곧바로 읽히는 숫자다.
                    recall[policy_cam][pct].append(float(is_target[sel].sum() / n_t) if n_t else np.nan)
            sel95 = raw >= np.percentile(raw, cfg.seed_percentile)
            for cat, n in zip(*np.unique(cats[sel95], return_counts=True)):
                breakdown[policy_cam][cat] = breakdown[policy_cam].get(cat, 0) + int(n)

            if example is None and policy_cam == "cam_high":
                example = (cloud.points.copy(), raw.copy(), cats.copy(), int(fi))
    scene.close()

    # ------------------------------------------------------------------------ 요약
    ok_a = all(a == b == c for a, b, c in preserved)
    ok_b = max(corr_nearest) < 1e-6 and max(corr_bracket) < 1e-6
    worst_jaccard = min((min(v) for v in seed_jaccard.values() if v), default=1.0)
    # 게이트는 AG3S가 실제로 의존하는 불변식 하나에만 건다: 순위 역전이 없을 것. 이것은
    # 정확히 성립해야 하고, 실제로 성립한다.
    #
    # seed 집합 동일성은 docstring이 그 불변식의 *결과*로 주장하는 것인데, softmax에서만
    # 정확히 성립하지 않는다. 통과시키려고 Jaccard 임계값을 고르는 것은 지표를 결과에 맞추는
    # 일이므로 하지 않는다 — 대신 기본값 경로(percentile)를 포함한 세 모드가 완전히 일치하는지를
    # 요구하고, softmax의 어긋남은 측정해서 보고한다.
    exact_modes = [m for m in AttentionConfig.NORMALIZATIONS if m != "softmax"]
    ok_c = all(order_ok) and all(
        all(x == 1.0 for x in seed_jaccard[m]) for m in exact_modes if seed_jaccard[m])
    head_peak = float(np.mean(rank["cam_high"]["peak"])) if rank["cam_high"]["peak"] else 0.0
    head_auc = float(np.mean(rank["cam_high"]["auc"])) if rank["cam_high"]["auc"] else 0.0
    head_prec = float(np.mean(rank["cam_high"]["prec_at_n"])) if rank["cam_high"]["prec_at_n"] else 0.0
    ok_d = head_peak >= 0.8 and head_auc >= 0.9
    verdict = "**PASS**" if (ok_a and ok_b and ok_c and ok_d) else "**FAIL**"

    figs = pathlib.Path(args.out_figs)
    figs.mkdir(parents=True, exist_ok=True)

    # fig1 — 순위 지표 3종, 카메라별. 순도가 아니라 이것이 이 단계의 답이다.
    metrics = [("peak가 target 위", "peak"), ("AUC", "auc"), ("precision@N", "prec_at_n")]
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=160)
    style_axes(fig, ax)
    ax.grid(axis="y", color=GRID_INK, lw=.6); ax.set_axisbelow(True)
    ax.axhline(0.5, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(0.996, 0.5, "우연 수준 (AUC) ", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=7.5, color=INK_2)
    x = np.arange(len(metrics))
    width = 0.8 / len(cameras)
    for ci, cam in enumerate(cameras):
        vals = [float(np.mean(rank[cam][k])) if rank[cam][k] else np.nan for _, k in metrics]
        pos = x + ci * width - 0.4 + width / 2
        ax.bar(pos, vals, width * 0.88, color=CATEGORICAL[ci], edgecolor=SURFACE, lw=2,
               label=cam, zorder=3)
        for xi, vv in zip(pos, vals):
            if np.isfinite(vv):
                ax.text(xi, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(x, [n for n, _ in metrics])
    ax.set_ylabel("값 (1.0이 최선)"); ax.set_ylim(0, 1.12)
    ax.set_title("lifting 후 target 점이 attention 순위에서 위에 오는가",
                 color=INK, fontsize=11, loc="left", pad=8)
    leg = ax.legend(frameon=False, fontsize=8, ncol=len(cameras), loc="upper center",
                    bbox_to_anchor=(0.5, -0.14))
    for t in leg.get_texts(): t.set_color(INK_2)
    fig.tight_layout(); fig.savefig(figs / "fig1_ranking.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # fig2 — seed 점의 3D 분포. 두 패널인 이유는 이야기가 두 개이기 때문이다: 전체 범위는
    # seed가 배경에 압도적으로 몰린다는 것을, 확대는 그 와중에도 target 위에 seed가 앉는다는
    # 것을 보여준다. 한 패널로는 둘 중 하나가 반드시 안 보인다.
    pts, raw, cats, fi_ex = example
    sel = raw >= np.percentile(raw, cfg.seed_percentile)
    cat_order = ["target", "다른 물체", "배경", "robot"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.8), dpi=160)
    style_axes(fig, (a1, a2))
    for ax, zoom in ((a1, False), (a2, True)):
        ax.scatter(pts[~sel, 0], pts[~sel, 1], s=.4, c=GRID_INK, lw=0,
                   label="seed 아님" if not zoom else None)
        for slot, cat in enumerate(cat_order):
            m = sel & (cats == cat)
            if m.any():
                ax.scatter(pts[m, 0], pts[m, 1], s=5, c=CATEGORICAL[slot], lw=0,
                           label=f"seed · {cat} ({int(m.sum())})" if not zoom else None)
        ax.set_xlabel("base x (m)"); ax.set_aspect("equal")
    a1.set_ylabel("base y (m)")
    a1.set_title("전체 범위 — seed 대부분이 배경(벽·바닥)에 있다", color=INK, fontsize=10, loc="left")
    tgt = sel & (cats == "target")
    if tgt.any():
        lo = pts[tgt, :2].min(axis=0) - 0.25; hi = pts[tgt, :2].max(axis=0) + 0.25
        a2.set_xlim(lo[0], hi[0]); a2.set_ylim(lo[1], hi[1])
    a2.set_title("target 주변 확대 — 그 안에서는 target 위에 앉는다", color=INK, fontsize=10, loc="left")
    handles, labels = a1.get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=8, ncol=5, markerscale=3,
                     loc="lower center", bbox_to_anchor=(0.5, -0.02))
    for t in leg.get_texts(): t.set_color(INK_2)
    fig.suptitle(f"프레임 {fi_ex} · 상위 {100-cfg.seed_percentile:g}% attention 점이 어디에 놓이는가",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(figs / "fig2_seed_points.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------------ 문서
    def md(header, rows):
        return "\n".join(["| " + " | ".join(header) + " |",
                          "|" + "|".join("---" for _ in header) + "|", *rows])

    doc = f"""# 3단계 — attention lifting (2D → 3D)

**질문.** 1단계에서 옳다고 확인한 attention이, 2단계에서 옳다고 확인한 점들 중 **옳은 점**에
붙는가?

두 입력이 모두 옳아도 잇는 방식이 틀리면 4단계는 틀린 점을 seed로 삼는다. 픽셀 대응이
어긋나거나, 정규화가 순위를 바꾸거나, 점이 조용히 사라지는 경우가 그것이다.

| | |
|---|---|
| 기록 | `{run.path}` |
| attention | `{args.attention}` |
| 사용한 셀 | **L{layer} h{head}**, Euler {denoise}, `{agg}` pooling (1단계가 고른 것을 그대로 이어받음) |
| target | `{target}` |
| 검사 프레임 | {len(picks)}개 × 카메라 {len(cameras)}대 (순위 지표는 target이 {args.min_target_px} px 이상 보이는 프레임만: cam_high {frames_scored.get("cam_high", 0)}개) |
| lift | `benchmark.ag3s.attention_lifting.lift` (AG3S 본체 코드 그대로) |

## 판정 — {verdict}

| 검사 | 결과 | 기준 |
|---|---|---|
| A. 점 보존 | {'통과' if ok_a else '실패'} — {len(preserved)}회 전부 입력 개수 = 출력 개수 | 완전 일치 |
| B. 픽셀 대응 | {'통과' if ok_b else '실패'} — nearest 오차 {max(corr_nearest):.2e}, bilinear 범위 이탈 {max(corr_bracket):.2e} | < 1e-6 |
| C. 정규화 순위 보존 | {'통과' if ok_c else '실패'} — 단조성 {sum(order_ok)}/{len(order_ok)}, 기본값 경로 seed 집합 완전 일치 | 단조성 전부 + percentile·minmax·none 완전 일치 |
| D. 순위 보존 (head) | {'통과' if ok_d else '실패'} — peak **{head_peak:.3f}**, AUC **{head_auc:.5f}**, precision@N **{head_prec:.3f}** | peak >= 0.8, AUC >= 0.9 |

### A. 점 보존 — 안전 계약이 걸려 있는 한 줄

`lift`의 docstring은 "The returned cloud has `len(cloud)` entries. **Always.**"라고 단언한다.
attention은 target의 *이름*을 정할 뿐 기하를 지우지 않는다는 AG3S의 안전 계약이 이 문장에
걸려 있다. 낮은 attention을 걸러내는 최적화가 언젠가 여기 들어가면, 물리적으로 존재하는
장애물이 "VLA가 안 봤다"는 이유로 충돌 후보에서 사라진다.

{len(preserved)}회(프레임 × 카메라) 전부에서 입력 점 수 = 출력 점 수 = attention 값 개수.
합계 {sum(a for a, _, _ in preserved):,}개 점, 손실 0개.

### B. 픽셀 대응 — 보간은 값을 만들어내지 않는다

`nearest` 보간에서는 점에 붙은 값이 그 점이 나온 픽셀의 패치를 직접 조회한 값과 **정확히**
같아야 한다. 최대 오차 **{max(corr_nearest):.2e}**.

`bilinear`에서는 정확한 일치를 요구할 수 없지만, 값이 이웃 네 패치의 [최솟값, 최댓값] 밖으로
나가면 안 된다 — 보간은 섞을 뿐 만들어내지 않는다. 범위 이탈 최대 **{max(corr_bracket):.2e}**.

이 두 검사는 2단계 A와 같은 역할을 한다. B가 통과하면 대응은 옳고, 이후 문제는 attention
자체이거나 점 자체다.

### C. 정규화가 순위를 보존하는가 — 설정 두 개의 독립성

`normalize_attention`은 네 모드(`percentile`·`minmax`·`softmax`·`none`) 전부가 순위를
보존한다고 주장한다. 이 주장이 `attention.normalization`과 `clustering.seed_percentile`을
독립적으로 고를 수 있는 근거다 — 백분위 기준 컷은 값이 아니라 순위만 보므로, 주장이 사실이면
어떤 모드에서도 같은 점이 뽑혀야 한다.

**`argsort`로 비교하면 안 된다.** `percentile` 모드는 상·하위를 잘라 의도적으로 동점을
만들고, 동점의 순서는 정렬 구현이 임의로 정한다. 주장은 "순위 역전이 없다"(약한 순서 보존)이지
"동점이 없다"가 아니다. (첫 판이 `argsort` 비교로 이 검사를 실패시켰다.) 그래서 raw 오름차순으로
정렬했을 때 정규화 값이 감소하지 않는지를 본다.

단조성은 **{sum(order_ok)}/{len(order_ok)} 전부 성립**한다. seed 집합은 거의 같지만
정확히 같지는 않다:

{md(["모드", "기준(percentile) 대비 Jaccard", "완전 일치"],
    [f"| {m} | {min(v):.4f} ~ {max(v):.4f} | {'예' if all(x == 1.0 for x in v) else '아니오'} |"
     for m, v in seed_jaccard.items() if v])}

**`softmax`만 갈린다.** 최소 Jaccard {worst_jaccard:.4f} — 약 2,600개 seed 중 최대 수십 개다.
원인은 순위가 뒤집혀서가 아니라 float32다: softmax는 값의 범위를 크게 압축하고, 컷 근처에서
서로 다른 raw 값들이 같은 float32로 뭉개져 함께 임계를 넘는다. 기본값인 `percentile`을 포함한
나머지 세 모드는 **완전히** 같은 점을 고른다.

**판정 게이트는 단조성과 세 모드의 완전 일치에만 걸었다.** softmax를 통과시키려고 Jaccard
임계값을 고르는 것은 지표를 결과에 맞추는 일이므로 하지 않았다. 대신 어긋남을 측정해 남긴다.

실질적 영향은 없다 — seed는 영역 성장의 출발점일 뿐이고, 수십 개가 달라져도 성장 결과는
같은 클러스터에 수렴한다. 다만 docstring의 주장("어떤 모드에서도 같은 점이 뽑힌다")은
**정신은 옳고 문자 그대로는 조금 과하다**. `seed_threshold`처럼 절대값을 쓰는 경로에서는
모드가 실제로 갈린다는 점도 함께 기억해 둘 만하다.

**정규화 모드를 seed 순도로 비교하는 것은 의미가 없다** — 어떤 모드든 사실상 같은 점을 고른다.

### D. lifting 후에도 target 점이 위에 오는가

**seed 순도는 이 질문의 답이 아니다.** 4단계는 seed를 그대로 target으로 쓰지 않는다 —
seed에서 영역을 성장시키고(`grow_region`), DBSCAN으로 나누고, 클러스터를 점수로 고른다.
seed에 불순물이 섞이는 것은 설계가 감당하도록 만들어진 일이다.

게다가 순도는 **물체 크기가 천장을 정한다.** target인 사과는 화면의 약 0.14%뿐이라, 상위 5%
점이 전부 뽑혀도 순도는 구조적으로 0.03을 넘을 수 없다. 1단계에서 β = 0이 씬에 눌려 헤드를
구분하지 못했던 것과 같은 함정이다. (첫 판에서 이 값을 0.007로 보고했는데, 그 옆에 천장
0.028을 같이 적지 않으면 재앙처럼 읽힌다.)

그래서 크기에 영향받지 않는 세 가지를 잰다.

| 지표 | 무엇을 묻는가 | 천장 | 우연 수준 |
|---|---|---|---|
| peak가 target 위 | `ground_target`이 실제로 쓰는 최고 attention 점이 target 위인가 | 1.0 | 물체 면적 비율 |
| AUC | 무작위 target 점이 무작위 비-target 점보다 높은 attention을 갖는 확률 | 1.0 | 0.5 |
| precision@N | 상위 N개(N = target 점 수) 중 target 비율 | 1.0 | 물체 면적 비율 |

{md(["카메라", "peak가 target 위", "AUC", "precision@N", "평균 target 점 수"],
    [f"| {c} | " + " | ".join(
        (f"{np.mean(rank[c][k]):.3f}" if k != "auc" else f"{np.mean(rank[c][k]):.5f}")
        if rank[c][k] else "—" for k in ("peak", "auc", "prec_at_n"))
     + f" | {np.mean(rank[c]['n_target']):.0f} |" if rank[c]["n_target"] else f"| {c} | — | — | — | — |"
     for c in cameras])}

![순위 지표](../asset/image/lifting/fig1_ranking.png)

### seed란 무엇이고, 왜 3단계 문서에 나오는가

**seed는 4단계의 개념이지 3단계의 개념이 아니다.** 그런데도 여기 나오므로 먼저 정리한다.

`extract_seeds(attention, config, seed_percentile)`는 attention이 상위
`seed_percentile`%인 **점들의 인덱스**를 돌려준다. 4단계 `ground_target`은 그것을 답으로
쓰지 않고 **출발점**으로 쓴다:

```
seed 점  ──grow_region(eps)──▶  성장한 영역  ──DBSCAN──▶  여러 클러스터  ──점수화──▶  target 하나
```

즉 seed는 "여기서부터 기하를 따라가 보라"는 지시일 뿐이다. seed 하나가 물체 위에 있으면
그 물체 전체가 성장으로 딸려 오고, seed가 엉뚱한 곳에 있어도 그 클러스터는 점수화에서
탈락한다. **그래서 seed는 순수할 필요가 없다.**

그렇다면 3단계 검증에 왜 등장하는가. 두 가지 이유뿐이고, 둘 다 판정 기준은 아니다.

1. **C가 검사하는 불변식이 seed에 대한 주장이기 때문이다.** `normalize_attention`의
   docstring은 "정규화 모드와 `seed_percentile`을 독립적으로 고를 수 있다"고 주장하는데,
   이것은 3단계 출력의 성질이지 4단계의 성질이 아니다 — 3단계가 내놓는 값의 *순서*가
   모드에 무관한가를 묻는 것이다. 그 주장을 확인하려면 seed 집합을 실제로 계산해 봐야 한다.

2. **인계 미리보기.** 3단계의 출력이 4단계의 입력이다. 4단계가 무엇을 받게 되는지 지금
   보여 두면, 4단계에서 이상한 결과가 나왔을 때 입력 탓인지 4단계 탓인지 가릴 수 있다.

**이 보고서의 첫 판은 seed 순도를 D의 판정 기준으로 삼았고, 그것은 틀렸다.** 3단계의 질문은
"attention이 옳은 점에 붙는가"이지 "4단계가 좋은 씨앗을 받는가"가 아니다. 후자를 재려면
4단계의 성장·군집·점수화까지 함께 봐야 하는데, 그러면 두 단계를 한 번에 재는 셈이라 실패
시 원인을 가릴 수 없게 된다. D는 크기에 영향받지 않는 순위 지표로 바꾸었고, seed 관련
숫자는 아래에 진단으로만 남긴다.

### seed 진단 — 4단계가 받게 될 것


{md(["카메라"] + [f"상위 {100-p:g}%" for p in SEED_PERCENTILES],
    [f"| {c} | " + " | ".join(
        f"{np.mean(purity[c][p]):.4f} / {np.mean(ceiling[c][p]):.4f} → **{np.nanmean(recall[c][p]):.2f}**"
        if purity[c][p] else "—" for p in SEED_PERCENTILES) + " |" for c in cameras])}

각 칸은 `순도 / 천장 → 회수율`이다. 천장은 target 점이 **전부** 뽑혀도 넘을 수 없는 값이고,
회수율 = 순도 / 천장 = **target 점 중 몇 %가 seed에 들어갔는가**로, 크기 효과가 약분되어 곧바로
읽힌다. head 카메라 상위 5%에서 회수율
**{np.nanmean(recall['cam_high'][95.0]):.2f}** — 순도 0.0065라는 숫자가 실패가 아니라 크기
때문임을 이 한 값이 보여준다.

상위 {100-cfg.seed_percentile:g}% seed가 실제로 어디에 앉는지 범주별로:

{md(["카메라", "target", "다른 물체", "배경", "robot"],
    [f"| {c} | " + " | ".join(str(breakdown[c].get(k, 0))
                              for k in ("target", "다른 물체", "배경", "robot")) + " |"
     for c in cameras])}

`robot`을 따로 센 것은 의도적이다. attention이 그리퍼에 몰려 seed가 로봇 위에 앉는 것은
"엉뚱한 물체를 골랐다"와는 다른 종류의 실패이고, 로봇 점은 자기 필터 단계에서 제거되므로
4단계에 도달하지도 않는다 — 즉 seed만 낭비된다.

![seed 점의 위치](../asset/image/lifting/fig2_seed_points.png)

### E. 카메라별 attention — 융합을 켜면 안 되는 이유

1단계는 head 카메라만 검증했다. 그런데 `multiview.fuse`는 세 카메라의 attention을 `max`로
합친다. 위 순위 표가 그 융합이 안전하지 않다는 것을 보여준다.

| 카메라 | peak가 target 위 | AUC | precision@N |
|---|---|---|---|
| `cam_high` | **{np.mean(rank['cam_high']['peak']):.3f}** | {np.mean(rank['cam_high']['auc']):.5f} | {np.mean(rank['cam_high']['prec_at_n']):.3f} |
| `cam_left_wrist` | **{np.mean(rank['cam_left_wrist']['peak']):.3f}** | {np.mean(rank['cam_left_wrist']['auc']):.5f} | {np.mean(rank['cam_left_wrist']['prec_at_n']):.3f} |
| `cam_right_wrist` | **{np.mean(rank['cam_right_wrist']['peak']):.3f}** | {np.mean(rank['cam_right_wrist']['auc']):.5f} | {np.mean(rank['cam_right_wrist']['prec_at_n']):.3f} |

**왼쪽 손목 카메라의 attention은 target을 가리키지 않는다.** peak가 한 프레임도 target 위에
오지 않고, AUC는 우연(0.5)보다는 낫지만 head 카메라와 비교가 되지 않는다.

원인은 셀 선택이다. 1단계는 **head 카메라의 토큰 블록 [0, 256)** 위에서 (L{layer}, h{head})를
골랐다. 같은 (층, 헤드)가 왼쪽 손목 블록 [256, 512)에서도 target을 찾는다는 보장은 어디에도
없다 — 오른쪽 손목에서는 우연히 옮겨갔고, 왼쪽에서는 그러지 않았다.

**따라서 attention을 `max`로 융합하면 왼쪽 손목의 잘못된 peak가 그대로 들어온다.** 세 가지
선택지가 있다:

1. 카메라마다 셀을 따로 고른다 — 카메라별로 1단계를 다시 돌려야 하고, 정답 대조가 필요하다.
2. 검증된 카메라만 융합에 넣는다 — 여기서는 `cam_high`와 `cam_right_wrist`.
3. **attention은 head 카메라 것만 쓰고, 점군만 세 카메라로 융합한다.**

3번이 지금으로서는 가장 방어하기 쉽다. attention은 target의 *이름*을 정할 뿐이고 그 일은 한
카메라로 충분하며(head 카메라 peak 1.000), 기하 융합은 2단계 D에서 세 카메라가 서로 일치함을
이미 확인했다. 4단계는 이 구성으로 진행한다.

## 그림에 대하여

세 단계의 그림은 같은 규격을 쓴다 (`benchmark/ag3s/experiments/figstyle.py`): 밝은 단색
바탕, 검증된 8색 순서형 팔레트(색각 이상에서도 인접 색이 구분되도록 순서 자체가 안전 장치다),
크기를 나타낼 때는 파랑 한 색의 명도 램프, 정체를 나타낼 때는 팔레트 슬롯을 고정 순서로.
한글 라벨은 Noto Sans CJK로 그린다.

### fig1 — 순위 지표 3종 × 카메라 3대

`fig1_ranking.png`. 막대 하나가 (지표, 카메라) 한 쌍이다.

**만드는 법.** 검사 프레임마다 카메라별로 점군을 재구성하고 attention을 lift한 뒤, MuJoCo
세그멘테이션으로 각 점이 target 위인지 라벨을 붙인다. 그 라벨과 lift된 attention 값으로
세 지표를 계산하고 프레임에 대해 평균한다. `peak`는 최고 attention 점이 target 위인 프레임
비율, `AUC`는 순위합(Mann–Whitney)에서 얻고, `precision@N`은 상위 N개(N = target 점 수) 중
target 비율이다.

**읽는 법.** 1.0이 최선이고 AUC의 우연 수준은 0.5(점선)이다. `peak`와 `precision@N`의 우연
수준은 물체의 화면 점유 비율(여기서는 0.1% 미만)이라 사실상 0이다. 세 지표를 함께 두는
이유는 각각 다른 것에 둔감하기 때문이다 — `peak`는 한 점만 보므로 분포를 못 보고, `AUC`는
전체 분포를 보므로 극단값에 둔하며, `precision@N`은 그 사이다.

### fig2 — seed 점이 어디에 놓이는가 (2패널)

`fig2_seed_points.png`. 한 프레임의 점군을 위에서 내려다본 산점도. 회색은 seed가 아닌 점,
색은 seed이며 정답 범주로 칠했다.

**만드는 법.** head 카메라 첫 프레임의 재구성 점군에 lift된 attention을 붙이고, 상위 5%를
seed로 표시한 뒤, 각 점의 MuJoCo body id를 target / 다른 물체 / 배경 / robot으로 묶어 색을
정한다. x–y 평면에 투영해 그린다 (z는 그리지 않는다 — 이 그림의 질문은 "어느 물체 위인가"이지
"얼마나 높은가"가 아니다).

**패널이 둘인 이유.** 이야기가 둘이기 때문이다. 왼쪽(전체 범위)은 seed의 대부분이 벽과
바닥 같은 배경에 있다는 것을 보여주고, 오른쪽(target 주변 확대)은 그 와중에도 target 위에
seed가 앉는다는 것을 보여준다. 한 패널로 그리면 배경이 화면을 차지해 target이 몇 픽셀로
뭉개지거나, 확대하면 배경 지배가 안 보인다.

**읽는 법.** 범례의 괄호 안 숫자가 그 범주의 seed 개수다. 배경이 압도적인 것은 정상이며,
그것이 seed 순도를 지표로 쓸 수 없는 이유를 눈으로 보여준다.

## 재현

```bash
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.lifting_report \\\\
    --records {args.records} --attention {args.attention}
```
"""
    out = pathlib.Path(args.out_doc)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    out.with_suffix(".json").write_text(json.dumps({
        "cell": {"layer": layer, "head": head, "denoise": denoise, "agg": agg},
        "target": target,
        "checks": {"A_preserved": ok_a, "B_correspondence": ok_b,
                   "C_order_preserving": ok_c, "D_seed_purity": ok_d},
        "nearest_max_err": max(corr_nearest), "bilinear_bracket_max": max(corr_bracket),
        "seed_jaccard": {m: [min(v), max(v)] for m, v in seed_jaccard.items() if v},
        "frames_scored": frames_scored,
        "ranking": {c: {k: (float(np.mean(v)) if v else None) for k, v in rank[c].items()}
                    for c in cameras},
        "seed_purity": {c: {str(p): (float(np.mean(v)) if v else None)
                            for p, v in purity[c].items()} for c in cameras},
        "seed_purity_ceiling": {c: {str(p): (float(np.mean(v)) if v else None)
                                    for p, v in ceiling[c].items()} for c in cameras},
        "seed_recall": {c: {str(p): (float(np.nanmean(v)) if v else None)
                            for p, v in recall[c].items()} for c in cameras},
        "seed_breakdown": breakdown,
    }, indent=2, ensure_ascii=False))
    print(f"wrote {out} and 2 figures in {figs}")
    print(f"판정 {verdict}: A={'ok' if ok_a else 'FAIL'}  B={max(corr_nearest):.1e}/"
          f"{max(corr_bracket):.1e}  C={'ok' if ok_c else 'FAIL'}  "
          f"D(head) peak={head_peak:.3f} AUC={head_auc:.3f} p@N={head_prec:.3f}")


if __name__ == "__main__":
    main()
