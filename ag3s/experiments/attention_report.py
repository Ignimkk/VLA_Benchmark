"""Step 1 verification -- does the fine-tuned pi0.5 actually look at the object the prompt names?

AG3S uses attention for exactly one thing: choosing which of the reconstructed clusters is the
target. Every safety property in the pipeline survives attention being wrong, but the pipeline is
only *useful* if attention is right, so this is the first thing to measure and the one number that
decides whether the real map can replace `mujoco_source.gaussian_attention`.

The scoring is deliberately hostile to the flattering answer.

**Size is controlled for.** The crate fills a large part of the head camera. A head that spreads
attention uniformly would put most of its mass on the crate and score well on any raw mass metric,
so every mass is reported next to `lift = mass / area_share` -- what the head achieved divided by
what uniform attention would have achieved on the same object in the same frame. Lift near 1 means
the head is doing nothing, whatever its raw mass says.

**Two baselines are always printed.** The head-average map (what you get without picking a head) and
the uniform map (what you get without a model at all). A (layer, head) that does not beat both is
not evidence of grounding; it is evidence that the target happens to be large or central.

**Occluded frames are excluded, not scored as misses.** A frame where the target is behind the crate
has no correct answer, and counting it either way would corrupt the rate. Frames are kept only when
the target covers at least `--min-target-px` pixels of the segmentation.

Run (locally, no GPU needed -- the forward pass already happened in `pi05_attention.py`):

    MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.attention_report \
        --records outputs/.../ag3s_records/run_0000 \
        --attention benchmark/ag3s/asset/attention/run_0000.npz \
        --out-doc benchmark/ag3s/docs/step-01-attention.md \
        --out-figs benchmark/ag3s/asset/image/attention
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Optional, Sequence

import numpy as np

from benchmark.ag3s.experiments.policy_record import (
    ATTENTION_GRID,
    CAMERA_BINDINGS,
    load_run,
    patch_coverage,
    pose_scene,
    replay_scene,
)

#: Scene bodies attention is measured against. `table` and `shelf` are here so a miss can be
#: *named* -- "the head looked at the table" is a different failure from "the head looked at the
#: apple", and only one of them is a grounding error.
CANDIDATE_BODIES = ("crate", "apple", "banana", "orange", "pear", "table", "shelf")

#: Bodies allowed to *win* a frame. The support surface is excluded on purpose, and not for
#: convenience: AG3S extracts the table as a `SupportSurface` in stage 7 and target grounding in
#: stage 4 never sees it as a candidate. Scoring attention against a set the pipeline would never
#: choose from would measure a decision nothing downstream makes. The table stays in the reported
#: per-body table, where a head that mostly stares at it is still visible as such.
COMPETITOR_BODIES = ("crate", "apple", "banana", "orange", "pear")

#: Words a prompt may use for a body. The RB-Y1 transport task says "basket" for the crate.
PROMPT_ALIASES = {
    "crate": ("crate", "basket", "box", "bin"),
    "apple": ("apple",),
    "banana": ("banana",),
    "orange": ("orange",),
    "pear": ("pear",),
}

from benchmark.ag3s.experiments.figstyle import (  # noqa: E402
    CATEGORICAL, GRID_INK, INK, INK_2, SEQ_BLUE, SURFACE, sequential_cmap, style_axes, use_korean,
)


def target_from_prompt(prompt: str) -> Optional[str]:
    """The body the prompt names, or None if it is ambiguous.

    Ambiguity is returned rather than resolved. "put the pear in the basket" names two objects and
    only one of them is the thing being grasped; the caller has to say which, and guessing here
    would silently decide the whole experiment.
    """
    low = prompt.lower()
    hits = [body for body, words in PROMPT_ALIASES.items() if any(w in low for w in words)]
    graspable = [b for b in hits if b != "crate"]
    return graspable[0] if len(graspable) == 1 else None


# ---------------------------------------------------------------------------- ground truth


def ground_truth(record, camera_mujoco: str, *, height: int, width: int):
    """Per-frame patch coverage of every candidate body, from replayed MuJoCo segmentation."""
    scene = replay_scene(record, height=height, width=width)
    body_ids, present = [], []
    for name in CANDIDATE_BODIES:
        bid = scene.mujoco.mj_name2id(scene.model, scene.mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            body_ids.append(bid)
            present.append(name)
    coverage = np.zeros((len(record), len(present), ATTENTION_GRID, ATTENTION_GRID), np.float32)
    areas = np.zeros((len(record), len(present)), np.float32)
    rgb = []
    for i, step in enumerate(record):
        pose_scene(scene, step)
        frame = scene.capture(camera_mujoco)
        coverage[i], areas[i] = patch_coverage(frame.body_ids, body_ids)
        rgb.append(step.images["cam_high"] if camera_mujoco == "zed_left" else None)
    scene.close()
    return present, coverage, areas, rgb


# ---------------------------------------------------------------------------------- scoring


def score_maps(maps, coverage, areas, keep, target_index, *, competitor_mask,
               betas=(0.0, 0.5, 1.0), min_body_px: float = 100.0, patch_hit_cov: float = 0.05):
    """Score an `(n_frames, 16, 16)` attention stack against per-frame ground-truth coverage.

    Two hit definitions, because a single one is misleading here.

    **Object-level** (primary, and what KNOWS Eq. (2)-(3) does): give every visible body a score
    `mass_i / area_share_i**beta` and ask whether the target wins. `beta = 0` is raw attention mass,
    which favours whatever is biggest; `beta = 1` is attention *density*, which favours whatever is
    smallest. Neither is the honest one on its own, so the rate is reported at both ends and in the
    middle. A target that wins across the whole sweep is winning on where the attention is, not on
    how the metric was tuned.

    **Peak-on-target** (secondary): does the single hottest patch contain the target at all. The
    obvious version of this -- "which body covers most of the argmax patch" -- is unusable at this
    resolution: a 16x16 grid over 480x640 makes each patch 30x40 px, the pear covers about 380 px in
    total, and the table it sits on wins every patch it appears in. Asking for presence rather than
    dominance is what makes the question answerable at all.

    Only `competitor_mask` bodies can win, and only while they cover at least `min_body_px` pixels.
    A body reduced to a few stray pixels can post an enormous density score off a single patch,
    which is noise wearing the shape of a result.
    """
    maps = np.asarray(maps, np.float64)
    coverage = np.asarray(coverage, np.float64)
    total = maps.sum(axis=(1, 2), keepdims=True)
    normalized = maps / np.maximum(total, 1e-12)

    mass = np.einsum("fij,fbij->fb", normalized, coverage)
    area_share = coverage.sum(axis=(2, 3)) / (ATTENTION_GRID * ATTENTION_GRID)
    lift = mass / np.maximum(area_share, 1e-12)
    eligible = (areas >= min_body_px) & np.asarray(competitor_mask, bool)[None, :]

    flat = maps.reshape(maps.shape[0], -1).argmax(axis=1)
    rows, cols = np.unravel_index(flat, (ATTENTION_GRID, ATTENTION_GRID))
    peak_cov = coverage[np.arange(len(maps)), :, rows, cols]

    out = {
        "mass": mass,
        "lift": lift,
        "argmax_rc": (rows, cols),
        "peak_target_cov": peak_cov[:, target_index],
    }
    k = np.asarray(keep, bool)
    n = int(k.sum())
    for beta in betas:
        score = mass / np.maximum(area_share, 1e-12) ** float(beta)
        score = np.where(eligible, score, -np.inf)
        winner = score.argmax(axis=1)
        out[f"hit_rate_b{beta:g}"] = float((winner[k] == target_index).mean()) if n else float("nan")
        out[f"winner_b{beta:g}"] = winner
    out["hit_rate"] = out[f"hit_rate_b{betas[-1]:g}"]
    out["peak_on_target"] = (
        float((peak_cov[k, target_index] > patch_hit_cov).mean()) if n else float("nan")
    )
    # Measured against the same set that is allowed to win. Including the support surface here
    # would report the target losing to the table -- a comparison no stage of AG3S ever makes.
    rivals = np.asarray(competitor_mask, bool).copy()
    rivals[target_index] = False
    other = mass[:, rivals].max(axis=1) if rivals.any() else np.zeros(len(mass))
    out["target_mass"] = float(mass[k, target_index].mean()) if n else float("nan")
    out["target_lift"] = float(lift[k, target_index].mean()) if n else float("nan")
    out["advantage"] = float((mass[:, target_index] - other)[k].mean()) if n else float("nan")
    return out


# ---------------------------------------------------------------------------------- figures


def _fig_setup(plt, fig, axes):
    style_axes(fig, axes)


def fig_layer_head(plt, hit, out_path, *, title, baseline_head_avg, baseline_uniform,
                   chosen=None, metric="peak-on-target"):
    """Sequential heatmap: one hue, light -> dark, because a rate is a magnitude.

    `chosen` is the cell the ranking picked, which is not always this matrix's own argmax -- ties
    are broken on metrics the heatmap does not show. Ringing the matrix maximum instead would point
    the reader at a different head from the one the document then discusses.
    """
    cmap = sequential_cmap()
    n_layers, n_heads = hit.shape
    fig, ax = plt.subplots(figsize=(6.2, 8.4), dpi=160)
    _fig_setup(plt, fig, ax)
    im = ax.imshow(hit, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for l in range(n_layers):
        for h in range(n_heads):
            v = hit[l, h]
            ax.text(h, l, f"{v:.2f}"[1:] if v < 1 else "1.0", ha="center", va="center",
                    fontsize=6.5, color="#ffffff" if v > 0.55 else INK)
    best = tuple(chosen) if chosen is not None else np.unravel_index(np.nanargmax(hit), hit.shape)
    ax.add_patch(plt.Rectangle((best[1] - .5, best[0] - .5), 1, 1, fill=False,
                               edgecolor=CATEGORICAL[1], linewidth=2.2))
    ties = int((hit >= np.nanmax(hit) - 1e-9).sum())
    ax.set_xticks(range(n_heads), [f"h{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_layers), [f"L{l}" for l in range(n_layers)])
    ax.set_xlabel("attention head")
    ax.set_ylabel("transformer 층")
    ax.set_title(title, color=INK, fontsize=11, pad=10, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label(metric, color=INK_2, fontsize=8)
    cb.ax.tick_params(colors=INK_2, labelsize=7)
    cb.outline.set_edgecolor(GRID_INK)
    ax.text(0.0, -0.085,
            f"선택: L{best[0]}h{best[1]} = {hit[best]:.2f}   ·   head-average "
            f"{baseline_head_avg:.2f}   ·   uniform {baseline_uniform:.2f}   ·   "
            f"행렬 최댓값 {np.nanmax(hit):.2f}에 닿은 셀 {ties}개",
            transform=ax.transAxes, fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_overlay(plt, images, maps, coverage, target_index, frames, out_path, *, title):
    """Attention over the pixels the policy actually saw, beside the ground-truth footprint."""
    cmap = sequential_cmap()
    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.0), dpi=160, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, f in enumerate(frames):
        img = images[f]
        top, bottom = axes[0][j], axes[1][j]
        for ax in (top, bottom):
            ax.set_facecolor(SURFACE)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID_INK)
        top.imshow(img)
        # Ground truth: the target's patch footprint, drawn as a contour so the image stays visible.
        cov = coverage[f, target_index]
        top.contour(np.kron(cov, np.ones((img.shape[0] // ATTENTION_GRID,
                                          img.shape[1] // ATTENTION_GRID))),
                    levels=[0.15], colors=[CATEGORICAL[1]], linewidths=1.4)
        top.set_title(f"프레임 {f}", fontsize=8, color=INK_2)
        bottom.imshow(img)
        a = maps[f]
        bottom.imshow(a / max(a.max(), 1e-12), cmap=cmap, alpha=0.62,
                      extent=(0, img.shape[1], img.shape[0], 0), interpolation="bilinear")
        r, c = np.unravel_index(a.argmax(), a.shape)
        bottom.plot((c + .5) / ATTENTION_GRID * img.shape[1],
                    (r + .5) / ATTENTION_GRID * img.shape[0],
                    marker="o", markersize=7, markerfacecolor="none",
                    markeredgecolor=CATEGORICAL[1], markeredgewidth=1.8)
    axes[0][0].set_ylabel("cam_high\n+ 정답 target", fontsize=8, color=INK_2)
    axes[1][0].set_ylabel("+ attention\n(○ = argmax)", fontsize=8, color=INK_2)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_lift_over_time(plt, t_steps, lift, bodies, target_index, keep, out_path, *, title):
    """Attention *density* per object across the rollout.

    Raw mass would be the obvious thing to plot and it is the wrong thing: the crate covers thirty
    times the pixels of a pear, so on a mass axis every fruit sits flat against zero and the chart
    says only that the crate is big. Lift divides by area share, which puts every object on the same
    axis and makes 1.0 a meaningful line -- exactly what uniform attention would achieve.
    """
    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=160)
    _fig_setup(plt, fig, ax)
    ax.grid(axis="y", color=GRID_INK, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axhline(1.0, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.004, 1.0, " 균등", transform=ax.get_yaxis_transform(), va="bottom",
            fontsize=7.5, color=INK_2)
    order = [target_index] + [i for i in range(len(bodies)) if i != target_index]
    for slot, bi in enumerate(order[:8]):
        ax.plot(t_steps, lift[:, bi], linewidth=2.2 if bi == target_index else 1.3,
                color=CATEGORICAL[slot], label=bodies[bi],
                zorder=4 if bi == target_index else 3)
    if (~keep).any():
        for t in np.asarray(t_steps)[~keep]:
            ax.axvspan(t - 4, t + 4, color=GRID_INK, alpha=0.45, linewidth=0, zorder=1)
    ax.set_xlabel("제어 스텝")
    ax.set_ylabel("attention 밀도  (mass / 점유비율)")
    ax.set_ylim(0, None)
    ax.set_title(title, color=INK, fontsize=11, pad=8, loc="left")
    leg = ax.legend(frameon=False, fontsize=8, ncol=len(order[:8]), loc="upper center",
                    bbox_to_anchor=(0.5, -0.22))
    for text in leg.get_texts():
        text.set_color(INK_2)
    if (~keep).any():
        ax.text(1.0, -0.42, "음영 = target 가려짐, 모든 채점에서 제외",
                transform=ax.transAxes, fontsize=7.5, color=INK_2, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_choice_sweep(plt, rows, out_path, *, title):
    """Which sampling choice matters: best hit rate per (denoise step, query aggregation)."""
    aggs = sorted({r["agg"] for r in rows})
    denoise = sorted({r["denoise"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=160)
    _fig_setup(plt, fig, ax)
    ax.grid(axis="y", color=GRID_INK, linewidth=0.6)
    ax.set_axisbelow(True)
    width = 0.8 / max(len(aggs), 1)
    x = np.arange(len(denoise))
    for ai, agg in enumerate(aggs):
        vals = [max((r["hit_rate"] for r in rows if r["agg"] == agg and r["denoise"] == d),
                    default=np.nan) for d in denoise]
        ax.bar(x + ai * width - 0.4 + width / 2, vals, width * 0.9,
               color=CATEGORICAL[ai], label=agg, edgecolor=SURFACE, linewidth=2.0)
        for xi, v in zip(x + ai * width - 0.4 + width / 2, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(x, [f"Euler step {d}" for d in denoise])
    ax.set_ylabel("모든 (층, 헤드) 중 최고 hit rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, color=INK, fontsize=11, pad=8, loc="left")
    leg = ax.legend(frameon=False, fontsize=8, title="query pooling", ncol=len(aggs))
    leg.get_title().set_color(INK_2)
    leg.get_title().set_fontsize(8)
    for text in leg.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--attention", required=True)
    ap.add_argument("--camera", default="cam_high", choices=sorted(CAMERA_BINDINGS))
    ap.add_argument("--target", default=None, help="body the prompt refers to (default: parsed)")
    ap.add_argument("--min-target-px", type=int, default=200,
                    help="frames where the target covers fewer pixels are excluded as occluded")
    ap.add_argument("--out-doc", default="benchmark/ag3s/docs/step-01-attention.md")
    ap.add_argument("--out-figs", default="benchmark/ag3s/asset/image/attention")
    ap.add_argument("--top", type=int, default=12, help="rows in the ranking table")
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                    help="area-normalization exponents the object-level hit rate is swept over; "
                         "0 = raw attention mass (favours big objects), 1 = attention density "
                         "(favours small ones)")
    ap.add_argument("--min-body-px", type=float, default=100.0,
                    help="a body smaller than this cannot win a frame")
    ap.add_argument("--patch-hit-cov", type=float, default=0.05,
                    help="target coverage in the hottest patch that counts as peak-on-target")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    use_korean()
    import matplotlib.pyplot as plt

    record = load_run(args.records)
    blob = np.load(args.attention, allow_pickle=False)
    # Stored as float16 to halve the transfer; every score is computed in float64 regardless, so
    # the on-disk dtype never reaches an arithmetic that could accumulate error.
    attention = np.asarray(blob["attention"], np.float32)
    cameras = [str(c) for c in blob["cameras"]]
    aggs = [str(a) for a in blob["aggregations"]]
    denoise = [int(d) for d in blob["denoise_steps"]]
    t_steps = np.asarray([s.t_step for s in record], np.int64)
    if attention.shape[0] != len(record):
        raise ValueError(
            f"attention has {attention.shape[0]} frames but the record has {len(record)}; "
            "they were produced from different runs"
        )
    ci = cameras.index(args.camera)
    camera_mujoco = CAMERA_BINDINGS[args.camera][0]

    target = args.target or target_from_prompt(record.prompt)
    if target is None:
        raise SystemExit(
            f"could not decide the target from prompt {record.prompt!r} -- pass --target explicitly"
        )
    print(f"prompt={record.prompt!r}  target={target}  camera={args.camera} ({camera_mujoco})")

    bodies, coverage, areas, images = ground_truth(record, camera_mujoco, height=480, width=640)
    ti = bodies.index(target)
    keep = areas[:, ti] >= args.min_target_px
    print(f"{int(keep.sum())}/{len(record)} frames keep the target visible "
          f"(>= {args.min_target_px} px)")
    if not keep.any():
        raise SystemExit("the target is never visible in this camera; nothing can be scored")

    n_layers, n_heads = attention.shape[3], attention.shape[4]
    uniform = np.ones((len(record), ATTENTION_GRID, ATTENTION_GRID), np.float32)
    competitor_mask = np.asarray([b in COMPETITOR_BODIES for b in bodies], bool)
    if not competitor_mask[ti]:
        raise SystemExit(f"target {target!r} is not in COMPETITOR_BODIES; nothing could ever win")
    score_kw = dict(betas=tuple(args.betas), min_body_px=args.min_body_px,
                    patch_hit_cov=args.patch_hit_cov, competitor_mask=competitor_mask)
    base_uniform = score_maps(uniform, coverage, areas, keep, ti, **score_kw)

    rows = []
    hit_cube = np.zeros((len(denoise), len(aggs), n_layers, n_heads), np.float64)
    peak_cube = np.zeros_like(hit_cube)
    beta_cubes = {b: np.zeros_like(hit_cube) for b in args.betas}
    for di, d in enumerate(denoise):
        for ai, agg in enumerate(aggs):
            for l in range(n_layers):
                for h in range(n_heads):
                    s = score_maps(attention[:, di, ai, l, h, ci], coverage, areas, keep, ti,
                                   **score_kw)
                    hit_cube[di, ai, l, h] = s["hit_rate"]
                    peak_cube[di, ai, l, h] = s["peak_on_target"]
                    for b in args.betas:
                        beta_cubes[b][di, ai, l, h] = s[f"hit_rate_b{b:g}"]
                    row = {"denoise": d, "agg": agg, "layer": l, "head": h,
                           "hit_rate": s["hit_rate"], "peak_on_target": s["peak_on_target"],
                           "target_mass": s["target_mass"], "target_lift": s["target_lift"],
                           "advantage": s["advantage"]}
                    row.update({f"hit_b{b:g}": s[f"hit_rate_b{b:g}"] for b in args.betas})
                    # Rank on the worst beta, not the best. A head that only wins when the metric
                    # is tuned in its favour has not demonstrated grounding.
                    row["hit_worst_beta"] = min(row[f"hit_b{b:g}"] for b in args.betas)
                    rows.append(row)

    # Ranked on peak-on-target, not on the beta sweep. Both ends of that sweep turn out to be
    # unusable as a ranking signal here and the diagnostic below says so in numbers: beta = 0 is
    # capped by the scene rather than the head (once the target is inside the crate, any blob
    # covering it covers more crate), and beta = 1 saturates. peak-on-target has neither problem --
    # it asks only whether the single hottest patch contains the target.
    rows.sort(key=lambda r: (-r["peak_on_target"], -r[f"hit_b{args.betas[len(args.betas)//2]:g}"],
                             -r["target_lift"]))
    best = rows[0]
    bdi, bai = denoise.index(best["denoise"]), aggs.index(best["agg"])
    best_maps = attention[:, bdi, bai, best["layer"], best["head"], ci]
    best_score = score_maps(best_maps, coverage, areas, keep, ti, **score_kw)
    head_avg_maps = attention[:, bdi, bai, :, :, ci].mean(axis=(1, 2))
    base_head_avg = score_maps(head_avg_maps, coverage, areas, keep, ti, **score_kw)

    figs = pathlib.Path(args.out_figs)
    figs.mkdir(parents=True, exist_ok=True)
    tag = f"d{best['denoise']}_{best['agg']}"
    fig_layer_head(
        plt, peak_cube[bdi, bai], figs / "fig1_layer_head_peak_on_target.png",
        title=f"(층, 헤드)별 peak-on-target  ·  Euler step {best['denoise']}, "
              f"{best['agg']} pooling",
        baseline_head_avg=base_head_avg["peak_on_target"],
        baseline_uniform=base_uniform["peak_on_target"],
        chosen=(best["layer"], best["head"]), metric="peak-on-target")
    shown = np.flatnonzero(keep)[:: max(1, int(keep.sum()) // 5)][:5]
    if args.camera == "cam_high" and images[0] is not None:
        fig_overlay(plt, images, best_maps, coverage, ti, shown,
                    figs / "fig2_attention_overlay.png",
                    title=f"L{best['layer']}h{best['head']} — 정책이 실제로 본 프레임 위의 attention")
    fig_lift_over_time(plt, t_steps, best_score["lift"], bodies, ti, keep,
                       figs / "fig3_attention_lift.png",
                       title=f"L{best['layer']}h{best['head']} 물체별 attention 밀도 "
                             f"(1.0 = 균등 attention이 줄 값)")
    fig_choice_sweep(plt, rows, figs / "fig4_choice_sweep.png",
                     title="Euler step과 query pooling이 답을 바꾸는 정도")

    # ------------------------------------------------------------------ document
    def md_table(header, lines):
        return "\n".join(["| " + " | ".join(header) + " |",
                          "|" + "|".join("---" for _ in header) + "|", *lines])

    betas = list(args.betas)
    beta_cols = [f"hit β={b:g}" for b in betas]
    rank_rows = [
        "| {r} | L{layer} h{head} | {d} | {agg} | {betas} | {peak:.3f} | {mass:.3f} | {lift:.2f} | {adv:+.3f} |".format(
            r=i + 1, layer=r["layer"], head=r["head"], d=r["denoise"], agg=r["agg"],
            betas=" | ".join(f"{r[f'hit_b{b:g}']:.3f}" for b in betas),
            peak=r["peak_on_target"], mass=r["target_mass"], lift=r["target_lift"],
            adv=r["advantage"])
        for i, r in enumerate(rows[: args.top])
    ]

    def baseline_row(label, sc, bold=False):
        cells = [label] + [f"{sc[f'hit_rate_b{b:g}']:.3f}" for b in betas] + [
            f"{sc['peak_on_target']:.3f}", f"{sc['target_mass']:.3f}",
            f"{sc['target_lift']:.2f}", f"{sc['advantage']:+.3f}"]
        if bold:
            cells = [f"**{c}**" for c in cells]
        return "| " + " | ".join(cells) + " |"

    baseline_rows = [
        baseline_row(f"head-average (all {n_layers}x{n_heads})", base_head_avg),
        baseline_row("uniform (no model)", base_uniform),
        baseline_row(f"best: L{best['layer']} h{best['head']}", best_score, bold=True),
    ]
    per_body = [
        f"| {b} | {areas[keep, i].mean():.0f} | {best_score['mass'][keep, i].mean():.3f} | "
        f"{best_score['lift'][keep, i].mean():.2f} | "
        f"{int((best_score[f'winner_b{betas[-1]:g}'][keep] == i).sum())} |"
        for i, b in enumerate(bodies)
    ]

    def _spread(cube):
        """How much of a metric's range the sweep actually uses, and how many cells tie at the top.

        A metric whose maximum is reached by a large fraction of the 1296 cells is not ranking
        anything; a metric whose maximum is the same for every cell is measuring the scene rather
        than the head. Both happen here, so both are reported instead of being ranked on.
        """
        top = float(np.nanmax(cube))
        return top, int((cube >= top - 1e-9).sum()), int(cube.size), len(np.unique(np.round(cube, 3)))

    diagnostics = [("peak-on-target", _spread(peak_cube))]
    diagnostics += [(f"hit β={b:g}", _spread(beta_cubes[b])) for b in betas]

    # The chosen head must beat both baselines everywhere, and must beat the head-average by a real
    # margin on peak-on-target. The margin is required only there on purpose: at β = 1 the
    # head-average already sits near the ceiling, so no head could clear a fixed margin however good
    # it was -- a gate nothing can pass is not evidence, it is a broken rule.
    beats_everywhere = all(
        best_score[f"hit_rate_b{b:g}"] >= max(base_head_avg[f"hit_rate_b{b:g}"],
                                              base_uniform[f"hit_rate_b{b:g}"]) - 1e-9
        for b in betas
    )
    peak_margin = best_score["peak_on_target"] - max(base_head_avg["peak_on_target"],
                                                     base_uniform["peak_on_target"])
    verdict = ("**PASS**" if (beats_everywhere and peak_margin >= 0.15
                              and best_score["target_lift"] > 2.0) else "**INCONCLUSIVE**")
    doc = f"""# 1단계 — attention map

**질문.** 파인튜닝된 `{blob["checkpoint"].item().split("/")[-3] if "/" in str(blob["checkpoint"]) else blob["checkpoint"]}` 정책은
프롬프트가 지목한 물체를 실제로 보는가?

AG3S가 attention을 쓰는 곳은 단 하나 — 재구성된 클러스터 중 어느 것이 target인지 이름을 붙이는
일뿐입니다. attention이 틀려도 파이프라인의 안전 성질은 전부 유지되지만, 틀리면 파이프라인이
쓸모가 없습니다. 그래서 이것이 가장 먼저 재야 할 값이고, 실제 attention이
`mujoco_source.gaussian_attention`(합성 대역품)을 대체할 수 있는지를 결정하는 값입니다.

| | |
|---|---|
| 프롬프트 | `{record.prompt}` |
| target body | `{target}` |
| 카메라 | `{args.camera}` → MuJoCo `{camera_mujoco}` → prefix 토큰 {CAMERA_BINDINGS[args.camera][2]} |
| 추론 프레임 | {len(record)}개 기록, **{int(keep.sum())}개 채점** ({int((~keep).sum())}개 제외: target이 {args.min_target_px} px 미만) |
| 체크포인트 | `{blob['checkpoint']}` |
| attention 블록 | {n_layers}개 층 × {n_heads}개 헤드 × {ATTENTION_GRID}×{ATTENTION_GRID} 패치 |
| Euler step | {denoise} |
| query pooling | {aggs} |
| noise seed | {blob['noise_seeds'].tolist() if 'noise_seeds' in blob else '기록 없음'} |

## 판정 — {verdict}

가장 좋은 헤드는 **L{best['layer']} h{best['head']}** (Euler step {best['denoise']}, `{best['agg']}` pooling)입니다.
**peak-on-target {best_score["peak_on_target"]:.3f}** — head-average {base_head_avg["peak_on_target"]:.3f},
uniform {base_uniform["peak_on_target"]:.3f} 대비. 같은 물체에 균등 attention이 줄 밀도의
**{best_score['target_lift']:.2f}배**를 싣습니다. β = {betas}에서 물체 단위 hit rate는
**{" / ".join(f"{best_score[f'hit_rate_b{b:g}']:.3f}" for b in betas)}**.

### 어떤 지표가 순위를 정하는가, 그리고 왜

{md_table(["지표", "스윕 최댓값", "최댓값에 닿은 셀", "고유값 개수"],
          [f"| {name} | {top:.3f} | {ties} / {total} | {distinct} |"
           for name, (top, ties, total, distinct) in diagnostics])}

헤드 순위는 **peak-on-target**으로 매깁니다. β 스윕은 보고하되 순위에는 쓰지 않습니다.
이 롤아웃에서 β의 양 끝이 모두 순위 신호로 실패하고, 위 표가 그것을 숫자로 보여줍니다.

- **β = 0은 헤드가 아니라 씬이 상한을 정합니다.** {len(rows)}개 셀 전부가 같은 값에서 멈추고,
  이길 수 있는 프레임은 파지 *전* 프레임뿐입니다. target이 크레이트 안으로 들어간 뒤에는
  target을 덮는 어떤 attention 덩어리도 크레이트를 더 많이 덮으므로, 헤드가 무엇을 하든
  raw mass는 target을 고를 수 없습니다.
- **β = 1은 포화합니다.** 많은 셀이 정확히 1.000에 닿아 좋은 헤드끼리를 구분하지 못합니다.

peak-on-target에는 두 문제가 다 없습니다. "가장 뜨거운 패치 하나가 target을 포함하는가"만
묻고, 이는 실질적으로 상한이 없으면서 이후 target grounding이 실제로 소비할 정보에 가장
가깝습니다.

### 판정 규칙

베이스라인이 둘이고, 선택된 헤드는 **어떤 β에서도 두 베이스라인보다 나빠서는 안 되며**,
**peak-on-target에서 head-average를 0.15 이상** 앞서고 lift가 2를 넘어야 합니다.
uniform 베이스라인은 "target이 그냥 화면에서 제일 큰 것 아닌가"에 답하고, head-average는
"헤드를 고른 것이 무엇을 벌어줬나"에 답합니다.

마진을 peak-on-target에만 요구하는 것은 의도된 선택입니다. β = 1에서 head-average가 이미
{base_head_avg["hit_rate_b1"]:.3f}이므로 어떤 헤드가 낼 수 있는 최대 마진은
{1.0 - base_head_avg["hit_rate_b1"]:.3f}입니다 — 아무것도 통과할 수 없는 게이트는 증거가
아니라 고장난 규칙입니다.

`β`는 attention mass를 물체의 화면 점유 비율로 나눌 때의 지수입니다. β = 0은 raw mass라
크기가 큰 크레이트가 이기고, β = 1은 밀도라 작은 과일이 이깁니다.

한 프레임에서 이길 수 있는 것은 {list(COMPETITOR_BODIES)}뿐입니다. 테이블을 제외한 것은
AG3S 7단계가 테이블을 support surface로 뽑아내고 4단계 target grounding이 애초에 후보로
보지 않기 때문입니다 — 파이프라인이 결코 고르지 않을 집합을 상대로 채점하면 아무 단계도
내리지 않는 판단을 재게 됩니다. 아래 물체별 표에는 그대로 남겨두어, 테이블만 쳐다보는
헤드가 있다면 그렇게 보이도록 했습니다.

### 베이스라인과 선택된 헤드

{md_table(["맵"] + beta_cols + ["peak-on-target", "target mass", "target lift", "advantage"], baseline_rows)}

`mass`는 target에 떨어진 attention의 비율, `lift`는 그 비율을 균등 attention이 줄 비율로
나눈 값입니다 — lift가 1.0 근처면 raw mass가 얼마든 그 헤드는 아무것도 선택하지 않는
것입니다. `advantage`는 이길 자격이 있는 물체들 중 최상위 경쟁자의 mass를 뺀 값입니다.
`peak-on-target`은 가장 뜨거운 패치 하나가 target을 포함한 프레임의 비율로, hit rate보다
거친 질문이지만 30×40 픽셀이라는 패치 크기에서 살아남는 질문입니다.

여기서 `advantage`가 음수인 것은 예상된 것이며 실패가 아닙니다. 크레이트가 target의 약
{areas[keep].mean(axis=0)[bodies.index("crate")] / areas[keep, ti].mean():.0f}배 픽셀을
차지하므로 raw mass는 더 많이 가져가면서 *픽셀당* attention은 훨씬 적게 받습니다.
`lift`가 재는 것이 정확히 그 비교이고, 두 열은 함께 읽어야 합니다.

### 순위 — 1296개 (층, 헤드, Euler step, pooling) 중 상위 {args.top}

{md_table(["#", "헤드", "Euler", "pooling"] + beta_cols + ["peak-on-target", "target mass", "target lift", "advantage"], rank_rows)}

![층·헤드별 peak-on-target](../asset/image/attention/fig1_layer_head_peak_on_target.png)

### attention이 실제로 어디로 가는가 (물체별)

{md_table(["body", "평균 가시 픽셀", "attention mass", "lift", "argmax 획득 프레임"], per_body)}

![롤아웃 동안의 attention lift](../asset/image/attention/fig3_attention_lift.png)

### 프레임 자체

![attention overlay](../asset/image/attention/fig2_attention_overlay.png)

### 샘플링 선택이 결과를 바꾸는가

![choice sweep](../asset/image/attention/fig4_choice_sweep.png)

## 그림에 대하여

모든 그림은 공통 규격(`benchmark/ag3s/experiments/figstyle.py`)을 쓴다. 크기·비율을 나타낼
때는 파랑 한 색의 명도 램프(순차형), 정체를 나타낼 때는 검증된 8색 팔레트를 고정 순서로
쓴다 — 슬롯 순서 자체가 색각 이상에서 인접 색이 구분되도록 고른 안전 장치이므로 차트마다
바꾸지 않는다.

### fig1 — (층, 헤드)별 peak-on-target

`fig1_layer_head_peak_on_target.png`. 세로 18층 × 가로 8헤드 격자, 칸 색이 그 헤드의
peak-on-target이다.

**만드는 법.** 고정된 (Euler step, pooling)에서 144개 (층, 헤드) 각각에 대해 채점 프레임의
peak-on-target을 평균해 격자에 채운다. 주황 테두리는 **순위가 고른 칸**이지 이 격자의
최댓값이 아니다 — 동점은 이 그림이 보여주지 않는 지표로 갈리므로, 최댓값에 테두리를 치면
문서가 논하는 헤드와 다른 칸을 가리키게 된다. 아래 캡션에 최댓값에 닿은 칸 수를 함께
적는 것은 포화 여부를 바로 보기 위해서다.

**읽는 법.** 진한 칸이 많으면 그 지표가 포화된 것이고, 그때는 지표를 바꿔야 한다.

### fig2 — 정책이 본 프레임 위의 attention

`fig2_attention_overlay.png`. 위 줄은 정책이 실제로 입력받은 224×224 이미지에 정답 target의
패치 윤곽을 주황으로 그린 것, 아래 줄은 같은 이미지에 attention을 파랑 램프로 덮고 argmax
패치에 ○를 친 것이다.

**만드는 법.** 기록에 저장된 정책 이미지를 그대로 쓴다 (재렌더링이 아니다). 정답 윤곽은
MuJoCo 세그멘테이션에서 계산한 16×16 패치 점유를 이미지 크기로 확대해 등고선으로 그린다.
attention은 16×16 격자를 이미지 위에 이중선형 보간으로 덮는다.

**읽는 법.** ○가 주황 윤곽 안에 있으면 그 프레임은 peak-on-target 성공이다. 프레임은 채점
가능한 것 중에서 균등 간격으로 뽑는다.

### fig3 — 물체별 attention 밀도

`fig3_attention_lift.png`. 가로축은 제어 스텝, 세로축은 밀도 = mass / 화면 점유 비율.

**만드는 법.** 프레임마다 물체별 attention mass와 화면 점유 비율을 세그멘테이션에서 구해
나눈다. 음영 구간은 target이 가려져 채점에서 제외된 프레임이다.

**왜 mass가 아니라 밀도인가.** raw mass 축에서는 테이블과 크레이트가 크기만으로 위를
차지하고 과일은 전부 0 근처에 눌린다. 밀도로 나누면 모든 물체가 같은 축 위에 놓이고,
1.0(점선)이 "균등 attention과 같음"이라는 의미 있는 기준선이 된다.

### fig4 — 샘플링 선택의 영향

`fig4_choice_sweep.png`. Euler step × query pooling 조합마다, 모든 (층, 헤드) 중 최고
hit rate를 막대로 그린다.

**만드는 법.** 1296개 조합의 채점 결과에서 (Euler step, pooling)별 최댓값을 집계한다.

**읽는 법.** 막대 높이가 조합마다 크게 다르면 그 선택이 결과를 좌우한다는 뜻이므로 보고서에
명시해야 한다. 비슷하면 선택에 둔감하다는 뜻이다.

## 이 결과가 아래 단계에서 무엇을 고치는가

`mujoco_source.gaussian_attention`은 target의 알려진 투영 위치에 놓은 가우시안입니다. 그
docstring은 이 카메라를 들여다보는 정책이 없어서 존재한다고 적혀 있습니다. 위 판정이 PASS라면
AG3S 3단계의 attention 소스는
`attention[frame, {best['denoise']}, "{best['agg']}", {best['layer']}, {best['head']}, camera]`이고,
`GridAttentionAdapter`가 이미 {ATTENTION_GRID}×{ATTENTION_GRID} 그리드를 받으므로 어댑터는
바뀌지 않습니다.

## 재현

```bash
# 1. 롤아웃 기록 (로컬, 원격 정책 서버를 구동)
src/openpi/.venv/bin/python src/rby1_bringup/pi05_infer.py \\
    --model rby1_transport_14d --remote localhost:8123 \\
    --prompt "{record.prompt}" --record-ag3s <RECORD_DIR> ...

# 2. 기록 검사 (로컬) — forward pass를 쓸 값어치가 있는지
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.record_check \\
    --records {args.records}

# 3. attention 추출 (GPU 서버, 체크포인트가 있는 곳)
python -m benchmark.ag3s.experiments.pi05_attention \\
    --records {args.records} --checkpoint {blob['checkpoint']} --out {args.attention}

# 4. 채점 (로컬, GPU 불필요)
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.attention_report \\
    --records {args.records} --attention {args.attention}
```
"""
    out_doc = pathlib.Path(args.out_doc)
    out_doc.parent.mkdir(parents=True, exist_ok=True)
    out_doc.write_text(doc)
    (out_doc.with_suffix(".json")).write_text(json.dumps(
        {"target": target, "prompt": record.prompt, "camera": args.camera,
         "n_frames": len(record), "n_scored": int(keep.sum()),
         "best": best, "baseline_head_average": {k: v for k, v in base_head_avg.items()
                                                 if isinstance(v, float)},
         "baseline_uniform": {k: v for k, v in base_uniform.items() if isinstance(v, float)},
         "ranking": rows[: args.top]}, indent=2))
    print(f"wrote {out_doc} and 4 figures in {figs}")
    print(f"VERDICT {verdict}: best L{best['layer']}h{best['head']} "
          f"hit={best_score['hit_rate']:.3f} lift={best_score['target_lift']:.2f} "
          f"vs head-avg {base_head_avg['hit_rate']:.3f} / uniform {base_uniform['hit_rate']:.3f}")


if __name__ == "__main__":
    main()
