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

# --- palette (dataviz reference instance, light surface) ---------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID_INK = "#d8d7d2"
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


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
    other = np.delete(mass, target_index, axis=1)
    out["target_mass"] = float(mass[k, target_index].mean()) if n else float("nan")
    out["target_lift"] = float(lift[k, target_index].mean()) if n else float("nan")
    out["advantage"] = float((mass[:, target_index] - other.max(axis=1))[k].mean()) if n else float("nan")
    return out


# ---------------------------------------------------------------------------------- figures


def _fig_setup(plt, fig, axes):
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID_INK)
        ax.tick_params(colors=INK_2, labelsize=8, length=3, color=GRID_INK)
        ax.xaxis.label.set_color(INK_2)
        ax.yaxis.label.set_color(INK_2)


def fig_layer_head(plt, hit, out_path, *, title, baseline_head_avg, baseline_uniform):
    """Sequential heatmap: one hue, light -> dark, because hit rate is a magnitude."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    n_layers, n_heads = hit.shape
    fig, ax = plt.subplots(figsize=(6.2, 8.4), dpi=160)
    _fig_setup(plt, fig, ax)
    im = ax.imshow(hit, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for l in range(n_layers):
        for h in range(n_heads):
            v = hit[l, h]
            ax.text(h, l, f"{v:.2f}"[1:] if v < 1 else "1.0", ha="center", va="center",
                    fontsize=6.5, color="#ffffff" if v > 0.55 else INK)
    best = np.unravel_index(np.nanargmax(hit), hit.shape)
    ax.add_patch(plt.Rectangle((best[1] - .5, best[0] - .5), 1, 1, fill=False,
                               edgecolor=CATEGORICAL[1], linewidth=2.2))
    ax.set_xticks(range(n_heads), [f"h{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_layers), [f"L{l}" for l in range(n_layers)])
    ax.set_xlabel("attention head")
    ax.set_ylabel("transformer layer")
    ax.set_title(title, color=INK, fontsize=11, pad=10, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("hit rate", color=INK_2, fontsize=8)
    cb.ax.tick_params(colors=INK_2, labelsize=7)
    cb.outline.set_edgecolor(GRID_INK)
    ax.text(0.0, -0.085,
            f"best L{best[0]}h{best[1]} = {hit[best]:.2f}   ·   head-average baseline "
            f"{baseline_head_avg:.2f}   ·   uniform baseline {baseline_uniform:.2f}",
            transform=ax.transAxes, fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_overlay(plt, images, maps, coverage, target_index, frames, out_path, *, title):
    """Attention over the pixels the policy actually saw, beside the ground-truth footprint."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
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
        top.set_title(f"frame {f}", fontsize=8, color=INK_2)
        bottom.imshow(img)
        a = maps[f]
        bottom.imshow(a / max(a.max(), 1e-12), cmap=cmap, alpha=0.62,
                      extent=(0, img.shape[1], img.shape[0], 0), interpolation="bilinear")
        r, c = np.unravel_index(a.argmax(), a.shape)
        bottom.plot((c + .5) / ATTENTION_GRID * img.shape[1],
                    (r + .5) / ATTENTION_GRID * img.shape[0],
                    marker="o", markersize=7, markerfacecolor="none",
                    markeredgecolor=CATEGORICAL[1], markeredgewidth=1.8)
    axes[0][0].set_ylabel("cam_high\n+ GT target", fontsize=8, color=INK_2)
    axes[1][0].set_ylabel("+ attention\n(o = argmax)", fontsize=8, color=INK_2)
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
    ax.text(0.004, 1.0, " uniform", transform=ax.get_yaxis_transform(), va="bottom",
            fontsize=7.5, color=INK_2)
    order = [target_index] + [i for i in range(len(bodies)) if i != target_index]
    for slot, bi in enumerate(order[:8]):
        ax.plot(t_steps, lift[:, bi], linewidth=2.2 if bi == target_index else 1.3,
                color=CATEGORICAL[slot], label=bodies[bi],
                zorder=4 if bi == target_index else 3)
    if (~keep).any():
        for t in np.asarray(t_steps)[~keep]:
            ax.axvspan(t - 4, t + 4, color=GRID_INK, alpha=0.45, linewidth=0, zorder=1)
    ax.set_xlabel("control step")
    ax.set_ylabel("attention lift  (mass / area share)")
    ax.set_ylim(0, None)
    ax.set_title(title, color=INK, fontsize=11, pad=8, loc="left")
    leg = ax.legend(frameon=False, fontsize=8, ncol=len(order[:8]), loc="upper center",
                    bbox_to_anchor=(0.5, -0.22))
    for text in leg.get_texts():
        text.set_color(INK_2)
    if (~keep).any():
        ax.text(1.0, -0.42, "shaded = target occluded, excluded from every score",
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
    ax.set_ylabel("best hit rate over all (layer, head)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, color=INK, fontsize=11, pad=8, loc="left")
    leg = ax.legend(frameon=False, fontsize=8, title="query aggregation", ncol=len(aggs))
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
    for di, d in enumerate(denoise):
        for ai, agg in enumerate(aggs):
            for l in range(n_layers):
                for h in range(n_heads):
                    s = score_maps(attention[:, di, ai, l, h, ci], coverage, areas, keep, ti,
                                   **score_kw)
                    hit_cube[di, ai, l, h] = s["hit_rate"]
                    row = {"denoise": d, "agg": agg, "layer": l, "head": h,
                           "hit_rate": s["hit_rate"], "peak_on_target": s["peak_on_target"],
                           "target_mass": s["target_mass"], "target_lift": s["target_lift"],
                           "advantage": s["advantage"]}
                    row.update({f"hit_b{b:g}": s[f"hit_rate_b{b:g}"] for b in args.betas})
                    # Rank on the worst beta, not the best. A head that only wins when the metric
                    # is tuned in its favour has not demonstrated grounding.
                    row["hit_worst_beta"] = min(row[f"hit_b{b:g}"] for b in args.betas)
                    rows.append(row)

    rows.sort(key=lambda r: (-r["hit_worst_beta"], -r["peak_on_target"], -r["target_lift"]))
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
        plt, hit_cube[bdi, bai], figs / "fig1_layer_head_hit_rate.png",
        title=f"Hit rate by (layer, head)  ·  Euler step {best['denoise']}, {best['agg']} query pooling",
        baseline_head_avg=base_head_avg["hit_rate"], baseline_uniform=base_uniform["hit_rate"])
    shown = np.flatnonzero(keep)[:: max(1, int(keep.sum()) // 5)][:5]
    if args.camera == "cam_high" and images[0] is not None:
        fig_overlay(plt, images, best_maps, coverage, ti, shown,
                    figs / "fig2_attention_overlay.png",
                    title=f"L{best['layer']}h{best['head']} attention on the frames the policy saw")
    fig_lift_over_time(plt, t_steps, best_score["lift"], bodies, ti, keep,
                       figs / "fig3_attention_lift.png",
                       title=f"L{best['layer']}h{best['head']} attention density per object "
                             f"(1.0 = what uniform attention would give)")
    fig_choice_sweep(plt, rows, figs / "fig4_choice_sweep.png",
                     title="Denoising step and query pooling both move the answer")

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

    beat_baselines = all(
        best_score[f"hit_rate_b{b:g}"] > max(base_head_avg[f"hit_rate_b{b:g}"],
                                             base_uniform[f"hit_rate_b{b:g}"]) + 0.15
        for b in betas
    )
    verdict = "**PASS**" if (beat_baselines and best_score["target_lift"] > 1.5) else "**INCONCLUSIVE**"
    doc = f"""# Step 1 — attention map

**Question.** Does the fine-tuned `rby1_transport_14d` policy attend to the object its prompt names?
AG3S uses attention for one thing only — naming which reconstructed cluster is the target — so if
the answer is no, every later step is measuring a stand-in.

| | |
|---|---|
| prompt | `{record.prompt}` |
| target body | `{target}` |
| camera | `{args.camera}` → MuJoCo `{camera_mujoco}` → prefix tokens {CAMERA_BINDINGS[args.camera][2]} |
| inference frames | {len(record)} recorded, **{int(keep.sum())} scored** ({int((~keep).sum())} excluded: target under {args.min_target_px} px) |
| checkpoint | `{blob['checkpoint']}` |
| attention block | {n_layers} layers x {n_heads} heads x {ATTENTION_GRID}x{ATTENTION_GRID} patches |
| Euler steps probed | {denoise} |
| query pooling | {aggs} |

## Verdict — {verdict}

The best head is **L{best['layer']} h{best['head']}** at Euler step {best['denoise']} with
`{best['agg']}` query pooling: object-level hit rate **{" / ".join(f"{best_score[f'hit_rate_b{b:g}']:.3f}" for b in betas)}**
across β = {betas}, peak-on-target **{best_score["peak_on_target"]:.3f}**, lift
**{best_score['target_lift']:.2f}x** over what uniform attention would score on the same object.

Both baselines are in the table below and both must be beaten **at every β** for this to mean
anything. The uniform baseline answers "is the target simply the biggest thing in view"; the
head-average baseline answers "did picking a head buy us anything". Heads are ranked by their
*worst* β, so a head that only wins when the metric leans its way does not reach the top.

`β` is the exponent that divides attention mass by the object's on-screen area share. β = 0 is raw
mass, which the crate wins by being large; β = 1 is attention density, which small fruit win by
being small. The honest reading is the column where the target has the least help.

Only {list(COMPETITOR_BODIES)} can win a frame. The table is excluded because AG3S extracts it as a
support surface in stage 7 and target grounding never considers it — scoring against a set the
pipeline would never choose from would measure a decision nothing downstream makes. It is still in
the per-object table below, so a head that mostly stares at the table is visible as one.

### Baselines and the chosen head

{md_table(["map"] + beta_cols + ["peak-on-target", "target mass", "target lift", "advantage"], baseline_rows)}

`mass` is the share of attention landing on the target; `lift` is that share divided by what
uniform attention would give it, so lift near 1.0 means the head is not selecting at all.
`advantage` is the target's mass minus the best distractor's. `peak-on-target` is the fraction of
frames whose single hottest patch contains the target — a coarser question than the hit rate, and
the one that survives the 30x40-pixel patch size.

### Ranking — top {args.top} of {len(rows)} (layer, head, Euler step, pooling)

{md_table(["#", "head", "Euler", "pooling"] + beta_cols + ["peak-on-target", "target mass", "target lift", "advantage"], rank_rows)}

![hit rate by layer and head](../asset/image/attention/fig1_layer_head_hit_rate.png)

### Where the attention actually goes, per object

{md_table(["body", "mean visible px", "attention mass", "lift", "argmax frames won"], per_body)}

![attention lift over the rollout](../asset/image/attention/fig3_attention_lift.png)

### The frames themselves

![attention overlay](../asset/image/attention/fig2_attention_overlay.png)

### Does the sampling choice matter?

![choice sweep](../asset/image/attention/fig4_choice_sweep.png)

## What this fixes downstream

`mujoco_source.gaussian_attention` is a Gaussian blob placed on the target's known projection. Its
docstring says it exists only because no policy looked through these cameras. If the verdict above
is PASS, the replacement for AG3S stage 3 is
`attention[frame, {best['denoise']}, "{best['agg']}", {best['layer']}, {best['head']}, camera]`,
resampled to the frame through `GridAttentionAdapter` — the adapter already expects a
{ATTENTION_GRID}x{ATTENTION_GRID} grid, so nothing else changes.

## Reproduce

```bash
# 1. record the rollout (local, drives the remote policy server)
src/openpi/.venv/bin/python src/rby1_bringup/pi05_infer.py \\
    --model rby1_transport_14d --remote localhost:8123 \\
    --prompt "{record.prompt}" --record-ag3s <RECORD_DIR> ...

# 2. extract attention (GPU server, where the checkpoint is)
src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.pi05_attention \\
    --records {args.records} --checkpoint {blob['checkpoint']} --out {args.attention}

# 3. score it (local, no GPU)
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
