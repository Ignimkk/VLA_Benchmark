"""Is this recording worth spending a GPU forward pass on?

Every later verification step consumes one recorded rollout, and the expensive part -- the attention
extraction -- happens on a different machine. A record that turns out to be unusable is discovered
there, after the transfer and the model load, unless it is checked here first.

Three things can make a record useless, and none of them are visible from the file listing:

**The target is never in frame.** Attention grounding cannot be scored on frames where the correct
answer does not exist. The rollout may be perfectly good and still leave too few scoreable frames.

**The images are degenerate.** A renderer that failed silently produces constant or near-constant
frames; the file sizes look normal because the shapes are right.

**The scene does not replay.** `qpos` is a flat vector whose meaning is positional, so a record made
against one XML and analyzed against another resolves every index and comes out quietly wrong.

Run:
    MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.sources.record_check \
        --records outputs/.../ag3s_records/run_0002
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from benchmark.ag3s.experiments.sources.policy_record import (
    ATTENTION_GRID,
    CAMERA_BINDINGS,
    POLICY_CAMERA_NAMES,
    load_run,
    patch_coverage,
    pose_scene,
    replay_scene,
)

from benchmark.ag3s.experiments.common.figstyle import (  # noqa: E402
    CATEGORICAL, GRID_INK, INK, INK_2, SURFACE, style_axes, use_korean,
)


def main() -> None:
    from benchmark.ag3s.experiments.reports.attention_report import CANDIDATE_BODIES, target_from_prompt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--camera", default="cam_high", choices=sorted(CAMERA_BINDINGS))
    ap.add_argument("--target", default=None)
    ap.add_argument("--min-target-px", type=int, default=200)
    ap.add_argument("--min-scored-frames", type=int, default=15,
                    help="below this the run is reported as too thin to score")
    ap.add_argument("--out-fig", default="benchmark/ag3s/asset/image/attention/fig0_record_sanity.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    use_korean()
    import matplotlib.pyplot as plt
    import mujoco

    run = load_run(args.records)
    target = args.target or target_from_prompt(run.prompt)
    print(f"records   : {run.path}")
    print(f"prompt    : {run.prompt!r}   -> target = {target}")
    print(f"steps     : {len(run)}  t_step {run.steps[0].t_step}..{run.steps[-1].t_step}")
    if target is None:
        raise SystemExit("could not decide the target from the prompt -- pass --target")

    spacing = np.diff([s.t_step for s in run])
    if len(set(spacing.tolist())) > 1:
        print(f"  ! t_step spacing is not constant: {sorted(set(spacing.tolist()))}")

    print("\nimages (frame 0):")
    for key in POLICY_CAMERA_NAMES:
        img = run.steps[0].images[key]
        print(f"  {key:16s} {img.shape} {img.dtype} mean={img.mean():6.1f} std={img.std():5.1f}")
        if img.std() < 5.0:
            print(f"  ! {key} is nearly constant -- the renderer may have failed")
    stack = np.stack([s.images[args.camera] for s in run]).astype(np.float64)
    motion = float(np.abs(np.diff(stack, axis=0)).mean())
    print(f"  {args.camera} frame-to-frame mean |diff| = {motion:.2f}"
          f"{'   ! frozen image' if motion < 0.5 else ''}")

    scene = replay_scene(run)
    body_ids, present = [], []
    for name in CANDIDATE_BODIES:
        bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            body_ids.append(bid)
            present.append(name)
    ti = present.index(target)
    visible = np.zeros((len(run), len(present)), np.int64)
    coverage = np.zeros((len(run), ATTENTION_GRID, ATTENTION_GRID), np.float32)
    camera_mujoco = CAMERA_BINDINGS[args.camera][0]
    for i, step in enumerate(run):
        pose_scene(scene, step)
        frame = scene.capture(camera_mujoco)
        cov, area = patch_coverage(frame.body_ids, body_ids)
        visible[i], coverage[i] = area, cov[ti]
    scene.close()

    keep = visible[:, ti] >= args.min_target_px
    t = np.asarray([s.t_step for s in run])
    print(f"\nvisible pixels in {args.camera} ({camera_mujoco}):")
    for j, name in enumerate(present):
        mark = "  <- target" if j == ti else ""
        print(f"  {name:8s} median={int(np.median(visible[:, j])):6d} min={visible[:, j].min():6d} "
              f"frames>={args.min_target_px}: {int((visible[:, j] >= args.min_target_px).sum())}"
              f"/{len(run)}{mark}")
    occluded = t[~keep]
    print(f"\nscoreable frames: {int(keep.sum())}/{len(run)}")
    if occluded.size:
        print(f"target below threshold at t_step {occluded.min()}..{occluded.max()} "
              f"({occluded.size} frames)")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.6), dpi=160, height_ratios=[1.7, 1])
    fig.patch.set_facecolor(SURFACE)
    style_axes(fig, (a1, a2))
    a1.grid(axis="y", color=GRID_INK, lw=0.6)
    a1.set_axisbelow(True)
    for lo in occluded:
        a1.axvspan(lo - 4, lo + 4, color=GRID_INK, alpha=0.55, lw=0, zorder=0)
    a1.plot(t, visible[:, ti], color=CATEGORICAL[0], lw=2.2, zorder=3)
    a1.axhline(args.min_target_px, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    a1.text(0.004, args.min_target_px, f" 채점 하한 {args.min_target_px} px",
            transform=a1.get_yaxis_transform(), va="bottom", fontsize=7.5, color=INK_2)
    a1.set_ylabel(f"{target} 가시 픽셀")
    a1.set_ylim(0, None)
    a1.set_title(f"{run.path.name} — {len(run)}프레임 중 {int(keep.sum())}개 채점 가능 "
                 f"(음영 = {target} 가려짐)", color=INK, fontsize=11, loc="left", pad=8)
    a2.imshow(coverage.reshape(len(run), -1).T, aspect="auto", cmap="Blues",
              extent=(float(t[0]), float(t[-1]), 0, 1), interpolation="nearest")
    a2.set_yticks([])
    a2.set_xlabel("제어 스텝")
    a2.set_ylabel(f"{target}\n패치 점유", fontsize=8)
    fig.tight_layout()
    out = pathlib.Path(args.out_fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    ok = int(keep.sum()) >= args.min_scored_frames and motion >= 0.5
    print(f"\n{'READY' if ok else 'NOT READY'}: "
          f"{int(keep.sum())} scoreable frames (need >= {args.min_scored_frames})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
