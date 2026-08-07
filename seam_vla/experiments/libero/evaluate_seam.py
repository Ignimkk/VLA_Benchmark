"""Evaluate SEAM (VLS enabled) pi0.5 on LIBERO-10 (in-process rollout).

Requires the LIBERO simulator. The full 10-task x 130-episode benchmark runs only with
--run-full-benchmark (or an explicit --config full_libero10.yaml + --num-episodes 130).

Example (single task, few episodes):
    python -m benchmark.seam_vla.experiments.libero.evaluate_seam \
        --config benchmark/seam_vla/configs/seam_libero10.yaml \
        --task-ids 0 --num-episodes 5 --out /tmp/seam_results.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json

from benchmark.seam_vla.experiments.config_io import RolloutConfig, load_experiment
from benchmark.seam_vla.experiments.libero.evaluator import DEFAULT_CKPT, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="benchmark/seam_vla/configs/seam_libero10.yaml")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out", default="/tmp/seam_seam_results.json")
    ap.add_argument("--task-ids", type=int, nargs="*", default=None)
    ap.add_argument("--num-episodes", type=int, default=None)
    ap.add_argument(
        "--run-full-benchmark",
        action="store_true",
        help="Authorize the full 10-task x 130-episode LIBERO-10 benchmark.",
    )
    args = ap.parse_args()

    seam_cfg, rollout_cfg = load_experiment(args.config)
    if args.task_ids is not None:
        rollout_cfg.task_ids = args.task_ids
    if args.num_episodes is not None:
        rollout_cfg.num_episodes = args.num_episodes

    is_full = len(rollout_cfg.task_ids) >= 10 and rollout_cfg.num_episodes >= 130
    if is_full and not args.run_full_benchmark:
        raise SystemExit(
            "Refusing to launch the full 10-task x 130-episode benchmark without --run-full-benchmark."
        )

    results = evaluate(seam_cfg, rollout_cfg, checkpoint_dir=args.checkpoint)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[seam] overall_success={results['overall_success']:.3f} "
          f"BJ={results['metrics'].get('BJ')} corr_mean={results['correction_norm_mean']:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
