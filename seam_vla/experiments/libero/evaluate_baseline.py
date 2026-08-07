"""Evaluate the BASELINE (SEAM disabled) pi0.5 policy on LIBERO-10 (in-process rollout).

Requires the LIBERO simulator. Writes a JSON results file.

Example:
    python -m benchmark.seam_vla.experiments.libero.evaluate_baseline \
        --config benchmark/seam_vla/configs/baseline_libero10.yaml \
        --out /tmp/seam_baseline.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json

from benchmark.seam_vla.experiments.config_io import load_experiment
from benchmark.seam_vla.experiments.libero.evaluator import DEFAULT_CKPT, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="benchmark/seam_vla/configs/baseline_libero10.yaml")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out", default="/tmp/seam_baseline_results.json")
    ap.add_argument("--task-ids", type=int, nargs="*", default=None)
    ap.add_argument("--num-episodes", type=int, default=None)
    args = ap.parse_args()

    seam_cfg, rollout_cfg = load_experiment(args.config)
    seam_cfg = dataclasses.replace(seam_cfg, seam_enabled=False)  # force baseline
    if args.task_ids is not None:
        rollout_cfg.task_ids = args.task_ids
    if args.num_episodes is not None:
        rollout_cfg.num_episodes = args.num_episodes

    results = evaluate(seam_cfg, rollout_cfg, checkpoint_dir=args.checkpoint)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[baseline] overall_success={results['overall_success']:.3f} "
          f"BJ={results['metrics'].get('BJ')} -> {args.out}")


if __name__ == "__main__":
    main()
