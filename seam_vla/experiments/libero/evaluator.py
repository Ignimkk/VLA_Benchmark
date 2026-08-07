"""Shared LIBERO evaluator: build a SEAM policy and run matched rollouts, collecting metrics.

Requires the LIBERO simulator (``libero``/``robosuite``/``mujoco``) to actually run episodes. The
policy-construction path only needs OpenPI + the local checkpoint.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any

import numpy as np

from benchmark.seam_vla.config import SeamConfig
from benchmark.seam_vla.experiments.config_io import RolloutConfig
from benchmark.seam_vla.metrics.motion import aggregate_metrics, compute_motion_metrics
from benchmark.seam_vla.policy.seam_policy import SeamPolicy, SeamPolicySession

DEFAULT_CKPT = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi05_libero")


def build_seam_policy(seam_config: SeamConfig, checkpoint_dir: str = DEFAULT_CKPT) -> SeamPolicy:
    """Load pi05_libero and wrap it in a SeamPolicy (asserts config vs. model)."""
    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc

    train_config = _config.get_config("pi05_libero")
    openpi_policy = _pc.create_trained_policy(train_config, checkpoint_dir)
    return SeamPolicy(
        openpi_policy,
        seam_config,
        rollout_execution_length=seam_config.seam_execution_length,
    )


def evaluate(
    seam_config: SeamConfig,
    rollout_config: RolloutConfig,
    *,
    checkpoint_dir: str = DEFAULT_CKPT,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the (in-process) LIBERO rollout for the given tasks/episodes and return a results dict.

    Lazily imports ``libero``; raises a clear error if the simulator is unavailable.
    """
    try:
        from libero.libero import benchmark as libero_benchmark
    except ImportError as e:  # pragma: no cover - depends on env
        raise RuntimeError(
            "LIBERO simulator not installed (need `libero`, `robosuite`, `mujoco`). See "
            "scripts/SETUP_CONTAINER.md. Policy code is validated separately without the sim."
        ) from e

    from benchmark.seam_vla.experiments.libero import rollout_common as rc

    seam_policy = build_seam_policy(seam_config, checkpoint_dir)
    session = SeamPolicySession(seam_policy)

    np.random.seed(rollout_config.seed)
    suite = libero_benchmark.get_benchmark_dict()[rollout_config.task_suite_name]()
    max_steps = rollout_config.max_steps or rc.max_steps_for(rollout_config.task_suite_name)

    per_episode_metrics = []
    per_task = {}
    all_corr_norms = []
    t_start = time.perf_counter()

    for task_id in rollout_config.task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, task_description = rc.make_env(task, rc.LIBERO_ENV_RESOLUTION, rollout_config.seed)
        successes = 0
        for ep in range(rollout_config.num_episodes):
            session.reset()  # first chunk of each episode uses baseline (no VLS)

            def predict(obs_dict):
                return session.predict_chunk(obs_dict, want_diagnostics=seam_config.seam_log_corrections)

            init = init_states[ep] if ep < len(init_states) else None
            result = rc.run_episode(
                env, task_description, predict, seam_config.seam_execution_length, max_steps, init
            )
            successes += int(result.success)
            all_corr_norms.extend(result.per_chunk_correction_norm)
            m = compute_motion_metrics(result.executed_actions, seam_config.seam_execution_length)
            m["success"] = bool(result.success)
            m["num_chunks"] = result.num_chunks
            m["episode_length"] = result.episode_length
            per_episode_metrics.append(m)
            if verbose:
                print(f"[task {task_id} ep {ep}] success={result.success} "
                      f"BJ={m['BJ']:.4f} CD={m['CD']:.4f} chunks={result.num_chunks}", flush=True)
        env.close()
        per_task[str(task_id)] = successes / max(rollout_config.num_episodes, 1)

    wall = time.perf_counter() - t_start
    agg = aggregate_metrics(per_episode_metrics)
    overall_success = float(np.mean([m["success"] for m in per_episode_metrics])) if per_episode_metrics else 0.0
    corr = np.asarray(all_corr_norms, dtype=np.float64)
    return {
        "seam_config": seam_config.to_dict(),
        "rollout_config": dataclasses.asdict(rollout_config),
        "overall_success": overall_success,
        "per_task_success": per_task,
        "metrics": agg,
        "correction_norm_mean": float(corr.mean()) if corr.size else 0.0,
        "correction_norm_max": float(corr.max()) if corr.size else 0.0,
        "num_vls_chunks": int((corr > 0).sum()),
        "num_episodes_total": len(per_episode_metrics),
        "wall_time_s": wall,
    }
