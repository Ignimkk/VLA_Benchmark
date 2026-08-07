"""Single-task smoke validation for SEAM.

Two modes:
  * real-env  (default when LIBERO is installed): runs a few matched baseline-vs-SEAM episodes on one
    LIBERO-10 task via the in-process evaluator.
  * sim-free  (--sim-free, or auto-fallback when LIBERO is missing): loads the real pi0.5 model and
    drives SeamPolicy over a repeated observation for several chunk queries, validating the full SEAM
    machinery (shapes, effective H/K/L/M/N/D, dim mask, model-space carry, VLS effect / correction
    norms, first-chunk & reset behavior, NaN/Inf, and compile-vs-steady latency) WITHOUT env dynamics.

Note: on <=8GB GPUs the pi0.5 model does not fit; run with JAX_PLATFORMS=cpu.

Example:
    JAX_PLATFORMS=cpu python -m benchmark.seam_vla.experiments.libero.smoke_test --sim-free
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.seam_vla.config import SeamConfig
from benchmark.seam_vla.experiments.config_io import load_experiment
from benchmark.seam_vla.experiments.libero.evaluator import DEFAULT_CKPT
from benchmark.seam_vla.metrics.latency import time_call
from benchmark.seam_vla.metrics.motion import compute_motion_metrics
from benchmark.seam_vla.policy.seam_policy import SeamPolicy
from benchmark.seam_vla.state import SeamState


def _libero_available() -> bool:
    try:
        import libero  # noqa: F401

        return True
    except Exception:
        return False


def run_sim_free(seam_cfg: SeamConfig, checkpoint: str, num_chunks: int = 6) -> dict:
    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc, libero_policy

    print("[load] loading pi05_libero (CPU recommended on <=8GB GPUs)...", flush=True)
    t0 = time.perf_counter()
    openpi_policy = _pc.create_trained_policy(_config.get_config("pi05_libero"), checkpoint)
    print(f"[load] done in {time.perf_counter()-t0:.1f}s device={jax.devices()[0]}", flush=True)

    seam_cfg_on = SeamConfig(**{**seam_cfg.to_dict(), "seam_enabled": True})
    sp = SeamPolicy(openpi_policy, seam_cfg_on, rollout_execution_length=seam_cfg_on.seam_execution_length)
    print(f"[dims] H={sp.H} K={sp.K} L={sp.L} M={sp.M} N={sp.N} D={sp.D} "
          f"guided_dims={int(sp.config.build_dim_mask().sum())}", flush=True)

    obs = libero_policy.make_libero_example()

    # Fixed rng/noise so baseline and SEAM(enabled=0) are exactly comparable.
    rng = jax.random.key(0)
    noise = jax.random.normal(jax.random.key(1), (1, sp.H, sp.D))

    # --- drive several chunk queries, carrying model-space tail ---
    state = SeamState.initial()
    executed = []
    per_chunk_corr = []
    for i in range(num_chunks):
        chunk, state, diag = sp.predict_chunk(dict(obs), state, want_diagnostics=True)
        assert chunk.shape == (sp.H, 7), chunk.shape
        assert np.all(np.isfinite(chunk)), "NaN/Inf in physical chunk"
        assert state.previous_chunk_model_space.shape == (sp.H, sp.D)
        used_vls = diag.used_vls
        per_chunk_corr.append(diag.correction_norm)
        print(f"[chunk {i}] used_vls={used_vls} corr_norm={diag.correction_norm:.5f} "
              f"model_shape={diag.model_space_shape} phys_shape={diag.physical_space_shape}", flush=True)
        # first chunk must be baseline (no VLS); subsequent chunks use VLS.
        if i == 0:
            assert used_vls is False, "first chunk must not use VLS"
        else:
            assert used_vls is True, "non-first chunk should use VLS"
            assert diag.correction_norm > 0.0, "VLS correction should be non-zero"
        executed.append(chunk[: sp.K])  # baseline K-execution semantics

    # reset -> first chunk baseline again
    state2 = state.reset()
    assert state2.is_first_chunk
    _, _, diag_reset = sp.predict_chunk(dict(obs), state2, want_diagnostics=True)
    assert diag_reset.used_vls is False
    print(f"[reset] first chunk after reset used_vls={diag_reset.used_vls} (expected False)", flush=True)

    executed = np.concatenate(executed, axis=0)
    metrics = compute_motion_metrics(executed, sp.K)
    print(f"[metrics] BJ={metrics['BJ']:.4f} IJ={metrics['IJ']:.4f} CD={metrics['CD']:.4f} "
          f"AVb={metrics['paper_avb']:.4f}", flush=True)

    # --- latency: compile vs steady-state (baseline sampler), synchronized ---
    from openpi.models import model as _model

    inputs = openpi_policy._input_transform(dict(obs))
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    obs_model = _model.Observation.from_dict(inputs)
    timing = time_call(
        lambda: sp._sampler.sample_baseline(rng, obs_model, noise), warmup=1, repeats=3
    )
    print(f"[latency] denoise-loop compile={timing.compile_ms:.0f}ms "
          f"steady={timing.steady_ms_mean:.0f}±{timing.steady_ms_std:.0f}ms", flush=True)

    print("SMOKE_SIM_FREE_OK", flush=True)
    return {
        "H": sp.H, "K": sp.K, "L": sp.L, "M": sp.M, "N": sp.N, "D": sp.D,
        "per_chunk_correction_norm": per_chunk_corr,
        "metrics": metrics,
        "latency": timing.to_dict(),
    }


def run_real_env(config_path: str, checkpoint: str, task_ids, num_episodes):
    from benchmark.seam_vla.experiments.libero.evaluator import evaluate

    seam_cfg, rollout_cfg = load_experiment(config_path)
    if task_ids is not None:
        rollout_cfg.task_ids = task_ids
    if num_episodes is not None:
        rollout_cfg.num_episodes = num_episodes
    return evaluate(seam_cfg, rollout_cfg, checkpoint_dir=checkpoint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="benchmark/seam_vla/configs/smoke_libero10.yaml")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--sim-free", action="store_true", help="Force sim-free pipeline validation.")
    ap.add_argument("--task-ids", type=int, nargs="*", default=[0])
    ap.add_argument("--num-episodes", type=int, default=2)
    args = ap.parse_args()

    seam_cfg, _ = load_experiment(args.config)
    if args.sim_free or not _libero_available():
        if not args.sim_free:
            print("[warn] LIBERO not installed; falling back to sim-free validation.", flush=True)
        run_sim_free(seam_cfg, args.checkpoint)
    else:
        res = run_real_env(args.config, args.checkpoint, args.task_ids, args.num_episodes)
        print(f"[real-env] overall_success={res['overall_success']:.3f} metrics={res['metrics']}")


if __name__ == "__main__":
    main()
