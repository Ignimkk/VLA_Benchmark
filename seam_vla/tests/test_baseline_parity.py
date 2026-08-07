"""Heavy: real-model baseline parity. Run with SEAM_RUN_MODEL_TESTS=1 (CPU recommended on 8GB GPUs)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.seam_vla.tests.conftest import run_model_tests

pytestmark = run_model_tests


def _setup():
    from benchmark.seam_vla.tests import _model_fixture as mf

    obs, noise, _ = mf.make_obs_and_noise()
    sampler = mf.build_sampler(guided_window=5)
    from benchmark.seam_vla.priors.aligned_tail import build_aligned_prior

    rng = jax.random.key(0)
    base = np.asarray(sampler.sample_baseline(rng, obs, noise))
    prior = build_aligned_prior(jnp.asarray(base[0]), mf.H, mf.K)[None]
    return sampler, obs, noise, rng, base, prior


def test_hook_none_vs_enabled_zero_exact():
    sampler, obs, noise, rng, base, prior = _setup()
    seam_off = np.asarray(sampler.sample_seam(rng, obs, prior, 0.1, 0.0, noise))
    assert np.max(np.abs(base - seam_off)) == 0.0


def test_lambda_zero_exact_baseline():
    sampler, obs, noise, rng, base, prior = _setup()
    seam_l0 = np.asarray(sampler.sample_seam(rng, obs, prior, 0.0, 1.0, noise))
    assert np.max(np.abs(base - seam_l0)) == 0.0


def test_seam_enabled_changes_output_finitely():
    sampler, obs, noise, rng, base, prior = _setup()
    seam_on = np.asarray(sampler.sample_seam(rng, obs, prior, 0.1, 1.0, noise))
    assert np.all(np.isfinite(seam_on))
    assert np.max(np.abs(base - seam_on)) > 0.0


def test_shapes_model_space():
    sampler, obs, noise, rng, base, prior = _setup()
    assert base.shape == (1, 10, 32)
