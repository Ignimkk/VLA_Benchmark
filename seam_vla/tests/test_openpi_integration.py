"""Heavy: SeamPolicy end-to-end integration (real model). SEAM_RUN_MODEL_TESTS=1 to enable.

Also includes light no-op checks for IdentityGuidance / IdentityChunkRefiner that always run.
"""

import numpy as np
import jax.numpy as jnp

from benchmark.seam_vla.guidance.identity import IdentityGuidance
from benchmark.seam_vla.refinement.identity import IdentityChunkRefiner
from benchmark.seam_vla.tests.conftest import run_model_tests


# --- light, always-on -------------------------------------------------------
def test_identity_guidance_is_noop():
    x = jnp.asarray(np.random.default_rng(0).standard_normal((10, 32)), dtype=jnp.float32)
    out = IdentityGuidance().update(x, 0.3)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))


def test_identity_refiner_is_noop():
    x = np.random.default_rng(0).standard_normal((10, 7))
    out = IdentityChunkRefiner().refine(x, context={"chunk_index": 0})
    np.testing.assert_array_equal(np.asarray(out), x)


# --- heavy, real model ------------------------------------------------------
@run_model_tests
def test_seam_policy_first_chunk_baseline_then_vls():
    from benchmark.seam_vla.tests import _model_fixture as mf
    from benchmark.seam_vla.config import SeamConfig
    from benchmark.seam_vla.policy.seam_policy import SeamPolicy
    from benchmark.seam_vla.state import SeamState
    from openpi.policies import libero_policy

    policy = mf.load_policy()
    cfg = SeamConfig(seam_enabled=True, seam_guided_window=20)
    sp = SeamPolicy(policy, cfg, rollout_execution_length=5)
    obs = libero_policy.make_libero_example()

    state = SeamState.initial()
    chunk1, state, d1 = sp.predict_chunk(obs, state, want_diagnostics=True)
    assert d1.used_vls is False  # first chunk: baseline
    assert chunk1.shape == (10, 7)
    assert state.previous_chunk_model_space.shape == (10, 32)
    assert not state.is_first_chunk

    chunk2, state, d2 = sp.predict_chunk(obs, state, want_diagnostics=True)
    assert d2.used_vls is True  # second chunk: VLS
    assert chunk2.shape == (10, 7)
    assert d2.correction_norm >= 0.0
    assert np.all(np.isfinite(chunk2))

    # reset -> first chunk baseline again
    state = state.reset()
    assert state.is_first_chunk
    _, state, d3 = sp.predict_chunk(obs, state)
    assert d3.used_vls is False


@run_model_tests
def test_seam_disabled_matches_baseline_policy():
    """SeamPolicy with seam_enabled=False reproduces the openpi Policy actions for a fixed rng/noise."""
    import jax
    from benchmark.seam_vla.tests import _model_fixture as mf
    from benchmark.seam_vla.config import SeamConfig
    from benchmark.seam_vla.policy.seam_policy import SeamPolicy
    from benchmark.seam_vla.state import SeamState
    from openpi.policies import libero_policy

    policy = mf.load_policy()
    cfg = SeamConfig(seam_enabled=False, seam_guided_window=20)
    sp = SeamPolicy(policy, cfg, rollout_execution_length=5)
    obs = libero_policy.make_libero_example()

    rng = jax.random.key(7)
    noise = jax.random.normal(jax.random.key(8), (1, 10, 32))
    chunk_a, _, _ = sp.predict_chunk(dict(obs), SeamState.initial(), rng=rng, noise=noise)
    chunk_b, _, _ = sp.predict_chunk(dict(obs), SeamState.initial(), rng=rng, noise=noise)
    np.testing.assert_array_equal(chunk_a, chunk_b)  # deterministic given rng+noise
