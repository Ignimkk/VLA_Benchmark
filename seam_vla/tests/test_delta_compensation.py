"""Base-state compensation math for rby1 (base-relative-delta model space). No model needed."""

import jax.numpy as jnp
import numpy as np

from benchmark.seam_vla.priors.aligned_tail import (
    build_aligned_prior,
    build_aligned_prior_delta_compensated,
)

H, K, D = 50, 8, 32
L = H - K
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def _quantile_unnorm(x_norm, q01, q99):
    return (x_norm + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    d_n_norm = rng.uniform(-1, 1, size=(H, D)).astype(np.float64)  # previous chunk normalized deltas
    q01 = rng.uniform(-2, -0.5, size=(D,))
    q99 = rng.uniform(0.5, 2.0, size=(D,))
    s_prev = rng.uniform(-1, 1, size=(D,))
    s_curr = rng.uniform(-1, 1, size=(D,))
    inv_scale = 2.0 / (q99 - q01 + 1e-6)
    guided = np.zeros((D,), dtype=bool)
    guided[ARM] = True
    return d_n_norm, q01, q99, s_prev, s_curr, inv_scale, guided


def test_compensated_prior_decodes_to_previous_absolute_tail():
    d_n_norm, q01, q99, s_prev, s_curr, inv_scale, guided = _setup()
    delta_state = s_curr - s_prev  # Δs (physical)
    prior = np.asarray(
        build_aligned_prior_delta_compensated(
            jnp.asarray(d_n_norm[None]), jnp.asarray(delta_state[None]),
            jnp.asarray(inv_scale), jnp.asarray(guided), H, K,
        )
    )[0]

    # Previous chunk's ABSOLUTE tail: unnorm(prev_delta[K:H]) + s_prev.
    prev_abs_tail = _quantile_unnorm(d_n_norm[K:H], q01, q99) + s_prev  # [L, D]
    # Decode the compensated prior with the NEW base state s_curr.
    prior_abs = _quantile_unnorm(prior[:L], q01, q99) + s_curr  # [L, D]

    # On guided dims they must match (that's the whole point of compensation).
    np.testing.assert_allclose(prior_abs[:, ARM], prev_abs_tail[:, ARM], rtol=1e-6, atol=1e-6)


def test_naive_vs_compensated_differ_by_delta_scaled():
    d_n_norm, q01, q99, s_prev, s_curr, inv_scale, guided = _setup(1)
    delta_state = s_curr - s_prev
    naive = np.asarray(build_aligned_prior(jnp.asarray(d_n_norm[None]), H, K))[0]
    comp = np.asarray(
        build_aligned_prior_delta_compensated(
            jnp.asarray(d_n_norm[None]), jnp.asarray(delta_state[None]),
            jnp.asarray(inv_scale), jnp.asarray(guided), H, K,
        )
    )[0]
    diff = naive - comp  # should equal Δs * inv_scale on guided dims, 0 elsewhere
    expected = (delta_state * inv_scale)
    np.testing.assert_allclose(diff[:L][:, ARM], np.broadcast_to(expected[ARM], (L, len(ARM))), rtol=1e-5, atol=1e-6)
    # unguided dims unchanged
    other = [i for i in range(D) if i not in ARM]
    np.testing.assert_allclose(diff[:, other], 0.0, atol=1e-7)


def test_zero_drift_equals_naive():
    d_n_norm, q01, q99, s_prev, s_curr, inv_scale, guided = _setup(2)
    zero = np.zeros((D,))
    naive = np.asarray(build_aligned_prior(jnp.asarray(d_n_norm[None]), H, K))
    comp = np.asarray(
        build_aligned_prior_delta_compensated(
            jnp.asarray(d_n_norm[None]), jnp.asarray(zero[None]),
            jnp.asarray(inv_scale), jnp.asarray(guided), H, K,
        )
    )
    np.testing.assert_allclose(comp, naive, atol=1e-7)
