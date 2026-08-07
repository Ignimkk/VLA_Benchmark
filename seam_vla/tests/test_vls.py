import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.seam_vla.guidance import vls as vls_mod
from benchmark.seam_vla.guidance.vls import (
    VLSGuidance,
    apply_guidance,
    compute_consistency_target,
    compute_vls_correction,
    make_position_mask,
)
from benchmark.seam_vla.priors.aligned_tail import build_aligned_prior

H, K, D = 10, 5, 32
L = H - K
M = 5
NVALID = 7


def _rng(seed):
    return np.random.default_rng(seed)


def _candidate(seed=0):
    return jnp.asarray(_rng(seed).standard_normal((H, D)), dtype=jnp.float32)


def _prior(seed=1):
    c = jnp.asarray(_rng(seed).standard_normal((H, D)), dtype=jnp.float32)
    return build_aligned_prior(c, H, K)


def _dim_mask(nvalid=NVALID):
    m = np.zeros((D,), dtype=bool)
    m[:nvalid] = True
    return m


def test_consistency_target_formula():
    prior = _prior()
    t_next = 0.3
    r = compute_consistency_target(prior, t_next)
    np.testing.assert_allclose(np.asarray(r), (1 - t_next) * np.asarray(prior), rtol=1e-6)


def test_correction_formula():
    cand = _candidate()
    prior = _prior()
    t_next = 0.4
    target = compute_consistency_target(prior, t_next)
    g = compute_vls_correction(cand, target)
    np.testing.assert_allclose(np.asarray(g), -2 * (np.asarray(cand) - np.asarray(target)), rtol=1e-6)


def test_manual_vls_update_matches_paper_eq8():
    cand = _candidate(2)
    prior = _prior(3)
    lam, t_next = 0.1, 0.2
    pos = make_position_mask(H, M)
    dim = _dim_mask()
    out = apply_guidance(cand, prior, t_next, lam, pos, dim, enabled=1.0)
    # Manual reference on the guided block only.
    ref = np.asarray(cand).copy()
    r = (1 - t_next) * np.asarray(prior)[:M, :NVALID]
    g = -2 * (np.asarray(cand)[:M, :NVALID] - r)
    ref[:M, :NVALID] = np.asarray(cand)[:M, :NVALID] + lam * (1 - t_next) * g
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-6)


def test_unguided_positions_equal_candidate():
    cand = _candidate(4)
    out = apply_guidance(cand, _prior(5), 0.5, 0.1, make_position_mask(H, M), _dim_mask())
    np.testing.assert_array_equal(np.asarray(out)[M:], np.asarray(cand)[M:])


def test_unguided_dims_equal_candidate():
    cand = _candidate(6)
    out = apply_guidance(cand, _prior(7), 0.5, 0.1, make_position_mask(H, M), _dim_mask())
    np.testing.assert_array_equal(np.asarray(out)[:, NVALID:], np.asarray(cand)[:, NVALID:])


def test_lambda_zero_is_baseline():
    cand = _candidate(8)
    out = apply_guidance(cand, _prior(9), 0.5, 0.0, make_position_mask(H, M), _dim_mask())
    np.testing.assert_array_equal(np.asarray(out), np.asarray(cand))


def test_enabled_zero_is_baseline():
    cand = _candidate(10)
    out = apply_guidance(cand, _prior(11), 0.5, 0.1, make_position_mask(H, M), _dim_mask(), enabled=0.0)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(cand))


def test_window_zero_is_baseline():
    cand = _candidate(12)
    out = apply_guidance(cand, _prior(13), 0.5, 0.1, make_position_mask(H, 0), _dim_mask())
    np.testing.assert_array_equal(np.asarray(out), np.asarray(cand))


def test_empty_dim_mask_is_baseline():
    cand = _candidate(14)
    empty = np.zeros((D,), dtype=bool)
    out = apply_guidance(cand, _prior(15), 0.5, 0.1, make_position_mask(H, M), empty)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(cand))


def test_window_clamped_by_position_mask():
    # M > L: only min(M, L) rows are guided when pos mask built with min.
    m_eff = min(20, L)
    pos = make_position_mask(H, m_eff)
    assert pos[:m_eff].all() and not pos[m_eff:].any()


def test_explicit_dim_mask():
    cand = _candidate(16)
    dim = np.zeros((D,), dtype=bool)
    dim[[1, 3, 6]] = True
    out = apply_guidance(cand, _prior(17), 0.4, 0.2, make_position_mask(H, M), dim)
    o, c = np.asarray(out), np.asarray(cand)
    changed = np.where(np.any(o != c, axis=0))[0]
    assert set(changed.tolist()).issubset({1, 3, 6})


def test_batch_inputs():
    B = 3
    cand = jnp.asarray(_rng(0).standard_normal((B, H, D)), dtype=jnp.float32)
    prior = jnp.asarray(_rng(1).standard_normal((B, H, D)), dtype=jnp.float32)
    out = apply_guidance(cand, prior, 0.3, 0.1, make_position_mask(H, M), _dim_mask())
    assert out.shape == (B, H, D)
    np.testing.assert_array_equal(np.asarray(out)[:, M:], np.asarray(cand)[:, M:])


def test_dtype_preserved():
    for dt in (jnp.float32, jnp.float16):
        cand = jnp.asarray(_rng(0).standard_normal((H, D)), dtype=dt)
        prior = jnp.asarray(_rng(1).standard_normal((H, D)), dtype=dt)
        out = apply_guidance(cand, prior, 0.3, 0.1, make_position_mask(H, M), _dim_mask())
        assert out.dtype == dt


def test_no_nan_inf():
    cand = _candidate(0)
    out = apply_guidance(cand, _prior(1), 0.99, 0.1, make_position_mask(H, M), _dim_mask())
    arr = np.asarray(out)
    assert np.all(np.isfinite(arr))


def test_jit_compatible():
    pos = make_position_mask(H, M)
    dim = _dim_mask()
    f = jax.jit(lambda cand, prior, t, lam: apply_guidance(cand, prior, t, lam, pos, dim))
    out = f(_candidate(0), _prior(1), 0.3, 0.1)
    assert out.shape == (H, D)


def test_guidance_class_matches_functional():
    cand = _candidate(21)
    prior = _prior(22)
    g = VLSGuidance(prior, 0.1, make_position_mask(H, M), _dim_mask(), enabled=1.0)
    out_cls = g.update(cand, 0.25)
    out_fn = apply_guidance(cand, prior, 0.25, 0.1, make_position_mask(H, M), _dim_mask(), 1.0)
    np.testing.assert_array_equal(np.asarray(out_cls), np.asarray(out_fn))


def test_no_autodiff_in_source():
    # VLS must be closed-form: no jax.grad / value_and_grad / jax.vjp / jax.jacobian in the module.
    src = inspect.getsource(vls_mod)
    for banned in ("jax.grad", "value_and_grad", "jax.vjp", "jax.jacobian", "jax.jacrev", "jax.jacfwd"):
        assert banned not in src, f"forbidden autodiff call {banned} found in vls.py"
