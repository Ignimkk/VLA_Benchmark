import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.seam_vla.priors.aligned_tail import build_aligned_prior

H, K, D = 10, 5, 32
L = H - K


def _chunk(seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((H, D)), dtype=jnp.float32)


def test_shape():
    a = build_aligned_prior(_chunk(), H, K)
    assert a.shape == (H, D)


def test_tail_copied_into_prefix():
    c = _chunk(1)
    a = build_aligned_prior(c, H, K)
    # First L rows equal the unexecuted tail c[K:H].
    np.testing.assert_array_equal(np.asarray(a[:L]), np.asarray(c[K:H]))


def test_last_tail_action_repeated():
    c = _chunk(2)
    a = build_aligned_prior(c, H, K)
    last_tail = np.asarray(c[H - 1])
    for row in range(L, H):
        np.testing.assert_array_equal(np.asarray(a[row]), last_tail)


def test_batch_shapes():
    B = 4
    c = jnp.asarray(np.random.default_rng(3).standard_normal((B, H, D)), dtype=jnp.float32)
    a = build_aligned_prior(c, H, K)
    assert a.shape == (B, H, D)
    np.testing.assert_array_equal(np.asarray(a[:, :L]), np.asarray(c[:, K:H]))


def test_arbitrary_dims():
    for h, k, d in [(6, 2, 3), (50, 10, 8), (10, 9, 1)]:
        c = jnp.asarray(np.random.default_rng(0).standard_normal((h, d)), dtype=jnp.float32)
        a = build_aligned_prior(c, h, k)
        assert a.shape == (h, d)
        np.testing.assert_array_equal(np.asarray(a[: h - k]), np.asarray(c[k:h]))


def test_empty_tail_rejected():
    c = _chunk()
    with pytest.raises(ValueError):
        build_aligned_prior(c, H, H)  # K == H -> L == 0


def test_bad_horizon_rejected():
    c = _chunk()
    with pytest.raises(ValueError):
        build_aligned_prior(c, H + 1, K)  # horizon mismatch


def test_jit_compatible():
    f = jax.jit(lambda c: build_aligned_prior(c, H, K))
    a = f(_chunk())
    assert a.shape == (H, D)
    assert not np.any(np.isnan(np.asarray(a)))
