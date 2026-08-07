import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.seam_vla.guidance.vls import apply_guidance, make_position_mask
from benchmark.seam_vla.priors.aligned_tail import build_aligned_prior
from benchmark.seam_vla.tests.conftest import requires_gpu

H, K, D, M, NVALID = 10, 5, 32, 5, 7


def _dim_mask():
    m = np.zeros((D,), dtype=bool)
    m[:NVALID] = True
    return m


@pytest.mark.parametrize("batch", [(), (1,), (4,), (2, 3)])
def test_arbitrary_batch_dims(batch):
    shape = batch + (H, D)
    cand = jnp.asarray(np.random.default_rng(0).standard_normal(shape), dtype=jnp.float32)
    prior = build_aligned_prior(cand, H, K)
    out = apply_guidance(cand, prior, 0.3, 0.1, make_position_mask(H, M), _dim_mask())
    assert out.shape == shape
    # unguided rows untouched
    np.testing.assert_array_equal(np.asarray(out)[..., M:, :], np.asarray(cand)[..., M:, :])


def test_device_placement_preserved():
    cand = jnp.asarray(np.random.default_rng(0).standard_normal((H, D)), dtype=jnp.float32)
    prior = build_aligned_prior(cand, H, K)
    out = apply_guidance(cand, prior, 0.3, 0.1, make_position_mask(H, M), _dim_mask())
    # Output lives on the same device set as the input.
    in_dev = list(cand.devices())
    out_dev = list(out.devices())
    assert out_dev == in_dev


def test_cpu_execution():
    with jax.default_device(jax.devices("cpu")[0]):
        cand = jnp.asarray(np.random.default_rng(0).standard_normal((H, D)), dtype=jnp.float32)
        prior = build_aligned_prior(cand, H, K)
        out = apply_guidance(cand, prior, 0.3, 0.1, make_position_mask(H, M), _dim_mask())
        assert "cpu" in str(next(iter(out.devices()))).lower()


@requires_gpu
def test_gpu_execution():
    gpu = jax.devices("gpu")[0]
    with jax.default_device(gpu):
        cand = jnp.asarray(np.random.default_rng(0).standard_normal((H, D)), dtype=jnp.float32)
        prior = build_aligned_prior(cand, H, K)
        out = apply_guidance(cand, prior, 0.3, 0.1, make_position_mask(H, M), _dim_mask())
        assert np.all(np.isfinite(np.asarray(out)))
