"""Normalization round-trip: Unnormalize(Normalize(x)) == x for the LIBERO quantile stats.

Light test (no model forward). Skips if the local norm_stats are unavailable.
"""

import os

import numpy as np
import pytest

from benchmark.seam_vla.tests.conftest import CKPT

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "assets/physical-intelligence/libero/norm_stats.json")),
    reason="local pi05_libero norm_stats.json not found",
)


def _load_stats():
    from openpi.shared import normalize as _normalize

    stats_dir = os.path.join(CKPT, "assets/physical-intelligence/libero")
    return _normalize.load(stats_dir)


def test_quantile_roundtrip_actions_and_state():
    from openpi import transforms

    stats = _load_stats()
    rng = np.random.default_rng(0)
    # actions: 7 physical dims; state: 8 dims. Use physical-range-ish values.
    data = {
        "actions": rng.uniform(-1, 1, size=(10, 7)).astype(np.float32),
        "state": rng.uniform(-1, 1, size=(8,)).astype(np.float32),
    }
    norm = transforms.Normalize(stats, use_quantiles=True)
    unnorm = transforms.Unnormalize(stats, use_quantiles=True)
    out = unnorm(norm({k: v.copy() for k, v in data.items()}))
    np.testing.assert_allclose(out["actions"], data["actions"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out["state"], data["state"], rtol=1e-4, atol=1e-4)


def test_padded_dims_pass_through_unnormalize():
    from openpi import transforms

    stats = _load_stats()
    # A model-space action padded to 32 dims: unnormalize should only touch the first 7.
    x = np.zeros((10, 32), dtype=np.float32)
    x[:, 7:] = 3.14  # padded dims
    # Unnormalize is strict over norm_stats keys, so include a state entry too.
    data = {"actions": x.copy(), "state": np.zeros((8,), dtype=np.float32)}
    out = transforms.Unnormalize(stats, use_quantiles=True)(data)["actions"]
    np.testing.assert_array_equal(out[:, 7:], x[:, 7:])  # padded dims unchanged


def test_stats_dims_match_libero():
    stats = _load_stats()
    assert stats["actions"].q01.shape[-1] == 7
    assert stats["state"].q01.shape[-1] == 8
