import numpy as np
import pytest

from benchmark.seam_vla.config import SeamConfig, SeamConfigError, build_dim_mask

D = 32


def test_all_physical_mask():
    m = build_dim_mask("all_physical", 7, D)
    assert m.shape == (D,)
    assert m[:7].all()
    assert not m[7:].any()


def test_all_physical_excludes_padding():
    m = build_dim_mask("all_physical", 7, D)
    assert m.sum() == 7  # padded dims 7..31 excluded


def test_index_list_mask():
    m = build_dim_mask([0, 2, 6], 7, D)
    assert set(np.where(m)[0].tolist()) == {0, 2, 6}


def test_bool_mask_passthrough():
    ref = np.zeros((D,), dtype=bool)
    ref[[1, 5, 9]] = True
    m = build_dim_mask(list(ref), 7, D)
    np.testing.assert_array_equal(m, ref)


def test_bad_index_rejected():
    with pytest.raises(SeamConfigError):
        build_dim_mask([0, 99], 7, D)


def test_empty_mask_rejected():
    with pytest.raises(SeamConfigError):
        build_dim_mask([], 7, D)


def test_config_build_dim_mask():
    cfg = SeamConfig()
    m = cfg.build_dim_mask()
    assert m.sum() == 7
