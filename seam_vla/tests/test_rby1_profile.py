import numpy as np
import pytest

from benchmark.seam_vla.config import SeamConfig, SeamConfigError


def _rby1_cfg(**kw):
    base = dict(
        seam_enabled=True, seam_horizon=50, seam_execution_length=8, seam_num_ode_steps=10,
        seam_guided_window=20, seam_guided_dimensions=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        num_valid_physical_dims=14, action_dim=32, seam_delta_base_compensation=True,
    )
    base.update(kw)
    return SeamConfig(**base)


def test_rby1_dims_and_overlap():
    c = _rby1_cfg()
    assert c.overlap_length == 42  # L = 50 - 8
    assert c.effective_window == 20  # min(20, 42)


def test_dim_mask_excludes_grippers():
    c = _rby1_cfg()
    m = c.build_dim_mask()
    assert m.shape == (32,)
    assert m[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].all()
    assert not m[6] and not m[13]  # grippers excluded
    assert not m[14:].any()  # padding excluded
    assert m.sum() == 12


def test_window_clamps_to_overlap():
    c = _rby1_cfg(seam_guided_window=100)
    assert c.effective_window == 42  # clamped to L


def test_compensation_requires_denoising_stage():
    with pytest.raises(SeamConfigError):
        _rby1_cfg(seam_stage="decoded", seam_delta_base_compensation=True)


def test_assert_matches_model_ok():
    c = _rby1_cfg()
    c.assert_matches_model(model_horizon=50, model_num_steps=10, rollout_execution_length=8)


def test_assert_matches_model_fails_on_wrong_horizon():
    c = _rby1_cfg()
    with pytest.raises(SeamConfigError):
        c.assert_matches_model(model_horizon=10, model_num_steps=10, rollout_execution_length=8)


def test_from_yaml_rby1():
    c = SeamConfig.from_yaml("benchmark/seam_vla/configs/seam_rby1.yaml")
    assert c.seam_enabled and c.seam_horizon == 50 and c.seam_execution_length == 8
    assert c.seam_delta_base_compensation is True
    assert c.build_dim_mask().sum() == 12
