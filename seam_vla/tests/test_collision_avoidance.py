import numpy as np
import pytest

from benchmark.seam_vla.refinement.base import DecodedChunkRefiner
from benchmark.seam_vla.refinement.collision_avoidance import CollisionAvoidanceRefiner

H, D = 10, 7


def _chunk(seed):
    return np.random.default_rng(seed).standard_normal((H, D))


def _distance_fn_constant(value):
    def fn(chunk):
        return np.full((chunk.shape[0], 1), value)

    return fn


def test_is_decoded_chunk_refiner_subclass():
    assert issubclass(CollisionAvoidanceRefiner, DecodedChunkRefiner)


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        CollisionAvoidanceRefiner(_distance_fn_constant(1.0), safety_margin=0.1, mode="bogus")


def test_refine_not_implemented():
    r = CollisionAvoidanceRefiner(_distance_fn_constant(1.0), safety_margin=0.1)
    with pytest.raises(NotImplementedError):
        r.refine(_chunk(0))


def test_would_violate_shape_and_dtype():
    r = CollisionAvoidanceRefiner(_distance_fn_constant(1.0), safety_margin=0.1)
    out = r.would_violate(_chunk(1))
    assert out.shape == (H,)
    assert out.dtype == np.bool_


def test_would_violate_flags_close_steps():
    # distance (0.05) below safety_margin (0.1) -> every step flagged unsafe.
    r = CollisionAvoidanceRefiner(_distance_fn_constant(0.05), safety_margin=0.1)
    out = r.would_violate(_chunk(2))
    assert np.all(out)


def test_would_violate_no_flags_when_safe():
    # distance (1.0) well above safety_margin (0.1) -> nothing flagged.
    r = CollisionAvoidanceRefiner(_distance_fn_constant(1.0), safety_margin=0.1)
    out = r.would_violate(_chunk(3))
    assert not np.any(out)


def test_would_violate_1d_distance_fn_broadcasts():
    def fn_1d(chunk):
        return np.zeros(chunk.shape[0])  # [H], not [H, n_obstacles]

    r = CollisionAvoidanceRefiner(fn_1d, safety_margin=0.1)
    out = r.would_violate(_chunk(4))
    assert out.shape == (H,)
    assert np.all(out)
