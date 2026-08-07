import numpy as np

from benchmark.seam_vla.refinement.overlap_steer import OverlapSteerRefiner

H, K, D = 50, 8, 32
L = H - K
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def _chunk(seed):
    return np.random.default_rng(seed).standard_normal((H, D))


def test_first_chunk_unchanged():
    r = OverlapSteerRefiner(K, 20, 0.3, ARM)
    c = _chunk(0)
    out = r.refine(c, context={"previous_physical_chunk": None})
    np.testing.assert_array_equal(np.asarray(out), c)


def test_lambda_zero_unchanged():
    r = OverlapSteerRefiner(K, 20, 0.0, ARM)
    c, p = _chunk(1), _chunk(2)
    out = r.refine(c, context={"previous_physical_chunk": p})
    np.testing.assert_array_equal(np.asarray(out), c)


def test_only_guided_dims_and_window_changed():
    r = OverlapSteerRefiner(K, 20, 0.5, ARM)
    c, p = _chunk(3), _chunk(4)
    out = np.asarray(r.refine(c, context={"previous_physical_chunk": p}))
    # gripper dims untouched
    np.testing.assert_array_equal(out[:, [6, 13]], c[:, [6, 13]])
    # rows beyond window M=20 untouched
    np.testing.assert_array_equal(out[20:], c[20:])
    # some guided rows changed
    assert np.any(out[:20, ARM] != c[:20, ARM])


def test_moves_head_toward_previous_tail():
    # With lam=1 and no decay, the first row snaps fully to the previous tail on guided dims.
    r = OverlapSteerRefiner(K, 20, 1.0, ARM, decay=False)
    c, p = _chunk(5), _chunk(6)
    out = np.asarray(r.refine(c, context={"previous_physical_chunk": p}))
    np.testing.assert_allclose(out[0, ARM], p[K, ARM], atol=1e-9)


def test_decay_weight_tapers():
    r = OverlapSteerRefiner(K, 10, 1.0, ARM, decay=True)
    c = np.zeros((H, D)); p = np.ones((H, D))
    out = np.asarray(r.refine(c, context={"previous_physical_chunk": p}))
    # weight w(j)=1*(1-j/10); correction = w*(1-0)=w. Row 0 -> ~1.0, taper to ~0 at j=9.
    assert out[0, 0] > out[1, 0] > out[9, 0]
    np.testing.assert_allclose(out[0, 0], 1.0, atol=1e-9)
