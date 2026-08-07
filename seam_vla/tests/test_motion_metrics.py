import numpy as np

from benchmark.seam_vla.metrics.motion import (
    boundary_indices,
    compute_motion_metrics,
    per_step_jerk,
)

K = 5


def test_boundary_indices():
    b = boundary_indices(21, K)  # T=21 -> centers 1..19 -> multiples of 5: 5,10,15
    np.testing.assert_array_equal(b, np.array([5, 10, 15]))


def test_boundary_excludes_last_center():
    # T=11 -> centers 1..9 -> only 5 (10 is out of range since T-2=9)
    b = boundary_indices(11, K)
    np.testing.assert_array_equal(b, np.array([5]))


def test_jerk_zero_for_linear_motion():
    # Constant velocity -> second difference is zero -> jerk 0.
    t = np.arange(20)[:, None].astype(float)
    a = np.concatenate([t, 2 * t, -t], axis=1)  # linear in t
    j = per_step_jerk(a)
    np.testing.assert_allclose(j, 0.0, atol=1e-9)


def test_synthetic_known_jerk():
    # 1-D signal with a single kink; compute jerk by hand.
    a = np.array([0, 1, 2, 10, 11, 12], dtype=float)[:, None]
    j = per_step_jerk(a)
    # centers t=1..4: second diffs = a[t+1]-2a[t]+a[t-1]
    #   t=1: 2-2+0=0 ; t=2: 10-4+1=7 ; t=3: 11-20+2=-7 ; t=4: 12-22+10=0
    ref = np.array([0, 7, -7, 0], dtype=float)
    np.testing.assert_allclose(j, np.abs(ref))


def test_metrics_keys_and_alias():
    a = np.cumsum(np.random.default_rng(0).standard_normal((60, 7)), axis=0)
    m = compute_motion_metrics(a, K)
    for key in ("BJ", "IJ", "CD", "paper_avb", "boundary_jerk_variance"):
        assert key in m
    assert m["paper_avb"] == m["boundary_jerk_variance"]


def test_avb_is_variance_of_boundary_jerk():
    rng = np.random.default_rng(1)
    a = np.cumsum(rng.standard_normal((51, 3)), axis=0)
    m = compute_motion_metrics(a, K)
    j = per_step_jerk(a)
    b = boundary_indices(a.shape[0], K)
    b = b[b <= a.shape[0] - 2]
    ref_var = float(np.var(j[b - 1]))
    assert abs(m["paper_avb"] - ref_var) < 1e-9


def test_boundary_and_interior_disjoint_cover():
    a = np.cumsum(np.random.default_rng(2).standard_normal((40, 4)), axis=0)
    m = compute_motion_metrics(a, K)
    assert m["num_boundary"] + m["num_interior"] == a.shape[0] - 2
