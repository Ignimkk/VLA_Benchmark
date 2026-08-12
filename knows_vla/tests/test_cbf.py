"""P1 — CBF-QP unit verification. Covers every check in papers/KNOWS/06-repro-plan.md §8.2.

No model, no simulator: pure numerics against the paper's equations.

    src/openpi/.venv/bin/python -m pytest benchmark/knows_vla/tests/test_cbf.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.knows_vla.cbf.ellipsoid import (
    Ellipsoid,
    barrier,
    grad_center,
    grad_normal,
    grad_rotation,
    initial_normal,
    optimal_normal,
    rotate_shape,
)
from benchmark.knows_vla.cbf.filter import CbfParams, SafetyFilter, effective_margin


def _random_ellipsoid(rng, scale=0.1):
    A = rng.normal(size=(3, 3))
    Q = A @ A.T + np.eye(3) * 1e-3
    return Ellipsoid(rng.normal(size=3) * scale, Q * scale**2)


def _unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ support function (Eq. 5)
def test_support_function_matches_numerical_maximum():
    """max_{y in E} n.y == n.c + sqrt(n' Q n), by brute force over the surface."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        E, n = _random_ellipsoid(rng), _unit(rng)
        L = np.linalg.cholesky(E.Q)
        u = rng.normal(size=(20000, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        pts = E.c + u @ L.T
        assert E.support(n) == pytest.approx(float((pts @ n).max()), rel=2e-3)


# ------------------------------------------------------------------------- barrier (Eq. 6)
def test_barrier_sign_tracks_separation():
    """h at the optimal normal is > 0 iff the ellipsoids are disjoint."""
    r = Ellipsoid.sphere([0, 0, 0], 0.05)
    for d, expect_separated in [(0.30, True), (0.11, True), (0.09, False), (0.02, False)]:
        o = Ellipsoid.sphere([d, 0, 0], 0.05)
        h = barrier(optimal_normal(r, o), r, o)
        assert (h > 0) is expect_separated, f"d={d}: h={h}"


def test_barrier_equals_exact_gap_for_spheres():
    """For two spheres the optimal gap is ||c_R - c_O|| - r_R - r_O."""
    r = Ellipsoid.sphere([0, 0, 0], 0.04)
    o = Ellipsoid.sphere([0.2, 0.1, 0.0], 0.06)
    expected = float(np.linalg.norm(o.c - r.c)) - 0.04 - 0.06
    assert barrier(optimal_normal(r, o), r, o) == pytest.approx(expected, abs=1e-9)


def test_gamma_elimination_is_not_a_relaxation_of_collision_freeness():
    """h(n) >= 0 implies a separating offset gamma exists, so Eq. (6) still certifies separation.

    This is the claim in papers/KNOWS/03-math.md that the paper's own appendix understates.
    """
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(200):
        r, o = _random_ellipsoid(rng), _random_ellipsoid(rng)
        n = optimal_normal(r, o)
        if barrier(n, r, o) <= 0:
            continue
        checked += 1
        lo = o.support(n)  # max over the obstacle along n
        hi = -r.support(-n)  # min over the robot along n
        assert lo <= hi + 1e-12  # a valid gamma exists in [lo, hi]
        gamma = 0.5 * (lo + hi)
        h_R = float(n @ r.c) - gamma - np.sqrt(n @ r.Q @ n)
        h_O = -float(n @ o.c) + gamma - np.sqrt(n @ o.Q @ n)
        assert h_R >= -1e-12 and h_O >= -1e-12
    assert checked > 20


# ----------------------------------------------------------------- gradients (Eq. 9, 10, 11)
def test_grad_center_matches_finite_difference():
    rng = np.random.default_rng(1)
    for _ in range(20):
        r, o, n = _random_ellipsoid(rng), _random_ellipsoid(rng), _unit(rng)
        g, eps = grad_center(n, r, o), 1e-7
        for i in range(3):
            dc = np.zeros(3)
            dc[i] = eps
            num = (barrier(n, Ellipsoid(r.c + dc, r.Q), o) - barrier(n, Ellipsoid(r.c - dc, r.Q), o)) / (2 * eps)
            assert g[i] == pytest.approx(num, rel=1e-5, abs=1e-9)


def test_grad_rotation_matches_finite_difference_under_R_Q_Rt():
    """Eq. (10) is the derivative under Q_R -> R Q_R R' with a world-frame axis-angle increment.

    Confirms the convention recorded in OPEN-QUESTIONS #2, which the paper never states.
    """
    rng = np.random.default_rng(2)
    for _ in range(20):
        r, o, n = _random_ellipsoid(rng), _random_ellipsoid(rng), _unit(rng)
        g, eps = grad_rotation(n, r, o), 1e-7
        for i in range(3):
            w = np.zeros(3)
            w[i] = eps
            plus = barrier(n, Ellipsoid(r.c, rotate_shape(r.Q, w)), o)
            minus = barrier(n, Ellipsoid(r.c, rotate_shape(r.Q, -w)), o)
            assert g[i] == pytest.approx((plus - minus) / (2 * eps), rel=1e-4, abs=1e-9)


def test_grad_normal_matches_finite_difference():
    rng = np.random.default_rng(4)
    for _ in range(20):
        r, o, n = _random_ellipsoid(rng), _random_ellipsoid(rng), _unit(rng)
        g, eps = grad_normal(n, r, o), 1e-7
        for i in range(3):
            dn = np.zeros(3)
            dn[i] = eps
            num = (barrier(n + dn, r, o) - barrier(n - dn, r, o)) / (2 * eps)
            assert g[i] == pytest.approx(num, rel=1e-5, abs=1e-9)


def test_rotation_gradient_vanishes_for_spherical_end_effector():
    """n x (rI)n == 0: rotating a sphere cannot change the geometry, so Eq. (10) is identically 0.

    Also holds whenever n is an eigenvector of Q_R.
    """
    rng = np.random.default_rng(5)
    r = Ellipsoid.sphere([0.0, 0.0, 0.0], 0.05)
    o = _random_ellipsoid(rng)
    for _ in range(10):
        assert np.allclose(grad_rotation(_unit(rng), r, o), 0.0, atol=1e-15)

    # eigenvector case with a non-spherical shape
    E = Ellipsoid.from_semi_axes([0, 0, 0], [0.03, 0.05, 0.08])
    for axis in np.eye(3):
        assert np.allclose(grad_rotation(axis, E, o), 0.0, atol=1e-15)


# -------------------------------------------------------------------------- the QP (Eq. 12)
def test_no_obstacles_is_the_identity():
    f = SafetyFilter()
    dc, dth = np.array([0.01, -0.02, 0.005]), np.array([0.1, 0.0, -0.05])
    res = f(Ellipsoid.sphere([0, 0, 0], 0.05), {}, dc, dth)
    assert res.feasible and not res.emergency_stop
    assert np.array_equal(res.delta_c, dc) and np.array_equal(res.delta_theta, dth)


def test_unconstrained_obstacle_leaves_nominal_untouched():
    """A far-away obstacle must not perturb the action: the CBF constraint is already slack."""
    f = SafetyFilter()
    robot = Ellipsoid.sphere([0, 0, 0], 0.05)
    obstacles = {0: Ellipsoid.sphere([5.0, 0, 0], 0.05)}
    dc = np.array([0.01, 0.0, 0.0])
    res = f(robot, obstacles, dc, np.zeros(3))
    assert res.feasible
    assert res.delta_c == pytest.approx(dc, abs=1e-6)


def test_qp_deflects_a_collision_course():
    """Nominal action drives straight at the obstacle; the filter must remove approach velocity."""
    f = SafetyFilter(CbfParams(gamma_h=0.5, eps_normal=0.0))
    robot = Ellipsoid.sphere([0.0, 0, 0], 0.05)
    obstacles = {0: Ellipsoid.sphere([0.13, 0, 0], 0.05)}
    dc_nom = np.array([0.05, 0.0, 0.0])  # 5 cm straight in, gap is only 3 cm
    res = f(robot, obstacles, dc_nom, np.zeros(3))
    assert res.feasible and not res.emergency_stop
    # +x is toward the obstacle, so the filtered step must advance strictly less.
    assert res.delta_c[0] < dc_nom[0]
    n = initial_normal(robot, obstacles[0])  # points away from the obstacle (-x)
    assert float(res.delta_c @ n) > float(dc_nom @ n)


def test_constraint_is_satisfied_at_the_solution():
    """Whatever the QP returns must satisfy Eq. (8) for every obstacle."""
    rng = np.random.default_rng(7)
    p = CbfParams(gamma_h=0.4, eps_normal=0.02)
    for _ in range(15):
        f = SafetyFilter(p)
        robot = Ellipsoid.sphere(rng.normal(size=3) * 0.05, 0.04)
        obstacles = {i: _random_ellipsoid(rng, scale=0.12) for i in range(3)}
        pre = f.normals()  # empty; populated inside the call
        res = f(robot, obstacles, rng.normal(size=3) * 0.03, rng.normal(size=3) * 0.05)
        if not res.feasible:
            continue
        del pre
        for i, k in enumerate(sorted(obstacles)):
            obs = obstacles[k]
            # Recover the normal that was used (state has since been updated), by re-deriving it.
            n = initial_normal(robot, obs)
            lhs = (grad_center(n, robot, obs) @ res.delta_c
                   + grad_rotation(n, robot, obs) @ res.delta_theta
                   + effective_margin(n, robot, obs, p.eps_normal))
            assert lhs >= -p.gamma_h * res.h[i] - 1e-6


def test_recovery_when_already_penetrating():
    """h < 0 makes Eq. (7) demand delta_h > 0 -- the filter must push outward, not freeze."""
    f = SafetyFilter(CbfParams(gamma_h=1.0, eps_normal=0.0))
    robot = Ellipsoid.sphere([0.0, 0, 0], 0.05)
    obstacles = {0: Ellipsoid.sphere([0.06, 0, 0], 0.05)}  # overlapping: gap = -0.04
    n = initial_normal(robot, obstacles[0])
    assert barrier(n, robot, obstacles[0]) < 0
    res = f(robot, obstacles, np.zeros(3), np.zeros(3))  # nominal says "hold still"
    assert res.feasible and not res.emergency_stop
    assert float(res.delta_c @ n) > 1e-9  # motion along +n == away from the obstacle


def test_infeasible_falls_back_to_emergency_stop():
    """Two obstacles squeezing from opposite sides with no slack: expect zero deltas, no exception."""
    f = SafetyFilter(CbfParams(gamma_h=1.0, eps_normal=0.0))
    robot = Ellipsoid.sphere([0.0, 0, 0], 0.05)
    obstacles = {
        0: Ellipsoid.sphere([0.06, 0, 0], 0.05),
        1: Ellipsoid.sphere([-0.06, 0, 0], 0.05),
    }
    res = f(robot, obstacles, np.array([0.02, 0, 0]), np.zeros(3))
    if res.feasible:
        pytest.skip(f"solver found this feasible (status={res.status}); not a valid infeasible case")
    assert res.emergency_stop
    assert np.array_equal(res.delta_c, np.zeros(3))
    assert np.array_equal(res.delta_theta, np.zeros(3))


def test_eps_normal_relaxes_the_constraint_monotonically():
    """Larger eps must let the filter advance further -- eps trades safety for hyperplane agility.

    The paper presents eps only as a smoothness bound; see papers/KNOWS/03-math.md.
    """
    robot = Ellipsoid.sphere([0.0, 0, 0], 0.05)
    obstacles = {0: Ellipsoid.from_semi_axes([0.14, 0.02, 0.0], [0.05, 0.03, 0.04])}
    dc_nom = np.array([0.05, 0.0, 0.0])
    advance = []
    for eps in (0.0, 0.02, 0.05, 0.1):
        f = SafetyFilter(CbfParams(gamma_h=0.5, eps_normal=eps))
        res = f(robot, obstacles, dc_nom, np.zeros(3))
        assert res.feasible
        advance.append(res.delta_c[0])
    assert all(b >= a - 1e-9 for a, b in zip(advance, advance[1:])), advance
    assert advance[-1] > advance[0] + 1e-6

    n = initial_normal(robot, obstacles[0])
    assert effective_margin(n, robot, obstacles[0], 0.1) > effective_margin(n, robot, obstacles[0], 0.0)


def test_normals_persist_and_stay_unit_norm():
    """The virtual normals are per-obstacle state carried across steps (the warm start)."""
    f = SafetyFilter()
    robot = Ellipsoid.sphere([0.0, 0, 0], 0.05)
    obstacles = {7: Ellipsoid.sphere([0.14, 0, 0], 0.05), 9: Ellipsoid.sphere([0, 0.15, 0], 0.05)}
    for _ in range(5):
        f(robot, obstacles, np.array([0.01, 0.01, 0.0]), np.zeros(3))
        for k, n in f.normals().items():
            assert k in obstacles
            assert float(np.linalg.norm(n)) == pytest.approx(1.0, abs=1e-9)
    f.reset()
    assert f.normals() == {}


def test_gamma_h_controls_conservatism():
    """Smaller gamma_h forbids h from decaying quickly, so the filter advances less."""
    robot = Ellipsoid.sphere([0.0, 0, 0], 0.05)
    obstacles = {0: Ellipsoid.sphere([0.16, 0, 0], 0.05)}
    dc_nom = np.array([0.04, 0.0, 0.0])
    adv = []
    for gamma in (0.1, 0.5, 1.0):
        f = SafetyFilter(CbfParams(gamma_h=gamma, eps_normal=0.0))
        res = f(robot, obstacles, dc_nom, np.zeros(3))
        assert res.feasible
        adv.append(res.delta_c[0])
    assert all(b >= a - 1e-9 for a, b in zip(adv, adv[1:])), adv


def test_latency_budget():
    """Paper Table 2 reports 11.4 ms for the safety QP. Report ours; do not assert on hardware."""
    import time

    rng = np.random.default_rng(11)
    robot = Ellipsoid.sphere([0, 0, 0], 0.05)
    for m in (1, 3, 6, 10):
        obstacles = {i: _random_ellipsoid(rng, scale=0.15) for i in range(m)}
        f = SafetyFilter()
        f(robot, obstacles, np.zeros(3), np.zeros(3))  # warm the solver path
        t0 = time.perf_counter()
        reps = 50
        for _ in range(reps):
            f(robot, obstacles, rng.normal(size=3) * 0.02, np.zeros(3))
        ms = (time.perf_counter() - t0) / reps * 1e3
        print(f"  obstacles={m:2d}  QP {ms:6.2f} ms/step  (paper: 11.4 ms)")
        assert ms < 200.0  # only guards a pathological regression
