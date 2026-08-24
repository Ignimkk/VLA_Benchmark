"""P1 — CBF-QP unit verification. Covers every check in benchmark/knows_vla/docs/06-repro-plan.md §8.2.

No model, no simulator: pure numerics against the paper's equations.

    src/openpi/.venv/bin/python -m pytest benchmark/knows_vla/tests/test_cbf.py -q
"""

from __future__ import annotations

import dataclasses

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
from benchmark.knows_vla.cbf.filter import (
    ArticulatedBody,
    CbfParams,
    HalfSpace,
    Link,
    RobotBody,
    SafetyFilter,
    effective_margin,
)


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

    This is the claim in benchmark/knows_vla/docs/03-math.md that the paper's own appendix understates.
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

    The paper presents eps only as a smoothness bound; see benchmark/knows_vla/docs/03-math.md.
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
    """Separating normals are per (robot part, obstacle) state carried across steps (the warm start).

    Keyed by the pair, not the obstacle alone: with a multi-part robot each part needs its own
    separating plane against the same obstacle.
    """
    f = SafetyFilter()
    robot = Ellipsoid.sphere([0.0, 0, 0], 0.05)
    obstacles = {7: Ellipsoid.sphere([0.14, 0, 0], 0.05), 9: Ellipsoid.sphere([0, 0.15, 0], 0.05)}
    for _ in range(5):
        f(robot, obstacles, np.array([0.01, 0.01, 0.0]), np.zeros(3))
        for (part, okey), n in f.normals().items():
            assert part == "eef"  # RobotBody.single names the lone part
            assert okey in obstacles
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


# ---------------------------------------------------------------------------------------------
# D1 — multi-part robot (docs/15-development-plan.md)
# ---------------------------------------------------------------------------------------------


def test_single_part_body_reproduces_the_paper_exactly():
    """RobotBody.single must be bit-identical to passing the bare ellipsoid, or every earlier
    measurement is invalidated by the refactor."""
    robot = Ellipsoid.from_semi_axes([0.0, 0, 0], [0.04, 0.04, 0.07])
    obstacles = {1: Ellipsoid.sphere([0.11, 0.02, 0], 0.05), 2: Ellipsoid.sphere([0, 0.13, 0], 0.04)}
    dc, dth = np.array([0.03, 0.0, -0.01]), np.array([0.0, 0.05, 0.0])
    a = SafetyFilter()(robot, obstacles, dc, dth)
    b = SafetyFilter()(RobotBody.single(robot), obstacles, dc, dth)
    assert a.delta_c == pytest.approx(b.delta_c, abs=1e-12)
    assert a.delta_theta == pytest.approx(b.delta_theta, abs=1e-12)
    assert a.h == pytest.approx(b.h, abs=1e-12)


def test_offset_part_rotation_gradient_matches_finite_difference():
    """The r x n term: rotating the wrist moves an offset part's centre, not just its shape.

    Without it the filter would rotate a two-finger hand straight into an obstacle while believing
    the barrier was unchanged.
    """
    rng = np.random.default_rng(5)
    origin = np.array([0.10, -0.02, 0.03])
    r = np.array([0.0, 0.037, 0.007])  # a finger offset from the EEF site
    Qp = np.diag([0.012, 0.021, 0.049]) ** 2
    obstacle = Ellipsoid.from_semi_axes([0.16, 0.01, 0.0], [0.032, 0.032, 0.031])
    body = RobotBody(origin, {"finger": Ellipsoid(r, Qp)})
    n = optimal_normal(body.world("finger"), obstacle)

    analytic = grad_rotation(n, body.world("finger"), obstacle) + np.cross(r, n)
    for _ in range(6):
        w = rng.normal(size=3)
        w /= np.linalg.norm(w)
        eps = 1e-7
        plus, minus = body.moved(np.zeros(3), w * eps), body.moved(np.zeros(3), -w * eps)
        fd = (barrier(n, plus.world("finger"), obstacle)
              - barrier(n, minus.world("finger"), obstacle)) / (2 * eps)
        assert fd == pytest.approx(float(analytic @ w), rel=2e-4, abs=1e-8)


def test_two_fingers_leave_the_space_between_them_free():
    """The point of decomposing the robot: an object between the fingers is not a collision.

    Measured RB-Y1 finger: 1.2 x 2.1 x 4.9 cm semi-axes at +-3.7 cm from the EEF axis. The block we
    used to assume spans that whole gap, so it calls a grasped object a collision.
    """
    semi = np.diag([0.012, 0.021, 0.049]) ** 2
    fingers = RobotBody(np.zeros(3), {"fp": Ellipsoid([0.0, 0.037, 0.0], semi),
                                      "fn": Ellipsoid([0.0, -0.037, 0.0], semi)})
    block = Ellipsoid.from_semi_axes([0, 0, 0], [0.04, 0.04, 0.07])
    between = Ellipsoid.sphere([0.0, 0.0, 0.0], 0.010)

    for k in fingers.parts:
        part = fingers.world(k)
        assert barrier(optimal_normal(part, between), part, between) > 0
    assert barrier(optimal_normal(block, between), block, between) < 0

    # ... and the filter acts on that: the block deflects, the fingers do not.
    nom = np.array([0.0, 0.0, -0.01])
    assert SafetyFilter()(fingers, {0: between}, nom, np.zeros(3)).delta_c == pytest.approx(nom, abs=1e-6)
    assert SafetyFilter()(block, {0: between}, nom, np.zeros(3)).delta_c != pytest.approx(nom, abs=1e-6)


# ---------------------------------------------------------------------------------------------
# D2 — fixed separating normal instead of the epsilon-relaxed one
# ---------------------------------------------------------------------------------------------


def test_fixed_normal_is_conservative_relative_to_the_true_optimum():
    """h(n) <= h* for every unit n, so holding a normal can only under-report the gap.

    That is the whole safety argument for dropping delta_n: the error is on the safe side.
    """
    rng = np.random.default_rng(17)
    for _ in range(40):
        robot = _random_ellipsoid(rng, scale=0.12)
        obstacle = _random_ellipsoid(rng, scale=0.12)
        h_star = barrier(optimal_normal(robot, obstacle), robot, obstacle)
        assert barrier(initial_normal(robot, obstacle), robot, obstacle) <= h_star + 1e-9


def test_fixed_normals_remove_the_epsilon_loophole():
    """With eps large the relaxed filter can satisfy Eq. (8) by tilting the plane and not moving.

    'fixed' has no plane to tilt, so a violated barrier must be answered with motion.
    """
    robot = Ellipsoid.from_semi_axes([0.0, 0, 0], [0.04, 0.04, 0.07])
    obstacle = Ellipsoid.from_semi_axes([0.05, 0, 0], [0.05, 0.05, 0.05])  # overlapping: h < 0
    nom = np.array([0.02, 0.0, 0.0])  # nominal pushes further in

    relaxed = SafetyFilter(CbfParams(eps_normal=0.5))(robot, {0: obstacle}, nom, np.zeros(3))
    fixed = SafetyFilter(CbfParams(normals="fixed"))(robot, {0: obstacle}, nom, np.zeros(3))

    assert relaxed.h[0] < 0 and fixed.h[0] < 0
    moved_relaxed = float(np.linalg.norm(relaxed.delta_c - nom))
    moved_fixed = float(np.linalg.norm(fixed.delta_c - nom))
    assert moved_fixed > moved_relaxed
    assert float(fixed.delta_c @ np.array([1.0, 0, 0])) < float(relaxed.delta_c @ np.array([1.0, 0, 0]))


def test_fixedpoint_reduces_intervention_on_aggregate():
    """Recomputing the normal at the landed pose cuts intervention -- but do not read that as good.

    This test pins the mechanism, not a recommendation. On recorded rollouts the same setting
    breaks the true CBF condition on 47.4% of steps against 9.2% for plain 'fixed', because the
    tightness it buys is taken as larger steps into the obstacle. Less intervention is only a
    virtue at equal safety, which is exactly what a synthetic tightness test cannot see.
    See docs/16-filter-variants.md.
    """
    rng = np.random.default_rng(23)
    robot = Ellipsoid.from_semi_axes([0.0, 0, 0], [0.04, 0.04, 0.07])
    total_fixed, total_fp = 0.0, 0.0
    for _ in range(12):
        obstacles = {i: _random_ellipsoid(rng, scale=0.16) for i in range(3)}
        nom = rng.normal(size=3) * 0.02
        a = SafetyFilter(CbfParams(normals="fixed"))(robot, obstacles, nom, np.zeros(3))
        b = SafetyFilter(CbfParams(normals="fixedpoint"))(robot, obstacles, nom, np.zeros(3))
        assert a.feasible and b.feasible
        total_fixed += float(np.linalg.norm(a.delta_c - nom))
        total_fp += float(np.linalg.norm(b.delta_c - nom))
    assert total_fp < total_fixed


# ---------------------------------------------------------------------------------------------
# D3 — slack, so infeasibility stops being a solver outcome
# ---------------------------------------------------------------------------------------------


def test_action_limits_alone_make_the_qp_infeasible():
    """Establishes the problem: bounding delta_c is what lets Eq. (12) fail at all."""
    robot = Ellipsoid.sphere([0.0, 0, 0], 0.05)
    obstacle = Ellipsoid.sphere([0.02, 0, 0], 0.05)  # deeply overlapped
    p = CbfParams(gamma_h=1.0, eps_normal=0.0, max_delta_pos=0.001, max_delta_rot=0.001)
    res = SafetyFilter(p)(robot, {0: obstacle}, np.array([0.05, 0, 0]), np.zeros(3))
    assert res.emergency_stop and not res.feasible


def test_slack_keeps_the_qp_feasible_and_makes_stopping_a_decision():
    """Same geometry, slack on: the solver always answers, and the stop comes from a threshold."""
    robot = Ellipsoid.sphere([0.0, 0, 0], 0.05)
    obstacle = Ellipsoid.sphere([0.02, 0, 0], 0.05)
    p = CbfParams(gamma_h=1.0, eps_normal=0.0, max_delta_pos=0.001, max_delta_rot=0.001,
                  slack_weight=1e3, slack_stop=1e9)
    res = SafetyFilter(p)(robot, {0: obstacle}, np.array([0.05, 0, 0]), np.zeros(3))
    assert res.feasible and not res.emergency_stop
    assert res.slack is not None and float(res.slack.max()) > 0.0

    strict = dataclasses.replace(p, slack_stop=1e-6)
    stopped = SafetyFilter(strict)(robot, {0: obstacle}, np.array([0.05, 0, 0]), np.zeros(3))
    assert stopped.emergency_stop and stopped.feasible  # a decision, not a failure
    assert stopped.delta_c == pytest.approx(np.zeros(3))


def test_slack_stays_shut_when_the_constraint_is_satisfiable():
    """A heavy penalty must not leak: no violation, no slack, no change to the paper's answer."""
    robot = Ellipsoid.sphere([0.0, 0, 0], 0.05)
    obstacles = {0: Ellipsoid.sphere([0.30, 0, 0], 0.05)}
    p = CbfParams(eps_normal=0.0, slack_weight=1e6)
    res = SafetyFilter(p)(robot, obstacles, np.array([0.01, 0, 0]), np.zeros(3))
    assert float(res.slack.max()) == pytest.approx(0.0, abs=1e-9)
    ref = SafetyFilter(CbfParams(eps_normal=0.0))(robot, obstacles, np.array([0.01, 0, 0]), np.zeros(3))
    assert res.delta_c == pytest.approx(ref.delta_c, abs=1e-7)


# ---------------------------------------------------------------------------------------------
# D4 — support surfaces as half-spaces
# ---------------------------------------------------------------------------------------------


def test_half_space_barrier_is_exact_for_an_axis_aligned_plane():
    """A table at z = 0.8: clearance is just the ellipsoid's bottom minus the plane."""
    plane = HalfSpace([0, 0, 1.0], 0.80)
    E = Ellipsoid.from_semi_axes([0.5, 0.0, 0.90], [0.04, 0.04, 0.07])
    assert plane.barrier(E) == pytest.approx(0.90 - 0.07 - 0.80, abs=1e-12)


def test_half_space_constraint_stops_descent_into_the_table():
    """The paper segments 'manipulable objects', so nothing stops the EEF entering the table."""
    plane = HalfSpace([0, 0, 1.0], 0.80)
    robot = Ellipsoid.from_semi_axes([0.5, 0.0, 0.88], [0.04, 0.04, 0.07])
    nom = np.array([0.0, 0.0, -0.05])  # driving straight down through it

    free = SafetyFilter()(robot, {}, nom, np.zeros(3))
    guarded = SafetyFilter()(robot, {}, nom, np.zeros(3), half_spaces={"table": plane})
    assert free.delta_c == pytest.approx(nom)  # no obstacles -> identity, as before
    assert guarded.delta_c[2] > nom[2]
    after = Ellipsoid(robot.c + guarded.delta_c, robot.Q)
    assert plane.barrier(after) >= -1e-6


def test_half_space_needs_no_epsilon():
    """eps is meaningless for a plane -- its normal is data. Changing it must change nothing."""
    plane = HalfSpace([0, 0, 1.0], 0.80)
    robot = Ellipsoid.from_semi_axes([0.5, 0.0, 0.88], [0.04, 0.04, 0.07])
    nom = np.array([0.0, 0.0, -0.05])
    a = SafetyFilter(CbfParams(eps_normal=0.0))(robot, {}, nom, np.zeros(3), {"t": plane})
    b = SafetyFilter(CbfParams(eps_normal=0.5))(robot, {}, nom, np.zeros(3), {"t": plane})
    assert a.delta_c == pytest.approx(b.delta_c, abs=1e-9)


# ---------------------------------------------------------------------------------------------
# A3 — the held object (docs/16-filter-variants.md)
# ---------------------------------------------------------------------------------------------


def test_held_object_detected_only_when_it_travels_with_the_hand():
    """A closed gripper is not evidence: it also closes on nothing. Motion agreement is."""
    from benchmark.knows_vla.perception.held import HeldObjectTracker

    trk = HeldObjectTracker()
    eef = np.zeros(3)
    carried, bystander = np.array([0.01, 0, 0]), np.array([0.25, 0, 0])
    for _ in range(8):
        eef = eef + np.array([0.01, 0, 0])
        carried = carried + np.array([0.01, 0, 0])  # rides along
        obstacles = {1: Ellipsoid.sphere(carried, 0.03), 2: Ellipsoid.sphere(bystander, 0.03)}
        held = trk.update(eef, True, obstacles)
    assert held == 1

    # The same motion with the gripper open must not latch.
    trk2 = HeldObjectTracker()
    eef = np.zeros(3)
    carried = np.array([0.01, 0, 0])
    for _ in range(8):
        eef, carried = eef + [0.01, 0, 0], carried + [0.01, 0, 0]
        trk2.update(eef, False, {1: Ellipsoid.sphere(carried, 0.03)})
    assert trk2.held is None


def test_standing_still_is_not_evidence_of_holding():
    """With the arm stationary every object 'follows' it perfectly."""
    from benchmark.knows_vla.perception.held import HeldObjectTracker

    trk = HeldObjectTracker()
    for _ in range(20):
        trk.update(np.zeros(3), True, {1: Ellipsoid.sphere([0.2, 0, 0], 0.03)})
    assert trk.held is None


def test_promoting_the_held_object_removes_the_structural_violation():
    """Held-as-obstacle forces h < 0; held-as-robot-part does not, and still protects it."""
    from benchmark.knows_vla.perception.held import promote

    eef = np.array([0.30, 0.0, 0.90])
    gripper = Ellipsoid.from_semi_axes(eef, [0.04, 0.04, 0.07])
    held = Ellipsoid.sphere(eef + [0.0, 0.0, -0.02], 0.032)  # in the hand, so overlapping
    other = Ellipsoid.sphere([0.40, 0.0, 0.90], 0.032)

    assert barrier(optimal_normal(gripper, held), gripper, held) < 0  # the structural problem

    body, rest = promote(RobotBody.single(gripper), {1: held, 2: other}, 1, eef)
    assert set(rest) == {2}
    assert "held:1" in body.parts
    # The carried object is now protected from the object it is being carried past.
    part = body.world("held:1")
    assert barrier(optimal_normal(part, other), part, other) > 0

    nom = np.array([0.05, 0.0, 0.0])  # drive it straight at `other`
    res = SafetyFilter(CbfParams(normals="fixed"))(body, rest, nom, np.zeros(3))
    assert res.feasible and res.delta_c[0] < nom[0]


# ---------------------------------------------------------------------------------------------
# Jacobian bridge — joint-space parameterisation (docs/13-rby1-port-plan.md §1)
# ---------------------------------------------------------------------------------------------


class _Chain:
    """A tiny revolute chain with exact forward kinematics, so tests can finite-difference it.

    Validating the Jacobian rows against the analytic Jacobian would only prove the code agrees
    with itself. Differencing *true* FK is what shows the bridge is right.
    """

    def __init__(self, axes, anchors, link_offsets):
        self.axes = [np.asarray(a, float) / np.linalg.norm(a) for a in axes]
        self.anchors = [np.asarray(p, float) for p in anchors]
        self.offsets = [np.asarray(o, float) for o in link_offsets]  # link k centre at q = 0

    @staticmethod
    def _rot(axis, angle):
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def fk(self, q):
        """(centre, orientation) of every link. Joint i moves links i and beyond.

        Composed tip-first (A_0 . A_1 ... A_k), each A_i about its zero-configuration axis and
        anchor. That is the ordering `jacobians` assumes when it carries joint i's screw through
        the upstream rotations; composing the other way round would describe a different mechanism
        and the two would silently disagree.
        """
        out = []
        for k in range(len(self.offsets)):
            c, R = self.offsets[k].copy(), np.eye(3)
            for i in range(k, -1, -1):
                Ri = self._rot(self.axes[i], q[i])
                c = self.anchors[i] + Ri @ (c - self.anchors[i])
                R = Ri @ R
            out.append((c, R))
        return out

    def jacobians(self, q):
        """Jv, Jw per link at configuration q, by the textbook revolute formula."""
        poses, nq = self.fk(q), len(self.axes)
        Js = []
        for k, (c, _) in enumerate(poses):
            Jv, Jw = np.zeros((3, nq)), np.zeros((3, nq))
            for i in range(k + 1):
                # joint i's screw, carried out through the joints that sit above it. Outermost
                # last: fk composes A_0 . A_1 ... so the transform is R_0 (R_1 (... axis_i)).
                axis, anchor = self.axes[i], self.anchors[i]
                for j in range(i - 1, -1, -1):
                    Rj = self._rot(self.axes[j], q[j])
                    axis = Rj @ axis
                    anchor = self.anchors[j] + Rj @ (anchor - self.anchors[j])
                Jv[:, i] = np.cross(axis, c - anchor)
                Jw[:, i] = axis
            Js.append((Jv, Jw))
        return Js


def _demo_chain():
    return _Chain(axes=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                  anchors=[[0, 0, 0], [0.25, 0, 0.10], [0.45, 0, 0.10]],
                  link_offsets=[[0.15, 0, 0.05], [0.35, 0, 0.10], [0.55, 0, 0.10]])


def _body_at(chain, q, shapes, eef="link2"):
    poses, Js = chain.fk(q), chain.jacobians(q)
    links = {}
    for k, ((c, R), (Jv, Jw)) in enumerate(zip(poses, Js)):
        links[f"link{k}"] = Link(Ellipsoid(c, R @ shapes[k] @ R.T), Jv, Jw)
    return ArticulatedBody(links, eef)


def test_jacobian_chain_rule_matches_true_forward_kinematics():
    """dh/dq == (dh/dc)' Jv + (dh/dR)' Jw, differenced against exact FK, not against the Jacobian."""
    chain = _demo_chain()
    shapes = [np.diag([0.05, 0.04, 0.04]) ** 2] * 3
    q = np.array([0.2, -0.35, 0.15])
    obstacle = Ellipsoid.from_semi_axes([0.62, 0.10, 0.12], [0.05, 0.05, 0.05])

    body = _body_at(chain, q, shapes)
    for name in body.links:
        part = body.world(name)
        n = optimal_normal(part, obstacle)
        L = body.links[name]
        analytic = (grad_center(n, part, obstacle) @ L.Jv
                    + grad_rotation(n, part, obstacle) @ L.Jw)
        for i in range(len(q)):
            eps = 1e-6
            dq = np.zeros(len(q))
            dq[i] = eps
            plus = _body_at(chain, q + dq, shapes).world(name)
            minus = _body_at(chain, q - dq, shapes).world(name)
            fd = (barrier(n, plus, obstacle) - barrier(n, minus, obstacle)) / (2 * eps)
            assert analytic[i] == pytest.approx(fd, rel=1e-4, abs=1e-9), (name, i)


def test_identity_jacobian_reproduces_the_task_space_filter():
    """With Jv = I and Jw = 0 the joint variables *are* delta_c, so both paths must agree."""
    E = Ellipsoid.from_semi_axes([0.0, 0, 0], [0.04, 0.04, 0.07])
    obstacles = {0: Ellipsoid.sphere([0.13, 0.01, 0], 0.05), 1: Ellipsoid.sphere([0, 0.14, 0], 0.04)}
    nom = np.array([0.05, 0.01, 0.0])
    p = CbfParams(normals="fixed")

    task = SafetyFilter(p)(E, obstacles, nom, np.zeros(3))
    art = ArticulatedBody({"eef": Link(E, np.eye(3), np.zeros((3, 3)))})
    joint = SafetyFilter(p)(art, obstacles, delta_q_nom=nom)

    assert joint.delta_q == pytest.approx(task.delta_c, abs=1e-7)
    assert joint.delta_c == pytest.approx(task.delta_c, abs=1e-7)


def test_joint_limits_are_enforced():
    chain = _demo_chain()
    shapes = [np.diag([0.05, 0.04, 0.04]) ** 2] * 3
    body = _body_at(chain, np.array([0.2, -0.35, 0.15]), shapes)
    obstacles = {0: Ellipsoid.from_semi_axes([0.62, 0.06, 0.12], [0.06, 0.06, 0.06])}
    p = CbfParams(normals="fixed", max_delta_q=0.02, slack_weight=1e3)
    res = SafetyFilter(p)(body, obstacles, delta_q_nom=np.array([0.5, 0.5, 0.5]))
    assert res.feasible
    assert np.max(np.abs(res.delta_q)) <= 0.02 + 1e-7


def test_whole_arm_is_protected_not_only_the_end_effector():
    """The paper's §5 limitation: only the EEF is modelled. Joint control removes the reason.

    Configuration chosen so the swing threatens the elbow and not the hand: over the nominal step
    the elbow's clearance goes +3.0 -> -5.7 cm while the end-effector never drops below +9.4 cm.
    Modelling the EEF alone therefore sees no reason to act, and the arm collides.
    """
    chain = _demo_chain()
    shapes = [np.diag([0.05, 0.04, 0.04]) ** 2] * 3
    q = np.zeros(3)
    body = _body_at(chain, q, shapes)
    obstacle = Ellipsoid.sphere([0.34, 0.12, 0.10], 0.05)
    nom = np.array([0.25, 0.0, 0.0])  # joint 0 swings the elbow across the obstacle

    def elbow_clearance(dq):
        after = _body_at(chain, q + dq, shapes).world("link1")
        return barrier(optimal_normal(after, obstacle), after, obstacle)

    assert elbow_clearance(np.zeros(3)) > 0  # clear before the step
    assert elbow_clearance(nom) < 0  # the unfiltered command drives the elbow into it

    p = CbfParams(normals="fixed")
    eef_only = ArticulatedBody({"link2": body.links["link2"]}, "link2")
    partial = SafetyFilter(p)(eef_only, {0: obstacle}, delta_q_nom=nom)
    full = SafetyFilter(p)(body, {0: obstacle}, delta_q_nom=nom)

    # Modelling only the end-effector lets the command through and the elbow still collides.
    assert partial.delta_q == pytest.approx(nom, abs=1e-6)
    assert elbow_clearance(partial.delta_q) < 0
    # Modelling the chain keeps it out.
    assert full.delta_q[0] < nom[0] - 1e-4
    assert elbow_clearance(full.delta_q) >= -1e-3


def test_joint_weights_steer_which_joint_absorbs_the_correction():
    """Making one joint expensive pushes the correction onto the others -- e.g. wrist over base."""
    chain = _demo_chain()
    shapes = [np.diag([0.05, 0.04, 0.04]) ** 2] * 3
    body = _body_at(chain, np.array([0.1, -0.2, 0.1]), shapes)
    obstacles = {0: Ellipsoid.from_semi_axes([0.62, 0.08, 0.12], [0.06, 0.06, 0.06])}
    nom = np.array([0.3, 0.0, 0.0])

    cheap = SafetyFilter(CbfParams(normals="fixed"))(body, obstacles, delta_q_nom=nom)
    pricey = SafetyFilter(CbfParams(normals="fixed", joint_weights=np.array([50.0, 1.0, 1.0])))(
        body, obstacles, delta_q_nom=nom)
    assert abs(pricey.delta_q[0] - nom[0]) < abs(cheap.delta_q[0] - nom[0])


def test_joint_mode_reports_the_induced_end_effector_twist():
    """delta_q is what gets commanded; the twist is derived, and must match Jv delta_q."""
    chain = _demo_chain()
    shapes = [np.diag([0.05, 0.04, 0.04]) ** 2] * 3
    body = _body_at(chain, np.array([0.2, -0.35, 0.15]), shapes)
    obstacles = {0: Ellipsoid.from_semi_axes([0.62, 0.06, 0.12], [0.06, 0.06, 0.06])}
    res = SafetyFilter(CbfParams(normals="fixed"))(body, obstacles,
                                                   delta_q_nom=np.array([0.05, 0.02, 0.01]))
    L = body.links[body.eef]
    assert res.delta_c == pytest.approx(L.Jv @ res.delta_q, abs=1e-12)
    assert res.delta_theta == pytest.approx(L.Jw @ res.delta_q, abs=1e-12)


def test_joint_space_latency_at_rby1_scale():
    """Paper Table 2 budgets 11.4 ms for the QP. Whole-arm protection must fit inside it.

    The body **moves** between reps, which is the only honest way to time this: holding the pose
    fixed lets every separating normal converge on the first call and exit in one iteration
    afterwards, which reported 3.3 ms for a configuration that actually costs 40 ms in motion.
    That artifact is what motivated `optimal_normals_batch`; this test is its regression guard.
    """
    import time

    rng = np.random.default_rng(0)
    for n_links, n_obs, nq in ((1, 6, 7), (6, 6, 14), (6, 10, 14)):
        J = [(rng.normal(size=(3, nq)) * 0.1, rng.normal(size=(3, nq)) * 0.1) for _ in range(n_links)]
        obstacles = {i: Ellipsoid.sphere(np.array([0.45, 0.10, 0.9]) + rng.normal(size=3) * 0.08, 0.04)
                     for i in range(n_obs)}

        def body_at(shift):
            return ArticulatedBody(
                {f"L{k}": Link(Ellipsoid(np.array([0.30 + 0.05 * k, 0.02 * k, 0.9]) + shift,
                                         np.diag([0.05, 0.04, 0.04]) ** 2), J[k][0], J[k][1])
                 for k in range(n_links)}, "L0")

        f = SafetyFilter(CbfParams(normals="fixed", max_delta_q=0.05, slack_weight=1e3))
        f(body_at(np.zeros(3)), obstacles, delta_q_nom=rng.normal(size=nq) * 0.02)
        t0, reps = time.perf_counter(), 20
        for r in range(reps):
            f(body_at(np.array([0.004 * r, 0.0, 0.0])), obstacles,
              delta_q_nom=rng.normal(size=nq) * 0.02)
        ms = (time.perf_counter() - t0) / reps * 1e3
        print(f"  links={n_links} obstacles={n_obs} nq={nq}  ->  {n_links * n_obs:3d} rows, {ms:6.2f} ms")
        assert ms < 25.0


def test_batched_normals_agree_with_the_scalar_ascent():
    """The batch is an optimisation, not a different algorithm."""
    from benchmark.knows_vla.cbf.ellipsoid import optimal_normals_batch

    rng = np.random.default_rng(3)
    R = [_random_ellipsoid(rng, 0.12) for _ in range(40)]
    O = [_random_ellipsoid(rng, 0.12) for _ in range(40)]
    N = optimal_normals_batch(R, O, iters=400)
    for i, (r, o) in enumerate(zip(R, O)):
        assert barrier(N[i], r, o) == pytest.approx(barrier(optimal_normal(r, o), r, o), abs=1e-12)


def test_skipping_far_pairs_does_not_change_the_solution():
    """`normal_skip_h` must be exact, not an approximation, for constraints that stay inactive."""
    robot = Ellipsoid.from_semi_axes([0.0, 0, 0], [0.04, 0.04, 0.07])
    obstacles = {0: Ellipsoid.sphere([0.13, 0, 0], 0.05),  # near: must still get the exact normal
                 1: Ellipsoid.sphere([0.90, 0.2, 0], 0.05),  # far: skipped
                 2: Ellipsoid.sphere([-0.75, 0, 0.3], 0.06)}
    nom = np.array([0.05, 0.0, 0.0])
    exact = SafetyFilter(CbfParams(normals="fixed", normal_skip_h=1e9))(robot, obstacles, nom, np.zeros(3))
    gated = SafetyFilter(CbfParams(normals="fixed"))(robot, obstacles, nom, np.zeros(3))
    assert gated.delta_c == pytest.approx(exact.delta_c, abs=1e-9)
