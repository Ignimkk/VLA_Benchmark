"""Separating-hyperplane CBF for ellipsoid pairs — KNOWS Eq. (5)-(11).

Convention (matches benchmark/knows_vla/docs/03-math.md):
    E = {c + Q^(1/2) u : ||u|| <= 1},  Q symmetric positive definite.
so the support function along a direction n is  max_{y in E} n.y = n.c + sqrt(n' Q n).

The end-effector ellipsoid rotates with the gripper as ``Q_R -> R Q_R R'``. That convention is
what makes Eq. (10) correct; see ``rotate_shape`` for the derivation.
"""

from __future__ import annotations

import dataclasses

import numpy as np

EPS_DENOM = 1e-12


@dataclasses.dataclass(frozen=True)
class Ellipsoid:
    """Centre and shape matrix. ``Q`` is the squared-radius matrix, not its square root."""

    c: np.ndarray  # (3,)
    Q: np.ndarray  # (3, 3) symmetric positive definite

    def __post_init__(self):
        object.__setattr__(self, "c", np.asarray(self.c, np.float64).reshape(3))
        Q = np.asarray(self.Q, np.float64).reshape(3, 3)
        object.__setattr__(self, "Q", 0.5 * (Q + Q.T))

    @staticmethod
    def sphere(c, radius: float) -> "Ellipsoid":
        return Ellipsoid(c, np.eye(3) * float(radius) ** 2)

    @staticmethod
    def from_semi_axes(c, semi_axes, R=None) -> "Ellipsoid":
        """Axis-aligned semi-axes, optionally rotated by ``R``."""
        Q = np.diag(np.asarray(semi_axes, np.float64) ** 2)
        if R is not None:
            R = np.asarray(R, np.float64)
            Q = R @ Q @ R.T
        return Ellipsoid(c, Q)

    def support(self, n) -> float:
        """max_{y in E} n.y  — the support function."""
        n = np.asarray(n, np.float64)
        return float(n @ self.c + np.sqrt(max(n @ self.Q @ n, 0.0)))


def rotate_shape(Q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Apply a world-frame rotation ``omega`` (axis-angle) to a shape matrix: Q -> R Q R'.

    Eq. (10) is the derivative of ``-sqrt(n' Q n)`` under exactly this transform. With
    R ~ I + skew(w):

        d(n' Q n) = 2 w . (Q n x n)  =>  d(-sqrt(n' Q n)) = w . (n x Q n) / sqrt(n' Q n)

    which is Eq. (10). So the paper's delta_theta is a **world-frame axis-angle increment** and the
    end-effector ellipsoid rotates rigidly with the gripper. The paper states neither
    (OPEN-QUESTIONS #2); test_cbf.py checks the identity numerically.
    """
    omega = np.asarray(omega, np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    if theta < EPS_DENOM:
        R = np.eye(3)
    else:
        k = omega / theta
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R @ Q @ R.T


def _sqrt_quad(n: np.ndarray, Q: np.ndarray) -> float:
    return float(np.sqrt(max(n @ Q @ n, 0.0)))


def barrier(n, robot: Ellipsoid, obstacle: Ellipsoid) -> float:
    """Eq. (6):  h(n) = n.(c_R - c_O) - sqrt(n' Q_R n) - sqrt(n' Q_O n).

    For a unit ``n`` this is the signed gap between the two ellipsoids measured along ``n``:
    h >= 0 certifies that the hyperplane with normal ``n`` separates them.
    """
    n = np.asarray(n, np.float64).reshape(3)
    return float(n @ (robot.c - obstacle.c)) - _sqrt_quad(n, robot.Q) - _sqrt_quad(n, obstacle.Q)


def grad_center(n, robot: Ellipsoid, obstacle: Ellipsoid) -> np.ndarray:
    """Eq. (9):  d h / d c_R = n."""
    del robot, obstacle
    return np.asarray(n, np.float64).reshape(3).copy()


def grad_rotation(n, robot: Ellipsoid, obstacle: Ellipsoid) -> np.ndarray:
    """Eq. (10):  d h / d theta = (n x Q_R n) / sqrt(n' Q_R n).

    Identically zero when ``n`` is an eigenvector of ``Q_R`` -- in particular for a spherical
    end-effector, where rotating cannot change the geometry.
    """
    del obstacle
    n = np.asarray(n, np.float64).reshape(3)
    denom = _sqrt_quad(n, robot.Q)
    if denom < EPS_DENOM:
        return np.zeros(3)
    return np.cross(n, robot.Q @ n) / denom


def grad_normal(n, robot: Ellipsoid, obstacle: Ellipsoid) -> np.ndarray:
    """Eq. (11):  d h / d n = (c_R - c_j) - Q_R n / sqrt(n' Q_R n) - Q_j n / sqrt(n' Q_j n)."""
    n = np.asarray(n, np.float64).reshape(3)
    dr, do = _sqrt_quad(n, robot.Q), _sqrt_quad(n, obstacle.Q)
    g = robot.c - obstacle.c
    if dr >= EPS_DENOM:
        g = g - robot.Q @ n / dr
    if do >= EPS_DENOM:
        g = g - obstacle.Q @ n / do
    return g


def initial_normal(robot: Ellipsoid, obstacle: Ellipsoid) -> np.ndarray:
    """Centre-to-centre direction — the initialization the paper specifies (Appendix 7.1)."""
    d = robot.c - obstacle.c
    nrm = float(np.linalg.norm(d))
    if nrm < EPS_DENOM:  # coincident centres: no direction is better than another
        return np.array([1.0, 0.0, 0.0])
    return d / nrm


def optimal_normal(robot: Ellipsoid, obstacle: Ellipsoid, *, iters: int = 400, init=None,
                   tol: float = 1e-7) -> np.ndarray:
    """Direction maximizing ``barrier`` — the tightest separating hyperplane at this pose.

    h(n) is concave in n, so projected gradient ascent from the centre-to-centre direction
    converges reliably.

    ``init`` warm-starts from the previous step's normal. The optimum moves only as far as the pose
    does, so a warm start reaches it in a handful of iterations -- which is what makes it affordable
    to drop the paper's ``delta_n`` variables and simply recompute the normal every step
    (``CbfParams.normals='fixed'``; see docs/15-development-plan.md D2).

    ``tol`` is what makes the warm start pay. Terminating only on the backtracking step decaying
    below 1e-10 costs ~65 iterations whatever the starting point, so a warm-started call was doing
    the same work as a cold one; stopping on the tangent gradient instead cuts a 6-link, 6-obstacle
    joint-space step from 39 ms to a few.
    """
    n = initial_normal(robot, obstacle) if init is None else np.asarray(init, np.float64).reshape(3)
    nrm = float(np.linalg.norm(n))
    n = n / nrm if nrm > EPS_DENOM else initial_normal(robot, obstacle)
    step = 1.0
    best_n, best_h = n.copy(), barrier(n, robot, obstacle)
    for _ in range(iters):
        g = grad_normal(n, robot, obstacle)
        g = g - n * float(g @ n)  # project onto the unit sphere's tangent
        if float(np.linalg.norm(g)) < tol:
            break  # at the maximum: h is concave, so a vanishing tangent gradient is the optimum
        cand = n + step * g
        cand = cand / max(float(np.linalg.norm(cand)), EPS_DENOM)
        h = barrier(cand, robot, obstacle)
        if h > best_h:
            best_n, best_h, n = cand.copy(), h, cand
        else:
            step *= 0.7
            if step < 1e-10:
                break
    return best_n


def optimal_normals_batch(robots, obstacles, *, inits=None, iters: int = 60,
                          tol: float = 1e-7) -> np.ndarray:
    """``optimal_normal`` for many pairs at once. Returns (m, 3).

    Same projected-gradient ascent with the same per-pair backtracking; only the loop is moved
    from Python into numpy. That matters because each iteration touches 3-vectors, so interpreter
    overhead -- not arithmetic -- dominates: protecting a whole arm means (links x obstacles)
    ascents per control step, which at ~1 ms each blows the paper's 11.4 ms budget by 6 links x 6
    obstacles. Batched, the same work is one pass of (m, 3) operations.

    Pairs that converge drop out of the active set, so a pair whose normal barely moved since the
    last step costs one iteration, not sixty.
    """
    m = len(robots)
    if m == 0:
        return np.zeros((0, 3))
    Cr = np.stack([e.c for e in robots])
    Qr = np.stack([e.Q for e in robots])
    Co = np.stack([e.c for e in obstacles])
    Qo = np.stack([e.Q for e in obstacles])
    D = Cr - Co

    if inits is None:
        N = D.copy()
    else:
        N = np.stack([D[i] if inits[i] is None else np.asarray(inits[i], np.float64)
                      for i in range(m)])
    nrm = np.linalg.norm(N, axis=1)
    fallback = np.linalg.norm(D, axis=1)
    N = np.where((nrm > EPS_DENOM)[:, None], N / np.maximum(nrm, EPS_DENOM)[:, None],
                 np.where((fallback > EPS_DENOM)[:, None], D / np.maximum(fallback, EPS_DENOM)[:, None],
                          np.array([1.0, 0.0, 0.0])))

    def _barrier(N):
        sr = np.sqrt(np.maximum(np.einsum("mi,mij,mj->m", N, Qr, N), 0.0))
        so = np.sqrt(np.maximum(np.einsum("mi,mij,mj->m", N, Qo, N), 0.0))
        return np.einsum("mi,mi->m", N, D) - sr - so, sr, so

    best_h, sr, so = _barrier(N)
    best_N = N.copy()
    step = np.ones(m)
    active = np.ones(m, bool)
    for _ in range(iters):
        G = (D - np.einsum("mij,mj->mi", Qr, N) / np.maximum(sr, EPS_DENOM)[:, None]
             - np.einsum("mij,mj->mi", Qo, N) / np.maximum(so, EPS_DENOM)[:, None])
        G = G - N * np.einsum("mi,mi->m", G, N)[:, None]  # project onto the tangent
        active &= np.linalg.norm(G, axis=1) >= tol  # converged pairs drop out
        if not active.any():
            break
        cand = N + step[:, None] * G
        cn = np.linalg.norm(cand, axis=1)
        cand = cand / np.maximum(cn, EPS_DENOM)[:, None]
        hc, scr, sco = _barrier(cand)
        better = active & (hc > best_h)
        best_N[better], best_h[better] = cand[better], hc[better]
        N[better], sr[better], so[better] = cand[better], scr[better], sco[better]
        step[active & ~better] *= 0.7
        active &= step > 1e-10
    return best_N
