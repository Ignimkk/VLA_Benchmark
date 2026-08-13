"""Discrete-time CBF-QP safety filter — KNOWS Eq. (7), (8), (12).

Projects the policy's nominal end-effector delta onto the safe set, keeping the virtual separating
hyperplane normals as per-obstacle state across steps.

Everything here is in **physical units** (metres, radians): Appendix 7.1 says the QP receives the
nominal action "scaled to physical units", so this must sit downstream of de-normalization.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import osqp
import scipy.sparse as sp

from benchmark.knows_vla.cbf.ellipsoid import (
    EPS_DENOM,
    Ellipsoid,
    barrier,
    grad_center,
    grad_normal,
    grad_rotation,
    initial_normal,
)


@dataclasses.dataclass(frozen=True)
class CbfParams:
    """None of these five values appear in the paper. See benchmark/knows_vla/docs/OPEN-QUESTIONS.md #1."""

    gamma_h: float = 0.5  # Eq. (7) decay, (0, 1]              [ASSUMPTION]
    W: float = 1.0  # Eq. (12) rotation vs translation weight   [ASSUMPTION]
    eps_normal: float = 0.05  # Eq. (12) ||delta n||_inf bound  [ASSUMPTION]
    q_eig_floor: float = 1e-8  # guards sqrt(n' Q n) -> 0 for near-degenerate shapes
    # OSQP tolerances. The paper names the solver [39] but gives no settings [ASSUMPTION].
    # Defaults (1e-3) leave primal residuals around 1e-5 m on the barrier constraint, i.e. the
    # returned action can violate Eq. (8) by microns. Harmless physically, but a safety filter
    # should not be the loosest link, and tightening costs little at this problem size.
    solver_eps_abs: float = 1e-9
    solver_eps_rel: float = 1e-9
    solver_max_iter: int = 20_000
    # NOT in the paper. Eq. (12) bounds delta_n but never delta_c or delta_theta, so a badly
    # violated barrier can demand an arbitrarily large correction -- measured up to 5 m against a
    # 5 cm OSC action limit (docs/11-p2b-results.md). Setting these adds box constraints, which
    # also makes the QP genuinely infeasible sometimes and so lets the paper's emergency-stop
    # fallback actually fire. Leave as None to reproduce Eq. (12) literally. See OPEN-QUESTIONS #13.
    max_delta_pos: float | None = None
    max_delta_rot: float | None = None


@dataclasses.dataclass
class FilterResult:
    delta_c: np.ndarray  # (3,) applied translational delta
    delta_theta: np.ndarray  # (3,) applied rotational delta (world-frame axis-angle)
    feasible: bool
    emergency_stop: bool
    h: np.ndarray  # (m,) barrier value per obstacle, before the step
    status: str


class SafetyFilter:
    """Stateful CBF-QP. The virtual normals persist across calls, which is what warm-starts it."""

    def __init__(self, params: CbfParams | None = None):
        self.p = params or CbfParams()
        self._normals: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self._normals.clear()

    def normals(self) -> dict[int, np.ndarray]:
        return {k: v.copy() for k, v in self._normals.items()}

    def _normal_for(self, key: int, robot: Ellipsoid, obstacle: Ellipsoid) -> np.ndarray:
        if key not in self._normals:
            self._normals[key] = initial_normal(robot, obstacle)
        return self._normals[key]

    def __call__(
        self,
        robot: Ellipsoid,
        obstacles: dict[int, Ellipsoid],
        delta_c_nom: np.ndarray,
        delta_theta_nom: np.ndarray,
    ) -> FilterResult:
        """Solve Eq. (12) for one control step.

        ``obstacles`` is keyed by a stable object id so each keeps its own virtual normal; the
        target must already have been removed by the attention stage (Eq. 4).
        """
        delta_c_nom = np.asarray(delta_c_nom, np.float64).reshape(3)
        delta_theta_nom = np.asarray(delta_theta_nom, np.float64).reshape(3)
        keys = sorted(obstacles)
        m = len(keys)

        if m == 0:  # nothing to avoid -> the filter must be the identity
            return FilterResult(delta_c_nom.copy(), delta_theta_nom.copy(), True, False,
                                np.zeros(0), "no_obstacles")

        n_var = 6 + 3 * m
        h_vals = np.zeros(m)
        A_rows = np.zeros((m, n_var))
        for i, k in enumerate(keys):
            obs = obstacles[k]
            n = self._normal_for(k, robot, obs)
            h_vals[i] = barrier(n, robot, obs)
            A_rows[i, 0:3] = grad_center(n, robot, obs)
            A_rows[i, 3:6] = grad_rotation(n, robot, obs)
            A_rows[i, 6 + 3 * i : 9 + 3 * i] = grad_normal(n, robot, obs)

        # min ||dc - dc_nom||^2 + W||dth - dth_nom||^2  ==  (1/2) x'Px + q'x  + const
        P = sp.diags(np.concatenate([np.full(3, 2.0), np.full(3, 2.0 * self.p.W), np.zeros(3 * m)])).tocsc()
        q = np.concatenate([-2.0 * delta_c_nom, -2.0 * self.p.W * delta_theta_nom, np.zeros(3 * m)])

        # Eq. (8) per obstacle, then the box ||delta n||_inf <= eps.
        box = np.zeros((3 * m, n_var))
        box[:, 6:] = np.eye(3 * m)
        rows = [A_rows, box]
        lower = [-self.p.gamma_h * h_vals, np.full(3 * m, -self.p.eps_normal)]
        upper = [np.full(m, np.inf), np.full(3 * m, self.p.eps_normal)]

        # Optional action limits (not in Eq. 12) -- see the CbfParams note.
        for slice_, bound in ((slice(0, 3), self.p.max_delta_pos), (slice(3, 6), self.p.max_delta_rot)):
            if bound is None:
                continue
            lim = np.zeros((3, n_var))
            lim[:, slice_] = np.eye(3)
            rows.append(lim)
            lower.append(np.full(3, -abs(bound)))
            upper.append(np.full(3, abs(bound)))

        A = sp.csc_matrix(np.vstack(rows))
        lower = np.concatenate(lower)
        upper = np.concatenate(upper)

        solver = osqp.OSQP()
        solver.setup(
            P=P, q=q, A=A, l=lower, u=upper,
            verbose=False,
            polishing=True,
            eps_abs=self.p.solver_eps_abs,
            eps_rel=self.p.solver_eps_rel,
            max_iter=self.p.solver_max_iter,
        )
        res = solver.solve()
        status = str(res.info.status)

        if res.x is None or not np.all(np.isfinite(res.x)) or "solved" not in status.lower():
            # Appendix 7.1: "If the QP is infeasible, we fall back to an emergency stop with zero
            # translational and rotational deltas."
            return FilterResult(np.zeros(3), np.zeros(3), False, True, h_vals, status)

        x = np.asarray(res.x, np.float64)
        for i, k in enumerate(keys):
            n_new = self._normals[k] + x[6 + 3 * i : 9 + 3 * i]
            nrm = float(np.linalg.norm(n_new))
            if nrm > EPS_DENOM:  # renormalize, per Appendix 7.1
                self._normals[k] = n_new / nrm

        return FilterResult(x[0:3].copy(), x[3:6].copy(), True, False, h_vals, status)


def effective_margin(n, robot: Ellipsoid, obstacle: Ellipsoid, eps: float) -> float:
    """How much the ``delta n`` box relaxes obstacle j's constraint: eps * ||grad_n h||_1.

    ``delta n`` appears in no objective term, so the QP is free to spend the whole box on making
    the constraint easier. The paper presents ``eps`` only as a smoothness bound on the hyperplane
    estimate and never notes that it also weakens safety. See benchmark/knows_vla/docs/03-math.md.
    """
    return float(eps * np.abs(grad_normal(n, robot, obstacle)).sum())
