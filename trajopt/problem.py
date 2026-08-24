"""Assembling the quadratic subproblem: what to preserve, and what the robot may physically do.

The decision vector is the trajectory itself rather than a step from it:

    z = [ vec(Q) ; s ],   Q in R^{nq x H} stored step-major,   s >= 0 the collision slacks

Step-major (``z[k*nq + j] = Q[j, k]``) is not cosmetic. Every collision constraint acts at one time
step, so in this ordering each collision row touches one contiguous block of `nq` variables. That
is what makes the constraint block's sparsity pattern fixed and predictable, which is in turn what
lets a QP solver keep its factorization between iterations.

Working in absolute coordinates rather than increments means the joint box, the trust region and the
initial-state anchor are all bounds on the same variable and can be intersected into one row block
instead of three. The trust region then reads as what it is: *this iteration may not move any joint
further than r from where the linearization was taken.*

Four things enter the cost, and only the first is about the task:

* **tracking** — SEAM's chunk is the answer the policy wants; the optimizer's job is to change it as
  little as collision avoidance permits. A tracking weight that is too small yields a safe
  trajectory that no longer performs the task, and no collision metric will notice.
* **smoothness** — second differences, i.e. jerk. This is the quantity SEAM exists to control, and
  an optimizer that ignored it would hand back exactly the discontinuities SEAM removed.
* **continuity** — against the previously refined chunk. See `config.CostConfig`: SEAM's next-chunk
  prior is built from the *pre-refinement* chunk, so without this term the two layers pull in
  opposite directions every chunk.
* **slack** — linear, and weighted far above tracking. Its purpose is feasibility, not permission.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import scipy.sparse as sp

from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.types import JointLimits


@dataclasses.dataclass(frozen=True)
class QpProblem:
    """One quadratic subproblem in OSQP's canonical form.

    ``minimize  1/2 z' P z + q' z    subject to    l <= A z <= u``

    `n_q` and `n_slack` are carried so a caller can split the solution without re-deriving the
    layout, and `row_blocks` names each contiguous group of constraint rows so a violated row can be
    reported as "velocity limit on step 12" rather than "row 731".
    """

    P: sp.csc_matrix
    q: np.ndarray
    A: sp.csc_matrix
    l: np.ndarray
    u: np.ndarray
    n_q: int  # nq * H
    n_slack: int  # collision slacks
    nq: int
    horizon: int
    row_blocks: dict[str, tuple[int, int]] = dataclasses.field(default_factory=dict)
    n_kinematic_slack: int = 0

    @property
    def n_var(self) -> int:
        return self.n_q + self.n_slack + self.n_kinematic_slack

    def split(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``z -> (Q[nq, H], s[n_slack])``. Kinematic slacks are diagnostics, read by `kinematic_slack`."""
        z = np.asarray(z, np.float64).reshape(-1)
        Q = z[: self.n_q].reshape(self.horizon, self.nq).T
        return np.ascontiguousarray(Q), z[self.n_q : self.n_q + self.n_slack]

    def kinematic_slack(self, z: np.ndarray) -> np.ndarray:
        """How far the solution had to break the robot's own velocity/acceleration limits."""
        z = np.asarray(z, np.float64).reshape(-1)
        return z[self.n_q + self.n_slack :]

    def flatten(self, Q: np.ndarray, slack: Optional[np.ndarray] = None) -> np.ndarray:
        """``(Q[nq, H], s) -> z``. The inverse of `split`, for warm starts."""
        Q = np.asarray(Q, np.float64)
        s = np.zeros(self.n_slack) if slack is None else np.asarray(slack, np.float64).reshape(-1)
        return np.concatenate([Q.T.reshape(-1), s, np.zeros(self.n_kinematic_slack)])


def difference_operator(nq: int, horizon: int, order: int) -> sp.csr_matrix:
    """First or second difference along time, as a sparse matrix acting on ``vec(Q)``.

    Order 1 gives ``Q[:, k+1] - Q[:, k]`` (a velocity in step units) and order 2 gives
    ``Q[:, k+2] - 2 Q[:, k+1] + Q[:, k]`` (an acceleration). Both are needed twice — once as a
    constraint on what the robot can do, once inside the smoothness cost — so they are built here
    rather than inline, where the two copies would eventually disagree about a sign.

    Built from diagonals rather than assembled block by block. In the step-major ordering a
    difference along time is simply a band at offset `nq`, and `sp.bmat` over `horizon x horizon`
    blocks costs 8 ms at H=50 where `sp.diags` costs microseconds — measured, and it was a third of
    the per-iteration budget before it was noticed.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    rows = horizon - order
    if rows <= 0:
        return sp.csr_matrix((0, nq * horizon))
    n_row = rows * nq
    stencil = {1: (-1.0, 1.0), 2: (1.0, -2.0, 1.0)}[order]
    diagonals = [np.full(n_row, coefficient) for coefficient in stencil]
    offsets = [offset * nq for offset in range(len(stencil))]
    return sp.diags(diagonals, offsets, shape=(n_row, nq * horizon), format="csr")


def build_problem(
    reference: np.ndarray,
    iterate: np.ndarray,
    limits: JointLimits,
    config: TrajOptConfig,
    *,
    q_now: Optional[np.ndarray] = None,
    previous_chunk: Optional[np.ndarray] = None,
    n_slack: int = 0,
    trust_radius: Optional[float] = None,
) -> QpProblem:
    """The QP around `iterate`, tracking `reference`, with room for `n_slack` collision rows.

    Args:
        reference: ``(nq, H)`` — SEAM's chunk, what the optimizer is trying to stay near.
        iterate: ``(nq, H)`` — where the linearization is taken; the trust region centres here.
        q_now: ``(nq,)`` the robot's current joint values. Used as the anchor: the first chunk entry
            may not be further than one step's travel from where the robot actually is. Note this is
            an *anchor*, not a pin — pinning ``Q[:, 0] = q_now`` would forbid the robot from moving
            during the first control period.
        previous_chunk: ``(nq, L)`` the overlapping tail of the previously refined chunk, for the
            continuity term. `None` disables it, which is correct for the first chunk of an episode.
        n_slack: how many collision rows the caller will add. Their slack variables and the
            ``s >= 0`` block are reserved here so the layout is complete before any geometry arrives.

    Collision rows are **not** added here — `linearize.append_collision_rows` does that. Keeping the
    split means this function can be tested against a problem with no geometry at all, where the
    right answer is known exactly: return the reference.
    """
    reference = np.asarray(reference, np.float64)
    iterate = np.asarray(iterate, np.float64)
    nq, horizon = reference.shape
    if iterate.shape != reference.shape:
        raise ValueError(f"iterate {iterate.shape} does not match reference {reference.shape}")
    if limits.nq != nq:
        raise ValueError(f"limits cover {limits.nq} joints but the trajectory has {nq}")
    n_q = nq * horizon
    radius = float(config.sqp.trust_radius if trust_radius is None else trust_radius)

    # ---- cost -------------------------------------------------------------------------
    cost = config.cost
    # Later steps are discarded (only K of H execute) and are the least certain, so tracking them
    # can be worth less. `track_decay = 1.0` keeps every step equal, which is the default.
    decay = cost.track_decay ** np.arange(horizon, dtype=np.float64)
    track_w = np.repeat(cost.w_track * decay, nq)  # step-major, so repeat per step

    diag = track_w.copy()
    linear = -track_w * reference.T.reshape(-1)

    if previous_chunk is not None and cost.w_continuity > 0.0:
        previous_chunk = np.asarray(previous_chunk, np.float64)
        if previous_chunk.shape[0] != nq:
            raise ValueError(
                f"previous_chunk has {previous_chunk.shape[0]} joints, expected {nq}"
            )
        overlap = min(previous_chunk.shape[1], horizon)
        cont_w = np.zeros(n_q)
        cont_w[: overlap * nq] = cost.w_continuity
        diag += cont_w
        linear[: overlap * nq] -= (
            cost.w_continuity * previous_chunk[:, :overlap].T.reshape(-1)
        )

    P_q = sp.diags(2.0 * diag, format="csc")
    if cost.w_smooth > 0.0 and horizon > 2:
        D2 = difference_operator(nq, horizon, 2)
        P_q = (P_q + 2.0 * cost.w_smooth * (D2.T @ D2)).tocsc()
    q_q = 2.0 * linear

    # Velocity and acceleration get slack of their own. A chunk that already violates the robot's
    # limits — which a policy trained on demonstrations can easily produce — would otherwise make the
    # QP primal-infeasible, and an infeasible QP hands the controller nothing at the moment it most
    # needs something. `benchmark/knows_vla/cbf/filter.py` records the same lesson (D3): soften the
    # rows so the solve always returns, and make the violation an explicit, reported number rather
    # than a solver status nobody sees.
    n_velocity = nq * max(horizon - 1, 0) if np.all(np.isfinite(limits.max_step)) else 0
    n_acceleration = (
        nq * max(horizon - 2, 0) if np.all(np.isfinite(limits.max_step_change)) else 0
    )
    n_kinematic = n_velocity + n_acceleration
    n_extra = n_slack + n_kinematic

    P = sp.block_diag(
        [P_q, sp.csc_matrix((n_slack + n_kinematic, n_slack + n_kinematic))], format="csc"
    )
    # Slack is penalised linearly, not quadratically: an L1 penalty on a non-negative variable is
    # exact above a finite weight, so a feasible problem gets s = 0 rather than a small smear of
    # violation spread across every row.
    q = np.concatenate([
        q_q,
        np.full(n_slack, cost.w_slack),
        # Weighted above the collision slack: breaking the robot is worse than grazing an obstacle,
        # and unlike a collision it is certain rather than predicted.
        np.full(n_kinematic, cost.w_slack * 10.0),
    ])

    # ---- constraints ------------------------------------------------------------------
    blocks: list[sp.spmatrix] = []
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    row_blocks: dict[str, tuple[int, int]] = {}
    cursor = 0

    def _add(name: str, on_q: sp.spmatrix, on_extra: sp.spmatrix, lo: np.ndarray, hi: np.ndarray):
        nonlocal cursor
        if on_q.shape[0] == 0:
            return
        blocks.append(sp.hstack([on_q, on_extra], format="csr"))
        lower.append(lo)
        upper.append(hi)
        row_blocks[name] = (cursor, cursor + on_q.shape[0])
        cursor += on_q.shape[0]

    zero_extra = lambda rows: sp.csr_matrix((rows, n_extra))  # noqa: E731

    # Box, trust region and anchor intersected into one block, because all three bound the same
    # variable and three separate row blocks would only make the QP larger.
    box_lo = np.tile(limits.lower, horizon)
    box_hi = np.tile(limits.upper, horizon)
    iterate_flat = iterate.T.reshape(-1)
    box_lo = np.maximum(box_lo, iterate_flat - radius)
    box_hi = np.minimum(box_hi, iterate_flat + radius)
    if q_now is not None:
        q_now = np.asarray(q_now, np.float64).reshape(-1)
        if q_now.shape[0] != nq:
            raise ValueError(f"q_now has {q_now.shape[0]} entries, expected {nq}")
        box_lo[:nq] = np.maximum(box_lo[:nq], q_now - limits.max_step)
        box_hi[:nq] = np.minimum(box_hi[:nq], q_now + limits.max_step)
    # The intersection can be empty when the robot starts outside its own limits — a real situation
    # after a fault or a bad hand-off. Collapsing to the midpoint keeps the QP solvable and lets the
    # trajectory walk back inside, rather than failing and leaving the caller with nothing.
    crossed = box_lo > box_hi
    if np.any(crossed):
        mid = 0.5 * (box_lo[crossed] + box_hi[crossed])
        box_lo[crossed] = mid
        box_hi[crossed] = mid
    _add("box", sp.identity(n_q, format="csr"), zero_extra(n_q), box_lo, box_hi)

    # Velocity and acceleration, each softened by one non-negative slack. A two-sided bound needs two
    # one-sided rows to share a slack: `a.x - s <= b` and `a.x + s >= -b`. One slack rather than two,
    # because a row cannot be violated on both sides at once.
    infinite = np.full(n_q, np.inf)
    for name, count, order, bound, offset in (
        ("velocity", n_velocity, 1, limits.max_step, n_slack),
        ("acceleration", n_acceleration, 2, limits.max_step_change, n_slack + n_velocity),
    ):
        if not count:
            continue
        operator = difference_operator(nq, horizon, order)
        limit = np.tile(bound, horizon - order)
        picker = sp.csr_matrix(
            (np.ones(count), (np.arange(count), offset + np.arange(count))),
            shape=(count, n_extra),
        )
        _add(name, operator, -picker, np.full(count, -np.inf), limit)
        _add(f"{name}_lower", operator, picker, -limit, np.full(count, np.inf))

    A = sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix((0, n_q + n_extra))
    l = np.concatenate(lower) if lower else np.zeros(0)
    u = np.concatenate(upper) if upper else np.zeros(0)

    if n_extra:
        slack_block = sp.hstack(
            [sp.csr_matrix((n_extra, n_q)), sp.identity(n_extra, format="csr")], format="csr"
        )
        row_blocks["slack"] = (A.shape[0], A.shape[0] + n_extra)
        A = sp.vstack([A, slack_block], format="csr")
        l = np.concatenate([l, np.zeros(n_extra)])
        u = np.concatenate([u, np.full(n_extra, np.inf)])

    return QpProblem(
        P=P.tocsc(),
        q=q,
        A=A.tocsc(),
        l=l,
        u=u,
        n_q=n_q,
        n_slack=n_slack,
        nq=nq,
        horizon=horizon,
        row_blocks=row_blocks,
        n_kinematic_slack=n_kinematic,
    )


def objective(
    trajectory: np.ndarray,
    reference: np.ndarray,
    config: TrajOptConfig,
    *,
    previous_chunk: Optional[np.ndarray] = None,
    slack: Optional[np.ndarray] = None,
) -> float:
    """The cost `build_problem` encodes, evaluated directly.

    Exists so the SQP loop can compare *actual* against *predicted* reduction without trusting the
    matrix assembly to be its own witness. A merit function computed from the same code that builds
    the QP would agree with it even when both are wrong.
    """
    Q = np.asarray(trajectory, np.float64)
    reference = np.asarray(reference, np.float64)
    cost = config.cost
    decay = cost.track_decay ** np.arange(Q.shape[1], dtype=np.float64)
    value = float(cost.w_track * np.sum(decay * np.sum((Q - reference) ** 2, axis=0)))
    if cost.w_smooth > 0.0 and Q.shape[1] > 2:
        value += float(cost.w_smooth * np.sum(np.diff(Q, n=2, axis=1) ** 2))
    if previous_chunk is not None and cost.w_continuity > 0.0:
        previous_chunk = np.asarray(previous_chunk, np.float64)
        overlap = min(previous_chunk.shape[1], Q.shape[1])
        value += float(
            cost.w_continuity * np.sum((Q[:, :overlap] - previous_chunk[:, :overlap]) ** 2)
        )
    if slack is not None:
        value += float(cost.w_slack * np.sum(np.abs(slack)))
    return value


__all__ = ["QpProblem", "build_problem", "difference_operator", "objective"]
