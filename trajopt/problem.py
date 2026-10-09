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
* **smoothness** — second differences of position, ``Q[:, k+2] - 2 Q[:, k+1] + Q[:, k]``, i.e.
  **acceleration** in step units (divide by ``dt**2`` for rad/s^2). It is *not* jerk: jerk would be
  the third difference. (Earlier docs and comments called this term "jerk"; the code has always been
  `np.diff(Q, n=2)` — the name was wrong, not the maths. T24, guide §7.1.) An optimizer that ignored
  it would hand back exactly the discontinuities SEAM removed.
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
from benchmark.trajopt.limits import velocity_rows
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


def deflection_rate_operator(nq: int, horizon: int, boundary: bool) -> sp.csr_matrix:
    """T43 TA (B) — ``M`` with ``M vec(D)`` = ``[D[:, 0] (if boundary); D[:, k+1] − D[:, k]]``."""
    D1 = difference_operator(nq, horizon, 1)
    if not boundary:
        return D1
    first = sp.csr_matrix((np.ones(nq), (np.arange(nq), np.arange(nq))), shape=(nq, nq * horizon))
    return sp.vstack([first, D1], format="csr")


def deflection_rate_cost(trajectory: np.ndarray, reference: np.ndarray, weight: float,
                         deflection_history: Optional[np.ndarray] = None) -> float:
    """T43 TA (B) — ``w (Σ_k ‖ΔD_k‖² + ‖D_0 − D_prev‖²)`` evaluated directly (``D = Q − Q_ref``)."""
    D = np.asarray(trajectory, np.float64) - np.asarray(reference, np.float64)
    value = float(np.sum(np.diff(D, axis=1) ** 2)) if D.shape[1] > 1 else 0.0
    if deflection_history is not None:
        value += float(np.sum((D[:, 0] - np.asarray(deflection_history, np.float64)) ** 2))
    return float(weight) * value


def servo_accel_rows(reference: np.ndarray, accel_bound: np.ndarray, mode: str,
                     max_step_change: Optional[np.ndarray] = None,
                     command_history: Optional[np.ndarray] = None):
    """T43 TA (C) — the second-difference rows of ``[u[-2], u[-1], Q[:, 0], …]`` and their bounds.

    Returns ``(A_q, const, bound)``: the row values are ``A_q @ vec(Q) + const`` (step-major
    ``vec``), ``const`` carrying the executed commands `command_history` ``(nq, 2)`` (oldest first;
    `None` = inside the chunk only). ``bound`` per row: ``"strict"`` = ``a_C``, ``"relaxed"`` =
    ``max(a_C, |rows of the reference|)``; both capped by the robot's own `max_step_change`.
    """
    reference = np.asarray(reference, np.float64)
    nq, horizon = reference.shape
    hist = None if command_history is None else np.asarray(command_history, np.float64)
    n_hist = 0 if hist is None else int(hist.shape[1])
    full = difference_operator(nq, horizon + n_hist, 2).tocsc()
    split = n_hist * nq
    A_q = full[:, split:].tocsr()
    const = (full[:, :split] @ hist.T.reshape(-1)) if n_hist else np.zeros(A_q.shape[0])
    rows = A_q.shape[0] // nq if nq else 0
    a_c = np.tile(np.asarray(accel_bound, np.float64).reshape(-1), rows)
    if mode == "relaxed":
        own = np.abs(A_q @ reference.T.reshape(-1) + const)
        bound = np.maximum(a_c, own)
    elif mode == "strict":
        bound = a_c
    else:
        raise ValueError(f"servo_accel mode must be 'strict' or 'relaxed', got {mode!r}")
    if max_step_change is not None and np.all(np.isfinite(max_step_change)):
        bound = np.minimum(bound, np.tile(np.asarray(max_step_change, np.float64), rows))
    return A_q, np.asarray(const, np.float64), bound


def boundary_velocity_overshoot(trajectory: np.ndarray, command_history: Optional[np.ndarray],
                                max_step: Optional[np.ndarray]) -> np.ndarray:
    """T43 TA (C) — ``max(0, |Q[:, 0] − u[−1]| − max_step)`` per joint (rad/row).

    The velocity rows of `build_problem` bound ``Q[:, k+1] − Q[:, k]`` inside the chunk, and row 0 is
    bounded only against the *measured* ``q_now`` (the anchor, `problem.py` box block). The servo
    lags the command by ≈ 2.5 rows (`servo.ServoParams.shape_coefficient`: τ = (kv + b)/kp), so
    ``|Q0 − u[−1]|`` was unbounded: V8 executed 3 boundary steps at 1.20–1.74 × ``v_max·0.9·Δt``.
    With C on and the executed history known, this row closes that gap.
    """
    if command_history is None or max_step is None or not np.all(np.isfinite(max_step)):
        return np.zeros(0)
    u_last = np.asarray(command_history, np.float64)[:, -1]
    return np.maximum(0.0, np.abs(np.asarray(trajectory, np.float64)[:, 0] - u_last)
                      - np.asarray(max_step, np.float64))


def servo_accel_overshoot(trajectory: np.ndarray, reference: np.ndarray, accel_bound, mode: str,
                          max_step_change=None, command_history=None) -> np.ndarray:
    """``max(0, |row| − bound)`` per C row (rad/row²) — what the L1 slack pays for."""
    A_q, const, bound = servo_accel_rows(reference, accel_bound, mode, max_step_change,
                                         command_history)
    value = A_q @ np.asarray(trajectory, np.float64).T.reshape(-1) + const
    return np.maximum(0.0, np.abs(value) - bound)


def box_bounds(limits: JointLimits, iterate: np.ndarray, radius: float, *,
               q_now: Optional[np.ndarray] = None,
               anchor_envelope: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """`(lower, upper)` of the box block — joint box ∩ trust region around `iterate` ∩ step-0 anchor
    (∩ the T41 b envelope). The only part of `build_problem` that depends on the iterate and the
    trust radius; `rebox_problem` (T43 TP) re-evaluates exactly this between SQP iterations."""
    iterate = np.asarray(iterate, np.float64)
    nq, horizon = iterate.shape
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
        if anchor_envelope is not None:
            # T41 b — clip (not max/min): the result lies inside the envelope even when the trust
            # region or the joint box does not reach it, so the midpoint collapse below stays inside.
            width = np.asarray(anchor_envelope, np.float64)[:, :horizon].T.reshape(-1)
            centre = np.tile(q_now, horizon)
            box_lo = np.clip(box_lo, centre - width, centre + width)
            box_hi = np.clip(box_hi, centre - width, centre + width)
    elif anchor_envelope is not None:
        raise ValueError("anchor_envelope needs q_now — the envelope is centred on it")
    # The intersection can be empty when the robot starts outside its own limits — a real situation
    # after a fault or a bad hand-off. Collapsing to the midpoint keeps the QP solvable and lets the
    # trajectory walk back inside, rather than failing and leaving the caller with nothing.
    crossed = box_lo > box_hi
    if np.any(crossed):
        mid = 0.5 * (box_lo[crossed] + box_hi[crossed])
        box_lo[crossed] = mid
        box_hi[crossed] = mid
    return box_lo, box_hi


def rebox_problem(template: "QpProblem", limits: JointLimits, iterate: np.ndarray, radius: float, *,
                  q_now: Optional[np.ndarray] = None,
                  anchor_envelope: Optional[np.ndarray] = None) -> "QpProblem":
    """T43 TP (`sqp.fast_eval`) — `build_problem` at a new iterate / trust radius from one built
    earlier **in the same solve**.

    Within a solve every input of `build_problem` but `iterate` and `trust_radius` is fixed
    (reference, limits, config, `q_now`, previous chunk, slack count, envelope, B/C history), and
    those two enter only the box block (`box_bounds`). So the template's `P`, `q`, `A` and every
    other row bound are already what `build_problem` would build; only the box rows of `l`/`u` are
    re-evaluated, by the same function. The result equals `build_problem`'s, array for array.
    """
    lo, hi = box_bounds(limits, iterate, radius, q_now=q_now, anchor_envelope=anchor_envelope)
    start, stop = template.row_blocks["box"]
    l = template.l.copy()
    u = template.u.copy()
    l[start:stop] = lo
    u[start:stop] = hi
    return dataclasses.replace(template, l=l, u=u)


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
    anchor_envelope: Optional[np.ndarray] = None,
    deflection_history: Optional[np.ndarray] = None,
    command_history: Optional[np.ndarray] = None,
    servo_accel_bound: Optional[np.ndarray] = None,
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
        anchor_envelope: ``(nq, H)`` or `None` (default — the anchor is step 0 only, as before).
            T41 b: when the robot starts **at rest** (`limits.rest_start_envelope`), every step is
            held to ``|Q[:, k] - q_now| <= anchor_envelope[:, k]``. It is intersected into the same
            box block, so the QP keeps its rows and sparsity; where the trust region and the envelope
            do not overlap, **the envelope wins** (the trust region is a numerical device, the
            envelope is what the robot can physically do from rest). Needs `q_now`.
        deflection_history: ``(nq,)`` or `None` — T43 TA (B): the deflection of the last executed
            command (`cost.w_deflection_rate`); `None` = no boundary term. Ignored when the weight is 0.
        command_history: ``(nq, 2)`` or `None` — T43 TA (C): the two last executed commands, oldest
            first; `None` = the C rows stay inside the chunk.
        servo_accel_bound: ``(nq,)`` `a_C` (rad/row²) or `None` (default — no C rows). The mode is
            `config.limits.servo_accel`.

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
    w_rate = float(getattr(cost, "w_deflection_rate", 0.0) or 0.0)
    if w_rate > 0.0:
        # T43 TA (B): w ‖M vec(Q − R) − c‖², c = [D_prev; 0] — expanded into P and q.
        boundary = deflection_history is not None
        M = deflection_rate_operator(nq, horizon, boundary)
        if M.shape[0]:
            MtM = (M.T @ M).tocsc()
            P_q = (P_q + 2.0 * w_rate * MtM).tocsc()
            q_q = q_q - 2.0 * w_rate * (MtM @ reference.T.reshape(-1))
            if boundary:
                c = np.zeros(M.shape[0])
                c[:nq] = np.asarray(deflection_history, np.float64).reshape(-1)
                q_q = q_q - 2.0 * w_rate * (M.T @ c)

    # Velocity and acceleration get slack of their own. A chunk that already violates the robot's
    # limits — which a policy trained on demonstrations can easily produce — would otherwise make the
    # QP primal-infeasible, and an infeasible QP hands the controller nothing at the moment it most
    # needs something. `benchmark/knows_vla/cbf/filter.py` records the same lesson (D3): soften the
    # rows so the solve always returns, and make the violation an explicit, reported number rather
    # than a solver status nobody sees.
    #
    # **속도 행은 `velocity_rows` 가 정한다** (T27). `max_step` 이 `inf` 이거나(예전 `enforce_velocity
    # =False`) `RelaxedJointLimits.velocity_rows=False`(`--no-limits`) 면 행이 없다. 후자에서도
    # `max_step` 은 유한하게 남아 아래 box 블록의 첫 스텝 anchor 가 그 값을 쓴다.
    n_velocity = nq * max(horizon - 1, 0) if velocity_rows(limits) else 0
    n_acceleration = (
        nq * max(horizon - 2, 0) if np.all(np.isfinite(limits.max_step_change)) else 0
    )
    servo = None
    if servo_accel_bound is not None:
        servo = servo_accel_rows(
            reference, servo_accel_bound, str(config.limits.servo_accel),
            limits.max_step_change if np.all(np.isfinite(limits.max_step_change)) else None,
            command_history)
    n_servo = 0 if servo is None else int(servo[0].shape[0])
    # T43 TA (C): the velocity row across the chunk start, |Q0 − u[−1]| <= max_step (with C and the
    # executed history only; the in-chunk velocity rows above are unchanged).
    n_bvel = (nq if servo is not None and command_history is not None and n_velocity else 0)
    n_kinematic = n_velocity + n_acceleration + n_servo + n_bvel
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
        np.full(n_velocity + n_acceleration, cost.w_slack * 10.0),
        # T43 TA (C): per rad/row², weighted like a collision metre (× servo_accel_weight) so that
        # avoidance can still buy acceleration when it must.
        np.full(n_servo, cost.w_slack * float(getattr(config.limits, "servo_accel_weight", 1.0))),
        # a robot velocity limit, weighted like the in-chunk velocity rows
        np.full(n_bvel, cost.w_slack * 10.0),
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
    #
    # With `limits.enforce_position=False` (`--no-limits`, T27) `lower/upper` are `∓inf`, so this
    # block is the trust region alone plus the step-0 anchor — the two SQP devices that stay.
    box_lo, box_hi = box_bounds(limits, iterate, radius, q_now=q_now,
                                anchor_envelope=anchor_envelope)
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
    if n_servo:
        # T43 TA (C): |A_q z + c| <= b + s, one slack per row (after the velocity/acceleration ones).
        A_s, c_s, b_s = servo
        offset = n_slack + n_velocity + n_acceleration
        picker = sp.csr_matrix(
            (np.ones(n_servo), (np.arange(n_servo), offset + np.arange(n_servo))),
            shape=(n_servo, n_extra))
        _add("servo_accel", A_s, -picker, np.full(n_servo, -np.inf), b_s - c_s)
        _add("servo_accel_lower", A_s, picker, -b_s - c_s, np.full(n_servo, np.inf))
    if n_bvel:
        first = sp.csr_matrix((np.ones(nq), (np.arange(nq), np.arange(nq))), shape=(nq, n_q))
        u_last = np.asarray(command_history, np.float64)[:, -1]
        offset = n_slack + n_velocity + n_acceleration + n_servo
        picker = sp.csr_matrix((np.ones(nq), (np.arange(nq), offset + np.arange(nq))),
                               shape=(nq, n_extra))
        vmax = np.asarray(limits.max_step, np.float64)
        _add("boundary_velocity", first, -picker, np.full(nq, -np.inf), u_last + vmax)
        _add("boundary_velocity_lower", first, picker, u_last - vmax, np.full(nq, np.inf))

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
    deflection_history: Optional[np.ndarray] = None,
    command_history: Optional[np.ndarray] = None,
    servo_accel_bound: Optional[np.ndarray] = None,
    max_step_change: Optional[np.ndarray] = None,
    max_step: Optional[np.ndarray] = None,
) -> float:
    """The cost `build_problem` encodes, evaluated directly.

    Exists so the SQP loop can compare *actual* against *predicted* reduction without trusting the
    matrix assembly to be its own witness. A merit function computed from the same code that builds
    the QP would agree with it even when both are wrong.

    T43 TA: the deflection-rate term (B, `cost.w_deflection_rate`) and the L1 price of the C rows'
    overshoot (`servo_accel_bound` given) are part of it — otherwise a reference that breaks the C
    bound would always win step acceptance against the candidate that respects it.
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
    w_rate = float(getattr(cost, "w_deflection_rate", 0.0) or 0.0)
    if w_rate > 0.0:
        value += deflection_rate_cost(Q, reference, w_rate, deflection_history)
    if servo_accel_bound is not None:
        over = servo_accel_overshoot(Q, reference, servo_accel_bound,
                                     str(config.limits.servo_accel), max_step_change,
                                     command_history)
        value += float(cost.w_slack * float(getattr(config.limits, "servo_accel_weight", 1.0))
                       * np.sum(over))
        value += float(cost.w_slack * 10.0 * np.sum(
            boundary_velocity_overshoot(Q, command_history, max_step)))
    if slack is not None:
        value += float(cost.w_slack * np.sum(np.abs(slack)))
    return value


__all__ = ["QpProblem", "build_problem", "boundary_velocity_overshoot", "deflection_rate_cost",
           "deflection_rate_operator",
           "difference_operator", "objective", "servo_accel_overshoot", "servo_accel_rows"]
