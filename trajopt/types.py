"""Trajectory-optimization data contracts.

Two ideas shape everything here.

**The optimizer may only move what the robot can actually be told to move.** π0.5's RB-Y1 action is
``[L arm_0..5, L gripper, R arm_0..5, R gripper]`` — fourteen numbers, of which twelve are joints.
The torso has no channel and `arm_6` is held fixed by `pi05_infer.apply_action`, so both are
*parameters* of the optimization, not variables. Optimizing a degree of freedom the action format
cannot carry produces a beautiful trajectory that cannot be executed, and the discrepancy shows up
as the robot going somewhere other than where the optimizer thought it would.

**A result that could not be made safe is still a result, and says so.** AG3S refuses to decide what
to do about geometry it cannot certify; the same rule applies one layer up. `TrajOptResult` carries
the trajectory it managed to produce *and* the worst constraint violation left in it, and
`TrajOptStatus.VIOLATED` is a report rather than an exception. Raising would leave the controller
with nothing to send at exactly the moment it most needs something, which is how a safety layer ends
up disabled in production.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Optional, Sequence

import numpy as np


class TrajOptStatus(str, enum.Enum):
    """How the solve ended. Only `OPTIMAL` and `FEASIBLE` are collision-free."""

    OPTIMAL = "optimal"  # converged inside the budget, no constraint violated
    FEASIBLE = "feasible"  # budget or iteration cap hit first, but nothing is violated
    VIOLATED = "violated"  # best effort; some constraint is still violated — caller decides
    SOLVER_FAILED = "solver_failed"  # every QP failed; the reference is returned unchanged
    UNCONSTRAINED = "unconstrained"  # AG3S emitted nothing to avoid; the reference passes through

    @property
    def safe(self) -> bool:
        """Whether the returned trajectory satisfies every constraint AG3S emitted.

        `UNCONSTRAINED` is deliberately **not** safe. "There was nothing to avoid" and "I checked and
        found nothing" look identical here and are not the same claim — an empty constraint set can
        mean the cameras failed. AG3S already distinguishes the two in `ConstraintValidity`; this
        flag refuses to launder the distinction away.
        """
        return self in (TrajOptStatus.OPTIMAL, TrajOptStatus.FEASIBLE)


@dataclasses.dataclass(frozen=True)
class ChunkLayout:
    """Which action-chunk columns are joints, which joints they are, and what is held fixed.

    The mapping is data rather than code because it differs per embodiment: RB-Y1's chunk is
    ``[L 6 joints, L gripper, R 6 joints, R gripper]`` while LIBERO's is a 7-vector of end-effector
    deltas. Only the first is optimizable in joint space, and a layout object makes the difference a
    configuration error rather than a silent misalignment.

    Attributes:
        action_dim: full padded width of the chunk (32 for π0.5), including columns nobody reads.
        n_valid: how many leading columns carry meaning (14 for RB-Y1).
        action_to_q: ``(action_column, q_index)`` for every joint the optimizer may move, in the
            order the optimizer's decision vector uses.
        fixed_q: q indices held at their current value — the torso (no action channel) and `arm_6`
            (pinned by the executor). They still enter forward kinematics; they are just not free.
        passthrough: action columns copied through untouched. Grippers live here: they are
            near-binary open/close and smoothing them would blur a discrete decision.
    """

    action_dim: int
    n_valid: int
    action_to_q: tuple[tuple[int, int], ...]
    fixed_q: tuple[int, ...]
    passthrough: tuple[int, ...]
    nq_model: int
    joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        columns = [a for a, _ in self.action_to_q]
        if len(set(columns)) != len(columns):
            raise ValueError(f"an action column is claimed twice by action_to_q: {columns}")
        q_indices = [q for _, q in self.action_to_q]
        if len(set(q_indices)) != len(q_indices):
            raise ValueError(f"a joint is claimed twice by action_to_q: {q_indices}")
        overlap = set(columns) & set(self.passthrough)
        if overlap:
            raise ValueError(f"columns {sorted(overlap)} are both optimized and passed through")
        clash = set(q_indices) & set(self.fixed_q)
        if clash:
            raise ValueError(f"joints {sorted(clash)} are both free and fixed")
        for column in columns + list(self.passthrough):
            if not 0 <= column < self.action_dim:
                raise ValueError(f"action column {column} is outside the {self.action_dim}-wide chunk")
        for q in q_indices + list(self.fixed_q):
            if not 0 <= q < self.nq_model:
                raise ValueError(f"q index {q} is outside the {self.nq_model}-joint model")

    @property
    def nq_opt(self) -> int:
        """Number of joints the optimizer may move."""
        return len(self.action_to_q)

    @property
    def action_columns(self) -> np.ndarray:
        return np.asarray([a for a, _ in self.action_to_q], np.int64)

    @property
    def q_indices(self) -> np.ndarray:
        return np.asarray([q for _, q in self.action_to_q], np.int64)

    def chunk_to_trajectory(self, chunk: np.ndarray) -> np.ndarray:
        """``chunk[H, action_dim]`` -> ``Q[nq_opt, H]``, the optimizer's decision variable."""
        chunk = np.asarray(chunk, np.float64)
        if chunk.ndim != 2:
            raise ValueError(f"chunk must be (H, action_dim), got {chunk.shape}")
        if chunk.shape[1] < self.n_valid:
            raise ValueError(
                f"chunk has {chunk.shape[1]} columns but the layout needs at least {self.n_valid}"
            )
        return chunk[:, self.action_columns].T.copy()

    def trajectory_to_chunk(self, trajectory: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Write ``Q[nq_opt, H]`` back into a copy of `template`, leaving every other column alone.

        Writing into a copy of the original chunk rather than building a fresh one is what makes the
        gripper columns and the padding pass through byte-identically: they are never touched, so
        they cannot drift.
        """
        trajectory = np.asarray(trajectory, np.float64)
        out = np.array(template, np.float64, copy=True)
        if trajectory.shape[0] != self.nq_opt:
            raise ValueError(
                f"trajectory must have {self.nq_opt} joints, got {trajectory.shape[0]}"
            )
        planned = trajectory.shape[1]
        if planned > out.shape[0]:
            raise ValueError(
                f"trajectory covers {planned} steps but the chunk has only {out.shape[0]}"
            )
        # A trajectory shorter than the chunk writes only its leading steps. The rest of the chunk is
        # the policy's own answer, untouched — it is discarded by the executor before it could run.
        out[:planned, self.action_columns] = trajectory.T
        return out

    def full_q(self, trajectory: np.ndarray, q_now: np.ndarray) -> np.ndarray:
        """``(nq_model, H)`` — the free joints from `trajectory`, everything else held at `q_now`.

        This is what forward kinematics consumes. The fixed joints are broadcast rather than dropped
        because the torso still positions the arms: leaving it out of FK would place every collision
        sphere as though the robot were standing straight.
        """
        trajectory = np.asarray(trajectory, np.float64)
        q_now = np.asarray(q_now, np.float64).reshape(-1)
        if q_now.shape[0] != self.nq_model:
            raise ValueError(f"q_now must have {self.nq_model} entries, got {q_now.shape[0]}")
        horizon = trajectory.shape[1]
        full = np.tile(q_now.reshape(-1, 1), (1, horizon))
        full[self.q_indices, :] = trajectory
        return full

    @staticmethod
    def rby1(
        joint_names: Sequence[str], *, action_dim: int = 32, n_valid: int = 14
    ) -> "ChunkLayout":
        """The RB-Y1 π0.5 layout, derived from the model's joint order rather than hard-coded.

        `pi05_infer.py` documents the chunk as
        ``14 = [L 6 abs joint, L grip, R 6 abs joint, R grip]`` and applies it to ``joints[:6]`` with
        `arm_6` held fixed. Both halves are reproduced here, and the q indices are looked up by name
        so a change to `DEFAULT_RBY1_JOINTS` cannot silently shift the mapping by one.
        """
        order = {name: i for i, name in enumerate(joint_names)}
        pairs: list[tuple[int, int]] = []
        for column, name in enumerate(f"left_arm_{i}" for i in range(6)):
            pairs.append((column, order[name]))
        for offset, name in enumerate(f"right_arm_{i}" for i in range(6)):
            pairs.append((7 + offset, order[name]))
        # Everything the action format cannot address: the six torso joints and both `arm_6`
        # wrists. They are not free, and they are not absent either — forward kinematics needs them.
        free = {q for _, q in pairs}
        fixed = tuple(i for i in range(len(joint_names)) if i not in free)
        passthrough = tuple(c for c in range(action_dim) if c not in {a for a, _ in pairs})
        return ChunkLayout(
            action_dim=int(action_dim),
            n_valid=int(n_valid),
            action_to_q=tuple(pairs),
            fixed_q=fixed,
            passthrough=passthrough,
            nq_model=len(joint_names),
            joint_names=tuple(joint_names),
        )


@dataclasses.dataclass(frozen=True)
class JointLimits:
    """Position, velocity and acceleration bounds for the optimized joints, already in step units.

    Velocity and acceleration arrive from the URDF in rad/s and rad/s², and are converted here using
    the control period so that the optimizer's constraints are plain bounds on differences of the
    decision variable. Doing the conversion once, in a named place, is what stops a stray `dt` from
    appearing in one constraint and not another.
    """

    lower: np.ndarray  # (nq_opt,) rad
    upper: np.ndarray  # (nq_opt,) rad
    max_step: np.ndarray  # (nq_opt,) rad per control step  = v_max * dt
    max_step_change: np.ndarray  # (nq_opt,) rad per step^2 = a_max * dt^2
    dt: float

    def __post_init__(self) -> None:
        shapes = {a.shape for a in (self.lower, self.upper, self.max_step, self.max_step_change)}
        if len(shapes) != 1:
            raise ValueError(f"limit vectors disagree in shape: {shapes}")
        if np.any(self.lower > self.upper):
            bad = np.flatnonzero(self.lower > self.upper)
            raise ValueError(f"joint(s) {bad.tolist()} have lower > upper")
        if np.any(self.max_step <= 0) or np.any(self.max_step_change <= 0):
            raise ValueError("velocity and acceleration limits must be positive")

    @property
    def nq(self) -> int:
        return int(self.lower.shape[0])


@dataclasses.dataclass(frozen=True)
class TrajOptResult:
    """What the optimizer produced, and how much it should be trusted.

    `max_violation` is measured on **every** constraint AG3S emitted, re-evaluated at the returned
    trajectory — not on the reduced set the QP actually saw. That distinction is the whole point of
    allowing the reduction: pruning is only safe if the answer is checked against the full problem,
    and checking it is cheap compared with solving it.
    """

    chunk: np.ndarray  # (H, action_dim) — ready to execute
    trajectory: np.ndarray  # (nq_opt, H) — the optimized joints alone
    status: TrajOptStatus
    iterations: int
    solve_time_ms: float
    max_violation: float  # metres of penetration at the solution; <= 0 means clear
    reference_violation: float  # the same measure on the input, so the change is visible
    slack_norm: float
    cost: float
    reference_deviation: float  # ||Q - Q_ref||, how far the optimizer moved SEAM's answer
    notes: list[str] = dataclasses.field(default_factory=list)
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def safe(self) -> bool:
        return self.status.safe

    @property
    def improved(self) -> bool:
        """Whether the optimizer actually reduced the violation it was handed."""
        return self.max_violation < self.reference_violation - 1e-12


__all__ = [
    "ChunkLayout",
    "JointLimits",
    "TrajOptResult",
    "TrajOptStatus",
]
