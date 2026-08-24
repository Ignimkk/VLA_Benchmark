"""Trajectory-optimization configuration.

Follows `benchmark.ag3s.config`: frozen dataclasses, per-section `validate()`, and **unknown keys
are a hard error**. That last rule matters more here than it looks. A silently ignored
`time_budget_ms: 20` under the wrong parent is a run that reports the default's latency under the
variant's name, and latency is the number this whole design exists to control.

Numbers that came from the deployment rather than from taste are marked as such. The rest are
starting points to be measured, and the README records what the measurement said.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Mapping


class TrajOptConfigError(ValueError):
    """Raised when a configuration is malformed or internally inconsistent."""


@dataclasses.dataclass(frozen=True)
class HorizonConfig:
    """Chunk geometry and the control period.

    All three are properties of the deployment, not choices: `pi05_infer.py` runs at
    ``CTRL_HZ = 15`` and consumes ``OPEN_LOOP_HORIZON = 8`` actions from each 50-step chunk. `dt`
    follows from the rate and converts the URDF's rad/s limits into per-step bounds.
    """

    horizon: int = 50  # H — must match the policy's action_horizon
    execution_length: int = 8  # K — how many steps actually run before the next chunk
    control_hz: float = 15.0
    # How many leading steps the optimizer actually plans. The rest of the chunk is passed through
    # untouched, because it is discarded: the executor runs `execution_length` actions and requests a
    # new chunk. This is the ordinary MPC separation of control horizon from prediction horizon, and
    # here it is what fits the budget — measured on RB-Y1 with a 53 mm near-miss, all of which reach
    # zero violation:
    #
    #     plan_horizon    50      32      24      16
    #     time (2 SQP)     -    22.4    20.0    15.0  ms
    #     time (3 SQP)  92.8    29.3    27.4    21.4  ms
    #
    # 32 keeps 24 steps of lookahead past the executed window — 1.6 s at 15 Hz — which is what stops
    # the optimizer from painting itself into a corner it can only see once it is too late. None
    # plans the whole chunk.
    plan_horizon: int | None = 32

    @property
    def dt(self) -> float:
        return 1.0 / float(self.control_hz)

    @property
    def overlap(self) -> int:
        """L = H - K, the tail that is discarded and that SEAM's prior is built from."""
        return self.horizon - self.execution_length

    def validate(self) -> None:
        if self.horizon < 2:
            raise TrajOptConfigError(f"horizon.horizon must be >= 2, got {self.horizon}")
        if not 0 < self.execution_length <= self.horizon:
            raise TrajOptConfigError(
                f"horizon.execution_length must satisfy 0 < K <= H; got K={self.execution_length}, "
                f"H={self.horizon}"
            )
        if self.control_hz <= 0.0:
            raise TrajOptConfigError(f"horizon.control_hz must be > 0, got {self.control_hz}")
        if self.plan_horizon is not None:
            if not 0 < self.plan_horizon <= self.horizon:
                raise TrajOptConfigError(
                    f"horizon.plan_horizon must satisfy 0 < P <= H; got P={self.plan_horizon}, "
                    f"H={self.horizon}"
                )
            if self.plan_horizon < self.execution_length:
                raise TrajOptConfigError(
                    f"horizon.plan_horizon ({self.plan_horizon}) is shorter than "
                    f"execution_length ({self.execution_length}): the optimizer would leave part of "
                    "what actually executes unplanned"
                )

    @property
    def planned(self) -> int:
        """Steps the optimizer plans. `horizon` when `plan_horizon` is None."""
        return self.horizon if self.plan_horizon is None else self.plan_horizon


@dataclasses.dataclass(frozen=True)
class CostConfig:
    """What the optimizer is trying to preserve while it makes the chunk safe.

    `w_track` dominates by design: SEAM's chunk is the task, and the optimizer's job is to change it
    as little as collision avoidance allows. A tracking weight that is too low produces a safe
    trajectory that no longer does what the policy intended, which is a failure the collision
    metrics will not catch.

    `w_continuity` exists because of a specific finding: SEAM builds the next chunk's prior from the
    **pre-refinement** model-space chunk (`seam_policy.py`, `with_chunk(model_chunk_np[0], ...)`), so
    the optimizer's corrections never reach it. Without a continuity term, SEAM re-proposes the same
    colliding trajectory every chunk, the optimizer pushes it away again, and the boundary jerk SEAM
    exists to suppress reappears one layer down. The term is applied against
    `context["previous_physical_chunk"]`, which `refine()` already receives.
    """

    w_track: float = 1.0
    w_smooth: float = 0.05  # on second differences — jerk, the quantity SEAM is about
    w_continuity: float = 0.5  # against the previously refined chunk's overlapping tail
    # Linear penalty on constraint violation. It must dominate `w_track` — otherwise the optimizer
    # can buy tracking accuracy with collision — but "dominate" is not "as large as possible": at
    # 1e4 the QP took 175 ADMM iterations where 1e3 takes 25, because a four-order-of-magnitude
    # spread in the cost wrecks the conditioning the solver's scaling has to fight. 1e3 against a
    # tracking weight of 1 is already far above any multiplier this problem produces.
    w_slack: float = 1e3
    # Later steps of the chunk are discarded (only K of H execute) and are also the least certain,
    # so tracking them is worth less. 1.0 keeps the paper-plain behaviour.
    track_decay: float = 1.0

    def validate(self) -> None:
        for name in ("w_track", "w_smooth", "w_continuity", "w_slack"):
            if getattr(self, name) < 0.0:
                raise TrajOptConfigError(f"cost.{name} must be >= 0, got {getattr(self, name)}")
        if self.w_track <= 0.0:
            raise TrajOptConfigError(
                "cost.w_track must be > 0: with no tracking term the optimizer is free to return "
                "any safe trajectory, including one that abandons the task"
            )
        if self.w_slack <= self.w_track:
            raise TrajOptConfigError(
                f"cost.w_slack ({self.w_slack}) must exceed cost.w_track ({self.w_track}), or the "
                "optimizer will buy tracking accuracy with constraint violation"
            )
        if not 0.0 < self.track_decay <= 1.0:
            raise TrajOptConfigError(f"cost.track_decay must be in (0, 1], got {self.track_decay}")


@dataclasses.dataclass(frozen=True)
class LimitsConfig:
    """How the URDF's own limits are applied.

    The scales are there to be conservative, not to be creative. A robot commanded to its exact
    velocity limit has no headroom for the tracking controller underneath, so 0.9 leaves some. The
    position margin keeps the optimizer off the hard stop for the same reason.
    """

    velocity_scale: float = 0.9
    acceleration_scale: float = 0.9
    position_margin: float = 0.02  # rad kept clear of each joint's hard stop
    # Joints whose URDF omits a velocity/acceleration limit. None means "raise instead of guessing";
    # see `UrdfSphereChain.velocity_limits`.
    default_velocity: float | None = None
    default_acceleration: float | None = None
    enforce_velocity: bool = True
    enforce_acceleration: bool = True

    def validate(self) -> None:
        for name in ("velocity_scale", "acceleration_scale"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise TrajOptConfigError(f"limits.{name} must be in (0, 1], got {value}")
        if self.position_margin < 0.0:
            raise TrajOptConfigError(
                f"limits.position_margin must be >= 0, got {self.position_margin}"
            )


@dataclasses.dataclass(frozen=True)
class ConstraintReductionConfig:
    """How many of AG3S's rows the QP actually sees.

    At H=50 with RB-Y1's 46 arm spheres and 32 candidate slots, AG3S emits about 78,000 rows against
    600 decision variables. Rows are the bottleneck, not the solver, and almost all of them are
    inactive by an enormous margin.

    Two reductions, both safety-preserving:

    * `activation_band` — hand the QP only rows whose current value is below the band. A row far
      from active has no influence on the QP solution, and the band must exceed the largest amount
      any row's value can change over one SQP step, which `trust_radius` bounds.
    * `temporal_stride` — enforce collision rows every `s`-th step while keeping *all* steps as
      decision variables. The lookahead is unchanged (a myopic optimizer avoids nothing it cannot
      see); what drops is how densely the swept volume is sampled, bounded by max joint speed x dt.

    Neither touches AG3S's fixed graph — the pruning happens on this side of the contract. The
    guarantee comes from re-evaluating **every** row at the solution, which `sqp` always does.
    """

    activation_band: float = 0.35  # in the units of h; squared-distance form makes this generous
    temporal_stride: int = 1  # 1 = every step
    # Collision rows the QP may hold **per step**, filled tightest-first. Budgeting per step rather
    # than globally is what keeps the QP's sparsity pattern fixed: row i of the block always belongs
    # to step `i // rows_per_step`, so it always touches the same block of decision variables, and a
    # solver can reuse its factorization across iterations. Unused rows are switched off the same way
    # AG3S switches off unused candidate slots — by making them trivially satisfied, not by removal.
    rows_per_step: int = 24
    # Metres of extra clearance demanded of the *linearized* row, so the *true* row is satisfied.
    #
    # A QP that satisfies ``h + grad h . d >= 0`` has not shown that ``h(Q + d) >= 0``: the distance
    # to a sphere is concave along a straight line in joint space, so the true clearance sits below
    # its own tangent. Measured on RB-Y1, an SQP that converged with the linearized rows exactly
    # satisfied still left 0.327 mm of real penetration. Asking the linearization for a few
    # millimetres more closes that gap, and it is the standard remedy rather than a fudge — the
    # alternative is to iterate until the step is small enough for the gap to vanish, which costs
    # iterations a real-time loop does not have.
    #
    # The size follows the trust region: a step of `trust_radius` moves a sphere by at most about
    # `trust_radius * reach`, and the second-order term is that squared over twice the clearance.
    linearization_backoff: float = 5e-3
    enabled: bool = True

    def validate(self) -> None:
        if self.activation_band <= 0.0:
            raise TrajOptConfigError(
                f"reduction.activation_band must be > 0, got {self.activation_band}"
            )
        if self.temporal_stride < 1:
            raise TrajOptConfigError(
                f"reduction.temporal_stride must be >= 1, got {self.temporal_stride}"
            )
        if self.linearization_backoff < 0.0:
            raise TrajOptConfigError(
                f"reduction.linearization_backoff must be >= 0, got {self.linearization_backoff}"
            )
        if self.rows_per_step < 1:
            raise TrajOptConfigError(
                f"reduction.rows_per_step must be >= 1, got {self.rows_per_step}"
            )


@dataclasses.dataclass(frozen=True)
class SqpConfig:
    """The real-time iteration budget.

    `max_iterations` and `time_budget_ms` are a **hard cap**, not a hint: whichever binds first, the
    current best iterate is returned. That is the real-time iteration scheme's whole idea — one (or
    a few) Newton-type steps per control cycle, with the feedback loop rather than the solver doing
    the converging. A solver that runs until it converges cannot be given a deadline.

    The budget follows from the deployment: at 15 Hz one control period is 66.7 ms, and the
    optimizer runs once per chunk while the loop waits. 25 ms keeps it well inside a single period
    and small next to π0.5 inference.
    """

    max_iterations: int = 3
    time_budget_ms: float = 50.0
    trust_radius: float = 0.15  # rad, initial
    trust_radius_min: float = 0.01
    trust_radius_max: float = 0.6
    trust_expand: float = 2.0
    trust_shrink: float = 0.5
    # Step acceptance: the ratio of actual to predicted merit reduction below which the step is
    # rejected and the trust region shrinks.
    accept_ratio: float = 0.1
    step_tolerance: float = 1e-5  # converged when the accepted step is smaller than this
    warm_start: bool = True

    def validate(self) -> None:
        if self.max_iterations < 1:
            raise TrajOptConfigError(f"sqp.max_iterations must be >= 1, got {self.max_iterations}")
        if self.time_budget_ms <= 0.0:
            raise TrajOptConfigError(f"sqp.time_budget_ms must be > 0, got {self.time_budget_ms}")
        if not 0.0 < self.trust_radius_min <= self.trust_radius <= self.trust_radius_max:
            raise TrajOptConfigError(
                "sqp trust radii must satisfy 0 < min <= initial <= max; got "
                f"{self.trust_radius_min} / {self.trust_radius} / {self.trust_radius_max}"
            )
        if self.trust_expand <= 1.0 or not 0.0 < self.trust_shrink < 1.0:
            raise TrajOptConfigError("sqp.trust_expand must be > 1 and trust_shrink in (0, 1)")
        if not 0.0 < self.accept_ratio < 1.0:
            raise TrajOptConfigError(f"sqp.accept_ratio must be in (0, 1), got {self.accept_ratio}")


@dataclasses.dataclass(frozen=True)
class QpConfig:
    """Which QP solver runs the subproblem.

    The default is measured, not assumed — `python -m benchmark.trajopt.qp --benchmark` times every
    available backend on the real RB-Y1 problem and the README records the result. What rules
    solvers *out* is structural: qpOASES is a dense online active-set method whose cost is
    superlinear in the constraint count, so thousands of rows are outside its range whatever the
    timing says.
    """

    solver: str = "osqp"  # osqp | proxqp | highs
    # 1e-3, not the 1e-5 that looks more careful. The subproblem is a *linearization* of a non-convex
    # problem, so solving the model far past the model's own fidelity buys nothing — this is the
    # standard inexact-SQP argument, and here it is also the difference between 70.7 ms and 4.6 ms
    # per solve (measured, RB-Y1, 1400 variables / 3364 rows). What makes it safe rather than merely
    # fast is that `sqp` re-checks the true constraints at full resolution afterwards, so a
    # subproblem solved loosely shows up as a violation the caller is told about, never as a silent
    # error.
    eps_abs: float = 1e-3
    eps_rel: float = 1e-3
    max_iter: int = 4000
    # Polishing refines the ADMM solution so the *duals* are accurate. Nothing here consumes duals
    # except the warm start, and it costs about 0.5 ms per solve. Off by default, on when a caller
    # wants trustworthy multipliers.
    polish: bool = False
    # OSQP refactorizes its KKT matrix whenever it adapts rho. Leaving adaptation on costs 4.1 ms
    # against 1.4 ms with a fixed rho, but a fixed rho is only better *on this problem* — adaptation
    # exists because a bad rho can be catastrophic, and 2.7 ms is a cheap insurance premium.
    adaptive_rho: bool = True
    verbose: bool = False

    #: `qrqp` and `ipqp` are absent for measured reasons, not preference. On the RB-Y1 subproblem
    #: (1,884 variables, 4,116 rows, 16,980 nonzeros) `qrqp` did not return inside 200 s, and `ipqp`
    #: **aborted the interpreter** with a heap corruption. A solver that can take down the control
    #: loop is not a solver this package will offer, whatever its timing would have been.
    SOLVERS = ("osqp", "proxqp", "highs")

    def validate(self) -> None:
        if self.solver not in self.SOLVERS:
            raise TrajOptConfigError(
                f"qp.solver must be one of {self.SOLVERS}, got {self.solver!r}. Absent on measured "
                "grounds: qpOASES is dense active-set and does not scale to this row count, `qrqp` "
                "did not return inside 200 s, and `ipqp` aborted the interpreter."
            )
        if self.max_iter < 1:
            raise TrajOptConfigError(f"qp.max_iter must be >= 1, got {self.max_iter}")


@dataclasses.dataclass(frozen=True)
class SafetyConfig:
    """Where "best effort" stops being good enough — and what happens then.

    Nothing here stops the robot. The optimizer reports `VIOLATED` and hands back the best
    trajectory it found; deciding between stopping, holding and re-planning belongs to whoever owns
    the task, exactly as AG3S refuses to decide what to do about `GEOMETRY_INCOMPLETE`. Putting the
    decision here would create a second safety state machine disagreeing with the first.
    """

    # 0.1 mm. Not tighter, because nothing upstream resolves better: the QP is solved to 1e-3, the
    # scene is voxelized at 6 mm and fused at 12 mm, and the trust region moves in units of 0.15 rad.
    # Reporting VIOLATED for 20 micrometres of numerical residue would be crying wolf, and a safety
    # status nobody believes is a safety status nobody reads.
    violation_tolerance: float = 1e-4
    # Trust the constraint set only when AG3S certified the geometry behind it. A frame AG3S marked
    # INCOMPLETE can still be optimized against, but the result cannot be called safe.
    require_certified_geometry: bool = True

    def validate(self) -> None:
        if self.violation_tolerance < 0.0:
            raise TrajOptConfigError(
                f"safety.violation_tolerance must be >= 0, got {self.violation_tolerance}"
            )


_SECTIONS: dict[str, type] = {
    "horizon": HorizonConfig,
    "cost": CostConfig,
    "limits": LimitsConfig,
    "reduction": ConstraintReductionConfig,
    "sqp": SqpConfig,
    "qp": QpConfig,
    "safety": SafetyConfig,
}


@dataclasses.dataclass(frozen=True)
class TrajOptConfig:
    """Root config. `TrajOptConfig()` is the RB-Y1 deployment's defaults."""

    horizon: HorizonConfig = dataclasses.field(default_factory=HorizonConfig)
    cost: CostConfig = dataclasses.field(default_factory=CostConfig)
    limits: LimitsConfig = dataclasses.field(default_factory=LimitsConfig)
    reduction: ConstraintReductionConfig = dataclasses.field(
        default_factory=ConstraintReductionConfig
    )
    sqp: SqpConfig = dataclasses.field(default_factory=SqpConfig)
    qp: QpConfig = dataclasses.field(default_factory=QpConfig)
    safety: SafetyConfig = dataclasses.field(default_factory=SafetyConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in _SECTIONS:
            getattr(self, name).validate()
        # Cross-section: the activation band has to cover everything one SQP step can change, or a
        # row pruned as "far away" could become violated within the same step.
        if self.reduction.enabled and self.reduction.activation_band <= self.sqp.trust_radius:
            raise TrajOptConfigError(
                f"reduction.activation_band ({self.reduction.activation_band}) must exceed "
                f"sqp.trust_radius ({self.sqp.trust_radius}): a row pruned as inactive must not be "
                "able to become violated inside one step"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TrajOptConfig":
        data = dict(data or {})
        unknown = set(data) - set(_SECTIONS)
        if unknown:
            raise TrajOptConfigError(
                f"unknown top-level config key(s): {sorted(unknown)}; expected {sorted(_SECTIONS)}"
            )
        kwargs: dict[str, Any] = {}
        for name, section_cls in _SECTIONS.items():
            if name in data:
                kwargs[name] = _build_section(section_cls, name, data[name])
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "TrajOptConfig":
        import yaml

        return cls.from_dict(yaml.safe_load(pathlib.Path(path).read_text()) or {})

    def with_overrides(self, overrides: Mapping[str, Any]) -> "TrajOptConfig":
        """Merge a partial dict of the same shape as the YAML. The ablation entry point."""
        merged = self.to_dict()
        for key, value in (overrides or {}).items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return TrajOptConfig.from_dict(merged)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: {f.name: getattr(getattr(self, name), f.name)
                   for f in dataclasses.fields(getattr(self, name))}
            for name in _SECTIONS
        }


def _build_section(section_cls: type, name: str, raw: Any) -> Any:
    if not isinstance(raw, Mapping):
        raise TrajOptConfigError(
            f"config section {name!r} must be a mapping, got {type(raw).__name__}"
        )
    fields = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(raw) - fields
    if unknown:
        raise TrajOptConfigError(
            f"unknown key(s) in {name!r}: {sorted(unknown)}; expected {sorted(fields)}"
        )
    return section_cls(**dict(raw))


__all__ = [
    "ConstraintReductionConfig",
    "CostConfig",
    "HorizonConfig",
    "LimitsConfig",
    "QpConfig",
    "SafetyConfig",
    "SqpConfig",
    "TrajOptConfig",
    "TrajOptConfigError",
]
