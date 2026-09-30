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


#: `HorizonConfig.plan_horizon` 의 sentinel — **실행되는 창만 계획한다** (`execution_length`).
#: 숫자로 적지 않는 이유가 있다: 실행 길이는 `pi05_infer.py` 의 `OPEN_LOOP_HORIZON` 에서 오고
#: 이 패키지에서는 `horizon.execution_length` 하나가 그것을 들고 있다. 같은 8 을 두 번째 자리에
#: 박으면 한쪽만 바뀌는 날이 오고, 그때 최적화기는 실행되지 않는 스텝을 계획하거나 실행되는
#: 스텝을 계획하지 않는다 — 둘 다 조용하다.
PLAN_EXECUTION_WINDOW = "execution"


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
    # Lookahead past the executed window buys foresight — an optimizer that can only see the window
    # it is about to run will steer into a corner it cannot get out of. **It also buys somewhere to
    # hide.** T6d measured what that costs on the real task: the refined chunk avoided the apple in
    # its leading steps (+89.86 mm from the fingertip) and only approached it in steps that never
    # execute (−17.61 mm over all 50). Those leading steps are the only ones the executor runs, so
    # every chunk re-planned the same dodge and the apple never moved (0.0 mm in four closed-loop
    # runs, against 245.2 mm for the policy's own chunk).
    #
    # `PLAN_EXECUTION_WINDOW` — the default since T6f — plans exactly the window that executes, so
    # there is no unexecuted tail to defer the approach into. An integer or `None` restores the older
    # behaviour (`None` plans the whole chunk); nothing else in the package reads this field, so the
    # revert is this one value.
    plan_horizon: int | str | None = PLAN_EXECUTION_WINDOW

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
        if isinstance(self.plan_horizon, str) and self.plan_horizon != PLAN_EXECUTION_WINDOW:
            raise TrajOptConfigError(
                f"horizon.plan_horizon must be an int, None, or {PLAN_EXECUTION_WINDOW!r}; got "
                f"{self.plan_horizon!r}. A misspelled sentinel that fell through to "
                "'plan the whole chunk' would be the T6d failure again, silently"
            )
        if self.plan_horizon is not None:
            planned = self.planned
            if not 0 < planned <= self.horizon:
                raise TrajOptConfigError(
                    f"horizon.plan_horizon must satisfy 0 < P <= H; got P={planned}, "
                    f"H={self.horizon}"
                )
            if planned < self.execution_length:
                raise TrajOptConfigError(
                    f"horizon.plan_horizon ({self.plan_horizon}) is shorter than "
                    f"execution_length ({self.execution_length}): the optimizer would leave part of "
                    "what actually executes unplanned"
                )

    @property
    def planned(self) -> int:
        """Steps the optimizer plans — and therefore the steps it refines.

        In this package those are one number, not two: `refiner.refine` slices the reference to
        `planned`, `trajectory_to_chunk` writes back only those steps, and `sqp._finish` measures
        `max_violation` over the same span. So "shrink the refined window" and "shrink the plan
        horizon" are the same edit.

        `horizon` when `plan_horizon` is None; `execution_length` for `PLAN_EXECUTION_WINDOW`.
        """
        if self.plan_horizon is None:
            return self.horizon
        if self.plan_horizon == PLAN_EXECUTION_WINDOW:
            return min(int(self.execution_length), int(self.horizon))
        return int(self.plan_horizon)

    @property
    def plans_execution_window_only(self) -> bool:
        """Is the whole planned window executed? Then no step of it can be deferred away."""
        return self.planned <= self.execution_length


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
    # Position smoothing: 2nd difference = acceleration in step units; jerk would be the 3rd.
    # (`problem.objective` is `np.diff(Q, n=2)`. The key keeps its name for flag compatibility —
    # `--w-smooth` and recorded configs use it — but it has never penalised jerk. T24, guide §7.1.)
    w_smooth: float = 0.05
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


#: `LimitsConfig.source` 의 값 (T31 G1). **joint position 범위를 어디서 읽나.**
#:
#: | 값 | 범위 | 쓰는 곳 |
#: |---|---|---|
#: | `model_xml` (기본) | 제어 대상 MJCF 의 `jnt_range` (position actuator `ctrlrange` 와 대조) | 시뮬레이션 — 정책이 학습된 로봇 |
#: | `urdf` | `robot_model.joint_limits()` (URDF `<limit lower upper>`) | T31 전 동작. 비교·재현용 |
#: | `config` | `limits.position_ranges` 에 적은 표 | 실기 — 그 로봇의 실제 범위를 사람이 적는다 |
#:
#: URDF 는 **형상**(구 모델·FK)의 근거로 남는다. 속도·가속도 한계는 셋 모두 URDF 에서 온다 —
#: MJCF 에는 관절 속도·가속도 한계가 없다 (`limits.py` 머리말).
LIMIT_SOURCES = ("model_xml", "urdf", "config")

#: `limits.model_xml` 이 비었을 때 읽는 MJCF — 시뮬레이션의 로봇
#: (`ag3s/experiments/sources/mujoco_source.TRANSPORT_MODEL` 과 같은 파일, 테스트가 박는다).
#: 작업공간 상대경로이고 `ag3s/runtime/asset_path.resolve_asset` 가 이 머신의 실제 자리를 찾는다.
DEFAULT_LIMITS_MODEL_XML = "src/rby1_description/models/rby1a/mujoco/model_transport.xml"


@dataclasses.dataclass(frozen=True)
class LimitsConfig:
    """How the robot's own limits are applied.

    The scales are there to be conservative, not to be creative. A robot commanded to its exact
    velocity limit has no headroom for the tracking controller underneath, so 0.9 leaves some. The
    position margin keeps the optimizer off the hard stop for the same reason.

    **Position 범위의 출처는 `source` 다** (T31 G1). T31 전에는 URDF 였고, 시뮬레이션의 로봇
    (`model_transport.xml`)은 손목 `arm_6` 이 ±2.967 인데 URDF 는 ±2.705 라 정책이 학습한 자세를
    TO 가 2.685 에서 잘랐다 (E3 ep1800·1808). 속도·가속도는 여전히 URDF 다.
    """

    #: position 범위의 출처 — `LIMIT_SOURCES`. 기본 `model_xml` (시뮬레이션).
    source: str = "model_xml"
    #: `source == "model_xml"` 일 때 읽는 MJCF. `None` 이면 `DEFAULT_LIMITS_MODEL_XML`.
    #: 서버는 `--model-xml` 을 여기 넣는다 (시뮬레이터가 돌리는 바로 그 파일).
    model_xml: str | None = None
    #: `source == "config"` 일 때만: ``{joint_name: [lower, upper]}`` (rad). TO 가 움직이는 관절은
    #: **전부** 적어야 한다 — 빠진 관절을 URDF 로 채우면 무엇이 어디서 왔는지 다시 섞인다.
    position_ranges: Mapping[str, Any] | None = None
    velocity_scale: float = 0.9
    acceleration_scale: float = 0.9
    position_margin: float = 0.02  # rad kept clear of each joint's hard stop
    # Joints whose URDF omits a velocity/acceleration limit. None means "raise instead of guessing";
    # see `UrdfSphereChain.velocity_limits`.
    default_velocity: float | None = None
    default_acceleration: float | None = None
    enforce_velocity: bool = True
    enforce_acceleration: bool = True
    #: **joint position box 를 거는가** (T27). `False` 면 `lower/upper` 가 `∓inf` 가 되어 box 행에는
    #: trust region(과 anchor)만 남는다. 기본 `True` — 지금까지의 TO 다.
    enforce_position: bool = True
    #: `enforce_velocity=False` 일 때만 뜻이 있다 (T27). `True` 면 **첫 스텝 anchor**
    #: ``|Q[:, 0] - q_now| <= v_max·dt`` 는 남기고 **스텝 사이 속도 행만** 뺀다. anchor 는 SQP 가
    #: 로봇이 지금 있는 곳에서 계획을 시작하게 하는 장치이고, `sqp._limit_overshoot` 가 초기 iterate
    #: 의 적격성을 그것으로 잰다 — 그래서 anchor 는 `max_step` 을 **유한하게 둔 채** 속도 행만
    #: 끄는 방식으로 남긴다 (`limits.RelaxedJointLimits`).
    #:
    #: 기본 `False` 는 예전 동작 그대로다: `enforce_velocity=False` 면 `max_step = inf` 이고 anchor 도
    #: 함께 사라진다 (`tests/trajopt/test_limits.py` 가 그것을 박고 있다).
    keep_anchor: bool = False

    @property
    def enforces_nothing(self) -> bool:
        """position · velocity · acceleration 을 **하나도** 걸지 않는가 (`--no-limits`, T27)."""
        return not (self.enforce_position or self.enforce_velocity or self.enforce_acceleration)

    def validate(self) -> None:
        for name in ("velocity_scale", "acceleration_scale"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise TrajOptConfigError(f"limits.{name} must be in (0, 1], got {value}")
        if self.position_margin < 0.0:
            raise TrajOptConfigError(
                f"limits.position_margin must be >= 0, got {self.position_margin}"
            )
        if self.source not in LIMIT_SOURCES:
            raise TrajOptConfigError(
                f"limits.source must be one of {LIMIT_SOURCES}, got {self.source!r}")
        if self.source == "config":
            if not self.position_ranges:
                raise TrajOptConfigError(
                    "limits.source='config' needs limits.position_ranges "
                    "({joint: [lower, upper]}) — a config source with no table would fall back to "
                    "something nobody wrote down")
            for name, pair in dict(self.position_ranges).items():
                try:
                    lo, hi = (float(v) for v in pair)
                except (TypeError, ValueError):
                    raise TrajOptConfigError(
                        f"limits.position_ranges[{name!r}] must be [lower, upper], got {pair!r}"
                    ) from None
                if not lo < hi:
                    raise TrajOptConfigError(
                        f"limits.position_ranges[{name!r}] must satisfy lower < upper, got "
                        f"[{lo}, {hi}]")
        elif self.position_ranges:
            # **조용히 무시하지 않는다** (이 모듈 머리말) — 표를 적었는데 다른 출처가 쓰이면 그
            # 실행은 적힌 표로 돈 것처럼 읽힌다.
            raise TrajOptConfigError(
                f"limits.position_ranges is only read when limits.source='config' "
                f"(got source={self.source!r})")


#: `--no-limits` 가 `limits` 절에 얹는 것 (T27). **한 곳에만 적는다** — 서버·테스트·기록이 같은
#: dict 를 본다. 무엇이 남는지: SQP 의 trust region (`sqp.trust_radius`, box 행 안) 과 첫 스텝
#: anchor (`keep_anchor`). 무엇이 빠지는지: joint position box · 스텝 사이 속도 행 · 가속도 행.
NO_LIMITS: dict[str, bool] = {
    "enforce_position": False,
    "enforce_velocity": False,
    "enforce_acceleration": False,
    "keep_anchor": True,
}


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

    **T31 — 반복 수 우선.** 위 계산은 한 반복이 수 ms 이던 primitive backend 의 것이다. ESDF
    backend 의 E3 에서는 linearize 하나가 ~100 ms 라 예산 50 ms 가 언제나 첫 반복 뒤에 걸렸다.
    `min_iterations` 만큼은 예산과 무관하게 돌고, 그 뒤부터 `time_budget_ms` 가 걸린다.
    `max_iterations` 가 `min_iterations` 보다 작으면 `max_iterations` 가 이긴다 (상한은 상한이다).
    """

    max_iterations: int = 3
    time_budget_ms: float = 50.0
    #: **벽시계 예산이 걸리기 전에 반드시 도는 SQP 반복 수** (T31 G2-i). 예산은 이 수만큼 돈 뒤부터
    #: 검사된다 — `max_iterations` 는 여전히 상한이고, 수렴(`step_tolerance`)·QP 실패로 더 일찍
    #: 끝날 수는 있다. 1 이면 T31 전 동작(첫 반복 뒤 곧바로 예산 검사)이다.
    #:
    #: 기본 3 (= `max_iterations`) 의 근거 — 둘로 나뉜다.
    #:
    #: 1. T30 E3 서버 기록 1125 청크 (E3a 675 · E3b 450, 각 run 첫 청크 제외): linearize 중앙값
    #:    92.5–100.2 ms (p90 115–183), assemble 2.9 · QP 2.1–2.2 · check 4.6–4.7 ms. 예산 50 ms 가
    #:    첫 반복보다 작아서 **1124/1125 청크가 1 반복**이었다.
    #: 2. 그 ~100 ms 는 **청크마다 새 거리장에 대한 첫 linearize** 의 값이다. 같은 입력(E3b chunk
    #:    302–311, T24 도구로 복원)을 3 반복으로 풀면 linearize 는 호출마다 121–145 ms · 2.7–3.2 ·
    #:    2.7–3.0 ms 다 (이 머신). 2·3 번째 반복은 QP·check 를 합쳐 ≈ 10 ms 씩이다.
    #:
    #: 같은 도구로 E3b 1800 r1 · 1808 r1 의 pre-grasp 150 청크를 1·2·3 반복(시계 끔, MJCF 한계)으로
    #: 풀면: 반환 merit / 초기 merit 중앙값 1.000 · 0.841 · 0.742, 충돌 위반 청크 0 · 0 · 0,
    #: 관절 편차(실행 창, °) 중앙값 0.00 · 0.95 · 1.10 / 최대 7.16 · 7.05 · 7.09. 그래서 3 반복은
    #: ≈ 20 ms 로 설계대로의 SQP(`max_iterations=3`)를 되돌리고, 반복 수가 부하와 무관하게 정해진다
    #: (시뮬레이션 재현성). 시뮬레이션 루프는 동기식이라 이 시간은 제어 주기를 깨지 않는다.
    #: **실기처럼 비동기로 도는 곳은 1 로 되돌린다** (`serve_safe --sqp-min-iterations 1`).
    min_iterations: int = 3
    #: 초기 iterate 가 **limit 만** 어기고 (충돌 위반 없음) 받아들여진 QP 후보가 없을 때, 거절된
    #: 후보 대신 **reference 의 최소 투영** (`limits.project_to_limits`: position box ∩ 첫 스텝
    #: anchor) 을 반환한다 — 그 투영이 limit 을 전부 지키고 충돌 검사를 통과할 때만 (T31 G2-ii).
    #: `False` 는 T24 규칙 그대로 (`best_unaccepted`). ablation 용 스위치다.
    limit_projection_fallback: bool = True
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
        if self.min_iterations < 1:
            raise TrajOptConfigError(
                f"sqp.min_iterations must be >= 1, got {self.min_iterations} — one step is the "
                "real-time iteration scheme's minimum unit of work")
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
class CollisionBackendConfig:
    """Which collision representation the optimizer consumes.

    `primitive` reads the sphere/plane blocks AG3S already packs into `ConstraintSpec`. `esdf` reads
    the distance field instead, one row per (step, robot sphere):

        d_esdf(p(q)) - collision_radius - safety_margin >= 0

    `both` enables both, which is the point of keeping the old path: the two describe the same scene,
    so an ablation can put them in the same QP and compare row for row rather than across runs.

    `esdf_margin` is deliberately separate from AG3S's `d_safe`. The primitive path already carries a
    per-(sphere, slot) margin from `ClearancePolicy`, and the ESDF has no slots to look it up by — a
    field answers about a *point*. Reusing the same number would either lose the phase-dependent
    relaxation or apply the grasp relaxation to every link at once, and the second is the mistake the
    clearance policy exists to prevent. Until the field carries a per-voxel semantic label, this stays
    one conservative number.
    """

    backend: str = "primitive"  # primitive | esdf | both
    esdf_margin: float = 0.05  # metres; the conservative clearance, matching geometry.safety_margin
    #: Consume the support-surface half-space rows. Turning this off does **not** stop AG3S from
    #: extracting the planes — target grounding needs them as an exclusion mask or region growing
    #: floods across the table and finds one cluster the size of the scene. It only stops the
    #: optimizer from reading them as constraints, which is what "let the field be the whole
    #: constraint" means: the table is then geometry in the ESDF like anything else.
    #:
    #: Leave it on to keep the older split, where a plane is one exact linear row at 10 mm and the
    #: field carries only what is not a plane. The two are a real trade: a half-space is exact and
    #: cheap but **unbounded**, so a tabletop plane forbids the entire volume beneath it — measured
    #: at 57 of 71 robot spheres in nominal violation on RB-Y1, all of them base and wheels resting
    #: on the floor.
    #:
    #: **Default off** (F15, 2026-09-14), paired with `AG3SConfig.esdf.exclude_support_surfaces`
    #: which also flipped. Both real deployments already ran this way. The pair must agree — carve
    #: and the plane row catches it, or leave it in the field and read no plane row — so
    #: `scene_from_constraint_set` refuses the combination that constrains a support surface with
    #: nothing (carved out of the field *and* no plane row).
    use_support_planes: bool = False
    #: **충돌 제약을 최적화기에 보이는가.** `True` 가 기본이고 그것이 지금까지의 TO 다.
    #:
    #: `False` 는 사용자가 요청한 ablation 이다 (*"TO 에 가장 기본적인 jerk 목적함수만 포함하고
    #: 나머지 제약을 모두 제거해 봐. 충돌 제약을 제거한 후에도 사과를 못 집으면 이건
    #: 궤적최적화의 문제로 볼 수 있지 않나?"*). 그 판정표가 이것이다 — 인용 속 "jerk" 는
    #: 요청 당시의 이름이고, 실제 항(`w_smooth`)은 **위치 2 차 차분 = 가속도**다 (jerk 는 3 차
    #: 차분이다. T24, 지침 §7.1):
    #:
    #: | `w_smooth` | 충돌 | 최적해가 무엇이어야 하나 | 아니면 |
    #: |---|---|---|---|
    #: | 0 | 없음 | **reference 그 자체** | optimizer 자체의 문제 (SQP 수렴·limit) |
    #: | 0.05 | 없음 | 2 차 차분(가속도) 항이 미는 만큼만 다름 | 평활 항의 기여를 격리 |
    #:
    #: **`--no-safe` 와 다르다.** 그쪽은 TO 를 아예 안 돌린다 (정책 청크가 그대로 나간다).
    #: 이것은 TO 가 **그대로 돌면서** 충돌 행만 없는 것이다 — 추적·평활(2 차 차분)·연속성·limit·
    #: SQP·trust region 이 전부 그대로다.
    #:
    #: **판정은 거짓말하지 않는다.** 여유거리 측정(`worst_row`)은 이 값과 무관하게 돌아가므로
    #: `max_violation_m` 은 여전히 참값이다. 꺼진 것은 **최적화기가 보는 것**이고, 그 사실은
    #: `notes` 와 `metrics["collision_enabled"]` 에 실린다 — 이 설정으로 돈 기록이 나중에
    #: "위반 0" 으로 읽히면 안 된다.
    enabled: bool = True

    BACKENDS = ("primitive", "esdf", "both")

    def validate(self) -> None:
        if self.backend not in self.BACKENDS:
            raise ValueError(f"collision.backend must be one of {self.BACKENDS}, got {self.backend!r}")
        if self.esdf_margin < 0.0:
            raise ValueError(f"collision.esdf_margin must be >= 0, got {self.esdf_margin}")

    @property
    def wants_esdf(self) -> bool:
        return self.backend in ("esdf", "both")

    @property
    def wants_primitive(self) -> bool:
        return self.backend in ("primitive", "both")


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
    "collision": CollisionBackendConfig,
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
    collision: CollisionBackendConfig = dataclasses.field(
        default_factory=CollisionBackendConfig
    )

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
    "DEFAULT_LIMITS_MODEL_XML",
    "LIMIT_SOURCES",
    "NO_LIMITS",
    "PLAN_EXECUTION_WINDOW",
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
