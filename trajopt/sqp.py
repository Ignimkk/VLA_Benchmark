"""The real-time iteration loop: linearize, solve, accept, re-check.

Collision avoidance is not convex — ``||p(q) - c|| >= R`` carves a hole out of the configuration
space — so the problem is solved as a sequence of convex ones. Each iteration linearizes the
clearances at the current trajectory, solves a QP inside a trust region, and either accepts the step
or shrinks the region and tries again.

**The budget is a deadline, not a suggestion.** `sqp.max_iterations` and `sqp.time_budget_ms` are
both hard caps, and whichever binds first ends the loop with the best iterate so far. That is the
real-time iteration scheme's central idea: at 15 Hz the optimizer runs once per chunk while the
control loop waits, so an algorithm that runs until it converges cannot be used at all. What makes
the truncation safe is not that the answer is optimal — it is that the answer is *checked*.

**T31 — iterations first.** The deadline is checked only after `sqp.min_iterations` iterations
(default 3). With the ESDF backend the first linearization on a new chunk's field costs ~100 ms
(later ones ~3 ms), so a 50 ms deadline checked after the first iteration meant *one QP per chunk,
always* — a rejected first step was returned without the retry the trust region exists to give it. The simulation loop is synchronous; a real
robot that needs the old wall-clock behaviour sets `min_iterations: 1`.

**A limit-only initial iterate falls back to its projection, not to a rejected step** (T31). When the
reference breaks only the joint limits (no collision violation) and no QP candidate is accepted, the
returned trajectory is the reference clipped onto the position box and the step-0 anchor — provided
that clip satisfies every limit and passes the full-resolution collision check. The rejected QP
candidate it replaces moved every joint (measured on E3b chunk 310: up to 8.7° on the wrist that
was out of range, and 0.4–0.9° on joints that were not).

**Every returned trajectory is checked at full resolution.** `linearize` may hand the QP a small
subset of the rows; `_finish` re-evaluates all of them. That is the whole basis on which the
reduction is allowed to exist: pruning may cost convergence, and it may not hide a collision. The
check costs about a twelfth of a solve, because it needs no Jacobian.

**A trajectory that could not be made safe is still returned.** `TrajOptStatus.VIOLATED` carries the
best effort together with the worst remaining violation in metres. Raising instead would leave the
controller with nothing to send at the exact moment it most needs something, which is how a safety
layer ends up switched off in production. Deciding between stopping, holding and re-planning belongs
to whoever owns the task — the same division of labour AG3S keeps with `ConstraintValidity`.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional

import numpy as np

from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.linearize import (
    CollisionBlock,
    CollisionLinearizer,
    SceneSnapshot,
    append_collision_rows,
)
from benchmark.trajopt.limits import (
    limit_report,
    project_to_limits,
    rest_start_envelope,
    velocity_rows,
)
from benchmark.trajopt.problem import build_problem, objective
from benchmark.trajopt.qp import QpSolver
from benchmark.trajopt.types import ChunkLayout, JointLimits, TrajOptResult, TrajOptStatus


#: 초기 iterate 가 "한계 안" 인가의 문턱 (rad 또는 rad/step). `_finish` 가 limit overshoot 를
#: 노트에 적는 문턱과 같다 — 두 곳이 다르면 "한계 안" 의 뜻이 갈라진다.
_LIMIT_TOLERANCE = 1e-6


class TrajectoryOptimizer:
    """Stateful across chunks, because that is where the warm start comes from.

    The QP backend keeps its factorization, and the previous solution seeds the next chunk's first
    iterate. Both are only valid while the problem's sparsity pattern holds, which is why the row
    budget is per step and fixed rather than per scene — see `linearize`.
    """

    def __init__(
        self,
        robot_model,
        layout: ChunkLayout,
        limits: JointLimits,
        config: Optional[TrajOptConfig] = None,
    ):
        self.config = config or TrajOptConfig()
        self.layout = layout
        self.limits = limits
        self.horizon = self.config.horizon.horizon
        self.linearizer = CollisionLinearizer(robot_model, layout, self.horizon)
        self.block = CollisionBlock(
            self.horizon, self.config.reduction.rows_per_step, layout.nq_opt
        )
        self.solver = QpSolver(self.config.qp)
        self._previous: Optional[np.ndarray] = None
        self.frame_index = 0
        #: T43 Q — servo model (`limits.servo_model`) and the rows between waypoints
        #: (`reduction.sweep_check`). Both `None` when both are off: nothing below reads them then.
        self.servo = None
        self.observer = None
        self.path = None
        if (getattr(self.config.limits, "servo_model", False)
                or getattr(self.config.reduction, "sweep_check", False)):
            self._build_path(robot_model)
        #: T43 TA (A) — first look-ahead step, or None (every planned step certified, as before).
        self.lookahead_from = self.config.horizon.lookahead_from
        self.linearizer.lookahead_from = self.lookahead_from
        #: T43 TA (A) — the gate's own optimizer: the executed window alone, every other setting
        #: as this one (`horizon.lookahead_gate`). `None` without look-ahead or without a gate.
        self.window_optimizer: Optional["TrajectoryOptimizer"] = None
        self.lookahead_gate = (None if self.lookahead_from is None
                               else self.config.horizon.lookahead_gate)
        if self.lookahead_gate is not None:
            k = int(self.lookahead_from)
            window_cfg = self.config.with_overrides(
                {"horizon": {"horizon": k, "execution_length": k, "plan_horizon": None,
                             "lookahead": "off"}})
            self.window_optimizer = TrajectoryOptimizer(robot_model, layout, limits, window_cfg)
        #: T43 TA (C) — `a_C` per optimized joint (rad/row²), or None when `limits.servo_accel` is off.
        self.servo_accel_bound = None
        self.servo_accel_record = None
        if getattr(self.config.limits, "servo_accel", "off") != "off":
            self._build_servo_accel(robot_model)
        #: T43 TA (B, C) — this solve's history (`solve(deflection_history=, command_history=)`).
        self._history: dict[str, Any] = {}

    def _build_path(self, robot_model) -> None:
        """T43 Q — the servo (from the controlled MJCF) and the path checker. Only when on."""
        from benchmark.trajopt.sweep import PathChecker

        cfg = self.config
        if getattr(cfg.limits, "servo_model", False):
            from benchmark.trajopt.limits import limits_model_xml
            from benchmark.trajopt.servo import ServoModel, ServoObserver, servo_params_from_xml

            names = [str(robot_model.joint_names[int(i)]) for i in self.layout.q_indices]
            params = servo_params_from_xml(limits_model_xml(cfg.limits), names,
                                           control_hz=cfg.horizon.control_hz)
            self.servo = ServoModel(params)
            self.observer = ServoObserver(self.servo)
        self.path = PathChecker(self.linearizer, self.layout, cfg, self.servo)
        self.path.warm_seconds = self.path.warm()

    def _build_servo_accel(self, robot_model) -> None:
        """T43 TA (C) — `a_C = ε Δt² / κ` from the controlled MJCF (`servo.ServoParams.accel_bound`)."""
        from benchmark.trajopt.limits import limits_model_xml
        from benchmark.trajopt.servo import servo_params_from_xml

        cfg = self.config
        names = [str(robot_model.joint_names[int(i)]) for i in self.layout.q_indices]
        params = servo_params_from_xml(limits_model_xml(cfg.limits), names,
                                       control_hz=cfg.horizon.control_hz)
        tol = float(cfg.limits.servo_accel_tolerance)
        bound = params.accel_bound(tol, row_seconds=cfg.horizon.dt)
        self.servo_accel_bound = np.asarray(bound, np.float64)
        self.servo_accel_record = {
            "mode": str(cfg.limits.servo_accel), "tolerance_rad": tol,
            "row_seconds": float(cfg.horizon.dt),
            "shape_coefficient_s2": [round(float(v), 6) for v in params.shape_coefficient()],
            "a_c_rad_per_row2": [round(float(v), 6) for v in bound],
            "joint_names": names, "weight": float(cfg.limits.servo_accel_weight)}

    def _max_step_change(self) -> Optional[np.ndarray]:
        step = np.asarray(self.limits.max_step_change, np.float64)
        return step if np.all(np.isfinite(step)) else None

    def _objective(self, trajectory, reference, previous_chunk) -> float:
        """`problem.objective` with this solve's T43 TA history (identical call when TA is off)."""
        extra: dict[str, Any] = {}
        if self._history.get("deflection") is not None:
            extra["deflection_history"] = self._history["deflection"]
        if self.servo_accel_bound is not None:
            step = np.asarray(self.limits.max_step, np.float64)
            extra.update(servo_accel_bound=self.servo_accel_bound,
                         command_history=self._history.get("command"),
                         max_step_change=self._max_step_change(),
                         max_step=step if velocity_rows(self.limits) else None)
        return objective(trajectory, reference, self.config, previous_chunk=previous_chunk, **extra)

    def reset(self) -> None:
        """Clear cross-chunk state. Between episodes, never mid-episode."""
        self.solver.reset()
        self._previous = None
        self.frame_index = 0
        if self.observer is not None:
            self.observer.reset()
        if self.window_optimizer is not None:
            self.window_optimizer.reset()

    # ------------------------------------------------------------------------------------
    def solve(
        self,
        reference: np.ndarray,
        q_now: np.ndarray,
        scene: Optional[SceneSnapshot],
        *,
        previous_chunk: Optional[np.ndarray] = None,
        template: Optional[np.ndarray] = None,
        geometry_certified: bool = True,
        start_at_rest: bool = False,
        servo_feedback: Optional[dict[str, Any]] = None,
        deflection_history: Optional[np.ndarray] = None,
        command_history: Optional[np.ndarray] = None,
    ) -> TrajOptResult:
        """Make `reference` safe against `scene`, or report how far short it fell.

        Args:
            reference: ``(nq_opt, H)`` — SEAM's chunk in joint space.
            q_now: ``(nq_model,)`` the robot's current configuration. Positions the fixed joints and
                anchors the first step.
            scene: AG3S's decoded constraint set, or None when there is nothing to avoid.
            previous_chunk: the overlapping tail of the last refined chunk, for continuity.
            template: the original ``(H, action_dim)`` chunk, so grippers and padding pass through
                untouched. Defaults to zeros, which is only right when the caller has no chunk.
            geometry_certified: AG3S's own verdict on the scene. A frame it could not certify may
                still be optimized against; the result simply cannot be called safe.
            start_at_rest: T41 b. `True` = the robot is known to be at rest at `q_now` (the previous
                chunk was held throughout). The step-0 anchor becomes the reach-from-rest envelope
                (`limits.rest_start_envelope`, `config.limits.rest_start_tolerance`) on every step —
                in the QP box, in the eligibility of the initial iterate and in the limit projection.
                `False` (default) leaves every number as it was. Who decides: `SafePolicy`, only when
                `limits.rest_start` is on.
            servo_feedback: T43 Q. ``{"feedback": exec_feedback, "seq": request seq}`` — what the
                servo observer propagates to estimate the arm's velocity at `q_now`
                (`servo.ServoObserver`). Read only when `limits.servo_model` is on; without it the
                arm is taken to be at rest and the record says so.
            deflection_history: T43 TA (B). ``(nq_opt,)`` deflection of the last executed command
                (`cost.w_deflection_rate`); `None` = no boundary term.
            command_history: T43 TA (C). ``(nq_opt, 2)`` the two last executed commands, oldest
                first (`limits.servo_accel`); `None` = the C rows stay inside the chunk.
        """
        started = time.perf_counter()
        cfg = self.config
        reference = np.asarray(reference, np.float64)
        q_now = np.asarray(q_now, np.float64).reshape(-1)
        notes: list[str] = []
        if self.lookahead_from is not None and previous_chunk is not None:
            # T43 TA (A): continuity only over the executed window, as without look-ahead. Pulling
            # the tail toward the previous chunk's stale prediction moved chunks that collide with
            # nothing (offline V8: 8 of 35 closing-set chunks the base left untouched).
            previous_chunk = np.asarray(previous_chunk, np.float64)[:, :self.lookahead_from]
        # T43 TA — only what is switched on is kept; with B and C off this stays empty.
        self._history = {}
        if deflection_history is not None and float(cfg.cost.w_deflection_rate) > 0.0:
            self._history["deflection"] = np.asarray(deflection_history, np.float64).reshape(-1)
        if command_history is not None and self.servo_accel_bound is not None:
            self._history["command"] = np.asarray(command_history, np.float64).reshape(
                reference.shape[0], -1)
        #: T41 b — `None` 이면 예전 anchor (첫 스텝 `max_step`) 그대로.
        envelope = (rest_start_envelope(self.limits, self.horizon,
                                        getattr(cfg.limits, "rest_start_tolerance", None))
                    if start_at_rest else None)

        servo_state = None
        if self.observer is not None:
            fb = (servo_feedback or {}).get("feedback")
            seq = (servo_feedback or {}).get("seq")
            servo_state = self.observer.update(q_now[self.layout.q_indices], fb, seq)

        if scene is None or scene.is_empty:
            return self._passthrough(
                reference, q_now, template, TrajOptStatus.UNCONSTRAINED,
                notes + ["AG3S emitted no geometry to avoid; the reference passes through unchanged"],
                started,
            )

        self._gate_record = None
        if self.window_optimizer is not None:
            gate = self._lookahead_gate(reference, q_now, scene)
            self._gate_record = {x: v for x, v in gate.items() if x != "open"}
            if not gate["open"]:
                return self._solve_window(
                    gate, reference, q_now, scene, previous_chunk=previous_chunk,
                    template=template, geometry_certified=geometry_certified,
                    start_at_rest=start_at_rest, servo_feedback=servo_feedback,
                    deflection_history=deflection_history, command_history=command_history)

        n_path = self.path.n_slack if self.path is not None else 0
        n_slack = self.block.n_rows + n_path
        iterate = reference.copy()
        if (cfg.sqp.warm_start and self._previous is not None
                and cfg.horizon.execution_length < self.horizon
                and self.lookahead_from is None):
            # Seed from the previous chunk shifted by the steps that have executed since. The tail is
            # held rather than extrapolated: extrapolating a trajectory past its horizon invents
            # motion the policy never proposed.
            #
            # **Skipped when the whole planned window has executed** (`execution_length >= horizon`,
            # which is what `PLAN_EXECUTION_WINDOW` produces): there is then no overlap between the
            # last plan and this one, and the shift would blend in `self._previous[:, -1]` held
            # constant — a pose from before the executed window, pulling the first iterate backwards.
            # With no overlap the reference *is* the best available start.
            k = min(cfg.horizon.execution_length, self.horizon - 1)
            shifted = np.concatenate(
                [self._previous[:, k:], np.tile(self._previous[:, -1:], (1, k))], axis=1
            )
            iterate = 0.5 * (iterate + shifted)

        # 쥔 물체를 질의점으로 올린다. 매 청크 다시 거는 것은 파지가 청크 사이에 시작되고 끝나기
        # 때문이고, 값이 같으면 하는 일이 없다 — 심볼릭 그래프는 링크마다 한 번만 만들어진다.
        self.linearizer.set_attached(
            getattr(scene, "attached_points", None), getattr(scene, "attached_parent_link", None)
        )
        if self.lookahead_from is not None:
            # T43 TA (A): what the reference does in the executed window — the certified span.
            reference_violation = -float(self.linearizer.step_clearances(
                reference, q_now, scene)[:self.lookahead_from].min(initial=np.inf))
        else:
            reference_violation = -self.linearizer.full_violation(reference, q_now, scene)
        if self.path is not None:
            # T43 Q — floors at `q_now` and the servo state, once per chunk (before any merit).
            self.path.begin(q_now, scene, servo_state)
        # Forward kinematics for a whole chunk is the most expensive thing left in the loop, so each
        # trajectory's sphere states are computed once and carried: the iterate's states feed both
        # the linearization and its merit, and an accepted candidate hands its states to the next
        # iteration rather than being recomputed there.
        iterate_states = self.linearizer.sphere_states(iterate, q_now)
        iterate_merit, iterate_violation = self._merit_terms(
            iterate, reference, previous_chunk, q_now, scene, states=iterate_states
        )
        # **초기 iterate 도 반환 후보다** (T24, 지침 §7.3). 예전에는 `best_merit = np.inf` 로
        # 시작해서, 첫 QP 후보가 초기 궤적보다 merit 이 나빠 반복점으로 **거절**되더라도 반환용
        # best 로는 저장됐다. 예산으로 끝나면 (T14 30/30 · T17 75/75 chunk 가 그랬다) 그 거절된
        # 후보가 나갔다. 이제 초기 iterate 와 그 merit 이 기준이고, 그보다 나쁜 후보는 반환되지
        # 않는다.
        #
        # **단, 초기 iterate 가 로봇 한계 안에 있을 때만이다.** merit 은 비용 + 충돌 위반이고
        # **관절 한계를 보지 않는다** — 한계는 QP 의 행(box·anchor 는 hard, 속도·가속도는 soft)이
        # 지킨다. QP 후보는 그 행을 거쳐 나오지만 초기 iterate 는 정책이 낸 그대로다. 그래서
        # reference 가 joint box 밖이면 box 안으로 끌어들인 후보가 추적 비용만큼 merit 이 나빠
        # **언제나 진다** — merit 만으로 고르면 한계를 넘는 reference 가 그대로 나간다. 두 후보의
        # 적격성이 같지 않은 것이다 (지침 §7.3 "모든 후보의 적격성이 같다고 가정하지 않는다").
        # 한계를 넘는 초기 iterate 는 비교 대상에서 빠지고 (예전 규칙 그대로), QP 가 하나도 못
        # 풀었을 때의 마지막 대비로만 남는다. **T31**: 그 초기 iterate 가 limit **만** 어겼고
        # 받아들여진 후보가 없으면, 거절된 후보 대신 reference 의 최소 투영이 나간다
        # (`_limit_projection`, 루프 뒤).
        #
        # 적격한 초기화 아래에서 best 는 언제나 수락된 반복점이다: 후보의 merit 이 `best_merit`
        # 보다 작으면 `best_merit <= iterate_merit` 이므로 수락 조건도 만족한다. 그래서
        # `best_unaccepted` 는 **초기 iterate 가 부적격일 때만** 나온다. `returned` 는 이 성질을
        # 가정하지 않고 후보 기록에서 읽어 싣는다.
        start_record = None
        if self.path is not None:
            # T43 Q — **a second starting point: stay where the robot is.** With the path rows a
            # reference that runs through an obstacle is a poor place to linearize: the waypoint rows
            # push the commands across the obstacle while the lagging path rows pull the robot back,
            # and three SQP iterations from there end VIOLATED (T40 C1807 t=216, margin 10 mm). The
            # plan `Q[:, k] = q_now` is the one the path rows always admit for a robot at rest (every
            # row sits at its floor) and the one a HOLD would execute. It replaces the reference as
            # the start only when its merit is lower — i.e. only when the reference violates.
            iterate, iterate_states, iterate_merit, iterate_violation, start_record = (
                self._stay_start(iterate, iterate_states, iterate_merit, iterate_violation,
                                 reference, previous_chunk, q_now, scene, envelope))
        initial_merit, initial_violation = iterate_merit, iterate_violation
        initial_overshoot = self._limit_overshoot(iterate, q_now, envelope)
        initial_eligible = max(initial_overshoot.values()) <= _LIMIT_TOLERANCE
        best = iterate
        best_states = iterate_states
        best_merit = iterate_merit if initial_eligible else np.inf
        #: `-1` = 초기 iterate. 그 밖은 `candidates` 의 인덱스.
        best_index = -1
        #: QP 가 풀린 모든 후보의 `(merit, 위반, 수락)`. 반환된 것이 무엇이었는지와 그 옆에 무엇이
        #: 있었는지를 기록으로 남긴다 — 후보 선택 규칙을 나중에 다시 실행하지 않고 검산할 수 있다.
        candidates: list[dict[str, Any]] = []
        qp_failures = 0
        # Per-stage timing, accumulated rather than sampled. Where the milliseconds go is the single
        # most useful thing to know about a real-time optimizer, and it changes with the scene.
        timing = {"linearize": 0.0, "assemble": 0.0, "qp": 0.0, "check": 0.0}
        best_slack = 0.0
        iterations = 0
        qp_iterations = 0
        radius = cfg.sqp.trust_radius
        budget_bound = False
        time_budget_hit = False

        # **반복 수 우선** (T31 G2-i): 벽시계 예산은 `min_iterations` 만큼 돈 뒤부터 검사한다.
        # `max_iterations` 가 더 작으면 그것이 이긴다 — 상한은 상한이다.
        min_iterations = max(1, min(int(getattr(cfg.sqp, "min_iterations", 1)),
                                    int(cfg.sqp.max_iterations)))
        while iterations < cfg.sqp.max_iterations:
            # Checked *after* the first iteration, never before it. A budget so tight that the
            # set-up alone exhausts it would otherwise return the reference untouched while
            # reporting success — the one outcome a safety layer must not produce. One step is the
            # real-time iteration scheme's minimum unit of work, and it is always taken.
            #
            # T31: "the first iteration" became "the first `min_iterations`". In E3 the first
            # iteration costs ~105 ms (the first linearize on a new field ~100 ms; later ones ~3 ms)
            # against a 50 ms budget, so the old check ended 1124/1125 chunks after a single QP — a
            # rejected first step never got its retry.
            if (iterations >= min_iterations
                    and (time.perf_counter() - started) * 1000.0 >= cfg.sqp.time_budget_ms):
                notes.append(
                    f"time budget {cfg.sqp.time_budget_ms:.0f} ms reached after {iterations} "
                    "iteration(s); returning the best iterate so far"
                )
                time_budget_hit = True
                break
            iterations += 1

            mark = time.perf_counter()
            rows = self.linearizer.linearize(iterate, q_now, scene, cfg, states=iterate_states)
            timing["linearize"] += (time.perf_counter() - mark) * 1000.0
            mark = time.perf_counter()
            budget_bound = budget_bound or bool(rows.budget_bound_steps)
            problem = build_problem(
                reference, iterate, self.limits, cfg,
                q_now=q_now[self.layout.q_indices], previous_chunk=previous_chunk,
                n_slack=n_slack, trust_radius=radius,
                **({"anchor_envelope": envelope} if envelope is not None else {}),
                **self._problem_history(),
            )
            problem = append_collision_rows(
                problem, rows, iterate, self.block, cfg.reduction.linearization_backoff,
                **({"extra_slack": n_path} if n_path else {})
            )
            timing["assemble"] += (time.perf_counter() - mark) * 1000.0
            if self.path is not None:
                mark = time.perf_counter()
                problem = self._append_path(problem, iterate, scene)
                timing["path"] = timing.get("path", 0.0) + (time.perf_counter() - mark) * 1000.0

            mark = time.perf_counter()
            solution = self.solver.solve(problem, warm_start=cfg.sqp.warm_start)
            timing["qp"] += (time.perf_counter() - mark) * 1000.0
            qp_iterations += solution.iterations
            if not solution.success:
                notes.append(f"QP failed at iteration {iterations}: {solution.status}")
                qp_failures += 1
                radius *= cfg.sqp.trust_shrink
                if radius < cfg.sqp.trust_radius_min:
                    break
                continue

            mark = time.perf_counter()
            candidate, slack = problem.split(solution.x)
            candidate_states = self.linearizer.sphere_states(candidate, q_now)
            merit, violation = self._merit_terms(
                candidate, reference, previous_chunk, q_now, scene, states=candidate_states
            )
            timing["check"] += (time.perf_counter() - mark) * 1000.0
            step_size = float(np.abs(candidate - iterate).max())
            accepted = bool(merit <= iterate_merit)
            candidates.append({
                "iteration": int(iterations),
                "merit": float(merit),
                "violation_m": None if violation is None else float(violation),
                "accepted": accepted,
                "step_size_rad": step_size,
                "trust_radius_rad": float(radius),
            })

            if merit < best_merit:
                best, best_merit = candidate, merit
                best_slack = float(np.sum(np.abs(slack)))
                best_states = candidate_states
                best_index = len(candidates) - 1
            # A step that improved the merit earns a wider trust region; one that did not gets a
            # narrower one and another attempt from the same iterate. `iterate_merit` is carried
            # rather than recomputed: it costs a full-resolution clearance sweep, which after the QP
            # became fast is the most expensive thing left in the loop.
            if accepted:
                iterate, iterate_merit, iterate_states = candidate, merit, candidate_states
                radius = min(radius * cfg.sqp.trust_expand, cfg.sqp.trust_radius_max)
            else:
                radius = max(radius * cfg.sqp.trust_shrink, cfg.sqp.trust_radius_min)

            if step_size < cfg.sqp.step_tolerance:
                break

        # **limit 만 어긴 초기 iterate 의 대비는 reference 의 최소 투영이다** (T31 G2-ii).
        projection = None
        if not initial_eligible and not any(c["accepted"] for c in candidates):
            projection = self._limit_projection(reference, q_now, scene, initial_violation,
                                                envelope)
            if projection["used"]:
                best, best_slack = projection.pop("_trajectory"), 0.0
                best_states = projection.pop("_states")
                best_index = _PROJECTION_INDEX
                notes.append(
                    "the initial iterate was outside the joint limits only (collision violation "
                    + ("not enforced" if initial_violation is None
                       else f"{initial_violation * 1000:.2f} mm")
                    + ") and no QP candidate was accepted; returning the minimal projection of the "
                    f"reference onto the limits (max change {projection['max_change_rad']:.4f} rad)"
                    " instead of the rejected QP step")
            projection.pop("_trajectory", None)
            projection.pop("_states", None)

        if budget_bound:
            notes.append(
                f"reduction.rows_per_step={cfg.reduction.rows_per_step} bound on at least one step; "
                "some near-active constraints were not shown to the QP (the final check still sees "
                "all of them)"
            )
        selection = {
            "returned": _returned_label(best_index, candidates),
            "returned_index": int(best_index),
            "initial_merit": float(initial_merit),
            "initial_violation_m": (None if initial_violation is None
                                    else float(initial_violation)),
            "initial_eligible": bool(initial_eligible),
            "initial_limit_overshoot": {k: float(v) for k, v in initial_overshoot.items()},
            "candidates": candidates,
            "n_candidates": len(candidates),
            "n_accepted": int(sum(1 for c in candidates if c["accepted"])),
            "qp_failures": int(qp_failures),
            "min_iterations": int(min_iterations),
            "projection": projection,
            # T43 Q — only with the path rows on.
            **({"initial_source": start_record} if start_record is not None else {}),
        }
        result = self._finish(
            best, reference, q_now, scene, template, iterations, best_slack,
            reference_violation, previous_chunk, notes, started, geometry_certified,
            states=best_states, timing=timing, qp_iterations=qp_iterations,
            time_budget_hit=time_budget_hit, selection=selection,
        )
        if envelope is not None:
            result.metrics["rest_start"] = self._rest_start_record(
                best, reference, q_now, envelope)
        return result

    def _lookahead_gate(self, reference, q_now, scene) -> dict[str, Any]:
        """T43 TA (A) — does the reference come within `lookahead_gate` of an obstacle in the tail?

        The tail's rows are the look-ahead ones (obstacle class only, `_lookahead_keep`); the value is
        their worst TO clearance (`d − r − margin`) over the reference's tail steps.
        """
        self.linearizer.set_attached(
            getattr(scene, "attached_points", None), getattr(scene, "attached_parent_link", None))
        k = int(self.lookahead_from)
        per_step = self.linearizer.step_clearances(reference, q_now, scene)
        tail = per_step[k:]
        tail_min = float(tail.min(initial=np.inf))
        gate = float(self.lookahead_gate)
        return {"open": bool(tail_min < gate), "gate_m": gate,
                "tail_reference_clearance_m": tail_min if np.isfinite(tail_min) else None,
                "tail_reference_worst_step": (int(k + int(np.argmin(tail)))
                                              if tail.size and np.isfinite(tail_min) else None)}

    def _solve_window(self, gate, reference, q_now, scene, *, previous_chunk, template,
                      geometry_certified, start_at_rest, servo_feedback, deflection_history,
                      command_history) -> TrajOptResult:
        """T43 TA (A) — gate closed: solve exactly as without look-ahead (`execution_length` steps,
        `self.window_optimizer`), and hand back the reference's tail untouched."""
        k = int(self.lookahead_from)
        inner = self.window_optimizer
        path = getattr(self.linearizer, "_param_path", None)
        inner.linearizer.set_joint_parameter_path(None if path is None else path[:, :k])
        result = inner.solve(
            reference[:, :k], q_now, scene,
            previous_chunk=None if previous_chunk is None else previous_chunk[:, :k],
            template=template, geometry_certified=geometry_certified,
            start_at_rest=start_at_rest, servo_feedback=servo_feedback,
            deflection_history=deflection_history, command_history=command_history)
        trajectory = np.concatenate([np.asarray(result.trajectory, np.float64),
                                     np.asarray(reference, np.float64)[:, k:]], axis=1)
        result = dataclasses.replace(result, trajectory=trajectory)
        result.metrics["lookahead"] = {"from": k, "planned": int(reference.shape[1]),
                                       "gate": "closed", **{x: v for x, v in gate.items()
                                                            if x != "open"}}
        return result

    def _problem_history(self) -> dict[str, Any]:
        """T43 TA — the `build_problem` keywords for B and C. `{}` when both are off."""
        out: dict[str, Any] = {}
        if self._history.get("deflection") is not None:
            out["deflection_history"] = self._history["deflection"]
        if self.servo_accel_bound is not None:
            out["servo_accel_bound"] = self.servo_accel_bound
            if self._history.get("command") is not None:
                out["command_history"] = self._history["command"]
        return out

    def _append_path(self, problem, iterate, scene):
        """T43 Q — the rows between waypoints, below the collision block (their own slacks)."""
        from benchmark.trajopt.sweep import append_path_rows

        if not getattr(self.config.collision, "enabled", True):
            # collision off (T15): the same shape, every row unused — as `LinearizedRows.all_unused`.
            from benchmark.trajopt.sweep import PathRows

            H, R, nq = self.horizon, self.path.rows_per_step, self.layout.nq_opt
            rows = PathRows(np.full((H, R), np.inf), np.zeros((H, R, H, nq)), np.zeros((H, R), bool),
                            np.zeros((H, R), int), np.zeros((H, R), int), 0)
        else:
            rows = self.path.linearize(iterate, scene)
        return append_path_rows(problem, rows, iterate, self.path.block, self.block.n_rows,
                                self.config.reduction.linearization_backoff)

    def _stay_start(self, iterate, states, merit, violation, reference, previous_chunk, q_now,
                    scene, envelope):
        """T43 Q — keep the given start, or switch to ``Q[:, k] = q_now`` if that has lower merit.

        Only looked at when the given start violates (`violation > violation_tolerance`); the stay
        plan must also be inside the limits (`_limit_overshoot`), as every other candidate.
        """
        tol = self.config.safety.violation_tolerance
        record = {"source": "reference", "stay_merit": None, "stay_violation_m": None}
        if violation is None or violation <= tol:
            return iterate, states, merit, violation, record
        stay = np.tile(np.asarray(q_now, np.float64)[self.layout.q_indices][:, None],
                       (1, iterate.shape[1]))
        if max(self._limit_overshoot(stay, q_now, envelope).values()) > _LIMIT_TOLERANCE:
            record["source"] = "reference (stay outside the limits)"
            return iterate, states, merit, violation, record
        stay_states = self.linearizer.sphere_states(stay, q_now)
        stay_merit, stay_violation = self._merit_terms(stay, reference, previous_chunk, q_now,
                                                       scene, states=stay_states)
        record.update(stay_merit=float(stay_merit),
                      stay_violation_m=None if stay_violation is None else float(stay_violation))
        if stay_merit < merit:
            record["source"] = "stay"
            return stay, stay_states, stay_merit, stay_violation, record
        return iterate, states, merit, violation, record

    def _path_violation(self, trajectory, scene) -> float:
        """T43 Q — penetration (m, ≥ 0) of the rows between waypoints, below their floors."""
        worst, _ = self.path.violation(trajectory, scene)
        return max(0.0, -worst)

    def _rest_start_record(self, trajectory, reference, q_now, envelope) -> dict[str, Any]:
        """T41 b — 정지 출발 범위가 무엇이었고 반환된 계획이 어디서 시작하나 (기록용, rad)."""
        q0 = np.asarray(q_now, np.float64).reshape(-1)[self.layout.q_indices][:, None]
        traj = np.asarray(trajectory, np.float64)
        ref = np.asarray(reference, np.float64)
        n = min(traj.shape[1], envelope.shape[1])
        off = np.abs(traj[:, :n] - q0)
        return {
            "applied": True,
            "tolerance_rad": (None if getattr(self.config.limits, "rest_start_tolerance", None)
                              is None else float(self.config.limits.rest_start_tolerance)),
            "envelope_rad_max_per_step": [float(v) for v in envelope[:, :n].max(axis=0)],
            "step0_offset_rad": float(off[:, 0].max(initial=0.0)) if n else 0.0,
            "reference_step0_offset_rad": (float(np.abs(ref[:, 0] - q0[:, 0]).max(initial=0.0))
                                           if ref.shape[1] else 0.0),
            "envelope_overshoot_rad": float(np.max(off - envelope[:, :n], initial=0.0)),
        }

    # --- helpers --------------------------------------------------------------------------
    def _merit(self, trajectory, reference, previous_chunk, q_now, scene, states=None) -> float:
        """Cost plus the true constraint violation, both at full resolution.

        Deliberately *not* computed from the QP's own linearized rows. A merit function built by the
        same code that builds the subproblem agrees with it even when both are wrong; this one
        re-evaluates the geometry and so can disagree, which is the only way the step-acceptance test
        can catch a bad linearization.
        """
        return self._merit_terms(trajectory, reference, previous_chunk, q_now, scene, states)[0]

    def _merit_terms(
        self, trajectory, reference, previous_chunk, q_now, scene, states=None
    ) -> tuple[float, Optional[float]]:
        """`(merit, violation_m)` — `_merit` 과 **같은 계산**이고, 위반량을 버리지 않고 돌려줄 뿐이다.

        후보 기록(T24)이 `(merit, 위반, 수락)` 을 싣기 위해 있다. 위반을 따로 한 번 더 재면
        후보마다 clearance sweep 이 하나 늘고, 그것은 예산 안의 반복 수를 바꾼다 — 기록하려고
        동작을 바꾸는 것이 된다.

        충돌이 꺼진 판에서는 `violation_m` 이 `None` 이다: merit 이 위반을 **재지 않으므로**
        (T15), 여기서 재면 역시 루프 비용이 는다. "0" 이 아니라 "재지 않음" 이다. 반환된 궤적의
        위반은 `_finish` 가 스위치와 무관하게 잰다.
        """
        cost = self._objective(trajectory, reference, previous_chunk)
        if not self.config.collision.enabled:
            # **꺼진 판에서는 벌점도 없어야 한다** (T15). 행만 비우고 merit 에 위반을 남기면
            # 충돌이 여전히 해를 밀고, 그러면 "충돌을 껐다" 가 거짓이 된다 — QP 가 아니라 step
            # 수락 기준을 통해 미는 것뿐이다. 측정(`worst_row`)은 이것과 무관하게 계속 돈다.
            return cost, None
        if self.lookahead_from is not None:
            # T43 TA (A): the window's and the tail's worst violations, summed — neither masks
            # the other in step acceptance (a max would let a stuck tail veto a window fix).
            per_step = self.linearizer.step_clearances(trajectory, q_now, scene, states)
            k = self.lookahead_from
            violation = (max(0.0, -float(per_step[:k].min(initial=np.inf)))
                         + max(0.0, -float(per_step[k:].min(initial=np.inf))))
            return cost + self.config.cost.w_slack * violation, violation
        violation = max(0.0, -self.linearizer.full_violation(trajectory, q_now, scene, states))
        if self.path is not None:
            violation = max(violation, self._path_violation(trajectory, scene))
        return cost + self.config.cost.w_slack * violation, violation

    def _limit_overshoot(self, trajectory: np.ndarray, q_now: np.ndarray,
                         envelope: Optional[np.ndarray] = None) -> dict[str, float]:
        """`limit_report` + **anchor** — QP 가 hard 로 지키는 첫 스텝 조건까지.

        `build_problem` 은 ``|Q[:, 0] - q_now| <= max_step`` 을 box 에 넣어 hard 로 지킨다. 초기
        iterate 가 그것을 넘으면 QP 후보와 같은 자격이 아니므로 여기서 함께 잰다.

        T41 b: `envelope` (정지 출발 범위) 가 있으면 anchor 는 **모든 스텝**의
        ``|Q[:, k] - q_now| - envelope[:, k]`` 최대값이다 — QP box 가 지키는 것과 같은 조건.
        """
        report = dict(limit_report(trajectory, self.limits))
        anchor = 0.0
        step = np.asarray(self.limits.max_step, np.float64)
        if trajectory.shape[1] and np.all(np.isfinite(step)):
            q0 = np.asarray(q_now, np.float64).reshape(-1)[self.layout.q_indices]
            anchor = float(np.max(np.abs(trajectory[:, 0] - q0) - step, initial=0.0))
        if envelope is not None and trajectory.shape[1]:
            q0 = np.asarray(q_now, np.float64).reshape(-1)[self.layout.q_indices][:, None]
            n = min(trajectory.shape[1], envelope.shape[1])
            anchor = max(anchor, float(np.max(np.abs(trajectory[:, :n] - q0) - envelope[:, :n],
                                              initial=0.0)))
        report["anchor"] = max(anchor, 0.0)
        return report

    def _limit_projection(self, reference, q_now, scene, initial_violation,
                          envelope: Optional[np.ndarray] = None) -> dict[str, Any]:
        """T31 G2-ii — **reference 의 최소 투영**을 반환할 수 있는가, 그리고 그 궤적.

        | 조건 | 아니면 |
        |---|---|
        | `sqp.limit_projection_fallback` 이 켜져 있다 | `reason: disabled` |
        | 초기 iterate 의 충돌 위반 ≤ `safety.violation_tolerance` (= limit **만** 어겼다) | `reason: initial_collision_violation` |
        | 투영이 존재한다 (box ∩ anchor 가 비지 않음) | `reason: no_projection` |
        | 투영이 limit 을 전부 지킨다 (`_limit_overshoot` — position·velocity·acceleration·anchor) | `reason: projection_outside_limits` |
        | 투영이 충돌 검사를 통과한다 (전 해상도 `full_violation` ≤ tolerance) | `reason: projection_collides` |

        충돌이 꺼진 판(`collision.enabled=False`)에서는 둘째·다섯째 줄을 보지 않는다 — merit 도
        충돌을 보지 않으므로 (T15), 여기서만 보면 "껐다" 가 거짓이 된다. 반환된 궤적의 위반은
        `_finish` 가 스위치와 무관하게 잰다.
        """
        cfg = self.config
        out: dict[str, Any] = {"tried": True, "used": False, "reason": None,
                               "max_change_rad": None, "violation_m": None,
                               "limit_overshoot": None}
        if not getattr(cfg.sqp, "limit_projection_fallback", True):
            out.update(tried=False, reason="disabled")
            return out
        collision = bool(cfg.collision.enabled)
        tolerance = cfg.safety.violation_tolerance
        if collision and (initial_violation is None or initial_violation > tolerance):
            out.update(tried=False, reason="initial_collision_violation")
            return out
        projected = project_to_limits(reference, self.limits,
                                      np.asarray(q_now, np.float64)[self.layout.q_indices],
                                      **({"envelope": envelope} if envelope is not None else {}))
        if projected is None:
            out["reason"] = "no_projection"
            return out
        out["max_change_rad"] = float(np.abs(projected - reference).max(initial=0.0))
        overshoot = self._limit_overshoot(projected, q_now, envelope)
        out["limit_overshoot"] = {k: float(v) for k, v in overshoot.items()}
        if max(overshoot.values()) > _LIMIT_TOLERANCE:
            out["reason"] = "projection_outside_limits"
            return out
        states = None
        if collision:
            # FK 는 한 번 — `_finish` 가 같은 상태로 최종 검증을 한다.
            states = self.linearizer.sphere_states(projected, q_now)
            violation = max(0.0, -self.linearizer.full_violation(projected, q_now, scene, states))
            if self.path is not None:
                violation = max(violation, self._path_violation(projected, scene))
            out["violation_m"] = float(violation)
            if violation > tolerance:
                out["reason"] = "projection_collides"
                return out
        out.update(used=True, reason="limit_only", _trajectory=projected, _states=states)
        return out

    def _passthrough(self, reference, q_now, template, status, notes, started) -> TrajOptResult:
        chunk = self._chunk_from(reference, template)
        return TrajOptResult(
            chunk=chunk,
            trajectory=reference,
            status=status,
            iterations=0,
            solve_time_ms=(time.perf_counter() - started) * 1000.0,
            max_violation=0.0,
            reference_violation=0.0,
            slack_norm=0.0,
            cost=0.0,
            reference_deviation=0.0,
            notes=list(notes),
            # 제약이 없었으므로 최악 행도 없다. **키는 둔다** — 없음이 `None` 으로 적히는 것과
            # 키가 빠지는 것은 읽는 쪽에 다른 뜻이다.
            metrics={"n_rows": 0, "max_violation_pair": None,
                     # 키 집합을 `_finish` 와 같게 둔다 — 읽는 쪽이 `.get` 두 갈래로 갈리지 않는다.
                     "sqp_iterations": 0, "max_iterations": int(self.config.sqp.max_iterations),
                     "time_budget_ms": float(self.config.sqp.time_budget_ms),
                     "time_budget_hit": False, "max_iterations_hit": False,
                     "collision_enabled": bool(self.config.collision.enabled),
                     # 후보 선택 키도 같게 둔다 (T24). 풀지 않았으므로 후보는 없고, 나간 것은
                     # reference 그 자체다.
                     **_no_selection()},
        )

    def _chunk_from(self, trajectory: np.ndarray, template: Optional[np.ndarray]) -> np.ndarray:
        if template is None:
            template = np.zeros((trajectory.shape[1], self.layout.action_dim))
        return self.layout.trajectory_to_chunk(trajectory, template)

    def _finish(
        self, trajectory, reference, q_now, scene, template, iterations, slack_norm,
        reference_violation, previous_chunk, notes, started, geometry_certified, states=None,
        timing=None, qp_iterations=0, time_budget_hit=False, selection=None,
    ) -> TrajOptResult:
        cfg = self.config
        # `full_violation` 이 아니라 `worst_row` 다. **같은 값을 내면서 그 값을 만든 행의 신원까지
        # 돌려준다** — clearance sweep 도 forward kinematics 도 늘지 않고 argmin 만 더 돈다.
        # `T5f` 가 *"violated 가 어느 제약인가"* 에서 막힌 것이 이 숫자에 이름이 없었기 때문이다.
        lookahead_record = None
        if self.lookahead_from is not None:
            # T43 TA (A): certify the executed window; the tail is recorded, never certified.
            k = self.lookahead_from
            clearance, worst_pair = self.linearizer.worst_row(trajectory, q_now, scene, states,
                                                              steps=k)
            per_step = self.linearizer.step_clearances(trajectory, q_now, scene, states)
            tail = per_step[k:]
            tail_min = float(tail.min(initial=np.inf))
            lookahead_record = {
                "from": int(k), "planned": int(trajectory.shape[1]),
                **({"gate": "open", **(getattr(self, "_gate_record", None) or {})}
                   if self.lookahead_gate is not None else {"gate": None}),
                "window_clearance_m": float(clearance),
                "tail_clearance_m": tail_min if np.isfinite(tail_min) else None,
                "tail_violation_m": max(0.0, -tail_min) if np.isfinite(tail_min) else 0.0,
                "tail_worst_step": (int(k + int(np.argmin(tail))) if tail.size
                                    and np.isfinite(tail_min) else None),
                **{key: v for key, v in (getattr(self.linearizer, "_last_lookahead", None)
                                         or {}).items() if key != "from"}}
        else:
            clearance, worst_pair = self.linearizer.worst_row(trajectory, q_now, scene, states)
        # T43 HM — the held rows' classes on what is returned (read before the path check
        # re-evaluates the field). `None` unless `held_obstacle_margin` is on and something is held.
        held_rows = (self.linearizer.held_rows_record()
                     if getattr(scene, "held_obstacle_margin", False)
                     and getattr(scene, "esdf", None) is not None
                     and hasattr(self.linearizer, "held_rows_record") else None)
        violation = max(0.0, -clearance)
        tolerance = cfg.safety.violation_tolerance
        path_record = None
        if self.path is not None:
            # T43 Q — the rows between waypoints, at full resolution, on what is returned. A
            # violation there makes the chunk VIOLATED like any other row; `clearance_m` and
            # `max_violation_pair` stay the waypoint numbers, the path's are in `path_check`.
            path_worst, path_record = self.path.violation(trajectory, scene, record=True)
            path_record["timing_ms"] = {k: round(v, 3) for k, v in self.path.timing.items()}
            path_violation = max(0.0, -path_worst)
            if path_violation > tolerance:
                notes.append(
                    f"path check ({self.path.mode}): {path_violation * 1000:.1f} mm below the floor "
                    f"at step {path_record['worst']['step']} "
                    f"({path_record['worst']['at_kind']} {path_record['worst']['at']:g}, "
                    f"{path_record['worst']['link']}) — between the waypoints")
            violation = max(violation, path_violation)

        if violation > tolerance:
            status = TrajOptStatus.VIOLATED
            notes.append(
                f"best effort: {violation * 1000:.1f} mm of penetration remains "
                f"(was {reference_violation * 1000:.1f} mm). The caller decides whether to execute."
            )
        elif iterations >= cfg.sqp.max_iterations or any("budget" in n for n in notes):
            status = TrajOptStatus.FEASIBLE
        else:
            status = TrajOptStatus.OPTIMAL

        if status.safe and cfg.safety.require_certified_geometry and not geometry_certified:
            # The trajectory clears everything AG3S described. AG3S said it could not account for
            # everything it saw, so "clear" is a statement about the model, not about the world.
            status = TrajOptStatus.VIOLATED
            notes.append(
                "the trajectory clears every constraint, but AG3S could not certify the geometry "
                "behind them; safety cannot be claimed for a scene that was not fully accounted for"
            )

        if not cfg.collision.enabled:
            # **이 기록이 "위반 0" 으로 읽히면 안 된다.** 위 `violation` 은 여전히 참값이지만
            # (측정은 이 스위치와 무관하다), 최적화기는 그것을 보지 않았다. 그 구별이 기록에
            # 없으면 이 실행의 궤적이 "충돌을 피한 궤적" 으로 읽힌다.
            notes.append(
                "COLLISION CONSTRAINTS WERE OFF (collision.enabled=False): the optimizer saw no "
                "collision rows and no violation penalty. max_violation below is still measured "
                "from the real geometry — it is what the trajectory does, not what was enforced."
            )
        limits = limit_report(trajectory, self.limits)
        if max(limits.values()) > _LIMIT_TOLERANCE:
            notes.append(
                f"limit overshoot (m or m/step): {  {k: round(v, 6) for k, v in limits.items()} }"
            )

        self._previous = trajectory
        self.frame_index += 1
        return TrajOptResult(
            chunk=self._chunk_from(trajectory, template),
            trajectory=trajectory,
            status=status,
            iterations=iterations,
            solve_time_ms=(time.perf_counter() - started) * 1000.0,
            max_violation=violation,
            reference_violation=max(0.0, reference_violation),
            slack_norm=float(slack_norm),
            cost=self._objective(trajectory, reference, previous_chunk),
            reference_deviation=float(np.linalg.norm(trajectory - reference)),
            notes=notes,
            metrics={
                "clearance_m": float(clearance),
                # **최악 행의 신원.** 활성 제약이 하나도 없으면 `None` 이다 — 없는 신원을
                # 지어내지 않는다. 키는 항상 있어서 소비 쪽이 `.get` 두 갈래로 갈리지 않는다.
                "max_violation_pair": worst_pair,
                "limit_overshoot": limits,
                "frame_index": self.frame_index,
                "geometry_certified": bool(geometry_certified),
                "qp_solver": cfg.qp.solver,
                "qp_iterations": int(qp_iterations),
                # **구조화된 값으로 싣는다** (T15). 예전에는 `notes` 의 문장
                # (`"time budget 50 ms reached after 1 iteration(s)"`) 뿐이라 **셀 수 없었다** —
                # "몇 번 돌았나" 가 다음 질문인데 그때마다 실행을 다시 해야 했다.
                "sqp_iterations": int(iterations),
                "max_iterations": int(cfg.sqp.max_iterations),
                "time_budget_ms": float(cfg.sqp.time_budget_ms),
                "time_budget_hit": bool(time_budget_hit),
                "max_iterations_hit": bool(iterations >= cfg.sqp.max_iterations),
                # 충돌이 꺼졌는지. `notes` 와 **둘 다** 싣는다 — 기계가 세는 쪽과 사람이 읽는 쪽.
                "collision_enabled": bool(cfg.collision.enabled),
                "timing_ms": {k: round(v, 3) for k, v in (timing or {}).items()},
                # **무엇이 반환됐나** (T24, 지침 §7.3). `returned` 는 `"initial"` · `"accepted"` ·
                # `"best_unaccepted"` 중 하나이고, 그 판정의 근거인 후보 목록이 옆에 실린다.
                # 위의 `clearance_m`·`limit_overshoot`·상태는 **반환된 그 궤적**을 최종 기하와
                # 실제 한계로 다시 잰 값이다 — 후보의 `violation_m` 은 루프 안의 merit 가 본 값이다.
                **(selection if selection is not None else _no_selection()),
                **({"path_check": path_record} if path_record is not None else {}),
                **({"held_rows": held_rows} if held_rows is not None else {}),
                **({"lookahead": lookahead_record} if lookahead_record is not None else {}),
                **self._ta_record(trajectory, reference),
            },
        )

    def _ta_record(self, trajectory, reference) -> dict[str, Any]:
        """T43 TA (B, C) — what the returned trajectory does to the new terms. `{}` when off."""
        out: dict[str, Any] = {}
        w = float(getattr(self.config.cost, "w_deflection_rate", 0.0) or 0.0)
        if w > 0.0:
            from benchmark.trajopt.problem import deflection_rate_cost

            D = np.asarray(trajectory) - np.asarray(reference)
            out["deflection_rate"] = {
                "weight": w, "boundary": self._history.get("deflection") is not None,
                "cost": deflection_rate_cost(trajectory, reference, w,
                                             self._history.get("deflection")),
                "max_step_change_rad": (float(np.abs(np.diff(D, axis=1)).max())
                                        if D.shape[1] > 1 else 0.0)}
        if self.servo_accel_bound is not None:
            from benchmark.trajopt.problem import servo_accel_overshoot

            over = servo_accel_overshoot(trajectory, reference, self.servo_accel_bound,
                                         str(self.config.limits.servo_accel),
                                         self._max_step_change(), self._history.get("command"))
            from benchmark.trajopt.problem import boundary_velocity_overshoot

            hist = self._history.get("command")
            step = np.asarray(self.limits.max_step, np.float64)
            bvel = (boundary_velocity_overshoot(trajectory, hist, step)
                    if velocity_rows(self.limits) else np.zeros(0))
            out["servo_accel"] = {**(self.servo_accel_record or {}),
                                  "boundary": hist is not None,
                                  "overshoot_max_rad": float(over.max(initial=0.0)),
                                  "n_over": int((over > 1e-9).sum()),
                                  "boundary_velocity_ratio": (
                                      None if hist is None else float(np.max(
                                          np.abs(np.asarray(trajectory)[:, 0] - hist[:, -1])
                                          / step))),
                                  "boundary_velocity_overshoot_rad": float(
                                      bvel.max(initial=0.0))}
        return out


#: `best_index` 의 표시 — reference 의 최소 투영이 반환됐다 (T31 G2-ii). `-1` 은 초기 iterate.
_PROJECTION_INDEX = -2


def _returned_label(index: int, candidates: list[dict[str, Any]]) -> str:
    """`best_index` → `"initial"` · `"accepted"` · `"best_unaccepted"` · `"projection"` (T31).

    수락 여부를 **후보 기록에서 읽는다.** 적격한 초기 iterate 로 시작하면 `best_unaccepted` 는
    나오지 않는다 (`solve` 의 주석) — 나오는 것은 초기 iterate 가 한계 밖이라 비교에서 빠졌을
    때뿐이다. 그 성질을 여기서 가정하면 그것이 깨진 날 기록이 거짓말을 한다.
    """
    if index == _PROJECTION_INDEX:
        return "projection"
    if index < 0:
        return "initial"
    return "accepted" if candidates[index]["accepted"] else "best_unaccepted"


def _no_selection() -> dict[str, Any]:
    """풀지 않은 결과의 후보 선택 키 — `_finish` 와 키 집합을 같게 둔다."""
    return {"returned": "initial", "returned_index": -1, "initial_merit": None,
            "initial_violation_m": None, "initial_eligible": None,
            "initial_limit_overshoot": None, "candidates": [], "n_candidates": 0,
            "n_accepted": 0, "qp_failures": 0, "min_iterations": None, "projection": None}


__all__ = ["TrajectoryOptimizer"]
