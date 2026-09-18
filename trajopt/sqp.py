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
from benchmark.trajopt.limits import limit_report
from benchmark.trajopt.problem import build_problem, objective
from benchmark.trajopt.qp import QpSolver
from benchmark.trajopt.types import ChunkLayout, JointLimits, TrajOptResult, TrajOptStatus


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

    def reset(self) -> None:
        """Clear cross-chunk state. Between episodes, never mid-episode."""
        self.solver.reset()
        self._previous = None
        self.frame_index = 0

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
        """
        started = time.perf_counter()
        cfg = self.config
        reference = np.asarray(reference, np.float64)
        q_now = np.asarray(q_now, np.float64).reshape(-1)
        notes: list[str] = []

        if scene is None or scene.is_empty:
            return self._passthrough(
                reference, q_now, template, TrajOptStatus.UNCONSTRAINED,
                notes + ["AG3S emitted no geometry to avoid; the reference passes through unchanged"],
                started,
            )

        n_slack = self.block.n_rows
        iterate = reference.copy()
        if cfg.sqp.warm_start and self._previous is not None:
            # Seed from the previous chunk shifted by the steps that have executed since. The tail is
            # held rather than extrapolated: extrapolating a trajectory past its horizon invents
            # motion the policy never proposed.
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
        reference_violation = -self.linearizer.full_violation(reference, q_now, scene)
        # Forward kinematics for a whole chunk is the most expensive thing left in the loop, so each
        # trajectory's sphere states are computed once and carried: the iterate's states feed both
        # the linearization and its merit, and an accepted candidate hands its states to the next
        # iteration rather than being recomputed there.
        iterate_states = self.linearizer.sphere_states(iterate, q_now)
        iterate_merit = self._merit(
            iterate, reference, previous_chunk, q_now, scene, states=iterate_states
        )
        best = iterate
        best_states = iterate_states
        best_merit = np.inf
        # Per-stage timing, accumulated rather than sampled. Where the milliseconds go is the single
        # most useful thing to know about a real-time optimizer, and it changes with the scene.
        timing = {"linearize": 0.0, "assemble": 0.0, "qp": 0.0, "check": 0.0}
        best_slack = 0.0
        iterations = 0
        qp_iterations = 0
        radius = cfg.sqp.trust_radius
        budget_bound = False

        while iterations < cfg.sqp.max_iterations:
            # Checked *after* the first iteration, never before it. A budget so tight that the
            # set-up alone exhausts it would otherwise return the reference untouched while
            # reporting success — the one outcome a safety layer must not produce. One step is the
            # real-time iteration scheme's minimum unit of work, and it is always taken.
            if iterations >= 1 and (time.perf_counter() - started) * 1000.0 >= cfg.sqp.time_budget_ms:
                notes.append(
                    f"time budget {cfg.sqp.time_budget_ms:.0f} ms reached after {iterations} "
                    "iteration(s); returning the best iterate so far"
                )
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
            )
            problem = append_collision_rows(
                problem, rows, iterate, self.block, cfg.reduction.linearization_backoff
            )
            timing["assemble"] += (time.perf_counter() - mark) * 1000.0

            mark = time.perf_counter()
            solution = self.solver.solve(problem, warm_start=cfg.sqp.warm_start)
            timing["qp"] += (time.perf_counter() - mark) * 1000.0
            qp_iterations += solution.iterations
            if not solution.success:
                notes.append(f"QP failed at iteration {iterations}: {solution.status}")
                radius *= cfg.sqp.trust_shrink
                if radius < cfg.sqp.trust_radius_min:
                    break
                continue

            mark = time.perf_counter()
            candidate, slack = problem.split(solution.x)
            candidate_states = self.linearizer.sphere_states(candidate, q_now)
            merit = self._merit(
                candidate, reference, previous_chunk, q_now, scene, states=candidate_states
            )
            timing["check"] += (time.perf_counter() - mark) * 1000.0
            step_size = float(np.abs(candidate - iterate).max())

            if merit < best_merit:
                best, best_merit = candidate, merit
                best_slack = float(np.sum(np.abs(slack)))
                best_states = candidate_states
            # A step that improved the merit earns a wider trust region; one that did not gets a
            # narrower one and another attempt from the same iterate. `iterate_merit` is carried
            # rather than recomputed: it costs a full-resolution clearance sweep, which after the QP
            # became fast is the most expensive thing left in the loop.
            if merit <= iterate_merit:
                iterate, iterate_merit, iterate_states = candidate, merit, candidate_states
                radius = min(radius * cfg.sqp.trust_expand, cfg.sqp.trust_radius_max)
            else:
                radius = max(radius * cfg.sqp.trust_shrink, cfg.sqp.trust_radius_min)

            if step_size < cfg.sqp.step_tolerance:
                break

        if budget_bound:
            notes.append(
                f"reduction.rows_per_step={cfg.reduction.rows_per_step} bound on at least one step; "
                "some near-active constraints were not shown to the QP (the final check still sees "
                "all of them)"
            )
        return self._finish(
            best, reference, q_now, scene, template, iterations, best_slack,
            reference_violation, previous_chunk, notes, started, geometry_certified,
            states=best_states, timing=timing, qp_iterations=qp_iterations,
        )

    # --- helpers --------------------------------------------------------------------------
    def _merit(self, trajectory, reference, previous_chunk, q_now, scene, states=None) -> float:
        """Cost plus the true constraint violation, both at full resolution.

        Deliberately *not* computed from the QP's own linearized rows. A merit function built by the
        same code that builds the subproblem agrees with it even when both are wrong; this one
        re-evaluates the geometry and so can disagree, which is the only way the step-acceptance test
        can catch a bad linearization.
        """
        violation = max(0.0, -self.linearizer.full_violation(trajectory, q_now, scene, states))
        cost = objective(trajectory, reference, self.config, previous_chunk=previous_chunk)
        return cost + self.config.cost.w_slack * violation

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
            metrics={"n_rows": 0},
        )

    def _chunk_from(self, trajectory: np.ndarray, template: Optional[np.ndarray]) -> np.ndarray:
        if template is None:
            template = np.zeros((trajectory.shape[1], self.layout.action_dim))
        return self.layout.trajectory_to_chunk(trajectory, template)

    def _finish(
        self, trajectory, reference, q_now, scene, template, iterations, slack_norm,
        reference_violation, previous_chunk, notes, started, geometry_certified, states=None,
        timing=None, qp_iterations=0,
    ) -> TrajOptResult:
        cfg = self.config
        clearance = self.linearizer.full_violation(trajectory, q_now, scene, states)
        violation = max(0.0, -clearance)
        tolerance = cfg.safety.violation_tolerance

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

        limits = limit_report(trajectory, self.limits)
        if max(limits.values()) > 1e-6:
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
            cost=objective(trajectory, reference, cfg, previous_chunk=previous_chunk),
            reference_deviation=float(np.linalg.norm(trajectory - reference)),
            notes=notes,
            metrics={
                "clearance_m": float(clearance),
                "limit_overshoot": limits,
                "frame_index": self.frame_index,
                "geometry_certified": bool(geometry_certified),
                "qp_solver": cfg.qp.solver,
                "qp_iterations": int(qp_iterations),
                "timing_ms": {k: round(v, 3) for k, v in (timing or {}).items()},
            },
        )


__all__ = ["TrajectoryOptimizer"]
