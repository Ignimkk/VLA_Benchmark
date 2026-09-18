"""The SEAM hook: a `DecodedChunkRefiner` that makes a decoded chunk collision-free.

`benchmark/seam_vla/refinement/base.py` reserved this slot — "future implementations may apply
trajectory optimization, MPC, ... collision or CBF safety filtering" — and left `refine()` abstract.
Nothing in `seam_vla` is modified to fill it; the class is imported and subclassed.

Two things about where this sits are easy to get wrong.

**Only the leading steps are planned.** The executor runs `execution_length` actions and then asks
for a new chunk, so the tail is discarded before it could execute. Planning all fifty steps costs
93 ms against 22 ms for thirty-two (measured) and buys motion that is thrown away. What the lookahead
past the executed window *does* buy is foresight: an optimizer that can only see eight steps ahead
will happily steer into a corner it cannot get out of. The unplanned tail is the policy's own answer,
returned untouched.

**SEAM never sees the correction.** `seam_policy.py` stores `model_chunk_np` — the *pre*-refinement
chunk — as the next chunk's prior, and keeps the refined physical chunk only for logging. So the
policy will propose the same colliding trajectory again next chunk, and this layer will push it away
again. Left alone that reintroduces exactly the boundary discontinuity SEAM exists to remove, one
level down. The mitigation is the continuity term in `problem.build_problem`, fed from
`context["previous_physical_chunk"]`, which `refine()` already receives. It is a mitigation and not a
fix: closing the loop properly needs a physical-to-model inverse and a change to SEAM.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Callable, Optional, Sequence

import numpy as np

from benchmark.seam_vla.refinement.base import DecodedChunkRefiner
from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.limits import build_limits
from benchmark.trajopt.linearize import SceneSnapshot
from benchmark.trajopt.sqp import TrajectoryOptimizer
from benchmark.trajopt.types import ChunkLayout, TrajOptResult, TrajOptStatus


class TrajOptChunkRefiner(DecodedChunkRefiner):
    """Refines a decoded physical-space chunk against AG3S's collision scene.

    Args:
        robot_model: the **same** `RobotCollisionModel` AG3S built its constraints against. Two
            different models would mean the optimizer avoiding geometry in one place while the
            constraints described it in another, and nothing would report the disagreement.
        layout: which chunk columns are joints. `ChunkLayout.rby1(...)` for RB-Y1.
        scene_fn: ``context -> (SceneSnapshot | None, q_now, geometry_certified)``. Supplied by the
            caller so this class never imports a simulator or a camera driver, and so a recorded
            scene and a live one use the same refiner.
        on_result: optional callback receiving every `TrajOptResult`. This is where a caller hooks
            its own policy for `VIOLATED` frames — stop, hold, re-plan. **Nothing here decides that**,
            for the same reason AG3S does not decide what to do about uncertified geometry.
    """

    def __init__(
        self,
        robot_model,
        layout: ChunkLayout,
        scene_fn: Callable[[Optional[dict]], tuple[Optional[SceneSnapshot], np.ndarray, bool]],
        config: Optional[TrajOptConfig] = None,
        *,
        on_result: Optional[Callable[[TrajOptResult], None]] = None,
    ):
        self.config = config or TrajOptConfig()
        self.layout = layout
        self.scene_fn = scene_fn
        self.on_result = on_result
        self.limits = build_limits(
            robot_model, layout, dt=self.config.horizon.dt, config=self.config.limits
        )
        planning = self.config.with_overrides(
            {"horizon": {"horizon": self.config.horizon.planned,
                         "execution_length": min(self.config.horizon.execution_length,
                                                 self.config.horizon.planned),
                         "plan_horizon": None}}
        )
        self.optimizer = TrajectoryOptimizer(robot_model, layout, self.limits, planning)
        self.last_result: Optional[TrajOptResult] = None
        #: 마지막으로 `scene_fn` 이 던진 예외의 요약, 또는 `None`.
        self.last_failure: Optional[str] = None
        #: 연속 실패 수. 한 프레임 드롭과 **영구 고장**을 가르는 것이 이 숫자다.
        self._failure_count = 0
        self._previous: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Between episodes. Drops the warm start and the continuity reference."""
        self.optimizer.reset()
        self.last_result = None
        self.last_failure = None
        self._failure_count = 0
        self._previous = None

    # ------------------------------------------------------------------------------------
    def refine(self, physical_chunk, context: dict[str, Any] | None = None):
        """Return a chunk of the same shape with its executed portion made collision-free.

        Falls back to the input chunk — unchanged, and with a note — whenever the scene cannot be
        obtained. A refiner that raised here would take down the policy over a dropped camera frame,
        which is a worse outcome than executing the policy's own answer and saying so.
        """
        chunk = np.asarray(physical_chunk, np.float64)
        planned = min(self.config.horizon.planned, chunk.shape[0])

        try:
            scene, q_now, certified = self.scene_fn(context)
        except Exception as exc:  # noqa: BLE001 - a perception fault must not kill the policy
            # **삼키되 크게 말한다.** 예외로 정책을 죽이지 않는 것은 맞다 — 카메라 한 프레임
            # 드롭으로 로봇이 서면 안 된다. 그러나 조용히 삼키면 **지각과 최적화가 통째로
            # 멈춘 채 응답만 계속 나간다.** 실측(2026-09-18): AG3S 가 파지 다음 프레임부터
            # 14 프레임 동안 한 번도 안 돌았는데 아무 데도 안 찍혔다. 원인은
            # `attached_parent_links` 미예약이었고, 그 한 줄이 로그에 있었으면 즉시 보였다.
            self.last_result = None
            self.last_failure = f"{type(exc).__name__}: {exc}"
            self._failure_count += 1
            logging.getLogger(__name__).exception(
                "scene_fn raised; the chunk passes through UNVERIFIED (consecutive failures: %d)",
                self._failure_count)
            self._note_failure(
                f"scene unavailable ({self.last_failure}); chunk passed through unchanged; "
                f"consecutive failures: {self._failure_count}\n"
                + traceback.format_exc(limit=6))
            return chunk

        reference = self.layout.chunk_to_trajectory(chunk)[:, :planned]
        previous = self._continuity_reference(context, planned)

        # 여기까지 왔으면 씬을 얻은 것이다 — 연속 실패 수를 지운다.
        self.last_failure = None
        self._failure_count = 0

        result = self.optimizer.solve(
            reference,
            q_now,
            scene,
            previous_chunk=previous,
            template=chunk,
            geometry_certified=certified,
        )
        # The join between the planned window and the untouched tail. It never executes, but it does
        # feed the next chunk's continuity term, so a large step here is worth seeing.
        if planned < chunk.shape[0]:
            join = float(
                np.abs(result.trajectory[:, -1] - reference[:, -1]).max()
            )
            result.metrics["plan_join_discontinuity_rad"] = join
            result.metrics["planned_steps"] = planned

        self.last_result = result
        self._previous = result.trajectory
        if self.on_result is not None:
            self.on_result(result)
        return result.chunk

    def would_violate(self, physical_chunk, context: dict[str, Any] | None = None) -> np.ndarray:
        """Per-step flag: does the *input* chunk violate anything? Diagnostics only.

        Mirrors the helper on `seam_vla`'s own skeleton refiner, so the offline analysis scripts that
        already call `would_violate` keep working against this implementation.
        """
        chunk = np.asarray(physical_chunk, np.float64)
        planned = min(self.config.horizon.planned, chunk.shape[0])
        scene, q_now, _ = self.scene_fn(context)
        flags = np.zeros(chunk.shape[0], bool)
        if scene is None or scene.is_empty:
            return flags
        trajectory = self.layout.chunk_to_trajectory(chunk)[:, :planned]
        candidate, plane, _ = self.optimizer.linearizer.clearances(trajectory, q_now, scene)
        worst = np.minimum(
            candidate.reshape(planned, -1).min(axis=1) if candidate.size else np.inf,
            plane.reshape(planned, -1).min(axis=1) if plane.size else np.inf,
        )
        flags[:planned] = worst < -self.config.safety.violation_tolerance
        return flags

    # --- helpers --------------------------------------------------------------------------
    def _continuity_reference(self, context: Optional[dict], planned: int) -> Optional[np.ndarray]:
        """The previously refined chunk's overlapping tail, aligned to this chunk's first step.

        `context["previous_physical_chunk"]` is what SEAM already passes. Its first
        `execution_length` steps have executed by now, so the part that overlaps this chunk starts at
        index K — the same alignment SEAM's own prior uses.
        """
        if self.config.cost.w_continuity <= 0.0:
            return None
        previous = None if context is None else context.get("previous_physical_chunk")
        if previous is None:
            return None
        previous = np.asarray(previous, np.float64)
        if previous.ndim != 2 or previous.shape[1] < self.layout.n_valid:
            return None
        tail = self.layout.chunk_to_trajectory(previous)[:, self.config.horizon.execution_length :]
        return tail[:, :planned] if tail.shape[1] else None

    def _note_failure(self, message: str) -> None:
        if self.on_result is not None:
            self.on_result(
                TrajOptResult(
                    chunk=np.zeros((0, 0)),
                    trajectory=np.zeros((self.layout.nq_opt, 0)),
                    status=TrajOptStatus.SOLVER_FAILED,
                    iterations=0,
                    solve_time_ms=0.0,
                    max_violation=0.0,
                    reference_violation=0.0,
                    slack_norm=0.0,
                    cost=0.0,
                    reference_deviation=0.0,
                    notes=[message],
                )
            )


__all__ = ["TrajOptChunkRefiner"]
