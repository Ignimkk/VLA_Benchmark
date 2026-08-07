"""Collision-avoidance chunk refiner (SKELETON — no working algorithm yet).

This is the concrete implementation target for the hook reserved in ``refinement/base.py``
("Future implementations may apply trajectory optimization, MPC, ... collision or CBF safety
filtering, etc."). Planned approach: treat the decoded absolute physical-space chunk as a candidate
trajectory and project it onto an approximate safe set defined by a signed-distance constraint

    h(a) = dist(robot(a), obstacle) - safety_margin >= 0

evaluated per predicted step via ``distance_fn`` (forward kinematics + scene geometry, or a MuJoCo
geom-distance query — supplied by the caller so this module stays simulator-agnostic between LIBERO
and RBY1). This week only the interface and a diagnostic-only helper (:meth:`would_violate`) are
implemented; :meth:`refine` raises ``NotImplementedError``. See
``docs/research_plan_week_ko.md`` §3.2 for the explicit scope decision.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from benchmark.seam_vla.refinement.base import DecodedChunkRefiner


class CollisionAvoidanceRefiner(DecodedChunkRefiner):
    """Physical-space refiner that will project a chunk away from predicted collisions.

    Args:
        distance_fn: ``q[H, D] -> distances[H, n_obstacles]``, signed distance from the robot's
            swept volume at each predicted step to each known obstacle (negative = penetrating).
            Supplied by the caller so this class does not depend on LIBERO/robosuite or MuJoCo
            directly; a LIBERO-side and an RBY1-side ``distance_fn`` can share this refiner.
        safety_margin: minimum allowed distance; steps with ``distance < safety_margin`` are
            considered unsafe.
        guided_dims: action-dim indices the refiner is allowed to modify (e.g. arm joints,
            excluding grippers). ``None`` means all dims.
        max_correction_norm: optional cap on the per-step correction magnitude, to avoid the
            refiner introducing a large jump (mirrors the jerk concern that motivated SEAM).
        mode: ``"project"`` (planned: solve a small QP / gradient step to push unsafe steps back
            onto the safe set) or ``"reject"`` (planned: leave the chunk unchanged but flag it,
            deferring to a higher-level fallback such as re-planning or a stop).
    """

    def __init__(
        self,
        distance_fn: Callable[[np.ndarray], np.ndarray],
        safety_margin: float,
        guided_dims: Sequence[int] | None = None,
        max_correction_norm: float | None = None,
        mode: str = "project",
    ):
        if mode not in ("project", "reject"):
            raise ValueError(f"mode must be 'project' or 'reject', got {mode!r}")
        self.distance_fn = distance_fn
        self.safety_margin = float(safety_margin)
        self.guided_dims = None if guided_dims is None else np.asarray(list(guided_dims), dtype=int)
        self.max_correction_norm = max_correction_norm
        self.mode = mode

    def refine(self, physical_chunk, context: dict[str, Any] | None = None):
        """Not implemented this week — see module docstring and research_plan_week_ko.md §3.2.

        Planned contract (once implemented): return a chunk of the same shape
        ``[H, action_dim_env]`` where steps flagged unsafe by :meth:`would_violate` have been
        nudged toward the safe set, respecting ``guided_dims`` and ``max_correction_norm``.
        """
        raise NotImplementedError(
            "CollisionAvoidanceRefiner.refine is a design-stage skeleton (this week's scope is "
            "interface design only, not a working algorithm). Use would_violate() for offline "
            "diagnostic flagging in the meantime."
        )

    def would_violate(self, physical_chunk, context: dict[str, Any] | None = None) -> np.ndarray:
        """Diagnostic-only helper: flag predicted steps that violate the safety margin.

        Usable today (does not require ``refine`` to be implemented), so the section-4 analysis
        scripts can reuse it for offline collision-risk flagging on recorded/rolled-out chunks.
        Requires ``distance_fn`` to be set; ``context`` is accepted for interface parity with
        :meth:`refine` (e.g. future callers may pass ``object_positions``/``contact_state`` that a
        concrete ``distance_fn`` closes over) but is not read directly by this base implementation.

        Returns:
            bool array of shape ``[H]``, True where any obstacle distance at that step is below
            ``safety_margin``.
        """
        chunk = np.asarray(physical_chunk)
        distances = np.asarray(self.distance_fn(chunk))  # [H, n_obstacles]
        if distances.ndim == 1:
            distances = distances[:, None]
        return np.any(distances < self.safety_margin, axis=-1)
