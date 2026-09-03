"""Stage 8 — the handover to trajectory optimization.

TO is not implemented in this workspace. `benchmark/seam_vla/refinement/collision_avoidance.py` is
where it will live — its `refine()` raises `NotImplementedError` today — and this module is the other
half of that interface, defined now so that when TO arrives there is nothing left to negotiate:

    VLA -> Action Chunk -> SEAM -> Reference Trajectory --.
                                                          +-> [TO: not implemented] -> Safe Chunk
    AG3S -> CollisionConstraintSet ----------------------'

What a TO gets from `to_casadi` is a plain NLP fragment — a constraint vector `g`, its bounds, the
parameter symbol and this frame's parameter values — with no AG3S types crossing the boundary and no
callback into AG3S at solve time. That last point is the one that matters: the expression is CasADi
arithmetic all the way down, so IPOPT differentiates it symbolically instead of finite-differencing
an opaque distance query.

The real-time contract is `refresh`. Structure is built once; each frame writes new numbers into the
same parameter vector, so the solver object, its sparsity pattern and its warm start all survive.
`warm_start_map` reports which slot holds which candidate id, so a TO carrying duals across frames
can tell when a slot changed owner.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from benchmark.ag3s.constraint_builder import ConstraintBuilder
from benchmark.ag3s.types import (
    AttachedCollisionGeometry,
    CollisionCandidate,
    CollisionConstraintSet,
    ConstraintSpec,
    ConstraintValidity,
    ContactPolicyContext,
    GroundingStatus,
    Phase,
    PipelineStatus,
    SupportSurface,
    TargetGeometry,
)


def to_casadi(spec: ConstraintSpec, Q: Any) -> dict[str, Any]:
    """Materialize the constraint fragment against a TO's own decision variables.

    Args:
        spec: this frame's specification.
        Q: the optimizer's state trajectory, `(nq, horizon)` CasADi `SX`/`MX`. AG3S never creates the
            decision variables — the TO owns those, and owns the dynamics and cost that go with them.

    Returns a dict with `g`, `lbg`, `ubg`, `p` (the parameter symbol) and `p_val` (this frame's
    numbers), ready to drop into `casadi.nlpsol`.
    """
    g = spec.expr_factory(Q, spec.parameter_symbol)
    return {
        "g": g,
        "lbg": spec.lower_bound,
        "ubg": spec.upper_bound,
        "p": spec.parameter_symbol,
        "p_val": spec.parameter_values,
    }


def warm_start_map(previous: Optional[ConstraintSpec], current: ConstraintSpec) -> np.ndarray:
    """`True` where a slot still holds the same candidate as last frame.

    A TO reusing duals or an initial guess should keep them only for slots this marks `True`. Slot
    ownership changes when an object appears, disappears, or overtakes another in radius order —
    rare, but silently reusing a dual across such a change pushes the robot toward the obstacle it
    just avoided.
    """
    if previous is None:
        return np.zeros(current.candidate_ids.shape, bool)
    n = min(previous.candidate_ids.size, current.candidate_ids.size)
    same = np.zeros(current.candidate_ids.shape, bool)
    same[:n] = (previous.candidate_ids[:n] == current.candidate_ids[:n]) & (
        current.candidate_ids[:n] >= 0
    )
    return same


def build_constraint_set(
    builder: ConstraintBuilder,
    *,
    candidates: Sequence[CollisionCandidate],
    support_surfaces: Sequence[SupportSurface],
    target: Optional[TargetGeometry],
    robot_state: np.ndarray,
    phase: Phase | str,
    grounding_status: GroundingStatus = GroundingStatus.OK,
    frame_id: str = "base",
    frame_index: int = 0,
    timestamp: Optional[float] = None,
    profile: Optional[dict[str, float]] = None,
    notes: Optional[list[str]] = None,
    contact_context: Optional[ContactPolicyContext] = None,
    validity: ConstraintValidity = ConstraintValidity.VALID,
    metrics: Optional[dict[str, Any]] = None,
    attached: Optional[AttachedCollisionGeometry] = None,
    esdf: Any = None,
    build_spec: bool = True,
) -> CollisionConstraintSet:
    """Assemble the final AG3S output.

    `esdf` carries the distance field when one was built. It is not a second copy of `candidates`:
    with `collision_backend: esdf` the candidate list is deliberately empty and the field *is* the
    obstacle set, so the status logic below has to consult it before reporting `NO_GEOMETRY`.

    `status` summarises the frame for a caller that will not read the rest: `NO_GEOMETRY` when there
    is nothing to constrain (treat as sensor failure, not as clear space), `NO_TARGET` when grounding
    declined to name one and every candidate is therefore held at full clearance, `DEGRADED` when a
    configured cap actually bound, `GEOMETRY_INCOMPLETE` when some physical geometry could not be
    represented at all, and `OK` otherwise.

    `validity` arrives from upstream stages and is combined with what happens here, never
    downgraded: a frame that reached this function already incomplete stays incomplete. The rule that
    matters is at the bottom — an incomplete frame is **never** reported as `OK`, because "I could
    not account for everything I saw" and "there is nothing there" must not look the same to a TO.
    """
    notes = list(notes or [])
    ctx = contact_context or ContactPolicyContext.make(phase)
    # An ESDF-only frame has no candidates by construction, and an empty candidate list must not be
    # read as "the cameras saw nothing". The two are opposite claims: one says the geometry lives in
    # the field, the other says there is no geometry. `has_field` keeps them apart below.
    has_field = esdf is not None and int(getattr(esdf, "stats", {}).get("n_occupied", 0)) > 0
    # `build_spec=False` skips the CasADi parameter vector entirely. There is nothing to pack when
    # the field is the constraint and no plane survived: every slot would be inactive and the
    # optimizer reads the field directly. Measured at 94 ms per frame on RB-Y1, which is three
    # times what building the field itself costs.
    spec = builder.build(
        candidates, support_surfaces, context=ctx, target_grounded=target is not None,
        attached=attached,
    ) if build_spec else None

    attached_dropped = int(spec.layout.get("n_attached_dropped", 0)) if spec is not None else 0
    if attached_dropped:
        # A held object whose shape does not fit its reserved slots is geometry the robot is carrying
        # and cannot see coming. There is no conservative aggregate to fall back to here — an
        # aggregate riding on the gripper would engulf the gripper — so it is a genuine hole.
        notes.append(
            f"{attached_dropped} attached-object sphere(s) exceeded max_attached_primitives="
            f"{spec.layout.get('max_attached_primitives') if spec is not None else '?'} "
            "and are NOT constrained"
        )
        validity = ConstraintValidity.worst(validity, ConstraintValidity.INCOMPLETE)

    overflow = int(spec.layout.get("n_overflow_spheres", 0)) if spec is not None else 0
    if overflow:
        notes.append(
            f"{overflow} constraint sphere(s) exceeded max_candidates={builder.max_candidates} "
            f"and were folded into {builder.constraint_config.reserved_overflow_slots} conservative "
            "aggregate slot(s); every one is still represented"
        )
    extra_planes = max(0, len(support_surfaces) - builder.max_support_surfaces)
    if extra_planes:
        # A plane that does not fit is a plane the robot can drive through. There is no conservative
        # aggregate for a half-space, so this is a real gap in the geometry rather than a coarsening.
        notes.append(
            f"{extra_planes} support surface(s) exceeded max_support_surfaces="
            f"{builder.max_support_surfaces} and are NOT constrained"
        )

    validity = ConstraintValidity.worst(
        validity,
        ConstraintValidity.INCOMPLETE if extra_planes else ConstraintValidity.VALID,
        ConstraintValidity.DEGRADED if overflow else ConstraintValidity.VALID,
    )

    if validity is ConstraintValidity.INCOMPLETE:
        status = PipelineStatus.GEOMETRY_INCOMPLETE
    elif not candidates and not support_surfaces and not has_field:
        status = PipelineStatus.NO_GEOMETRY
    elif validity is ConstraintValidity.DEGRADED:
        status = PipelineStatus.DEGRADED
    elif target is None:
        status = PipelineStatus.NO_TARGET
    else:
        status = PipelineStatus.OK

    return CollisionConstraintSet(
        constraints=spec,
        candidates=list(candidates),
        target=target,
        support_surfaces=list(support_surfaces),
        robot_state=np.asarray(robot_state, np.float64).reshape(-1),
        phase=Phase.parse(phase),
        timestamp=time.time() if timestamp is None else float(timestamp),
        frame_id=frame_id,
        status=status,
        grounding_status=grounding_status,
        frame_index=frame_index,
        profile=dict(profile or {}),
        notes=notes,
        validity=validity,
        contact_context=ctx,
        metrics=dict(metrics or {}),
        attached=attached,
        esdf=esdf,
    )


def refresh(
    constraint_set: CollisionConstraintSet,
    builder: ConstraintBuilder,
    *,
    candidates: Optional[Iterable[CollisionCandidate]] = None,
    support_surfaces: Optional[Iterable[SupportSurface]] = None,
    contact_context: Optional[ContactPolicyContext] = None,
    attached: Optional[AttachedCollisionGeometry] = None,
) -> np.ndarray:
    """Recompute only the parameter values for a new scene. **No symbolic work.**

    This is the real-time path: the TO keeps its `nlpsol` object, its sparsity pattern and its warm
    start, and hands the returned array in as `p`. `to_casadi` is called once, at setup.

    A phase change or a change of authorized manipulator comes through here too — it is a change of
    numbers, not of structure, which is the whole reason the margins are parameters.
    """
    values, _, _ = builder.parameter_values(
        constraint_set.candidates if candidates is None else candidates,
        constraint_set.support_surfaces if support_surfaces is None else support_surfaces,
        context=contact_context or constraint_set.contact_context,
        target_grounded=constraint_set.target is not None,
        attached=constraint_set.attached if attached is None else attached,
    )
    return values


def summary(constraint_set: CollisionConstraintSet) -> dict[str, Any]:
    """A flat, JSON-friendly digest for logs and regression comparison."""
    spec = constraint_set.constraints
    by_type: dict[str, int] = {}
    for candidate in constraint_set.candidates:
        by_type[candidate.source_type.value] = by_type.get(candidate.source_type.value, 0) + 1
    return {
        "status": constraint_set.status.value,
        "validity": constraint_set.validity.value,
        "geometry_certified": constraint_set.geometry_certified,
        "grounding_status": constraint_set.grounding_status.value,
        "phase": constraint_set.phase.value,
        "active_manipulators": sorted(
            m.value for m in constraint_set.contact_context.active_manipulators
        ),
        "attached": None if constraint_set.attached is None else constraint_set.attached.label,
        "esdf": (None if constraint_set.esdf is None else
                 {k: v for k, v in constraint_set.esdf.stats.items() if k != "per_camera"}),
        "frame_index": constraint_set.frame_index,
        "frame_id": constraint_set.frame_id,
        "n_candidates": len(constraint_set.candidates),
        "candidates_by_type": by_type,
        "n_support_surfaces": len(constraint_set.support_surfaces),
        "has_target": constraint_set.has_target,
        "n_constraints": None if spec is None else spec.n_constraints,
        "n_active_slots": None if spec is None else int(spec.active_mask.sum()),
        "max_candidates": None if spec is None else spec.max_candidates,
        "horizon": None if spec is None else spec.horizon,
        "squared_distance": None if spec is None else spec.squared_distance,
        "notes": list(constraint_set.notes),
        "profile_ms": dict(constraint_set.profile),
        "metrics": dict(constraint_set.metrics),
    }


__all__ = ["build_constraint_set", "refresh", "summary", "to_casadi", "warm_start_map"]
