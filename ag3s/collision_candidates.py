"""Stage 5b — collision candidates.

The term is the design. "Obstacle" invites the question *is this thing an obstacle?*, which invites
an answer based on semantics, which is how a pipeline ends up deciding that the object it is
reaching for cannot be hit. A **collision candidate** is anything that could touch the robot along
the current or future trajectory, so the question is only *is there mass here?* — and the answer
comes from geometry, never from attention.

Three consequences run through this module:

**The target stays in the set, with its full margin.** It is emitted with `source_type=TARGET` and
`collision_enabled=True` in every phase. Deleting it is what makes a filter blind to the one object
it is closest to, and re-adding it at grasp time makes the barrier discontinuous exactly when
precision matters most. `benchmark/knows_vla` measured the other half of this: with the target
removed and the held object demoted back to an obstacle, its barrier was negative on 58.8% of steps
by construction.

The `safety_margin` written here is the **conservative** one, applying to every robot link. Contact
permission is not a property of the candidate — it is a property of a `(robot link, candidate)`
pair, and it is resolved in `clearance.ClearancePolicy` at constraint-build time. Putting a relaxed
scalar on the candidate would relax it against the torso too.

**Unclassifiable geometry is kept.** `preserve_unknown` emits `UNKNOWN_GEOMETRY` for anything that
clusters too small to be an object. A point that is not explained is still mass.

**Identity is stable across frames.** `CandidateTracker` matches by centroid so a candidate keeps
its `id` and accumulates `track_age`. TO warm-starting depends on it: with ids reassigned per frame,
the previous solution's slot *k* would refer to a different object this frame, and the warm start
would push the robot toward the thing it just avoided.

Phase is an **injected input**. There is no transition logic here — only a lookup of the rule the
caller's phase selects.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional, Sequence

import numpy as np

from benchmark.ag3s.config import CollisionCandidateConfig, ContactConfig, GeometryConfig
from benchmark.ag3s.geometry import containment_report, fit_primitive
from benchmark.ag3s.target_grounding import connected_components_3d, dbscan
from benchmark.ag3s.types import (
    CollisionCandidate,
    Phase,
    PointCloud,
    SourceType,
    SupportSurface,
    TargetGeometry,
)


def _pose_from_centroid(centroid: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = centroid
    return pose


def _half_extents(points: np.ndarray) -> np.ndarray:
    """Axis-aligned half-extents — for logging and visualization, not for constraints."""
    if points.shape[0] == 0:
        return np.zeros(3)
    return (points.max(axis=0) - points.min(axis=0)) / 2.0


# ------------------------------------------------------------------------------- tracking


class CandidateTracker:
    """Stable ids across frames by centroid nearest-neighbour association.

    Matching is a global assignment (`scipy.optimize.linear_sum_assignment`) rather than greedy
    nearest-first. Greedy gets two nearby objects wrong in exactly the case that matters: whichever
    is processed first claims the shared nearest track, and the other is forced to a new id, so two
    adjacent objects can swap identities frame to frame. A global assignment minimises total
    displacement and keeps both.

    Tracks unmatched for `track_max_missed` consecutive frames are dropped. Until then they are
    *retained but not emitted*, so an object that flickers out for a frame — occluded by the arm,
    say — comes back with its original id and its accumulated `track_age`, and TO's warm start
    survives the gap.
    """

    def __init__(self, config: CollisionCandidateConfig | None = None):
        self.config = config or CollisionCandidateConfig()
        self._next_id = 0
        self._centroids: dict[int, np.ndarray] = {}
        self._age: dict[int, int] = {}
        self._missed: dict[int, int] = {}

    def reset(self) -> None:
        self.__init__(self.config)

    @property
    def active_ids(self) -> list[int]:
        return sorted(self._centroids)

    def track_age(self, track_id: int) -> int:
        return self._age.get(track_id, 0)

    def update(self, centroids: Sequence[np.ndarray]) -> list[int]:
        """Assign an id to each detection, in the order given. Call once per frame."""
        from scipy.optimize import linear_sum_assignment

        detections = (
            np.asarray(centroids, np.float64).reshape(-1, 3)
            if len(centroids)
            else np.zeros((0, 3), np.float64)
        )
        existing = self.active_ids
        assigned: list[Optional[int]] = [None] * detections.shape[0]

        if existing and detections.shape[0]:
            previous = np.stack([self._centroids[i] for i in existing])
            cost = np.linalg.norm(detections[:, None, :] - previous[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.config.track_association_radius:
                    assigned[r] = existing[c]

        matched = {i for i in assigned if i is not None}
        for track_id in existing:
            if track_id in matched:
                self._missed[track_id] = 0
            else:
                self._missed[track_id] = self._missed.get(track_id, 0) + 1
                if self._missed[track_id] > self.config.track_max_missed:
                    self._centroids.pop(track_id, None)
                    self._age.pop(track_id, None)
                    self._missed.pop(track_id, None)

        out: list[int] = []
        for i, track_id in enumerate(assigned):
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._age[track_id] = 0
            else:
                self._age[track_id] = self._age.get(track_id, 0) + 1
            self._centroids[track_id] = detections[i]
            self._missed[track_id] = 0
            out.append(track_id)
        return out


# ---------------------------------------------------------------------------- the stage


def _spatial_groups(points: np.ndarray, indices: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Split `indices` into at most `n_groups` spatially coherent runs along their longest axis.

    Crude on purpose. The alternative — one aggregate for everything — produces a sphere spanning the
    whole scene, which is conservative and also forbids every trajectory, so the robot stops rather
    than detours. Sorting along the direction the points are most spread out and cutting into equal
    runs keeps each aggregate near its members at a cost of one argsort, which is what this stage can
    afford on the residue of a frame.
    """
    if indices.size == 0 or n_groups <= 0:
        return []
    if indices.size <= n_groups:
        return [indices[i : i + 1] for i in range(indices.size)]
    spread = points.max(axis=0) - points.min(axis=0)
    order = np.argsort(points[:, int(np.argmax(spread))], kind="stable")
    return [indices[chunk] for chunk in np.array_split(order, n_groups) if chunk.size]


def _cluster(points: np.ndarray, config: CollisionCandidateConfig) -> np.ndarray:
    if config.cluster_method == "dbscan":
        return dbscan(points, config.eps, config.min_points)
    return connected_components_3d(points, config.eps)


def generate_candidates(
    cloud: PointCloud,
    *,
    phase: Phase | str,
    target: Optional[TargetGeometry] = None,
    support_surfaces: Iterable[SupportSurface] = (),
    support_mask: Optional[np.ndarray] = None,
    config: CollisionCandidateConfig | None = None,
    geometry_config: GeometryConfig | None = None,
    contact_config: ContactConfig | None = None,
    tracker: Optional[CandidateTracker] = None,
    timestamp: Optional[float] = None,
) -> tuple[list[CollisionCandidate], dict[str, int]]:
    """Turn a reconstructed cloud into the candidate set TO must respect.

    Note what is *not* a parameter: there is no attention argument. Everything attention had to say
    was said in `target_grounding`, and letting it back in here is precisely the mistake this design
    exists to avoid.

    Returns `(candidates, stats)`. `stats` reports `n_object`, `n_unknown`, `n_support`,
    `n_target`, `n_residual_points` and `n_unassigned_points` — the last being points that were too
    isolated even for `unknown_min_points`, surfaced as a number rather than dropped silently.
    """
    cfg = config or CollisionCandidateConfig()
    geo = geometry_config or GeometryConfig()
    contact = contact_config or ContactConfig()
    phase = Phase.parse(phase)
    ts = time.time() if timestamp is None else float(timestamp)
    tracker = tracker if tracker is not None else CandidateTracker(cfg)

    n = len(cloud)
    support_mask = (
        np.zeros(n, bool) if support_mask is None else np.asarray(support_mask, bool).reshape(-1)
    )
    target_mask = np.zeros(n, bool)
    if target is not None:
        target_mask[target.point_indices] = True

    residual_idx = np.nonzero(~support_mask & ~target_mask)[0]
    stats = {
        "n_object": 0, "n_unknown": 0, "n_support": 0, "n_target": 0, "n_overflow": 0,
        "n_residual_points": int(residual_idx.size), "n_unassigned_points": 0,
        "n_overflow_points": 0,
    }

    # --- B. residual clustering ---------------------------------------------------------
    groups: list[tuple[SourceType, np.ndarray]] = []
    leftovers: list[np.ndarray] = []
    if residual_idx.size:
        labels = _cluster(cloud.points[residual_idx], cfg)
        for label in np.unique(labels):
            members = residual_idx[labels == label]
            if label < 0:
                leftovers.append(members)  # DBSCAN noise
            elif members.size >= cfg.min_points:
                groups.append((SourceType.OBJECT, members))
            else:
                leftovers.append(members)

    # --- C. unknown geometry: never discard what did not classify ------------------------
    residue: list[np.ndarray] = []  # points headed for a conservative aggregate
    if cfg.preserve_unknown and leftovers:
        loose = np.concatenate(leftovers)
        # Re-group the leftovers among themselves. Points DBSCAN called noise because they were not
        # dense *enough* are often still a small connected body — a cable, a thin handle, a partly
        # occluded object — and that body is what the robot would hit.
        sub = connected_components_3d(cloud.points[loose], cfg.eps)
        for label in np.unique(sub):
            members = loose[sub == label]
            if members.size >= cfg.unknown_min_points:
                groups.append((SourceType.UNKNOWN_GEOMETRY, members))
            else:
                # Too few points to fit a shape *individually*. That says nothing about whether
                # there is mass here, so they go to the overflow aggregate rather than to a counter.
                residue.append(members)
    elif leftovers:
        residue.extend(leftovers)

    # Biggest first, so the `max_clusters` cap sends the least massive things to the aggregate rather
    # than an arbitrary slice. Ties broken by lowest index for determinism.
    groups.sort(key=lambda g: (-g[1].size, int(g[1][0])))
    if len(groups) > cfg.max_clusters:
        residue.extend(members for _, members in groups[cfg.max_clusters :])
        groups = groups[: cfg.max_clusters]

    # --- association: one id per detection, stable across frames -------------------------
    detections: list[tuple[SourceType, np.ndarray, np.ndarray]] = []
    for source_type, members in groups:
        detections.append((source_type, members, cloud.points[members].mean(axis=0)))
    if target is not None:
        detections.append((SourceType.TARGET, target.point_indices, target.centroid))
    ids = tracker.update([centroid for _, _, centroid in detections])

    target_rule = contact.rule_for(phase)
    candidates: list[CollisionCandidate] = []
    primitive_seconds = 0.0
    for (source_type, members, centroid), track_id in zip(detections, ids):
        points = target.points if source_type is SourceType.TARGET else cloud.points[members]
        is_target = source_type is SourceType.TARGET
        # The **conservative** margin — the one that applies to every robot link that is not
        # explicitly authorized to make contact. Any relaxation is per (robot sphere, slot) and lives
        # in `clearance.ClearancePolicy`; carrying a single relaxed number here would silently grant
        # the torso the clearance meant for the fingertips.
        margin = geo.safety_margin
        fit_started = time.perf_counter()
        primitive = fit_primitive(
            points,
            geo.primitive_type,
            min_radius=geo.min_radius,
            uncertainty=geo.perception_uncertainty,
            semantic_role=source_type.value,
            candidate_id=track_id,
            safety_margin=margin,
            collision_enabled=True,
            contact_permission=target_rule.contact_permission if is_target else False,
        )
        primitive_seconds += time.perf_counter() - fit_started
        candidates.append(
            CollisionCandidate(
                id=track_id,
                source_type=source_type,
                semantic_role=source_type.value,
                geometry=[primitive],
                pose=_pose_from_centroid(centroid),
                dimensions=_half_extents(points),
                confidence=target.confidence if is_target else 1.0,
                safety_margin=margin,
                # Always on. A candidate is switched off only by ceasing to exist physically, which
                # is not something a phase can do.
                collision_enabled=True,
                contact_permission=target_rule.contact_permission if is_target else False,
                phase_rule=phase.value if is_target else "always_on",
                timestamp=ts,
                frame_id=cloud.frame_id,
                track_age=tracker.track_age(track_id),
                centroid=centroid,
                point_indices=members,
                point_count=int(members.size),
            )
        )
        stats["n_target" if is_target else
              ("n_object" if source_type is SourceType.OBJECT else "n_unknown")] += 1

    # --- D. overflow: geometry that could not get its own candidate ----------------------
    # Nothing here is deleted. Points that lost the `max_clusters` competition, and residue too
    # sparse for `unknown_min_points`, are folded into a small number of containing primitives. The
    # shape is coarse and the excess volume is real; both are reported. What matters is that the mass
    # is still in the constraint set, because a trajectory routed through where it used to be hits it
    # just as hard as if it had been a properly fitted object.
    if residue:
        loose = np.concatenate(residue)
        stats["n_overflow_points"] = int(loose.size)
        if cfg.overflow_groups <= 0:
            # The ablation. The points are genuinely gone, and the caller must be told.
            stats["n_unassigned_points"] += int(loose.size)
        else:
            fit_started = time.perf_counter()
            for members in _spatial_groups(cloud.points[loose], loose, cfg.overflow_groups):
                points = cloud.points[members]
                # Forced to SPHERE regardless of `geometry.primitive`: an aggregate of unrelated
                # fragments has no meaningful principal axis, and the bounding sphere is the one fit
                # whose containment needs no argument.
                primitive = fit_primitive(
                    points,
                    "sphere",
                    min_radius=geo.min_radius,
                    uncertainty=geo.perception_uncertainty,
                    semantic_role=SourceType.OVERFLOW.value,
                    candidate_id=-1,
                    safety_margin=geo.safety_margin,
                    collision_enabled=True,
                    contact_permission=False,
                )
                centroid = points.mean(axis=0)
                candidates.append(
                    CollisionCandidate(
                        id=-(1000 + stats["n_overflow"]),  # negative: not a tracked object identity
                        source_type=SourceType.OVERFLOW,
                        semantic_role=SourceType.OVERFLOW.value,
                        geometry=[primitive],
                        pose=_pose_from_centroid(centroid),
                        dimensions=_half_extents(points),
                        confidence=0.0,  # nothing was identified; only located
                        safety_margin=geo.safety_margin,
                        collision_enabled=True,
                        contact_permission=False,
                        phase_rule="always_on",
                        timestamp=ts,
                        frame_id=cloud.frame_id,
                        track_age=0,
                        centroid=centroid,
                        point_indices=members,
                        point_count=int(members.size),
                    )
                )
                stats["n_overflow"] += 1
                report = containment_report(primitive, points)
                stats["overflow_containment_rate"] = min(
                    stats.get("overflow_containment_rate", 1.0), report["containment_rate"]
                )
                stats["overflow_excess_volume_ratio"] = max(
                    stats.get("overflow_excess_volume_ratio", 0.0), report["excess_volume_ratio"]
                )
            primitive_seconds += time.perf_counter() - fit_started

    # Spec §6 wants primitive fitting as its own row in the latency table, but it is naturally
    # interleaved with candidate construction. Reporting it as a number here keeps the two stages
    # separable without threading a profiler through this function's signature.
    stats["primitive_fit_ms"] = primitive_seconds * 1000.0

    # --- A. support surfaces --------------------------------------------------------------
    # Emitted as candidates for completeness of the set, but with **no primitives**: a plane's
    # constraint is the exact half-space `n . p >= d + margin` carried on the `SupportSurface`
    # itself, and approximating it with spheres would be both wrong and enormous. The constraint
    # builder reads planes from `CollisionConstraintSet.support_surfaces`.
    for surface in support_surfaces:
        candidates.append(
            CollisionCandidate(
                id=-(surface.id + 1),  # negative ids: planes are tracked by the fit, not by centroid
                source_type=SourceType.SUPPORT_SURFACE,
                semantic_role="support_surface",
                geometry=[],
                pose=_pose_from_centroid(surface.normal * surface.offset),
                dimensions=np.zeros(3),
                confidence=1.0,
                safety_margin=surface.safety_margin,
                collision_enabled=True,
                contact_permission=False,
                phase_rule="always_on",
                timestamp=ts,
                frame_id=cloud.frame_id,
                track_age=0,
                centroid=surface.normal * surface.offset,
                point_indices=surface.point_indices,
                point_count=surface.point_count,
            )
        )
        stats["n_support"] += 1

    return candidates, stats


__all__ = ["CandidateTracker", "generate_candidates"]
