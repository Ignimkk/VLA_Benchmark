"""Which clusters may become the **manipulated object**, and which one is the **destination** (T26).

## Why this exists

T14 seq 12–21: the apple was behind the fingers, the crate led the attention ranking at score 0.015,
grounding named it the target, and `_target_ball` built a 246 mm ball around it and deleted 15.8k
crate seeds from the field the fingers were checked against. T20 stopped the *hand-over* (a hidden
object is `occluded`, not replaced), but nothing said the crate could not be the manipulated object
in the first place — `TargetConfirm`'s `first` adopted whatever led.

A radius cap would be the wrong fix (user ruling 2026-09-28: "crate 는 어떤 상황에서도 공이 될 수
없다"; "반지름 상한은 답이 아니다"). A cap would still let the crate *be* the manipulated object and
merely limit the damage. The rule here makes that state unreachable:

    admissible  =  graspable                 narrowest principal extent  ≤  max gripper opening
                   AND not the destination   fewer than `destination_overlap_fraction` of the
                                             cluster's points lie within one voxel of the
                                             registered destination's points

`max_opening` is **computed from the robot model** (the fingers' collision geometry at the finger
joints' open limit — `robot_models.UrdfSphereChain.gripper_openings`), never written as a number.
When nobody can say what the gripper's opening is, *nothing* is graspable (fail-closed): the only
alternative is to admit anything, which is the T14 failure.

## The destination is registered from perception

The crate is a **free body** in `model_transport.xml` (`crate`, joint 31, `free`), so the static
geometry channel (`--static-geometry auto`, which drops free bodies) cannot supply it. It is
therefore registered from what the cameras see: a cluster that is **known not graspable** and leads
the attention ranking at `target_switch_min_score` or above for `target_confirm_frames` frames running
becomes the destination (`DestinationRegistry`). A caller that knows the destination injects its
points instead (`AG3S.reset(destination_points=...)`), and the injection takes precedence.

The destination is **never removed from any field** (F18's thin margin for the *held* object's
points against it is unchanged). It only takes part here, as the thing a manipulated object must
not be.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

#: Sources of `MaxOpening.value_m`.
FROM_ROBOT_MODEL, FROM_CONFIG, UNAVAILABLE = "robot_model", "config", "unavailable"
#: Sources of a `DestinationGeometry`.
INJECTED, REGISTERED = "injected", "registered"

#: `Admissibility.reason` vocabulary. `admissible` is the only one that admits.
ADMISSIBLE = "admissible"
NOT_GRASPABLE = "not_graspable"
MAX_OPENING_UNKNOWN = "max_opening_unknown"
NO_EXTENT = "no_extent"
DESTINATION_OVERLAP = "destination_overlap"


class ExclusionInvariantViolation(RuntimeError):
    """Exclusion geometry was about to be built from something with no admissibility evidence.

    Raised only in strict mode (`AG3S(strict_invariants=True)` or `AG3S_STRICT_INVARIANTS=1`, which
    the unit tests set). In production the pipeline refuses the geometry and records
    `invariant_violation` instead — admissible-only is not a limit to be enforced at the consumer,
    it is a property of what reaches it; reaching it anyway is a bug to be found, not a case to
    handle.
    """

    def __init__(self, detail: str, record: Optional[dict] = None):
        super().__init__(detail)
        self.detail = str(detail)
        self.record = dict(record or {})


# ------------------------------------------------------------------------------ geometry


def principal_extents(points: np.ndarray) -> np.ndarray:
    """`(3,)` extents of `points` along their principal axes, **ascending** (m).

    PCA axes, so the answer does not depend on how the object sits in the base frame: a banana
    lying diagonally on the table measures its real thickness, not the diagonal of its AABB. Fewer
    than two points have no extent (zeros); a degenerate set (collinear/coplanar) has zeros in the
    missing directions, which is the truth about that set.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return np.zeros(3, np.float64)
    centred = pts - pts.mean(axis=0)
    cov = centred.T @ centred / max(len(pts) - 1, 1)
    _, axes = np.linalg.eigh(cov)
    proj = centred @ axes
    return np.sort(proj.max(axis=0) - proj.min(axis=0))


@dataclasses.dataclass(frozen=True)
class MaxOpening:
    """The widest object the gripper can close around, and where that number came from.

    `value_m` is the **inner-face gap** of the finger collision geometry at the finger joints' open
    limit (not the finger-origin gap, which is larger by the fingers' own thickness). `None` means
    no source could say — every cluster is then inadmissible (`max_opening_unknown`).
    """

    value_m: Optional[float]
    source: str
    detail: dict = dataclasses.field(default_factory=dict)

    @property
    def known(self) -> bool:
        return self.value_m is not None and math.isfinite(float(self.value_m)) \
            and float(self.value_m) > 0.0

    def record(self) -> dict:
        return {
            "value_mm": None if self.value_m is None else round(float(self.value_m) * 1000.0, 2),
            "source": self.source,
            **self.detail,
        }


def max_opening_from_models(*models: Any) -> MaxOpening:
    """Ask each model, in order, for its grippers' openings; the first that answers wins.

    A model answers through `gripper_openings()` (`UrdfSphereChain` implements it from the finger
    links' spheres and the finger joints' limits). The smallest opening over its grippers is used —
    "graspable" must hold for whichever hand closes.
    """
    tried = []
    for model in models:
        if model is None:
            continue
        fn = getattr(model, "gripper_openings", None)
        name = type(model).__name__
        if not callable(fn):
            tried.append(f"{name}: no gripper_openings()")
            continue
        openings = list(fn() or ())
        if not openings:
            tried.append(f"{name}: no parallel gripper found")
            continue
        best = min(openings, key=lambda o: float(o["inner_gap_m"]))
        return MaxOpening(
            value_m=float(best["inner_gap_m"]),
            source=FROM_ROBOT_MODEL,
            detail={
                "model": name,
                "rule": "min over grippers of the finger collision geometry's inner-face gap "
                        "along the closing axis, finger joints at the open limit",
                "grippers": [_jsonable(o) for o in openings],
            })
    return MaxOpening(None, UNAVAILABLE, {"tried": tried or ["no robot model"]})


def resolve_max_opening(config_value: Optional[float], *models: Any) -> MaxOpening:
    """`clustering.gripper_max_opening` when set (offline tools with no robot model), else the
    robot model's own number, else unavailable."""
    if config_value is not None:
        return MaxOpening(float(config_value), FROM_CONFIG,
                          {"rule": "clustering.gripper_max_opening"})
    return max_opening_from_models(*models)


# ---------------------------------------------------------------------------- destination


@dataclasses.dataclass(frozen=True)
class DestinationGeometry:
    """Where the manipulated object is being taken — the crate (guide §4.1's third role)."""

    id: int
    source: str                      # `injected` | `registered`
    points: np.ndarray               # (N, 3) base frame, last observation
    registered_frame: int
    last_seen_frame: int
    n_observations: int
    score: Optional[float] = None
    extents_m: Optional[np.ndarray] = None
    #: KD-tree over `points`, built on first use (the overlap test runs per cluster per frame).
    _kdtree: Any = dataclasses.field(default=None, compare=False, repr=False)

    def tree(self):
        if self._kdtree is None:
            from scipy.spatial import cKDTree

            object.__setattr__(self, "_kdtree",
                               cKDTree(np.asarray(self.points, np.float64).reshape(-1, 3)))
        return self._kdtree

    @property
    def centroid(self) -> np.ndarray:
        return np.asarray(self.points, np.float64).reshape(-1, 3).mean(axis=0)

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(self.points, np.float64).reshape(-1, 3)
        return pts.min(axis=0), pts.max(axis=0)

    @property
    def label(self) -> str:
        """Name for `GraspLatch.set_destination` and logs."""
        return f"destination{self.id}"

    def record(self) -> dict:
        lo, hi = self.aabb
        return {
            "id": int(self.id),
            "label": self.label,
            "source": self.source,
            "n_points": int(len(self.points)),
            "centroid": [round(float(v), 5) for v in self.centroid],
            "aabb_min": [round(float(v), 5) for v in lo],
            "aabb_max": [round(float(v), 5) for v in hi],
            "extents_mm": (None if self.extents_m is None
                           else [round(float(v) * 1000.0, 1) for v in self.extents_m]),
            "registered_frame": int(self.registered_frame),
            "last_seen_frame": int(self.last_seen_frame),
            "n_observations": int(self.n_observations),
            "score": None if self.score is None else float(self.score),
        }


def destination_overlap(points: np.ndarray, destination: Optional[DestinationGeometry],
                        distance: float) -> float:
    """Fraction of `points` within `distance` of any destination point (0.0 without one)."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if destination is None or not len(pts) or not len(destination.points):
        return 0.0
    d, _ = destination.tree().query(pts, k=1,
                                    distance_upper_bound=float(distance) * (1.0 + 1e-9))
    return float(np.mean(np.isfinite(d)))


class DestinationRegistry:
    """The destination, registered from perception or injected by the caller.

    Registration: the **attention leader** (rank 0) is *known not graspable* and scores at least
    `min_score`, for `frames` frames running on the same object. "Same object" is a centroid within
    `tolerance` of the running candidate's centroid **or inside its AABB grown by `tolerance`** — the
    crate is 0.18 × 0.30 m, and which part of it the cameras see moves its partial-view centroid by
    more than the 60 mm that separates two small objects.

    Once registered, a non-graspable cluster matching it by the same rule refreshes its geometry
    (last observation, not a union: a registration that grew forever would swallow what is put into
    it). A graspable-looking fragment of it (one crate wall seen edge-on) does not refresh it; it is
    kept out of the manipulated role by the overlap rule instead.

    An injected destination is never replaced or refreshed by perception.
    """

    def __init__(self, *, frames: int, min_score: float, tolerance: float):
        self.frames = int(frames)
        self.min_score = float(min_score)
        self.tolerance = float(tolerance)
        self.current: Optional[DestinationGeometry] = None
        self._next_id = 0
        self._candidate: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._streak = 0
        #: What the last `observe()` did — `registered` / `updated` / None.
        self.last_event: Optional[str] = None

    def reset(self, injected_points: Optional[np.ndarray] = None) -> None:
        self.current = None
        self._next_id = 0
        self._candidate = None
        self._streak = 0
        self.last_event = None
        if injected_points is not None:
            self.inject(injected_points, frame=-1)

    def inject(self, points: np.ndarray, *, frame: int) -> None:
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        if not len(pts):
            return
        prev = self.current
        self.current = DestinationGeometry(
            id=prev.id if prev is not None and prev.source == INJECTED else self._take_id(),
            source=INJECTED, points=pts, registered_frame=int(frame), last_seen_frame=int(frame),
            n_observations=1, extents_m=principal_extents(pts))
        self.last_event = "injected"

    @property
    def streak(self) -> int:
        return self._streak

    def observe(self, clusters: Sequence[Any], cluster_points: Sequence[np.ndarray],
                graspable: Sequence[Optional[bool]], *, frame: int) -> Optional[str]:
        """One frame. `clusters` score-sorted; `graspable[i]` True/False/None (unknown)."""
        self.last_event = None
        if not clusters:
            return None
        if self.current is not None:
            if self.current.source == REGISTERED:
                i = self._match(clusters, graspable, self.current)
                if i is not None:
                    self.current = dataclasses.replace(
                        self.current, points=np.asarray(cluster_points[i], np.float64),
                        _kdtree=None,
                        last_seen_frame=int(frame),
                        n_observations=self.current.n_observations + 1,
                        score=float(clusters[i].target_score),
                        extents_m=_extents_of(clusters[i], cluster_points[i]))
                    self.last_event = "updated"
            return self.last_event

        leader = clusters[0]
        if graspable[0] is False and float(leader.target_score) >= self.min_score:
            pts = np.asarray(cluster_points[0], np.float64).reshape(-1, 3)
            box = (np.asarray(leader.centroid, np.float64).reshape(3),
                   pts.min(axis=0), pts.max(axis=0))
            if self._candidate is not None and self._same(box[0], self._candidate):
                self._streak += 1
            else:
                self._streak = 1
            self._candidate = box
            if self._streak >= self.frames:
                self.current = DestinationGeometry(
                    id=self._take_id(), source=REGISTERED, points=pts,
                    registered_frame=int(frame), last_seen_frame=int(frame), n_observations=1,
                    score=float(leader.target_score), extents_m=_extents_of(leader, pts))
                self._candidate, self._streak = None, 0
                self.last_event = "registered"
        else:
            self._candidate, self._streak = None, 0
        return self.last_event

    # --- internals ---------------------------------------------------------------------
    def _take_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _same(self, centroid, box) -> bool:
        c0, lo, hi = box
        c = np.asarray(centroid, np.float64).reshape(3)
        if float(np.linalg.norm(c - c0)) <= self.tolerance:
            return True
        return bool(np.all(c >= lo - self.tolerance) and np.all(c <= hi + self.tolerance))

    def _match(self, clusters, graspable, dest: DestinationGeometry) -> Optional[int]:
        lo, hi = dest.aabb
        box = (dest.centroid, lo, hi)
        best, best_d = None, float("inf")
        for i, c in enumerate(clusters):
            if graspable[i] is not False:
                continue
            if not self._same(c.centroid, box):
                continue
            d = float(np.linalg.norm(np.asarray(c.centroid, np.float64).reshape(3) - box[0]))
            if d < best_d:
                best, best_d = i, d
        return best


def _extents_of(cluster: Any, points: np.ndarray) -> np.ndarray:
    ext = getattr(cluster, "principal_extents", None)
    return principal_extents(points) if ext is None else np.asarray(ext, np.float64)


# ---------------------------------------------------------------------------- the verdict


@dataclasses.dataclass(frozen=True)
class Admissibility:
    """Why a cluster may or may not become the manipulated object — the evidence, kept.

    `graspable` is tri-state: None when it could not be decided (no max opening, no extent), and
    only `True` admits. The pipeline re-checks `narrowest_extent_m ≤ max_opening_m` and the overlap
    before it builds any exclusion geometry (`AG3S._exclusion_gate`).
    """

    graspable: Optional[bool]
    narrowest_extent_m: Optional[float]
    extents_m: tuple[float, ...]
    max_opening_m: Optional[float]
    destination_overlap: float
    overlaps_destination: bool
    destination_id: Optional[int]
    reason: str

    @property
    def admissible(self) -> bool:
        return self.reason == ADMISSIBLE

    def record(self) -> dict:
        return {
            "admissible": bool(self.admissible),
            "reason": self.reason,
            "graspable": self.graspable,
            "narrowest_extent_mm": (None if self.narrowest_extent_m is None
                                    else round(float(self.narrowest_extent_m) * 1000.0, 2)),
            "extents_mm": [round(float(v) * 1000.0, 1) for v in self.extents_m],
            "max_opening_mm": (None if self.max_opening_m is None
                               else round(float(self.max_opening_m) * 1000.0, 2)),
            "destination_overlap": round(float(self.destination_overlap), 4),
            "overlaps_destination": bool(self.overlaps_destination),
            "destination_id": self.destination_id,
        }


def graspable(extents: Optional[np.ndarray], max_opening_m: Optional[float]) -> Optional[bool]:
    """True/False, or None when either side is unknown."""
    if max_opening_m is None or extents is None:
        return None
    ext = np.asarray(extents, np.float64).reshape(-1)
    if not ext.size or not np.all(np.isfinite(ext)):
        return None
    return bool(float(ext.min()) <= float(max_opening_m))


def assess(points: np.ndarray, *, extents: Optional[np.ndarray], max_opening_m: Optional[float],
           destination: Optional[DestinationGeometry], overlap_distance: float,
           overlap_fraction: float) -> Admissibility:
    """The admissibility of one cluster (or one observed geometry)."""
    ext = principal_extents(points) if extents is None else np.asarray(extents, np.float64)
    g = graspable(ext, max_opening_m)
    narrow = float(ext.min()) if ext.size and len(np.asarray(points).reshape(-1, 3)) >= 2 else None
    overlap = destination_overlap(points, destination, overlap_distance)
    overlaps = bool(destination is not None and overlap >= float(overlap_fraction))
    if max_opening_m is None:
        reason = MAX_OPENING_UNKNOWN
    elif narrow is None or g is None:
        reason = NO_EXTENT
    elif not g:
        reason = NOT_GRASPABLE
    elif overlaps:
        reason = DESTINATION_OVERLAP
    else:
        reason = ADMISSIBLE
    return Admissibility(
        graspable=g, narrowest_extent_m=narrow,
        extents_m=tuple(float(v) for v in ext.reshape(-1)),
        max_opening_m=None if max_opening_m is None else float(max_opening_m),
        destination_overlap=float(overlap), overlaps_destination=overlaps,
        destination_id=None if destination is None else int(destination.id), reason=reason)


# ------------------------------------------------------------------ pre-grasp association (T30 F3)


#: `Association.failed` vocabulary.
ASSOC_SHIFT, ASSOC_NARROWEST, ASSOC_MIDDLE = "shift", "narrowest", "middle"


@dataclasses.dataclass(frozen=True)
class Association:
    """Whether this frame's cluster may be **the same object** as the manipulated object's last
    admissible geometry, before a grasp (T30 F3) — the evidence, kept.

    An object resting on the table does not move until the hand touches it. So before the grasp a
    cluster is the same object only if

      (i)  its centroid is within the object's own bounding radius of the last admissible
           geometry's centroid (`shift_m ≤ radius_m`), and
      (ii) its narrowest and middle principal extents are each within a factor `max_ratio` of the
           last admissible geometry's (`1/max_ratio ≤ new/ref ≤ max_ratio`).

    (i) stops a neighbour (the orange 105 mm away) from being taken as the apple; (ii) stops a
    cluster that is the apple *merged* with a neighbour — whose centroid can still be close — and a
    small fragment of something else. The largest extent is not tested: it is the axis along which
    a partial view and a merge both change a cluster most, and the two tested axes already bound it.
    """

    accepted: bool
    shift_m: float
    radius_m: float
    #: new / reference, narrowest and middle principal extent.
    extent_ratio: tuple[float, float]
    extents_m: tuple[float, ...]
    reference_extents_m: tuple[float, ...]
    max_ratio: float
    #: Which tests failed (`shift` · `narrowest` · `middle`), empty when accepted.
    failed: tuple[str, ...] = ()
    #: T30 F3b — fraction of the anchor's points within `anchor_distance` of this cluster's points
    #: (how much of the anchor it shows), and of this cluster's points near the anchor's. None when
    #: the anchor rule did not run.
    coverage: Optional[float] = None
    containment: Optional[float] = None
    #: True when accepted but `coverage` < the minimum: the observation drops part of the anchor,
    #: so it must not replace it (`subset_kept_anchor`).
    subset: bool = False

    @property
    def outcome(self) -> str:
        """`rejected` · `subset_kept_anchor` · `accepted`."""
        if not self.accepted:
            return "rejected"
        return "subset_kept_anchor" if self.subset else "accepted"

    def record(self) -> dict:
        return {
            "accepted": bool(self.accepted),
            "shift_mm": round(float(self.shift_m) * 1000.0, 2),
            "radius_mm": round(float(self.radius_m) * 1000.0, 2),
            "shift_over_radius": (round(float(self.shift_m / self.radius_m), 3)
                                  if self.radius_m > 0.0 else None),
            "extent_ratio": [round(float(v), 3) for v in self.extent_ratio],
            "extents_mm": [round(float(v) * 1000.0, 1) for v in self.extents_m],
            "reference_extents_mm": [round(float(v) * 1000.0, 1) for v in self.reference_extents_m],
            "max_ratio": float(self.max_ratio),
            "failed": list(self.failed),
            "outcome": self.outcome,
            "coverage": None if self.coverage is None else round(float(self.coverage), 4),
            "containment": None if self.containment is None else round(float(self.containment), 4),
        }


def _ratio(new: float, ref: float) -> float:
    """new / ref, with 0/0 = 1 (both degenerate) and x/0 = inf."""
    if ref <= 1e-9:
        return 1.0 if new <= 1e-9 else math.inf
    return float(new) / float(ref)


def associate(centroid: np.ndarray, extents: np.ndarray, *, reference_centroid: np.ndarray,
              reference_radius: float, reference_extents: np.ndarray,
              max_ratio: float) -> Association:
    """The T30 F3 test of one cluster against the manipulated object's last admissible geometry.

    `extents` / `reference_extents`: principal extents (m, any order — sorted here).
    """
    ext = np.sort(np.asarray(extents, np.float64).reshape(-1))
    ref = np.sort(np.asarray(reference_extents, np.float64).reshape(-1))
    shift = float(np.linalg.norm(np.asarray(centroid, np.float64).reshape(3)
                                 - np.asarray(reference_centroid, np.float64).reshape(3)))
    radius = float(reference_radius)
    ratios = (_ratio(ext[0], ref[0]), _ratio(ext[1], ref[1]))
    k = float(max_ratio)
    failed = []
    if not shift <= radius:
        failed.append(ASSOC_SHIFT)
    for name, r in zip((ASSOC_NARROWEST, ASSOC_MIDDLE), ratios):
        if not (1.0 / k <= r <= k):
            failed.append(name)
    return Association(
        accepted=not failed, shift_m=shift, radius_m=radius, extent_ratio=ratios,
        extents_m=tuple(float(v) for v in ext), reference_extents_m=tuple(float(v) for v in ref),
        max_ratio=k, failed=tuple(failed))


def anchor_cover(assoc: Association, anchor_points: np.ndarray, points: np.ndarray, *,
                 distance: float, min_coverage: float) -> Association:
    """Add the T30 F3b anchor numbers to an accepted `Association`.

    `coverage` = fraction of `anchor_points` within `distance` of `points` — how much of the
    anchor this observation shows. Below `min_coverage` it drops part of the anchor (the hand hides
    the apple and only the orange beside it is left in the blob), so it is a `subset` and must not
    replace the anchor. `containment` (the reverse fraction) is recorded for reading, not tested.
    """
    from scipy.spatial import cKDTree

    a = np.asarray(anchor_points, np.float64).reshape(-1, 3)
    b = np.asarray(points, np.float64).reshape(-1, 3)
    if not len(a) or not len(b):
        cov, cont = 0.0, 0.0
    else:
        d_ab, _ = cKDTree(b).query(a, k=1)
        d_ba, _ = cKDTree(a).query(b, k=1)
        cov = float(np.mean(d_ab <= float(distance)))
        cont = float(np.mean(d_ba <= float(distance)))
    return dataclasses.replace(assoc, coverage=cov, containment=cont,
                               subset=bool(cov < float(min_coverage)))


@dataclasses.dataclass(frozen=True)
class HandSpheres:
    """The hands' collision spheres at one robot state, in the cloud's (base) frame (T32b).

    What `TargetConfirm.note_hand` reads to tell "the hand is hiding the object" from "the object
    is gone". `links[i]` names the link of sphere `i` (palm `ee_*` or a finger).
    """

    centres: np.ndarray
    radii: np.ndarray
    links: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        c = np.asarray(self.centres, np.float64).reshape(-1, 3)
        r = np.asarray(self.radii, np.float64).reshape(-1)
        if len(c) != len(r):
            raise ValueError(f"HandSpheres: {len(c)} centres but {len(r)} radii")
        object.__setattr__(self, "centres", c)
        object.__setattr__(self, "radii", r)
        object.__setattr__(self, "links", tuple(str(n) for n in self.links))

    def __len__(self) -> int:
        return int(len(self.radii))

    def gap(self, points: np.ndarray) -> Optional[float]:
        """Smallest distance from any sphere's **surface** to any of `points` (m; negative = a
        point inside a sphere). None when there are no spheres or no points."""
        from scipy.spatial import cKDTree

        p = np.asarray(points, np.float64).reshape(-1, 3)
        if not len(p) or not len(self):
            return None
        d, _ = cKDTree(p).query(self.centres, k=1)
        return float(np.min(d - self.radii))


def hand_links(robot_model: Any) -> tuple[str, ...]:
    """The links of every parallel gripper — the palm (the fingers' parent) and both fingers —
    found **structurally** (`gripper_joint_pairs`, T29), never by name. () when the model cannot say."""
    pairs = getattr(robot_model, "gripper_joint_pairs", None)
    if pairs is None:
        return ()
    out: list[str] = []
    for g in pairs():
        for name in (g["parent_link"], *g["children"]):
            if name not in out:
                out.append(str(name))
    return tuple(out)


def hand_spheres(robot_model: Any, robot_state: Optional[np.ndarray]) -> Optional[HandSpheres]:
    """The hands' collision spheres at `robot_state` (T32b), or None when the model or the state
    is missing or the model has no gripper. Same FK as the self-filter
    (`sphere_centers_numeric`, finger joints at their current values)."""
    if robot_model is None or robot_state is None:
        return None
    links = set(hand_links(robot_model))
    names = getattr(robot_model, "sphere_link_names", None)
    if not links or names is None:
        return None
    keep = np.array([n in links for n in names], bool)
    if not keep.any():
        return None
    centres, radii = robot_model.sphere_centers_numeric(np.asarray(robot_state, np.float64))
    return HandSpheres(np.asarray(centres)[keep], np.asarray(radii)[keep],
                       tuple(n for n, k in zip(names, keep) if k))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


__all__ = [
    "ADMISSIBLE",
    "ASSOC_MIDDLE",
    "ASSOC_NARROWEST",
    "ASSOC_SHIFT",
    "Admissibility",
    "Association",
    "DESTINATION_OVERLAP",
    "DestinationGeometry",
    "DestinationRegistry",
    "ExclusionInvariantViolation",
    "FROM_CONFIG",
    "FROM_ROBOT_MODEL",
    "HandSpheres",
    "INJECTED",
    "MAX_OPENING_UNKNOWN",
    "MaxOpening",
    "NOT_GRASPABLE",
    "NO_EXTENT",
    "REGISTERED",
    "UNAVAILABLE",
    "anchor_cover",
    "assess",
    "associate",
    "destination_overlap",
    "graspable",
    "hand_links",
    "hand_spheres",
    "max_opening_from_models",
    "principal_extents",
    "resolve_max_opening",
]
