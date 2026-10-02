"""Stage 4 — attention-guided 3D target grounding.

This is where attention earns its keep and where its limits are contained. The sequence is:

    seeds from attention  ->  3D connectivity growth  ->  clustering  ->  metrics  ->  score

and the ordering is the design. Attention picks *where to start looking*; 3D Euclidean connectivity
decides *how far the object extends*. That split is what makes the stage robust to the failure mode
that defeats a pure-2D pipeline: attention smears across neighbouring objects, so a 2D threshold
merges a mug with the bowl beside it, while in 3D the 5 cm air gap between their surfaces is wider
than any sane `eps` and they simply are not connected.

The growth reaching into the *full* cloud rather than staying inside the seed set is equally
deliberate. The far side of an object gets little attention — it is barely lit, oblique to the
camera, sometimes shadowed — and a cluster built only from high-attention points would have a
centroid biased toward the camera on top of the single-view bias that is already there. Geometry
decides extent; attention only decided which object.

**Support surfaces must be excluded before this runs.** A cup standing on a table is within `eps` of
the table, and connectivity growth would flood from the cup into the entire tabletop and out to
every other object on it. `exclude_mask` is how the caller says "these points are the table". Spec
§3.4 numbers plane fitting after grounding, but the plane fit does not depend on the target, so the
pipeline runs it first and passes the mask in; that is a scheduling choice, not a change of method.

**Failure is reported, never invented.** If no seed survives, or nothing clusters, the result is
`target=None` with a `GroundingStatus` saying which. Stage 5 then treats all geometry as conservative
collision candidates. Fabricating a target out of nothing would be worse than useless: the phase
rules would relax that object's clearance, so a wrong guess actively disables a constraint.

**A low score is not one of those failures any more (T5e, 2026-09-25).** It was: the best cluster
scoring under `target_score_threshold` returned `LOW_SCORE`, and on ep1807 that rejected 35 of 51
chunks whose rank-1 cluster was the right object (T5d measured all 51 — zero empty chunks once only
the ranking is read). The score mixes mean attention with two exp() decays, so its absolute level
tracks the scene's attention scale and does not mean the same thing in two episodes; the *ranking*
does. So the default threshold is 0.0 — rank 1 is the target, and only an empty cluster list is "no
target". The threshold remains configurable, and above 0 it rejects exactly as before.

What a score cannot do, a *count* can: `TargetConfirm` makes a different object earn the target over
`target_confirm_frames` consecutive frames of leading. That is the same rule `GraspLatch` uses, and
it is what actually suppressed the ep1807 flicker (2 chunks of pear, 1 of apple) that a threshold
never addressed.

**Hidden is not replaced (T20, 2026-09-28).** `TargetConfirm` also carries the *manipulated
object*: an id we assign, its last observed geometry and an observation state (`visible |
occluded | lost`). When the held object produces no cluster, the frame's target is that object's
last observed geometry (`occluded`), not the leader — on T14 seq 12 the leader was a crate at score
0.015 because the apple was behind the fingers. After `target_lost_frames` it is `lost`: no target,
`GroundingStatus.LOST`. A switch still happens, on the T5e count, from challengers scoring at least
`target_switch_min_score` (default 0.0 = every challenger counts, as before).

**Only admissible clusters compete (T26, 2026-09-28).** A cluster may become the manipulated object
only if it is *graspable* — its narrowest principal extent fits between the fingers at their open
limit, a number computed from the robot model — and it is not the *destination* (`stages/
admissibility.py`). `first` adopts the highest-ranked admissible cluster, not the leader; a challenger
is the highest-ranked admissible cluster that is not the held one; a held object whose matching
cluster is inadmissible (the apple merged with the fingers or the crate) is `occluded` with reason
`inadmissible_match` and keeps its last admissible geometry. With no admissible cluster and nothing
held the frame is `GroundingStatus.NO_ADMISSIBLE`. The crate therefore never becomes the manipulated
object — whatever its attention score — and is registered as the destination instead. The default
`target_switch_min_score` is 0.1 since T26.

**Before a grasp, a cluster is the held object only if it looks like it (T30 F3, 2026-09-29).** E3
ep1808 t=96: the manipulated id 0 slid onto the orange (the held centroid 42.6 → 46.7 → 85.5 mm from
the apple) and ep1800 onto the banana — each frame's nearest cluster within the 60 mm identity
tolerance was taken as the apple and the held centroid followed it. An object on the table does not
move until the hand touches it, so while `TargetConfirm` is not frozen the nearest cluster is the
held object only if (i) its centroid is within the object's own bounding radius of the last
admissible geometry and (ii) its narrowest and middle extents are within
`clustering.manipulated_extent_ratio` of that geometry's (`admissibility.associate`). Otherwise
the object is `occluded` with reason `association_rejected` and keeps its last admissible geometry;
the rejected cluster is an ordinary challenger under the unchanged T26 switch rule.

**A cluster that may be two objects is split by attention (T31b, 2026-09-29).** The E3 apple was
adopted at t=0 merged with the orange (ep1808) or the banana (ep1800): 30 mm connectivity bridged
the gap between them, and the merged cluster's narrowest extent still fit the gripper. Before a
grasp (`confirm` not frozen), a cluster whose narrowest extent fits the gripper but whose largest
does not is re-clustered at `clustering.attention_split_radius`; when attention separates the
components (`clustering.attention_split_ratio`), each becomes its own cluster, the one with the
largest attention mass being the target candidate (`stages/attention_split.py`). Every decision is
kept: `GroundingResult.splits`, `ClusterInfo.split`, and `ManipulatedIdentity.split` in the record.

**Two side effects of that split, closed in `TargetConfirm` (T32b, 2026-09-29).** (S2) The split is
off while frozen, so the geometry a grasp attempt leaves standing can be the merge again; on the
first call after the freeze is released it is split once (`clustering.split_anchor_on_release`,
with the attention it was observed with). (S3) With the neighbour split out as a clean cluster, it
could lead while the hand hid the apple and win the switch early; a frame where the held object is
unobserved or a subset *and* a hand sphere (`TargetConfirm.note_hand`) is within
`clustering.hand_occlusion_reach` of it does not count toward a switch.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from benchmark.ag3s.config import KDTREE_WORKERS, ClusteringConfig
from benchmark.ag3s.stages.admissibility import (
    Admissibility,
    Association,
    DestinationGeometry,
    DestinationRegistry,
    HandSpheres,
    anchor_cover,
    assess,
    associate,
    graspable,
    principal_extents,
)
from benchmark.ag3s.stages.attention_split import SplitDecision, split_by_attention
from benchmark.ag3s.stages.geometry import fit_sphere, rms_radius
from benchmark.ag3s.types import AttentionPointCloud, GroundingStatus, TargetGeometry

_EPS = 1e-9
#: "Not given" for `TargetConfirm(lost_frames=...)`, where None already means "never lost".
_FROM_CONFIG = object()
#: `TargetConfirm(max_opening=UNCHECKED)`: admissibility is not evaluated at all (T20 behaviour).
#: Only for standalone/offline use — `AG3S` always passes a number, or `None` (= unknown =
#: nothing graspable), and its exclusion gate refuses geometry without admissibility evidence.
UNCHECKED = "unchecked"
#: `ConfirmDecision.mode` when nothing is held and no cluster is admissible (T26).
NO_ADMISSIBLE = "no_admissible"
#: `ConfirmDecision.reason` when the held object's matching cluster was inadmissible (T26).
INADMISSIBLE_MATCH = "inadmissible_match"
#: `ConfirmDecision.reason` when, before a grasp, the cluster at the held object's place failed the
#: association test — moved further than the object's own radius, or changed size (T30 F3).
ASSOCIATION_REJECTED = "association_rejected"
#: `ConfirmDecision.reason` when, before a grasp, the cluster at the held object's place is the
#: object but covers only part of its anchor geometry — the anchor is kept (T30 F3b).
SUBSET_KEPT_ANCHOR = "subset_kept_anchor"
#: `ConfirmDecision.reason` when, before a grasp, the anchor is a component of an attention split
#: and the cluster at its place is a cluster that may again be several objects but could not be
#: split this frame — the anchor is kept, the object counts as seen (T31b).
UNSPLIT_KEPT_ANCHOR = "unsplit_kept_anchor"


# ----------------------------------------------------------------------------- clustering


def dbscan(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """DBSCAN over 3D points. Returns a label per point, `-1` for noise.

    Hand-rolled because scikit-learn is not installed in this workspace and the dependency policy
    says not to add it. The implementation is the textbook one, expressed so that scipy does the
    heavy lifting: neighbour counts come from one KD-tree query, core-core connectivity from
    `query_pairs` plus `connected_components`, and border assignment from a nearest-core query. No
    Python loop runs per point.

    `min_points` counts the point itself, matching scikit-learn's `min_samples`.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pts = np.asarray(points, np.float64).reshape(-1, 3)
    n = pts.shape[0]
    labels = np.full(n, -1, np.int64)
    if n == 0:
        return labels

    tree = cKDTree(pts)
    counts = tree.query_ball_point(pts, eps, return_length=True, workers=KDTREE_WORKERS)
    core = counts >= min_points
    if not core.any():
        return labels

    core_idx = np.nonzero(core)[0]
    core_tree = cKDTree(pts[core_idx])
    pairs = core_tree.query_pairs(eps, output_type="ndarray")
    m = core_idx.size
    graph = coo_matrix(
        (np.ones(pairs.shape[0], np.int8), (pairs[:, 0], pairs[:, 1])), shape=(m, m)
    )
    _, comp = connected_components(graph, directed=False)
    labels[core_idx] = comp

    # Border points: within eps of a core point, but not dense enough to be core themselves. They
    # join the cluster of their nearest core point, which is DBSCAN's rule.
    non_core = np.nonzero(~core)[0]
    if non_core.size:
        dist, j = core_tree.query(pts[non_core], k=1, distance_upper_bound=eps,
                                  workers=KDTREE_WORKERS)
        hit = np.isfinite(dist)
        labels[non_core[hit]] = comp[j[hit]]
    return labels


def connected_components_3d(points: np.ndarray, radius: float) -> np.ndarray:
    """Euclidean connected components — DBSCAN with the core-point test removed.

    This is what `clustering.method: region_growing` uses. The difference from DBSCAN is not
    cosmetic: without a density requirement, a handful of stray points bridging two objects is enough
    to merge them, whereas DBSCAN's `min_points` prunes exactly such sparse bridges. Keeping both
    makes that trade-off measurable instead of assumed.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pts = np.asarray(points, np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n == 0:
        return np.zeros(0, np.int64)
    pairs = cKDTree(pts).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(pairs.shape[0], np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    return comp.astype(np.int64)


def grow_region(
    points: np.ndarray,
    seed_indices: np.ndarray,
    radius: float,
    *,
    max_points: Optional[int] = None,
) -> np.ndarray:
    """Breadth-first growth from seeds through the cloud, one wave at a time.

    Wave-at-a-time so each expansion is a single batched KD-tree query rather than one query per
    point; the number of iterations is the region's diameter in `radius` units, which is small.

    Returns the indices of the grown set, sorted, seeds included.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, np.float64).reshape(-1, 3)
    n = pts.shape[0]
    seeds = np.asarray(seed_indices, np.int64).reshape(-1)
    visited = np.zeros(n, bool)
    if seeds.size == 0:
        return np.zeros(0, np.int64)
    visited[seeds] = True

    tree = cKDTree(pts)
    frontier = seeds
    while frontier.size:
        neighbours = tree.query_ball_point(pts[frontier], radius, workers=KDTREE_WORKERS)
        # T38 B4: the same flattening, iterated in C (`chain.from_iterable`) instead of a Python
        # generator — ≈ 470 k generator steps per frame were most of this function's own time.
        flat = np.fromiter(
            itertools.chain.from_iterable(neighbours), np.int64,
            count=sum(map(len, neighbours)),
        )
        if flat.size == 0:
            break
        candidate = np.unique(flat)
        frontier = candidate[~visited[candidate]]
        visited[frontier] = True
        if max_points is not None and int(visited.sum()) >= max_points:
            break
    return np.nonzero(visited)[0]


# ---------------------------------------------------------------------------------- seeds


def extract_seeds(attention: np.ndarray, config: ClusteringConfig, seed_percentile: float,
                  seed_threshold: Optional[float] = None) -> np.ndarray:
    """Indices of the points attention considers interesting.

    An absolute `seed_threshold` wins over the percentile when set, because a percentile *always*
    yields seeds — even from a flat map, where the top 5% is an arbitrary 5%. An absolute threshold
    can legitimately return nothing, which is how the `NO_SEED` failure mode becomes reachable.

    The `max_seed_points` cap keeps the strongest seeds (ties broken by index, so it is
    deterministic). Capping matters less for correctness than for latency: growth is seeded from
    every one of these.
    """
    att = np.asarray(attention, np.float32).reshape(-1)
    if att.size == 0:
        return np.zeros(0, np.int64)

    if seed_threshold is not None:
        keep = np.nonzero(att >= float(seed_threshold))[0]
    else:
        cut = float(np.percentile(att, seed_percentile))
        keep = np.nonzero(att >= cut)[0]

    if keep.size > config.max_seed_points:
        order = np.argsort(-att[keep], kind="stable")[: config.max_seed_points]
        keep = np.sort(keep[order])
    return keep


# --------------------------------------------------------------------------------- results


@dataclasses.dataclass(frozen=True)
class ClusterInfo:
    """One target candidate, with the six per-cluster metrics spec §3.3 asks for."""

    id: int
    point_indices: np.ndarray
    centroid: np.ndarray
    mean_attention: float
    max_attention: float
    attention_mass: float
    point_count: int
    spatial_compactness: float
    distance_from_attention_peak: float
    target_score: float
    rms_radius: float = 0.0
    #: `(3,)` extents along the cluster's principal axes, ascending (m) — T26's "graspable" reads
    #: the first. None for a hand-built `ClusterInfo` (then it is computed from the points).
    principal_extents: Optional[np.ndarray] = None
    #: T31b: `{role, split_index, parent_n_points, parent_extents_mm}` (`split_index` indexes
    #: `GroundingResult.splits`). `role` is `target`/`obstacle` for a component of a split parent,
    #: `unsplit` (+ `reason`) for a triggered cluster the split left whole. None when not triggered.
    split: Optional[dict] = None


@dataclasses.dataclass(frozen=True)
class GroundingResult:
    """Everything the stage produced, including on failure.

    `clusters` is returned even when `target is None` so a caller can see *what* was rejected and by
    how much — a run that reports LOW_SCORE with a best score of 0.24 against a 0.25 threshold is a
    tuning problem, and one with no clusters at all is a perception problem.
    """

    target: Optional[TargetGeometry]
    status: GroundingStatus
    clusters: tuple[ClusterInfo, ...] = ()
    seed_indices: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0, np.int64))
    attention_peak_index: int = -1
    best_score: float = 0.0
    #: T31b: this frame's split decisions, one per triggered cluster (accepted or not), in the
    #: order the parents were grown. Empty when nothing was triggered or the split did not run.
    splits: tuple[SplitDecision, ...] = ()

    @property
    def ok(self) -> bool:
        return self.target is not None


# --------------------------------------------------------------- cross-frame target confirmation


#: Observation states of the manipulated object (T20, guide §4.3 — the names are the guide's
#: suggestion). `visible`: one of this frame's clusters is it. `occluded`: it produced no cluster
#: this frame, for at most `clustering.target_lost_frames` frames running; its last observed geometry
#: stands in. `lost`: unobserved for longer than that; there is no target, and the reason says so.
VISIBLE, OCCLUDED, LOST = "visible", "occluded", "lost"
MANIPULATED_STATES = (VISIBLE, OCCLUDED, LOST)


@dataclasses.dataclass(frozen=True)
class ManipulatedIdentity:
    """The object this grasp is about, as of one frame (T20, guide §4.1).

    Three things used to be one: the **attention target** (what the policy looked at this frame),
    the **manipulated object** (what this grasp is about) and the destination. `ground_target` is
    about the first; this is the second. Before a grasp they coincide, so nothing downstream behaves
    differently — they part exactly when the apple disappears behind the fingers or attention moves
    to the basket, which is when contact permission and target exclusion must *not* follow attention.

    `id` is ours, assigned by `TargetConfirm` and stable across frames. Cluster indices are not: they
    are this frame's position in a sorted list and mean nothing one frame later.

    `geometry` is the **last observed** `TargetGeometry` of this object — this frame's when it is
    `visible`, an older one when `occluded` or `lost`. Its `point_indices` index the cloud of the
    frame it was observed in, so only `points` / `centroid` / `bounding_geometry` are meaningful
    across frames. It is `None` only between `select()` and `observe()` of an adoption frame, i.e.
    never after `ground_target` returns.
    """

    id: int
    state: str
    geometry: Optional[TargetGeometry]
    #: Where the object was last seen (the held centroid). Follows the object while it is visible.
    centroid: np.ndarray
    #: Consecutive frames without an observation. 0 when `visible`.
    age_frames: int
    #: Grounding-call index of the last observation (`TargetConfirm` counts calls since `reset()`,
    #: which inside `AG3S` equals `frame_index`). -1 if it has never been observed.
    last_seen_frame: int
    #: `TargetGeometry.timestamp` of that observation, or None.
    last_seen_time: Optional[float]
    #: The id this one took over from by a `switch`; None for the episode's first object.
    switched_from: Optional[int]
    #: Grounding-call index this snapshot describes.
    frame: int
    #: T26: the admissibility evidence of the **last observed geometry** (`narrowest_extent_m ≤
    #: max_opening_m`, destination overlap) — what `AG3S._exclusion_gate` checks before any
    #: exclusion geometry is built. None when `TargetConfirm` runs `UNCHECKED`.
    admissibility: Optional[Admissibility] = None
    #: T30 F3: this frame's association test of the cluster at the object's place (accepted on a
    #: `keep`/`hold`, rejected on an `association_rejected` frame), or None when it did not run
    #: (frozen, `UNCHECKED`, test off, no cluster there, nothing to compare with).
    association: Optional[Association] = None
    #: T30 F3b: the current geometry as the pre-grasp anchor — `{manipulated_id, active, source
    #: (first|switch|refined|observed), since_frame, kept_frames, extents_mm, radius_mm, n_points,
    #: centroid}` — or None before any geometry.
    anchor: Optional[dict] = None
    #: T31b: this frame's attention-split record — `{ran, skipped, max_opening_mm, n_triggered,
    #: n_split, decisions: [SplitDecision.record()], manipulated_from_split}` — or None when
    #: grounding did not reach clustering this frame.
    split: Optional[dict] = None
    #: T32b: this frame's hand record — `{available, gap_mm, reach_mm, occluded_by_hand,
    #: challenger_gap_mm, n_spheres, links}`
    #: (`gap_mm` = nearest hand sphere surface to the standing geometry) — or None when the caller
    #: gave no hand this frame.
    hand: Optional[dict] = None

    @property
    def usable(self) -> bool:
        """True when consumers should treat this object as present: `visible` or `occluded`."""
        return self.state in (VISIBLE, OCCLUDED) and self.geometry is not None

    @property
    def points(self) -> np.ndarray:
        if self.geometry is None:
            return np.zeros((0, 3), np.float64)
        return np.asarray(self.geometry.points, np.float64).reshape(-1, 3)

    @property
    def radius(self) -> float:
        """Bounding-sphere radius of the last observation (0.0 if none)."""
        if self.geometry is None:
            return 0.0
        return float(np.asarray(self.geometry.bounding_geometry.dimensions, np.float64)[0])

    def record(self) -> dict[str, Any]:
        """JSON-friendly per-frame record (T20 §3): `metrics["manipulated"]` and the chunk summary."""
        return {
            "id": int(self.id),
            "state": str(self.state),
            "age_frames": int(self.age_frames),
            "centroid": [float(v) for v in np.asarray(self.centroid, np.float64).reshape(3)],
            "radius": float(self.radius),
            "n_points": int(len(self.points)),
            "last_seen_frame": int(self.last_seen_frame),
            "switched_from": None if self.switched_from is None else int(self.switched_from),
            "frame": int(self.frame),
            "admissibility": None if self.admissibility is None else self.admissibility.record(),
            "association": None if self.association is None else self.association.record(),
            "anchor": None if self.anchor is None else dict(self.anchor),
            "split": None if self.split is None else dict(self.split),
            "hand": None if self.hand is None else dict(self.hand),
        }


@dataclasses.dataclass(frozen=True)
class ConfirmDecision:
    """Which object this frame's target is, and why that one (T5e, T20).

    Kept as a record rather than a bare return value because the *why* is the observable part: a
    frame where the leader was overruled looks identical to a frame where it won, unless the
    decision says so.
    """

    #: The chosen candidate. **None** when the manipulated object produced no cluster this frame
    #: (`occluded` / `lost`): the leader is deliberately *not* put here (T20).
    cluster: Optional["ClusterInfo"]
    #: Its position in the score-sorted list. 0 means the leader was taken; -1 = not in this frame.
    rank: int
    #: `first` — nothing was held yet, so the leader is taken and becomes the held object.
    #: `keep`  — the leader *is* the held object.
    #: `hold`  — a different cluster leads, but not for long enough yet; the held object is kept.
    #: `switch` — a different cluster has led `target_confirm_frames` frames running, each at
    #: `target_switch_min_score` or above; it takes over under a new id.
    #: `occluded` — the held object is not in this frame at all and no switch is due. Until T20 this
    #: was `unobserved` and handed the frame to the leader; the held object is now kept, with its
    #: last observed geometry, and the switch count keeps running (T14 seq 12: the leader was a
    #: crate at score 0.015 because the apple was behind the fingers).
    #: `lost` — as `occluded`, but for more than `target_lost_frames` frames: no target at all.
    mode: str
    #: How many frames running the current challenger has led. 0 whenever the leader is held.
    streak: int
    #: `target_confirm_frames`, so a reader can see `streak` against its threshold.
    frames: int
    #: Observation state of the manipulated object after this decision.
    state: str = VISIBLE
    #: Its id after this decision.
    manipulated_id: Optional[int] = None
    #: Consecutive unobserved frames after this decision (0 when visible).
    age_frames: int = 0
    #: T26: why the held object is unobserved (`unobserved` | `inadmissible_match`), or
    #: `no_admissible` for a frame with nothing held and nothing admissible. None otherwise.
    reason: Optional[str] = None
    #: T26: admissibility of `cluster` (or of the inadmissible match on `inadmissible_match`).
    admissibility: Optional[Admissibility] = None
    #: T26: how many of this frame's clusters were admissible (None when `UNCHECKED`).
    n_admissible: Optional[int] = None
    #: T26: whether a challenger was barred from counting because the grasp is in progress.
    frozen: bool = False
    #: T30 F3: the association test of the cluster nearest the held object (see
    #: `ManipulatedIdentity.association`). Rejected ⇔ `reason == "association_rejected"`.
    association: Optional[Association] = None
    #: T32b: the held object was hidden by the hand this frame (unobserved or `subset_kept_anchor`
    #: with a hand sphere within `clustering.hand_occlusion_reach`), so a challenger did not count.
    hand_occluded: bool = False
    #: T32b: nearest hand sphere surface ↔ the standing geometry (m), or None without a hand.
    hand_gap_m: Optional[float] = None


class TargetConfirm:
    """Hysteresis over *which* object is the target, and the identity of that object (T5e, T20).

    `ground_target` is frame-independent by design and stays that way: it ranks this frame's
    clusters and knows nothing about the last one. This object holds the only cross-frame state —
    the centroid of the object currently being called the target, its id, its last observed geometry
    and whether it is currently seen — and is owned by whoever owns the episode (`AG3S` holds one
    and clears it in `reset()`).

    **Why hysteresis and not a latch.** `GraspLatch._Confirm` locks a name once and never changes it
    until `release()`; that is right for "which object is in the hand", and wrong here, because
    the target legitimately changes — on ep1807 it is the apple, then the crate it goes into. What is
    not legitimate is changing on a single frame. So the rule is the latch's counting rule without
    the lock: a challenger must lead `target_confirm_frames` frames in a row — and, since T20, score at
    least `target_switch_min_score` on each of them.

    **Unobserved is not a new object (T20).** When the held object produces no cluster, the frame is
    `occluded`: the held object is returned with its *last observed* geometry, never the leader.
    The switch count keeps running, so a challenger that earns the switch still takes over on
    schedule; after `target_lost_frames` unobserved frames the object is `lost` and the frame has no
    target, with that as the stated reason. Re-appearing within tolerance of the last centroid makes
    it `visible` again under the same id.

    Three further reasons this is not `_Confirm` reused:
      - `_Confirm` gates each frame on `score >= score_ratio * runner_up`, which is a score-based
        rejection — the exact thing T5e removes. An ambiguous frame would not count, so under a
        steady 1.1x margin the target would never switch at all.
      - it compares *names*, and grounding has no names: `TargetGeometry.id` is this frame's cluster
        index and means nothing across frames. Identity here is centroid proximity, same rule and
        same tolerance as `CentroidIdentity`; the id is assigned here.
      - `benchmark/ag3s` does not import `benchmark/trajopt`; the dependency runs the other way
        (`trajopt/safe_policy.py` imports the pipeline). Reaching up would invert it.

    **The held centroid follows its object.** A grasped apple travels, so a centroid pinned to where
    it was first seen stops matching it within a few frames. `CentroidIdentity` deliberately does not
    update its anchors, but it is feeding a permanent lock, where a drifting anchor can never be
    undone; here the decision is remade every frame, so tracking the object is the safer error.
    While the object is unobserved the centroid stays where it was last seen — a pushed object is
    the known limit of that (T20.task "되돌아올 지점").

    **Hidden by the hand does not count toward a switch (T32b S3).** The count keeps running while
    the object is unobserved — unless the caller's hand spheres (`note_hand`) say a palm or finger
    is within `clustering.hand_occlusion_reach` of it, and no nearer the challenger. Then the
    object is hidden by the hand, not gone, and a challenger's frame is a gap (as while frozen).
    Once the hand is away — or has gone to the challenger — the count runs again, so a real change
    of object still switches on the T5e schedule.

    **A released grasp re-splits its anchor (T32b S2).** While frozen the geometry follows the
    observation unsplit; on the first call after the freeze is released that geometry goes through
    the T31b attention split once (`split_anchor_on_release`), so the neighbour a failed grasp
    left merged with the apple does not become the anchor.
    """

    def __init__(
        self,
        config: ClusteringConfig | None = None,
        *,
        frames: Optional[int] = None,
        tolerance: Optional[float] = None,
        switch_min_score: Optional[float] = None,
        lost_frames: Any = _FROM_CONFIG,
        max_opening: Any = _FROM_CONFIG,
        overlap_distance: Optional[float] = None,
        overlap_fraction: Optional[float] = None,
        destination_points: Optional[np.ndarray] = None,
        extent_ratio: Any = _FROM_CONFIG,
        anchor_coverage: Any = _FROM_CONFIG,
        anchor_distance: Optional[float] = None,
        hand_reach: Any = _FROM_CONFIG,
        split_on_release: Optional[bool] = None,
    ):
        cfg = config or ClusteringConfig()
        self.frames = int(cfg.target_confirm_frames if frames is None else frames)
        self.tolerance = float(cfg.target_identity_tolerance if tolerance is None else tolerance)
        self.switch_min_score = float(
            cfg.target_switch_min_score if switch_min_score is None else switch_min_score)
        # None is a legal value (never lost), so "not given" needs its own sentinel.
        lost = cfg.target_lost_frames if lost_frames is _FROM_CONFIG else lost_frames
        self.lost_frames: Optional[int] = None if lost is None else int(lost)
        # --- admissibility (T26) ---
        # Not given: `clustering.gripper_max_opening` if set, otherwise UNCHECKED (standalone use).
        # `None` given explicitly = "no one knows the opening" = nothing is graspable (fail-closed).
        if max_opening is _FROM_CONFIG:
            max_opening = (UNCHECKED if cfg.gripper_max_opening is None
                           else float(cfg.gripper_max_opening))
        if max_opening is not UNCHECKED and max_opening is not None:
            max_opening = float(max_opening)
        self.max_opening: Any = max_opening
        self.overlap_distance = float(
            overlap_distance if overlap_distance is not None
            else (cfg.destination_overlap_distance if cfg.destination_overlap_distance is not None
                  else cfg.eps))
        self.overlap_fraction = float(
            cfg.destination_overlap_fraction if overlap_fraction is None else overlap_fraction)
        # --- pre-grasp association (T30 F3) ---
        #: None = the test is off. Runs only when admissibility is checked and nothing is frozen.
        self.extent_ratio: Optional[float] = (
            None if extent_ratio is _FROM_CONFIG and cfg.manipulated_extent_ratio is None
            else float(cfg.manipulated_extent_ratio) if extent_ratio is _FROM_CONFIG
            else None if extent_ratio is None else float(extent_ratio))
        self._association: Optional[Association] = None
        # --- pre-grasp anchor (T30 F3b) ---
        #: Minimum fraction of the anchor geometry's points an observation must cover (within
        #: `anchor_distance`) to *replace* it before a grasp. None = off (every accepted observation
        #: replaces it, the T26 behaviour).
        self.anchor_coverage: Optional[float] = (
            (None if cfg.manipulated_anchor_coverage is None
             else float(cfg.manipulated_anchor_coverage)) if anchor_coverage is _FROM_CONFIG
            else None if anchor_coverage is None else float(anchor_coverage))
        #: Same-surface distance for that coverage. Default: the destination-overlap distance (one
        #: cloud voxel inside `AG3S`) — "the same surface at the resolution it was observed at".
        self.anchor_distance = float(self.overlap_distance if anchor_distance is None
                                     else anchor_distance)
        #: The **anchor**: the geometry adopted at `first`/`switch` (or last observed during a grasp,
        #: once the grasp ends), its extents and frame. Coverage is always measured against it, so
        #: a blob eroding a few percent per frame cannot walk the geometry off the object.
        self._anchor_geom: Optional[TargetGeometry] = None
        self._anchor_extents: Optional[np.ndarray] = None
        self._anchor_source: Optional[str] = None
        self._anchor_frame = -1
        #: Where the current geometry came from (`first|switch|refined|observed`) and when.
        self._geometry_source: Optional[str] = None
        self._geometry_frame = -1
        #: Consecutive frames the anchor was kept over a subset observation.
        self._anchor_kept = 0
        #: This frame kept the anchor (`observe` must not replace the geometry).
        self._keep_anchor = False
        #: T31b: this frame's split record (`note_split`), None until grounding reports one.
        self._split: Optional[dict] = None
        # --- hand occlusion (T32b S3) ---
        #: Hand-sphere surface ↔ standing geometry distance under which an unobserved (or subset)
        #: object counts as hidden by the hand. None = off.
        hr = cfg.hand_occlusion_reach if hand_reach is _FROM_CONFIG else hand_reach
        self.hand_reach: Optional[float] = None if hr is None else float(hr)
        #: The hand spheres for the *next* grounding call (`note_hand`); consumed by it.
        self._hand: Optional[HandSpheres] = None
        #: This frame's hand record (None when no hand was given).
        self._hand_record: Optional[dict] = None
        # --- anchor split on release (T32b S2) ---
        self.split_on_release = bool(
            cfg.split_anchor_on_release if split_on_release is None else split_on_release)
        #: The freeze was released and the next grounding call must split the standing geometry.
        self._release_pending = False
        #: Per-point attention of `_geometry` (the frame it was observed in), for that split.
        self._geometry_attention: Optional[np.ndarray] = None
        self._hand_occluded, self._hand_gap = False, None
        self._frame_hand: Optional[HandSpheres] = None
        #: The release split's record while the anchor it produced stands (anchor record).
        self._release_split: Optional[dict] = None
        #: This call's release event (split record), shown in `ManipulatedIdentity.split`.
        self._release_event: Optional[dict] = None
        #: The destination (T26): registered from perception, or injected. Same count and minimum
        #: score as a switch — the destination has to earn its role the way a target does.
        self.destinations = DestinationRegistry(
            frames=self.frames, min_score=self.switch_min_score, tolerance=self.tolerance)
        #: True while a grasp is in progress (closing … held): a challenger does not count, so the
        #: manipulated object cannot change between the closing command and the release (T26 §4).
        self.frozen = False
        self._held: Optional[np.ndarray] = None
        self._challenger: Optional[np.ndarray] = None
        self._streak = 0
        # --- identity (T20) ---
        self._id: Optional[int] = None
        self._next_id = 0
        self._switched_from: Optional[int] = None
        self._geometry: Optional[TargetGeometry] = None
        self._admissibility: Optional[Admissibility] = None
        self._age = 0
        self._last_seen_frame = -1
        self._last_seen_time: Optional[float] = None
        #: Index of the current grounding call since `reset()`; -1 before the first.
        self._frame = -1
        #: The last decision, for logging. `None` before the first frame with a cluster.
        self.last: Optional[ConfirmDecision] = None
        #: This frame's per-cluster admissibility, score order (None when UNCHECKED / no clusters).
        self.last_admissibility: Optional[list[Admissibility]] = None
        if destination_points is not None:
            self.destinations.reset(destination_points)

    # --- queries ---------------------------------------------------------------------------
    @property
    def checked(self) -> bool:
        """True when admissibility is evaluated (always inside `AG3S`)."""
        return self.max_opening is not UNCHECKED

    @property
    def destination(self) -> Optional[DestinationGeometry]:
        """The registered or injected destination, or None (T26)."""
        return self.destinations.current

    @property
    def held_centroid(self) -> Optional[np.ndarray]:
        """Centroid of the object currently being called the target, or None."""
        return None if self._held is None else self._held.copy()

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def state(self) -> Optional[str]:
        """Observation state of the manipulated object, or None when nothing is held."""
        if self._held is None:
            return None
        if self._age == 0:
            return VISIBLE
        if self.lost_frames is not None and self._age > self.lost_frames:
            return LOST
        return OCCLUDED

    @property
    def last_geometry(self) -> Optional[TargetGeometry]:
        """The manipulated object's last observed geometry (see `ManipulatedIdentity.geometry`)."""
        return self._geometry

    @property
    def manipulated(self) -> Optional[ManipulatedIdentity]:
        """A snapshot of the manipulated object after the latest grounding call, or None."""
        if self._held is None or self._id is None:
            return None
        return ManipulatedIdentity(
            id=int(self._id),
            state=str(self.state),
            geometry=self._geometry,
            centroid=self._held.copy(),
            age_frames=int(self._age),
            last_seen_frame=int(self._last_seen_frame),
            last_seen_time=self._last_seen_time,
            switched_from=self._switched_from,
            frame=int(self._frame),
            admissibility=self._admissibility,
            association=self._association,
            anchor=self._anchor_record(),
            split=self._split_snapshot(),
            hand=None if self._hand_record is None else dict(self._hand_record),
        )

    def reset(self, destination_points: Optional[np.ndarray] = None) -> None:
        """Episode boundary. The next frame's leader is taken with no argument, under a fresh id.

        `destination_points` (T26): the caller knows the destination — injected, it takes
        precedence over anything perception would register and is never replaced by it.
        """
        self._held = self._challenger = None
        self._streak = 0
        self._id = None
        self._next_id = 0
        self._switched_from = None
        self._geometry = None
        self._admissibility = None
        self._age = 0
        self._last_seen_frame = -1
        self._last_seen_time = None
        self._frame = -1
        self.last = None
        self.last_admissibility = None
        self._association = None
        self._anchor_geom, self._anchor_extents = None, None
        self._anchor_source, self._anchor_frame, self._anchor_kept = None, -1, 0
        self._geometry_source, self._geometry_frame = None, -1
        self._keep_anchor = False
        self._split = None
        self._hand, self._hand_record, self._frame_hand = None, None, None
        self._release_pending, self._geometry_attention = False, None
        self._release_split, self._release_event = None, None
        self.frozen = False
        self.destinations.reset(destination_points)

    def inject_destination(self, points: np.ndarray) -> None:
        """The caller names the destination mid-episode (source `injected`, sticky until reset)."""
        self.destinations.inject(points, frame=self._frame)

    def freeze(self, on: bool) -> None:
        """Bar challengers while a grasp is in progress (closing → held → release). T26 §4.

        The release (frozen → not) arms the one-off anchor split of T32b S2 for the next grounding
        call; freezing again before that call disarms it (the next release arms it again).
        """
        on = bool(on)
        if self.frozen and not on:
            self._release_pending = self.split_on_release
        elif on:
            self._release_pending = False
        self.frozen = on
        if self.frozen:
            self._challenger, self._streak = None, 0

    def note_hand(self, hand: Optional[HandSpheres]) -> None:
        """The hands' collision spheres at this frame's robot state, in the cloud frame (T32b S3).

        Given before each grounding call (as `freeze`) and consumed by it; None (or never calling
        this) = no hand information, and then nothing is suppressed. Build it with
        `stages.admissibility.hand_spheres(robot_model, robot_state)`.
        """
        if hand is not None and not isinstance(hand, HandSpheres):
            raise TypeError(f"note_hand expects HandSpheres or None, got {type(hand).__name__}")
        self._hand = hand if hand is not None and len(hand) else None

    def split_anchor_on_release(self, config: ClusteringConfig, *,
                                min_radius: float = 0.005) -> Optional[dict]:
        """T32b S2: on the first grounding call after a freeze is released, split the standing
        geometry (the anchor the grasp attempt left) once with the T31b attention split.

        Uses the attention the geometry was observed with (`observe(..., attention=)`). When the
        split is taken and its target component is admissible, that component becomes the geometry
        and the anchor (source `release_split`, `attention_split = 1` so an unsplit merge cannot
        replace it later — T31b `unsplit_kept_anchor`). Returns this call's record (also in
        `ManipulatedIdentity.split["release"]`), or None when no release was pending.
        """
        self._release_event = None
        if not self._release_pending:
            return None
        self._release_pending = False
        geom, att = self._geometry, self._geometry_attention
        event: dict[str, Any] = {"frame": int(self._frame + 1), "replaced": False,
                                 "skipped": None, "decision": None}
        self._release_event = event
        opening, skipped = self.split_opening(config)
        if opening is None:
            event["skipped"] = skipped
            return event
        if geom is None or self._id is None:
            event["skipped"] = "no_geometry"
            return event
        pts = np.asarray(geom.points, np.float64).reshape(-1, 3)
        if att is None or len(att) != len(pts):
            event["skipped"] = "no_attention"
            return event
        dec = split_by_attention(
            pts, np.asarray(att, np.float64), np.arange(len(pts)), max_opening=opening,
            radius=float(config.attention_split_radius), min_points=int(config.min_points),
            min_ratio=float(config.attention_split_ratio))
        if dec is None:
            event["skipped"] = "not_triggered"
            return event
        event["decision"] = dec.record()
        if not dec.accepted:
            return event
        idx = np.sort(dec.target.indices)
        sub = pts[idx]
        ext = principal_extents(sub)
        adm = None
        if self.checked:
            adm = assess(sub, extents=ext, max_opening_m=self.max_opening,
                         destination=self.destinations.current,
                         overlap_distance=self.overlap_distance,
                         overlap_fraction=self.overlap_fraction)
            if not adm.admissible:
                event["skipped"] = "component_inadmissible"
                return event
        metrics = dict(geom.metrics)
        metrics.update({"attention_split": 1.0, "attention_split_target": 1.0,
                        "attention_split_parent_n_points": float(len(pts)),
                        "attention_split_on_release": 1.0, "point_count": float(len(sub))})
        new = dataclasses.replace(
            geom, points=sub, point_indices=np.zeros(0, np.int64), centroid=sub.mean(axis=0),
            bounding_geometry=fit_sphere(sub, min_radius=min_radius, semantic_role="target"),
            metrics=metrics)
        self._geometry, self._geometry_attention = new, np.asarray(att, np.float64)[idx]
        self._admissibility = adm
        self._held = np.asarray(new.centroid, np.float64).reshape(3)
        self._anchor_geom, self._anchor_extents = new, ext
        self._anchor_source = self._geometry_source = "release_split"
        self._anchor_frame = self._geometry_frame = int(self._frame + 1)
        self._anchor_kept = 0
        event["replaced"] = True
        event["n_points"] = [int(len(pts)), int(len(sub))]
        self._release_split = event
        return event

    def split_opening(self, config: ClusteringConfig) -> tuple[Optional[float], Optional[str]]:
        """The max opening the T31b split tests against this frame, or None with the reason.

        No split while a grasp is in progress (`frozen`: the fingers cut the held object into
        pieces and it moves with the hand), with the split off in the config, or when the opening
        is unknown (then nothing is graspable anyway). `UNCHECKED` falls back on
        `clustering.gripper_max_opening`, as the rest of this class does.
        """
        if config.attention_split_radius is None:
            return None, "off"
        if self.frozen:
            return None, "frozen"
        if self.checked:
            return (None, "max_opening_unknown") if self.max_opening is None else (
                float(self.max_opening), None)
        if config.gripper_max_opening is None:
            return None, "max_opening_unknown"
        return float(config.gripper_max_opening), None

    def note_split(self, record: Optional[dict]) -> None:
        """`ground_target` reports this frame's split decisions (T31b); see `ManipulatedIdentity.split`."""
        self._split = record

    # --- one frame -------------------------------------------------------------------------
    def select(self, clusters: "Sequence[ClusterInfo]",
               points: Optional[np.ndarray] = None) -> ConfirmDecision:
        """Pick this frame's target from `clusters`, which must be sorted by descending score.

        Returns a decision whose `cluster` is None when the manipulated object is `occluded` or
        `lost` — the caller decides what stands in for it (`ground_target`: the last observed
        geometry, or no target) — or when nothing is held and nothing is admissible
        (`no_admissible`). It never substitutes the leader for an unobserved object, and never
        adopts an inadmissible cluster (T26).

        `points`: the cloud the clusters' `point_indices` index. Required when admissibility is
        checked (the destination overlap and the extents are measured on it).
        """
        if not clusters:
            raise ValueError("TargetConfirm.select needs at least one cluster")
        self._frame += 1
        self._association = None
        self._keep_anchor = False
        self._hand_occluded, self._hand_gap = False, self._take_hand()
        adm = self._assess_frame(clusters, points)
        candidates = (list(range(len(clusters))) if adm is None
                      else [i for i, a in enumerate(adm) if a.admissible])

        if self._held is None:
            self._challenger, self._streak = None, 0
            if not candidates:
                return self._record(None, -1, NO_ADMISSIBLE, reason=NO_ADMISSIBLE)
            first = candidates[0]
            self._adopt(clusters[first], switched_from=None, adm=_at(adm, first))
            return self._record(clusters[first], first, "first")

        held_rank = self._nearest(clusters, self._held)
        reason = None
        inadmissible = None
        if held_rank is not None and adm is not None and not adm[held_rank].admissible:
            # The cluster at the held object's place is not something the hand can hold (or it is
            # the destination): the apple merged with the fingers or the crate. Its geometry must
            # not stand in for the apple's — the object counts as unobserved and keeps its last
            # admissible geometry (T26 `inadmissible_match`).
            reason, inadmissible, held_rank = INADMISSIBLE_MATCH, adm[held_rank], None
        elif held_rank is not None:
            # Before a grasp the object has not moved, so the nearest cluster is it only if it sits
            # within the object's own radius and has its size (T30 F3). A rejected cluster is not
            # the held object this frame; it may still be a challenger under the T26 switch rule.
            self._association = self._associate(clusters[held_rank], points)
            if self._association is not None and not self._association.accepted:
                reason, held_rank = ASSOCIATION_REJECTED, None
            elif self._association is not None and self._association.subset:
                # It is the object, but it shows only part of the anchor (the hand hides the rest):
                # the anchor stays the exclusion geometry, the object counts as seen (T30 F3b).
                reason, self._keep_anchor = SUBSET_KEPT_ANCHOR, True
            elif self._association is not None and self._unsplit_over_split_anchor(
                    clusters[held_rank]):
                # The anchor is the apple split out of the apple+orange blob; this frame's cluster
                # at its place is the blob again, which the split could not separate (the gap
                # closed under 15 mm, or attention did not tell them apart). It shows the anchor,
                # so it is the object — but it must not replace it and bring the neighbour back
                # into the exclusion (T31b).
                reason, self._keep_anchor = UNSPLIT_KEPT_ANCHOR, True
        lead = candidates[0] if candidates else None
        # T32b S3: the held object is not seen whole this frame (unobserved for any reason, or only
        # a part of it), a hand sphere is within reach of it, and the hand is not nearer the
        # challenger than the held object: the hand is hiding the held object. That is not a new
        # object, so the challenger does not count this frame (T20). A hand that has gone to the
        # challenger is the policy changing object — that frame counts (E3 ep1800: the hand 31 mm
        # from the apple but at the banana it then grasped). A `keep` frame is never suppressed —
        # there the held object leads and the count restarts anyway.
        self._hand_occluded = self._hidden_by_hand(
            hidden=held_rank is None or reason == SUBSET_KEPT_ANCHOR,
            challenger=None if lead is None or lead == held_rank else clusters[lead],
            points=points)

        if held_rank is not None and lead == held_rank:
            # The held object is the best admissible candidate. Track it and forget any challenger:
            # leading again after a gap has to start the count over.
            self._seen(clusters[held_rank], _at(adm, held_rank))
            self._challenger, self._streak = None, 0
            return self._record(clusters[held_rank], held_rank, "keep", reason=reason)

        challenger = clusters[lead] if lead is not None else None
        if (challenger is not None and not self.frozen and not self._hand_occluded
                and float(challenger.target_score) >= self.switch_min_score):
            if self._challenger is not None and self._within(challenger.centroid, self._challenger):
                self._streak += 1
            else:
                self._streak = 1
            self._challenger = np.asarray(challenger.centroid, np.float64).reshape(3)
        else:
            # A challenger under the minimum does not count, and it breaks the run: "N frames
            # running at or above the minimum" is the rule, so a sub-minimum frame is a gap. No
            # admissible challenger at all is a gap too, and so is a frozen (grasping) frame and a
            # frame where the hand hides the held object (T32b).
            self._challenger, self._streak = None, 0

        if self._streak >= self.frames:
            self._adopt(challenger, switched_from=self._id, adm=_at(adm, lead))
            self._challenger, self._streak = None, 0
            return self._record(challenger, lead, "switch")

        if held_rank is None:
            # The held object produced no (admissible) cluster this frame. It is not replaced by
            # the leader — being hidden by the hand is not being a different object. The count is
            # *not* reset: if it stays gone, a qualifying challenger still takes over on schedule.
            self._age += 1
            state = self.state
            return self._record(None, -1, LOST if state == LOST else OCCLUDED,
                                reason=reason or "unobserved", admissibility=inadmissible)

        held = clusters[held_rank]
        self._seen(held, _at(adm, held_rank))
        return self._record(held, held_rank, "hold", reason=reason)

    def unseen(self) -> Optional[str]:
        """A grounding call that produced no clusters at all (NO_SEED, NO_CLUSTER, ...).

        The manipulated object was not observed either, so it ages; nothing else moves — the switch
        count is untouched, as it was before T20 when such a frame never reached `select`. Returns
        the resulting state (None when nothing is held).
        """
        self._frame += 1
        self.last_admissibility = None
        self._hand_occluded, self._hand_gap = False, self._take_hand()
        if self._held is None:
            return None
        self._age += 1
        return self.state

    def observe(self, target: TargetGeometry, attention: Optional[np.ndarray] = None) -> None:
        """Record this frame's geometry of the manipulated object (called by `ground_target` right
        after a `first`/`keep`/`hold`/`switch` decision). This is what `occluded` falls back on.

        `attention`: per-point attention of `target.points` in this frame (T32b S2 — what the
        release split of this geometry reads). None = unknown (the release split is skipped)."""
        self._last_seen_time = float(target.timestamp)
        if self._keep_anchor:
            return  # T30 F3b: a subset observation never replaces the anchor
        self._geometry = target
        att = None if attention is None else np.asarray(attention, np.float64).reshape(-1)
        self._geometry_attention = (
            att if att is not None and len(att) == len(np.asarray(target.points).reshape(-1, 3))
            else None)
        decision = self.last.mode if self.last is not None else None
        source = ("observed" if self.frozen
                  else decision if decision in ("first", "switch") else "refined")
        self._geometry_source, self._geometry_frame = source, self._frame
        self._anchor_kept = 0
        if source != "refined" or self._anchor_geom is None:
            # A new anchor: a new object, or the object as the hand last had it (grasp in progress
            # — after the release this is where it was put down). A refinement keeps the anchor.
            self._release_split = None
            self._anchor_geom = target
            self._anchor_extents = (
                np.asarray(self._admissibility.extents_m, np.float64)
                if self._admissibility is not None and len(self._admissibility.extents_m) == 3
                else principal_extents(target.points))
            self._anchor_source, self._anchor_frame = source, self._frame

    def observe_seen(self, timestamp: float) -> None:
        """A `subset_kept_anchor` frame: the object was seen at `timestamp`, the geometry stays."""
        self._last_seen_time = float(timestamp)

    @property
    def anchor_active(self) -> bool:
        """True when the pre-grasp anchor rule runs this frame (T30 F3b)."""
        return (not self.frozen and self.checked and self.anchor_coverage is not None
                and self.extent_ratio is not None)

    # --- internals -------------------------------------------------------------------------
    def _take_hand(self) -> Optional[float]:
        """Consume this call's hand spheres (T32b): the record, and the gap to the standing
        geometry (the last observed / kept anchor — what stands in while the object is hidden)."""
        hand, self._hand = self._hand, None
        self._frame_hand = hand
        if hand is None:
            self._hand_record = None
            return None
        geom = self._geometry
        gap = None if geom is None else hand.gap(np.asarray(geom.points, np.float64))
        self._hand_record = {
            "available": True,
            "gap_mm": None if gap is None else round(gap * 1000.0, 1),
            "reach_mm": None if self.hand_reach is None else round(self.hand_reach * 1000.0, 1),
            "occluded_by_hand": False,
            "challenger_gap_mm": None,
            "n_spheres": int(len(hand)),
            "links": sorted(set(hand.links)),
        }
        return gap

    def _hidden_by_hand(self, *, hidden: bool, challenger: Optional["ClusterInfo"],
                        points) -> bool:
        """T32b S3 (see `select`). Records the challenger-side gap in the hand record."""
        hand, gap = self._frame_hand, self._hand_gap
        if (not hidden or self.frozen or self.hand_reach is None or hand is None or gap is None
                or gap > self.hand_reach):
            return False
        out = True
        if challenger is not None and points is not None:
            pts = np.asarray(points, np.float64).reshape(-1, 3)[
                np.asarray(challenger.point_indices, np.int64)]
            cgap = hand.gap(pts)
            if cgap is not None:
                self._hand_record["challenger_gap_mm"] = round(cgap * 1000.0, 1)
                out = gap <= cgap
        self._hand_record["occluded_by_hand"] = out
        return out

    def _associate(self, cluster: "ClusterInfo", points) -> Optional[Association]:
        """T30 F3 test of `cluster` against the last admissible geometry, or None when it does not
        apply: a grasp in progress (the object moves with the hand then), admissibility not
        checked (`UNCHECKED` has no admissible geometry), the test off, or no geometry yet."""
        geom, ref_adm = self._geometry, self._admissibility
        if (self.frozen or not self.checked or self.extent_ratio is None or geom is None
                or len(np.asarray(geom.points).reshape(-1, 3)) < 2):
            return None
        ref_ext = (np.asarray(ref_adm.extents_m, np.float64)
                   if ref_adm is not None and len(ref_adm.extents_m) == 3
                   else principal_extents(geom.points))
        cluster_pts = None
        if points is not None:
            cluster_pts = np.asarray(points, np.float64).reshape(-1, 3)[
                np.asarray(cluster.point_indices, np.int64)]
        ext = cluster.principal_extents
        if ext is None:
            if cluster_pts is None:
                return None
            ext = principal_extents(cluster_pts)
        assoc = associate(
            cluster.centroid, ext,
            reference_centroid=np.asarray(geom.centroid, np.float64),
            reference_radius=float(np.asarray(geom.bounding_geometry.dimensions, np.float64)[0]),
            reference_extents=ref_ext, max_ratio=self.extent_ratio)
        if assoc.accepted and self.anchor_active and cluster_pts is not None:
            anchor = self._anchor_geom if self._anchor_geom is not None else geom
            assoc = anchor_cover(assoc, np.asarray(anchor.points, np.float64), cluster_pts,
                                 distance=self.anchor_distance,
                                 min_coverage=float(self.anchor_coverage))
        return assoc

    def _unsplit_over_split_anchor(self, cluster: "ClusterInfo") -> bool:
        """T31b: the anchor came from an attention split and `cluster` is a triggered cluster the
        split left whole (it may hold the anchor's neighbour again)."""
        anchor = self._anchor_geom
        return bool(self.anchor_active and anchor is not None
                    and anchor.metrics.get("attention_split", 0.0)
                    and cluster.split is not None and cluster.split.get("role") == "unsplit")

    def _anchor_record(self) -> Optional[dict[str, Any]]:
        """The pre-grasp anchor and the current geometry, per frame (T30 F3b record)."""
        anchor = self._anchor_geom
        if anchor is None or self._id is None:
            return None
        ext = (self._anchor_extents if self._anchor_extents is not None
               else principal_extents(anchor.points))
        return {
            "manipulated_id": int(self._id),
            "active": bool(self.anchor_active),
            "source": self._anchor_source,
            "since_frame": int(self._anchor_frame),
            "kept_frames": int(self._anchor_kept),
            "extents_mm": [round(float(v) * 1000.0, 1) for v in ext],
            "radius_mm": round(float(np.asarray(anchor.bounding_geometry.dimensions)[0]) * 1000.0,
                               1),
            "n_points": int(len(np.asarray(anchor.points).reshape(-1, 3))),
            "centroid": [float(v) for v in np.asarray(anchor.centroid, np.float64).reshape(3)],
            "geometry_source": self._geometry_source,
            "geometry_frame": int(self._geometry_frame),
            # T31b: the anchor is a component of a cluster split by attention (the apple out of
            # the apple+orange blob), not the blob.
            "from_split": bool(anchor.metrics.get("attention_split", 0.0)),
            # T32b S2: this anchor is the target component of the release split (its record).
            "release_split": None if self._release_split is None else {
                k: v for k, v in self._release_split.items() if k != "decision"},
        }

    def _split_snapshot(self) -> Optional[dict[str, Any]]:
        """This frame's split record plus whether the manipulated geometry came from a split."""
        if self._split is None and self._release_event is None:
            return None
        geom = self._geometry
        out = dict(self._split) if self._split is not None else {"ran": False, "skipped": None}
        out["manipulated_from_split"] = bool(
            geom is not None and geom.metrics.get("attention_split", 0.0))
        if self._release_event is not None:
            out["release"] = dict(self._release_event)  # T32b S2, the call it happened on
        return out

    def _assess_frame(self, clusters, points) -> Optional[list[Admissibility]]:
        """Per-cluster admissibility, destination registration first (T26). None when UNCHECKED."""
        if not self.checked:
            self.last_admissibility = None
            return None
        if points is None:
            raise ValueError(
                "TargetConfirm.select needs the cloud `points` when admissibility is checked — the "
                "extents and the destination overlap are measured on the cluster's points")
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        cluster_points = [pts[np.asarray(c.point_indices, np.int64)] for c in clusters]
        extents = [c.principal_extents if c.principal_extents is not None
                   else principal_extents(p) for c, p in zip(clusters, cluster_points)]
        grasp = [graspable(e, self.max_opening) for e in extents]
        self.destinations.observe(clusters, cluster_points, grasp, frame=self._frame)
        dest = self.destinations.current
        out = [assess(p, extents=e, max_opening_m=self.max_opening, destination=dest,
                      overlap_distance=self.overlap_distance,
                      overlap_fraction=self.overlap_fraction)
               for p, e in zip(cluster_points, extents)]
        self.last_admissibility = out
        return out

    def _adopt(self, cluster: "ClusterInfo", *, switched_from: Optional[int],
               adm: Optional[Admissibility] = None) -> None:
        self._keep_anchor = False  # a new object has no anchor to keep
        self._id = self._next_id
        self._next_id += 1
        self._switched_from = switched_from
        self._geometry = None  # the previous object's shape must never stand in for this one
        self._last_seen_time = None
        self._seen(cluster, adm)

    def _seen(self, cluster: "ClusterInfo", adm: Optional[Admissibility] = None) -> None:
        self._age = 0
        self._last_seen_frame = self._frame
        if self._keep_anchor:
            # Seen, but the anchor stays: its centroid and its admissibility (they describe the
            # geometry the exclusion gate will read), not this partial cluster's (T30 F3b).
            self._anchor_kept += 1
            return
        self._held = np.asarray(cluster.centroid, np.float64).reshape(3)
        self._admissibility = adm

    def _record(self, cluster: Optional["ClusterInfo"], rank: int, mode: str, *,
                reason: Optional[str] = None,
                admissibility: Optional[Admissibility] = None) -> ConfirmDecision:
        adm = self.last_admissibility
        if admissibility is None and adm is not None and cluster is not None and rank >= 0:
            admissibility = adm[rank]
        self.last = ConfirmDecision(
            cluster, int(rank), mode, self._streak, self.frames,
            state="none" if self.state is None else str(self.state),
            manipulated_id=self._id, age_frames=int(self._age), reason=reason, admissibility=admissibility,
            n_admissible=None if adm is None else sum(1 for a in adm if a.admissible),
            frozen=bool(self.frozen), association=self._association,
            hand_occluded=bool(self._hand_occluded), hand_gap_m=self._hand_gap)
        return self.last

    def _within(self, a, b) -> bool:
        return float(np.linalg.norm(np.asarray(a, np.float64).reshape(3)
                                    - np.asarray(b, np.float64).reshape(3))) <= self.tolerance

    def _nearest(self, clusters, centroid) -> Optional[int]:
        """Index of the cluster closest to `centroid` within `tolerance`, or None.

        Nearest rather than first-within-tolerance — with two candidates both inside 6 cm, the
        closer one is the better claim to being the same object, and the ordering of `clusters` is
        by score, which has nothing to do with it.
        """
        best, best_d = None, float("inf")
        for i, c in enumerate(clusters):
            d = float(np.linalg.norm(np.asarray(c.centroid, np.float64).reshape(3) - centroid))
            if d <= self.tolerance and d < best_d:
                best, best_d = i, d
        return best


def _at(adm: Optional[list], i: Optional[int]) -> Optional[Admissibility]:
    return None if adm is None or i is None else adm[i]


# ------------------------------------------------------------------------------- scoring


def _score_cluster(
    cluster_id: int,
    indices: np.ndarray,
    points: np.ndarray,
    attention: np.ndarray,
    peak_point: np.ndarray,
    config: ClusteringConfig,
) -> ClusterInfo:
    pts = points[indices]
    att = attention[indices]
    centroid = pts.mean(axis=0)
    rms = rms_radius(pts)
    peak_distance = float(np.linalg.norm(centroid - peak_point))

    # Both terms are in (0, 1] and decay with a configured length scale, so the weights stay
    # interpretable and the threshold means the same thing across scenes.
    compactness = float(np.exp(-rms / config.compactness_scale))
    peak_proximity = float(np.exp(-peak_distance / config.peak_scale))
    # Geometric mean: a cluster must be both compact *and* near the peak. A sum would let a huge
    # blob centred on the peak score as well as the object actually being attended to.
    spatial_consistency = float(np.sqrt(compactness * peak_proximity))

    attention_score = float(att.mean())
    w_a, w_g = config.w_attention, config.w_geometry
    score = float((w_a * attention_score + w_g * spatial_consistency) / (w_a + w_g))

    return ClusterInfo(
        id=cluster_id,
        point_indices=indices,
        centroid=centroid,
        mean_attention=attention_score,
        max_attention=float(att.max()),
        attention_mass=float(att.sum()),
        point_count=int(indices.size),
        spatial_compactness=compactness,
        distance_from_attention_peak=peak_distance,
        target_score=score,
        rms_radius=rms,
        principal_extents=principal_extents(pts),
    )


# ----------------------------------------------------------------------------- the stage


def ground_target(
    attention_cloud: AttentionPointCloud,
    config: ClusteringConfig | None = None,
    *,
    seed_percentile: float = 95.0,
    seed_threshold: Optional[float] = None,
    exclude_mask: Optional[np.ndarray] = None,
    min_radius: float = 0.005,
    timestamp: Optional[float] = None,
    confirm: Optional["TargetConfirm"] = None,
) -> GroundingResult:
    """Find the object attention is pointing at, or report why there isn't one.

    Args:
        attention_cloud: every reconstructed point, with its attention value.
        config: clustering settings.
        seed_percentile / seed_threshold: from `AttentionConfig`; passed explicitly so this stage
            depends on one config section rather than the whole tree.
        exclude_mask: ``True`` where a point belongs to a support surface and must not participate in
            connectivity. Without it, growth floods across the table — see the module docstring.
        min_radius: floor for the bounding sphere of a very small cluster.
        confirm: optional cross-frame hysteresis (T5e). Without it the ranking of *this* frame
            decides alone, which keeps the function reproducible from a single frame and is what
            every offline caller gets. With it, the target only moves to a different object after
            that object has led `target_confirm_frames` frames in a row; the caller owns the object
            because the caller owns the episode boundary. With one, a frame where the held object
            produced no cluster returns its **last observed** geometry (`occluded`; empty
            `point_indices`, `metrics["manipulated_occluded"] = 1`) or, once `lost`, no target with
            `GroundingStatus.LOST` — never the leader (T20). A frame that ends with no cluster at
            all reports its usual status and only ages the manipulated object (`confirm.unseen()`).
            When `confirm` checks admissibility (T26; always inside `AG3S`), only admissible
            clusters compete, and a frame with nothing held and nothing admissible returns no
            target with `GroundingStatus.NO_ADMISSIBLE`.

    Note the excluded points are excluded from *connectivity and clustering only*. They remain in the
    cloud and go on to become `SUPPORT_SURFACE` candidates in stage 5; nothing is deleted here.
    """
    cfg = config or ClusteringConfig()
    ts = time.time() if timestamp is None else float(timestamp)
    if confirm is not None:
        confirm.note_split(None)  # T31b: until this frame reaches clustering, no split record
        # T32b S2: the first call after a grasp attempt's freeze is released splits its anchor once.
        confirm.split_anchor_on_release(cfg, min_radius=min_radius)
    points = attention_cloud.points
    attention = attention_cloud.attention
    n = len(attention_cloud)

    if n == 0:
        _unseen(confirm)
        return GroundingResult(None, GroundingStatus.NO_GEOMETRY)

    if float(attention.max()) - float(attention.min()) <= _EPS:
        # A flat map carries no information about which object is the target. Taking its "top 5%"
        # would return an arbitrary slice of the scene and dress it up as a grounded target.
        _unseen(confirm)
        return GroundingResult(None, GroundingStatus.NO_ATTENTION)

    # The peak comes from the *unclipped* signal. Percentile normalization deliberately saturates its
    # top percentile, which ties hundreds of points at 1.0 and makes `argmax` return whichever one
    # happens to be first in raster order — on the default fixture, a table point 12 cm from the
    # object. `AttentionPointCloud.peak_signal` keeps the two concerns apart.
    peak_index = int(np.argmax(attention_cloud.peak_signal))
    peak_point = points[peak_index]

    usable = np.ones(n, bool)
    if exclude_mask is not None:
        usable &= ~np.asarray(exclude_mask, bool).reshape(-1)

    seed_mask = np.zeros(n, bool)
    seed_mask[extract_seeds(attention, cfg, seed_percentile, seed_threshold)] = True
    seed_mask &= usable
    seeds = np.nonzero(seed_mask)[0]
    if seeds.size == 0:
        _unseen(confirm)
        return GroundingResult(None, GroundingStatus.NO_SEED, seed_indices=seeds,
                               attention_peak_index=peak_index)

    if not cfg.use_3d_connectivity:
        # Ablation: "2D attention only". The seed set *is* the target, with no geometric growth and
        # no separation of touching objects. This is the behaviour AG3S exists to improve on, kept
        # reachable so the improvement can be measured rather than asserted.
        groups = {0: seeds}
        member_points, member_attention = points, attention
    else:
        candidate_idx = np.nonzero(usable)[0]
        local_seed = np.searchsorted(candidate_idx, seeds)
        grown_local = grow_region(points[candidate_idx], local_seed, cfg.eps)
        grown = candidate_idx[grown_local]
        if grown.size == 0:
            _unseen(confirm)
            return GroundingResult(None, GroundingStatus.NO_CLUSTER, seed_indices=seeds,
                                   attention_peak_index=peak_index)

        if cfg.method == "dbscan":
            labels = dbscan(points[grown], cfg.eps, cfg.min_points)
        else:
            labels = connected_components_3d(points[grown], cfg.region_growing_radius or cfg.eps)

        groups = {}
        for label in np.unique(labels):
            if label < 0:  # DBSCAN noise
                continue
            members = grown[labels == label]
            if members.size >= cfg.min_points:
                groups[int(label)] = members
        member_points, member_attention = points, attention

    if not groups:
        _unseen(confirm)
        return GroundingResult(None, GroundingStatus.NO_CLUSTER, seed_indices=seeds,
                               attention_peak_index=peak_index)

    # T31b: a cluster that may be several objects is split by attention before anything is scored,
    # so the components compete — and are excluded or kept — as the separate objects they are.
    members, splits = _split_groups(
        [idx for _, idx in sorted(groups.items())], points, attention, cfg, confirm,
        enabled=cfg.use_3d_connectivity)

    clusters = [
        dataclasses.replace(
            _score_cluster(i, idx, member_points, member_attention, peak_point, cfg), split=info)
        for i, (idx, info) in enumerate(members)
    ]
    # Deterministic ordering: score descending, ties broken by the lowest member index.
    clusters.sort(key=lambda c: (-c.target_score, int(c.point_indices[0])))
    best = clusters[0]

    # The score gate is **off by default** since T5e: `target_score_threshold` defaults to 0.0 and the
    # comparison is `<`, so nothing is rejected and the ranking alone names the target. Measured
    # reason in `config.py` — on ep1807 this line rejected 35 of 51 chunks whose rank-1 identity was
    # correct. It is still here, and still first, because a caller that sets the threshold above 0
    # wants rejection *before* any of the state below moves: a frame we refuse to name must not count
    # toward switching the target either.
    if best.target_score < cfg.target_score_threshold:
        _unseen(confirm)
        return GroundingResult(
            None, GroundingStatus.LOW_SCORE, tuple(clusters), seeds, peak_index, best.target_score,
            splits=splits,
        )

    # Which candidate wins. Without a `confirm` this is rank 1 and the frame stands alone, which is
    # what every direct caller gets and what makes this function reproducible from one frame. With
    # one, a challenger has to lead `target_confirm_frames` frames running before the target moves.
    decision = confirm.select(clusters, points=points) if confirm is not None else None
    if decision is not None and decision.mode == NO_ADMISSIBLE:
        # Nothing held yet and no cluster may become the manipulated object (T26) — e.g. the crate
        # is all attention found. The leader is not named, whatever its score.
        return GroundingResult(None, GroundingStatus.NO_ADMISSIBLE, tuple(clusters), seeds,
                               peak_index, best.target_score, splits=splits)
    if decision is not None and decision.cluster is None:
        # The manipulated object is not among this frame's clusters (T20). The leader is *not*
        # named in its place: a hand in front of the apple does not make the crate the object.
        return dataclasses.replace(
            _unobserved_result(decision, confirm, tuple(clusters), seeds, peak_index,
                               best.target_score), splits=splits)
    if decision is not None and decision.reason in (SUBSET_KEPT_ANCHOR, UNSPLIT_KEPT_ANCHOR):
        # The object was seen, but only part of it: its anchor geometry stands in (T30 F3b).
        return dataclasses.replace(
            _anchor_result(decision, confirm, tuple(clusters), seeds, peak_index,
                           best.target_score, ts), splits=splits)
    chosen = clusters[0] if decision is None else decision.cluster
    # The strongest *other* candidate. Identical to `clusters[1]` whenever rank 1 was taken, which is
    # the definition `GraspLatch` was written against; when the held object was kept over a stronger
    # leader, the leader is the competitor, and reporting the ambiguity is the honest answer.
    others = [c.target_score for c in clusters if c is not chosen]
    cluster_points = points[chosen.point_indices]
    target = TargetGeometry(
        id=chosen.id,
        points=cluster_points,
        point_indices=chosen.point_indices,
        centroid=chosen.centroid,
        bounding_geometry=fit_sphere(cluster_points, min_radius=min_radius, semantic_role="target"),
        attention_score=chosen.mean_attention,
        confidence=chosen.target_score,
        timestamp=ts,
        frame_id=attention_cloud.cloud.frame_id,
        metrics={
            "mean_attention": chosen.mean_attention,
            "max_attention": chosen.max_attention,
            "attention_mass": chosen.attention_mass,
            "point_count": float(chosen.point_count),
            "spatial_compactness": chosen.spatial_compactness,
            "distance_from_attention_peak": chosen.distance_from_attention_peak,
            "rms_radius": chosen.rms_radius,
            # Where the chosen cluster sat in the ranking (0 = the leader) and how many frames the
            # current challenger has led. Both are 0 on every frame that has no `confirm`, so a log
            # showing anything else is showing hysteresis actually doing something.
            "target_rank": float(0 if decision is None else decision.rank),
            "target_confirm_streak": float(0 if decision is None else decision.streak),
            # The margin over the second-best candidate. `GraspLatch` reads exactly this key to
            # decide whether a frame is unambiguous enough to count toward a lock
            # (`trajopt/grasp_latch.py`: `confident = score >= score_ratio * max(runner_up, 1e-9)`),
            # and while it was missing `metrics.get("runner_up_score", 0.0)` returned 0.0 on every
            # frame — the threshold collapsed to 1.3e-9, so *any* grounded frame read as confident
            # and the ambiguity test did nothing. `clusters` is sorted by descending score, so with
            # no `confirm` the chosen cluster is [0] and the runner-up is [1].
            #
            # A single candidate has no runner-up, and 0.0 is both the literal value of "the second
            # score" and the right latch answer: with no competitor there is nothing to be ambiguous
            # about. It is also unambiguous as a signal, because a scored cluster can never reach
            # 0.0 — `_score_cluster` mixes mean attention with `exp(-d/scale)` terms that are
            # strictly positive — so `runner_up_score == 0.0` means "one candidate", never "the
            # second one scored nothing". This matches `_scores()` in
            # `experiments/diagrams/ppt_target_grounding.py`, which already reports the pair this way.
            "runner_up_score": max(others) if others else 0.0,
            # T26: the admissibility evidence, as numbers (the full record is on the decision).
            **_admissibility_metrics(None if decision is None else decision.admissibility),
            **_association_metrics(None if decision is None else decision.association),
            **_split_metrics(chosen),
            **_hand_metrics(decision),
        },
    )
    if confirm is not None:
        # The manipulated object's latest observation — what an `occluded` frame falls back on.
        confirm.observe(target, attention=attention[chosen.point_indices])
    return GroundingResult(
        target, GroundingStatus.OK, tuple(clusters), seeds, peak_index, best.target_score,
        splits=splits,
    )


def _split_groups(groups, points, attention, cfg, confirm, *, enabled):
    """T31b: split the triggered clusters of this frame (`stages/attention_split.py`).

    Returns `([(member_indices, ClusterInfo.split), ...], splits)` — the parents in their grown
    order, each replaced by its components (target first) when the split was taken — and reports
    the frame's record to `confirm`.
    """
    if confirm is not None:
        opening, skipped = confirm.split_opening(cfg)
    elif cfg.attention_split_radius is None:
        opening, skipped = None, "off"
    elif cfg.gripper_max_opening is None:
        opening, skipped = None, "max_opening_unknown"
    else:
        opening, skipped = float(cfg.gripper_max_opening), None
    if not enabled and skipped is None:
        opening, skipped = None, "no_3d_connectivity"
    out, decisions = [], []
    for idx in groups:
        dec = None if opening is None else split_by_attention(
            points, attention, idx, max_opening=opening, radius=float(cfg.attention_split_radius),
            min_points=int(cfg.min_points), min_ratio=float(cfg.attention_split_ratio))
        if dec is None:
            out.append((idx, None))
            continue
        decisions.append(dec)
        if not dec.accepted:
            # Recorded on the cluster too: a triggered cluster left whole may still be several
            # objects, so it must not replace an anchor that a split produced (T31b).
            out.append((idx, {"role": "unsplit", "reason": dec.reason,
                              "split_index": len(decisions) - 1, "parent_n_points": int(len(idx)),
                              "parent_extents_mm": [round(float(v) * 1000.0, 1)
                                                    for v in dec.parent_extents_m]}))
            continue
        parent = {"split_index": len(decisions) - 1, "parent_n_points": int(len(idx)),
                  "parent_extents_mm": [round(float(v) * 1000.0, 1) for v in dec.parent_extents_m]}
        for j, g in enumerate(dec.groups()):
            out.append((np.sort(g), {"role": "target" if j == 0 else "obstacle", **parent}))
    if confirm is not None:
        confirm.note_split({
            "ran": opening is not None,
            "skipped": skipped,
            "max_opening_mm": None if opening is None else round(float(opening) * 1000.0, 2),
            "n_triggered": len(decisions),
            "n_split": sum(1 for d in decisions if d.accepted),
            "decisions": [d.record() for d in decisions],
        })
    return out, tuple(decisions)


def _split_metrics(chosen: "ClusterInfo") -> dict[str, float]:
    """T31b: the chosen cluster is a split component (`attention_split = 1`) — and which role."""
    if chosen.split is None:
        return {}
    if chosen.split.get("role") == "unsplit":
        return {"attention_split": 0.0, "attention_split_unsplit": 1.0}
    return {"attention_split": 1.0,
            "attention_split_target": float(chosen.split.get("role") == "target"),
            "attention_split_parent_n_points": float(chosen.split.get("parent_n_points", 0))}


def _hand_metrics(decision: Optional[ConfirmDecision]) -> dict[str, float]:
    """T32b S3: hand gap to the standing geometry and whether the hand hid it (flat metrics)."""
    if decision is None or decision.hand_gap_m is None:
        return {}
    return {"manipulated_hand_gap_m": float(decision.hand_gap_m),
            "manipulated_hand_occluded": float(decision.hand_occluded)}


def _association_metrics(assoc: Optional[Association]) -> dict[str, float]:
    """T30 F3 numbers as flat metrics (the full record is `ManipulatedIdentity.association`)."""
    if assoc is None:
        return {}
    return {"association_shift_m": float(assoc.shift_m),
            "association_radius_m": float(assoc.radius_m),
            "association_ratio_narrowest": float(assoc.extent_ratio[0]),
            "association_ratio_middle": float(assoc.extent_ratio[1])}


def _admissibility_metrics(adm: Optional[Admissibility]) -> dict[str, float]:
    if adm is None:
        return {}
    out = {"destination_overlap": float(adm.destination_overlap),
           "admissible": float(adm.admissible)}
    if adm.narrowest_extent_m is not None:
        out["narrowest_extent_m"] = float(adm.narrowest_extent_m)
    if adm.max_opening_m is not None:
        out["gripper_max_opening_m"] = float(adm.max_opening_m)
    return out


def _unseen(confirm: Optional["TargetConfirm"]) -> None:
    """A frame with no clusters: the manipulated object (if any) ages; the result is unchanged."""
    if confirm is not None:
        confirm.unseen()


def _anchor_result(decision, confirm, clusters, seeds, peak_index, best_score, ts):
    """`subset_kept_anchor` (T30 F3b) / `unsplit_kept_anchor` (T31b) → the kept geometry is this
    frame's target.

    Marked like an occluded stand-in so no consumer takes it for a fresh cluster: empty
    `point_indices`, `target_rank` = the observed cluster's rank, `manipulated_subset_kept_anchor
    = 1` and this frame's association numbers.
    """
    confirm.observe_seen(ts)
    last = confirm.last_geometry
    others = [c.target_score for c in clusters if c is not decision.cluster]
    metrics = {k: v for k, v in last.metrics.items() if not k.startswith("association_")}
    metrics.update({
        "target_rank": float(decision.rank),
        "target_confirm_streak": float(decision.streak),
        "runner_up_score": max(others) if others else 0.0,
        "manipulated_occluded": 0.0,
        "manipulated_subset_kept_anchor": 1.0,
        # T31b: kept because the cluster at its place was an unsplit merge (not a subset).
        "manipulated_unsplit_kept_anchor": float(decision.reason == UNSPLIT_KEPT_ANCHOR),
        **_association_metrics(decision.association),
        **_hand_metrics(decision),
    })
    target = dataclasses.replace(
        last, point_indices=np.zeros(0, np.int64), metrics=metrics,
        seed_camera=None, supporting_cameras=())
    return GroundingResult(target, GroundingStatus.OK, clusters, seeds, peak_index, best_score)


def _unobserved_result(decision, confirm, clusters, seeds, peak_index, best_score):
    """`occluded` → the last observed geometry as this frame's target; `lost` → no target.

    The occluded target is the last observation **as it was**: its points, centroid and bounding
    sphere, and its `timestamp` (when those were true). Two things are changed so no consumer
    mistakes it for a fresh cluster: `point_indices` is empty (none of *this* cloud's points are
    it) and `metrics` carries `manipulated_occluded = 1`, `manipulated_age_frames` and
    `target_rank = -1`. `runner_up_score` is the best score of this frame — every cluster is a
    competitor, and `GraspLatch` must see that ambiguity, not a stale one.
    """
    last = confirm.last_geometry
    if decision.state == LOST or last is None:
        return GroundingResult(None, GroundingStatus.LOST, clusters, seeds, peak_index, best_score)
    # The last observation's own association numbers describe *that* frame; this frame's (if any)
    # are added below.
    metrics = {k: v for k, v in last.metrics.items() if not k.startswith("association_")}
    metrics.update({
        "target_rank": -1.0,
        "target_confirm_streak": float(decision.streak),
        "runner_up_score": max(c.target_score for c in clusters),
        "manipulated_occluded": 1.0,
        "manipulated_age_frames": float(decision.age_frames),
        # T26: 1 when the object's place held a cluster that could not be it (merged with fingers
        # or the crate) — the geometry above is the last *admissible* observation.
        "manipulated_inadmissible_match": float(decision.reason == INADMISSIBLE_MATCH),
        # T30 F3: 1 when the nearest cluster failed the pre-grasp association test; its numbers.
        "manipulated_association_rejected": float(decision.reason == ASSOCIATION_REJECTED),
        **_association_metrics(decision.association),
        **_hand_metrics(decision),
    })
    target = dataclasses.replace(
        last, id=-1, point_indices=np.zeros(0, np.int64), metrics=metrics,
        seed_camera=None, supporting_cameras=())
    return GroundingResult(target, GroundingStatus.OK, clusters, seeds, peak_index, best_score)


__all__ = [
    "ASSOCIATION_REJECTED",
    "SUBSET_KEPT_ANCHOR",
    "UNSPLIT_KEPT_ANCHOR",
    "INADMISSIBLE_MATCH",
    "NO_ADMISSIBLE",
    "UNCHECKED",
    "ClusterInfo",
    "ConfirmDecision",
    "GroundingResult",
    "LOST",
    "MANIPULATED_STATES",
    "ManipulatedIdentity",
    "OCCLUDED",
    "TargetConfirm",
    "VISIBLE",
    "connected_components_3d",
    "dbscan",
    "extract_seeds",
    "ground_target",
    "grow_region",
]
