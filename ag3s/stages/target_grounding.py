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
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from benchmark.ag3s.config import ClusteringConfig
from benchmark.ag3s.stages.geometry import fit_sphere, rms_radius
from benchmark.ag3s.types import AttentionPointCloud, GroundingStatus, TargetGeometry

_EPS = 1e-9
#: "Not given" for `TargetConfirm(lost_frames=...)`, where None already means "never lost".
_FROM_CONFIG = object()


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
    counts = tree.query_ball_point(pts, eps, return_length=True, workers=-1)
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
        dist, j = core_tree.query(pts[non_core], k=1, distance_upper_bound=eps, workers=-1)
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
        neighbours = tree.query_ball_point(pts[frontier], radius, workers=-1)
        flat = np.fromiter(
            (j for group in neighbours for j in group), np.int64,
            count=sum(len(g) for g in neighbours),
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
    """

    def __init__(
        self,
        config: ClusteringConfig | None = None,
        *,
        frames: Optional[int] = None,
        tolerance: Optional[float] = None,
        switch_min_score: Optional[float] = None,
        lost_frames: Any = _FROM_CONFIG,
    ):
        cfg = config or ClusteringConfig()
        self.frames = int(cfg.target_confirm_frames if frames is None else frames)
        self.tolerance = float(cfg.target_identity_tolerance if tolerance is None else tolerance)
        self.switch_min_score = float(
            cfg.target_switch_min_score if switch_min_score is None else switch_min_score)
        # None is a legal value (never lost), so "not given" needs its own sentinel.
        lost = cfg.target_lost_frames if lost_frames is _FROM_CONFIG else lost_frames
        self.lost_frames: Optional[int] = None if lost is None else int(lost)
        self._held: Optional[np.ndarray] = None
        self._challenger: Optional[np.ndarray] = None
        self._streak = 0
        # --- identity (T20) ---
        self._id: Optional[int] = None
        self._next_id = 0
        self._switched_from: Optional[int] = None
        self._geometry: Optional[TargetGeometry] = None
        self._age = 0
        self._last_seen_frame = -1
        self._last_seen_time: Optional[float] = None
        #: Index of the current grounding call since `reset()`; -1 before the first.
        self._frame = -1
        #: The last decision, for logging. `None` before the first frame with a cluster.
        self.last: Optional[ConfirmDecision] = None

    # --- queries ---------------------------------------------------------------------------
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
        )

    def reset(self) -> None:
        """Episode boundary. The next frame's leader is taken with no argument, under a fresh id."""
        self._held = self._challenger = None
        self._streak = 0
        self._id = None
        self._next_id = 0
        self._switched_from = None
        self._geometry = None
        self._age = 0
        self._last_seen_frame = -1
        self._last_seen_time = None
        self._frame = -1
        self.last = None

    # --- one frame -------------------------------------------------------------------------
    def select(self, clusters: "Sequence[ClusterInfo]") -> ConfirmDecision:
        """Pick this frame's target from `clusters`, which must be sorted by descending score.

        Returns a decision whose `cluster` is None when the manipulated object is `occluded` or
        `lost` — the caller decides what stands in for it (`ground_target`: the last observed
        geometry, or no target). It never substitutes the leader for an unobserved object.
        """
        if not clusters:
            raise ValueError("TargetConfirm.select needs at least one cluster")
        self._frame += 1
        leader = clusters[0]

        if self._held is None:
            self._adopt(leader, switched_from=None)
            self._challenger, self._streak = None, 0
            return self._record(leader, 0, "first")

        held_rank = self._nearest(clusters, self._held)
        if held_rank == 0:
            # The leader is the object we are already calling the target. Track it and forget any
            # challenger: leading again after a gap has to start the count over.
            self._seen(leader)
            self._challenger, self._streak = None, 0
            return self._record(leader, 0, "keep")

        if float(leader.target_score) >= self.switch_min_score:
            if self._challenger is not None and self._within(leader.centroid, self._challenger):
                self._streak += 1
            else:
                self._streak = 1
            self._challenger = np.asarray(leader.centroid, np.float64).reshape(3)
        else:
            # A leader under the minimum does not count, and it breaks the run: "N frames running
            # at or above the minimum" is the rule, so a sub-minimum frame is a gap.
            self._challenger, self._streak = None, 0

        if self._streak >= self.frames:
            self._adopt(leader, switched_from=self._id)
            self._challenger, self._streak = None, 0
            return self._record(leader, 0, "switch")

        if held_rank is None:
            # The held object produced no cluster this frame. It is not replaced by the leader —
            # being hidden by the hand is not being a different object. The count is *not* reset:
            # if it stays gone, a qualifying challenger still takes over on schedule.
            self._age += 1
            state = self.state
            return self._record(None, -1, LOST if state == LOST else OCCLUDED)

        held = clusters[held_rank]
        self._seen(held)
        return self._record(held, held_rank, "hold")

    def unseen(self) -> Optional[str]:
        """A grounding call that produced no clusters at all (NO_SEED, NO_CLUSTER, ...).

        The manipulated object was not observed either, so it ages; nothing else moves — the switch
        count is untouched, as it was before T20 when such a frame never reached `select`. Returns
        the resulting state (None when nothing is held).
        """
        self._frame += 1
        if self._held is None:
            return None
        self._age += 1
        return self.state

    def observe(self, target: TargetGeometry) -> None:
        """Record this frame's geometry of the manipulated object (called by `ground_target` right
        after a `first`/`keep`/`hold`/`switch` decision). This is what `occluded` falls back on."""
        self._geometry = target
        self._last_seen_time = float(target.timestamp)

    # --- internals -------------------------------------------------------------------------
    def _adopt(self, cluster: "ClusterInfo", *, switched_from: Optional[int]) -> None:
        self._id = self._next_id
        self._next_id += 1
        self._switched_from = switched_from
        self._geometry = None  # the previous object's shape must never stand in for this one
        self._last_seen_time = None
        self._seen(cluster)

    def _seen(self, cluster: "ClusterInfo") -> None:
        self._held = np.asarray(cluster.centroid, np.float64).reshape(3)
        self._age = 0
        self._last_seen_frame = self._frame

    def _record(self, cluster: Optional["ClusterInfo"], rank: int, mode: str) -> ConfirmDecision:
        self.last = ConfirmDecision(
            cluster, int(rank), mode, self._streak, self.frames,
            state=str(self.state), manipulated_id=self._id, age_frames=int(self._age))
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

    Note the excluded points are excluded from *connectivity and clustering only*. They remain in the
    cloud and go on to become `SUPPORT_SURFACE` candidates in stage 5; nothing is deleted here.
    """
    cfg = config or ClusteringConfig()
    ts = time.time() if timestamp is None else float(timestamp)
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

    clusters = [
        _score_cluster(i, idx, member_points, member_attention, peak_point, cfg)
        for i, (_, idx) in enumerate(sorted(groups.items()))
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
            None, GroundingStatus.LOW_SCORE, tuple(clusters), seeds, peak_index, best.target_score
        )

    # Which candidate wins. Without a `confirm` this is rank 1 and the frame stands alone, which is
    # what every direct caller gets and what makes this function reproducible from one frame. With
    # one, a challenger has to lead `target_confirm_frames` frames running before the target moves.
    decision = confirm.select(clusters) if confirm is not None else None
    if decision is not None and decision.cluster is None:
        # The manipulated object is not among this frame's clusters (T20). The leader is *not*
        # named in its place: a hand in front of the apple does not make the crate the object.
        return _unobserved_result(decision, confirm, tuple(clusters), seeds, peak_index,
                                  best.target_score)
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
        },
    )
    if confirm is not None:
        # The manipulated object's latest observation — what an `occluded` frame falls back on.
        confirm.observe(target)
    return GroundingResult(
        target, GroundingStatus.OK, tuple(clusters), seeds, peak_index, best.target_score
    )


def _unseen(confirm: Optional["TargetConfirm"]) -> None:
    """A frame with no clusters: the manipulated object (if any) ages; the result is unchanged."""
    if confirm is not None:
        confirm.unseen()


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
    metrics = dict(last.metrics)
    metrics.update({
        "target_rank": -1.0,
        "target_confirm_streak": float(decision.streak),
        "runner_up_score": max(c.target_score for c in clusters),
        "manipulated_occluded": 1.0,
        "manipulated_age_frames": float(decision.age_frames),
    })
    target = dataclasses.replace(
        last, id=-1, point_indices=np.zeros(0, np.int64), metrics=metrics,
        seed_camera=None, supporting_cameras=())
    return GroundingResult(target, GroundingStatus.OK, clusters, seeds, peak_index, best_score)


__all__ = [
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
