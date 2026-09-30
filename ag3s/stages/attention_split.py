"""Stage 4b — split a cluster that may hold more than one object, by attention (T31b, 2026-09-29).

**Why.** Target grounding grows clusters with `clustering.eps` (30 mm) connectivity, so two objects
whose surfaces come within 30 mm of each other become one cluster. In E3 that happened to the
apple on 2 of 3 episodes: ep1808 adopted the apple merged with the orange (54·67·144 mm), ep1800 the
apple merged with the banana (59·65·164 mm). The narrowest extent fits the gripper, so the merged
cluster was admissible (T26), became the manipulated object, and the exclusion geometry (ball,
target-free layer, contact permission) covered the neighbour too — 249 of 408 pre-grasp chunks
(T30b.impl §F3-4).

**What attention says.** Inside those merged clusters the attention is on the apple: its share of
the cluster's attention mass has median 0.75–0.90, and the per-point mean attention apple ÷
neighbour is at least 2.2 (t=0: 1.00 vs banana 0.40, 0.883 vs orange 0.001). Telling the target from
the obstacle with the VLA's attention is the project's purpose, so the split is decided by it.

**The rule** (`split_by_attention`), for one cluster:

1. *Trigger — the cluster may hold more than one object:* its narrowest principal extent fits the
   gripper (`≤ max_opening`) but its largest does not (`> max_opening`). An admissible cluster
   longer than the opening along some axis is either one elongated object or several graspable
   objects side by side. Clusters wider than the opening in every direction are inadmissible
   anyway (T26) and are never split — splitting them could only manufacture a graspable fragment
   of, say, the crate.
2. *Geometry — re-cluster inside it* with a smaller connectivity radius
   (`clustering.attention_split_radius`, 15 mm = 3 cloud voxels). Components with at least
   `clustering.min_points` points are the candidate objects; smaller components (stragglers) are
   re-attached to the nearest candidate, so no point is dropped. Fewer than two candidates: one
   object — no split (`one_component`).
3. *Attention — who is the target:* the component with the largest attention mass is the target
   candidate. The split is taken only if attention separates it from the others: its mean attention
   must be at least `clustering.attention_split_ratio` (2.0) times that of **every** other
   component. Otherwise attention does not tell the objects apart and the cluster is left whole
   (`attention_not_discriminative`).
4. The other components stay in the frame as **their own clusters** — obstacles, scored like any
   other cluster (and, like any other cluster, challengers under the T26 switch rule).

Evidence for the radius (E3 T28+T30 records, T14, T17 — 12 + 15 + 2 runs; `T31b.impl.md`): a single
apple never re-clusters into two ≥ 20-point components at 15 mm before the grasp (0 of 139 triggered
apple-only clusters, uneven attention included), because its own surface is connected at ≤ 17 mm
(MST longest edge, 95th percentile), while the apple–neighbour gap inside the merged clusters has
median 24 mm (5th percentile 14–20 mm): 469 of 510 merged clusters separate, the target component
being all apple (apple precision and recall 1.0 at the 10th percentile).

**Not during a grasp.** `ground_target` does not split while `TargetConfirm` is frozen (closing …
held): the fingers cut the held object's surface into pieces (9 of 102 frozen apple-only clusters
would split at 15 mm, apple recall 0.61–0.65), and the object moves with the hand then — a gap there
is the hand, not a second object.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np

from benchmark.ag3s.stages.admissibility import principal_extents

#: `SplitDecision.method` — the only method implemented (the choice and its evidence: module doc).
SPLIT_METHOD = "attention_recluster"
#: `SplitDecision.reason` vocabulary.
SPLIT = "split"
ONE_COMPONENT = "one_component"
ATTENTION_NOT_DISCRIMINATIVE = "attention_not_discriminative"
SPLIT_REASONS = (SPLIT, ONE_COMPONENT, ATTENTION_NOT_DISCRIMINATIVE)

_EPS = 1e-12


@dataclasses.dataclass(frozen=True)
class SplitComponent:
    """One component of a triggered cluster (cloud indices + the numbers the decision read)."""

    indices: np.ndarray
    extents_m: tuple[float, float, float]
    attention_mass: float
    mean_attention: float
    centroid: np.ndarray

    def record(self) -> dict[str, Any]:
        return {
            "n_points": int(len(self.indices)),
            "extents_mm": [round(float(v) * 1000.0, 1) for v in self.extents_m],
            "attention_mass": round(float(self.attention_mass), 4),
            "mean_attention": round(float(self.mean_attention), 4),
            "centroid": [round(float(v), 5) for v in np.asarray(self.centroid).reshape(3)],
        }


@dataclasses.dataclass(frozen=True)
class SplitDecision:
    """The split decision for one triggered cluster (T31b) — kept whether or not it split.

    `components` are sorted by attention mass, descending: `components[0]` is the target
    candidate. When `accepted`, `groups()` returns them as separate clusters (target first);
    otherwise the parent stays one cluster.
    """

    parent_indices: np.ndarray
    parent_extents_m: tuple[float, float, float]
    parent_attention_mass: float
    max_opening_m: float
    radius_m: float
    min_points: int
    min_ratio: float
    reason: str
    components: tuple[SplitComponent, ...] = ()
    #: Mean attention of the target candidate ÷ the largest mean attention among the others.
    #: None with fewer than two components; inf when every other component has zero attention.
    ratio: Optional[float] = None
    method: str = SPLIT_METHOD

    @property
    def accepted(self) -> bool:
        return self.reason == SPLIT

    @property
    def target(self) -> Optional[SplitComponent]:
        return self.components[0] if self.accepted else None

    def groups(self) -> list[np.ndarray]:
        """The cluster(s) this parent becomes: the components (target first) or the parent."""
        if not self.accepted:
            return [self.parent_indices]
        return [c.indices for c in self.components]

    def record(self) -> dict[str, Any]:
        tgt = self.target
        return {
            "method": self.method,
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "trigger": "narrowest <= max_opening < largest",
            "max_opening_mm": round(float(self.max_opening_m) * 1000.0, 2),
            "radius_mm": round(float(self.radius_m) * 1000.0, 2),
            "min_points": int(self.min_points),
            "min_ratio": float(self.min_ratio),
            "ratio": (None if self.ratio is None
                      else ("inf" if not np.isfinite(self.ratio) else round(float(self.ratio), 3))),
            "before": {
                "n_points": int(len(self.parent_indices)),
                "extents_mm": [round(float(v) * 1000.0, 1) for v in self.parent_extents_m],
                "attention_mass": round(float(self.parent_attention_mass), 4),
            },
            "after": None if tgt is None else {
                "n_points": int(len(tgt.indices)),
                "extents_mm": [round(float(v) * 1000.0, 1) for v in tgt.extents_m],
                "attention_mass": round(float(tgt.attention_mass), 4),
                "attention_share": round(float(tgt.attention_mass
                                               / max(self.parent_attention_mass, _EPS)), 4),
            },
            "components": [c.record() for c in self.components],
        }


def triggered(extents: np.ndarray, max_opening: Optional[float]) -> bool:
    """The cluster may hold more than one object: narrowest ≤ max opening < largest."""
    if max_opening is None:
        return False
    ext = np.sort(np.asarray(extents, np.float64).reshape(-1))
    return bool(ext.size == 3 and ext[0] <= float(max_opening) < ext[-1])


def split_by_attention(
    points: np.ndarray,
    attention: np.ndarray,
    members: np.ndarray,
    *,
    max_opening: Optional[float],
    radius: float,
    min_points: int,
    min_ratio: float,
    extents: Optional[np.ndarray] = None,
) -> Optional[SplitDecision]:
    """Decide whether the cluster `members` (indices into `points`) is split (module doc).

    Returns None when the cluster is not triggered (it is one cluster, and there is nothing to
    record), else the decision. Pure: the same cloud and attention always give the same answer.
    """
    from scipy.spatial import cKDTree

    from benchmark.ag3s.stages.target_grounding import connected_components_3d

    members = np.asarray(members, np.int64).reshape(-1)
    pts_all = np.asarray(points, np.float64).reshape(-1, 3)
    att_all = np.asarray(attention, np.float64).reshape(-1)
    pts = pts_all[members]
    ext = principal_extents(pts) if extents is None else np.sort(np.asarray(extents, np.float64))
    if max_opening is None or not triggered(ext, max_opening):
        return None
    att = att_all[members]
    base = dict(parent_indices=members, parent_extents_m=tuple(float(v) for v in ext),
                parent_attention_mass=float(att.sum()), max_opening_m=float(max_opening),
                radius_m=float(radius), min_points=int(min_points), min_ratio=float(min_ratio))

    labels = connected_components_3d(pts, float(radius))
    ids, counts = np.unique(labels, return_counts=True)
    big = ids[counts >= int(min_points)]
    if big.size < 2:
        return SplitDecision(reason=ONE_COMPONENT, **base)

    # Stragglers (components under `min_points`) join the candidate object nearest to them, so a
    # split never drops a point: geometry decides extent, attention only decided which object.
    small = ~np.isin(labels, big)
    if small.any():
        is_big = ~small
        tree = cKDTree(pts[is_big])
        _, j = tree.query(pts[small], k=1)
        labels = labels.copy()
        labels[np.nonzero(small)[0]] = labels[np.nonzero(is_big)[0][j]]

    comps = []
    for k in big:
        local = np.nonzero(labels == k)[0]
        a = att[local]
        comps.append(SplitComponent(
            indices=members[local],
            extents_m=tuple(float(v) for v in principal_extents(pts[local])),
            attention_mass=float(a.sum()), mean_attention=float(a.mean()),
            centroid=pts[local].mean(axis=0)))
    # Largest attention mass first; ties by the lowest member index (deterministic).
    comps.sort(key=lambda c: (-c.attention_mass, int(c.indices.min())))
    top = comps[0].mean_attention
    rest = max(c.mean_attention for c in comps[1:])
    if top <= _EPS:
        ratio = 0.0
    elif rest <= _EPS:
        ratio = float("inf")
    else:
        ratio = top / rest
    reason = SPLIT if ratio >= float(min_ratio) else ATTENTION_NOT_DISCRIMINATIVE
    return SplitDecision(reason=reason, components=tuple(comps), ratio=ratio, **base)


__all__ = [
    "ATTENTION_NOT_DISCRIMINATIVE",
    "ONE_COMPONENT",
    "SPLIT",
    "SPLIT_METHOD",
    "SPLIT_REASONS",
    "SplitComponent",
    "SplitDecision",
    "split_by_attention",
    "triggered",
]
