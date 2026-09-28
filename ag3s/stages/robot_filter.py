"""Stage 2 — remove the robot's own body from the scene cloud.

The arm is the largest thing in most depth frames and it is not an obstacle to itself. Left in, it
becomes a cluster that tracks the end-effector perfectly, and TO is then asked to keep the robot
away from the robot.

The filter is a ball query against the same sphere chain the constraints are written against
(`RobotCollisionModel.sphere_centers_numeric`), which is the reason both come from one FK
implementation: if the filter used a *different* geometry from the constraints, points could be
deleted as "robot" that the constraint side considers free space, and the mismatch would be
invisible until something hit something.

The query runs **per sphere, not per point**, and that is the whole performance story. Each sphere
has its own radius, so one ball query cannot answer all of them at once; the choice is between a
Python iteration per scene point (~60k) and one per robot sphere (~200). Measured on the RB-Y1
transport scene with a 194-sphere whole-body model, per-point cost 257 ms per frame and per-sphere
8 ms — identical output, thirty times faster, because the loop is over the small set. The
whole-robot model that made this matter is itself the point: a filter built from RB-Y1's arm-only
URDF capsules leaves the base in the cloud, and the base then clusters into a phantom obstacle.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from benchmark.ag3s.config import AG3SConfigError, PointCloudConfig
from benchmark.ag3s.types import PointCloud, RobotCollisionModel


def inflation_link_groups() -> dict[str, tuple[str, ...]]:
    """`{group: member links}` for the group names `self_filter_inflation_by_link` accepts.

    The member lists are **read, not copied**, from `grounding_report` — the canonical definition
    of `serve_safe --links arms|gripper`. Written out a second time they would drift, and the day
    they drift `gripper` stops meaning the same links in the constraint model and in the filter.
    Imported lazily: that module imports this one, and the default (no override) path never needs it.
    """
    from benchmark.ag3s.experiments.reports.grounding_report import ARM_LINKS, GRIPPER_LINKS

    groups = {"arms": tuple(ARM_LINKS), "gripper": tuple(GRIPPER_LINKS)}
    missing = set(PointCloudConfig.INFLATION_GROUPS) - set(groups)
    if missing:  # pragma: no cover - guards the two lists against drifting apart
        raise AG3SConfigError(f"no member list for inflation group(s) {sorted(missing)}")
    return groups


def resolve_self_filter_inflation(
    link_names: Optional[Sequence[str]],
    config: PointCloudConfig,
) -> float | np.ndarray:
    """The inflation the self-filter applies: a **float** when there is no override, else `(S,)`.

    With `self_filter_inflation_by_link` empty this returns `config.self_filter_inflation` itself —
    the exact scalar the filter used before T19 — so the default path is not merely equivalent but
    the same arithmetic. With overrides it returns one value per sphere, aligned with
    `link_names` (the model's `sphere_link_names`, in sphere order).

    Precedence is by specificity: `arms` < `gripper` < an explicit link name, whatever the dict
    order. A key that is neither a group nor a link of this model raises `AG3SConfigError` rather
    than falling back to the global value; so does a group none of whose links the model has.
    """
    base = float(config.self_filter_inflation)
    by_link = dict(config.self_filter_inflation_by_link or {})
    if not by_link:
        return base
    if link_names is None:
        raise AG3SConfigError(
            "pointcloud.self_filter_inflation_by_link is set but the self-filter robot model has no "
            "sphere_link_names; a per-link margin cannot be placed on unnamed spheres")
    names = tuple(str(n) for n in link_names)
    present = set(names)
    group_keys = [k for k in by_link if k in PointCloudConfig.INFLATION_GROUPS]
    groups = inflation_link_groups() if group_keys else {}

    unknown = [k for k in by_link if k not in groups and k not in present]
    if unknown:
        raise AG3SConfigError(
            f"pointcloud.self_filter_inflation_by_link names link(s) the self-filter model does not "
            f"have: {unknown}. Accepted: a link name from the model or a group "
            f"{list(PointCloudConfig.INFLATION_GROUPS)}. Falling back to the global value silently "
            f"would leave a filter you believe you tuned untouched. Model links: {sorted(present)}")

    per_link: dict[str, float] = {}
    # Larger group first so a smaller one inside it (`gripper` inside `arms`) wins; stable for ties.
    for key in sorted(group_keys, key=lambda k: -len(groups[k])):
        members = [link for link in groups[key] if link in present]
        if not members:
            raise AG3SConfigError(
                f"pointcloud.self_filter_inflation_by_link[{key!r}]: none of the group's links "
                f"{list(groups[key])} has a sphere in the self-filter model")
        for link in members:
            per_link[link] = float(by_link[key])
    for key, value in by_link.items():
        if key not in groups:
            per_link[key] = float(value)
    return np.asarray([per_link.get(n, base) for n in names], np.float64)


def inflation_table(
    link_names: Optional[Sequence[str]], inflation: float | np.ndarray
) -> dict[str, float]:
    """`{link: effective inflation}` in first-appearance order — what the start log prints.

    A scalar inflation gives every link the same value; with no link names the table is
    `{"*": value}`.
    """
    if link_names is None:
        return {"*": float(np.max(inflation)) if np.ndim(inflation) else float(inflation)}
    names = tuple(str(n) for n in link_names)
    values = np.broadcast_to(np.asarray(inflation, np.float64), (len(names),))
    out: dict[str, float] = {}
    for name, value in zip(names, values):
        out.setdefault(name, float(value))
    return out


def robot_sphere_mask(
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    inflation: float | np.ndarray = 0.0,
    *,
    guard_centre: Optional[np.ndarray] = None,
    guard_radius: float = 0.0,
    return_protected: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """``True`` where a point lies inside some robot sphere, inflated by `inflation`.

    Inflation absorbs calibration error, depth noise and the gap between the capsule chain and the
    real mesh. It is the conservative direction for *this* stage in the sense that matters: deleting
    a few extra points near the arm loses a sliver of table, while under-filtering leaves a phantom
    obstacle welded to the gripper.

    `inflation` is a scalar or one value per sphere (`resolve_self_filter_inflation`).

    **Target guard** (T19): points within `guard_radius` of `guard_centre` are tested against the
    **bare** sphere radius (inflation 0) instead. That is not an exemption — a guarded point inside a
    robot sphere is still deleted — so the robot surface goes and the object beside it stays. With
    `guard_centre is None` or `guard_radius <= 0` the guard does nothing.

    With `return_protected=True` also returns the points the guard kept: inside some inflated
    sphere, outside every bare one, and within the guard.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, np.float64).reshape(-1, 3)
    centers = np.asarray(centers, np.float64).reshape(-1, 3)
    radii = np.asarray(radii, np.float64).reshape(-1)
    if centers.shape[0] != radii.shape[0]:
        raise ValueError(f"{centers.shape[0]} sphere centres but {radii.shape[0]} radii")
    if np.ndim(inflation) and np.shape(inflation) != radii.shape:
        raise ValueError(
            f"inflation has shape {np.shape(inflation)} but there are {radii.shape[0]} spheres")
    inside = np.zeros(points.shape[0], bool)
    if points.shape[0] == 0 or centers.shape[0] == 0:
        return (inside, inside.copy()) if return_protected else inside

    guarded = None
    if guard_centre is not None and float(guard_radius) > 0.0:
        centre_g = np.asarray(guard_centre, np.float64).reshape(3)
        guarded = np.linalg.norm(points - centre_g, axis=1) <= float(guard_radius)
        if not guarded.any():
            guarded = None
    touched = np.zeros(points.shape[0], bool) if guarded is not None else None

    tree = cKDTree(points)
    for centre, bare, radius in zip(centers, radii, radii + inflation):
        # Exact rather than bounded: each sphere is queried at its own radius, so nothing relies on a
        # shared upper bound and no second distance test is needed.
        hits = tree.query_ball_point(centre, float(radius), workers=-1)
        if not hits:
            continue
        if guarded is None:
            inside[hits] = True
            continue
        hits = np.asarray(hits, np.intp)
        touched[hits] = True
        in_guard = guarded[hits]
        if in_guard.any():
            g = hits[in_guard]
            keep = np.linalg.norm(points[g] - centre, axis=1) <= float(bare)
            hits = np.concatenate([hits[~in_guard], g[keep]])
        inside[hits] = True
    if not return_protected:
        return inside
    protected = np.zeros_like(inside) if touched is None else (touched & ~inside)
    return inside, protected


def self_filter_mask(
    points: np.ndarray,
    robot_model: RobotCollisionModel,
    robot_state: np.ndarray,
    config: PointCloudConfig,
    *,
    inflation: float | np.ndarray | None = None,
    guard_centre: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The one self-filter decision, shared by the cloud filter and the depth robot mask.

    Both call sites go through here so they cannot disagree on the margin: if they did, grounding
    (which reads the filtered cloud) and the TSDF (which reads the masked depth) would each see a
    different robot. `inflation=None` resolves it from `config` and the model's link names; the
    pipeline resolves once at construction and passes the same object to both.

    Returns ``(inside, guard_stats)``.
    """
    centers, radii = robot_model.sphere_centers_numeric(np.asarray(robot_state, np.float64))
    if inflation is None:
        inflation = resolve_self_filter_inflation(
            _link_names_or_none(robot_model, config), config)
    radius = float(config.self_filter_target_guard_radius)
    active = radius > 0.0 and guard_centre is not None
    inside, protected = robot_sphere_mask(
        points, centers, radii, inflation,
        guard_centre=guard_centre if active else None, guard_radius=radius,
        return_protected=True)
    return inside, {
        "guard_enabled": radius > 0.0,
        "guard_active": bool(active),
        "guard_radius": radius,
        "guard_centroid": (None if not active
                           else [float(x) for x in np.asarray(guard_centre).reshape(3)]),
        "n_guard_protected": int(protected.sum()),
    }


def _link_names_or_none(robot_model: Any, config: PointCloudConfig) -> Optional[Sequence[str]]:
    """`sphere_link_names`, read only when an override needs it (mock models may not have it)."""
    if not config.self_filter_inflation_by_link:
        return None
    return getattr(robot_model, "sphere_link_names", None)


def filter_robot_points(
    cloud: PointCloud,
    robot_model: Optional[RobotCollisionModel],
    robot_state: Optional[np.ndarray],
    config: PointCloudConfig | None = None,
    *,
    inflation: float | np.ndarray | None = None,
    guard_centre: Optional[np.ndarray] = None,
) -> tuple[PointCloud, dict[str, Any]]:
    """Drop points inside the robot's collision geometry at `robot_state`.

    A no-op — returning the cloud unchanged and reporting `enabled: False` — when the self-filter is
    switched off (`pointcloud.self_filter: false`, an ablation axis), when no robot model was
    injected, or when no robot state was supplied. Silently skipping is right here: AG3S must stay
    usable without a robot model, and a caller who wanted filtering will see `enabled: False` in the
    stats rather than having to infer it.

    `inflation` / `guard_centre`: see `self_filter_mask`.

    Returns ``(cloud, stats)`` with `n_in`, `n_removed`, `n_out`, `enabled`, and the guard fields
    `guard_enabled`, `guard_active`, `guard_radius`, `guard_centroid`, `n_guard_protected`.
    """
    cfg = config or PointCloudConfig()
    n_in = len(cloud)
    if not cfg.self_filter or robot_model is None or robot_state is None or cloud.is_empty:
        return cloud, {"n_in": n_in, "n_removed": 0, "n_out": n_in, "enabled": False,
                       **_guard_off(cfg)}

    q = np.asarray(robot_state, np.float64).reshape(-1)
    nq = int(robot_model.nq)
    if q.shape[0] != nq:
        raise ValueError(f"robot_state has {q.shape[0]} entries but the robot model expects nq={nq}")

    inside, guard = self_filter_mask(
        cloud.points, robot_model, q, cfg, inflation=inflation, guard_centre=guard_centre)
    kept = np.nonzero(~inside)[0]
    return cloud.select(kept), {
        "n_in": n_in,
        "n_removed": int(inside.sum()),
        "n_out": int(kept.size),
        "enabled": True,
        **guard,
    }


def _guard_off(cfg: PointCloudConfig) -> dict[str, Any]:
    radius = float(cfg.self_filter_target_guard_radius)
    return {"guard_enabled": radius > 0.0, "guard_active": False, "guard_radius": radius,
            "guard_centroid": None, "n_guard_protected": 0}


__all__ = [
    "filter_robot_points",
    "inflation_link_groups",
    "inflation_table",
    "resolve_self_filter_inflation",
    "robot_sphere_mask",
    "self_filter_mask",
]
