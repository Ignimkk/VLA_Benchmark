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

from typing import Optional

import numpy as np

from benchmark.ag3s.config import PointCloudConfig
from benchmark.ag3s.types import PointCloud, RobotCollisionModel


def robot_sphere_mask(
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    inflation: float = 0.0,
) -> np.ndarray:
    """``True`` where a point lies inside some robot sphere, inflated by `inflation`.

    Inflation absorbs calibration error, depth noise and the gap between the capsule chain and the
    real mesh. It is the conservative direction for *this* stage in the sense that matters: deleting
    a few extra points near the arm loses a sliver of table, while under-filtering leaves a phantom
    obstacle welded to the gripper.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, np.float64).reshape(-1, 3)
    centers = np.asarray(centers, np.float64).reshape(-1, 3)
    radii = np.asarray(radii, np.float64).reshape(-1)
    if centers.shape[0] != radii.shape[0]:
        raise ValueError(f"{centers.shape[0]} sphere centres but {radii.shape[0]} radii")
    if points.shape[0] == 0 or centers.shape[0] == 0:
        return np.zeros(points.shape[0], bool)

    tree = cKDTree(points)
    inside = np.zeros(points.shape[0], bool)
    for centre, radius in zip(centers, radii + inflation):
        # Exact rather than bounded: each sphere is queried at its own radius, so nothing relies on a
        # shared upper bound and no second distance test is needed.
        hits = tree.query_ball_point(centre, float(radius), workers=-1)
        if hits:
            inside[hits] = True
    return inside


def filter_robot_points(
    cloud: PointCloud,
    robot_model: Optional[RobotCollisionModel],
    robot_state: Optional[np.ndarray],
    config: PointCloudConfig | None = None,
) -> tuple[PointCloud, dict[str, int | bool]]:
    """Drop points inside the robot's collision geometry at `robot_state`.

    A no-op — returning the cloud unchanged and reporting `enabled: False` — when the self-filter is
    switched off (`pointcloud.self_filter: false`, an ablation axis), when no robot model was
    injected, or when no robot state was supplied. Silently skipping is right here: AG3S must stay
    usable without a robot model, and a caller who wanted filtering will see `enabled: False` in the
    stats rather than having to infer it.

    Returns ``(cloud, stats)`` with `n_in`, `n_removed`, `n_out`, `enabled`.
    """
    cfg = config or PointCloudConfig()
    n_in = len(cloud)
    if not cfg.self_filter or robot_model is None or robot_state is None or cloud.is_empty:
        return cloud, {"n_in": n_in, "n_removed": 0, "n_out": n_in, "enabled": False}

    q = np.asarray(robot_state, np.float64).reshape(-1)
    nq = int(robot_model.nq)
    if q.shape[0] != nq:
        raise ValueError(f"robot_state has {q.shape[0]} entries but the robot model expects nq={nq}")

    centers, radii = robot_model.sphere_centers_numeric(q)
    inside = robot_sphere_mask(cloud.points, centers, radii, cfg.self_filter_inflation)
    kept = np.nonzero(~inside)[0]
    return cloud.select(kept), {
        "n_in": n_in,
        "n_removed": int(inside.sum()),
        "n_out": int(kept.size),
        "enabled": True,
    }


__all__ = ["filter_robot_points", "robot_sphere_mask"]
