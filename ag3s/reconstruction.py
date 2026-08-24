"""Stage 1 — 3D scene reconstruction.

Depth (or a point cloud handed in directly) becomes a filtered, downsampled cloud in the robot base
frame, with each point's originating pixel carried alongside it.

Three decisions here are load-bearing rather than incidental:

**`(u, v)` survives every step.** Attention lifting needs to know which pixel a 3D point came from,
so voxel downsampling keeps a *representative point* rather than the voxel centroid. A centroid has
no pixel — averaging three points from three different pixels leaves nothing to look the attention
value up with, and inventing one by nearest-neighbour would put a fabricated correspondence at the
very start of the pipeline.

**Subsampling is deterministic.** `max_points` is a real-time requirement (a 640x480 frame is 307k
points and every KD-tree downstream is superlinear), but an ablation that reruns with a different
random subset is measuring the sampler, not the change. Both the voxel filter and the cap are
order-preserving functions of the input alone — no RNG, no hashing that varies by build.

**Nothing is dropped for being uninteresting.** Only physically invalid depth (0, NaN, Inf, outside
the sensor's trustworthy range) is removed. Every remaining point reaches candidate generation, which
is what "geometry decides what can collide" means operationally.

**And nothing is dropped for being late in the array.** `max_points` used to be enforced by picking
`max_points` indices out of the cloud, which is a statement about the index range and none at all
about space: the points not picked were simply gone, and geometry that is never sampled cannot
become a candidate. The default now grows the voxel instead (`cap_strategy: "voxel"`), so every
occupied voxel still contributes a representative and the unrepresented gap is bounded by the final
voxel size. It is a coarser cloud, not a partial one, and the coarseness is reported.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from benchmark.ag3s.config import PointCloudConfig
from benchmark.ag3s.types import PointCloud


def _check_intrinsics(K: np.ndarray) -> tuple[float, float, float, float]:
    K = np.asarray(K, np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"camera_intrinsics must be (3, 3), got {K.shape}")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if abs(fx) < 1e-9 or abs(fy) < 1e-9:
        raise ValueError(f"degenerate intrinsics: fx={fx}, fy={fy}")
    return fx, fy, float(K[0, 2]), float(K[1, 2])


def _check_transform(T: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(T, np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be (4, 4), got {T.shape}")
    return T


def backproject(
    depth: np.ndarray,
    camera_intrinsics: np.ndarray,
    T_base_cam: np.ndarray,
    config: PointCloudConfig | None = None,
    *,
    frame_id: str = "base",
) -> PointCloud:
    """Pinhole back-projection, fully vectorized.

        X = (u - cx) * D / fx,   Y = (v - cy) * D / fy,   Z = D,   P_base = T_base_cam @ P_cam

    `u` is the column and `v` the row, matching the intrinsics convention. Points whose depth is
    zero, non-finite, or outside ``[depth_min, depth_max]`` are dropped here and never enter the
    cloud — a zero-depth pixel back-projects to the camera origin, which would put a phantom
    obstacle inside the robot.

    Note there is no image flipping. `benchmark/knows_vla` needs `flip_row`/`flip_col` because
    robosuite stores observations pre-flipped and the collection script rotates them again; AG3S
    takes depth in the same orientation as the intrinsics describe and leaves any such fix-up to the
    caller, where it can be validated against ground truth (see that package's
    `validate_backprojection.py` for why guessing is not an option).
    """
    cfg = config or PointCloudConfig()
    depth = np.asarray(depth, np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W), got shape {depth.shape}")
    fx, fy, cx, cy = _check_intrinsics(camera_intrinsics)
    T = _check_transform(T_base_cam, "T_base_cam")

    d = depth * cfg.depth_scale
    valid = np.isfinite(d) & (d > 0.0) & (d >= cfg.depth_min) & (d <= cfg.depth_max)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return PointCloud.empty(frame_id)

    z = d[rows, cols]
    pts_cam = np.empty((rows.size, 3), np.float64)
    pts_cam[:, 0] = (cols - cx) * z / fx
    pts_cam[:, 1] = (rows - cy) * z / fy
    pts_cam[:, 2] = z

    if cfg.range_max is not None:
        # `depth_max` bounds z-depth, not distance. On a wide lens the two diverge badly — see
        # `PointCloudConfig.range_max` — so an explicit radial gate is the only way to say
        # "this is the workspace" and mean it.
        within = np.linalg.norm(pts_cam, axis=1) <= cfg.range_max
        pts_cam, rows, cols = pts_cam[within], rows[within], cols[within]
        if rows.size == 0:
            return PointCloud.empty(frame_id)

    pts_base = pts_cam @ T[:3, :3].T + T[:3, 3]
    uv = np.stack([cols, rows], axis=1).astype(np.int32)  # (u, v) = (col, row)
    return PointCloud(pts_base, uv, frame_id)


def from_pointcloud(
    points: np.ndarray,
    config: PointCloudConfig | None = None,
    *,
    T_base_cam: Optional[np.ndarray] = None,
    uv: Optional[np.ndarray] = None,
    frame_id: str = "base",
) -> PointCloud:
    """The back-projection-free path, for sensors that already deliver 3D.

    Pass `T_base_cam` if the points are in camera coordinates; omit it if they are already in the
    base frame. `uv` is optional and, when absent, attention lifting falls back to projecting points
    into the image itself — see `attention_lifting.lift`.
    """
    cfg = config or PointCloudConfig()
    pts = np.asarray(points, np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pointcloud must be (N, 3), got {pts.shape}")

    finite = np.isfinite(pts).all(axis=1)
    if T_base_cam is not None:
        T = _check_transform(T_base_cam, "T_base_cam")
        # Range gating is a camera-frame notion, so it only applies when we know where the camera is.
        depth_cam = pts[:, 2]
        finite &= (depth_cam >= cfg.depth_min) & (depth_cam <= cfg.depth_max)
        pts = pts @ T[:3, :3].T + T[:3, 3]

    keep = np.nonzero(finite)[0]
    out_uv = None if uv is None else np.asarray(uv, np.int32).reshape(-1, 2)[keep]
    return PointCloud(pts[keep], out_uv, frame_id)


def voxel_downsample(cloud: PointCloud, voxel_size: float) -> PointCloud:
    """Keep one representative point per occupied voxel — the first in input order.

    "First in input order" rather than "the centroid" is deliberate: a representative point keeps its
    `(u, v)`, and a centroid does not. It is also what makes the result a deterministic function of
    the input, since `np.unique(..., return_index=True)` reports first occurrences.

    Output order matches input order, which keeps raster locality intact for the `max_points` stride
    cap that usually follows.
    """
    if voxel_size <= 0.0 or cloud.is_empty:
        return cloud
    keys = np.floor(cloud.points / float(voxel_size)).astype(np.int64)
    # Pack the three voxel coordinates into one sortable key. Shifting by the per-axis minimum keeps
    # the packed value non-negative without an unbounded hash, so it stays exact in int64 for any
    # scene under ~2 million voxels per axis.
    keys -= keys.min(axis=0)
    extent = keys.max(axis=0) + 1
    if float(extent[0]) * float(extent[1]) * float(extent[2]) >= 2.0**62:
        raise ValueError(
            f"voxel grid {tuple(extent)} overflows int64; voxel_size={voxel_size} is too small for this cloud"
        )
    packed = (keys[:, 0] * extent[1] + keys[:, 1]) * extent[2] + keys[:, 2]
    _, first = np.unique(packed, return_index=True)
    first.sort()  # restore input order
    return cloud.select(first)


def coverage_preserving_cap(
    cloud: PointCloud, config: PointCloudConfig
) -> tuple[PointCloud, dict[str, float | int | bool]]:
    """Enforce `max_points` by coarsening, not by selecting. Returns ``(cloud, stats)``.

    The voxel size is multiplied by `cap_voxel_growth` until the cloud fits. The safety property is
    what survives each step: `voxel_downsample` keeps one point per occupied voxel, so after the
    loop every occupied voxel of the final grid still has a representative and nothing physical is
    more than one voxel away from a point AG3S will consider. Compare with index selection, which
    can leave an entire object unsampled while reporting the same point count.

    Re-downsampling the already-downsampled cloud rather than the raw one is deliberate and safe: an
    occupied coarse voxel necessarily contains at least one of the fine representatives, so the
    coverage guarantee is unaffected and the work is proportional to the small cloud.

    `stats['covered']` is False only if the loop ran out of iterations — the caller turns that into
    `GEOMETRY_INCOMPLETE` rather than continuing with a cloud it cannot vouch for.
    """
    n_in = len(cloud)
    stats: dict[str, float | int | bool] = {
        "capped": False, "covered": True, "voxel_size": float(config.voxel_size), "n_growth_steps": 0,
    }
    if n_in <= config.max_points:
        return cloud, stats

    stats["capped"] = True
    voxel = float(config.voxel_size) if config.voxel_size > 0.0 else 0.001
    working = cloud
    for step in range(1, int(config.cap_max_iterations) + 1):
        voxel *= float(config.cap_voxel_growth)
        working = voxel_downsample(working, voxel)
        stats.update(voxel_size=voxel, n_growth_steps=step)
        if len(working) <= config.max_points:
            return working, stats

    # Out of iterations. Rather than truncate — which would silently delete geometry — say so.
    stats["covered"] = False
    return working, stats


def cap_points(cloud: PointCloud, max_points: int) -> tuple[PointCloud, bool]:
    """Deterministic uniform cap by index selection. Returns ``(cloud, was_capped)``.

    Retained as the `cap_strategy: "stride"` ablation and for callers that want the old behaviour.
    It samples the whole index range with a bounded gap, which is a fine property to have and not
    the one that matters: index adjacency is raster adjacency, not spatial adjacency, so a thin
    object seen edge-on can fall entirely between two selected indices. Prefer
    `coverage_preserving_cap`.

    Two properties, in priority order:

    1. **Deterministic** — a pure function of `(len(cloud), max_points)`, with no RNG. Reruns of an
       ablation compare like with like.
    2. **Full coverage** — the selected indices span the entire cloud with a bounded gap, so no
       region of the image is systematically unsampled. This is why the selection is `linspace`
       across the whole range rather than an integer stride: a stride of ``n // max_points``,
       truncated to `max_points`, leaves the tail of the cloud (up to ~8% of the frame at awkward
       ratios) never sampled at all, and geometry that is never sampled cannot become a candidate.

    What this does *not* give is index-set stability: adding one point to the cloud shifts roughly
    half the chosen indices by one. That is fine, and deliberately not traded for — clusters carry
    hundreds of points, so their centroids move by well under a millimetre under a resample, which is
    what the candidate tracker actually keys on. `test_reconstruction.py` measures that directly
    rather than assuming it.
    """
    n = len(cloud)
    if max_points < 1:
        raise ValueError(f"max_points must be >= 1, got {max_points}")
    if n <= max_points:
        return cloud, False
    idx = np.unique(np.linspace(0, n - 1, max_points).astype(np.int64))
    return cloud.select(idx), True


def reconstruct(
    *,
    depth: Optional[np.ndarray] = None,
    pointcloud: Optional[np.ndarray] = None,
    camera_intrinsics: Optional[np.ndarray] = None,
    T_base_cam: Optional[np.ndarray] = None,
    uv: Optional[np.ndarray] = None,
    config: PointCloudConfig | None = None,
    frame_id: str = "base",
) -> tuple[PointCloud, dict[str, int | bool]]:
    """Full stage: back-project (or accept a cloud), filter, voxel-downsample, cap.

    Robot self-filtering is *not* done here — it is a separate stage in `robot_filter`, both because
    it needs the robot model AG3S does not own and because the spec's latency table reports it
    separately.

    Returns the cloud and a small stats dict (`n_raw`, `n_voxel`, `n_final`, `capped`) that the
    profiler and the pipeline's `DEGRADED` status read.
    """
    cfg = config or PointCloudConfig()
    if (depth is None) == (pointcloud is None):
        raise ValueError("provide exactly one of depth= or pointcloud=")

    if depth is not None:
        if camera_intrinsics is None or T_base_cam is None:
            raise ValueError("depth input requires camera_intrinsics and T_base_cam")
        cloud = backproject(depth, camera_intrinsics, T_base_cam, cfg, frame_id=frame_id)
    else:
        cloud = from_pointcloud(pointcloud, cfg, T_base_cam=T_base_cam, uv=uv, frame_id=frame_id)

    n_raw = len(cloud)
    cloud = voxel_downsample(cloud, cfg.voxel_size)
    n_voxel = len(cloud)
    if cfg.cap_strategy == "voxel":
        cloud, cap_stats = coverage_preserving_cap(cloud, cfg)
    else:
        cloud, capped = cap_points(cloud, cfg.max_points)
        # Index selection makes no spatial promise, so a cap that actually bound leaves geometry
        # unaccounted for. Reporting `covered=False` is what turns the ablation into an honest one.
        cap_stats = {
            "capped": capped, "covered": not capped,
            "voxel_size": float(cfg.voxel_size), "n_growth_steps": 0,
        }
    return cloud, {
        "n_raw": n_raw, "n_voxel": n_voxel, "n_final": len(cloud),
        "capped": bool(cap_stats["capped"]), "covered": bool(cap_stats["covered"]),
        "final_voxel_size": float(cap_stats["voxel_size"]),
        "n_growth_steps": int(cap_stats["n_growth_steps"]),
    }


__all__ = [
    "backproject",
    "cap_points",
    "coverage_preserving_cap",
    "from_pointcloud",
    "reconstruct",
    "voxel_downsample",
]
