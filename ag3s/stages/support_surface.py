"""Stage 5a — support surfaces (tables, floors, shelves) as half-spaces.

A table is not a blob, and forcing it through the object pipeline goes wrong in both directions: a
bounding primitive large enough to contain it reaches metres in every direction, and voxelizing it
costs the optimizer thousands of sphere rows to describe a shape that one linear inequality
describes exactly. So planes are fitted and managed separately, and reach TO as

    n . p >= d + margin

which is one row, exact, and needs no separating-normal machinery — `n` is data, not a variable.
`benchmark/knows_vla/cbf/filter.py` reached the same conclusion from the other direction: its
`HalfSpace` exists because the ellipsoid enclosing the room came out with a 13.9 m semi-axis.

Two properties this module has to have:

**Deterministic.** RANSAC is the one obviously random step in AG3S, so it is seeded and its
hypothesis set is a pure function of `(points, seed, max_iterations)`. An ablation that resampled its
planes would move every downstream cluster boundary.

**Fitted before grounding runs.** Objects stand *on* the table, within any usable `eps` of it, so
Euclidean connectivity floods from an object across the whole surface unless the plane is already
identified. Spec §3.4 numbers this after grounding, but the fit does not depend on the target, so the
pipeline runs it first and hands grounding the inlier mask.

Nothing is deleted. Plane inliers are excluded from *clustering*, and then become
`SUPPORT_SURFACE` candidates in their own right — the table is geometry the robot can hit.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from benchmark.ag3s.config import SupportSurfaceConfig
from benchmark.ag3s.types import PointCloud, SupportSurface

_EPS = 1e-12


def _plane_from_points(triplets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normals and offsets for a batch of point triplets, shape `(M, 3, 3)` -> `(M, 3)`, `(M,)`."""
    p0, p1, p2 = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.where(lengths < _EPS, 1.0, lengths))
    return normals, np.einsum("ij,ij->i", normals, p0)


def _refit(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares plane through `points` — the smallest singular direction.

    RANSAC's winning hypothesis was defined by three points, so its normal carries the noise of
    exactly three samples. Refitting on the whole inlier set is what turns a hypothesis into a
    measurement; on the synthetic fixture it takes the normal error from ~1e-3 rad to ~1e-8.
    """
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    return normal, float(normal @ centroid)


def _orient(normal: np.ndarray, offset: float, reference: np.ndarray) -> tuple[np.ndarray, float]:
    """Point the normal into free space, i.e. the same side as `normal_reference` (usually +z)."""
    if float(normal @ reference) < 0.0:
        return -normal, -offset
    return normal, offset


def fit_plane_ransac(
    points: np.ndarray,
    config: SupportSurfaceConfig | None = None,
    *,
    seed: Optional[int] = None,
    min_inliers: Optional[int] = None,
) -> tuple[Optional[np.ndarray], float, np.ndarray]:
    """One plane. Returns `(normal, offset, inlier_indices)`; normal is None if none qualified.

    `min_inliers` overrides the size gate. `fit_support_surfaces` uses it to hold every plane to the
    *original* cloud size — see there for why that matters.

    Hypotheses are scored in chunks rather than one enormous `(N, max_iterations)` distance matrix —
    at 40k points and 200 hypotheses that array is 64 MB, and the chunked version is no slower.

    Candidate planes whose normal is more than `max_normal_angle_deg` from `normal_reference` are
    rejected *before* scoring. Without that gate the largest plane in a tabletop scene is sometimes a
    wall or the side of a box, and the pipeline would then subtract a wall and cluster the table.
    """
    cfg = config or SupportSurfaceConfig()
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n < 3:
        return None, 0.0, np.zeros(0, np.int64)

    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    triplet_idx = rng.integers(0, n, size=(cfg.max_iterations, 3))
    # Degenerate triplets (repeated indices) give a zero normal; drop them up front.
    distinct = (
        (triplet_idx[:, 0] != triplet_idx[:, 1])
        & (triplet_idx[:, 1] != triplet_idx[:, 2])
        & (triplet_idx[:, 0] != triplet_idx[:, 2])
    )
    triplet_idx = triplet_idx[distinct]
    if triplet_idx.shape[0] == 0:
        return None, 0.0, np.zeros(0, np.int64)

    normals, offsets = _plane_from_points(pts[triplet_idx])
    reference = np.asarray(cfg.normal_reference, np.float64)
    reference = reference / max(float(np.linalg.norm(reference)), _EPS)
    cos_limit = float(np.cos(np.deg2rad(cfg.max_normal_angle_deg)))
    aligned = np.abs(normals @ reference) >= cos_limit
    normals, offsets = normals[aligned], offsets[aligned]
    if normals.shape[0] == 0:
        return None, 0.0, np.zeros(0, np.int64)

    best_count, best_hypothesis = -1, -1
    for start in range(0, normals.shape[0], 32):
        block = slice(start, start + 32)
        distance = np.abs(pts @ normals[block].T - offsets[block])
        counts = (distance <= cfg.distance_threshold).sum(axis=0)
        local = int(np.argmax(counts))
        if int(counts[local]) > best_count:
            best_count, best_hypothesis = int(counts[local]), start + local

    gate = max(cfg.min_inliers, int(cfg.min_inlier_ratio * n)) if min_inliers is None else int(min_inliers)
    if best_count < gate:
        return None, 0.0, np.zeros(0, np.int64)

    normal, offset = normals[best_hypothesis], float(offsets[best_hypothesis])
    inliers = np.nonzero(np.abs(pts @ normal - offset) <= cfg.distance_threshold)[0]
    normal, offset = _refit(pts[inliers])
    # Refitting moves the plane slightly, so re-select inliers against the measured plane.
    inliers = np.nonzero(np.abs(pts @ normal - offset) <= cfg.distance_threshold)[0]
    normal, offset = _orient(normal, offset, reference)
    return normal, offset, inliers


def fit_support_surfaces(
    cloud: PointCloud,
    config: SupportSurfaceConfig | None = None,
    *,
    timestamp: Optional[float] = None,
) -> tuple[list[SupportSurface], np.ndarray]:
    """Fit up to `max_planes`, largest first. Returns `(surfaces, mask)`.

    `mask` is `True` for every point belonging to some surface — what `target_grounding` and residual
    clustering exclude from connectivity.

    Planes are found sequentially, each on the points the previous ones did not claim. A tabletop
    scene usually needs one; a second slot catches the floor, or a shelf below.
    """
    cfg = config or SupportSurfaceConfig()
    ts = time.time() if timestamp is None else float(timestamp)
    n = len(cloud)
    mask = np.zeros(n, bool)
    surfaces: list[SupportSurface] = []
    if not cfg.enabled or n == 0:
        return surfaces, mask

    # The size gate is measured against the ORIGINAL cloud, not the shrinking remainder. Scoring it
    # against the remainder makes it collapse as points are consumed: on the default fixture the
    # table claims 96% of the cloud, so a 5% ratio drops from 2,066 points to 82 and the second slot
    # accepts a 661-point sliver across the objects' tops as a "support surface" — removing real
    # object geometry from clustering. Each plane has to clear the same absolute bar.
    size_gate = max(cfg.min_inliers, int(cfg.min_inlier_ratio * n))

    remaining = np.arange(n, dtype=np.int64)
    for plane_id in range(cfg.max_planes):
        if remaining.size < 3:
            break
        normal, offset, local_inliers = fit_plane_ransac(
            cloud.points[remaining], cfg, seed=cfg.seed + plane_id, min_inliers=size_gate
        )
        if normal is None or local_inliers.size == 0:
            break
        inliers = remaining[local_inliers]
        rms = float(np.sqrt((((cloud.points[inliers] @ normal) - offset) ** 2).mean()))
        surfaces.append(
            SupportSurface(
                id=plane_id,
                normal=normal,
                offset=offset,
                point_indices=inliers,
                point_count=int(inliers.size),
                rms_error=rms,
                safety_margin=cfg.safety_margin,
                frame_id=cloud.frame_id,
                timestamp=ts,
            )
        )
        mask[inliers] = True
        remaining = np.setdiff1d(remaining, inliers, assume_unique=True)
    return surfaces, mask


__all__ = ["fit_plane_ransac", "fit_support_surfaces"]
