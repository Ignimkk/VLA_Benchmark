"""Stage 6 — point clusters approximated by TO-friendly primitives.

Primitives, not an ESDF. A signed-distance field has to be rebuilt and re-interpolated every frame
and gives the optimizer a lookup rather than a derivative; a handful of convex shapes give a
closed-form constraint whose gradient CasADi can differentiate symbolically. Spec §3.5 sets the
preference order **sphere -> capsule -> box -> ellipsoid**, which is also the order of increasing
cost and decreasing robustness, and `fit_primitive` walks down it when a fit is degenerate.

The governing rule everywhere in this module: **a primitive must contain its cluster.**
Under-approximating an obstacle is the one error direction that produces a collision rather than a
detour, so every fit here errs outward and `test_geometry.py` asserts containment rather than
tightness.

One honest limitation, stated up front. The constraint form in §3.5 is sphere-versus-sphere, so
`to_spheres` is what actually reaches the optimizer. A capsule genuinely benefits — it becomes a
chain of spheres along its axis, the same trick `robot_models/urdf_sphere_chain.py` uses on the
robot side. A box or an ellipsoid currently falls back to its bounding sphere for constraint
purposes, so those two types tighten the *reported* geometry (`type`, `orientation`, `dimensions`
travel to TO intact) without yet tightening the emitted inequality. Making them tighter needs
box/ellipsoid distance support on the TO side, which does not exist yet.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from benchmark.ag3s.types import Primitive, PrimitiveType

_EPS = 1e-12


# ------------------------------------------------------------------------------- helpers


def rms_radius(points: np.ndarray) -> float:
    """Root-mean-square distance from the centroid — what `spatial_compactness` is built on.

    RMS rather than the max: the max is set by a single outlying point, so it measures the worst
    stray reading rather than the shape, and a compactness score built on it would rank a clean
    object below a noisy one.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sqrt(((pts - pts.mean(axis=0)) ** 2).sum(axis=1).mean()))


def _principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right-handed principal axes as columns, plus their variances, largest last.

    Largest *last* so that column 2 — the local +z that `Primitive` uses for a capsule's segment —
    is the long direction. Handedness is forced because a reflection is a valid eigenbasis but not a
    valid rotation, and `Primitive.orientation` is documented as a rotation.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    centred = pts - pts.mean(axis=0)
    cov = (centred.T @ centred) / max(pts.shape[0] - 1, 1)
    values, vectors = np.linalg.eigh(cov)  # ascending
    if np.linalg.det(vectors) < 0.0:
        vectors[:, 0] = -vectors[:, 0]
    return vectors, values


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to the segment `a`-`b`."""
    d = b - a
    length_sq = float(d @ d)
    if length_sq < _EPS:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(((points - a) @ d) / length_sq, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * d), axis=1)


# ------------------------------------------------------------------------------------ fits


def fit_sphere(points: np.ndarray, *, min_radius: float = 0.0, **fields: Any) -> Primitive:
    """Centroid, with the radius that reaches the furthest point.

    Guaranteed to contain the cluster by construction, which a minimum-volume fit only achieves with
    care. It is loose for elongated clusters — that is what the capsule is for — but loose outward.

    `min_radius` floors the result so a three-point cluster still has a usable shape instead of a
    degenerate zero-radius sphere that no constraint can push against.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise ValueError("cannot fit a sphere to an empty cluster")
    centre = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - centre, axis=1).max()) if pts.shape[0] > 1 else 0.0
    return Primitive(
        type=PrimitiveType.SPHERE,
        center=centre,
        dimensions=np.full(3, max(radius, float(min_radius))),
        **fields,
    )


def fit_capsule(points: np.ndarray, *, min_radius: float = 0.0, **fields: Any) -> Primitive:
    """Segment along the principal axis, swept by the radius that reaches the furthest point.

    Tighter than a sphere for anything elongated, which is most of what a robot has to avoid: a
    cable, an arm, a bottle, a table leg. The segment spans the extreme projections onto the axis,
    so every point projects *inside* it and its distance to the segment is purely perpendicular —
    which makes `radius = max distance to segment` both exact and containing.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return fit_sphere(pts, min_radius=min_radius, **fields)

    axes, _ = _principal_axes(pts)
    axis = axes[:, 2]
    centroid = pts.mean(axis=0)
    t = (pts - centroid) @ axis
    a, b = centroid + t.min() * axis, centroid + t.max() * axis
    radius = max(float(_segment_distance(pts, a, b).max()), float(min_radius))
    return Primitive(
        type=PrimitiveType.CAPSULE,
        center=(a + b) / 2.0,
        dimensions=np.array([radius, float(t.max() - t.min()) / 2.0, radius]),
        orientation=axes,
        **fields,
    )


def fit_box(points: np.ndarray, *, min_radius: float = 0.0, **fields: Any) -> Primitive:
    """Oriented bounding box on the principal axes.

    Not the minimum-volume OBB — that needs a rotating-calipers search over the convex hull. The PCA
    box is within a few percent for the compact, roughly-convex clusters this pipeline produces, is
    O(n), and contains the cluster exactly, which is the property that matters.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if pts.shape[0] < 4:
        return fit_capsule(pts, min_radius=min_radius, **fields)

    axes, _ = _principal_axes(pts)
    local = pts @ axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    half = np.maximum((hi - lo) / 2.0, float(min_radius))
    return Primitive(
        type=PrimitiveType.BOX,
        center=axes @ ((lo + hi) / 2.0),
        dimensions=half,
        orientation=axes,
        **fields,
    )


def fit_ellipsoid(points: np.ndarray, *, min_radius: float = 0.0, **fields: Any) -> Primitive:
    """Covariance ellipsoid, inflated until it contains every point.

    The shape comes from the covariance and the *scale* from the worst point's Mahalanobis distance,
    so containment holds by construction rather than by tolerance. A true minimum-volume enclosing
    ellipsoid is tighter; `fit_ellipsoid_mvee` reuses the Khachiyan implementation that already
    exists in this workspace when a tighter fit is worth its cost.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if pts.shape[0] < 4:
        return fit_capsule(pts, min_radius=min_radius, **fields)

    axes, variances = _principal_axes(pts)
    sigma = np.sqrt(np.maximum(variances, _EPS))
    centroid = pts.mean(axis=0)
    local = (pts - centroid) @ axes
    scale = float(np.sqrt(((local / sigma) ** 2).sum(axis=1)).max())
    semi_axes = np.maximum(sigma * max(scale, _EPS), float(min_radius))
    return Primitive(
        type=PrimitiveType.ELLIPSOID,
        center=centroid,
        dimensions=semi_axes,
        orientation=axes,
        **fields,
    )


def fit_ellipsoid_mvee(points: np.ndarray, *, min_radius: float = 0.0, **fields: Any) -> Primitive:
    """Minimum-volume enclosing ellipsoid, reusing `benchmark.knows_vla`'s Khachiyan solver.

    Imported lazily and on purpose: AG3S must not depend on the KNOWS package, which is a comparison
    baseline with its own lifecycle. Falls back to the covariance fit if it is unavailable or the
    solve is degenerate.
    """
    try:
        from benchmark.knows_vla.perception.ellipsoid_fit import mvee
    except Exception:  # pragma: no cover - depends on a sibling package
        return fit_ellipsoid(points, min_radius=min_radius, **fields)

    pts = np.asarray(points, np.float64).reshape(-1, 3)
    try:
        centre, Q = mvee(pts)
        values, vectors = np.linalg.eigh(0.5 * (Q + Q.T))
    except Exception:
        return fit_ellipsoid(points, min_radius=min_radius, **fields)
    if np.linalg.det(vectors) < 0.0:
        vectors[:, 0] = -vectors[:, 0]
    semi_axes = np.maximum(np.sqrt(np.maximum(values, 0.0)), _EPS)
    # Khachiyan's method is iterative and stops at a tolerance, so its "enclosing" ellipsoid can
    # leave a few points marginally outside. Containment is this module's invariant, so the result
    # is scaled by the worst point's Mahalanobis distance before it is trusted. The scale is >= 1
    # only when the solver fell short; a converged fit is untouched.
    centre = np.asarray(centre, np.float64).reshape(3)
    local = (pts - centre) @ vectors
    scale = max(1.0, float(np.sqrt(((local / semi_axes) ** 2).sum(axis=1)).max()))
    semi_axes = np.maximum(semi_axes * scale, float(min_radius))
    return Primitive(
        type=PrimitiveType.ELLIPSOID,
        center=centre,
        dimensions=semi_axes,
        orientation=vectors,
        **fields,
    )


_FITTERS = {
    PrimitiveType.SPHERE: fit_sphere,
    PrimitiveType.CAPSULE: fit_capsule,
    PrimitiveType.BOX: fit_box,
    PrimitiveType.ELLIPSOID: fit_ellipsoid,
}

#: Spec §3.5's preference order. `fit_primitive` walks *down* it when a fit is not viable, so a
#: two-point cluster asked for a box degrades to a capsule and then to a sphere rather than raising.
PRIORITY = (PrimitiveType.SPHERE, PrimitiveType.CAPSULE, PrimitiveType.BOX, PrimitiveType.ELLIPSOID)


def fit_primitive(
    points: np.ndarray,
    primitive_type: PrimitiveType | str = PrimitiveType.SPHERE,
    *,
    min_radius: float = 0.0,
    uncertainty: float = 0.0,
    **fields: Any,
) -> Primitive:
    """Fit the requested type, degrading toward `SPHERE` if the cluster cannot support it.

    Degrading rather than raising is the safe behaviour: a candidate that fails to produce a shape
    produces no constraint at all, and a missing constraint is a collision. Every fitter here already
    delegates downward on its own, so this is mostly a single dispatch plus a guaranteed fallback.

    `uncertainty` is the perception budget — depth noise, voxel quantization, registration and
    hand-eye error — and it is added to the fitted **dimensions**, growing the shape outward. It is
    applied here and nowhere else. In particular it is not also added to `d_safe`: the same
    centimetre counted in both places doubles every detour while buying no additional safety, which
    is the failure mode of stacking margins whose provenance nobody tracks.
    """
    requested = PrimitiveType(primitive_type)
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise ValueError("cannot fit a primitive to an empty cluster")

    order = [requested] + [t for t in reversed(PRIORITY) if t is not requested]
    fitted = None
    for candidate_type in order:
        try:
            attempt = _FITTERS[candidate_type](pts, min_radius=min_radius, **fields)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.all(np.isfinite(attempt.center)) and np.all(np.isfinite(attempt.dimensions)):
            fitted = attempt
            break
    if fitted is None:
        fitted = fit_sphere(pts, min_radius=min_radius, **fields)
    return inflate(fitted, uncertainty)


def inflate(primitive: Primitive, amount: float) -> Primitive:
    """Grow a primitive outward by `amount` metres in every direction.

    For a sphere and a capsule this is the radius; for a box and an ellipsoid every half-extent. All
    four grow the surface by `amount` along its own normal, so one number means the same physical
    thing whichever shape a cluster happened to fit — which is what lets the uncertainty budget be a
    single configured value rather than four.
    """
    amount = float(amount)
    if amount <= 0.0:
        return primitive
    if primitive.type is PrimitiveType.CAPSULE:
        # (radius, half_length, radius): only the radius grows. Extending the half-length too would
        # add `amount` twice at the caps, which are already hemispheres of the grown radius.
        dims = primitive.dimensions + np.array([amount, 0.0, amount])
    else:
        dims = primitive.dimensions + amount
    return dataclasses.replace(primitive, dimensions=dims)


def containment_report(primitive: Primitive, points: np.ndarray) -> dict[str, float]:
    """How conservative a fit actually is: `containment_rate` first, `excess_volume` second.

    The order is the priority. A tight fit that leaves 2% of the cluster outside is worse than a
    loose one that holds all of it, because the excluded points are geometry the optimizer is told
    does not exist. `excess_volume_ratio` is the cost of the conservatism — the primitive's volume
    over the point cloud's own axis-aligned volume — and exists so an over-inflated shape can be
    noticed rather than merely tolerated.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return {"containment_rate": 1.0, "excess_volume_ratio": 1.0, "n_outside": 0.0}
    inside = contains(primitive, pts)
    d = primitive.dimensions
    if primitive.type is PrimitiveType.SPHERE:
        volume = 4.0 / 3.0 * np.pi * d[0] ** 3
    elif primitive.type is PrimitiveType.CAPSULE:
        volume = np.pi * d[0] ** 2 * (2.0 * d[1]) + 4.0 / 3.0 * np.pi * d[0] ** 3
    elif primitive.type is PrimitiveType.BOX:
        volume = 8.0 * float(np.prod(d))
    else:
        volume = 4.0 / 3.0 * np.pi * float(np.prod(d))
    extent = np.maximum(pts.max(axis=0) - pts.min(axis=0), _EPS)
    return {
        "containment_rate": float(inside.mean()),
        "excess_volume_ratio": float(volume / float(np.prod(extent))),
        "n_outside": float((~inside).sum()),
    }


# --------------------------------------------------------------------- constraint spheres


def to_spheres(primitive: Primitive, *, max_spheres: int = 8) -> list[tuple[np.ndarray, float]]:
    """The sphere set a sphere-versus-sphere constraint actually uses.

    A capsule becomes a chain along its segment with centres at most one radius apart, so the union
    of the spheres contains the capsule — the same construction `UrdfSphereChain` applies to the
    robot. Everything else collapses to its bounding sphere, which is conservative and, for a box or
    an ellipsoid, currently the best the constraint form can express (see the module docstring).
    """
    if primitive.type is PrimitiveType.CAPSULE:
        radius = float(primitive.dimensions[0])
        half_length = float(primitive.dimensions[1])
        axis = primitive.orientation[:, 2]
        count = int(np.clip(int(np.ceil(2.0 * half_length / max(radius, _EPS))) + 1, 1, max_spheres))
        if count == 1:
            return [(primitive.center.copy(), primitive.bounding_radius)]
        offsets = np.linspace(-half_length, half_length, count)
        # When `max_spheres` binds, the centres end up further apart than one radius and the naive
        # chain leaves gaps *between* the spheres — an under-approximation, which is the error
        # direction that causes a collision rather than a detour. Inflating each sphere to
        # sqrt(r^2 + (spacing/2)^2) is exactly enough to cover the midpoint between two centres, so
        # the union contains the capsule for any cap.
        spacing = float(offsets[1] - offsets[0])
        effective = float(np.sqrt(radius**2 + (spacing / 2.0) ** 2))
        return [(primitive.center + float(t) * axis, effective) for t in offsets]
    return [(primitive.center.copy(), primitive.bounding_radius)]


def contains(primitive: Primitive, points: np.ndarray, *, tolerance: float = 1e-9) -> np.ndarray:
    """Per-point containment test — the invariant every fit in this module has to satisfy."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    delta = pts - primitive.center
    if primitive.type is PrimitiveType.SPHERE:
        return np.linalg.norm(delta, axis=1) <= primitive.dimensions[0] + tolerance
    if primitive.type is PrimitiveType.CAPSULE:
        axis = primitive.orientation[:, 2]
        half_length = float(primitive.dimensions[1])
        a = primitive.center - half_length * axis
        b = primitive.center + half_length * axis
        return _segment_distance(pts, a, b) <= primitive.dimensions[0] + tolerance
    local = delta @ primitive.orientation
    if primitive.type is PrimitiveType.BOX:
        return np.all(np.abs(local) <= primitive.dimensions + tolerance, axis=1)
    return ((local / np.maximum(primitive.dimensions, _EPS)) ** 2).sum(axis=1) <= 1.0 + tolerance


__all__ = [
    "PRIORITY",
    "containment_report",
    "contains",
    "fit_box",
    "fit_capsule",
    "fit_ellipsoid",
    "fit_ellipsoid_mvee",
    "fit_primitive",
    "fit_sphere",
    "inflate",
    "rms_radius",
    "to_spheres",
]
