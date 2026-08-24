"""Depth back-projection and MVEE ellipsoid fitting — KNOWS §3.2.

    At episode start an instance segmentation model produces a binary mask per object; we
    back-project the masked depth into 3D through the known camera intrinsics and extrinsics,
    fuse it across the available camera views into a single point cloud, and fit a
    minimum-volume enclosing ellipsoid (MVEE) to obtain E_i.
    Crucially, all ellipsoid shape matrices Q_1..N are fixed at t = 0. At each subsequent step we
    recompute only the centroid p_i from the updated segmentation mask.

Only the agentview camera is used here: LIBERO's wrist camera has no usable extrinsics for static
scene geometry, so the "fuse across views" step is a single view in this port (Layer B deviation).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from benchmark.knows_vla.cbf.ellipsoid import Ellipsoid


def real_depth(depth_buffer: np.ndarray, znear: float, zfar: float) -> np.ndarray:
    """MuJoCo's depth buffer is normalized; convert to metres.

    Same formula as ``robosuite.utils.camera_utils.get_real_depth_map``:
        z = near / (1 - d (1 - near/far))
    """
    d = np.asarray(depth_buffer, np.float64)
    return znear / (1.0 - d * (1.0 - znear / zfar))


@dataclasses.dataclass(frozen=True)
class CameraModel:
    """Agentview intrinsics/extrinsics plus how stored pixels map to raw render pixels."""

    K: np.ndarray  # (3, 3)
    cam_to_world: np.ndarray  # (4, 4)
    znear: float
    zfar: float
    flip_row: bool = False
    flip_col: bool = True  # see `pixel_to_raw`

    @staticmethod
    def from_json_entry(entry: dict, *, flip_row: bool = False, flip_col: bool = True) -> "CameraModel":
        return CameraModel(
            K=np.asarray(entry["intrinsic"], np.float64),
            cam_to_world=np.asarray(entry["cam_to_world"], np.float64),
            znear=float(entry["znear"]),
            zfar=float(entry["zfar"]),
            flip_row=flip_row,
            flip_col=flip_col,
        )

    def pixel_to_raw(self, rows: np.ndarray, cols: np.ndarray, h: int, w: int):
        """Map indices in our stored arrays back to raw-render pixel indices.

        robosuite already stores observations vertically flipped (IMAGE_CONVENTION), and
        ``collect_p0b.py`` rotates them a further 180 degrees to match training preprocessing. Net
        effect should be a horizontal mirror, but that is a chain of assumptions, so the flips are
        configurable and ``validate_backprojection.py`` picks the combination empirically.
        """
        r = (h - 1 - rows) if self.flip_row else rows
        c = (w - 1 - cols) if self.flip_col else cols
        return r, c

    def world_to_pixel(self) -> np.ndarray:
        K_exp = np.eye(4)
        K_exp[:3, :3] = self.K
        return K_exp @ np.linalg.inv(self.cam_to_world)

    def backproject(self, rows: np.ndarray, cols: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """Stored-array pixels + metric depth -> world points, shape (n, 3).

        Mirrors ``robosuite.utils.camera_utils.transform_from_pixels_to_world``: the homogeneous
        camera vector is ``[col*z, row*z, z, 1]`` in raw-render indexing.
        """
        h, w = depth_m.shape
        r_raw, c_raw = self.pixel_to_raw(rows, cols, h, w)
        z = depth_m[rows, cols]
        cam_pts = np.stack([c_raw * z, r_raw * z, z, np.ones_like(z)], axis=-1)
        world = cam_pts @ np.linalg.inv(self.world_to_pixel()).T
        return world[:, :3]


def mvee(points: np.ndarray, tol: float = 1e-3, max_iter: int = 10_000):
    """Minimum-volume enclosing ellipsoid (Khachiyan) — the paper's [35].

    Returns ``(center, Q)`` with the convention used throughout this port,
    E = {c + Q^(1/2) u : ||u|| <= 1}, so ``Q`` is directly usable by ``cbf.ellipsoid.Ellipsoid``.
    """
    P = np.asarray(points, np.float64)
    n, d = P.shape
    if n <= d:
        raise ValueError(f"MVEE needs more than {d} points, got {n}")

    Q_lift = np.vstack([P.T, np.ones(n)])
    u = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        # Scale columns instead of building diag(u): that matrix is n x n, so materializing it
        # turns each iteration into an O(n^2) matmul (128 MB at n=4000) and the fit never finishes.
        X = (Q_lift * u) @ Q_lift.T
        M = np.einsum("ij,jk,ki->i", Q_lift.T, np.linalg.inv(X), Q_lift)
        j = int(np.argmax(M))
        step = (M[j] - d - 1.0) / ((d + 1.0) * (M[j] - 1.0))
        if step <= 0 or not np.isfinite(step):
            break
        new_u = (1.0 - step) * u
        new_u[j] += step
        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u

    c = P.T @ u
    A = np.linalg.inv((P.T * u) @ P - np.outer(c, c)) / d  # (x-c)' A (x-c) <= 1
    return c, np.linalg.inv(0.5 * (A + A.T))


def _trim_by_depth(rows, cols, depth_m, lo_pct: float, hi_pct: float):
    """Drop mask pixels whose depth is an outlier for that object.

    Segmentation masks bleed a few pixels onto whatever is in front of or behind the object. MVEE
    is a *minimum-volume enclosing* fit, so it must contain every point: a single stray pixel
    stretches the ellipsoid along the viewing ray. Measured on LIBERO, alphabet_soup_1 spans 4.8 cm
    between its 5th and 95th depth percentiles but 38 cm between min and max, which inflates a
    semi-axis to 24 cm.

    The paper specifies MVEE (its [35]) and says nothing about outlier rejection
    (OPEN-QUESTIONS). This trim is an engineering addition, not part of the method.
    """
    z = depth_m[rows, cols]
    lo, hi = np.percentile(z, [lo_pct, hi_pct])
    keep = (z >= lo) & (z <= hi)
    return rows[keep], cols[keep]


def fit_objects(
    seg: np.ndarray,
    depth_buffer: np.ndarray,
    cam: CameraModel,
    object_ids,
    *,
    min_pixels: int = 12,
    subsample: int = 4000,
    depth_trim: tuple[float, float] | None = (2.0, 98.0),
    rng: np.random.Generator | None = None,
    depth_is_metric: bool = False,
) -> dict[int, Ellipsoid]:
    """Fit one MVEE per object id from a single (segmentation, depth) frame.

    ``depth_is_metric`` for sources that already give metres -- `mujoco.Renderer` does, whereas
    robosuite hands back a normalized buffer. Converting an already-metric map produces a
    plausible-looking scene at the wrong scale rather than an error, so it has to be explicit.

    The default ``depth_trim`` is not optional in practice: silhouette pixels take the depth of
    whatever is behind the object, which stretches the fit along the line of sight. Measured on a
    4 cm sphere rendered by MuJoCo, its mask spans 26 cm of depth and the untrimmed fit comes out
    as 11 x 17 x 7 cm; trimming brings it to 4.1 x 4.1 x 4.4 cm.
    """
    rng = rng or np.random.default_rng(0)
    depth_m = np.asarray(depth_buffer, np.float64) if depth_is_metric else real_depth(
        depth_buffer, cam.znear, cam.zfar)
    out: dict[int, Ellipsoid] = {}
    for oid in object_ids:
        rows, cols = np.nonzero(seg == oid)
        if rows.size < min_pixels:
            continue
        if depth_trim is not None:
            rows, cols = _trim_by_depth(rows, cols, depth_m, *depth_trim)
            if rows.size < min_pixels:
                continue
        if rows.size > subsample:  # MVEE is superlinear in the point count
            pick = rng.choice(rows.size, subsample, replace=False)
            rows, cols = rows[pick], cols[pick]
        pts = cam.backproject(rows, cols, depth_m)
        try:
            c, Q = mvee(pts)
        except (ValueError, np.linalg.LinAlgError):
            continue
        out[int(oid)] = Ellipsoid(c, Q)
    return out


def centroid_of(seg: np.ndarray, depth_buffer: np.ndarray, cam: CameraModel, oid: int,
                *, min_pixels: int = 12,
                depth_trim: tuple[float, float] | None = (2.0, 98.0)) -> np.ndarray | None:
    """Per-step update: only the centroid moves, the shape matrix stays fixed (§3.2)."""
    rows, cols = np.nonzero(seg == oid)
    if rows.size < min_pixels:
        return None
    depth_m = real_depth(depth_buffer, cam.znear, cam.zfar)
    if depth_trim is not None:
        rows, cols = _trim_by_depth(rows, cols, depth_m, *depth_trim)
        if rows.size < min_pixels:
            return None
    return cam.backproject(rows, cols, depth_m).mean(axis=0)


class ObjectTracker:
    """Shapes frozen at t=0, centroids recomputed per step, frozen while occluded.

    The paper also re-associates identity swaps with an HSV colour-histogram Bhattacharyya match.
    That is unnecessary here because LIBERO gives ground-truth instance ids; it becomes relevant
    only once a real detector replaces them (OPEN-QUESTIONS #9).
    """

    def __init__(self, cam: CameraModel, shapes: dict[int, Ellipsoid]):
        self.cam = cam
        self._Q = {k: e.Q.copy() for k, e in shapes.items()}
        self._c = {k: e.c.copy() for k, e in shapes.items()}
        self.frozen: set[int] = set()

    def update(self, seg: np.ndarray, depth_buffer: np.ndarray) -> dict[int, Ellipsoid]:
        depth_m = real_depth(depth_buffer, self.cam.znear, self.cam.zfar)
        self.frozen.clear()
        for oid in self._Q:
            rows, cols = np.nonzero(seg == oid)
            if rows.size >= 12:
                rows, cols = _trim_by_depth(rows, cols, depth_m, 2.0, 98.0)
            if rows.size < 12:  # occluded by the arm: hold the last known position
                self.frozen.add(oid)
                continue
            self._c[oid] = self.cam.backproject(rows, cols, depth_m).mean(axis=0)
        return self.current()

    def current(self) -> dict[int, Ellipsoid]:
        return {k: Ellipsoid(self._c[k], self._Q[k]) for k in self._Q}


@dataclasses.dataclass(frozen=True)
class PerceptionMargin:
    """Inflate a fitted ellipsoid by what a single viewpoint could not see.

    An MVEE fitted to one depth frame is systematically **too small**: the camera never sees the
    far side of the object. Measured on the transport meshes, fitting only the upward-facing half
    shrinks the orange's vertical semi-axis from 3.1 cm to 2.0 cm and the pear's from 5.5 to 3.8.
    Under-approximating an obstacle is the dangerous direction, so the fit must be inflated back.

    The error is anisotropic -- it lies along the line of sight, not around it -- so a single scalar
    padding would be wrong twice over: too little along the ray, too much across it, where the mask
    boundary is the only source of error. Hence

        Q <- Q + sigma_ray^2 d d' + sigma_lat^2 (I - d d')

    with ``d`` the unit vector from the camera to the object centre.

    Note this is a *bias* correction, not noise: the unseen half is missing every frame, so
    averaging over frames does not remove it.
    """

    sigma_ray: float = 0.0
    sigma_lateral: float = 0.0

    def inflate(self, E: Ellipsoid, camera_pos) -> Ellipsoid:
        d = np.asarray(E.c, np.float64) - np.asarray(camera_pos, np.float64).reshape(3)
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-9:
            return E
        d = d / nrm
        P = np.outer(d, d)
        return Ellipsoid(E.c, E.Q + self.sigma_ray**2 * P + self.sigma_lateral**2 * (np.eye(3) - P))

    def apply(self, objects: dict, camera_pos) -> dict:
        return {k: self.inflate(v, camera_pos) for k, v in objects.items()}


def visible_half(points: np.ndarray, camera_pos) -> np.ndarray:
    """The points a camera at ``camera_pos`` could see, approximated by the near-facing half.

    Crude on purpose: it is a calibration aid for `PerceptionMargin`, not a renderer. Comparing the
    MVEE of this subset against the MVEE of the full mesh gives the shortfall the margin has to
    make up.
    """
    P = np.asarray(points, np.float64)
    d = P.mean(0) - np.asarray(camera_pos, np.float64).reshape(3)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-9:
        return P
    return P[(P - P.mean(0)) @ (d / nrm) <= 0.0]


def complete_backface(E: Ellipsoid, camera_pos) -> Ellipsoid:
    """Extend a single-view fit behind the surface the camera can see. No tuning constant.

    A depth frame gives only the near face, so an MVEE fitted to it stops around the object's
    midline: measured against the full transport meshes, a single-view fit places the far surface
    1.7-3.1 cm too close. Under-approximating an obstacle is the dangerous direction, so this has
    to be corrected, and it is a *bias* -- the far side is missing in every frame, so averaging
    frames will not remove it.

    The correction assumes the object is about as deep as it is wide, which is the weakest useful
    prior for a graspable object: hold the measured near face, and set the along-ray semi-axis to
    the observed lateral one.

        rho  = mean semi-axis of Q projected onto the plane perpendicular to the ray
        c   <- c + (rho - sqrt(d' Q d)) d
        Q   <- Q + (rho^2 - d' Q d) d d'

    Directions across the ray are untouched: those were observed.

    Accuracy on the transport meshes (far-surface error, + is conservative):
    apple +0.4 cm, orange +0.5, banana -0.5, pear -0.4; a sphere comes back to 3.19 cm from a true
    3.00. The two negatives are the elongated objects, where depth genuinely is less than width, so
    stack a small `PerceptionMargin(sigma_ray~0.005)` on top to keep the residual on the safe side.

    An earlier version doubled the along-ray semi-axis instead, on the theory that the MVEE of a
    hemispherical shell is half the sphere. It is not -- the fit bulges to cover the rim -- and that
    rule over-inflated by 0.7-2.8 cm.

    **Only for compact objects.** "As deep as it is wide" is exactly wrong for a plate: the crate's
    0.9 cm side comes back 8.3 cm thick and its 0.7 cm floor 11.3 cm, refilling the interior that
    decomposing the crate existed to open. One view cannot recover a plate's thickness -- nothing in
    the image constrains it -- so structure like a crate or shelf should come from its model, which
    is a known tracked asset anyway. See docs/16-filter-variants.md §4.
    """
    d = np.asarray(E.c, np.float64) - np.asarray(camera_pos, np.float64).reshape(3)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-9:
        return E
    d = d / nrm
    s2 = float(d @ E.Q @ d)
    perp = np.eye(3) - np.outer(d, d)
    lateral = np.sqrt(np.maximum(np.linalg.eigvalsh(perp @ E.Q @ perp), 0.0))
    rho = float(lateral[-2:].mean())  # drop the (zero) eigenvalue along d
    if rho**2 <= s2:  # already deeper than it is wide; nothing to add
        return E
    return Ellipsoid(E.c + (rho - np.sqrt(s2)) * d, E.Q + (rho**2 - s2) * np.outer(d, d))
