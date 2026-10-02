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

import dataclasses
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from benchmark.ag3s.config import KDTREE_WORKERS, AG3SConfigError, PointCloudConfig
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
    held_columns: Optional[np.ndarray] = None,
    return_parts: bool = False,
    device: Optional[str] = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | "SphereMaskParts":
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

    T38 B2: this stays the **one** entry point every self-filter decision goes through (cloud filter,
    depth robot mask, held count). `return_parts=True` returns the whole `SphereMaskParts`
    (`inside`, `protected`, and `held` for `held_columns`) from one pass. `device`: ``None`` reads
    `config.SELF_FILTER_DEVICE`; see `sphere_mask_parts` for the rule both backends share.
    """
    parts = sphere_mask_parts(points, centers, radii, inflation, guard_centre=guard_centre,
                              guard_radius=guard_radius, held_columns=held_columns, device=device)
    if return_parts:
        return parts
    return (parts.inside, parts.protected) if return_protected else parts.inside


@dataclasses.dataclass
class SphereMaskParts:
    """Per-point self-filter outcome (T38 B2). All arrays are `(N,)` bool."""

    #: The self-filter decision: inside some (inflated, or bare when guarded) sphere.
    inside: np.ndarray
    #: Kept by the target guard: inside some inflated sphere, within the guard, not `inside`.
    protected: np.ndarray
    #: Inside one of the `held_columns` spheres at its bare radius (no inflation, no guard) — the
    #: T32 H1 count. None when no held columns were named.
    held: Optional[np.ndarray]
    #: ``"cuda"`` or ``"cpu"`` — which backend computed it.
    backend: str


def sphere_mask_parts(
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    inflation: float | np.ndarray = 0.0,
    *,
    guard_centre: Optional[np.ndarray] = None,
    guard_radius: float = 0.0,
    held_columns: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> SphereMaskParts:
    """`robot_sphere_mask` and the held-sphere count in **one** pass over the points (T38 B2).

    Decision rule, identical on both backends (cuRobo `RobotSegmenter`'s point-vs-sphere test, made
    exact): with ``sq_ij = (dx·dx + dy·dy) + dz·dz`` for ``d = p_i − c_j`` in float64 and
    ``R_j = r_j + δ_j``,

    * ``hit_ij  ⇔ sq_ij ≤ R_j·R_j`` — scipy `query_ball_point`'s own arithmetic, so the GPU path and
      the KD-tree path agree to the bit, boundary points included;
    * ``guarded_i ⇔ ‖p_i − g‖ ≤ ρ`` (``np.linalg.norm`` order, ``sqrt((x² + y²) + z²)``), only when
      `guard_centre` is given, ``ρ > 0`` and at least one point is guarded;
    * ``inside_i ⇔ ∃j hit_ij``, except guarded points: ``∃j hit_ij ∧ sqrt(sq_ij) ≤ r_j``;
    * ``protected_i ⇔ guarded_i ∧ (∃j hit_ij) ∧ ¬inside_i``;
    * ``held_i ⇔ ∃j∈held_columns sq_ij ≤ r_j·r_j`` (bare radius, no guard — T32 H1's count).

    Every value is a pure function of the point and the spheres, so a subset of points (a voxel
    cloud's representatives) gets exactly the decisions its members got in the full set.

    `device`: ``None`` reads `config.SELF_FILTER_DEVICE` (``"auto" | "cuda" | "cpu"``).

    T38 B4: `points` may also be an ``(N, 3)`` float64 **CUDA tensor** (a back-projection already on the
    GPU, `device_recon`). Then the same GPU kernel runs where the points are and the parts come back
    as CUDA bool tensors (`backend == "cuda"`); `device` is not consulted.
    """
    if _is_cuda_tensor(points):
        return _sphere_mask_parts_on_device(points, centers, radii, inflation, guard_centre,
                                            guard_radius, held_columns)
    points = np.asarray(points, np.float64).reshape(-1, 3)
    centers = np.asarray(centers, np.float64).reshape(-1, 3)
    radii = np.asarray(radii, np.float64).reshape(-1)
    if centers.shape[0] != radii.shape[0]:
        raise ValueError(f"{centers.shape[0]} sphere centres but {radii.shape[0]} radii")
    if np.ndim(inflation) and np.shape(inflation) != radii.shape:
        raise ValueError(
            f"inflation has shape {np.shape(inflation)} but there are {radii.shape[0]} spheres")
    cols = None if held_columns is None else np.asarray(held_columns, np.intp).reshape(-1)
    torch_device = _torch_device(device)
    backend = "cpu" if torch_device is None else "cuda"
    n = points.shape[0]
    if n == 0 or centers.shape[0] == 0:
        empty = np.zeros(n, bool)
        return SphereMaskParts(empty, empty.copy(), None if cols is None else empty.copy(), backend)

    guard = None
    if guard_centre is not None and float(guard_radius) > 0.0:
        guard = (np.asarray(guard_centre, np.float64).reshape(3), float(guard_radius))
    if torch_device is None:
        inside, protected = _robot_sphere_mask_kdtree(points, centers, radii, inflation, guard)
        held = None
        if cols is not None:
            held, _ = _robot_sphere_mask_kdtree(points, centers[cols], radii[cols], 0.0, None)
        return SphereMaskParts(inside, protected, held, backend)
    inside, protected, held = _sphere_mask_torch(
        points, centers, radii, radii + inflation, guard, cols, torch_device)
    return SphereMaskParts(inside, protected, held, backend)


def _is_cuda_tensor(x) -> bool:
    return type(x).__module__.startswith("torch") and getattr(getattr(x, "device", None), "type", "") == "cuda"


def _sphere_mask_parts_on_device(points, centers, radii, inflation, guard_centre, guard_radius,
                                 held_columns) -> SphereMaskParts:
    """`sphere_mask_parts` for points resident on the GPU (T38 B4) — same checks, same rule."""
    import torch

    if points.dtype != torch.float64 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"device points must be (N, 3) float64, got {tuple(points.shape)} {points.dtype}")
    centers = np.asarray(centers, np.float64).reshape(-1, 3)
    radii = np.asarray(radii, np.float64).reshape(-1)
    if centers.shape[0] != radii.shape[0]:
        raise ValueError(f"{centers.shape[0]} sphere centres but {radii.shape[0]} radii")
    if np.ndim(inflation) and np.shape(inflation) != radii.shape:
        raise ValueError(
            f"inflation has shape {np.shape(inflation)} but there are {radii.shape[0]} spheres")
    cols = None if held_columns is None else np.asarray(held_columns, np.intp).reshape(-1)
    guard = None
    if guard_centre is not None and float(guard_radius) > 0.0:
        guard = (np.asarray(guard_centre, np.float64).reshape(3), float(guard_radius))
    inside, protected, held = _sphere_mask_device(points, centers, radii, radii + inflation, guard, cols)
    return SphereMaskParts(inside, protected, held, "cuda")


def _robot_sphere_mask_kdtree(points, centers, radii, inflation, guard):
    """The CPU backend: one `query_ball_point` per sphere (the pre-T38 implementation, unchanged).

    Each sphere has its own radius, so one ball query cannot answer all of them at once; the loop is
    over the small set (spheres), not the points. Returns ``(inside, protected)``.
    """
    from scipy.spatial import cKDTree

    inside = np.zeros(points.shape[0], bool)
    guarded = None
    if guard is not None:
        centre_g, radius_g = guard
        guarded = np.linalg.norm(points - centre_g, axis=1) <= radius_g
        if not guarded.any():
            guarded = None
    touched = np.zeros(points.shape[0], bool) if guarded is not None else None

    tree = cKDTree(points)
    for centre, bare, radius in zip(centers, radii, radii + inflation):
        # Exact rather than bounded: each sphere is queried at its own radius, so nothing relies on a
        # shared upper bound and no second distance test is needed.
        hits = tree.query_ball_point(centre, float(radius), workers=KDTREE_WORKERS)
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
    protected = np.zeros_like(inside) if touched is None else (touched & ~inside)
    return inside, protected


_TORCH_STATE: dict[str, Any] = {}


def _torch_device(device: Optional[str]):
    """The torch CUDA device the GPU backend runs on, or None for the CPU backend."""
    from benchmark.ag3s import config as _config

    choice = str(device if device is not None else _config.SELF_FILTER_DEVICE).lower()
    if choice not in ("auto", "cuda", "cpu"):
        raise AG3SConfigError(f"self-filter device must be auto|cuda|cpu, got {choice!r}")
    if choice == "cpu":
        return None
    if "available" not in _TORCH_STATE:
        try:
            import torch

            _TORCH_STATE["available"] = bool(torch.cuda.is_available())
        except ImportError:
            _TORCH_STATE["available"] = False
    if not _TORCH_STATE["available"]:
        if choice == "cuda":
            raise RuntimeError("SELF_FILTER_DEVICE='cuda' but torch sees no CUDA device")
        return None
    import torch

    return torch.device("cuda", torch.cuda.current_device())


def _sphere_mask_torch(points, centers, radii, inflated, guard, cols, device):
    """The GPU backend of `sphere_mask_parts`: (points × spheres) in float64, point axis chunked.

    The squared distance is built one elementwise kernel at a time — ``d = px − cx; d *= d;
    t = py − cy; t *= t; d += t; …`` — so nothing is fused into an FMA and every rounding is the one
    numpy/scipy perform. Returns ``(inside, protected, held)`` as numpy bool arrays.
    """
    import torch

    P = torch.as_tensor(np.ascontiguousarray(points), device=device)
    inside, protected, held = _sphere_mask_device(P, centers, radii, inflated, guard, cols)
    out_held = None if held is None else held.cpu().numpy()
    return inside.cpu().numpy(), protected.cpu().numpy(), out_held


def _sphere_mask_device(P, centers, radii, inflated, guard, cols):
    """`_sphere_mask_torch` on points already resident on the GPU (T38 B4): ``(N, 3)`` float64 tensor
    in, ``(inside, protected, held)`` bool **tensors** out. Same kernels, same order."""
    import torch

    from benchmark.ag3s import config as _config

    device = P.device
    n, s = int(P.shape[0]), centers.shape[0]
    if n == 0 or s == 0:
        empty = torch.zeros(n, dtype=torch.bool, device=device)
        return empty, empty.clone(), None if cols is None else empty.clone()
    C = torch.as_tensor(np.ascontiguousarray(centers.T), device=device)          # (3, S)
    R2 = torch.as_tensor(inflated * inflated, device=device)                      # scipy: r * r
    bare = torch.as_tensor(np.ascontiguousarray(radii), device=device)
    inside = torch.empty(n, dtype=torch.bool, device=device)
    held = None
    if cols is not None:
        cols_t = torch.as_tensor(cols, device=device)
        r2_held = torch.as_tensor(radii[cols] * radii[cols], device=device)
        held = torch.zeros(n, dtype=torch.bool, device=device)

    guarded_idx = None
    if guard is not None:
        g = torch.as_tensor(guard[0], device=device)
        dg = P - g
        sqg = dg[:, 0] * dg[:, 0]
        sqg += dg[:, 1] * dg[:, 1]
        sqg += dg[:, 2] * dg[:, 2]
        guarded_idx = torch.nonzero(torch.sqrt(sqg) <= guard[1]).reshape(-1).cpu().numpy()
        if guarded_idx.size == 0:
            guarded_idx = None
    protected = torch.zeros(n, dtype=torch.bool, device=device)

    rows = max(1, int(_config.SELF_FILTER_GPU_CHUNK_BYTES) // (8 * s))
    d_buf = torch.empty((min(rows, n), s), dtype=torch.float64, device=device)
    t_buf = torch.empty_like(d_buf)
    for a in range(0, n, rows):
        b = min(n, a + rows)
        p = P[a:b]
        d, t = d_buf[: b - a], t_buf[: b - a]
        torch.sub(p[:, 0:1], C[0], out=d)
        d.mul_(d)
        torch.sub(p[:, 1:2], C[1], out=t)
        t.mul_(t)
        d.add_(t)
        torch.sub(p[:, 2:3], C[2], out=t)
        t.mul_(t)
        d.add_(t)
        hit = d <= R2
        touched = hit.any(dim=1)
        inside[a:b] = touched
        if held is not None:
            held[a:b] = (d.index_select(1, cols_t) <= r2_held).any(dim=1)
        if guarded_idx is not None:
            lo, hi = np.searchsorted(guarded_idx, [a, b])
            if hi > lo:
                local = torch.as_tensor(guarded_idx[lo:hi] - a, device=device)
                dd = d.index_select(0, local)
                kept = (hit.index_select(0, local) & (torch.sqrt(dd) <= bare)).any(dim=1)
                was = touched.index_select(0, local)
                inside[a:b].index_copy_(0, local, kept)
                protected[a:b].index_copy_(0, local, was & ~kept)
    return inside, protected, held


@dataclasses.dataclass
class DepthRobotMask:
    """One camera's self-filter, decided **once** per back-projected pixel (T38 B2).

    The TSDF reads `mask` directly. The voxel cloud cut from the same back-projection reads it too:
    a voxel representative *is* one of the back-projected points (`voxel_downsample` keeps the
    first point of each voxel, with its pixel), so its keep/drop is that pixel's decision — the same
    rule as testing the representative itself, which is what the cloud filter always did
    (`filter_cloud`). Since every decision is a pure function of the point (`sphere_mask_parts`),
    the two are equal, not merely close.
    """

    #: `(H, W)` True where the pixel's back-projected point is the robot (TSDF depth mask).
    mask: np.ndarray
    #: `(H, W)` True where the target guard kept the pixel (see `SphereMaskParts.protected`).
    #: T38 B4: None when decided on the device (no host reader needs it on that path — the
    #: counts are taken on the GPU); `protected_mask()` pulls it on demand.
    protected: Optional[np.ndarray]
    #: Pixels inside the held object's spheres (T32 H1), or None when the model holds nothing.
    n_held_px: Optional[int]
    #: The guard fields `self_filter_mask` reports (`n_guard_protected` = protected pixels).
    guard: dict[str, Any]
    backend: str
    #: T38 B4: the GPU-resident side when the mask was decided on device points
    #: (`depth_robot_mask_device`) — per-point `inside` / `protected` of the back-projection, the
    #: `(H, W)` mask tensor and the device depth in metres. None on the host paths.
    device: Any = None

    def masked_depth_f32_device(self):
        """The TSDF's depth on the GPU: ``float32(depth_m)`` with robot pixels set to 0 — the same
        values `CuroboFieldBuilder._observations` makes on the host (``np.where(mask, 0, f32(d))``)
        from the same depth, without uploading it again. None without the device side."""
        if self.device is None:
            return None
        import torch

        return self.device.depth_m.to(torch.float32).masked_fill_(self.device.mask, 0.0)

    @property
    def n_guard_protected_px(self) -> int:
        return int(self.guard["n_guard_protected"])

    def protected_mask(self) -> np.ndarray:
        """`protected` on the host, pulled from the device side when it was decided there (B4)."""
        if self.protected is not None:
            return self.protected
        return self.device.protected_px.cpu().numpy()

    def filter_cloud(self, cloud: PointCloud) -> tuple[PointCloud, dict[str, Any]]:
        """`filter_robot_points` for a cloud cut from **this** back-projection (needs its `uv`).

        Returns the same ``(cloud, stats)`` — `n_guard_protected` counts this cloud's points.
        """
        n_in = len(cloud)
        if cloud.uv is None:
            raise ValueError("DepthRobotMask.filter_cloud needs the cloud's (u, v)")
        u, v = cloud.uv[:, 0], cloud.uv[:, 1]
        inside = self.mask[v, u]
        kept = np.nonzero(~inside)[0]
        return cloud.select(kept), {
            "n_in": n_in,
            "n_removed": int(inside.sum()),
            "n_out": int(kept.size),
            "enabled": True,
            **{**self.guard, "n_guard_protected": int(self.protected_mask()[v, u].sum())},
        }


def depth_robot_mask(
    cloud: PointCloud,
    image_hw: tuple[int, int],
    robot_model: RobotCollisionModel,
    robot_state: np.ndarray,
    config: PointCloudConfig,
    *,
    inflation: float | np.ndarray | None = None,
    guard_centre: Optional[np.ndarray] = None,
    device: Optional[str] = None,
    fk_cache: Optional[dict] = None,
) -> DepthRobotMask:
    """The self-filter of one camera's full-resolution back-projection `cloud`, as pixel masks.

    One `sphere_mask_parts` pass over the model's spheres — the robot's and, while attached, the
    held object's (`HeldSphereFilterModel` appends them) — gives the self-filter decision, the guard
    count and the held-sphere count together. `inflation` / `guard_centre`: as `self_filter_mask`.

    `fk_cache` (T38 B4): a dict shared by the cameras of one frame; the sphere FK of a `(model, q)`
    pair is computed once and reused (`_mask_spheres`).
    """
    q = np.asarray(robot_state, np.float64)
    centers, radii, held_columns, extra = _mask_spheres(robot_model, q, fk_cache)
    if inflation is None:
        inflation = resolve_self_filter_inflation(_link_names_or_none(robot_model, config), config)
    radius = float(config.self_filter_target_guard_radius)
    active = radius > 0.0 and guard_centre is not None

    parts = robot_sphere_mask(
        cloud.points, centers, radii, inflation,
        guard_centre=guard_centre if active else None, guard_radius=radius,
        held_columns=held_columns, return_parts=True, device=device)
    n_held = None if held_columns is None else int(parts.held.sum())
    if extra is not None:
        n_held = int(robot_sphere_mask(cloud.points, extra[0], extra[1], 0.0, device=device).sum())

    h, w = int(image_hw[0]), int(image_hw[1])
    mask = np.zeros((h, w), bool)
    protected = np.zeros((h, w), bool)
    uv = cloud.uv
    mask[uv[parts.inside, 1], uv[parts.inside, 0]] = True
    protected[uv[parts.protected, 1], uv[parts.protected, 0]] = True
    return DepthRobotMask(
        mask=mask, protected=protected, n_held_px=n_held,
        guard={
            "guard_enabled": radius > 0.0,
            "guard_active": bool(active),
            "guard_radius": radius,
            "guard_centroid": (None if not active
                               else [float(x) for x in np.asarray(guard_centre).reshape(3)]),
            "n_guard_protected": int(parts.protected.sum()),
        },
        backend=parts.backend)


def _mask_spheres(robot_model, q: np.ndarray, fk_cache: Optional[dict] = None):
    """``(centers, radii, held_columns, extra)`` for the depth robot mask at `q`.

    `held_columns`: the model's own trailing spheres when they are exactly the held object's
    (`HeldSphereFilterModel` appends them), else None and `extra = (hc, hr)` for a separate count.

    With `fk_cache` the result is memoised per ``(id(robot_model), q bytes)`` — FK is a pure function
    of `q`, and the three cameras of a frame are usually captured at the same `q` (T38 B4: all 75
    chunks of the T38 replay). The cache must not outlive the models it keys on (one frame).
    """
    key = None
    if fk_cache is not None:
        key = (id(robot_model), np.ascontiguousarray(q, np.float64).tobytes())
        hit = fk_cache.get(key)
        if hit is not None and hit[0] is robot_model:
            return hit[1]
    centers, radii = robot_model.sphere_centers_numeric(q)
    centers = np.asarray(centers, np.float64).reshape(-1, 3)
    radii = np.asarray(radii, np.float64).reshape(-1)
    held_columns, extra = None, None
    held_fn = getattr(robot_model, "held_spheres", None)
    if callable(held_fn):
        hc, hr = held_fn(q)
        hc = np.asarray(hc, np.float64).reshape(-1, 3)
        hr = np.asarray(hr, np.float64).reshape(-1)
        k = hr.shape[0]
        if (k <= radii.shape[0] and np.array_equal(centers[radii.shape[0] - k:], hc)
                and np.array_equal(radii[radii.shape[0] - k:], hr)):
            held_columns = np.arange(radii.shape[0] - k, radii.shape[0])   # the appended spheres
        else:
            extra = (hc, hr)
    out = (centers, radii, held_columns, extra)
    if key is not None:
        fk_cache[key] = (robot_model, out)
    return out


@dataclasses.dataclass
class _DeviceMaskSide:
    inside: Any       # (N,) bool tensor, per back-projected point
    protected: Any    # (N,) bool tensor
    mask: Any         # (H, W) bool tensor
    protected_px: Any  # (H, W) bool tensor
    depth_m: Any      # (H, W) float64 tensor, depth × depth_scale


def depth_robot_mask_device(
    full: Any,
    robot_model: RobotCollisionModel,
    robot_state: np.ndarray,
    config: PointCloudConfig,
    *,
    inflation: float | np.ndarray | None = None,
    guard_centre: Optional[np.ndarray] = None,
    fk_cache: Optional[dict] = None,
) -> DepthRobotMask:
    """`depth_robot_mask` for a back-projection resident on the GPU (`device_recon.DeviceCloud`, T38 B4).

    Same spheres (`_mask_spheres`), same kernel (`_sphere_mask_device`, the B2 rule) on the points where
    they already are — no 12 MB point upload. The `(H, W)` masks come back to the host (the host
    `CameraDepth.robot_mask` and the counts read them); the per-point decisions stay on the device for
    the voxel cloud cut (`device_recon.camera_front_end`).
    """
    import torch

    q = np.asarray(robot_state, np.float64)
    centers, radii, held_columns, extra = _mask_spheres(robot_model, q, fk_cache)
    if inflation is None:
        inflation = resolve_self_filter_inflation(_link_names_or_none(robot_model, config), config)
    radius = float(config.self_filter_target_guard_radius)
    active = radius > 0.0 and guard_centre is not None
    # Through the one seam every self-filter decision takes (`robot_sphere_mask`), with the points
    # where they are: a CUDA tensor in, CUDA bool tensors out.
    parts = robot_sphere_mask(
        full.points, centers, radii, inflation,
        guard_centre=guard_centre if active else None, guard_radius=radius,
        held_columns=held_columns, return_parts=True)
    inside, protected, held = parts.inside, parts.protected, parts.held
    if extra is not None:
        held = robot_sphere_mask(full.points, extra[0], extra[1], 0.0)

    h, w = full.image_hw
    dev = full.points.device
    mask = torch.zeros((h, w), dtype=torch.bool, device=dev)
    prot = torch.zeros((h, w), dtype=torch.bool, device=dev)
    mask[full.rows[inside], full.cols[inside]] = True
    prot[full.rows[protected], full.cols[protected]] = True
    counts = torch.stack([protected.sum(), (held.sum() if held is not None
                                            else torch.zeros((), dtype=torch.int64, device=dev))])
    n_prot, n_held = (int(x) for x in counts.tolist())
    return DepthRobotMask(
        mask=mask.cpu().numpy(), protected=None,
        n_held_px=None if held is None else n_held,
        guard={
            "guard_enabled": radius > 0.0,
            "guard_active": bool(active),
            "guard_radius": radius,
            "guard_centroid": (None if not active
                               else [float(x) for x in np.asarray(guard_centre).reshape(3)]),
            "n_guard_protected": n_prot,
        },
        backend="cuda",
        device=_DeviceMaskSide(inside, protected, mask, prot, full.depth_m))


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
    "DepthRobotMask",
    "SphereMaskParts",
    "depth_robot_mask",
    "depth_robot_mask_device",
    "filter_robot_points",
    "inflation_link_groups",
    "inflation_table",
    "resolve_self_filter_inflation",
    "robot_sphere_mask",
    "self_filter_mask",
    "sphere_mask_parts",
]
