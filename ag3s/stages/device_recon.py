"""T38 B4 — one camera's scene reconstruction on the GPU, with the numpy path's exact bits.

`multiview.process_observation` (fused ESDF path) used to do, per camera, in numpy: back-project the full
depth image (≈ 170 k points), voxel-downsample it (`np.unique` over packed keys), cap it, upload the full
cloud (12 MB) for the GPU self-filter (T38 B2), and cut the voxel cloud with the pixel mask. Here the depth
image goes up **once** (2.4 MB, float64) and everything up to the filtered voxel cloud happens on the
device; only what the CPU consumers need comes back — the kept voxel cloud (points + pixel), the `(H, W)`
robot / guard pixel masks and a handful of counts. The same device depth, masked, is what the cuRobo TSDF
integrates (`DepthRobotMask.masked_depth_f32_device`), so there is one H2D per camera.

**Why the bits are the same** — each step against the numpy code it replaces:

=====================  ==========================================  =========================================
step                   numpy (`reconstruction.py`)                  here
=====================  ==========================================  =========================================
metres                 ``depth * depth_scale`` (f64)                same op, one rounding
valid                  ``isfinite & > 0 & >= min & <= max``        same comparisons against the same doubles
pixels                 ``np.nonzero(valid)`` (row-major)            ``torch.nonzero`` (row-major)
camera x / y           ``(col - cx) * z / fx``                      ``__ddiv_rn(__dmul_rn(__dsub_rn(col, cx), z), fx)``
radial gate            ``norm(axis=1) = sqrt((x² + y²) + z²)``      same order, separate kernels (B2)
base frame             ``pts_cam @ R.T + t`` (OpenBLAS dgemm)       ``__dadd_rn(fma(z, r2, fma(y, r1, x·r0)), t)``
voxel key              ``floor(p / vs)`` → int64, shift, pack      same; the divisor is a CUDA value (*)
representative         ``np.unique(return_index)`` = first index    ``scatter_reduce(amin)`` of the index
cap                    `coverage_preserving_cap` / `cap_points`     the same loop on index sets
self-filter            `sphere_mask_parts` (B2 rule)                the same GPU kernel, on resident points
=====================  ==========================================  =========================================

(*) torch CUDA rewrites ``tensor / python_scalar`` as ``tensor * (1 / scalar)`` (B3: 133 k of 1 M values off
by one ulp). Every division here has its denominator as a CUDA value or inside a jiterator kernel.

The dgemm row is a measured fact about the BLAS numpy calls, not a definition: on this machine (numpy
1.26.4 · OpenBLAS 0.3.23 · Xeon 6767P) ``(N, 3) @ (3, m)`` for N ≥ 2, m ≥ 2 is the k-ordered FMA chain,
bit for bit, at every size tried (`tests/o4/test_gpu_recon.py` re-measures it on every run, so a different
BLAS shows up as a failing test, not as silent drift). N = 1 is a gemv in numpy and rounds differently —
a camera with fewer than two valid pixels takes the numpy path (`camera_front_end` returns None).

Every kernel is a separate elementwise op or an explicit ``__*_rn`` intrinsic, so nothing is contracted
into an FMA that numpy does not perform.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np

from benchmark.ag3s.config import AG3SConfigError, PointCloudConfig
from benchmark.ag3s.types import PointCloud

_STATE: dict[str, Any] = {}

#: jiterator kernels (compiled once per process, on first use). Double only — the intrinsics are the
#: double-precision round-to-nearest ones, which is what pins the arithmetic.
_KERNELS = {
    # numpy: (pix - c) * z / f, evaluated left to right, one rounding per op.
    "cam_axis": ("template <typename T> T cam_axis(T pix, T z, T c, T f) "
                 "{ return __ddiv_rn(__dmul_rn(__dsub_rn(pix, c), z), f); }",
                 dict(c=0.0, f=1.0)),
    # numpy: (pts_cam @ R.T)[:, j] + t[j] — dgemm's k-ordered FMA chain, then the broadcast add.
    "rot_row": ("template <typename T> T rot_row(T x, T y, T z, T a, T b, T c, T t) "
                "{ return __dadd_rn(__fma_rn(z, c, __fma_rn(y, b, __dmul_rn(x, a))), t); }",
                dict(a=0.0, b=0.0, c=0.0, t=0.0)),
    # numpy: (pts @ N.T)[i, j] for the RANSAC scoring (same dgemm chain, no offset).
    "dot3": ("template <typename T> T dot3(T x, T y, T z, T a, T b, T c) "
             "{ return __fma_rn(z, c, __fma_rn(y, b, __dmul_rn(x, a))); }",
             {}),
}


def kernel(name: str):
    """The jiterator function `name` (compiled lazily, cached)."""
    fn = _STATE.get(("kernel", name))
    if fn is None:
        from torch.cuda import jiterator

        code, kwargs = _KERNELS[name]
        fn = jiterator._create_jit_fn(code, **kwargs)
        _STATE[("kernel", name)] = fn
    return fn


def torch_device(choice: Optional[str], *, what: str):
    """The CUDA device for ``"auto" | "cuda" | "cpu"``, or None for the numpy path.

    Same contract as `robot_filter._torch_device`: ``auto`` falls back to numpy without torch/CUDA or
    without a working jiterator (it needs NVRTC); ``cuda`` raises instead.
    """
    choice = str(choice).lower()
    if choice not in ("auto", "cuda", "cpu"):
        raise AG3SConfigError(f"{what} device must be auto|cuda|cpu, got {choice!r}")
    if choice == "cpu":
        return None
    if "available" not in _STATE:
        ok = False
        try:
            import torch

            if torch.cuda.is_available():
                from torch.cuda import jiterator  # noqa: F401  (private but stable since 1.13)

                ok = True
        except ImportError:
            ok = False
        _STATE["available"] = ok
    if not _STATE["available"]:
        if choice == "cuda":
            raise RuntimeError(f"{what} device 'cuda' but torch/jiterator sees no CUDA device")
        return None
    import torch

    return torch.device("cuda", torch.cuda.current_device())


def recon_device():
    from benchmark.ag3s import config as _config

    return torch_device(_config.RECON_DEVICE, what="RECON")


# ------------------------------------------------------------------------------ back-projection


@dataclasses.dataclass
class DeviceCloud:
    """A full-resolution back-projection resident on the GPU (B4)."""

    points: Any        # (N, 3) float64, base frame
    rows: Any          # (N,) int64  — v
    cols: Any          # (N,) int64  — u
    depth_m: Any       # (H, W) float64 — depth × depth_scale (what the TSDF reads, before masking)
    image_hw: tuple[int, int]

    def __len__(self) -> int:
        return int(self.points.shape[0])


def upload_depth(depth: np.ndarray, device):
    """One H2D of the observation's depth, as float64 (what `backproject` reads)."""
    import torch

    d = np.asarray(depth, np.float64)
    if d.ndim != 2:
        raise ValueError(f"depth must be (H, W), got shape {d.shape}")
    return torch.as_tensor(np.ascontiguousarray(d), device=device)


def backproject_device(depth_dev, camera_intrinsics: np.ndarray, T_base_cam: np.ndarray,
                       config: PointCloudConfig | None = None) -> Optional[DeviceCloud]:
    """`reconstruction.backproject` on a device depth image. None when fewer than two points survive.

    None is the caller's cue to run the numpy path instead: with 0 points there is nothing to do and
    with 1 point numpy multiplies by the rotation through gemv, whose rounding this does not mirror.
    """
    import torch

    from benchmark.ag3s.stages.reconstruction import _check_intrinsics, _check_transform

    cfg = config or PointCloudConfig()
    fx, fy, cx, cy = _check_intrinsics(camera_intrinsics)
    T = _check_transform(T_base_cam, "T_base_cam")
    d = depth_dev * float(cfg.depth_scale)
    valid = torch.isfinite(d) & (d > 0.0) & (d >= float(cfg.depth_min)) & (d <= float(cfg.depth_max))
    rc = torch.nonzero(valid)
    if rc.shape[0] < 2:
        return None
    rows, cols = rc[:, 0].contiguous(), rc[:, 1].contiguous()
    z = d[rows, cols]
    axis = kernel("cam_axis")
    x = axis(cols.to(torch.float64), z, c=cx, f=fx)
    y = axis(rows.to(torch.float64), z, c=cy, f=fy)

    if cfg.range_max is not None:
        s = x * x
        s += y * y
        s += z * z
        within = torch.sqrt(s) <= float(cfg.range_max)
        keep = torch.nonzero(within).reshape(-1)
        if keep.shape[0] < 2:
            return None
        x, y, z, rows, cols = x[keep], y[keep], z[keep], rows[keep], cols[keep]

    rot = kernel("rot_row")
    R, t = T[:3, :3], T[:3, 3]
    pts = torch.stack([rot(x, y, z, a=float(R[j, 0]), b=float(R[j, 1]), c=float(R[j, 2]),
                           t=float(t[j])) for j in range(3)], dim=1)
    return DeviceCloud(pts, rows, cols, d, (int(depth_dev.shape[0]), int(depth_dev.shape[1])))


# ------------------------------------------------------------------------------ voxel downsample


def voxel_first_device(points, voxel_size: float):
    """`reconstruction.voxel_downsample`'s selection: the first index (input order) of each occupied
    voxel, ascending. Returns an int64 index tensor into `points`."""
    import torch

    n = int(points.shape[0])
    dev = points.device
    if voxel_size <= 0.0 or n == 0:
        return torch.arange(n, device=dev)
    vs = torch.tensor(float(voxel_size), dtype=torch.float64, device=dev)
    keys = torch.floor(points / vs).to(torch.int64)
    keys -= keys.min(dim=0).values
    extent = (keys.max(dim=0).values + 1).tolist()
    if float(extent[0]) * float(extent[1]) * float(extent[2]) >= 2.0**62:
        raise ValueError(
            f"voxel grid {tuple(extent)} overflows int64; voxel_size={voxel_size} is too small for this cloud"
        )
    packed = (keys[:, 0] * int(extent[1]) + keys[:, 1]) * int(extent[2]) + keys[:, 2]
    uniq, inverse = torch.unique(packed, sorted=True, return_inverse=True)
    first = torch.full((int(uniq.shape[0]),), n, dtype=torch.int64, device=dev)
    first.scatter_reduce_(0, inverse, torch.arange(n, device=dev), reduce="amin", include_self=True)
    return torch.sort(first).values


def downsample_and_cap_device(points, config: PointCloudConfig | None = None):
    """`reconstruction.downsample_and_cap` on a device cloud: ``(index tensor, stats)``.

    The index selects the surviving points of `points` in input order; `stats` is the same dict
    (same keys, same values) the numpy function returns.
    """
    import torch

    cfg = config or PointCloudConfig()
    n_raw = int(points.shape[0])
    idx = voxel_first_device(points, cfg.voxel_size)
    n_voxel = int(idx.shape[0])
    if cfg.cap_strategy == "voxel":
        # `coverage_preserving_cap`, step for step (the voxel grows from the configured size).
        stats = {"capped": False, "covered": True, "voxel_size": float(cfg.voxel_size),
                 "n_growth_steps": 0}
        if n_voxel > cfg.max_points:
            stats["capped"] = True
            voxel = float(cfg.voxel_size) if cfg.voxel_size > 0.0 else 0.001
            working = idx
            done = False
            for step in range(1, int(cfg.cap_max_iterations) + 1):
                voxel *= float(cfg.cap_voxel_growth)
                working = working[voxel_first_device(points[working], voxel)]
                stats.update(voxel_size=voxel, n_growth_steps=step)
                if int(working.shape[0]) <= cfg.max_points:
                    done = True
                    break
            if not done:
                stats["covered"] = False
            idx = working
    else:
        # `cap_points`: deterministic index selection over the voxel cloud.
        capped = False
        if cfg.max_points < 1:
            raise ValueError(f"max_points must be >= 1, got {cfg.max_points}")
        if n_voxel > cfg.max_points:
            sel = np.unique(np.linspace(0, n_voxel - 1, cfg.max_points).astype(np.int64))
            idx = idx[torch.as_tensor(sel, device=idx.device)]
            capped = True
        stats = {"capped": capped, "covered": not capped, "voxel_size": float(cfg.voxel_size),
                 "n_growth_steps": 0}
    return idx, {
        "n_raw": n_raw, "n_voxel": n_voxel, "n_final": int(idx.shape[0]),
        "capped": bool(stats["capped"]), "covered": bool(stats["covered"]),
        "final_voxel_size": float(stats["voxel_size"]),
        "n_growth_steps": int(stats["n_growth_steps"]),
    }


# ------------------------------------------------------------------------------ one camera


@dataclasses.dataclass
class FrontEnd:
    """What `camera_front_end` hands back to `process_observation`."""

    cloud: PointCloud              # the filtered voxel cloud, on the host
    recon_stats: dict
    filter_stats: dict
    depth_mask: Any                # `robot_filter.DepthRobotMask` (host masks + device extras)
    points_device: Any             # (K, 3) float64 — `cloud.points` on the GPU (fusion keys)


def camera_front_end(depth: np.ndarray, camera_intrinsics: np.ndarray, T_base_cam: np.ndarray,
                     config: PointCloudConfig, robot_model, q: np.ndarray, *, inflation=None,
                     guard_centre=None, fk_cache: Optional[dict] = None, frame_id: str = "base",
                     device=None) -> Optional[FrontEnd]:
    """`backproject` → `downsample_and_cap` → `depth_robot_mask` → `filter_cloud`, on the GPU.

    Returns None (→ numpy path) when the back-projection keeps fewer than two points. The result equals
    `multiview.process_observation`'s numpy branch: the same cloud (points · pixels · order), stats and
    pixel masks.
    """
    import torch

    from benchmark.ag3s.stages.robot_filter import depth_robot_mask_device

    dev = device if device is not None else recon_device()
    depth_dev = upload_depth(depth, dev)
    full = backproject_device(depth_dev, camera_intrinsics, T_base_cam, config)
    if full is None:
        return None
    rep, recon_stats = downsample_and_cap_device(full.points, config)
    mask = depth_robot_mask_device(full, robot_model, q, config, inflation=inflation,
                                   guard_centre=guard_centre, fk_cache=fk_cache)
    # `DepthRobotMask.filter_cloud`: a representative is dropped iff its pixel is masked, and its
    # pixel's decision is its own (the (u, v) → point map of one back-projection is 1:1).
    inside_rep = mask.device.inside[rep]
    kept = rep[~inside_rep]
    counts = torch.stack([inside_rep.sum(), mask.device.protected[rep].sum()]).tolist()
    pts = full.points[kept]
    uv = torch.stack([full.cols[kept], full.rows[kept]], dim=1).to(torch.int32)
    cloud = PointCloud(pts.cpu().numpy(), uv.cpu().numpy(), frame_id)
    filter_stats = {
        "n_in": int(rep.shape[0]),
        "n_removed": int(counts[0]),
        "n_out": int(kept.shape[0]),
        "enabled": True,
        **{**mask.guard, "n_guard_protected": int(counts[1])},
    }
    return FrontEnd(cloud, recon_stats, filter_stats, mask, pts)


__all__ = [
    "DeviceCloud",
    "FrontEnd",
    "backproject_device",
    "camera_front_end",
    "downsample_and_cap_device",
    "kernel",
    "recon_device",
    "torch_device",
    "upload_depth",
    "voxel_first_device",
]
