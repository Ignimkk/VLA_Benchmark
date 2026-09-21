"""Stage 3 — 2D attention lifted onto 3D points.

Produces `P_A = {(X, Y, Z, A)}`: every reconstructed point, with an attention value attached.

The word doing the work is *every*. This stage never removes a point for having low attention, and
the return type has exactly as many entries as the cloud it was given. Attention's only job starts
in `target_grounding`, where it picks seeds; the geometry that reaches collision candidates is the
geometry that physically exists, whatever the VLA happened to be looking at.

Two boundaries are drawn deliberately:

**`AttentionAdapter` isolates the VLA.** π0.5 returns a `[layer, head, token, patch]` block over a
16x16 agent-view grid (`benchmark/knows_vla/probe_p0b.py`); another model might return a dense map,
or a different grid, or nothing. Adapters convert whatever it is into an `(H, W)` float32 map, and
nothing downstream of `to_pixel_map` knows which VLA produced it.

**Normalization happens on the lifted per-point values, not on the pixel map.** A percentile taken
over the image includes pixels with no geometry behind them — sky, out-of-range, the hole where
invalid depth was dropped. Taking it over the points instead makes `attention.normalization` and
`clustering.seed_percentile` refer to the same population, which is the only way the two settings
compose predictably.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from benchmark.ag3s.config import AttentionConfig
from benchmark.ag3s.types import AttentionPointCloud, PointCloud

_EPS = 1e-12


# ------------------------------------------------------------------------------- resampling


def _resample(src: np.ndarray, out_hw: tuple[int, int], mode: str) -> np.ndarray:
    """Resize an `(h, w)` map to `(H, W)`.

    `nearest` is a pure index map — exact, and the right choice when the source is already at or
    above the target resolution. `bilinear` uses pixel-centre alignment (the `+0.5 ... -0.5` shift),
    which matters at these ratios: a 16x16 grid stretched over 480x640 is a factor of 30-40, and
    corner-aligned interpolation would bias every patch centre by half a patch, i.e. ~15 pixels.
    """
    src = np.asarray(src, np.float32)
    if src.ndim != 2:
        raise ValueError(f"attention map must be 2-D, got shape {src.shape}")
    h, w = src.shape
    H, W = int(out_hw[0]), int(out_hw[1])
    if (h, w) == (H, W):
        return src

    if mode == "nearest":
        gy = np.minimum((np.arange(H) * h) // H, h - 1)
        gx = np.minimum((np.arange(W) * w) // W, w - 1)
        return src[np.ix_(gy, gx)]

    if mode != "bilinear":
        raise ValueError(f"unknown interpolation {mode!r}")

    fy = np.clip((np.arange(H) + 0.5) * h / H - 0.5, 0.0, h - 1.0)
    fx = np.clip((np.arange(W) + 0.5) * w / W - 0.5, 0.0, w - 1.0)
    y0 = np.floor(fy).astype(np.int64)
    x0 = np.floor(fx).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (fy - y0).astype(np.float32)[:, None]
    wx = (fx - x0).astype(np.float32)[None, :]

    top = src[np.ix_(y0, x0)] * (1.0 - wx) + src[np.ix_(y0, x1)] * wx
    bot = src[np.ix_(y1, x0)] * (1.0 - wx) + src[np.ix_(y1, x1)] * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def _sample_at(pixel_map: np.ndarray, uv: np.ndarray, mode: str) -> np.ndarray:
    """Read the map at `(u, v)`. Integer `uv` is an exact lookup; float `uv` is interpolated."""
    H, W = pixel_map.shape
    uv = np.asarray(uv)
    if np.issubdtype(uv.dtype, np.integer):
        u = np.clip(uv[:, 0], 0, W - 1)
        v = np.clip(uv[:, 1], 0, H - 1)
        return pixel_map[v, u].astype(np.float32)

    u = np.clip(uv[:, 0].astype(np.float64), 0.0, W - 1.0)
    v = np.clip(uv[:, 1].astype(np.float64), 0.0, H - 1.0)
    if mode == "nearest":
        return pixel_map[np.rint(v).astype(np.int64), np.rint(u).astype(np.int64)].astype(np.float32)
    x0, y0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    wx, wy = (u - x0).astype(np.float32), (v - y0).astype(np.float32)
    top = pixel_map[y0, x0] * (1.0 - wx) + pixel_map[y0, x1] * wx
    bot = pixel_map[y1, x0] * (1.0 - wx) + pixel_map[y1, x1] * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


# --------------------------------------------------------------------------------- adapters


class DenseAttentionAdapter:
    """For models that already emit a dense 2-D map; resamples it to the depth resolution."""

    def __init__(self, interpolation: str = "bilinear"):
        self.interpolation = interpolation

    def to_pixel_map(self, raw: Any, out_hw: tuple[int, int]) -> np.ndarray:
        return _resample(np.asarray(raw, np.float32), out_hw, self.interpolation)


class GridAttentionAdapter:
    """For models that emit per-patch attention over a square grid.

    Accepts, in decreasing order of rawness:

    * ``(L, Hd, T, P)`` — π0.5's block: layers x heads x suffix tokens x patches. Tokens are averaged
      (matching `AGGS["mean"]` in `benchmark/knows_vla/probe_p0b.py`), then `(layer, head)` selects
      the cell. Both must be configured; there is no defensible default, and the KNOWS probe found
      the paper's (12, 3) is not obviously best on this checkpoint.
    * ``(L, Hd, P)`` — same without the token axis.
    * ``(P,)`` or ``(G, G)`` — a single already-selected map.

    `P` must be a perfect square; `G = sqrt(P)` is the grid side. π0.5's agent-view block is 256
    patches, so `G = 16`.
    """

    def __init__(
        self,
        *,
        interpolation: str = "bilinear",
        layer: Optional[int] = None,
        head: Optional[int] = None,
        grid: Optional[int] = None,
    ):
        self.interpolation = interpolation
        self.layer = layer
        self.head = head
        self.grid = grid

    @classmethod
    def from_config(cls, config: AttentionConfig) -> "GridAttentionAdapter":
        return cls(interpolation=config.interpolation, layer=config.layer, head=config.head)

    def _select(self, raw: np.ndarray) -> np.ndarray:
        if raw.ndim == 4:  # (L, heads, tokens, patches)
            raw = raw.mean(axis=2)
        if raw.ndim == 3:  # (L, heads, patches)
            if self.layer is None or self.head is None:
                raise ValueError(
                    f"attention has shape {raw.shape}; set attention.layer and attention.head to "
                    "choose a cell (there is no safe default)"
                )
            raw = raw[self.layer, self.head]
        return raw

    def to_grid(self, raw: Any) -> np.ndarray:
        """Whatever the VLA gave us, as a `(G, G)` float32 patch grid."""
        arr = self._select(np.asarray(raw, np.float32))
        if arr.ndim == 2:
            return arr
        if arr.ndim != 1:
            raise ValueError(f"cannot interpret attention of shape {np.asarray(raw).shape}")
        g = self.grid or int(round(float(np.sqrt(arr.size))))
        if g * g != arr.size:
            raise ValueError(f"{arr.size} patches is not a square grid; pass grid= explicitly")
        return arr.reshape(g, g)

    def to_pixel_map(self, raw: Any, out_hw: tuple[int, int]) -> np.ndarray:
        return _resample(self.to_grid(raw), out_hw, self.interpolation)


def make_adapter(raw: Any, config: AttentionConfig) -> Any:
    """Pick an adapter by shape: a 2-D input the size of the image is dense, anything else is a grid.

    Convenience only. A caller with a known VLA should construct the adapter explicitly and pass it
    to `lift`, which is the supported path and the one that keeps AG3S uncoupled.
    """
    arr = np.asarray(raw)
    if arr.ndim == 2 and min(arr.shape) > 64:
        return DenseAttentionAdapter(config.interpolation)
    return GridAttentionAdapter.from_config(config)


# ---------------------------------------------------------------------------- normalization


def normalize_attention(values: np.ndarray, config: AttentionConfig) -> np.ndarray:
    """Map raw attention onto a comparable scale. Order is preserved by every mode.

    That last property is what lets `clustering.seed_percentile` and `attention.normalization` be
    chosen independently: normalization changes the *values* a threshold sees, never the *ranking*,
    so a percentile-based seed cut selects the same points under any of these.

    * `percentile` (default) — clip to `percentile_range` and rescale. Robust to the single hot patch
      that min-max normalization would otherwise let dominate the whole map.
    * `minmax` — the textbook version, kept as the ablation baseline.
    * `softmax` — temperature-scaled, then rescaled so the peak is 1. Without that rescale the values
      would be ~1/N and any absolute threshold would be meaningless.
    * `none` — pass through, for a VLA that already normalizes.
    """
    a = np.asarray(values, np.float32).reshape(-1)
    if a.size == 0:
        return a

    mode = config.normalization
    if mode == "none":
        return a
    if mode == "softmax":
        z = (a - a.max()) / float(config.temperature)
        p = np.exp(z, dtype=np.float32)
        total = float(p.sum())
        if total <= _EPS:
            return np.zeros_like(a)
        p /= total
        peak = float(p.max())
        return p / peak if peak > _EPS else np.zeros_like(a)

    if mode == "percentile":
        lo, hi = np.percentile(a, config.percentile_range)
    elif mode == "minmax":
        lo, hi = float(a.min()), float(a.max())
    else:
        raise ValueError(f"unknown attention.normalization {mode!r}")

    if hi - lo <= _EPS:
        # Degenerate input: every value identical, or the percentile band collapsed. Returning zeros
        # rather than dividing is what lets `target_grounding` report NO_ATTENTION instead of
        # producing a target out of numerical noise.
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------------- lifting


def lift(
    cloud: PointCloud,
    attention: Any,
    config: AttentionConfig | None = None,
    *,
    adapter: Any = None,
    image_hw: Optional[tuple[int, int]] = None,
    camera_intrinsics: Optional[np.ndarray] = None,
    T_base_cam: Optional[np.ndarray] = None,
) -> AttentionPointCloud:
    """Attach an attention value to every point of `cloud`.

    Correspondence comes from `cloud.uv`, which reconstruction preserved for exactly this. If the
    cloud has no `uv` — a sensor that delivers 3D with no image behind it — the points are projected
    into the image instead, which requires `camera_intrinsics` and `T_base_cam`.

    `image_hw` is the resolution the attention map should be expanded to. It defaults to the extent
    of `cloud.uv`, which is right when the cloud came from `backproject`; pass it explicitly if the
    cloud has been heavily downsampled and its `uv` no longer reaches the image border.

    The returned cloud has `len(cloud)` entries. Always.
    """
    cfg = config or AttentionConfig()
    adapter = adapter or make_adapter(attention, cfg)

    if cloud.is_empty:
        return AttentionPointCloud(cloud, np.zeros(0, np.float32), np.zeros(0, np.float32))

    if cloud.uv is not None:
        hw = image_hw or (int(cloud.uv[:, 1].max()) + 1, int(cloud.uv[:, 0].max()) + 1)
        uv = cloud.uv
    else:
        if camera_intrinsics is None or T_base_cam is None or image_hw is None:
            raise ValueError(
                "cloud has no uv; lifting then needs camera_intrinsics, T_base_cam and image_hw "
                "to project the points back into the image"
            )
        hw = image_hw
        uv = _project(cloud.points, camera_intrinsics, T_base_cam)

    pixel_map = adapter.to_pixel_map(attention, hw)
    raw_values = _sample_at(pixel_map, uv, cfg.interpolation)
    # Both are kept: normalization is lossy at the top (percentile clipping creates ties), and
    # `target_grounding` needs an unambiguous peak. See `AttentionPointCloud`.
    return AttentionPointCloud(cloud, normalize_attention(raw_values, cfg), raw_values)


def _project(points: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """Base-frame points -> float `(u, v)`. The inverse of `reconstruction.backproject`."""
    K = np.asarray(K, np.float64)
    T = np.asarray(T_base_cam, np.float64)
    cam = (np.asarray(points, np.float64) - T[:3, 3]) @ T[:3, :3]
    z = np.where(np.abs(cam[:, 2]) < _EPS, _EPS, cam[:, 2])
    return np.stack([K[0, 0] * cam[:, 0] / z + K[0, 2], K[1, 1] * cam[:, 1] / z + K[1, 2]], axis=1)


__all__ = [
    "DenseAttentionAdapter",
    "GridAttentionAdapter",
    "lift",
    "make_adapter",
    "normalize_attention",
]
