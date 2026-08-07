"""Motion-quality metrics (paper-exact), computed on executed physical-space actions.

Definitions (SEAM paper, Eqs. 9-13):
    j_t = || a_{t+1} - 2 a_t + a_{t-1} ||_2                          (per-step jerk, Eq. 9)
    B   = { t | t % K == 0, t > 0 } with t-1, t+1 valid              (chunk boundaries)
    I   = remaining valid jerk centers not in B                      (interior)
    BJ  = mean(j_t for t in B)                                       (Eq. 10)
    IJ  = mean(j_t for t in I)                                       (Eq. 11)
    CD  = mean(|| a_t - a_{t-1} ||_2 for t in B)                     (Eq. 12)
    AVb = var(j_t for t in B)                                        (Eq. 13)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def per_step_jerk(actions: np.ndarray) -> np.ndarray:
    """Return j_t for t = 1 .. T-2 (length T-2). ``actions`` is ``[T, D]``."""
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"actions must be [T, D], got shape {a.shape}")
    if a.shape[0] < 3:
        return np.zeros((0,), dtype=np.float64)
    second_diff = a[2:] - 2.0 * a[1:-1] + a[:-2]  # centers t = 1 .. T-2
    return np.linalg.norm(second_diff, axis=-1)


def boundary_indices(num_steps: int, execution_length: int) -> np.ndarray:
    """Chunk-boundary centers B = {t | t % K == 0, t > 0, 1 <= t <= T-2}."""
    k = int(execution_length)
    if k <= 0:
        raise ValueError(f"execution_length must be > 0, got {k}")
    valid = np.arange(1, max(num_steps - 1, 1))  # t in [1, T-2]
    return valid[(valid % k == 0) & (valid > 0)]


def compute_motion_metrics(actions: np.ndarray, execution_length: int) -> dict:
    """Compute BJ, IJ, CD, AVb for one episode's executed actions ``[T, D]``.

    Returns a dict with per-metric values plus the paper key ``paper_avb`` and its alias
    ``boundary_jerk_variance``. Empty-set metrics are reported as NaN.
    """
    a = np.asarray(actions, dtype=np.float64)
    t_total = a.shape[0]
    jerk = per_step_jerk(a)  # index i corresponds to center t = i + 1
    centers = np.arange(1, t_total - 1) if t_total >= 3 else np.zeros((0,), dtype=int)

    b = boundary_indices(t_total, execution_length)
    b = b[np.isin(b, centers)]
    interior = np.setdiff1d(centers, b)

    def _jerk_at(ts: np.ndarray) -> np.ndarray:
        return jerk[ts - 1] if ts.size else np.zeros((0,), dtype=np.float64)

    bj_vals = _jerk_at(b)
    ij_vals = _jerk_at(interior)

    # CD uses first differences at boundary centers.
    if b.size:
        cd_vals = np.linalg.norm(a[b] - a[b - 1], axis=-1)
    else:
        cd_vals = np.zeros((0,), dtype=np.float64)

    def _mean(x):
        return float(np.mean(x)) if x.size else float("nan")

    def _var(x):
        return float(np.var(x)) if x.size else float("nan")

    avb = _var(bj_vals)
    return {
        "BJ": _mean(bj_vals),
        "IJ": _mean(ij_vals),
        "CD": _mean(cd_vals),
        "paper_avb": avb,
        "boundary_jerk_variance": avb,  # clearer alias for the same quantity
        "num_boundary": int(b.size),
        "num_interior": int(interior.size),
        "num_steps": int(t_total),
    }


def aggregate_metrics(per_episode: Sequence[dict]) -> dict:
    """Average metrics across episodes/tasks, ignoring NaNs."""
    if not per_episode:
        return {}
    keys = ["BJ", "IJ", "CD", "paper_avb", "boundary_jerk_variance"]
    out = {}
    for key in keys:
        vals = np.asarray([m[key] for m in per_episode if key in m], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        out[key] = float(vals.mean()) if vals.size else float("nan")
    return out
