"""Velocity-guided Loss Steering (VLS) — the pure JAX core of SEAM.

Closed-form correction applied to the model-space post-Euler candidate. No autodiff / no policy
backward pass. For a candidate ``x_cand = x_t + dt*v_t`` at ODE time ``t_next``:

    r      = (1 - t_next) * a_aligned                     # consistency target (Eq. 4)
    g      = -2 * (x_cand - r)                            # -grad of ||x_cand - r||^2 (Eq. 7)
    x_next = x_cand + lambda*(1 - t_next) * g   (masked)  # (Eq. 8)

Only the first ``M_eff`` overlap rows and the guided action dimensions are corrected; everything else
equals ``x_cand`` exactly. The masks are concrete boolean constants; ``aligned_prior``, ``lam`` and
``enabled`` may be traced arrays.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from benchmark.seam_vla.guidance.base import DenoisingGuidance


def make_position_mask(horizon: int, window: int) -> np.ndarray:
    """Boolean mask of shape [H], True for the first ``min(window, H)`` rows."""
    m = int(min(max(window, 0), horizon))
    mask = np.zeros((horizon,), dtype=bool)
    mask[:m] = True
    return mask


def compute_consistency_target(aligned_prior, t_next):
    """Time-interpolated consistency target ``r = (1 - t_next) * a_aligned`` (full [..,H,D])."""
    return (1.0 - t_next) * aligned_prior


def compute_vls_correction(candidate, target):
    """Closed-form negative gradient ``g = -2 * (candidate - target)`` (full [..,H,D])."""
    return -2.0 * (candidate - target)


def apply_guidance(candidate, aligned_prior, t_next, lam, pos_mask, dim_mask, enabled=1.0):
    """Apply the masked VLS correction and return the corrected candidate.

    Args:
        candidate: Euler candidate ``[..., H, D]`` (model space).
        aligned_prior: aligned prior ``[..., H, D]`` (model space).
        t_next: scalar ODE time in [0, 1].
        lam: guidance strength (scalar; may be traced).
        pos_mask: boolean ``[H]`` selecting guided overlap rows.
        dim_mask: boolean ``[D]`` selecting guided action dims.
        enabled: scalar 0/1 gate (folded into the strength; 0 => exact baseline).

    Returns:
        Corrected candidate, same shape/dtype as ``candidate``.
    """
    target = compute_consistency_target(aligned_prior, t_next)
    g = compute_vls_correction(candidate, target)
    strength = jnp.asarray(lam, dtype=candidate.dtype) * jnp.asarray(enabled, dtype=candidate.dtype)
    correction = strength * (1.0 - t_next) * g  # [..., H, D]
    mask2d = jnp.asarray(pos_mask, dtype=bool)[:, None] & jnp.asarray(dim_mask, dtype=bool)[None, :]
    step = jnp.where(mask2d, correction, jnp.zeros((), dtype=candidate.dtype))
    return (candidate + step).astype(candidate.dtype)


class VLSGuidance(DenoisingGuidance):
    """DenoisingGuidance implementing VLS.

    Constructed *inside* the traced sampling function so it may close over traced arrays
    (``aligned_prior``, ``lam``, ``enabled``). ``pos_mask`` and ``dim_mask`` are concrete constants.
    """

    def __init__(self, aligned_prior, lam, pos_mask, dim_mask, enabled=1.0):
        self.aligned_prior = aligned_prior
        self.lam = lam
        self.pos_mask = pos_mask
        self.dim_mask = dim_mask
        self.enabled = enabled

    def update(self, candidate, t_next):
        return apply_guidance(
            candidate,
            self.aligned_prior,
            t_next,
            self.lam,
            self.pos_mask,
            self.dim_mask,
            self.enabled,
        )
