"""Aligned-prior construction for SEAM (pure JAX, model space).

Given the previous decoded chunk ``c_n`` of shape ``[..., H, D]``, the aligned prior repeats the last
unexecuted-tail action to keep the prior length-compatible with the generated chunk:

    a_tail   = c_n[..., K:H, :]              # length L = H - K
    a_aligned = concat(a_tail, repeat(a_tail[-1], K), axis=-2)   # length H

Only the first ``M <= L`` rows of ``a_aligned`` are ever guided; the repeated rows are inert padding.
"""

from __future__ import annotations

import jax.numpy as jnp


def build_aligned_prior(previous_chunk, horizon: int, execution_length: int):
    """Construct the aligned prior in model space.

    Args:
        previous_chunk: array of shape ``[..., H, D]`` (the previous **model-space** chunk).
        horizon: H, the chunk horizon (static int). Must equal ``previous_chunk.shape[-2]``.
        execution_length: K, the number of executed actions (static int). ``0 <= K < H``.

    Returns:
        Aligned prior of shape ``[..., H, D]``.

    Raises:
        ValueError: on shape/parameter inconsistency or an empty tail (L <= 0).
    """
    if previous_chunk.ndim < 2:
        raise ValueError(f"previous_chunk must have >= 2 dims [.., H, D], got shape {previous_chunk.shape}")
    h = previous_chunk.shape[-2]
    if h != horizon:
        raise ValueError(f"previous_chunk horizon {h} != horizon arg {horizon}")
    if not (0 <= execution_length < horizon):
        raise ValueError(
            f"execution_length must satisfy 0 <= K < H; got K={execution_length}, H={horizon}"
        )
    overlap = horizon - execution_length
    if overlap <= 0:
        raise ValueError(f"empty tail: L = H - K = {overlap} must be > 0")

    tail = previous_chunk[..., execution_length:horizon, :]  # [..., L, D]
    return _extend(tail, horizon, overlap)


def _extend(tail, horizon: int, overlap: int):
    """Repeat the last tail row so the prior has length H."""
    last = tail[..., -1:, :]  # [..., 1, D]
    pad_len = horizon - overlap  # == execution_length == K
    if pad_len > 0:
        repeat_shape = tuple(tail.shape[:-2]) + (pad_len, tail.shape[-1])
        pad = jnp.broadcast_to(last, repeat_shape)
        return jnp.concatenate([tail, pad], axis=-2)
    return tail


def build_aligned_prior_delta_compensated(
    previous_chunk,
    delta_state_phys,
    inv_scale,
    guided_mask,
    horizon: int,
    execution_length: int,
):
    """Aligned prior for base-relative-delta model spaces (e.g. rby1 with AbsoluteActions).

    The model denoises deltas relative to the proprio state at inference time, so the previous chunk's
    delta tail must be re-expressed relative to the NEW base state before it can serve as the overlap
    target:

        prior_delta_norm[j] = prev_tail_delta_norm[K+j] - Δs · inv_scale   (guided dims only)

    where ``Δs = s_curr - s_prev`` (physical proprio change) and ``inv_scale[d] = 2/(q99-q01)`` for the
    action normalization of dim ``d``. Unguided dims are unchanged (they are masked out downstream too).

    Args:
        previous_chunk: previous model-space chunk ``[..., H, D]`` (normalized delta).
        delta_state_phys: physical proprio change ``[..., D]`` (only guided dims used).
        inv_scale: per-dim ``2/(q99-q01)`` as ``[D]`` (0 on non-guided dims is fine).
        guided_mask: boolean ``[D]`` selecting compensated dims.
        horizon: H. execution_length: K.

    Returns:
        Aligned prior ``[..., H, D]``.
    """
    if previous_chunk.ndim < 2:
        raise ValueError(f"previous_chunk must be [.., H, D], got {previous_chunk.shape}")
    if previous_chunk.shape[-2] != horizon:
        raise ValueError(f"horizon mismatch: {previous_chunk.shape[-2]} != {horizon}")
    overlap = horizon - execution_length
    if overlap <= 0:
        raise ValueError(f"empty tail: L = {overlap}")

    tail = previous_chunk[..., execution_length:horizon, :]  # [..., L, D] normalized delta
    mask = jnp.asarray(guided_mask, dtype=bool)
    shift = (jnp.asarray(delta_state_phys) * jnp.asarray(inv_scale))  # [..., D] physical→norm units
    shift = jnp.where(mask, shift, jnp.zeros((), dtype=tail.dtype))
    # Broadcast the per-dim shift across the L overlap rows and subtract.
    tail_comp = tail - shift[..., None, :]
    return _extend(tail_comp, horizon, overlap)
