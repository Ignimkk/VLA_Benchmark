"""Physical-space overlap-steering refiner (SEAM applied at the decoded-chunk stage).

Operates on the decoded **absolute** action chunk, after Unnormalize/AbsoluteActions. It nudges the new
chunk's first ``M`` steps toward the previous chunk's unexecuted absolute tail by a small factor ``λ``
(SEAM-style *steering*, NOT averaging), reducing the chunk-boundary discontinuity. Because it works in
absolute space, it is correct regardless of the model's internal (delta) representation and needs only
the previous decoded chunk — which makes it verifiable offline on recorded chunks.

    prev_tail[j] = prev_chunk[K + j]                       (absolute, j in [0, L))
    w(j)        = λ · (1 - j/M)   if decay else λ           (taper across the window)
    new[j, g]  += w(j) · (prev_tail[j, g] - new[j, g])      for j in [0, M), guided dims g

Empirically on real RB-Y1 records a **uniform** weight worsens boundary jerk (the post-hoc-averaging
failure the SEAM paper warns about), while a small-λ **decaying** weight gives a modest reduction. The
decaying schedule is the physical-space analog of the paper's ``(1-t)`` denoising schedule and is the
default. This refiner is a lightweight decoded-stage smoother; the primary, paper-faithful method for
RB-Y1 is denoising-stage VLS with base-state compensation (server-side).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from benchmark.seam_vla.refinement.base import DecodedChunkRefiner


class OverlapSteerRefiner(DecodedChunkRefiner):
    """Steer the new chunk's overlap prefix toward the previous chunk's absolute tail.

    Args:
        execution_length: K (executed per chunk); the previous tail starts at index K.
        guided_window: M (number of overlap steps to steer); clamped to min(M, L).
        lam: steering strength in [0, 1]. 0 = no-op; small values nudge, 1 = snap to prev tail.
        guided_dims: action-dim indices to steer (e.g. arm joints, excluding grippers). None = all.
        decay: if True (default), weight tapers as ``λ·(1-j/M)`` across the window.
    """

    def __init__(
        self,
        execution_length: int,
        guided_window: int,
        lam: float,
        guided_dims: Sequence[int] | None = None,
        decay: bool = True,
    ):
        self.K = int(execution_length)
        self.M = int(guided_window)
        self.lam = float(lam)
        self.decay = bool(decay)
        self.guided_dims = None if guided_dims is None else np.asarray(list(guided_dims), dtype=int)

    def refine(self, physical_chunk, context: dict[str, Any] | None = None):
        chunk = np.array(physical_chunk, dtype=np.float64, copy=True)
        context = context or {}
        prev = context.get("previous_physical_chunk")
        if prev is None or self.lam <= 0.0:
            return physical_chunk  # first chunk / disabled: unchanged
        prev = np.asarray(prev, dtype=np.float64)

        H = chunk.shape[0]
        L = H - self.K
        m = int(min(self.M, L, prev.shape[0] - self.K, H))
        if m <= 0:
            return physical_chunk
        prev_tail = prev[self.K : self.K + m]  # [m, D]
        new_head = chunk[:m]                    # [m, D]
        j = np.arange(m, dtype=np.float64)
        w = self.lam * (1.0 - j / m) if self.decay else np.full(m, self.lam)  # [m]
        correction = w[:, None] * (prev_tail - new_head)
        if self.guided_dims is not None:
            g = self.guided_dims
            chunk[:m, g] = new_head[:, g] + correction[:, g]
        else:
            chunk[:m] = new_head + correction
        return chunk
