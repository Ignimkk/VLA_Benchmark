"""SEAM diagnostics: correction-norm and application-count accumulation (host side).

These helpers run outside the JIT boundary (after ``block_until_ready``) so they may use NumPy.
"""

from __future__ import annotations

import dataclasses
from typing import List

import numpy as np


def correction_norm(candidate, corrected) -> float:
    """L2 norm of the applied correction over the whole chunk."""
    diff = np.asarray(corrected, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    return float(np.linalg.norm(diff.reshape(-1)))


@dataclasses.dataclass
class SeamDiagnostics:
    """Accumulates VLS correction statistics across steps and chunks within an episode."""

    per_step_norms: List[float] = dataclasses.field(default_factory=list)
    per_chunk_norms: List[float] = dataclasses.field(default_factory=list)
    num_applications: int = 0
    num_chunks: int = 0

    def record_step(self, norm: float) -> None:
        self.per_step_norms.append(float(norm))
        if norm > 0.0:
            self.num_applications += 1

    def record_chunk(self, norm: float) -> None:
        self.per_chunk_norms.append(float(norm))
        self.num_chunks += 1

    def summary(self) -> dict:
        steps = np.asarray(self.per_step_norms, dtype=np.float64) if self.per_step_norms else np.zeros(0)
        chunks = np.asarray(self.per_chunk_norms, dtype=np.float64) if self.per_chunk_norms else np.zeros(0)
        return {
            "num_vls_applications": int(self.num_applications),
            "num_chunks": int(self.num_chunks),
            "mean_step_correction_norm": float(steps.mean()) if steps.size else 0.0,
            "max_step_correction_norm": float(steps.max()) if steps.size else 0.0,
            "mean_chunk_correction_norm": float(chunks.mean()) if chunks.size else 0.0,
            "max_chunk_correction_norm": float(chunks.max()) if chunks.size else 0.0,
        }
