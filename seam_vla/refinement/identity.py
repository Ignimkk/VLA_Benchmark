"""Identity chunk refiner: returns the physical chunk unchanged."""

from __future__ import annotations

from typing import Any

from benchmark.seam_vla.refinement.base import DecodedChunkRefiner


class IdentityChunkRefiner(DecodedChunkRefiner):
    """No-op refiner. ``refine`` returns the physical chunk exactly."""

    def refine(self, physical_chunk, context: dict[str, Any] | None = None):
        return physical_chunk
