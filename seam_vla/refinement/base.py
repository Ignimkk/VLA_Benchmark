"""DecodedChunkRefiner interface (reserved for trajectory optimization / MPC).

A ``DecodedChunkRefiner`` operates on the **physical-space** decoded chunk, AFTER baseline
unnormalization and output transforms and BEFORE execution. This is intentionally separate from
``DenoisingGuidance`` (which acts in model space inside the denoising loop). Future implementations may
apply trajectory optimization, MPC, joint/velocity/acceleration limits, collision or CBF safety
filtering, etc. For this work only :class:`IdentityChunkRefiner` is provided.
"""

from __future__ import annotations

import abc
from typing import Any


class DecodedChunkRefiner(abc.ABC):
    """Refines a decoded physical-space action chunk before execution."""

    @abc.abstractmethod
    def refine(self, physical_chunk, context: dict[str, Any] | None = None):
        """Return a (possibly refined) physical-space chunk of the same shape ``[H, action_dim_env]``."""
