"""Per-session SEAM state (no process-global state).

``SeamState`` holds the two previous-chunk representations plus a chunk counter. It is immutable;
transitions return a new instance (functional update). One instance per rollout environment / policy
session.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class SeamState:
    """State carried across chunk queries within a single episode.

    Attributes:
        previous_chunk_model_space: raw sampler output of the previous chunk, ``[H, D]`` (model space),
            or ``None`` before the first chunk / after reset.
        previous_chunk_physical_space: post-transform previous chunk, ``[H, 7]`` (physical space), or
            ``None``. Used only for logging/metrics/validation — never for VLS.
        chunk_index: number of chunks produced so far (0 before the first).
    """

    previous_chunk_model_space: Optional[Any] = None
    previous_chunk_physical_space: Optional[Any] = None
    # Physical proprio state at the previous inference (for base-relative-delta compensation, e.g. rby1).
    previous_proprio_state: Optional[Any] = None
    chunk_index: int = 0

    @property
    def is_first_chunk(self) -> bool:
        """True when no usable previous chunk exists (first chunk or just-reset)."""
        return self.previous_chunk_model_space is None

    @classmethod
    def initial(cls) -> "SeamState":
        return cls()

    def reset(self) -> "SeamState":
        """Clear both previous-chunk representations and the counter."""
        return SeamState.initial()

    def with_chunk(self, model_space_chunk, physical_space_chunk, proprio_state=None) -> "SeamState":
        """Return a new state recording the just-produced chunk and advancing the counter."""
        return dataclasses.replace(
            self,
            previous_chunk_model_space=model_space_chunk,
            previous_chunk_physical_space=physical_space_chunk,
            previous_proprio_state=proprio_state,
            chunk_index=self.chunk_index + 1,
        )
