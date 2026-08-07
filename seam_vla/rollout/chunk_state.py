"""Rollout chunk-state management.

Re-exports the immutable :class:`SeamState` and provides a thin stateful ``SeamSession`` wrapper for
single-environment execution (holds one ``SeamState`` and offers ``reset()``). State is per-session;
never process-global.
"""

from __future__ import annotations

from benchmark.seam_vla.state import SeamState

__all__ = ["SeamState", "SeamSession"]


class SeamSession:
    """Mutable per-environment holder around an immutable :class:`SeamState`."""

    def __init__(self) -> None:
        self._state = SeamState.initial()

    @property
    def state(self) -> SeamState:
        return self._state

    @property
    def is_first_chunk(self) -> bool:
        return self._state.is_first_chunk

    @property
    def chunk_index(self) -> int:
        return self._state.chunk_index

    def reset(self) -> None:
        """Clear previous-chunk state (first chunk after this uses baseline inference)."""
        self._state = self._state.reset()

    def record_chunk(self, model_space_chunk, physical_space_chunk) -> None:
        self._state = self._state.with_chunk(model_space_chunk, physical_space_chunk)
