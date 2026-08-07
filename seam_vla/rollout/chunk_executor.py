"""Chunk executor: replicates the baseline LIBERO K-of-H execution cadence.

Mirrors ``examples/libero/main.py`` (a ``collections.deque`` that is refilled with the first
``replan_steps`` actions of each new chunk and popped one action per env step). SEAM does not change
this cadence — it only changes how the chunk is generated. The executor also records the
previous-chunk representations into the SEAM state *before* the queue discards the unexecuted tail.
"""

from __future__ import annotations

import collections
from typing import Callable, Optional

import numpy as np


class ChunkExecutor:
    """Consumes K actions per chunk from a policy that returns full [H, D] chunks.

    Args:
        predict_chunk_fn: callable ``(obs_dict) -> (physical_chunk[H, D], diagnostics)``. This is
            typically ``SeamPolicySession.predict_chunk``. It is responsible for updating SEAM state.
        execution_length: K, the number of actions executed before requesting a new chunk.
    """

    def __init__(self, predict_chunk_fn: Callable, execution_length: int):
        if execution_length <= 0:
            raise ValueError(f"execution_length must be > 0, got {execution_length}")
        self._predict = predict_chunk_fn
        self.K = int(execution_length)
        self._queue: collections.deque = collections.deque()
        self.last_diagnostics = None
        self.num_chunks = 0

    def reset(self) -> None:
        self._queue.clear()
        self.num_chunks = 0
        self.last_diagnostics = None

    def act(self, obs_dict: dict) -> np.ndarray:
        """Return the next single action, requesting a new chunk when the queue is empty."""
        if not self._queue:
            chunk, diag = self._predict(obs_dict)
            chunk = np.asarray(chunk)
            if chunk.shape[0] < self.K:
                raise AssertionError(
                    f"policy predicted {chunk.shape[0]} actions but K={self.K} are required per chunk"
                )
            # Execute only the first K actions (baseline semantics); discard the tail.
            self._queue.extend(chunk[: self.K])
            self.last_diagnostics = diag
            self.num_chunks += 1
        return self._queue.popleft()
