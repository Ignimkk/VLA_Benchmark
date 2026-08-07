"""Synchronized JAX latency measurement.

Separates one-time JIT/compilation cost from steady-state cost. Always calls ``block_until_ready`` on
the measured output before stopping the timer so device execution is actually synchronized.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, List

import jax


def _block(x):
    """Block until all leaves of a pytree are ready."""
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return x


@dataclasses.dataclass
class TimingResult:
    compile_ms: float
    steady_ms_mean: float
    steady_ms_std: float
    samples_ms: List[float]

    def to_dict(self) -> dict:
        return {
            "compile_ms": self.compile_ms,
            "steady_ms_mean": self.steady_ms_mean,
            "steady_ms_std": self.steady_ms_std,
            "steady_samples_ms": list(self.samples_ms),
        }


def time_call(fn: Callable, *args, warmup: int = 1, repeats: int = 5, **kwargs) -> TimingResult:
    """Time ``fn(*args, **kwargs)`` with a warm-up (compile) pass and steady-state repeats.

    The first ``warmup`` call(s) trigger compilation and are reported separately; the following
    ``repeats`` calls are averaged. Each call is synchronized via ``block_until_ready``.
    """
    # Warm-up / compile.
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _block(out)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    for _ in range(max(warmup - 1, 0)):
        _block(fn(*args, **kwargs))

    samples: List[float] = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        _block(out)
        samples.append((time.perf_counter() - t) * 1000.0)

    import numpy as np

    arr = np.asarray(samples, dtype=float)
    return TimingResult(
        compile_ms=compile_ms,
        steady_ms_mean=float(arr.mean()),
        steady_ms_std=float(arr.std()),
        samples_ms=samples,
    )
