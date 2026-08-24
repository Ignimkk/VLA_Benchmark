"""Trajectory optimization: AG3S's collision constraints + SEAM's reference chunk -> a safe chunk.

Fills the `DecodedChunkRefiner` slot `benchmark/seam_vla/refinement/base.py` reserved. Nothing in
`seam_vla` is modified; `TrajOptChunkRefiner` imports and subclasses the hook.

Imports here stay light — `casadi`, `osqp` and `scipy` are pulled in by the modules that need them,
so `from benchmark.trajopt import TrajOptConfig` cannot fail for environment reasons.
"""

from benchmark.trajopt.config import TrajOptConfig, TrajOptConfigError
from benchmark.trajopt.types import (
    ChunkLayout,
    JointLimits,
    TrajOptResult,
    TrajOptStatus,
)

__all__ = [
    "ChunkLayout",
    "JointLimits",
    "TrajOptConfig",
    "TrajOptConfigError",
    "TrajOptResult",
    "TrajOptStatus",
]
