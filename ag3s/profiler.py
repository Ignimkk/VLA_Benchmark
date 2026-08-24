"""Per-stage timing.

Spec §6 wants a latency number for each of the eight stages, and it wants the *correct baseline*
measured before anything is optimized — otherwise the optimization target is chosen by intuition
rather than by measurement. This module is deliberately dumb: a context manager, a dict of
accumulated milliseconds, and a table formatter.

The shape mirrors `benchmark/seam_vla/metrics/latency.py` (`mean`, `std`, `samples`, `to_dict`), so
AG3S timings and SEAM timings can sit in the same report without a converter. What it does *not*
copy is that module's `block_until_ready` handling — AG3S is numpy and scipy on the CPU, with no
device queue to synchronize against, so `perf_counter` is already honest here.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from typing import Iterator, Optional

#: The eight stages spec §6 asks for, in pipeline order. Named here so a table always has the same
#: rows even when a stage was skipped this frame (no robot model, no attention, and so on).
STAGE_ORDER: tuple[str, ...] = (
    "scene_reconstruction",
    "robot_self_filter",
    "support_surface",
    "attention_lifting",
    "target_grounding",
    "collision_candidates",
    "primitive_fitting",
    "constraint_generation",
)


@dataclasses.dataclass
class StageTiming:
    """Samples for one stage, in milliseconds."""

    name: str
    samples: list[float] = dataclasses.field(default_factory=list)

    def add(self, milliseconds: float) -> None:
        self.samples.append(float(milliseconds))

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def total_ms(self) -> float:
        return float(sum(self.samples))

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    @property
    def std_ms(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.mean_ms
        return float((sum((s - mean) ** 2 for s in self.samples) / self.count) ** 0.5)

    @property
    def last_ms(self) -> float:
        return self.samples[-1] if self.samples else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "total_ms": self.total_ms,
            "last_ms": self.last_ms,
        }


class StageProfiler:
    """Accumulates per-stage timings across frames.

    `warmup_frames` exists because the first frame through the pipeline pays for scipy's KD-tree
    import, CasADi's first symbolic build, and numpy's first big allocation. Reporting that as
    steady-state latency overstates it several-fold, so those frames are timed and then discarded
    rather than quietly not measured.
    """

    def __init__(self, enabled: bool = True, warmup_frames: int = 0):
        self.enabled = bool(enabled)
        self.warmup_frames = int(warmup_frames)
        self.frame_index = 0
        self.stages: dict[str, StageTiming] = {}

    def reset(self) -> None:
        self.frame_index = 0
        self.stages.clear()

    @property
    def counting(self) -> bool:
        return self.enabled and self.frame_index >= self.warmup_frames

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block. A no-op when profiling is off, so callers need no conditional."""
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            if self.counting:
                self.stages.setdefault(name, StageTiming(name)).add(elapsed)

    def record(self, name: str, milliseconds: float) -> None:
        """Record a duration measured elsewhere — e.g. a sub-stage a callee timed for us."""
        if self.counting:
            self.stages.setdefault(name, StageTiming(name)).add(float(milliseconds))

    def end_frame(self) -> None:
        self.frame_index += 1

    # --- reporting ------------------------------------------------------------------------
    def last_frame_ms(self) -> dict[str, float]:
        """What the most recent frame cost, stage by stage — what goes into a constraint set."""
        return {name: timing.last_ms for name, timing in self.stages.items()}

    def total_ms(self) -> float:
        return float(sum(t.mean_ms for t in self.stages.values()))

    def to_dict(self) -> dict[str, object]:
        return {
            "frames": self.frame_index,
            "warmup_frames": self.warmup_frames,
            "total_mean_ms": self.total_ms(),
            "stages": {name: timing.to_dict() for name, timing in self.stages.items()},
        }

    def to_table(self, title: Optional[str] = None) -> str:
        """A fixed-row latency table. Stages that did not run show as `skipped`, not as zero."""
        measured = self.frame_index - self.warmup_frames
        lines = [
            title or f"AG3S per-stage latency ({max(measured, 0)} frame(s) after "
            f"{self.warmup_frames} warm-up)",
            f"  {'stage':<26} {'mean ms':>9} {'std':>7} {'n':>4}",
            f"  {'-' * 26} {'-' * 9} {'-' * 7} {'-' * 4}",
        ]
        known = list(STAGE_ORDER) + [n for n in self.stages if n not in STAGE_ORDER]
        for name in known:
            timing = self.stages.get(name)
            if timing is None or timing.count == 0:
                lines.append(f"  {name:<26} {'skipped':>9} {'':>7} {'':>4}")
            else:
                lines.append(
                    f"  {name:<26} {timing.mean_ms:9.2f} {timing.std_ms:7.2f} {timing.count:4d}"
                )
        lines.append(f"  {'-' * 26} {'-' * 9} {'-' * 7} {'-' * 4}")
        lines.append(f"  {'TOTAL':<26} {self.total_ms():9.2f}")
        return "\n".join(lines)


__all__ = ["STAGE_ORDER", "StageProfiler", "StageTiming"]
