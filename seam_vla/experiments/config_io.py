"""Experiment config IO: load SeamConfig + rollout knobs from a YAML file."""

from __future__ import annotations

import dataclasses
from typing import Any, List

import yaml

from benchmark.seam_vla.config import SeamConfig


@dataclasses.dataclass
class RolloutConfig:
    task_suite_name: str = "libero_10"
    task_ids: List[int] = dataclasses.field(default_factory=lambda: [0])
    num_episodes: int = 2
    max_steps: int = 120
    seed: int = 7


def load_experiment(path: str) -> tuple[SeamConfig, RolloutConfig]:
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    seam = SeamConfig.from_dict(data.get("seam", {}))
    rollout = RolloutConfig(**data.get("rollout", {}))
    return seam, rollout
