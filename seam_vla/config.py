"""SEAM configuration.

Fields default to the *derived* profile for the local ``pi05_libero`` checkpoint (H=10, K=5, N=10),
NOT the paper profile (H=50, K=10, N=10). ``seam_assert_baseline_shape`` cross-checks the horizon /
number of ODE steps against the live model and the execution length against the rollout, and raises a
clear error on mismatch instead of silently altering baseline behavior.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Sequence

import numpy as np

GuidedDimsSpec = Literal["all_physical"] | Sequence[int] | Sequence[bool]


class SeamConfigError(ValueError):
    """Raised when a SEAM configuration is inconsistent with the actual model/rollout."""


@dataclasses.dataclass(frozen=True)
class SeamConfig:
    """Velocity-guided Loss Steering (VLS) configuration.

    Attributes mirror the ``seam_*`` fields in the spec. Values are validated against the live model
    via :meth:`assert_matches_model` when ``seam_assert_baseline_shape`` is true.
    """

    seam_enabled: bool = False
    seam_lambda: float = 0.1
    seam_guided_window: int = 5  # M; clamped to min(M, L) at runtime. Paper uses 20 (needs L>=20).
    seam_horizon: int = 10  # H; must match model.action_horizon.
    seam_execution_length: int = 5  # K; must match rollout replan_steps. L = H - K.
    seam_num_ode_steps: int = 10  # N; must match sampler num_steps.
    seam_guided_dimensions: GuidedDimsSpec = "all_physical"
    seam_target_mode: Literal["repeat_last_tail"] = "repeat_last_tail"
    seam_log_corrections: bool = True
    seam_assert_baseline_shape: bool = True
    # Number of valid physical action dims (LIBERO: 7; rby1: 14). Used for the "all_physical" mask.
    num_valid_physical_dims: int = 7
    # Model action dim D (padded). pi0.5: 32.
    action_dim: int = 32
    # Where SEAM applies. "denoising": VLS inside the flow-matching ODE (model space, paper-faithful).
    # "decoded": a physical-space overlap-steering refiner on the decoded absolute chunk.
    seam_stage: Literal["denoising", "decoded"] = "denoising"
    # rby1 uses AbsoluteActions (model space = base-relative delta). When true, the aligned prior is
    # compensated by the base-state change so it targets the previous chunk's ABSOLUTE tail. Only valid
    # with seam_stage="denoising". Leave false for LIBERO (no AbsoluteActions).
    seam_delta_base_compensation: bool = False

    # --- derived quantities -------------------------------------------------
    @property
    def overlap_length(self) -> int:
        """L = H - K."""
        return self.seam_horizon - self.seam_execution_length

    @property
    def effective_window(self) -> int:
        """M_eff = min(M, L), clamped to [0, L]."""
        return int(min(max(self.seam_guided_window, 0), max(self.overlap_length, 0)))

    # --- validation ---------------------------------------------------------
    def __post_init__(self) -> None:
        if self.seam_horizon <= 0:
            raise SeamConfigError(f"seam_horizon must be > 0, got {self.seam_horizon}")
        if not (0 <= self.seam_execution_length < self.seam_horizon):
            raise SeamConfigError(
                f"seam_execution_length must satisfy 0 <= K < H; got K={self.seam_execution_length}, "
                f"H={self.seam_horizon} (overlap L={self.overlap_length} must be > 0)"
            )
        if self.overlap_length <= 0:
            raise SeamConfigError(
                f"overlap L = H - K must be > 0; got H={self.seam_horizon}, "
                f"K={self.seam_execution_length}"
            )
        if self.seam_num_ode_steps <= 0:
            raise SeamConfigError(f"seam_num_ode_steps must be > 0, got {self.seam_num_ode_steps}")
        if self.seam_lambda < 0:
            raise SeamConfigError(f"seam_lambda must be >= 0, got {self.seam_lambda}")
        if self.seam_guided_window < 0:
            raise SeamConfigError(f"seam_guided_window must be >= 0, got {self.seam_guided_window}")
        if self.seam_target_mode != "repeat_last_tail":
            raise SeamConfigError(
                f"only seam_target_mode='repeat_last_tail' is supported, got {self.seam_target_mode!r}"
            )
        if self.seam_stage not in ("denoising", "decoded"):
            raise SeamConfigError(f"seam_stage must be 'denoising' or 'decoded', got {self.seam_stage!r}")
        if self.seam_delta_base_compensation and self.seam_stage != "denoising":
            raise SeamConfigError(
                "seam_delta_base_compensation requires seam_stage='denoising' (it compensates the "
                "model-space aligned prior)."
            )

    def assert_matches_model(
        self,
        *,
        model_horizon: int,
        model_num_steps: int,
        rollout_execution_length: int | None = None,
    ) -> None:
        """Fail loudly if the config disagrees with the actual model/rollout.

        This is the safeguard that prevents silently reproducing the paper profile (H=50, K=10) on a
        checkpoint that is actually H=10, K=5.
        """
        if not self.seam_assert_baseline_shape:
            return
        problems = []
        if self.seam_horizon != model_horizon:
            problems.append(
                f"seam_horizon={self.seam_horizon} != model.action_horizon={model_horizon}. "
                f"The horizon is baked into the checkpoint and cannot be changed here."
            )
        if self.seam_num_ode_steps != model_num_steps:
            problems.append(
                f"seam_num_ode_steps={self.seam_num_ode_steps} != sampler num_steps={model_num_steps}."
            )
        if rollout_execution_length is not None and self.seam_execution_length != rollout_execution_length:
            problems.append(
                f"seam_execution_length={self.seam_execution_length} != rollout "
                f"replan_steps={rollout_execution_length}."
            )
        if problems:
            raise SeamConfigError(
                "SEAM config does not match the actual model/rollout (baseline would be altered):\n  - "
                + "\n  - ".join(problems)
                + "\nSet the config to the derived values or disable seam_assert_baseline_shape only if "
                "you understand the consequences."
            )

    def build_dim_mask(self) -> np.ndarray:
        """Boolean mask of shape [action_dim] selecting guided action dimensions."""
        return build_dim_mask(self.seam_guided_dimensions, self.num_valid_physical_dims, self.action_dim)

    # --- serialization ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeamConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - fields
        if unknown:
            raise SeamConfigError(f"unknown SEAM config keys: {sorted(unknown)}")
        kwargs = dict(data)
        gd = kwargs.get("seam_guided_dimensions")
        if isinstance(gd, list):
            kwargs["seam_guided_dimensions"] = tuple(gd)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "SeamConfig":
        import yaml  # local import; pyyaml is available in the openpi venv

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Allow a top-level "seam" section or a flat mapping.
        if "seam" in data and isinstance(data["seam"], dict):
            data = data["seam"]
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def build_dim_mask(
    guided_dimensions: GuidedDimsSpec,
    num_valid_physical_dims: int,
    action_dim: int,
) -> np.ndarray:
    """Build a boolean action-dimension mask of shape [action_dim].

    Supports:
      - "all_physical": first ``num_valid_physical_dims`` dims True.
      - a sequence of ints: those indices True.
      - a boolean sequence of length ``action_dim``: used as-is.
    """
    mask = np.zeros((action_dim,), dtype=bool)
    if isinstance(guided_dimensions, str):
        if guided_dimensions != "all_physical":
            raise SeamConfigError(f"unknown guided_dimensions string {guided_dimensions!r}")
        if not (0 < num_valid_physical_dims <= action_dim):
            raise SeamConfigError(
                f"num_valid_physical_dims={num_valid_physical_dims} out of range for D={action_dim}"
            )
        mask[:num_valid_physical_dims] = True
        return mask
    seq = list(guided_dimensions)
    if len(seq) == action_dim and all(isinstance(x, (bool, np.bool_)) for x in seq):
        return np.asarray(seq, dtype=bool)
    # treat as index list
    for idx in seq:
        i = int(idx)
        if not (0 <= i < action_dim):
            raise SeamConfigError(f"guided dimension index {i} out of range [0, {action_dim})")
        mask[i] = True
    if not mask.any():
        raise SeamConfigError("guided_dimensions resolved to an empty mask")
    return mask
