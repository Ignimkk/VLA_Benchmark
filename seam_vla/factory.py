"""Factory helpers to build a SeamPolicy from a loaded openpi policy (shared by server + client-local).

Keeps the norm-stat -> inv_scale derivation (for base-relative-delta compensation) in one place.
"""

from __future__ import annotations

import numpy as np

from benchmark.seam_vla.config import SeamConfig
from benchmark.seam_vla.policy.seam_policy import SeamPolicy


def action_inv_scale(train_config, ckpt_dir) -> np.ndarray:
    """inv_scale[d] = 2/(q99-q01) from the ACTION quantile norm stats (base-delta compensation)."""
    from openpi.training import checkpoints as _checkpoints

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("asset_id required to load norm stats for compensation")
    norm_stats = _checkpoints.load_norm_stats(ckpt_dir / "assets", data_config.asset_id)
    act = norm_stats["actions"]
    q01 = np.asarray(act.q01, dtype=np.float32)
    q99 = np.asarray(act.q99, dtype=np.float32)
    return 2.0 / (q99 - q01 + 1e-6)


def build_seam_policy(openpi_policy, train_config, ckpt_dir, seam_cfg: SeamConfig) -> SeamPolicy:
    """Wrap an already-loaded openpi Policy with SEAM, computing inv_scale if compensation is on."""
    inv_scale = action_inv_scale(train_config, ckpt_dir) if seam_cfg.seam_delta_base_compensation else None
    return SeamPolicy(
        openpi_policy,
        seam_cfg,
        rollout_execution_length=seam_cfg.seam_execution_length,
        action_inv_scale=inv_scale,
    )
