"""Pre-flight check for RB-Y1 SEAM deployment (read-only; run on the GPU server).

Loads the SERVER's pi05_rby1_lora config and prints the facts SEAM depends on, so the base-relative-
delta compensation and guided-dim choices can be confirmed against the server-only config.py before
serving. Does NOT load the model weights.

    python -m benchmark.seam_vla.serving.preflight_rby1 --config pi05_rby1_lora \
        --dir /mnt/dev/work/pi05_TO_hybrid/checkpoints/pi05_rby1_lora/full_run_30k/29999 \
        --seam-config benchmark/seam_vla/configs/seam_rby1.yaml
"""

from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_rby1_lora")
    ap.add_argument("--dir", required=True, help="checkpoint dir (for norm stats)")
    ap.add_argument("--seam-config", default="benchmark/seam_vla/configs/seam_rby1.yaml")
    args = ap.parse_args()

    from openpi.training import config as _config
    from openpi.shared import download
    from openpi.training import checkpoints as _checkpoints
    from benchmark.seam_vla.config import SeamConfig

    seam = SeamConfig.from_yaml(args.seam_config)
    tc = _config.get_config(args.config)
    print("=== model ===")
    print(f"action_horizon (H) = {tc.model.action_horizon}   (seam_horizon={seam.seam_horizon})")
    print(f"action_dim     (D) = {tc.model.action_dim}   (seam action_dim={seam.action_dim})")

    dc = tc.data.create(tc.assets_dirs, tc.model)
    print(f"asset_id           = {dc.asset_id}")
    print(f"use_quantile_norm  = {getattr(dc, 'use_quantile_norm', '?')}")

    print("\n=== data_transforms.outputs (run at inference; look for AbsoluteActions + its mask) ===")
    for t in dc.data_transforms.outputs:
        mask = getattr(t, "mask", None)
        print(f"  {type(t).__name__:20s} mask={None if mask is None else list(np.asarray(mask).astype(int))}")
    print("=== data_transforms.inputs (training; DeltaActions mask marks delta dims) ===")
    for t in dc.data_transforms.inputs:
        mask = getattr(t, "mask", None)
        print(f"  {type(t).__name__:20s} mask={None if mask is None else list(np.asarray(mask).astype(int))}")

    ckpt = download.maybe_download(args.dir)
    ns = _checkpoints.load_norm_stats(ckpt / "assets", dc.asset_id)
    act = ns["actions"]
    inv = 2.0 / (np.asarray(act.q99) - np.asarray(act.q01) + 1e-6)
    print(f"\n=== action norm stats ({len(act.q01)}-dim) ===")
    print(f"q01 len={len(act.q01)}  q99 len={len(act.q99)}  inv_scale(2/(q99-q01)) first14="
          f"{np.round(inv[:14], 3).tolist()}")

    print("\n=== SEAM guided dims (compensation applied to these) ===")
    m = seam.build_dim_mask()
    print(f"guided indices = {np.where(m)[0].tolist()}  (expect arm joints 0-5,7-12; grippers 6,13 EXCLUDED)")

    print("\nCHECK:")
    print(f"  [{'OK' if tc.model.action_horizon == seam.seam_horizon else 'MISMATCH'}] H matches seam_horizon")
    print(f"  [{'OK' if tc.model.action_dim == seam.action_dim else 'MISMATCH'}] D matches action_dim")
    has_abs = any("Absolute" in type(t).__name__ for t in dc.data_transforms.outputs)
    print(f"  [{'OK' if has_abs == seam.seam_delta_base_compensation else 'CHECK'}] "
          f"AbsoluteActions present={has_abs} vs seam_delta_base_compensation={seam.seam_delta_base_compensation}")
    print("  -> If AbsoluteActions present: model space is delta, compensation is correct (keep true).")
    print("  -> If NOT present (model outputs absolute): set seam_delta_base_compensation: false.")


if __name__ == "__main__":
    main()
