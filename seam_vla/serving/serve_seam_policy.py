"""Standalone SEAM-enabled policy server (no OpenPI edits).

Builds an openpi policy from a trained checkpoint, wraps it with server-side VLS (SeamServerPolicy),
and serves it over the same websocket protocol as ``scripts/serve_policy.py``. Deploy this on the GPU
server where the rby1 config/checkpoint live.

Example (RB-Y1):
    python -m benchmark.seam_vla.serving.serve_seam_policy \
        --config pi05_rby1_lora \
        --dir /mnt/dev/work/pi05_TO_hybrid/checkpoints/pi05_rby1_lora/full_run_30k/29999 \
        --seam-config benchmark/seam_vla/configs/seam_rby1.yaml \
        --port 8000
"""

from __future__ import annotations

import argparse
import dataclasses
import logging

from benchmark.seam_vla.config import SeamConfig
from benchmark.seam_vla.factory import build_seam_policy
from benchmark.seam_vla.serving.seam_server_policy import SeamServerPolicy


def build_server_policy(
    config_name: str,
    checkpoint_dir: str,
    seam_config_path: str,
    *,
    verify_first_correction: bool = False,
):
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from openpi.training import config as _config

    seam_cfg = SeamConfig.from_yaml(seam_config_path)
    if verify_first_correction:
        seam_cfg = dataclasses.replace(seam_cfg, seam_log_corrections=True)
    train_config = _config.get_config(config_name)
    ckpt_dir = download.maybe_download(checkpoint_dir)

    openpi_policy = _policy_config.create_trained_policy(train_config, ckpt_dir)
    seam_policy = build_seam_policy(openpi_policy, train_config, ckpt_dir, seam_cfg)
    logging.info(
        "[SEAM] server ready: H=%d K=%d L=%d M=%d N=%d D=%d guided_dims=%d compensation=%s",
        seam_policy.H, seam_policy.K, seam_policy.L, seam_policy.M, seam_policy.N, seam_policy.D,
        int(seam_policy._dim_mask.sum()), seam_cfg.seam_delta_base_compensation,
    )
    return SeamServerPolicy(seam_policy, metadata={"seam": seam_cfg.to_dict()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="training config name, e.g. pi05_rby1_lora")
    ap.add_argument("--dir", required=True, help="checkpoint directory")
    ap.add_argument("--seam-config", required=True, help="path to a SEAM yaml (e.g. configs/seam_rby1.yaml)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--verify-first-correction",
        action="store_true",
        help="run one extra baseline sample for the first guided chunk and log its measured "
             "guided-vs-baseline correction norm",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    from openpi.serving import websocket_policy_server

    policy = build_server_policy(
        args.config,
        args.dir,
        args.seam_config,
        verify_first_correction=args.verify_first_correction,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    )
    logging.info("Serving SEAM policy on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
