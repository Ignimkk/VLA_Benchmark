"""Shared, cached loader for the pi05_libero policy (heavy). Used by guarded model tests."""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.seam_vla.tests.conftest import CKPT

H, D, N, K, NVALID = 10, 32, 10, 5, 7


@functools.lru_cache(maxsize=1)
def load_policy():
    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc

    cfg = _config.get_config("pi05_libero")
    return _pc.create_trained_policy(cfg, CKPT)


@functools.lru_cache(maxsize=1)
def make_obs_and_noise():
    from openpi.policies import libero_policy
    from openpi.models import model as _model

    policy = load_policy()
    example = libero_policy.make_libero_example()
    inputs = policy._input_transform(dict(example))
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    obs = _model.Observation.from_dict(inputs)
    noise = jax.random.normal(jax.random.key(123), (1, H, D))
    return obs, noise, inputs


def build_sampler(guided_window=5):
    from benchmark.seam_vla.integration.openpi_jax import SeamSampler
    from benchmark.seam_vla.guidance.vls import make_position_mask
    from benchmark.seam_vla.config import build_dim_mask

    policy = load_policy()
    pos = make_position_mask(H, min(guided_window, H - K))
    dim = build_dim_mask("all_physical", NVALID, D)
    return SeamSampler(policy._model, num_steps=N, pos_mask=pos, dim_mask=dim)
