"""OpenPI denoising integration for SEAM.

Builds a single jitted sampling function around ``Pi0.sample_actions`` that injects VLS as the
``guidance_fn`` post-Euler hook. The aligned prior, guidance strength, and enable-gate are passed as
**traced arrays** so one compilation is reused across chunks; the position/dim masks and ``num_steps``
are static constants captured at build time.

Mirrors ``openpi.shared.nnx_utils.module_jit``: the NNX module is split into (graphdef, state) once,
and ``state`` is threaded as the first traced argument.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from benchmark.seam_vla.guidance.vls import VLSGuidance


class SeamSampler:
    """Jitted π0.5 sampler with an optional VLS post-Euler correction.

    Args:
        model: the ``Pi0`` nnx module (already parameter-loaded).
        num_steps: N, the number of Euler steps (static).
        pos_mask: boolean ``[H]`` mask for guided overlap rows (static constant).
        dim_mask: boolean ``[D]`` mask for guided action dims (static constant).
    """

    def __init__(self, model, *, num_steps: int, pos_mask: np.ndarray, dim_mask: np.ndarray):
        self._graphdef, self._state = nnx.split(model)
        self._num_steps = int(num_steps)
        self._pos_mask = jnp.asarray(np.asarray(pos_mask, dtype=bool))
        self._dim_mask = jnp.asarray(np.asarray(dim_mask, dtype=bool))
        self._jit_seam = jax.jit(self._call_seam)
        self._jit_baseline = jax.jit(self._call_baseline)

    # --- traced bodies ------------------------------------------------------
    def _call_seam(self, state, rng, observation, noise, aligned_prior, lam, enabled):
        model = nnx.merge(self._graphdef, state)
        guidance = VLSGuidance(aligned_prior, lam, self._pos_mask, self._dim_mask, enabled)
        return model.sample_actions(
            rng,
            observation,
            num_steps=self._num_steps,
            noise=noise,
            guidance_fn=guidance.as_fn(),
        )

    def _call_baseline(self, state, rng, observation, noise):
        model = nnx.merge(self._graphdef, state)
        return model.sample_actions(
            rng,
            observation,
            num_steps=self._num_steps,
            noise=noise,
            guidance_fn=None,
        )

    # --- public API ---------------------------------------------------------
    def sample_baseline(self, rng, observation, noise=None):
        """Sample a model-space chunk with no guidance (exact baseline)."""
        return self._jit_baseline(self._state, rng, observation, noise)

    def sample_seam(self, rng, observation, aligned_prior, lam, enabled, noise=None):
        """Sample a model-space chunk with VLS guidance.

        ``aligned_prior`` ``[..., H, D]``, ``lam`` and ``enabled`` scalars (traced). Set ``enabled=0``
        (or ``lam=0``) to recover the baseline exactly.
        """
        return self._jit_seam(
            self._state,
            rng,
            observation,
            noise,
            jnp.asarray(aligned_prior),
            jnp.asarray(lam, dtype=jnp.float32),
            jnp.asarray(enabled, dtype=jnp.float32),
        )
