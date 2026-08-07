"""SEAM policy: orchestrates transforms + jitted sampler + per-session state.

Wraps an OpenPI ``Policy`` (for the model, input/output transforms, and RNG) and adds VLS guidance in
model space plus a physical-space :class:`DecodedChunkRefiner` (identity for this work). Exposes a
pure state-passing ``predict_chunk`` and a stateful ``SeamPolicySession`` wrapper with ``reset()``.

Pipeline per chunk:
    input transform -> Observation
      -> [first chunk / disabled] baseline sample   (model space)
         [else]                  VLS-guided sample  (model space, aligned prior from prev tail)
      -> Unnormalize + output transforms            (physical space)
      -> DecodedChunkRefiner(identity)              (physical space)
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.seam_vla.config import SeamConfig
from benchmark.seam_vla.diagnostics import correction_norm
from benchmark.seam_vla.guidance.vls import make_position_mask
from benchmark.seam_vla.integration.openpi_jax import SeamSampler
from benchmark.seam_vla.priors.aligned_tail import (
    build_aligned_prior,
    build_aligned_prior_delta_compensated,
)
from benchmark.seam_vla.refinement.base import DecodedChunkRefiner
from benchmark.seam_vla.refinement.identity import IdentityChunkRefiner
from benchmark.seam_vla.state import SeamState

# Imported lazily-friendly: only needed for Observation.from_dict.
from openpi.models import model as _model


@dataclasses.dataclass
class ChunkDiagnostics:
    chunk_index: int
    used_vls: bool
    correction_norm: float = 0.0
    model_space_shape: tuple = ()
    physical_space_shape: tuple = ()


class SeamPolicy:
    """Stateless-per-call SEAM policy (state passed in/out explicitly)."""

    def __init__(
        self,
        openpi_policy,
        config: SeamConfig,
        *,
        refiner: Optional[DecodedChunkRefiner] = None,
        rollout_execution_length: int | None = None,
        action_inv_scale=None,
        proprio_key: str = "state",
    ):
        self._policy = openpi_policy
        self._model = openpi_policy._model
        self._input_transform = openpi_policy._input_transform
        self._output_transform = openpi_policy._output_transform
        self.config = config
        self.refiner = refiner or IdentityChunkRefiner()
        self._proprio_key = proprio_key

        model_H = int(self._model.action_horizon)
        model_D = int(self._model.action_dim)
        # num_steps: honor openpi sample_kwargs override if present, else default 10.
        self._num_steps = int(openpi_policy._sample_kwargs.get("num_steps", 10))

        # Fail loudly if the config disagrees with the actual model / rollout.
        config.assert_matches_model(
            model_horizon=model_H,
            model_num_steps=self._num_steps,
            rollout_execution_length=rollout_execution_length,
        )
        self.H, self.D, self.N = model_H, model_D, self._num_steps
        self.K = config.seam_execution_length
        self.L = self.H - self.K
        self.M = config.effective_window

        pos_mask = make_position_mask(self.H, self.M)
        self._dim_mask = config.build_dim_mask()
        self._sampler = SeamSampler(
            self._model, num_steps=self.N, pos_mask=pos_mask, dim_mask=self._dim_mask
        )
        self._rng = getattr(openpi_policy, "_rng", jax.random.key(0))

        # Base-relative-delta compensation (rby1). inv_scale[d] = 2/(q99-q01) of the ACTION norm stats.
        self._compensate = bool(config.seam_delta_base_compensation)
        if self._compensate:
            if action_inv_scale is None:
                raise ValueError(
                    "seam_delta_base_compensation=True requires action_inv_scale (2/(q99-q01) per dim); "
                    "pass it from the loaded norm stats."
                )
            iv = np.zeros((self.D,), dtype=np.float32)
            src = np.asarray(action_inv_scale, dtype=np.float32).reshape(-1)
            iv[: src.shape[0]] = src[: self.D]
            self._inv_scale = jnp.asarray(iv)

    # --- helpers ------------------------------------------------------------
    def _to_observation(self, obs_dict: dict):
        inputs = self._input_transform(jax.tree.map(lambda x: x, obs_dict))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return inputs, _model.Observation.from_dict(inputs)

    def _extract_proprio(self, obs_dict: dict) -> np.ndarray:
        """Return the raw physical proprio vector from the observation (for delta compensation).

        For rby1 the observation carries a 14-dim ``state`` in the same layout as the action vector,
        so guided action dims map directly onto proprio dims.
        """
        if self._proprio_key not in obs_dict:
            raise KeyError(
                f"proprio key {self._proprio_key!r} not in observation; needed for "
                f"seam_delta_base_compensation. Available keys: {list(obs_dict)}"
            )
        return np.asarray(obs_dict[self._proprio_key], dtype=np.float32).reshape(-1)

    def _to_physical(self, model_actions_1hd, state_1s):
        # Mirror Policy.infer: strip batch, run output transform (Unnormalize + LiberoOutputs).
        outputs = {"state": np.asarray(state_1s[0]), "actions": np.asarray(model_actions_1hd[0])}
        outputs = self._output_transform(outputs)
        return np.asarray(outputs["actions"])

    # --- main API -----------------------------------------------------------
    def predict_chunk(
        self,
        obs_dict: dict,
        seam_state: SeamState,
        *,
        rng=None,
        noise=None,
        want_diagnostics: bool = True,
    ):
        """Predict one chunk.

        Returns ``(physical_chunk, new_state, diagnostics)`` where ``physical_chunk`` is ``[H, 7]``.
        """
        if rng is None:
            self._rng, rng = jax.random.split(self._rng)
        inputs, obs = self._to_observation(obs_dict)
        state_1s = inputs["state"]
        # Raw physical proprio (pre-normalization), for base-relative-delta compensation.
        curr_proprio = self._extract_proprio(obs_dict) if self._compensate else None

        first = seam_state.is_first_chunk
        use_vls = self.config.seam_enabled and (not first) and self.config.seam_lambda > 0 and self.M > 0

        if not use_vls:
            model_chunk = self._sampler.sample_baseline(rng, obs, noise)
            corr_norm = 0.0
        else:
            prev_model = jnp.asarray(seam_state.previous_chunk_model_space)[None]  # [1, H, D]
            if self._compensate and seam_state.previous_proprio_state is not None:
                delta_state = np.asarray(curr_proprio) - np.asarray(seam_state.previous_proprio_state)
                delta_state_d = np.zeros((self.D,), dtype=np.float32)
                delta_state_d[: delta_state.shape[0]] = delta_state[: self.D]
                prior = build_aligned_prior_delta_compensated(
                    prev_model, jnp.asarray(delta_state_d)[None], self._inv_scale, self._dim_mask,
                    self.H, self.K,
                )
            else:
                prior = build_aligned_prior(prev_model, self.H, self.K)
            model_chunk = self._sampler.sample_seam(
                rng, obs, prior, self.config.seam_lambda, 1.0, noise
            )
            corr_norm = 0.0
            if want_diagnostics and self.config.seam_log_corrections:
                base_chunk = self._sampler.sample_baseline(rng, obs, noise)
                corr_norm = correction_norm(base_chunk, model_chunk)

        model_chunk_np = np.asarray(jax.block_until_ready(model_chunk))
        physical_chunk = self._to_physical(model_chunk_np, np.asarray(state_1s))
        physical_chunk = np.asarray(
            self.refiner.refine(
                physical_chunk,
                context={
                    "chunk_index": seam_state.chunk_index,
                    "previous_physical_chunk": seam_state.previous_chunk_physical_space,
                },
            )
        )

        # Store model-space chunk (strip batch) for the next chunk's prior; physical for logging;
        # proprio for the next chunk's compensation.
        new_state = seam_state.with_chunk(model_chunk_np[0], physical_chunk, proprio_state=curr_proprio)
        diag = ChunkDiagnostics(
            chunk_index=seam_state.chunk_index,
            used_vls=use_vls,
            correction_norm=float(corr_norm),
            model_space_shape=tuple(model_chunk_np.shape),
            physical_space_shape=tuple(physical_chunk.shape),
        )
        return physical_chunk, new_state, diag


class SeamPolicySession:
    """Stateful single-environment wrapper around :class:`SeamPolicy`."""

    def __init__(self, seam_policy: SeamPolicy):
        self._policy = seam_policy
        self._state = SeamState.initial()

    @property
    def state(self) -> SeamState:
        return self._state

    def reset(self) -> None:
        self._state = self._state.reset()

    def predict_chunk(self, obs_dict: dict, **kwargs):
        physical_chunk, self._state, diag = self._policy.predict_chunk(obs_dict, self._state, **kwargs)
        return physical_chunk, diag
