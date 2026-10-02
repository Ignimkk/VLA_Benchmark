"""Decorates a policy so its `infer()` result also carries a per-camera attention map.

**Why a second model instance.** `Pi0Config.return_attn_probs=True` bakes `return_probs=True` into
`gemma.Module` at construction time, and once that is on, `PaliGemma.llm(...)` returns a 3-tuple
everywhere it's called. `Pi0.sample_actions` (`pi0.py:245` and `:269`) still unpacks a 2-tuple, so
turning the flag on for the policy that serves actions breaks serving outright. Fixing that is an
openpi change, so instead: load a **second** copy of the same checkpoint with the flag on, and use it
only through `AttentionSampler` (`benchmark/ag3s/experiments/sources/pi05_attention.py`), which never calls
`sample_actions` — it drives `PaliGemma.llm` directly with the 3-tuple unpack it already expects.
That sampler is the same one `step-01-attention.json` was scored with; nothing here is reimplemented.

**Why this can't perturb the served actions.** The policy that actually produces `result["actions"]`
is the caller's own, untouched, called exactly as it would be without this wrapper. Attention comes
from a wholly separate model object, a separate forward pass, and is merged into the result dict
afterward. There is no shared state that a bug here could feed back into the action path.

**Subtask label (SUBTASK-c).** The same prefix pass also yields the `kv_L4` feature (prefix KV
cache, no extra forward), and the exported linear probe (`benchmark/ag3s/asset/subtask_probe/`,
`benchmark/ag3s/stages/subtask_probe.py`) turns it into `result["subtask"] =
{"p": {"pick", "place", "home"}, "argmax": str, "probe": "kv_L4_v1"}`. Like the attention, a
failure there is logged and leaves the key out; it never fails the policy call.

**Cost.** Only Euler step 0 is sampled — the (layer, head, agg) cell `step-01-attention.json` scored
best used `denoise=0` — so this is one prefix+suffix pass, not the full ~10-step denoising loop.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

import numpy as np

__all__ = ["AttentionPolicy", "load_attention_model"]

#: `benchmark/ag3s/docs/archive/step-verification-20260904/step-01-attention.json` -> `best`.
_DENOISE_STEP = 0
_AGG = "last"
_LAYER = 8
_HEAD = 2


def load_attention_model(config_name: str, checkpoint: str):
    """A `Pi0` model from the same checkpoint, with `return_attn_probs=True`.

    Returns the bare model (not a `Policy`) — that's all `AttentionSampler` needs.
    """
    from benchmark.ag3s.experiments.sources.pi05_attention import load_policy

    return load_policy(config_name, checkpoint)._model  # noqa: SLF001


class AttentionPolicy:
    """Wraps `policy` so `infer()`'s result carries `result["attention"]`.

    `attention` is `{policy camera name: [16, 16] float32}` — raw softmax mass, not normalized or
    thresholded (`AG3S` does that itself, and it needs the peak on the true scale to do it).

    `subtask_probe`: `"default"` loads the shipped `kv_L4_v1` asset, a path loads that asset, a
    `SubtaskProbe` is used as is, None turns the label off (no `result["subtask"]`). A probe that
    fails to load is logged and turned off — the attention path is unaffected.
    """

    def __init__(self, policy, attn_model, *, num_denoise_steps: int = 10, noise_seed: int = 0,
                 subtask_probe: Any = "default"):
        from benchmark.ag3s.experiments.sources.pi05_attention import AttentionSampler

        self._policy = policy
        self._sampler = AttentionSampler(attn_model, num_steps=num_denoise_steps)
        self._noise_seed = int(noise_seed)
        self._subtask_probe = _load_probe(subtask_probe)

    @property
    def metadata(self) -> dict[str, Any]:
        return getattr(self._policy, "metadata", {})

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    @property
    def subtask_probe(self):
        """The probe in use (`SubtaskProbe`), or None when the label is off."""
        return self._subtask_probe

    def infer(self, obs: dict[str, Any], **kwargs) -> dict[str, Any]:
        result = self._policy.infer(obs, **kwargs)
        try:
            result = dict(result)
            attention, feature = self._attention_and_feature(obs)
            result["attention"] = attention
        except Exception:  # noqa: BLE001 — attention failing must not fail the policy call
            logging.warning("attention extraction failed:\n%s", traceback.format_exc())
            return result
        if feature is not None:
            try:
                result["subtask"] = self._subtask_probe.predict(feature)
            except Exception:  # noqa: BLE001 — the label failing must not fail the policy call
                logging.warning("subtask probe failed:\n%s", traceback.format_exc())
        return result

    def _attention(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return self._attention_and_feature(obs, with_feature=False)[0]

    def _attention_and_feature(self, obs: dict[str, Any], *, with_feature: Optional[bool] = None):
        """`(attention dict, kv_L4 feature or None)` from one prefix pass."""
        import jax
        import jax.numpy as jnp

        from benchmark.ag3s.experiments.sources.pi05_attention import AGGREGATIONS
        from benchmark.ag3s.experiments.sources.policy_record import (
            ATTENTION_GRID, CAMERA_BINDINGS, POLICY_CAMERA_NAMES)
        from openpi.models import model as _model

        inputs = self._policy._input_transform(dict(obs))  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        model_obs = _model.Observation.from_dict(inputs)

        if with_feature is None:
            with_feature = self._subtask_probe is not None
        if with_feature:
            blocks, feature = self._sampler.attention_and_subtask_feature(
                model_obs, [_DENOISE_STEP], noise_seed=self._noise_seed)
        else:
            blocks = self._sampler.attention(model_obs, [_DENOISE_STEP],
                                             noise_seed=self._noise_seed)
            feature = None
        pooled = AGGREGATIONS[_AGG](blocks[_DENOISE_STEP])  # [L, heads, 768]
        cell = np.asarray(pooled[_LAYER, _HEAD])  # [768]

        out = {}
        for camera in POLICY_CAMERA_NAMES:
            lo, hi = CAMERA_BINDINGS[camera][2]
            out[camera] = cell[lo:hi].reshape(ATTENTION_GRID, ATTENTION_GRID)
        return out, feature


def _load_probe(spec: Any):
    """`AttentionPolicy(subtask_probe=...)` → a `SubtaskProbe` or None (logged, never raised)."""
    if spec is None or spec is False:
        return None
    try:
        from benchmark.ag3s.stages.subtask_probe import SubtaskProbe

        if isinstance(spec, SubtaskProbe):
            return spec
        probe = SubtaskProbe.load(None if spec == "default" else spec)
        logging.info("subtask probe %s loaded (classes %s, dim %d)", probe.name,
                     list(probe.classes), probe.dim)
        return probe
    except Exception:  # noqa: BLE001 — no label is a degraded mode, not a failed server
        logging.warning("subtask probe could not be loaded (%r) — result['subtask'] is off:\n%s",
                        spec, traceback.format_exc())
        return None
