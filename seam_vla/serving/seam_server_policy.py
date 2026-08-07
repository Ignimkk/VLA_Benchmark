"""Server-side SEAM policy: a BasePolicy wrapper that applies VLS with per-session state.

Drops into ``WebsocketPolicyServer`` in place of a plain openpi ``Policy``. Because the websocket
server calls a single shared ``policy.infer(obs)`` per message, per-episode state is reset via an
explicit ``obs["seam_reset"]`` flag (the RB-Y1 client sends it on the first request of a run). The
model + denoising loop run here (server-side), which is the only place VLS can access model space.
"""

from __future__ import annotations

import logging
import time
from typing import Dict

import numpy as np

try:
    from openpi_client.base_policy import BasePolicy
except Exception:  # pragma: no cover - openpi_client always present where this runs
    class BasePolicy:  # minimal fallback so the module imports without openpi_client
        def infer(self, obs):  # noqa: D401
            raise NotImplementedError

        def reset(self):
            pass

from benchmark.seam_vla.policy.seam_policy import SeamPolicy, SeamPolicySession

logger = logging.getLogger(__name__)


class SeamServerPolicy(BasePolicy):
    """Wrap a :class:`SeamPolicy` as a websocket-servable BasePolicy with per-session state.

    Args:
        seam_policy: the constructed SEAM policy (already validated against the model).
        reset_key: obs key whose truthy value triggers a per-episode session reset.
        metadata: optional metadata surfaced to clients.
    """

    def __init__(self, seam_policy: SeamPolicy, *, reset_key: str = "seam_reset", metadata: dict | None = None):
        self._session = SeamPolicySession(seam_policy)
        self._reset_key = reset_key
        self._metadata = metadata or {}
        # Measuring the guided-vs-baseline output requires a second model sample.  Keep it opt-in and
        # do it only for the first guided chunk; reporting the default (unmeasured) value as 0.0 is
        # misleading because it looks like VLS made no correction.
        self._measure_first_correction = bool(seam_policy.config.seam_log_corrections)
        self._logged_first_guided = False

    @property
    def metadata(self) -> dict:
        return self._metadata

    def reset(self) -> None:
        self._session.reset()
        self._logged_first_guided = False

    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        obs = dict(obs)
        if obs.pop(self._reset_key, False):
            self.reset()

        measure_correction = self._measure_first_correction and not self._logged_first_guided
        t0 = time.perf_counter()
        chunk, diag = self._session.predict_chunk(obs, want_diagnostics=measure_correction)
        seam_ms = (time.perf_counter() - t0) * 1000.0

        correction_measured = bool(diag.used_vls and measure_correction)
        if diag.used_vls and not self._logged_first_guided:
            self._logged_first_guided = True
            if correction_measured:
                logger.info(
                    "[SEAM] first guided chunk: correction_norm=%.6e (measured) "
                    "model_shape=%s phys_shape=%s",
                    diag.correction_norm, diag.model_space_shape, diag.physical_space_shape,
                )
            else:
                logger.info(
                    "[SEAM] first guided chunk: guidance active; correction_norm=not_measured "
                    "(restart with --verify-first-correction to measure once) "
                    "model_shape=%s phys_shape=%s",
                    diag.model_space_shape, diag.physical_space_shape,
                )

        return {
            "actions": np.asarray(chunk),
            "seam_timing": {"seam_ms": seam_ms, "used_vls": bool(diag.used_vls),
                            "chunk_index": int(diag.chunk_index),
                            "correction_norm_measured": correction_measured,
                            "correction_norm": float(diag.correction_norm) if correction_measured else None},
        }
