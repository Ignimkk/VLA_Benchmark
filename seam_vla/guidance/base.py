"""DenoisingGuidance interface.

A ``DenoisingGuidance`` operates on the **model-space** post-Euler candidate at each flow-matching
step. It is injected into the OpenPI sampler as a plain callable ``(candidate, t_next) -> corrected``
via :meth:`DenoisingGuidance.as_fn`, so the sampler never imports benchmark code.

Implementations must be pure/JAX-traceable (the sampler runs inside ``jax.lax.while_loop`` under
``jax.jit``): no Python-side mutable state, no host transfers, static shapes.
"""

from __future__ import annotations

import abc
from typing import Callable


class DenoisingGuidance(abc.ABC):
    """Post-Euler guidance applied during flow-matching denoising (model space)."""

    @abc.abstractmethod
    def update(self, candidate, t_next):
        """Return the corrected denoising state.

        Args:
            candidate: the standard Euler candidate ``x_t + dt * v_t``, shape ``[..., H, D]``.
            t_next: the ODE time of ``candidate`` (scalar, in [0, 1]; t=1 noise, t=0 target).

        Returns:
            Corrected state of the same shape/dtype as ``candidate``.
        """

    def as_fn(self) -> Callable:
        """Return a plain closure ``(candidate, t_next) -> corrected`` for the sampler hook."""
        return lambda candidate, t_next: self.update(candidate, t_next)
