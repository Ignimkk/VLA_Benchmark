"""Identity guidance: returns the Euler candidate unchanged (exact baseline)."""

from __future__ import annotations

from benchmark.seam_vla.guidance.base import DenoisingGuidance


class IdentityGuidance(DenoisingGuidance):
    """No-op guidance. ``update`` returns the candidate exactly.

    Used for the first chunk / after reset, and to prove baseline parity.
    """

    def update(self, candidate, t_next):
        return candidate
