"""Phase x manipulator x source x link -> required clearance.

This module replaces the single most dangerous thing AG3S used to do. The old contract expressed
"the gripper may now touch the cup" by setting `collision_enabled=False` on the target candidate,
which did not relax a margin — it **removed the target's constraint rows from the NLP entirely**.
For the duration of the grasp the optimizer was free to drive the left arm, the forearm and the
torso straight through the object, because as far as the solver could see the object was not there.

The fix is the distinction the design document already argued for and the implementation had not
caught up with:

    existence of a constraint   <- geometry, always
    required clearance of it    <- phase, manipulator, source and robot link

So the target keeps its slot in every phase. What changes at `GRASP` is one number in one cell of a
`(robot sphere, candidate slot)` margin matrix: the right fingertips get `contact_margin`, which is
zero by default. **Zero is not "off".** The row `||p_robot(q) - p_cand||^2 - (r_r + r_c + 0)^2 >= 0`
still forbids penetration; it merely permits touching. Every other cell in that column — forearm,
torso, opposite arm, and any link nobody has heard of — keeps the full safety margin.

Everything unknown fails closed. An unrecognised phase, source type, link name, or manipulator does
not produce an exception and does not produce a relaxation: it produces `safety_margin`. The reason
is asymmetric cost. A wrongly-tight margin makes the robot take a detour; a wrongly-loose one makes
it hit something. Only a *successfully grounded* target on an *explicitly named* link of an
*explicitly authorized* manipulator is ever relaxed, and `OBJECT`, `UNKNOWN_GEOMETRY`, `OVERFLOW`
and support surfaces are never relaxed at all.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np

from benchmark.ag3s.config import ContactConfig, GeometryConfig, SupportSurfaceConfig
from benchmark.ag3s.types import (
    ContactPolicyContext,
    Manipulator,
    Phase,
    RobotCollisionModel,
    SourceType,
)

#: What a sphere's link is called when the robot model does not say. Chosen to be a name no URDF
#: uses, so it can never accidentally appear in a contact allowlist.
UNKNOWN_LINK = "<unknown-link>"


def resolve_link_names(model: Optional[RobotCollisionModel], n_spheres: int) -> tuple[str, ...]:
    """One link name per sphere, padded or truncated to `n_spheres`.

    A model predating `sphere_link_names`, or one whose list has drifted out of sync with its sphere
    count, yields `UNKNOWN_LINK` rather than an error. That is deliberate: the policy treats an
    unknown link as non-contact, so a stale model loses the ability to grasp but never the ability to
    avoid. Failing loudly here would take out an otherwise working perception pipeline over a
    metadata gap; failing closed degrades exactly one capability.
    """
    names = getattr(model, "sphere_link_names", None)
    if names is None:
        return (UNKNOWN_LINK,) * n_spheres
    resolved = [str(n) for n in names]
    if len(resolved) < n_spheres:
        resolved.extend([UNKNOWN_LINK] * (n_spheres - len(resolved)))
    return tuple(resolved[:n_spheres])


@dataclasses.dataclass(frozen=True)
class ClearancePolicy:
    """The lookup table `d(q; link_i, candidate_j) >= m(phase, manipulators, source, link)`.

    Constructed from the config sections that already own these numbers, so there is one place a
    margin can come from. `safety_margin` is the full clearance for object-like geometry;
    `support_margin` for planes, which have their own value because a table is fitted far more
    precisely than a cluster is.
    """

    contact: ContactConfig = dataclasses.field(default_factory=ContactConfig)
    safety_margin: float = 0.05
    support_margin: float = 0.01

    @classmethod
    def from_config(
        cls,
        contact: ContactConfig | None = None,
        geometry: GeometryConfig | None = None,
        support_surface: SupportSurfaceConfig | None = None,
    ) -> "ClearancePolicy":
        geo = geometry or GeometryConfig()
        sup = support_surface or SupportSurfaceConfig()
        return cls(
            contact=contact or ContactConfig(),
            safety_margin=float(geo.safety_margin),
            support_margin=float(sup.safety_margin),
        )

    # --- the full margin for a source, before any relaxation is considered ---------------
    def full_margin(self, source: SourceType | str) -> float:
        """Clearance for `source` with no contact authorization anywhere. Never relaxed below this
        except by the one explicitly-authorized target path in `lookup`."""
        try:
            parsed = SourceType.parse(source)
        except ValueError:
            return self.safety_margin  # an unnameable source is treated as solid geometry
        if parsed is SourceType.SUPPORT_SURFACE:
            return self.support_margin
        return self.safety_margin

    # --- contact authorization ----------------------------------------------------------
    def authorized_links(self, context: ContactPolicyContext) -> frozenset[str]:
        """Links this context permits to touch the target. Empty when nobody is authorized.

        The union over the active manipulators, so a bimanual grasp needs no separate code path — it
        is `{LEFT, RIGHT}` and the union is both allowlists.
        """
        links: set[str] = set()
        for manipulator in context.active_manipulators:
            links.update(self.contact.links_for(manipulator))
        return frozenset(links)

    def lookup(
        self,
        *,
        phase: Phase | str,
        candidate_source: SourceType | str,
        robot_link: str,
        active_manipulators: Sequence[Manipulator | str] | frozenset | None = None,
        target_grounded: bool = True,
    ) -> float:
        """Required surface clearance for one (robot link, candidate) pair, in metres.

        Returns `contact.contact_margin` only when **all** of the following hold. Any one of them
        failing returns the full margin:

        * the candidate is a successfully grounded `TARGET`,
        * `robot_link` is in the allowlist of an active manipulator,
        * the phase rule for `phase` sets `contact_permission`,
        * `contact.phase_aware` is on.

        With `contact_permission` off but the link still authorized, the phase's `margin_scale`
        applies — the graded approach (1.0 -> 0.4 -> 0.1) that lets the gripper close in before
        contact is permitted. Note that this graded relaxation is also scoped to authorized links:
        the torso never gets it.
        """
        context = ContactPolicyContext.make(phase, active_manipulators)
        try:
            source = SourceType.parse(candidate_source)
        except ValueError:
            return self.safety_margin

        base = self.full_margin(source)
        if source is not SourceType.TARGET or not target_grounded:
            return base
        if not self.contact.phase_aware:
            return base
        if str(robot_link) not in self.authorized_links(context):
            return base

        rule = self.contact.rule_for(context.phase)
        if rule.contact_permission:
            return float(self.contact.contact_margin)
        return float(base * rule.margin_scale)

    # --- the matrix the constraint builder consumes -------------------------------------
    def margin_matrix(
        self,
        link_names: Sequence[str],
        sources: Sequence[SourceType | str],
        *,
        context: ContactPolicyContext,
        target_grounded: bool = True,
    ) -> np.ndarray:
        """`(len(link_names), len(sources))` of required clearances.

        Built column-wise rather than cell-by-cell: every non-target column is one constant, and the
        target column differs only between authorized and unauthorized links. On RB-Y1's 61-sphere
        constraint model with 32 slots that is 1,952 cells, and computing them with a Python call
        each would put the policy on the per-frame hot path for no benefit.
        """
        links = [str(n) for n in link_names]
        out = np.empty((len(links), len(sources)), np.float64)
        authorized = self.authorized_links(context)
        rule = self.contact.rule_for(context.phase)
        is_authorized = np.asarray([name in authorized for name in links], bool)

        for j, source in enumerate(sources):
            try:
                parsed = SourceType.parse(source)
            except ValueError:
                parsed = SourceType.UNKNOWN_GEOMETRY
            base = self.full_margin(parsed)
            out[:, j] = base
            if parsed is not SourceType.TARGET or not target_grounded:
                continue
            if not self.contact.phase_aware or not authorized:
                continue
            relaxed = (
                float(self.contact.contact_margin)
                if rule.contact_permission
                else float(base * rule.margin_scale)
            )
            out[is_authorized, j] = relaxed
        return out

    # --- diagnostics --------------------------------------------------------------------
    def describe(self, context: ContactPolicyContext, *, target_grounded: bool = True) -> dict:
        """A JSON-friendly record of what this context actually authorizes, for logs and reports."""
        authorized = sorted(self.authorized_links(context))
        rule = self.contact.rule_for(context.phase)
        return {
            "phase": context.phase.value,
            "active_manipulators": sorted(m.value for m in context.active_manipulators),
            "phase_aware": self.contact.phase_aware,
            "target_grounded": bool(target_grounded),
            "authorized_links": authorized,
            "contact_permission": bool(rule.contact_permission and target_grounded and authorized),
            "contact_margin": float(self.contact.contact_margin),
            "target_margin_authorized": self.lookup(
                phase=context.phase,
                candidate_source=SourceType.TARGET,
                robot_link=authorized[0] if authorized else UNKNOWN_LINK,
                active_manipulators=context.active_manipulators,
                target_grounded=target_grounded,
            ),
            "target_margin_other": self.lookup(
                phase=context.phase,
                candidate_source=SourceType.TARGET,
                robot_link=UNKNOWN_LINK,
                active_manipulators=context.active_manipulators,
                target_grounded=target_grounded,
            ),
            "full_margin": self.safety_margin,
        }


__all__ = ["UNKNOWN_LINK", "ClearancePolicy", "resolve_link_names"]
