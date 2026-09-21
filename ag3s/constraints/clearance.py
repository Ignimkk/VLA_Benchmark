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
from typing import Any, Optional, Sequence

import numpy as np

from benchmark.ag3s.config import ContactConfig, GeometryConfig, SupportSurfaceConfig
from benchmark.ag3s.stages.geometry import to_spheres
from benchmark.ag3s.types import (
    AttachedCollisionGeometry,
    ContactPolicyContext,
    Manipulator,
    Phase,
    RobotCollisionModel,
    SourceType,
    TargetGeometry,
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


# ------------------------------------------------------------------ 조작 대상 대 주목 대상


@dataclasses.dataclass(frozen=True)
class ManipulatedObject:
    """무엇을 **조작**하고 있는가 — 접촉 권한이 붙어야 할 대상.

    이 모듈의 첫 문단이 말한 것과 같은 종류의 구분이 하나 더 있고, 구현이 아직 따라오지 못했다
    (F11, `docs/AG3S_REVIEW_LOG.md`). 두 대상은 **같지 않다**:

        주목 대상 (`TargetGeometry`)          attention 이 가리키는 것
        조작 대상 (`AttachedCollisionGeometry`) 로봇이 실제로 쥐고 있는 것

    실측에서 정책의 attention 은 **파지에 착수하는 순간 목적지로 옮겨간다.** run_0004 와 run_0005
    양쪽에서, 손이 사과를 쥐고 있는 내내 grounding 이 낸 target 은 바구니였다. 접촉 권한을 주목
    대상에 걸면 손끝은 *바구니* 를 만질 허가를 받고, 정작 쥐고 있는 *사과* 는 완전 여유거리를
    요구하는 장애물로 남는다 — 두 기록에서 최악 -138.7 mm / -97.5 mm 로 측정됐다.

    그래서 권한은 **조작 대상**에 건다. 규칙은 한 줄이다:

        쥔 것이 있으면 그것. 없으면 주목 대상.

    파지 전에는 접근 중인 물체가 곧 보고 있는 물체이므로 둘이 일치하고, 이때 동작은 예전과
    **완전히 같다** — 바뀌는 것은 `attach()` 가 불린 뒤부터다.

    표현이 둘인 것은 일부러다. 주목 대상은 관측된 점구름이라 거리를 점으로 재는 것이 정확하고,
    쥔 물체는 primitive + FK 자세라 **해석적 거리**가 정확하다. 하나로 합치면 어느 한쪽이
    나빠진다 — 주목 대상을 경계구로 줄이면 헐거워지고, 쥔 물체를 점으로 샘플링하면 샘플 밀도에
    정확도가 매달린다.
    """

    #: `"attached"` 면 쥔 물체, `"target"` 이면 주목 대상. 로그와 시험이 어느 경로였는지 본다.
    source: str
    #: 주목 대상 경로 — 관측된 점구름 `(M, 3)`.
    points: Optional[np.ndarray] = None
    #: 쥔 물체 경로 — base 프레임 구 `(S, 3)` 와 `(S,)`. FK 로 놓은 스냅샷이다.
    sphere_centers: Optional[np.ndarray] = None
    sphere_radii: Optional[np.ndarray] = None

    @property
    def is_held(self) -> bool:
        return self.source == "attached"


def manipulated_object(
    target: Optional["TargetGeometry"] = None,
    attached: Optional["AttachedCollisionGeometry"] = None,
    *,
    robot_model: Any = None,
    robot_state: Optional[np.ndarray] = None,
) -> Optional[ManipulatedObject]:
    """쥔 것이 있으면 그것, 없으면 주목 대상. 둘 다 없으면 `None`.

    쥔 물체는 `parent_link` 를 타고 다니므로 base 프레임 좌표를 얻으려면 그 순간의 FK 가 필요하다.
    `robot_model` 이나 `robot_state` 가 없어서 놓을 수 없으면 **주목 대상으로 물러서지 않고
    `None` 을 돌려준다** — "쥐고 있는데 어디 있는지 모른다" 를 "바구니를 쥐고 있다" 로 바꾸는 것이
    이 함수가 막으려는 바로 그 오류이기 때문이다. 권한이 없는 쪽이 닫히는 방향이다.
    """
    if attached is not None:
        pose_fn = getattr(robot_model, "link_pose", None) or getattr(
            robot_model, "link_pose_numeric", None
        )
        if pose_fn is None or robot_state is None:
            return None
        T = np.asarray(pose_fn(np.asarray(robot_state, np.float64).reshape(-1),
                               attached.parent_link), np.float64)
        T_base_object = T @ np.asarray(attached.T_parent_object, np.float64)
        R, origin = T_base_object[:3, :3], T_base_object[:3, 3]
        centres: list[np.ndarray] = []
        radii: list[float] = []
        for primitive in attached.primitives:
            for local, radius in to_spheres(primitive):
                centres.append(R @ np.asarray(local, np.float64) + origin)
                radii.append(float(radius))
        if not centres:
            return None
        return ManipulatedObject(
            source="attached",
            sphere_centers=np.asarray(centres, np.float64).reshape(-1, 3),
            sphere_radii=np.asarray(radii, np.float64).reshape(-1),
        )
    if target is not None:
        return ManipulatedObject(
            source="target", points=np.asarray(target.points, np.float64).reshape(-1, 3)
        )
    return None


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
        if parsed is SourceType.DESTINATION:
            # Thin, but never relaxed further and never zero: the held object goes *into* the
            # destination, it does not touch it. `contact_margin` (zero) stays reserved for the
            # object actually being held.
            return float(self.contact.destination_margin)
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
