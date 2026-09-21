"""Stage 7 — collision candidates as a CasADi constraint specification.

This module is the contract with a trajectory optimizer that does not exist yet, so its shape is
determined entirely by what a real-time TO needs rather than by what is convenient to emit.

**The structure is built once and never rebuilt.** `ConstraintBuilder.__init__` creates the symbolic
expression — a fixed `max_candidates` slots, a fixed horizon, a fixed robot sphere count — and every
subsequent frame only writes new numbers into a parameter vector. If the NLP were rebuilt per frame,
IPOPT would redo symbolic setup and Jacobian sparsity detection each time, and warm-starting would be
meaningless because the variable and constraint indices would not refer to the same things. Fixed
slots are why `max_candidates` exists at all.

**Unused slots are switched off by a parameter, not by removal.** Each slot carries an `active` flag
in {0, 1}; an inactive slot's row evaluates to a positive constant and its gradient to zero, so the
row count and the sparsity pattern are invariant while the scene changes.

**Clearance is per `(robot sphere, slot)`, not per slot.** `d_safe` used to be one number per
candidate, which cannot express "the right fingertips may touch the cup but the torso may not", and
the old code expressed contact by *skipping* the candidate — deleting its rows and letting the whole
robot pass through it. Now every candidate occupies its slot in every phase and the margin varies
across the column; see `clearance.ClearancePolicy`. The parameter block grows from `m` to
`n_robot_spheres * m` values, and nothing else about the graph moves: same rows, same sparsity in
`q`, same solver object.

**Distances are squared by default.** `||.||` is not differentiable at the origin, and an
interior-point method will walk a robot sphere straight through a candidate centre — exactly where
the gradient is undefined. So the emitted row is

    h(q) = ||p_robot(q) - p_cand||^2 - (r_robot + r_cand + d_safe)^2 >= 0

`constraint.squared_distance: false` selects the non-squared form for comparison, guarded by a small
epsilon inside the square root so the ablation is runnable rather than merely configurable.

**hpp-fcl is not wrapped as a black box.** Every expression here is CasADi arithmetic over `q`, so
the optimizer differentiates it symbolically. A callback into an external distance query would give
IPOPT a function it can only finite-difference.

AG3S does not own the robot side: the sphere chain arrives through the injected
`RobotCollisionModel` Protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

import numpy as np

from benchmark.ag3s.config import ConstraintConfig, GeometryConfig
from benchmark.ag3s.stages.geometry import to_spheres
from benchmark.ag3s.types import (
    CollisionCandidate,
    ConstraintSpec,
    ContactPolicyContext,
    RobotCollisionModel,
    SourceType,
    SupportSurface,
)

if TYPE_CHECKING:  # pragma: no cover
    from benchmark.ag3s.constraints.clearance import ClearancePolicy

_SQRT_EPS = 1e-9


class ConstraintBuilder:
    """Builds the symbolic structure once; `build()` then only refreshes parameter values.

    Args:
        robot_model: injected `RobotCollisionModel`. Its `sphere_centers_symbolic(q)` is the only
            thing AG3S knows about the robot.
        constraint_config / geometry_config: horizon, squared form, slot count, margins.
        max_support_surfaces: half-space slots. Planes get their own block because their constraint
            is linear and exact, and collapsing a table into spheres would be both wrong and
            enormous.

    Row count is `horizon * n_robot_spheres * max_candidates` for the sphere block plus
    `horizon * n_robot_spheres * max_support_surfaces` for the plane block. That grows quickly with
    the robot's sphere count — RB-Y1's full 61-sphere chain at H=20 and 32 slots is ~39k rows — so a
    real deployment should inject a link-filtered model covering the arm that actually moves. The
    number is reported in `ConstraintSpec.n_constraints` rather than hidden.
    """

    def __init__(
        self,
        robot_model: RobotCollisionModel,
        constraint_config: ConstraintConfig | None = None,
        geometry_config: GeometryConfig | None = None,
        *,
        max_support_surfaces: int = 2,
        clearance_policy: Optional["ClearancePolicy"] = None,
        attached_parent_links: Sequence[str] = (),
    ):
        import casadi as ca

        from benchmark.ag3s.constraints.clearance import ClearancePolicy, resolve_link_names

        self.ca = ca
        self.robot_model = robot_model
        self.constraint_config = constraint_config or ConstraintConfig()
        self.geometry_config = geometry_config or GeometryConfig()
        self.horizon = int(self.constraint_config.horizon)
        self.max_candidates = int(self.geometry_config.max_candidates)
        self.max_support_surfaces = int(max_support_surfaces)
        self.nq = int(robot_model.nq)

        probe = robot_model.sphere_centers_symbolic(ca.SX.sym("q_probe", self.nq))
        self.n_robot_spheres = len(probe)
        self.robot_radii = np.asarray([float(r) for _, r in probe], np.float64)
        self.sphere_link_names = resolve_link_names(robot_model, self.n_robot_spheres)
        self.clearance_policy = clearance_policy or ClearancePolicy.from_config(
            geometry=self.geometry_config
        )

        # Attached-object slots are reserved per *parent link*, because the parent is part of the
        # symbolic expression (`link_pose_symbolic(q, link)`) rather than a number, and switching it
        # would rebuild the graph. Declaring the links a session might grasp with up front — usually
        # the two grippers — keeps attach and detach to a parameter change. The default is empty, so
        # a caller that never attaches anything pays nothing: same rows, same sparsity as before.
        self.attached_parent_links = tuple(str(link) for link in attached_parent_links)
        self.max_attached = (
            int(self.constraint_config.max_attached_primitives) if self.attached_parent_links else 0
        )
        if self.max_attached and not hasattr(robot_model, "link_pose_symbolic"):
            raise ValueError(
                "attached_parent_links was given but the robot model has no link_pose_symbolic; "
                "a held object's pose has to be differentiable in q"
            )

        m, k, s = self.max_candidates, self.max_support_surfaces, self.n_robot_spheres
        a, np_ = self.max_attached, len(self.attached_parent_links)
        # `d_safe` is (sphere, slot) in row-major order: index `sphere * m + slot`. It is by far the
        # largest block now (61 x 32 on RB-Y1), which is the price of being able to say which part
        # of the robot a clearance applies to.
        base = 4 * m
        self.layout = {
            "pos": (0, 3 * m),
            "radius": (3 * m, 4 * m),
            "d_safe": (base, base + s * m),
            "active": (base + s * m, base + s * m + m),
            "plane_normal": (base + s * m + m, base + s * m + m + 3 * k),
            "plane_offset": (base + s * m + m + 3 * k, base + s * m + m + 4 * k),
            "plane_active": (base + s * m + m + 4 * k, base + s * m + m + 5 * k),
        }
        cursor = base + s * m + m + 5 * k
        # One block per (parent link, attached slot): centre in the parent frame, radius, clearance,
        # an on/off flag, and a per-robot-sphere mask that zeroes the pairs the caller allowlisted as
        # intended contact (the fingers actually holding the object).
        for name, width in (
            ("attached_pos", 3 * a * np_),
            ("attached_radius", a * np_),
            ("attached_dsafe", a * np_),
            ("attached_active", a * np_),
            ("attached_self_mask", a * np_ * s),
        ):
            self.layout[name] = (cursor, cursor + width)
            cursor += width
        self.layout["n_params"] = cursor

        self.parameter_symbol = ca.SX.sym("ag3s_p", self.layout["n_params"])
        self.n_constraints = self.horizon * self.n_robot_spheres * (m + k)
        # attached vs candidates, vs planes, and vs the robot's own spheres. The last is what covers
        # §23's "opposite arm", "torso" and "non-contact same-arm links" with one mechanism.
        self.n_attached_rows = self.horizon * a * np_ * (m + k + s)
        self.n_constraints += self.n_attached_rows

    # --- symbolic ------------------------------------------------------------------------
    def _slice(self, name: str) -> Any:
        lo, hi = self.layout[name]
        return self.parameter_symbol[lo:hi]

    def expr_factory(self, Q: Any, p: Any = None) -> Any:
        """`Q` is `(nq, horizon)`; returns the stacked constraint vector `h(Q) >= 0`.

        Rows are ordered `(step, robot_sphere, slot)` with slot varying fastest, and the plane block
        follows the sphere block. `ConstraintSpec.layout` records the same ordering so a caller can
        map a violated row back to the candidate that produced it.
        """
        ca = self.ca
        p = self.parameter_symbol if p is None else p
        m, k = self.max_candidates, self.max_support_surfaces
        lo_pos, _ = self.layout["pos"]
        lo_r, _ = self.layout["radius"]
        lo_d, _ = self.layout["d_safe"]
        lo_a, _ = self.layout["active"]
        lo_pn, _ = self.layout["plane_normal"]
        lo_po, _ = self.layout["plane_offset"]
        lo_pa, _ = self.layout["plane_active"]

        rows: list[Any] = []
        plane_rows: list[Any] = []
        for step in range(self.horizon):
            q_k = Q[:, step]
            for sphere_index, (centre, robot_radius) in enumerate(
                self.robot_model.sphere_centers_symbolic(q_k)
            ):
                for slot in range(m):
                    candidate_centre = p[lo_pos + 3 * slot : lo_pos + 3 * slot + 3]
                    # The clearance for *this* sphere against *this* slot, which is what lets the
                    # right fingertips approach the target while the torso does not.
                    d_safe = p[lo_d + sphere_index * m + slot]
                    total_radius = robot_radius + p[lo_r + slot] + d_safe
                    delta = centre - candidate_centre
                    if self.constraint_config.squared_distance:
                        h = ca.dot(delta, delta) - total_radius * total_radius
                    else:
                        h = ca.sqrt(ca.dot(delta, delta) + _SQRT_EPS) - total_radius
                    active = p[lo_a + slot]
                    # Inactive slots evaluate to +1 and contribute no gradient, so the row count and
                    # the Jacobian sparsity are the same whatever the scene contains.
                    rows.append(active * h + (1.0 - active) * 1.0)

                for plane in range(k):
                    normal = p[lo_pn + 3 * plane : lo_pn + 3 * plane + 3]
                    offset = p[lo_po + plane]
                    active = p[lo_pa + plane]
                    h = ca.dot(normal, centre) - offset - robot_radius
                    plane_rows.append(active * h + (1.0 - active) * 1.0)
        return ca.vertcat(*(rows + plane_rows + self._attached_rows(Q, p)))

    def _attached_rows(self, Q: Any, p: Any) -> list[Any]:
        """Rows for a held object: against the scene, the support planes, and the robot itself.

        The object's pose is `link_pose_symbolic(q, parent) @ centre_in_parent`, so the optimizer
        differentiates the held geometry's motion with respect to the joints carrying it. Passing a
        numeric pose instead would make the object a constant obstacle that happens to sit near the
        hand — correct at one configuration and wrong at every other point on the trajectory.

        The robot block is one mechanism covering everything §23 asks for. A mask parameter zeroes
        exactly the pairs the caller allowlisted as intended contact; the forearm, the torso, the
        opposite arm and every unrecognised link keep their rows.
        """
        if not self.max_attached:
            return []
        ca = self.ca
        m, k, s, a = self.max_candidates, self.max_support_surfaces, self.n_robot_spheres, self.max_attached
        lo_pos, _ = self.layout["pos"]
        lo_r, _ = self.layout["radius"]
        lo_pn, _ = self.layout["plane_normal"]
        lo_po, _ = self.layout["plane_offset"]
        lo_pa, _ = self.layout["plane_active"]
        lo_ap, _ = self.layout["attached_pos"]
        lo_ar, _ = self.layout["attached_radius"]
        lo_ad, _ = self.layout["attached_dsafe"]
        lo_aa, _ = self.layout["attached_active"]
        lo_am, _ = self.layout["attached_self_mask"]

        out: list[Any] = []
        for step in range(self.horizon):
            q_k = Q[:, step]
            robot = list(self.robot_model.sphere_centers_symbolic(q_k))
            for parent_index, parent in enumerate(self.attached_parent_links):
                T = self.robot_model.link_pose_symbolic(q_k, parent)
                R, origin = T[:3, :3], T[:3, 3]
                for slot in range(a):
                    flat = parent_index * a + slot
                    local = p[lo_ap + 3 * flat : lo_ap + 3 * flat + 3]
                    centre = ca.mtimes(R, local) + origin
                    radius = p[lo_ar + flat]
                    margin = p[lo_ad + flat]
                    active = p[lo_aa + flat]

                    for candidate_slot in range(m):
                        other = p[lo_pos + 3 * candidate_slot : lo_pos + 3 * candidate_slot + 3]
                        total = radius + p[lo_r + candidate_slot] + margin
                        delta = centre - other
                        both = active * p[self.layout["active"][0] + candidate_slot]
                        h = (
                            ca.dot(delta, delta) - total * total
                            if self.constraint_config.squared_distance
                            else ca.sqrt(ca.dot(delta, delta) + _SQRT_EPS) - total
                        )
                        out.append(both * h + (1.0 - both) * 1.0)

                    for plane in range(k):
                        normal = p[lo_pn + 3 * plane : lo_pn + 3 * plane + 3]
                        both = active * p[lo_pa + plane]
                        h = ca.dot(normal, centre) - p[lo_po + plane] - radius - margin
                        out.append(both * h + (1.0 - both) * 1.0)

                    for sphere_index, (robot_centre, robot_radius) in enumerate(robot):
                        mask = p[lo_am + flat * s + sphere_index]
                        both = active * mask
                        delta = centre - robot_centre
                        total = radius + robot_radius + margin
                        h = (
                            ca.dot(delta, delta) - total * total
                            if self.constraint_config.squared_distance
                            else ca.sqrt(ca.dot(delta, delta) + _SQRT_EPS) - total
                        )
                        out.append(both * h + (1.0 - both) * 1.0)
        return out

    # --- numeric -------------------------------------------------------------------------
    def parameter_values(
        self,
        candidates: Iterable[CollisionCandidate] = (),
        support_surfaces: Iterable[SupportSurface] = (),
        *,
        context: Optional[ContactPolicyContext] = None,
        target_grounded: bool = True,
        attached: Optional[Any] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pack this frame's scene into the parameter vector.

        Returns `(values, active_mask, candidate_ids)`. A slot holds one **constraint sphere**, not
        one candidate: a capsule occupies several, which is how an elongated obstacle gets a tighter
        description than its bounding sphere. `candidate_ids[j]` records the owning candidate so TO
        can carry a warm start slot-by-slot across frames.

        **Every candidate occupies slots, in every phase.** The old code skipped candidates with
        `collision_enabled=False`, which is how the target vanished from the graph at `GRASP`. Only
        support surfaces are excluded here, and only because they are exact as half-spaces in their
        own block rather than because they are unimportant.

        `context` (phase + authorized manipulators) drives the `(sphere, slot)` margin block. With no
        context the policy sees an empty manipulator set and returns full clearance everywhere,
        which is the right default for a caller that has not said who is allowed to touch what.

        Slots are filled largest-radius-first, so when a scene has more spheres than slots the ones
        at risk are the least able to cause a collision. Overflow is handled conservatively rather
        than by deletion — see `_aggregate_overflow` — and reported via
        `ConstraintSpec.layout['n_overflow_spheres']`.
        """
        values = np.zeros(self.layout["n_params"], np.float64)
        m, k, s = self.max_candidates, self.max_support_surfaces, self.n_robot_spheres
        lo_pos, _ = self.layout["pos"]
        lo_r, _ = self.layout["radius"]
        lo_d, _ = self.layout["d_safe"]
        lo_a, _ = self.layout["active"]
        ctx = context or ContactPolicyContext()

        entries: list[tuple[float, np.ndarray, float, int, SourceType]] = []
        for candidate in candidates:
            if candidate.source_type is SourceType.SUPPORT_SURFACE:
                continue  # exact as a half-space; lives in the plane block below
            for primitive in candidate.geometry:
                for centre, radius in to_spheres(primitive):
                    entries.append(
                        (float(radius), np.asarray(centre, np.float64),
                         float(radius), int(candidate.id), candidate.source_type)
                    )
        entries.sort(key=lambda e: (-e[0], e[3]))
        entries, overflow_count = self._aggregate_overflow(entries, m)

        candidate_ids = np.full(m, -1, np.int64)
        slot_sources: list[SourceType] = [SourceType.UNKNOWN_GEOMETRY] * m
        for slot, (_, centre, radius, owner, source) in enumerate(entries[:m]):
            values[lo_pos + 3 * slot : lo_pos + 3 * slot + 3] = centre
            values[lo_r + slot] = radius
            values[lo_a + slot] = 1.0
            candidate_ids[slot] = owner
            slot_sources[slot] = source

        # The margin block. Inactive slots get the full margin too — harmless, since `active=0`
        # makes their row a constant, and it means a slot that becomes active mid-frame can never
        # inherit a stale relaxation.
        margins = self.clearance_policy.margin_matrix(
            self.sphere_link_names, slot_sources, context=ctx, target_grounded=target_grounded
        )
        values[lo_d : lo_d + s * m] = margins.reshape(-1)

        lo_pn, _ = self.layout["plane_normal"]
        lo_po, _ = self.layout["plane_offset"]
        lo_pa, _ = self.layout["plane_active"]
        for plane, surface in enumerate(list(support_surfaces)[:k]):
            normal, offset = surface.to_halfspace()
            values[lo_pn + 3 * plane : lo_pn + 3 * plane + 3] = normal
            values[lo_po + plane] = offset
            values[lo_pa + plane] = 1.0

        self._pack_attached(values, attached)

        active = values[lo_a : lo_a + m].copy()
        self._last_overflow = overflow_count
        return values, active, candidate_ids

    def _pack_attached(self, values: np.ndarray, attached: Optional[Any]) -> None:
        """Write the held object into its reserved slots, or leave every flag at zero.

        Detaching is the `else` branch: the flags go to zero and every attached row becomes the
        constant 1. No symbol is touched, so the graph, the sparsity and the solver object are
        identical before and after — which is the entire point of reserving the slots.
        """
        if not self.max_attached:
            if attached is not None:
                raise ValueError(
                    f"cannot attach to {attached.parent_link!r}: this builder reserved no attached "
                    "slots. Pass attached_parent_links=(...) when constructing it."
                )
            return

        a, s = self.max_attached, self.n_robot_spheres
        lo_ap, _ = self.layout["attached_pos"]
        lo_ar, _ = self.layout["attached_radius"]
        lo_ad, _ = self.layout["attached_dsafe"]
        lo_aa, _ = self.layout["attached_active"]
        lo_am, _ = self.layout["attached_self_mask"]
        # Cleared every frame: a stale active flag would keep constraining an object that was put
        # down three frames ago.
        values[lo_aa : lo_aa + a * len(self.attached_parent_links)] = 0.0
        if attached is None:
            return

        from benchmark.ag3s.constraints.attached import attached_spheres, self_collision_mask

        if attached.parent_link not in self.attached_parent_links:
            raise ValueError(
                f"cannot attach to {attached.parent_link!r}: this builder reserved slots for "
                f"{list(self.attached_parent_links)}. The parent link is part of the symbolic "
                "expression, so it has to be declared before the graph is built."
            )
        parent_index = self.attached_parent_links.index(attached.parent_link)
        spheres = attached_spheres(attached)
        mask = self_collision_mask(attached, self.sphere_link_names)
        self._last_attached_dropped = max(0, len(spheres) - a)

        for slot, (centre, radius) in enumerate(spheres[:a]):
            flat = parent_index * a + slot
            values[lo_ap + 3 * flat : lo_ap + 3 * flat + 3] = centre
            values[lo_ar + flat] = radius
            values[lo_ad + flat] = float(self.geometry_config.safety_margin)
            values[lo_aa + flat] = 1.0
            values[lo_am + flat * s : lo_am + flat * s + s] = mask

    # --- conservative slot overflow -----------------------------------------------------
    def _aggregate_overflow(
        self, entries: list, max_slots: int
    ) -> tuple[list, int]:
        """Fold spheres beyond `max_slots` into reserved aggregate slots that **contain** them.

        The old behaviour was to keep the largest `max_slots` spheres and drop the rest. Dropping is
        under-approximation: a small sphere is small, not absent, and a robot routed through where it
        used to be hits it just as hard. So the tail is grouped and each group replaced by one sphere
        that provably contains every member — centre at the group's mean, radius
        ``max_i(||c_i - c|| + r_i)``.

        The grouping is spatial (a coordinate sort into `reserved` contiguous runs, taken along the
        tail's longest axis) rather than "everything into one ball". With distant clutter, a single
        aggregate would span the workspace and forbid every trajectory; several local ones stay
        conservative without being useless. Excess volume is the price and is reported, not hidden.
        """
        n = len(entries)
        if n <= max_slots:
            return entries, 0

        reserved = max(1, min(int(self.constraint_config.reserved_overflow_slots), max_slots))
        keep = entries[: max_slots - reserved]
        tail = entries[max_slots - reserved :]
        if not tail:
            return entries[:max_slots], 0

        centres = np.stack([e[1] for e in tail])
        radii = np.asarray([e[2] for e in tail], np.float64)
        spread = centres.max(axis=0) - centres.min(axis=0)
        axis = int(np.argmax(spread))
        order = np.argsort(centres[:, axis], kind="stable")
        groups = np.array_split(order, min(reserved, len(order)))

        aggregates: list = []
        for group in groups:
            if group.size == 0:
                continue
            member_c, member_r = centres[group], radii[group]
            centre = member_c.mean(axis=0)
            radius = float(np.max(np.linalg.norm(member_c - centre, axis=1) + member_r))
            aggregates.append((radius, centre, radius, -1, SourceType.OVERFLOW))
        return keep + aggregates, len(tail)

    def build(
        self,
        candidates: Iterable[CollisionCandidate] = (),
        support_surfaces: Iterable[SupportSurface] = (),
        *,
        context: Optional[ContactPolicyContext] = None,
        target_grounded: bool = True,
        attached: Optional[Any] = None,
    ) -> ConstraintSpec:
        """A `ConstraintSpec` for this frame. The symbolic parts are shared, not rebuilt."""
        values, active, candidate_ids = self.parameter_values(
            candidates, support_surfaces, context=context, target_grounded=target_grounded,
            attached=attached,
        )
        return ConstraintSpec(
            expr_factory=self.expr_factory,
            parameter_symbol=self.parameter_symbol,
            parameter_values=values,
            active_mask=active,
            lower_bound=np.zeros(self.n_constraints),
            upper_bound=np.full(self.n_constraints, np.inf),
            candidate_ids=candidate_ids,
            horizon=self.horizon,
            max_candidates=self.max_candidates,
            n_robot_spheres=self.n_robot_spheres,
            squared_distance=self.constraint_config.squared_distance,
            layout={
                **self.layout,
                "nq": self.nq,
                "max_support_surfaces": self.max_support_surfaces,
                "row_order": "(step, robot_sphere, slot); plane block last",
                "n_sphere_rows": self.horizon * self.n_robot_spheres * self.max_candidates,
                "n_plane_rows": self.horizon * self.n_robot_spheres * self.max_support_surfaces,
                # Spheres folded into conservative aggregates rather than dropped. Zero means every
                # sphere got its own slot.
                "n_overflow_spheres": getattr(self, "_last_overflow", 0),
                "n_dropped_spheres": 0,  # nothing is dropped any more; kept so old readers see zero
                "sphere_link_names": list(self.sphere_link_names),
                "d_safe_order": "(robot_sphere, slot) row-major",
                "attached_parent_links": list(self.attached_parent_links),
                "max_attached_primitives": self.max_attached,
                "n_attached_rows": self.n_attached_rows,
                "n_attached_dropped": getattr(self, "_last_attached_dropped", 0),
            },
        )


def evaluate(spec: ConstraintSpec, q_trajectory: np.ndarray, builder: ConstraintBuilder) -> np.ndarray:
    """Numeric value of every constraint row at `q_trajectory`, shape `(nq, horizon)`.

    Used by tests and diagnostics to check a solution without going through a solver.
    """
    ca = builder.ca
    Q = ca.SX.sym("Q", builder.nq, builder.horizon)
    fn = ca.Function("h", [Q, spec.parameter_symbol], [spec.expr_factory(Q, spec.parameter_symbol)])
    return np.asarray(fn(np.asarray(q_trajectory, np.float64), spec.parameter_values)).reshape(-1)


__all__ = ["ConstraintBuilder", "evaluate"]
