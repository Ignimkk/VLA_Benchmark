"""Turning AG3S's collision scene into linear rows the QP can hold — and choosing which ones.

## Why this does not call AG3S's CasADi fragment

`ag3s.to_adapter.to_casadi()` hands over the whole constraint vector as one symbolic expression, and
that is the right shape for a general NLP solver that wants to differentiate the problem itself. It
is the wrong shape here, and the difference is not marginal. Measured on RB-Y1 (61 spheres, 32
candidate slots, H=50, 103,700 rows):

| how the rows are obtained            | build (once) | evaluate h and its gradient |
|--------------------------------------|--------------|------------------------------|
| one monolithic CasADi graph          | ~19 s        | ~19 ms                       |
| per-step graph, mapped over 50 steps | 0.31 s       | 12.5 ms                      |
| **sphere FK + position Jacobian**    | **0.066 s**  | **1.07 ms**                  |

The last row is not an approximation of the first two. For a sphere at ``p(q)`` and a candidate at
``c``, the constraint and its gradient are available in closed form:

    h      = ||p - c|| - (r_robot + r_cand + d_safe)
    grad h = (p - c)^T / ||p - c|| * dp/dq

so all that is needed from the robot is ``p`` and ``dp/dq`` — and those are **shared by every
candidate slot**, which is exactly the redundancy the monolithic graph pays for 32 times over. The
rows themselves are then assembled in numpy, where skipping the ones that do not matter is free.

AG3S is still the single source of truth for *what* to avoid and *by how much*: the candidate
positions, radii and the per-``(sphere, slot)`` clearance matrix are read straight out of the
parameter vector AG3S packed, via the layout it publishes. Nothing is recomputed, no clearance policy
is duplicated, and there is no call into AG3S while the solver runs.

## Distance, not squared distance

AG3S emits the squared form because an interior-point method walks through the origin of ``||.||``
where the gradient is undefined. An SQP never evaluates a gradient there — it linearizes at the
current iterate and the trust region keeps it away — so this module works in metres instead. The
feasible sets are identical (``h_sq >= 0`` iff ``h_dist >= 0``), and metres are what makes the
activation band, the trust radius and the reported violation comparable quantities that a person can
judge. The degenerate case ``p == c`` (a sphere centre exactly inside a candidate centre) is handled
explicitly rather than left to produce a NaN.

## Which rows the QP sees

Two reductions, and the guarantee that makes them safe is the same for both: **every** row is
re-evaluated at the solution, at full resolution, by `full_violation`. Pruning can cost convergence;
it cannot hide a collision.

* **activation band** — a row whose current clearance exceeds the band cannot become violated within
  one trust-region step, so it has no influence on this QP.
* **per-step budget** — each step keeps its `rows_per_step` tightest rows. Budgeting per step rather
  than globally keeps the QP's sparsity fixed (row *i* always belongs to step ``i // rows_per_step``)
  and stops one crowded step from starving the rest. When a step's budget binds, that is reported.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.types import ChunkLayout

_EPS = 1e-12

#: `LinearizedRows.slot` value for an ESDF row. Distinct from every candidate slot (>= 0) and from
#: every plane (-1 - k, k < max_support_surfaces), so a consumer can tell the three apart.
ESDF_SLOT = -99999


@dataclasses.dataclass(frozen=True)
class SceneSnapshot:
    """One frame of AG3S's scene, decoded from the parameter vector it already packs.

    Reading `ConstraintSpec.parameter_values` through `ConstraintSpec.layout` rather than
    re-deriving anything is deliberate: the clearance policy, the overflow aggregation and the
    attached-object block all end up in that vector, so a change on the AG3S side reaches the
    optimizer without a second implementation to keep in sync.
    """

    candidate_pos: np.ndarray  # (M, 3) base frame
    candidate_radius: np.ndarray  # (M,)
    candidate_active: np.ndarray  # (M,) bool
    d_safe: np.ndarray  # (S, M) required clearance per (robot sphere, slot)
    plane_normal: np.ndarray  # (K, 3) unit, free side is n.p >= offset
    plane_offset: np.ndarray  # (K,)
    plane_active: np.ndarray  # (K,) bool
    robot_radii: np.ndarray  # (S,)
    candidate_ids: np.ndarray  # (M,) owning candidate, -1 for empty
    #: Optional ESDF backend. When present it adds one row per (step, robot sphere) —
    #: `d_esdf(p) - r_robot - esdf_margin >= 0` — alongside whatever candidate and plane rows the
    #: primitive backend produced. Both can be active at once, which is what an ablation needs: the
    #: two descriptions of the same scene then sit in the same QP and can be compared row for row.
    esdf: Any = None
    esdf_margin: float = 0.0
    #: The **manipulated** object, in whichever form measures its distance best, plus the per-sphere
    #: margin that goes with it. `manipulated_link_margin` is `(S,)`, AG3S's
    #: `ClearancePolicy.margin_matrix` TARGET column for this constraint model
    #: (`CollisionConstraintSet.manipulated_link_margin`). Together they let `_esdf_clearance` tell
    #: "this sphere's nearest field obstacle is the object its own link is authorized to touch" from
    #: "this sphere is near something else", without AG3S carving that object out of the shared field
    #: for every sphere (`docs/AG3S_REVIEW_LOG.md` Step 2/3, E1).
    #:
    #: Two forms, because the two objects are measured differently (F11, Step 6):
    #:
    #: * `manipulated_points` `(T, 3)` — the attention target's observed cloud, when nothing is held.
    #: * `manipulated_spheres` `(S2, 3)` + `manipulated_sphere_radii` `(S2,)` — a **held** object,
    #:   placed by FK. Analytic, so accuracy does not depend on how densely a surface was sampled.
    #:
    #: At most one is set. All are `None` when there is nothing being manipulated.
    manipulated_points: Optional[np.ndarray] = None
    manipulated_spheres: Optional[np.ndarray] = None
    manipulated_sphere_radii: Optional[np.ndarray] = None
    manipulated_link_margin: Optional[np.ndarray] = None
    #: The **held** object as query points riding `attached_parent_link`, in that link's frame
    #: (`AttachedCollisionGeometry.points`). Once a grasp closes the object stops being part of the
    #: world and becomes part of the robot, so it belongs on the query side of the field rather than
    #: inside it -- these points are appended to the robot's spheres with radius zero.
    #:
    #: Points rather than a fitted primitive, because a primitive is measurably worse (F19): one
    #: sphere through an apple's cloud is r = 57.4 mm since it must cover the stem and leaf, which
    #: costs 13.2 mm of clearance on average against the basket and in one frame reports a collision
    #: that is not there.
    #:
    #: They are `None` together whenever nothing is held.
    attached_points: Optional[np.ndarray] = None
    attached_parent_link: Optional[str] = None
    #: Where the held object is being **put**. `destination_label` is the name it carries in the
    #: field's label layer; `destination_margin` is the clearance required against it.
    #:
    #: Thin (20 mm) rather than the 50 mm every other obstacle gets, because the geometry does not
    #: leave room for 50: the held apple's points pass within 32.9 mm of the basket's inner wall
    #: while it is lowered in (F18). This is not a global relaxation -- it applies only where the
    #: label layer says the nearest surface *is* the destination, so every other obstacle, the
    #: table included, keeps its full margin.
    #:
    #: `None` whenever no destination was named, and naming one is the caller's job: AG3S is never
    #: told what the task is, so it cannot know which object is the destination.
    destination_label: Optional[str] = None
    destination_margin: float = 0.0

    @property
    def has_esdf(self) -> bool:
        return self.esdf is not None

    @property
    def n_slots(self) -> int:
        return int(self.candidate_pos.shape[0])

    @property
    def n_spheres(self) -> int:
        return int(self.robot_radii.shape[0])

    @property
    def n_planes(self) -> int:
        return int(self.plane_normal.shape[0])

    @property
    def n_esdf_rows(self) -> int:
        """One row per robot sphere when the field is present. The ESDF answers per *point*, not
        per slot, so there is nothing to enumerate — that is most of why it is cheaper."""
        return self.n_spheres if self.esdf is not None else 0

    @property
    def is_empty(self) -> bool:
        return not (self.candidate_active.any() or self.plane_active.any()
                    or self.esdf is not None)

    @classmethod
    def from_spec(cls, spec, robot_radii: np.ndarray) -> "SceneSnapshot":
        """Decode an AG3S `ConstraintSpec`. `robot_radii` comes from the same model AG3S used."""
        values = np.asarray(spec.parameter_values, np.float64)
        layout = spec.layout
        m = int(spec.max_candidates)
        s = int(spec.n_robot_spheres)
        k = int(layout["max_support_surfaces"])
        radii = np.asarray(robot_radii, np.float64).reshape(-1)
        if radii.shape[0] != s:
            raise ValueError(
                f"the spec was built against {s} robot spheres but {radii.shape[0]} radii were given;"
                " the optimizer and AG3S must share one constraint model"
            )

        def block(name: str) -> np.ndarray:
            lo, hi = layout[name]
            return values[lo:hi]

        return cls(
            candidate_pos=block("pos").reshape(m, 3),
            candidate_radius=block("radius"),
            candidate_active=block("active") > 0.5,
            d_safe=block("d_safe").reshape(s, m),
            plane_normal=block("plane_normal").reshape(k, 3),
            plane_offset=block("plane_offset"),
            plane_active=block("plane_active") > 0.5,
            robot_radii=radii,
            candidate_ids=np.asarray(spec.candidate_ids, np.int64),
        )


@dataclasses.dataclass(frozen=True)
class LinearizedRows:
    """The linearized rows, in a shape whose **sparsity never changes**.

    `gradient` is ``(H, rows_per_step, nq_opt)`` and `value` is ``(H, rows_per_step)``, dense and
    fully allocated whether or not the scene fills them. A step with three near-active constraints
    and a step with sixty produce the same arrays; the unused rows carry a zero gradient and a
    clearance of ``+inf``, which makes them trivially satisfied.

    That padding is the point. A QP solver factorizes its KKT matrix against a sparsity pattern and
    reuses the factorization only while that pattern holds — measured here at 193 ms per solve when
    the pattern moved every iteration, against a 25 ms budget for the whole optimization. Fixing the
    shape is what turns re-setup into a numeric update, and it is the same device AG3S uses for its
    own empty candidate slots.
    """

    value: np.ndarray  # (H, rows_per_step) clearance in metres; +inf where unused
    gradient: np.ndarray  # (H, rows_per_step, nq_opt); zero where unused
    used: np.ndarray  # (H, rows_per_step) bool
    slot: np.ndarray  # (H, rows_per_step) candidate slot, -1-plane_id for a plane, -999 unused
    sphere: np.ndarray  # (H, rows_per_step)
    n_considered: int
    budget_bound_steps: tuple[int, ...] = ()

    @property
    def n_rows(self) -> int:
        """Rows actually carrying a constraint. The QP always holds `value.size` of them."""
        return int(self.used.sum())

    @property
    def capacity(self) -> int:
        return int(self.value.size)


class CollisionBlock:
    """The fixed sparsity pattern of the collision rows, built once and refilled every iteration.

    Row ``i`` always belongs to step ``i // rows_per_step`` and always touches that step's `nq_opt`
    decision variables plus its own slack. Nothing about the scene can move a nonzero, so the same
    `indices`/`indptr` serve every solve and only `data` is rewritten.
    """

    def __init__(self, horizon: int, rows_per_step: int, nq_opt: int):
        self.horizon = int(horizon)
        self.rows_per_step = int(rows_per_step)
        self.nq_opt = int(nq_opt)
        self.n_rows = self.horizon * self.rows_per_step
        n_q = self.nq_opt * self.horizon

        # Each row: nq_opt entries in its own step's block, then a single +1 on its slack.
        step = np.repeat(np.arange(self.horizon), self.rows_per_step)
        base = (step * self.nq_opt).repeat(self.nq_opt)
        offset = np.tile(np.arange(self.nq_opt), self.n_rows)
        q_cols = base + offset
        slack_cols = n_q + np.arange(self.n_rows)

        indices = np.empty(self.n_rows * (self.nq_opt + 1), np.int32)
        block = self.nq_opt + 1
        for column in range(self.nq_opt):
            indices[column::block] = q_cols[column :: self.nq_opt]
        indices[self.nq_opt :: block] = slack_cols
        self._indices = indices
        self._indptr = np.arange(self.n_rows + 1, dtype=np.int32) * block
        self._n_col = n_q + self.n_rows

    def matrix(self, rows: "LinearizedRows", n_col: Optional[int] = None) -> sp.csr_matrix:
        """The block with this iteration's gradients written in. Same pattern, new numbers.

        `n_col` is the *problem's* width, which exceeds this block's own columns whenever the problem
        carries slack the collision rows do not use — the softened velocity and acceleration bounds,
        for instance. The pattern is unaffected: the extra columns are structurally empty.
        """
        block = self.nq_opt + 1
        data = np.empty(self.n_rows * block, np.float64)
        flat = rows.gradient.reshape(self.n_rows, self.nq_opt)
        for column in range(self.nq_opt):
            data[column::block] = flat[:, column]
        data[self.nq_opt :: block] = 1.0
        width = self._n_col if n_col is None else int(n_col)
        if width < self._n_col:
            raise ValueError(
                f"the problem has {width} columns but this block indexes up to {self._n_col}"
            )
        return sp.csr_matrix((data, self._indices, self._indptr), shape=(self.n_rows, width))


class CollisionLinearizer:
    """Evaluates clearances and their gradients for a whole chunk, and selects the rows that matter.

    The forward-kinematics function is built once per robot model and mapped over the horizon, which
    is where the measured cost went from ~19 s of graph construction to 66 ms.
    """

    def __init__(self, robot_model, layout: ChunkLayout, horizon: int):
        import casadi as ca

        self.robot_model = robot_model
        self.layout = layout
        self.horizon = int(horizon)
        self.nq_opt = layout.nq_opt

        q = ca.SX.sym("q", int(robot_model.nq), 1)
        spheres = robot_model.sphere_centers_symbolic(q)
        self.n_spheres = len(spheres)
        centres = ca.horzcat(*[c for c, _ in spheres])  # 3 x S
        free = [int(i) for i in layout.q_indices]
        jac = ca.jacobian(ca.reshape(centres, -1, 1), q[free])  # 3S x nq_opt
        self._fk = ca.Function("fk_jac", [q], [centres, jac])
        self._fk_map = self._fk.map(self.horizon)
        self.robot_radii = np.asarray([float(r) for _, r in spheres], np.float64)
        #: Extra query points riding a link -- the held object. See `set_attached`.
        self._attached_local: Optional[np.ndarray] = None
        self._attached_link: Optional[str] = None
        self._probe_maps: dict = {}
        #: Radii for every query point: the robot's spheres, then zeros for attached points.
        self.query_radii = self.robot_radii

        # Converting a CasADi DM to numpy costs about 45 ns per *dense* entry regardless of how it is
        # asked for, and the mapped Jacobian is 183 x 600 with only 23% of its entries structurally
        # nonzero. Pulling the nonzeros out and scattering them straight into the layout this class
        # wants takes 1.8 ms where densifying and transposing took 5.7 — measured, and it was the
        # single largest cost in the whole optimization once the QP had been fixed.
        probe = self._fk_map(np.zeros((int(robot_model.nq), self.horizon)))
        rows, cols = probe[1].sparsity().get_triplet()
        rows = np.asarray(rows, np.int64)
        cols = np.asarray(cols, np.int64)
        # A Jacobian row is (sphere, xyz) because CasADi reshapes in column order; a column is
        # (step, joint) because `map` concatenates the calls horizontally.
        sphere, xyz = rows // 3, rows % 3
        step, joint = cols // self.nq_opt, cols % self.nq_opt
        self._jac_shape = (self.horizon, self.n_spheres, 3, self.nq_opt)
        self._jac_dest = np.ravel_multi_index((step, sphere, xyz, joint), self._jac_shape)
        # Centres arrive dense as (3, H*S) in step-major column order; this permutation lands them
        # in (H, S, 3) without a transpose.
        c_step, c_sphere = np.divmod(np.arange(self.horizon * self.n_spheres), self.n_spheres)
        self._centre_dest = np.ravel_multi_index(
            (np.repeat(c_step, 3), np.repeat(c_sphere, 3), np.tile(np.arange(3), c_step.size)),
            (self.horizon, self.n_spheres, 3),
        )

    # --- held object ---------------------------------------------------------------------
    def set_attached(self, points_local, parent_link: Optional[str]) -> None:
        """Hold `points_local` (N, 3, in `parent_link`'s frame) as extra query points, or clear them.

        The points are appended to the robot's own spheres with radius zero, so every consumer that
        already walks the sphere axis picks them up: the ESDF rows, the activation band, the budget,
        and `full_violation`'s guarantee that the reported number came from every row.

        **The QP's sparsity does not move.** The block is `horizon x reduction.rows_per_step`
        regardless of how many query points exist -- widening the candidate list changes what
        *competes* for those rows, not how many there are. That is why a grasp can start and end
        mid-rollout without invalidating the factorization.

        Nothing is rebuilt symbolically when the point set changes. `_link_probe_map` differentiates
        four points per link -- the origin and its three unit axes -- and every attached point is an
        exact affine combination of those four, because a rigid transform is affine. So a new grasp
        costs one numpy matmul, not a CasADi graph.
        """
        if points_local is None or parent_link is None or len(points_local) == 0:
            self._attached_local = None
            self._attached_link = None
            self.query_radii = self.robot_radii
            return
        self._attached_local = np.asarray(points_local, np.float64).reshape(-1, 3)
        self._attached_link = str(parent_link)
        self.query_radii = np.concatenate(
            [self.robot_radii, np.zeros(self._attached_local.shape[0])]
        )

    @property
    def n_attached(self) -> int:
        return 0 if self._attached_local is None else int(self._attached_local.shape[0])

    def _link_probe_map(self, link: str):
        """Cached FK for four points rigidly fixed to `link`: its origin and its three unit axes.

        Returned mapped over the horizon. Any point `p` with local coordinates `(a, b, c)` is

            p(q) = o(q) + a (x(q) - o(q)) + b (y(q) - o(q)) + c (z(q) - o(q))

        which is linear in the four probe positions with weights summing to one, so both the point
        and its Jacobian follow from the probes by the same weights. Four probes therefore cover
        every possible attached point set on that link, and the graph is built once per link.
        """
        import casadi as ca

        cached = self._probe_maps.get(link)
        if cached is not None:
            return cached
        q = ca.SX.sym("q", int(self.robot_model.nq), 1)
        T = self.robot_model.link_pose_symbolic(q, link)
        origin = T[:3, 3]
        probes = ca.horzcat(origin, origin + T[:3, 0], origin + T[:3, 1], origin + T[:3, 2])
        free = [int(i) for i in self.layout.q_indices]
        jac = ca.jacobian(ca.reshape(probes, -1, 1), q[free])
        fn = ca.Function(f"probe_{link}", [q], [probes, jac]).map(self.horizon)

        probe_dm, jac_dm = fn(np.zeros((int(self.robot_model.nq), self.horizon)))
        rows, cols = jac_dm.sparsity().get_triplet()
        rows = np.asarray(rows, np.int64)
        cols = np.asarray(cols, np.int64)
        probe, xyz = rows // 3, rows % 3
        step, joint = cols // self.nq_opt, cols % self.nq_opt
        shape = (self.horizon, 4, 3, self.nq_opt)
        dest = np.ravel_multi_index((step, probe, xyz, joint), shape)
        c_step, c_probe = np.divmod(np.arange(self.horizon * 4), 4)
        c_dest = np.ravel_multi_index(
            (np.repeat(c_step, 3), np.repeat(c_probe, 3), np.tile(np.arange(3), c_step.size)),
            (self.horizon, 4, 3),
        )
        cached = (fn, dest, shape, c_dest)
        self._probe_maps[link] = cached
        return cached

    def attached_states(self, full: np.ndarray):
        """``(pos[H, N, 3], jac[H, N, 3, nq_opt])`` for the held object, or empty when nothing is held."""
        if self._attached_local is None:
            return (np.zeros((self.horizon, 0, 3)),
                    np.zeros((self.horizon, 0, 3, self.nq_opt)))
        fn, dest, shape, c_dest = self._link_probe_map(self._attached_link)
        probe_dm, jac_dm = fn(full)
        probes = np.zeros(self.horizon * 4 * 3)
        probes[c_dest] = np.asarray(probe_dm).T.reshape(-1)
        probes = probes.reshape(self.horizon, 4, 3)
        pj = np.zeros(int(np.prod(shape)))
        pj[dest] = np.asarray(jac_dm.nonzeros(), np.float64)
        pj = pj.reshape(shape)

        a = self._attached_local
        w = np.column_stack([1.0 - a.sum(axis=1), a])          # (N, 4), rows sum to 1
        pos = np.einsum("nk,hkj->hnj", w, probes)
        jac = np.einsum("nk,hkjm->hnjm", w, pj)
        return pos, jac

    # --- raw geometry --------------------------------------------------------------------
    def sphere_states(self, trajectory: np.ndarray, q_now: np.ndarray):
        """``(centres[H, Q, 3], jacobians[H, Q, 3, nq_opt])`` along the whole chunk.

        ``Q = S + N``: the robot's own spheres first, then the held object's points (radius zero).
        The candidate and plane blocks slice off the robot's part; the ESDF block uses all of it,
        because once a grasp closes the held object is part of the robot and has to be checked
        against the world like any other piece of it (E3 -- the held object never reached the
        optimizer at all before this).
        """
        full = self.layout.full_q(trajectory, q_now)
        centres_dm, jac_dm = self._fk_map(full)

        centres = np.zeros(self.horizon * self.n_spheres * 3)
        centres[self._centre_dest] = np.asarray(centres_dm).T.reshape(-1)
        jac = np.zeros(int(np.prod(self._jac_shape)))
        jac[self._jac_dest] = np.asarray(jac_dm.nonzeros(), np.float64)
        centres = centres.reshape(self.horizon, self.n_spheres, 3)
        jac = jac.reshape(self._jac_shape)
        if self._attached_local is None:
            return centres, jac
        a_pos, a_jac = self.attached_states(full)
        return (np.concatenate([centres, a_pos], axis=1),
                np.concatenate([jac, a_jac], axis=1))

    # --- clearances ----------------------------------------------------------------------
    def clearances(
        self, trajectory: np.ndarray, q_now: np.ndarray, scene: SceneSnapshot, states=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(candidate[H, S, M], plane[H, S, K])`` clearances in metres. Inactive slots are +inf.

        No Jacobian, so this is the cheap call. `full_violation` and the activation band both use it,
        which is why the safety re-check costs a fraction of a solve.
        """
        centres = (states or self.sphere_states(trajectory, q_now))[0]
        return self._clearances_from(centres, scene)

    def esdf_clearance(self, trajectory: np.ndarray, q_now: np.ndarray, scene: SceneSnapshot,
                       states=None) -> np.ndarray:
        """``(H, S, 1)`` ESDF clearances, or an empty last axis when no field is attached."""
        centres = (states or self.sphere_states(trajectory, q_now))[0]
        return self._esdf_clearance(centres, scene)

    def _esdf_clearance(self, centres: np.ndarray, scene: SceneSnapshot) -> np.ndarray:
        """``(H, S)`` — ``d_esdf(p) - r_robot - margin``. Inactive (no field) is an empty array.

        No slot loop and no selection over candidates: the field answers for the point directly.
        That is why this block is `S` rows where the primitive block is `S * M`.

        **Per-sphere margin for the manipulated object, not a global carve (E1).** AG3S no longer
        removes that object from the field, so `d` already reflects it as an obstacle for every
        sphere. A sphere whose nearest field obstacle *is* that object uses
        `scene.manipulated_link_margin` for itself instead of the flat `esdf_margin`. An unauthorized
        sphere's entry is the full margin already (AG3S builds it that way), so there is nothing else
        to check here: whether a link may touch it is entirely encoded in the numbers AG3S handed
        over.

        **The object is the one being manipulated, not the one attention is looking at** (F11).
        Those are the same until a grasp closes; afterwards the policy's attention moves to the
        destination while the hand still holds the object, and keying the permission on attention
        leaves the held object demanding full clearance from the fingers holding it.

        **The identification test is two-sided** (F13). Comparing only `d >= d_object - tol` is
        correct only while that object is actually in the field; when it is missing — a held object
        erased by the self-filter is exactly that case (F12) — the one-sided form also accepts
        spheres whose field obstacle is much *further* than the object, and relaxes them wrongly.
        Requiring the two distances to agree within a voxel says what was meant.
        """
        if scene.esdf is None:
            return np.zeros((centres.shape[0], centres.shape[1], 0))
        n_query = centres.shape[1]
        radii = self.query_radii
        if radii.shape[0] != n_query:
            raise ValueError(
                f"{n_query} query points but {radii.shape[0]} radii; `set_attached` and the states "
                "passed in disagree about what is being held"
            )
        flat = centres.reshape(-1, 3)
        d = np.asarray(scene.esdf.distance(flat), np.float64).reshape(centres.shape[:2])
        margin = np.full(centres.shape[:2], float(scene.esdf_margin))
        # 목적지: 가장 가까운 표면이 목적지인 질의점만 얇은 마진을 쓴다. 필드가 라벨을 함께
        # 답하게 된 덕분에 이 구분이 가능해졌다 — 그 전에는 거리장이 익명이라 "지금 가까운 것이
        # 목적지인가" 를 물을 수 없었고, 그래서 전역 마진을 내리는 것 말고는 방법이 없었다.
        #
        # **그리고 쥔 물체에만 붙는다.** 목적지 마진을 "가장 가까운 것이 목적지인 모든 질의점"
        # 에 걸었더니 로봇 팔과 몸통의 여유거리까지 함께 완화됐다 — 실측으로 잡았다(프레임 18
        # 에서 로봇 구 최악 -77.8 mm 가 -47.8 mm 로). 그것은 E1(조작 대상을 필드에서 파내면
        # 손끝뿐 아니라 전신에게 사라진다)과 같은 종류의 실수다. 바구니 안으로 들어가야 하는
        # 것은 **쥔 물체**이지 팔꿈치가 아니다.
        #
        # 손가락은 이 완화가 필요 없다 — 실측상 바구니 안에서 50.4~51.1 mm 여유가 있어 전역
        # 마진으로 충분하다. 필요해지는 날 (링크, 목적지) 쌍 권한으로 넓히면 되고, 그것은
        # `ClearancePolicy` 가 접촉 권한에 이미 쓰는 구조다.
        if scene.destination_label and getattr(scene.esdf, "has_labels", False):
            is_dest = np.asarray(
                scene.esdf.is_label(flat, scene.destination_label)
            ).reshape(centres.shape[:2])
            held = np.zeros(centres.shape[:2], bool)
            held[:, self.n_spheres:] = True
            margin = np.where(is_dest & held, float(scene.destination_margin), margin)
        if scene.manipulated_link_margin is not None:
            d_object = None
            if scene.manipulated_spheres is not None and len(scene.manipulated_spheres):
                # Held object: analytic distance to the union of its spheres, so no surface sampling
                # sits between the geometry and the test.
                delta = flat[:, None, :] - scene.manipulated_spheres[None, :, :]
                d_object = (np.linalg.norm(delta, axis=2)
                            - scene.manipulated_sphere_radii[None, :]).min(axis=1)
            elif scene.manipulated_points is not None and len(scene.manipulated_points):
                from scipy.spatial import cKDTree
                d_object = cKDTree(scene.manipulated_points).query(flat)[0]
            if d_object is not None:
                d_object = d_object.reshape(centres.shape[:2])
                tol = float(scene.esdf.grid.voxel_size)
                is_object = np.abs(d - d_object) <= tol
                per_link = np.asarray(scene.manipulated_link_margin, np.float64).reshape(-1)
                if per_link.shape[0] < n_query:
                    # The held object's own points get no relaxation: they *are* the object, so
                    # "may this link touch it" is not a question about them. Padding with the full
                    # margin is the fail-closed direction and keeps the array conformable.
                    per_link = np.concatenate(
                        [per_link, np.full(n_query - per_link.shape[0], float(scene.esdf_margin))]
                    )
                margin = np.where(is_object, per_link[None, :], margin)
        return (d - radii[None, :] - margin)[..., None]

    def _clearances_from(self, centres: np.ndarray, scene: SceneSnapshot):
        """``(candidate[H, S, M], plane[H, S, K], distance[H, S, M])`` — no ``delta`` array.

        Distances come from the expansion ``||p - c||^2 = ||p||^2 - 2 p.c + ||c||^2``, so the
        sphere-to-candidate term is one BLAS matmul instead of a materialized ``(H, S, M, 3)``
        difference — 2.3 MB per call at RB-Y1 scale, allocated three times per SQP iteration. The
        direction vector is only needed for the handful of rows that survive selection, and
        `linearize` computes those individually.
        """
        centres = centres[:, : self.n_spheres]   # 후보·평면 행은 로봇 구에만 붙는다
        flat = centres.reshape(-1, 3)  # (H*S, 3)
        cross = flat @ scene.candidate_pos.T  # (H*S, M)
        squared = (
            np.einsum("ij,ij->i", flat, flat)[:, None]
            - 2.0 * cross
            + np.einsum("ij,ij->i", scene.candidate_pos, scene.candidate_pos)[None, :]
        )
        shape = centres.shape[:2]
        distance = np.sqrt(np.maximum(squared, 0.0)).reshape(*shape, -1)
        required = (
            scene.robot_radii[None, :, None]
            + scene.candidate_radius[None, None, :]
            + scene.d_safe[None, :, :]
        )
        candidate = distance - required
        # An inactive slot is not "far away", it is *absent*: +inf keeps it out of every selection
        # and out of the violation report, rather than contributing a large positive clearance that
        # would flatter the numbers.
        candidate = np.where(scene.candidate_active[None, None, :], candidate, np.inf)

        plane = (
            centres @ scene.plane_normal.T
            - scene.plane_offset[None, None, :]
            - scene.robot_radii[None, :, None]
        )
        plane = np.where(scene.plane_active[None, None, :], plane, np.inf)
        return candidate, plane, distance

    def full_violation(
        self, trajectory: np.ndarray, q_now: np.ndarray, scene: SceneSnapshot, states=None
    ) -> float:
        """Worst clearance over **every** row, in metres. Positive means clear.

        This is the guarantee behind every reduction in this module. Selection decides what the QP
        optimizes; this decides what the caller is told, and it never looks at a subset.
        """
        states = states or self.sphere_states(trajectory, q_now)
        candidate, plane, _ = self._clearances_from(states[0], scene)
        worst = np.inf
        if candidate.size:
            worst = min(worst, float(np.min(candidate)))
        if plane.size:
            worst = min(worst, float(np.min(plane)))
        # The ESDF block is checked here too. The whole guarantee of the activation band is that
        # the *reported* number comes from every row, and adding a backend without adding it here
        # would quietly exempt it from that.
        esdf = self._esdf_clearance(states[0], scene)
        if esdf.size:
            worst = min(worst, float(np.min(esdf)))
        return worst

    def worst_row(
        self, trajectory: np.ndarray, q_now: np.ndarray, scene: SceneSnapshot, states=None
    ) -> tuple[float, Optional[dict]]:
        """`(full_violation 과 같은 값, 그 값을 만든 행의 신원)`.

        `T5f` 가 *"`violated` 가 **어느** 제약인가"* 에서 막혔다. 판정에는 `max_violation_m`
        한 숫자만 실려 있고, 그 숫자를 만든 행이 어느 link 대 어느 obstacle 인지는 여기 안에만
        있었다. 그것을 꺼내는 것이 이 메서드의 전부다.

        **`full_violation` 을 대신 부르는 것이 아니라 그것과 같은 일을 한 번 더 한다** — 그래서
        merit 평가 경로(SQP 안쪽 루프)는 손대지 않았다. `sqp._finish` 는 어차피 마지막에
        `full_violation` 을 한 번 부르므로, 거기서 이 메서드로 바꾸면 **추가 비용은 argmin
        세 번**뿐이다. clearance sweep 도 forward kinematics 도 늘지 않는다.

        신원의 키:

        | 키 | 무엇 |
        |---|---|
        | `block` | `candidate` · `plane` · `esdf` — 어느 제약 계열인가 |
        | `step` | chunk 안의 몇 번째 스텝인가 (0-based) |
        | `query` / `link` | 어느 질의점인가. `link` 는 그 구가 붙은 link 이름 |
        | `slot` | candidate slot 번호, 또는 plane 번호. ESDF 는 `None` |
        | `candidate_id` | 그 slot 을 소유한 AG3S candidate. slot 이 비면 `-1` |
        | `obstacle` | ESDF label 층이 답한 **가장 가까운 물체의 이름**. 없으면 `None` |
        | `point_m` | 그 질의점의 base frame 좌표 — 그림에 바로 찍을 수 있다 |

        **`None` 을 돌려주는 경우가 있다**: 활성 제약이 하나도 없으면 모든 행이 `+inf` 라
        argmin 에 뜻이 없다. 그때 첫 값은 `inf` 이고 신원은 `None` 이다 — 없는 신원을
        지어내지 않는다.
        """
        states = states or self.sphere_states(trajectory, q_now)
        centres = states[0]
        candidate, plane, _ = self._clearances_from(centres, scene)
        esdf = self._esdf_clearance(centres, scene)

        worst = np.inf
        best: Optional[dict] = None
        for name, block in (("candidate", candidate), ("plane", plane), ("esdf", esdf)):
            if not block.size:
                continue
            flat = int(np.argmin(block))
            value = float(block.reshape(-1)[flat])
            # `inf` 는 "그 행이 없다" 는 뜻이므로 신원의 후보가 아니다 (`_clearances_from` 이
            # 비활성 slot 을 그렇게 표시한다).
            if not np.isfinite(value) or value >= worst:
                continue
            idx = np.unravel_index(flat, block.shape)
            worst = value
            best = self._identify(name, idx, centres, scene, value)
        if best is None:
            # 신원이 없으면 값도 `full_violation` 과 같아야 한다 — 전부 `inf` 인 경우다.
            return float(self.full_violation(trajectory, q_now, scene, states)), None
        return worst, best

    def _identify(self, block: str, idx, centres: np.ndarray, scene: SceneSnapshot,
                  value: float) -> dict:
        """`worst_row` 가 고른 인덱스를 **사람이 읽을 이름**으로. 계산은 하지 않는다."""
        step, query = int(idx[0]), int(idx[1])
        slot = int(idx[2]) if len(idx) > 2 else None
        point = np.asarray(centres[step, query], np.float64)
        out: dict[str, Any] = {
            "block": block,
            "clearance_m": float(value),
            "step": step,
            "query": query,
            "link": self._query_name(query),
            "slot": None if block == "esdf" else slot,
            "candidate_id": None,
            "obstacle": None,
            "point_m": [float(v) for v in point],
        }
        if block == "candidate" and slot is not None:
            ids = np.asarray(scene.candidate_ids, np.int64).reshape(-1)
            if slot < ids.shape[0]:
                out["candidate_id"] = int(ids[slot])
        if block == "esdf":
            out["obstacle"] = self._esdf_label_name(point, scene)
        return out

    def _query_name(self, query: int) -> str:
        """질의점 하나의 이름. 로봇 구는 그 link 이름, 쥔 물체의 점은 `attached:<link>[i]`.

        **구 인덱스를 그대로 적지 않는 이유**는 T5f 가 막힌 지점 그 자체다 — 아무도 해석할 수
        없는 숫자는 신원이 아니다.
        """
        if query >= self.n_spheres:
            return f"attached:{self._attached_link}[{query - self.n_spheres}]"
        names = getattr(self.robot_model, "sphere_link_names", ()) or ()
        if query < len(names):
            return str(names[query])
        return f"sphere[{query}]"

    def _esdf_label_name(self, point: np.ndarray, scene: SceneSnapshot) -> Optional[str]:
        """ESDF label 층이 답하는 **가장 가까운 표면의 물체 이름**, 없으면 `None`.

        점 하나만 묻는다 — 비용이 없다. label 층이 없는 backend(라벨 없이 지은 필드)에서는
        `None` 이고, 그것은 *"이름을 모른다"* 이지 *"물체가 없다"* 가 아니다.
        """
        field = scene.esdf
        if field is None or not getattr(field, "has_labels", False):
            return None
        try:
            label = int(np.asarray(field.label(point.reshape(1, 3))).reshape(-1)[0])
            names = tuple(getattr(field, "label_names", ()) or ())
        except Exception:  # noqa: BLE001 — 이름을 못 읽는 것으로 판정 기록이 죽지 않는다
            return None
        if label < 0 or label >= len(names):
            return None
        return str(names[label])

    # --- selection + linearization -------------------------------------------------------
    def linearize(
        self,
        trajectory: np.ndarray,
        q_now: np.ndarray,
        scene: SceneSnapshot,
        config: TrajOptConfig,
        states=None,
    ) -> LinearizedRows:
        """Evaluate, select and differentiate the rows the QP will hold.

        Fully vectorized. An earlier version looped over the selected rows in Python and cost 13 ms
        at H=50 — half the budget for the entire optimization — for arithmetic numpy does in
        microseconds.
        """
        reduction = config.reduction
        budget = reduction.rows_per_step
        centres, jac = states or self.sphere_states(trajectory, q_now)
        candidate, plane, distance = self._clearances_from(centres, scene)
        esdf = self._esdf_clearance(centres, scene)

        horizon = centres.shape[0]
        # Two counts, deliberately. Candidate and plane rows are per *robot sphere*; ESDF rows are
        # per *query point*, which includes the held object's points (they carry no candidate slot
        # because they are not in the world, they are on the robot).
        n_spheres = self.n_spheres
        n_slots, n_planes = scene.n_slots, scene.n_planes
        band = reduction.activation_band if reduction.enabled else np.inf
        n_candidate = n_spheres * n_slots
        n_plane_rows = n_spheres * n_planes

        # One list per step: candidates, planes and the ESDF together. The budget should go to
        # whatever is tightest, and a table the gripper is about to hit matters exactly as much as a
        # crate — or as much as a voxel the field says is 2 cm away.
        flat = np.concatenate(
            [candidate.reshape(horizon, -1), plane.reshape(horizon, -1),
             esdf.reshape(horizon, -1)], axis=1
        )  # (H, S*M + S*K + S)
        if reduction.enabled and reduction.temporal_stride > 1:
            # Steps that are not enforced get +inf everywhere, so they select nothing while keeping
            # their rows allocated — the pattern must not depend on the stride either.
            mask = np.ones(horizon, bool)
            mask[:: reduction.temporal_stride] = False
            flat = flat.copy()
            flat[mask] = np.inf
        considered = int(np.isfinite(flat).sum())

        width = flat.shape[1]
        take = min(budget, width)
        # argpartition, not argsort: the order inside the selection is irrelevant and partitioning
        # is linear where sorting is not.
        picked = np.argpartition(flat, take - 1, axis=1)[:, :take] if take < width else (
            np.tile(np.arange(width), (horizon, 1))
        )
        if take < budget:
            # A scene with fewer possible rows than the budget still gets a full-width block. The
            # shape of the problem must depend on the *configuration*, never on what the cameras
            # happened to see this frame — that is the entire basis for reusing a factorization.
            padding = np.zeros((horizon, budget - take), np.int64)
            picked = np.concatenate([picked, padding], axis=1)
            take = budget
        step_index = np.repeat(np.arange(horizon), take)
        picked_flat = picked.reshape(-1)
        values = flat[step_index, picked_flat]
        used = np.isfinite(values) & (values < band)
        if width < budget:
            # The padded columns repeat row 0; mark them unused so they cannot be counted twice.
            used = used.reshape(horizon, take)
            used[:, width:] = False
            used = used.reshape(-1)

        # How many rows each step *wanted*, to report a budget that actually bound.
        eligible_per_step = np.sum(np.isfinite(flat) & (flat < band), axis=1)
        bound = tuple(int(k) for k in np.flatnonzero(eligible_per_step > take))

        is_candidate = picked_flat < n_candidate
        is_esdf = picked_flat >= n_candidate + n_plane_rows
        is_plane = (~is_candidate) & (~is_esdf)
        plane_local = picked_flat - n_candidate
        esdf_local = picked_flat - n_candidate - n_plane_rows
        sphere = np.where(
            is_candidate, picked_flat // max(n_slots, 1),
            np.where(is_plane, plane_local // max(n_planes, 1), esdf_local),
        )
        # `slot` identifies which thing a row is about: >= 0 a candidate slot, -1-k a plane, and
        # ESDF rows get one sentinel because there is nothing to enumerate — the field is a single
        # object and the row is about a *point*, not about a slot.
        slot = np.where(
            is_candidate,
            picked_flat % max(n_slots, 1),
            np.where(is_plane, -1 - (plane_local % max(n_planes, 1)), ESDF_SLOT),
        )
        sphere = np.where(used, sphere, 0)  # keep indices in range for the gather below

        # Direction: away from the candidate for a sphere row, along the normal for a plane row.
        direction = np.zeros((step_index.size, 3))
        if np.any(is_candidate & used):
            sel = is_candidate & used
            norm = distance[step_index[sel], sphere[sel], slot[sel]]
            # Only the selected rows need a direction, so the (H, S, M, 3) difference array is never
            # built — this gathers the few hundred vectors that matter.
            unit = (
                centres[step_index[sel], sphere[sel]] - scene.candidate_pos[slot[sel]]
            ) / np.maximum(norm, _EPS)[:, None]
            # p == c exactly: no separating direction exists. Push along +x rather than emit NaN.
            unit[norm <= _EPS] = np.array([1.0, 0.0, 0.0])
            direction[sel] = unit
        if np.any(is_plane & used):
            sel = is_plane & used
            direction[sel] = scene.plane_normal[-1 - slot[sel]]
        if np.any(is_esdf & used):
            # The ESDF's own gradient is the separating direction, and it is already unit-magnitude
            # in free space because a distance field satisfies the eikonal equation. Normalising it
            # anyway would hide a field that had gone wrong, so it is used as computed and only
            # rescued where it vanishes (deep inside an obstacle, where the field is flat).
            sel = is_esdf & used
            grad = np.asarray(scene.esdf.gradient(centres[step_index[sel], sphere[sel]]), float)
            norm = np.linalg.norm(grad, axis=1)
            grad = np.where(norm[:, None] > _EPS, grad / np.maximum(norm, _EPS)[:, None],
                            np.array([1.0, 0.0, 0.0]))
            direction[sel] = grad

        gathered = jac[step_index, sphere]  # (n, 3, nq_opt)
        gradient = np.einsum("ni,nij->nj", direction, gathered)
        gradient[~used] = 0.0
        values = np.where(used, values, np.inf)

        shape = (horizon, take)
        return LinearizedRows(
            value=values.reshape(shape),
            gradient=gradient.reshape(horizon, take, self.nq_opt),
            used=used.reshape(shape),
            slot=np.where(used, slot, -999).reshape(shape),
            sphere=sphere.reshape(shape),
            n_considered=considered,
            budget_bound_steps=bound,
        )



def _check_support_surface_invariant(constraint_set, config) -> None:
    """지지면을 제약하는 것이 **하나는** 있는지 확인한다 (F15).

    스위치가 둘이고 **서로 다른 config 에 산다**:

        AG3SConfig.esdf.exclude_support_surfaces   지각 쪽 — 표면을 필드에서 파낼 것인가
        TrajOptConfig.collision.use_support_planes 최적화 쪽 — 평면 행을 읽을 것인가

    네 조합 중 둘만 일관된다. 파냈으면 평면 행이 받아야 하고, 안 파냈으면 필드가 이미 담고 있다.
    **`파냄 + 평면 행 없음` 은 지지면을 아무도 제약하지 않는 상태**이고, 그것이 이 검사가 막는
    것이다. 실측(run_0004, 3 카메라): 파내면 상판 z=0.823 에서 필드가 **-30.5 mm → +6.0 mm** 로
    바뀐다 — 표면이 자유공간이 된다. 손끝이 실제로 다투는 대역에서 **최대 36.5 mm 낙관적**이다.

    도달하기 쉬운 조합이라는 것이 문제의 핵심이다. `SafePolicy` 는 `to_config` 기본값으로
    `use_support_planes: False` 를 쓰면서 `ag3s` 는 **밖에서 주입받는다.** 그래서

        ag3s = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf"}), ...)   # exclude 기본 True
        SafePolicy(policy, ag3s=ag3s)                                            # use_planes 기본 False

    이라는 지극히 자연스러운 코드가 위험 조합을 만든다. 두 config 중 어느 쪽도 상대를 볼 수
    없으므로, **둘이 만나는 이 지점**이 검사할 수 있는 유일한 자리다.

    파냈다는 사실은 추측하지 않고 필드가 스스로 보고한 것을 읽는다
    (`EsdfField.stats["n_support_voxels_carved"]`, `esdf.py:568`).
    """
    field = getattr(constraint_set, "esdf", None)
    if field is None:
        return
    stats = getattr(field, "stats", None) or {}
    carved = int(stats.get("n_support_voxels_carved", 0) or 0)
    if carved <= 0:
        return
    if getattr(getattr(config, "collision", None), "use_support_planes", True):
        return
    n_planes = len(getattr(constraint_set, "support_surfaces", ()) or ())
    raise ValueError(
        f"지지면을 제약하는 것이 없습니다 (F15): AG3S 가 지지면 복셀 {carved} 개를 필드에서 "
        f"파냈는데(esdf.exclude_support_surfaces=True) collision.use_support_planes=False 라 "
        f"평면 {n_planes} 개가 최적화기에 도달하지 않습니다. 테이블 상판이 자유공간으로 보입니다 "
        f"— 실측 +36.5 mm. 둘 중 하나로 맞추세요: "
        f"exclude_support_surfaces=False (필드가 담게, 두 실제 경로가 쓰는 쪽) 또는 "
        f"use_support_planes=True (평면 행이 받게)."
    )


def scene_from_constraint_set(constraint_set, robot_radii: np.ndarray,
                             config) -> Optional[SceneSnapshot]:
    """AG3S 의 `CollisionConstraintSet` -> 이 optimizer 가 실제로 쓸 `SceneSnapshot`.

    이 함수가 있는 이유는 **AG3S 가 backend 와 무관하게 candidate 를 항상 내놓기** 때문이다.
    target/obstacle 분리는 candidate 목록의 성질이고 clearance policy 가 읽는 것도 그것이라,
    ESDF 를 켠다고 후보가 사라지지 않는다. 무엇을 제약으로 삼을지는 **소비자의 선택**이고
    그 선택이 여기서 일어난다.

    * `primitive` — 후보 구와 평면. 필드는 붙이지 않는다.
    * `esdf` — 필드와 평면. **후보 슬롯을 전부 비활성화한다.** 그러지 않으면 같은 기하가 두 번
      제약되고, 그건 `both` 이지 `esdf` 가 아니다.
    * `both` — 셋 다. 두 표현을 한 QP 에 넣어 행 단위로 비교할 때 쓴다.

    평면(지지면)은 어느 backend 에서도 남는다. 평면 하나는 선형 행 하나로 정확한데 같은 표면을
    복셀로 옮기면 수천 행이 되고 근사도 나빠지기 때문이며, ESDF 쪽은 그 표면을 필드에서 파낸다.
    """
    spec = constraint_set.constraints
    backend = getattr(getattr(config, "collision", None), "backend", "primitive")
    radii = np.asarray(robot_radii, np.float64).reshape(-1)
    _check_support_surface_invariant(constraint_set, config)
    # E1: AG3S no longer carves the manipulated object out of the field, so a sphere on an authorized
    # link needs its own relaxed margin instead — read straight off `CollisionConstraintSet`, nothing
    # recomputed here. All are `None` together whenever nothing is being manipulated this frame.
    # Which object that is — the held one, or the attention target when nothing is held — is AG3S's
    # decision (`clearance.manipulated_object`), not one this adapter re-derives (F11).
    manipulated = getattr(constraint_set, "manipulated", None)
    man_points = getattr(manipulated, "points", None)
    man_spheres = getattr(manipulated, "sphere_centers", None)
    man_radii = getattr(manipulated, "sphere_radii", None)
    manipulated_link_margin = getattr(constraint_set, "manipulated_link_margin", None)
    # 쥔 물체는 이제 **로봇 쪽 질의점**이다 (E3 — 쥔 물체가 optimizer 에 도달하지 않던 문제).
    # primitive 가 아니라 점인 이유는 F19: 관측 점구름에 구 하나를 맞추면 꼭지·잎까지 덮느라
    # 반지름이 57.4 mm 가 되어 여유를 중앙 13.2 mm 먹고, 한 프레임에서는 없는 충돌을 만든다.
    attached = getattr(constraint_set, "attached", None)
    attached_points = getattr(attached, "points", None) if attached is not None else None
    attached_link = getattr(attached, "parent_link", None) if attached is not None else None
    if attached_points is not None and len(attached_points) == 0:
        attached_points, attached_link = None, None
    # 목적지 마진. AG3S 가 정책에서 뽑아 실어 보낸 값을 그대로 읽는다 — 여기서 다시 계산하지
    # 않는 것은 마진이 나오는 곳이 하나여야 하기 때문이다 (`ClearancePolicy`).
    destination_label = getattr(constraint_set, "destination_label", None)
    destination_margin = getattr(constraint_set, "destination_margin", None)
    if destination_margin is None:
        destination_label = None
        destination_margin = 0.0
    if manipulated_link_margin is not None and manipulated_link_margin.shape[0] != radii.size:
        raise ValueError(
            f"manipulated_link_margin has {manipulated_link_margin.shape[0]} entries but "
            f"{radii.size} robot radii were given; the optimizer and AG3S must share one "
            "constraint model"
        )
    if spec is None:
        # No `ConstraintSpec` at all. With the field that is a complete scene — an ESDF-only frame
        # never builds the primitive parameter vector — so return a field-only snapshot rather than
        # `None`, which the optimizer reads as "nothing to avoid".
        field = getattr(constraint_set, "esdf", None)
        if field is None or backend == "primitive":
            return None
        empty = SceneSnapshot(
            candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
            candidate_active=np.zeros(0, bool), d_safe=np.zeros((radii.size, 0)),
            plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0),
            plane_active=np.zeros(0, bool), robot_radii=radii,
            candidate_ids=np.zeros(0, np.int64),
            esdf=field, esdf_margin=float(config.collision.esdf_margin),
            manipulated_points=man_points, manipulated_spheres=man_spheres,
            manipulated_sphere_radii=man_radii,
            manipulated_link_margin=manipulated_link_margin,
            attached_points=attached_points, attached_parent_link=attached_link,
            destination_label=destination_label,
            destination_margin=float(destination_margin))
        return empty
    scene = SceneSnapshot.from_spec(spec, radii)
    if backend not in ("primitive", "esdf", "both"):
        raise ValueError(f"알 수 없는 collision backend: {backend!r}")

    if backend in ("esdf", "both"):
        field = getattr(constraint_set, "esdf", None)
        if field is None:
            raise ValueError(
                f"collision.backend={backend!r} 인데 CollisionConstraintSet 에 ESDF 가 없습니다. "
                "AG3S 쪽 `collision_backend` 를 'esdf' 또는 'both' 로 두고, depth 를 넘겼는지 "
                "확인하세요 — point cloud 만으로는 투영 TSDF 를 만들 수 없습니다.")
        scene = dataclasses.replace(
            scene, esdf=field, esdf_margin=float(config.collision.esdf_margin),
            manipulated_points=man_points, manipulated_spheres=man_spheres,
            manipulated_sphere_radii=man_radii,
            manipulated_link_margin=manipulated_link_margin,
            attached_points=attached_points, attached_parent_link=attached_link,
            destination_label=destination_label,
            destination_margin=float(destination_margin))

    if not getattr(config.collision, "use_support_planes", True):
        # 평면도 끈다(지우지 않는다). AG3S 는 계속 평면을 뽑는다 — grounding 이 그 마스크 없이는
        # 테이블을 타고 번지기 때문이다. 여기서 정하는 것은 **최적화기가 무엇을 제약으로 읽는가**
        # 뿐이고, 표면 자체는 필드에 남아 있다.
        scene = dataclasses.replace(
            scene, plane_active=np.zeros_like(scene.plane_active))

    if backend == "esdf":
        # 후보를 지우지 않고 **끈다**. 슬롯 배치가 그대로여야 희소성 패턴이 유지되고, 그래야
        # QP 의 인수분해를 프레임 간에 재사용할 수 있다 — 이 optimizer 의 실시간성이 거기 걸려 있다.
        scene = dataclasses.replace(
            scene, candidate_active=np.zeros_like(scene.candidate_active))
    return scene


def append_collision_rows(
    problem,
    rows: LinearizedRows,
    iterate: np.ndarray,
    block: CollisionBlock,
    backoff: float = 0.0,
):
    """Write the linearized rows into a `QpProblem` that reserved slack for them.

    The row is ``value + grad . (Q - Q_k) + s >= backoff``, rearranged so the variable side is on
    the left:

        grad . Q + s  >=  grad . Q_k - value + backoff

    `backoff` is what makes the *true* constraint hold rather than only its tangent — see
    `config.ConstraintReductionConfig.linearization_backoff`.

    An unused row has a zero gradient and a lower bound of zero, leaving ``s >= 0`` — already true,
    so the row costs the solver nothing while keeping the pattern intact.
    """
    if block.n_rows != problem.n_slack:
        raise ValueError(
            f"the collision block holds {block.n_rows} rows but {problem.n_slack} slacks were "
            "reserved; both come from reduction.rows_per_step x horizon and must agree"
        )
    iterate_flat = np.asarray(iterate, np.float64).T.reshape(-1)
    gradient = rows.gradient.reshape(block.n_rows, -1)
    step = np.repeat(np.arange(rows.value.shape[0]), rows.value.shape[1])
    # grad . Q_k, evaluated per row against its own step's block only.
    at_iterate = np.einsum(
        "nj,nj->n", gradient, iterate_flat.reshape(-1, block.nq_opt)[step]
    )
    lower = np.where(
        rows.used.reshape(-1), at_iterate - rows.value.reshape(-1) + float(backoff), 0.0
    )

    A = sp.vstack([problem.A, block.matrix(rows, problem.A.shape[1])], format="csc")
    start = problem.A.shape[0]
    row_blocks = dict(problem.row_blocks)
    row_blocks["collision"] = (start, start + block.n_rows)
    return dataclasses.replace(
        problem,
        A=A,
        l=np.concatenate([problem.l, lower]),
        u=np.concatenate([problem.u, np.full(block.n_rows, np.inf)]),
        row_blocks=row_blocks,
    )


__all__ = [
    "CollisionBlock",
    "CollisionLinearizer",
    "LinearizedRows",
    "SceneSnapshot",
    "append_collision_rows",
]
