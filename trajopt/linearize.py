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

    # --- raw geometry --------------------------------------------------------------------
    def sphere_states(self, trajectory: np.ndarray, q_now: np.ndarray):
        """``(centres[H, S, 3], jacobians[H, S, 3, nq_opt])`` along the whole chunk."""
        full = self.layout.full_q(trajectory, q_now)
        centres_dm, jac_dm = self._fk_map(full)

        centres = np.zeros(self.horizon * self.n_spheres * 3)
        centres[self._centre_dest] = np.asarray(centres_dm).T.reshape(-1)
        jac = np.zeros(int(np.prod(self._jac_shape)))
        jac[self._jac_dest] = np.asarray(jac_dm.nonzeros(), np.float64)
        return (
            centres.reshape(self.horizon, self.n_spheres, 3),
            jac.reshape(self._jac_shape),
        )

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
        """
        if scene.esdf is None:
            return np.zeros((centres.shape[0], centres.shape[1], 0))
        flat = centres.reshape(-1, 3)
        d = np.asarray(scene.esdf.distance(flat), np.float64).reshape(centres.shape[:2])
        return (d - scene.robot_radii[None, :] - float(scene.esdf_margin))[..., None]

    def _clearances_from(self, centres: np.ndarray, scene: SceneSnapshot):
        """``(candidate[H, S, M], plane[H, S, K], distance[H, S, M])`` — no ``delta`` array.

        Distances come from the expansion ``||p - c||^2 = ||p||^2 - 2 p.c + ||c||^2``, so the
        sphere-to-candidate term is one BLAS matmul instead of a materialized ``(H, S, M, 3)``
        difference — 2.3 MB per call at RB-Y1 scale, allocated three times per SQP iteration. The
        direction vector is only needed for the handful of rows that survive selection, and
        `linearize` computes those individually.
        """
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

        horizon, n_spheres = centres.shape[0], centres.shape[1]
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
            esdf=field, esdf_margin=float(config.collision.esdf_margin))
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
            scene, esdf=field, esdf_margin=float(config.collision.esdf_margin))

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
