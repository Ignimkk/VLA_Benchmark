"""Discrete-time CBF-QP safety filter — KNOWS Eq. (7), (8), (12), plus four measured extensions.

Everything here is in **physical units** (metres, radians): Appendix 7.1 says the QP receives the
nominal action "scaled to physical units", so this must sit downstream of de-normalization.

`CbfParams` defaults reproduce the paper exactly. Each extension is off until asked for, so a run
can always be pinned to "Eq. (12) as written" for comparison. What they are and why
(docs/15-development-plan.md):

D1  `RobotBody`      several ellipsoids for the robot instead of one. A container decomposed into
                     walls has a genuinely empty interior, and a held object can be promoted from
                     obstacle to robot part rather than being dropped from Eq. (4) entirely.
D2  `normals=`       drop the virtual-normal variables. h(n) >= 0 for *any* fixed unit n already
                     certifies separation, so `delta_n` buys tightness, not safety -- and the
                     epsilon box that bounds it relaxes the constraint by more than gamma_h|h|
                     demands, which is how the filter ends up inert (docs/11 §4).
D3  `slack_weight`   soften the barrier rows so the QP is always feasible, and make emergency stop
                     an explicit threshold on the slack rather than a solver failure. Eq. (12) has
                     no bound on delta_c, so it never goes infeasible and the paper's own
                     emergency-stop branch is unreachable (measured: 0.0% over 5,879 steps).
D4  `half_spaces=`   support surfaces as planes. A table is not an ellipsoid, and the paper's
                     "manipulable objects" segmentation leaves it unprotected entirely.
D7  `ArticulatedBody` solve for joint increments instead of an end-effector delta, via the
                     Jacobian. Required for RB-Y1, which is commanded in joint space and cannot
                     accept Eq. (12)'s answer at all -- and it dissolves the paper's own §5
                     limitation, since protecting only the end-effector was a consequence of only
                     being able to command it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import osqp
import scipy.sparse as sp

from benchmark.knows_vla.cbf.ellipsoid import (
    EPS_DENOM,
    Ellipsoid,
    barrier,
    grad_center,
    grad_normal,
    grad_rotation,
    initial_normal,
    optimal_normal,
    optimal_normals_batch,
    rotate_shape,
)


@dataclasses.dataclass(frozen=True)
class HalfSpace:
    """Keep-out plane: the free side is ``a . x >= b``, with ``a`` a unit normal.

    Exact, not an approximation -- a table top really is a half-space, whereas the ellipsoid that
    would have to enclose it reaches metres in every direction (the room body came out with a
    13.9 m semi-axis). It also needs no virtual normal: ``a`` is fixed, so there is nothing to
    relax and no epsilon.
    """

    a: np.ndarray
    b: float

    def __post_init__(self):
        a = np.asarray(self.a, np.float64).reshape(3)
        nrm = float(np.linalg.norm(a))
        object.__setattr__(self, "a", a / nrm if nrm > EPS_DENOM else np.array([0.0, 0.0, 1.0]))
        object.__setattr__(self, "b", float(self.b))

    def barrier(self, E: Ellipsoid) -> float:
        """Signed clearance of an ellipsoid from the plane: a.c - b - sqrt(a' Q a)."""
        return float(self.a @ E.c) - self.b - float(np.sqrt(max(self.a @ E.Q @ self.a, 0.0)))


@dataclasses.dataclass(frozen=True)
class RobotBody:
    """The robot as one or more ellipsoids rigidly attached to the end-effector frame.

    ``parts`` centres are **relative to ``origin``**, which is what makes the lever arm explicit:
    rotating the wrist by ``w`` moves an offset part's centre by ``w x r``, contributing ``r x n``
    to the rotation row (see ``_rows``). With a single part at the origin that term vanishes and
    the constraint is Eq. (9)-(10) unchanged.
    """

    origin: np.ndarray
    parts: dict[str, Ellipsoid]

    def __post_init__(self):
        object.__setattr__(self, "origin", np.asarray(self.origin, np.float64).reshape(3))

    @staticmethod
    def single(E: Ellipsoid) -> "RobotBody":
        """The paper's model: one ellipsoid, centred on the end-effector."""
        return RobotBody(E.c, {"eef": Ellipsoid(np.zeros(3), E.Q)})

    def world(self, name: str) -> Ellipsoid:
        p = self.parts[name]
        return Ellipsoid(self.origin + p.c, p.Q)

    @property
    def n_var(self) -> int:
        return 6  # (delta_c, delta_theta), the paper's decision variables

    def project(self, name: str, A_c: np.ndarray, A_rot: np.ndarray) -> np.ndarray:
        """Task-space gradients at one part -> its row over (delta_c, delta_theta).

        The lever arm lives here: rotating the wrist by ``w`` translates an offset part's centre by
        ``w x r``, which contributes ``r x A_c`` to the rotation block. Eq. (9)-(10) unchanged when
        ``r`` is zero.
        """
        return np.concatenate([A_c, A_rot + np.cross(self.parts[name].c, A_c)])

    def apply(self, x: np.ndarray):
        return x[0:3].copy(), x[3:6].copy(), None

    def moved(self, delta_c: np.ndarray, delta_theta: np.ndarray) -> "RobotBody":
        """Apply a step: parts rotate rigidly with the wrist, so their offsets rotate too."""
        w = np.asarray(delta_theta, np.float64).reshape(3)
        R = np.eye(3) + np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        return RobotBody(self.origin + np.asarray(delta_c, np.float64).reshape(3),
                         {k: Ellipsoid(R @ p.c, rotate_shape(p.Q, w)) for k, p in self.parts.items()})


@dataclasses.dataclass(frozen=True)
class Link:
    """One rigid piece of the robot, with the Jacobian that maps joint motion to its own motion."""

    ellipsoid: Ellipsoid  # world pose
    Jv: np.ndarray  # (3, nq) d(centre) / d(q)
    Jw: np.ndarray  # (3, nq) d(orientation) / d(q), world-frame angular velocity

    def __post_init__(self):
        for f in ("Jv", "Jw"):
            object.__setattr__(self, f, np.asarray(getattr(self, f), np.float64).reshape(3, -1))
        if self.Jv.shape != self.Jw.shape:
            raise ValueError(f"Jv {self.Jv.shape} and Jw {self.Jw.shape} disagree on nq")


@dataclasses.dataclass(frozen=True)
class ArticulatedBody:
    """The robot as links in joint space -- the decision variable becomes ``delta_q``.

    Why this exists: Eq. (12) solves for an end-effector delta, which presumes the robot is
    commanded that way. RB-Y1 is not; the policy emits joint targets, so ``delta_c`` and
    ``delta_theta`` are not quantities anyone can send. Substituting the kinematics,

        delta_c = Jv delta_q,   delta_theta = Jw delta_q

    turns every barrier row into a row over ``delta_q`` and leaves Eq. (5)-(11) untouched.

    It also removes the limitation the paper states in its own §5 -- that only the end-effector is
    protected, "however, our safety filter assumes control of only the end effector position and
    pose". That assumption is what forced the restriction, and commanding joints dissolves it: give
    each link its Jacobian and every (link, obstacle) pair contributes a row.

    `reproduction.md` class: **engineering adaptation**. The method's intent (project the nominal
    action onto the safe set) is preserved exactly; only the parameterisation of "action" changes.
    """

    links: dict[str, Link]
    eef: str | None = None  # which link's twist to report back; defaults to the first

    def __post_init__(self):
        if not self.links:
            raise ValueError("ArticulatedBody needs at least one link")
        nqs = {L.Jv.shape[1] for L in self.links.values()}
        if len(nqs) != 1:
            raise ValueError(f"links disagree on nq: {nqs}")
        if self.eef is None:
            object.__setattr__(self, "eef", sorted(self.links)[0])

    @property
    def parts(self) -> dict:
        return self.links

    @property
    def nq(self) -> int:
        return next(iter(self.links.values())).Jv.shape[1]

    @property
    def n_var(self) -> int:
        return self.nq

    def world(self, name: str) -> Ellipsoid:
        return self.links[name].ellipsoid

    def project(self, name: str, A_c: np.ndarray, A_rot: np.ndarray) -> np.ndarray:
        """Chain rule: dh/dq = (dh/dc)' Jv + (dh/dR)' Jw. No lever-arm term -- Jv already carries it."""
        L = self.links[name]
        return A_c @ L.Jv + A_rot @ L.Jw

    def apply(self, x: np.ndarray):
        """Returns (twist of the reporting link, delta_q). delta_q is what gets commanded."""
        L = self.links[self.eef]
        return L.Jv @ x, L.Jw @ x, x.copy()

    def moved(self, delta_q: np.ndarray, _unused=None) -> "ArticulatedBody":
        """First-order step. Jacobians are held fixed: re-deriving them needs the kinematic model,
        and over one control step the change is second order."""
        dq = np.asarray(delta_q, np.float64).reshape(-1)
        links = {}
        for k, L in self.links.items():
            w = L.Jw @ dq
            links[k] = Link(Ellipsoid(L.ellipsoid.c + L.Jv @ dq, rotate_shape(L.ellipsoid.Q, w)),
                            L.Jv, L.Jw)
        return ArticulatedBody(links, self.eef)


@dataclasses.dataclass(frozen=True)
class CbfParams:
    """Defaults reproduce the paper. The five values it never states are marked [ASSUMPTION].

    See docs/OPEN-QUESTIONS.md #1.
    """

    gamma_h: float = 0.5  # Eq. (7) decay, (0, 1]              [ASSUMPTION]
    W: float = 1.0  # Eq. (12) rotation vs translation weight   [ASSUMPTION]
    eps_normal: float = 0.05  # Eq. (12) ||delta n||_inf bound  [ASSUMPTION]
    q_eig_floor: float = 1e-8  # guards sqrt(n' Q n) -> 0 for near-degenerate shapes
    # OSQP tolerances. The paper names the solver [39] but gives no settings [ASSUMPTION].
    # Defaults (1e-3) leave primal residuals around 1e-5 m on the barrier constraint, i.e. the
    # returned action can violate Eq. (8) by microns. Harmless physically, but a safety filter
    # should not be the loosest link, and tightening costs little at this problem size.
    solver_eps_abs: float = 1e-9
    solver_eps_rel: float = 1e-9
    solver_max_iter: int = 20_000
    # NOT in the paper. Eq. (12) bounds delta_n but never delta_c or delta_theta, so a badly
    # violated barrier can demand an arbitrarily large correction -- measured up to 5 m against a
    # 5 cm OSC action limit (docs/11-p2b-results.md). See OPEN-QUESTIONS #13.
    max_delta_pos: float | None = None
    max_delta_rot: float | None = None

    # --- D2: how the separating normal is chosen -------------------------------------------
    # 'relaxed'    Eq. (12) as written: delta_n decision variables inside an eps box.
    # 'fixed'      no delta_n. Recompute the tightest normal at the current pose each step and hold
    #              it. Since h(n) <= h* for every unit n, a fixed normal can only *under*-report
    #              the gap, so this is conservative -- and it deletes eps from the parameter set.
    # 'fixedpoint' 'fixed', then re-solve once with the normal recomputed at the resulting pose.
    #              DO NOT USE. It does cut intervention (44% of steps vs 63%, p95 correction 4.6 cm
    #              vs 7.5 cm) but it buys that by taking steps that break the true CBF condition on
    #              47.4% of steps -- worse than the paper's own 26.5%. Kept only so the measurement
    #              is reproducible; see docs/16-filter-variants.md.
    normals: str = "relaxed"
    normal_iters: int = 60  # warm-started ascent budget for the two fixed modes
    # Skip the ascent for pairs that cannot become active this step, and reuse the warm normal.
    # Exact, not an approximation: h(n) <= h* for any unit n, so h(warm) above this threshold
    # proves the true gap is too. With |delta_c| bounded by the controller (5 cm) such a row stays
    # inactive, and an inactive row's normal does not affect the solution. Without it, whole-arm
    # protection costs ~1 ms per (link, obstacle) pair -- 60 ms at 6 links x 6 obstacles.
    normal_skip_h: float = 0.15

    # --- D3: slack ---------------------------------------------------------------------------
    # None reproduces the paper: hard barrier rows, emergency stop when the solver reports
    # infeasible. A float turns each row into `>= -gamma_h h - s` with `slack_weight * ||s||^2`
    # added to the cost, so the QP always has an answer and stopping becomes a decision about how
    # much violation was unavoidable rather than a solver outcome.
    # Keep this within ~1e3 of the position cost (which is 2.0): the ratio is the QP's condition
    # number, and at the 1e-9 tolerances above a weight of 1e4 needs >20k iterations to converge,
    # at which point the solver's "max iterations" is misread as infeasibility -- exactly the
    # failure mode slack exists to remove.
    slack_weight: float | None = None
    slack_stop: float = 0.02  # metres of unavoidable violation that trigger an emergency stop

    # --- joint-space parameterisation (ArticulatedBody) --------------------------------------
    # The analogue of max_delta_pos once the decision variable is delta_q: a per-step joint limit,
    # scalar or per-joint. Not optional on hardware -- an unbounded Eq. (12) asked for 5 m of
    # end-effector correction (docs/11 §3), and the joint-space version is no better behaved.
    max_delta_q: float | np.ndarray | None = None
    # Relative cost of deviating on each joint. Useful to make the filter prefer moving the wrist
    # over the base, which is the difference between a small correction and driving the robot.
    joint_weights: np.ndarray | None = None


@dataclasses.dataclass
class FilterResult:
    delta_c: np.ndarray  # (3,) applied translational delta
    delta_theta: np.ndarray  # (3,) applied rotational delta (world-frame axis-angle)
    feasible: bool
    emergency_stop: bool
    h: np.ndarray  # (m,) barrier value per constraint row, before the step
    status: str
    slack: np.ndarray | None = None  # (m,) per-row violation the QP could not avoid
    rows: tuple = ()  # (part, obstacle-key) per row, so a caller can name what is binding
    delta_q: np.ndarray | None = None  # joint-space answer; None unless the body was articulated


class SafetyFilter:
    """Stateful CBF-QP. The separating normals persist across calls, which is what warm-starts it."""

    def __init__(self, params: CbfParams | None = None):
        self.p = params or CbfParams()
        if self.p.normals not in ("relaxed", "fixed", "fixedpoint"):
            raise ValueError(f"unknown normals mode: {self.p.normals}")
        self._normals: dict[tuple, np.ndarray] = {}

    def reset(self) -> None:
        self._normals.clear()

    def normals(self) -> dict[tuple, np.ndarray]:
        return {k: v.copy() for k, v in self._normals.items()}

    def _batch_normals(self, body, obstacles: dict):
        """Separating normals for every (part, obstacle) pair, in one vectorised ascent.

        Doing this pair-by-pair is what made whole-arm protection unaffordable: the ascent touches
        3-vectors, so the Python loop dominates and 36 pairs cost ~40 ms against an 11.4 ms budget.
        Skipped pairs (`normal_skip_h`) never enter the batch at all.
        """
        pairs, robots, obs_list, inits = [], [], [], []
        for pname in sorted(body.parts):
            part = body.world(pname)
            for okey in sorted(obstacles, key=repr):
                key = (pname, okey)
                obs = obstacles[okey]
                prev = self._normals.get(key)
                if self.p.normals == "relaxed":
                    if prev is None:
                        self._normals[key] = initial_normal(part, obs)
                    continue
                n = prev if prev is not None else initial_normal(part, obs)
                if barrier(n, part, obs) > self.p.normal_skip_h:
                    self._normals[key] = n  # inactive: its exact direction cannot matter
                    continue
                pairs.append(key)
                robots.append(part)
                obs_list.append(obs)
                inits.append(n)
        if pairs:
            N = optimal_normals_batch(robots, obs_list, inits=inits, iters=self.p.normal_iters)
            for key, n in zip(pairs, N):
                self._normals[key] = n

    def _rows(self, body, obstacles: dict, half_spaces: dict):
        """One constraint row per (robot part, obstacle) pair.

        Returns (keys, h, A_c, A_rot, A_n) with the gradients **at the part**, before any
        projection onto the decision variables: A_c is dh/d(part centre) and A_rot is the shape
        term of dh/d(part orientation). Keeping them unprojected is what lets the same rows serve
        an end-effector-commanded robot and a joint-commanded one -- `body.project` supplies the
        lever arm in the first case and the Jacobian in the second. A_n is None for half-space
        rows: a plane's normal is data, not a variable.
        """
        self._batch_normals(body, obstacles)
        keys, h, A_c, A_rot, A_n = [], [], [], [], []
        for pname in sorted(body.parts):
            part = body.world(pname)
            for okey in sorted(obstacles, key=repr):
                obs = obstacles[okey]
                n = self._normals[(pname, okey)]
                keys.append((pname, okey))
                h.append(barrier(n, part, obs))
                A_c.append(grad_center(n, part, obs))
                A_rot.append(grad_rotation(n, part, obs))
                A_n.append(grad_normal(n, part, obs) if self.p.normals == "relaxed" else None)
            for hkey in sorted(half_spaces, key=repr):
                hs = half_spaces[hkey]
                denom = float(np.sqrt(max(hs.a @ part.Q @ hs.a, 0.0)))
                keys.append((pname, hkey))
                h.append(hs.barrier(part))
                A_c.append(hs.a.copy())
                A_rot.append(np.cross(hs.a, part.Q @ hs.a) / denom if denom > EPS_DENOM else np.zeros(3))
                A_n.append(None)
        return keys, np.asarray(h, np.float64), A_c, A_rot, A_n

    def _solve(self, body, keys, h_vals, A_c, A_rot, A_n, nominal):
        """Assemble and solve the QP. ``nominal`` is the cost's target in the body's own variables.

        The barrier / virtual-normal / slack machinery is identical for both parameterisations;
        only the projection of each row and the tracking cost differ, and both come from ``body``.
        """
        m = len(keys)
        n_pose = body.n_var
        n_free = sum(1 for g in A_n if g is not None)
        idx_n = {i: n_pose + 3 * k
                 for k, i in enumerate(i for i, g in enumerate(A_n) if g is not None)}
        use_slack = self.p.slack_weight is not None
        base = n_pose + 3 * n_free
        n_var = base + (m if use_slack else 0)

        A_rows = np.zeros((m, n_var))
        for i, (pname, _) in enumerate(keys):
            A_rows[i, 0:n_pose] = body.project(pname, A_c[i], A_rot[i])
            if A_n[i] is not None:
                A_rows[i, idx_n[i] : idx_n[i] + 3] = A_n[i]
            if use_slack:
                A_rows[i, base + i] = 1.0  # ... >= -gamma h - s  <=>  ... + s >= -gamma h

        w, target = nominal
        p_diag = np.concatenate([2.0 * w, np.zeros(3 * n_free)])
        q = np.concatenate([-2.0 * w * target, np.zeros(3 * n_free)])
        if use_slack:
            p_diag = np.concatenate([p_diag, np.full(m, 2.0 * self.p.slack_weight)])
            q = np.concatenate([q, np.zeros(m)])
        P = sp.diags(p_diag).tocsc()

        rows = [A_rows]
        lower = [-self.p.gamma_h * h_vals]
        upper = [np.full(m, np.inf)]

        if n_free:  # Eq. (12) box on the virtual normals
            box = np.zeros((3 * n_free, n_var))
            box[:, n_pose : n_pose + 3 * n_free] = np.eye(3 * n_free)
            rows.append(box)
            lower.append(np.full(3 * n_free, -self.p.eps_normal))
            upper.append(np.full(3 * n_free, self.p.eps_normal))
        if use_slack:  # s >= 0
            nn = np.zeros((m, n_var))
            nn[:, base:] = np.eye(m)
            rows.append(nn)
            lower.append(np.zeros(m))
            upper.append(np.full(m, np.inf))
        for sl, bound in self._action_limits(body):
            if bound is None:
                continue
            k = sl.stop - sl.start
            lim = np.zeros((k, n_var))
            lim[:, sl] = np.eye(k)
            rows.append(lim)
            lower.append(-np.abs(np.broadcast_to(bound, (k,))))
            upper.append(np.abs(np.broadcast_to(bound, (k,))))

        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=sp.csc_matrix(np.vstack(rows)),
                     l=np.concatenate(lower), u=np.concatenate(upper),
                     verbose=False, polishing=True,
                     eps_abs=self.p.solver_eps_abs, eps_rel=self.p.solver_eps_rel,
                     max_iter=self.p.solver_max_iter)
        res = solver.solve()
        status = str(res.info.status)
        ok = res.x is not None and np.all(np.isfinite(res.x)) and "solved" in status.lower()
        x = np.asarray(res.x, np.float64) if ok else None
        slack = x[base:] if (ok and use_slack) else None
        return ok, status, x, idx_n, slack

    def _action_limits(self, body):
        """Box bounds on the pose block, in whatever variables the body uses."""
        if body.n_var == 6:
            return [(slice(0, 3), self.p.max_delta_pos), (slice(3, 6), self.p.max_delta_rot)]
        return [(slice(0, body.n_var), self.p.max_delta_q)]

    def __call__(
        self,
        robot,
        obstacles: dict,
        delta_c_nom=None,
        delta_theta_nom=None,
        half_spaces: dict | None = None,
        delta_q_nom=None,
    ) -> FilterResult:
        """Solve Eq. (12) for one control step.

        ``obstacles`` is keyed by a stable object id so each keeps its own separating normal; the
        target must already have been removed by the attention stage (Eq. 4).

        ``robot`` selects the parameterisation. An ``Ellipsoid`` or ``RobotBody`` solves for
        ``(delta_c, delta_theta)`` as the paper does. An ``ArticulatedBody`` solves for
        ``delta_q`` and wants ``delta_q_nom`` -- the joint step the policy asked for.
        """
        if isinstance(robot, (RobotBody, ArticulatedBody)):
            body = robot
        else:
            body = RobotBody.single(robot)
        joint_mode = isinstance(body, ArticulatedBody)
        half_spaces = half_spaces or {}

        if joint_mode:
            if delta_q_nom is None:
                raise ValueError("ArticulatedBody needs delta_q_nom")
            target = np.asarray(delta_q_nom, np.float64).reshape(-1)
            if target.size != body.nq:
                raise ValueError(f"delta_q_nom has {target.size} entries, body has nq={body.nq}")
            w = np.broadcast_to(np.asarray(self.p.joint_weights if self.p.joint_weights is not None
                                           else 1.0, np.float64), (body.nq,)).astype(np.float64)
            nominal = (w, target)
            identity = (target.copy(), None)
        else:
            dc = np.asarray(delta_c_nom, np.float64).reshape(3)
            dth = np.asarray(delta_theta_nom, np.float64).reshape(3)
            nominal = (np.concatenate([np.ones(3), np.full(3, self.p.W)]), np.concatenate([dc, dth]))
            identity = (dc, dth)

        if not obstacles and not half_spaces:  # nothing to avoid -> the filter must be the identity
            dq = identity[0].copy() if joint_mode else None
            dc, dth = ((body.links[body.eef].Jv @ dq, body.links[body.eef].Jw @ dq) if joint_mode
                       else (identity[0].copy(), identity[1].copy()))
            return FilterResult(dc, dth, True, False, np.zeros(0), "no_obstacles", None, (), dq)

        keys, h_vals, A_c, A_rot, A_n = self._rows(body, obstacles, half_spaces)
        ok, status, x, idx_n, slack = self._solve(body, keys, h_vals, A_c, A_rot, A_n, nominal)
        if not ok:
            # Appendix 7.1: "If the QP is infeasible, we fall back to an emergency stop with zero
            # translational and rotational deltas." Reachable only without slack -- see D3.
            zq = np.zeros(body.nq) if joint_mode else None
            return FilterResult(np.zeros(3), np.zeros(3), False, True, h_vals, status,
                                None, tuple(keys), zq)

        if self.p.normals == "relaxed":
            for i, key in enumerate(keys):
                if A_n[i] is None:
                    continue
                n_new = self._normals[key] + x[idx_n[i] : idx_n[i] + 3]
                nrm = float(np.linalg.norm(n_new))
                if nrm > EPS_DENOM:  # renormalize, per Appendix 7.1
                    self._normals[key] = n_new / nrm
        elif self.p.normals == "fixedpoint":
            # Recompute the normals at the pose this solve lands on, then solve once more. The
            # first solve is already feasible, so this only tightens.
            moved = body.moved(x[0 : body.n_var]) if joint_mode else body.moved(x[0:3], x[3:6])
            keys, h_vals, A_c, A_rot, A_n = self._rows(moved, obstacles, half_spaces)
            ok2, status2, x2, idx_n, slack2 = self._solve(body, keys, h_vals, A_c, A_rot, A_n, nominal)
            if ok2:
                x, status, slack = x2, status2, slack2

        dc, dth, dq = body.apply(x[0 : body.n_var])
        stop = bool(slack is not None and slack.size and float(slack.max()) > self.p.slack_stop)
        if stop:
            return FilterResult(np.zeros(3), np.zeros(3), True, True, h_vals,
                                f"slack_stop({float(slack.max()):.4f})", slack, tuple(keys),
                                np.zeros(body.nq) if joint_mode else None)
        return FilterResult(dc, dth, True, False, h_vals, status, slack, tuple(keys), dq)


def effective_margin(n, robot: Ellipsoid, obstacle: Ellipsoid, eps: float) -> float:
    """How much the epsilon box can relax the Eq. (8) row, in metres.

    The delta_n freedom adds at most ``eps * ||grad_n h||_1`` to the constraint's left-hand side.
    When that exceeds the ``gamma_h * |h|`` the barrier demands, the row can be satisfied by tilting
    the plane alone and the filter stops moving the robot -- the mechanism behind the inert filter
    in docs/11-p2b-results.md §4, and the reason `normals='fixed'` exists.
    """
    return float(eps) * float(np.abs(grad_normal(n, robot, obstacle)).sum())
