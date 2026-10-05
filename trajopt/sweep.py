"""T43 Q (2) — collision rows **between** waypoints: the interpolated path, or the servo's own path.

## The gap this closes

`linearize` checks the plan at its waypoints `Q[:, 0..H-1]` and nowhere else. Two things happen in
between that the TO never saw (T41 b):

* the hand **sweeps** from one waypoint to the next — up to ~150 mm per control row at the anchor
  bound — so a 24 mm bar can sit between two clear waypoints;
* the robot is **not at the waypoints**. A position servo lags about one control row behind its
  command (`servo.py`), so the executed path runs behind and between the commands.

## What is checked

A set of intermediate configurations per control row `k`, each an **affine** function of the plan:

| mode | configurations in row `k` | `X = offset + Σ_i W_i ⊙ Q[:, i]` touches |
|---|---|---|
| `interp` (`reduction.sweep_check`, no servo) | joint-space interpolation between `Q[:, k-1]` (`q_now` for `k = 0`) and `Q[:, k]`, interior points only | `Q[:, k-1]`, `Q[:, k]` |
| `servo` (`limits.servo_model`) | the servo's predicted state at sub-steps of row `k` — the row end always, interior sub-steps when `sweep_check` is on | `Q[:, 0..k]` |

**How many.** With `sweep_check` on, each row's path is first measured on a grid of 8 points
(positions-only FK): `L_k = Σ_cells max_query |Δp|`. By the triangle inequality no query sphere moves
further than the `L` gained between two configurations, so row `k` is cut at equal steps of `L_k / m_k`,
`m_k = ceil(L_k / sweep_max_m)`, capped at `sweep_max_samples`. `sweep_max_m` defaults to 12 mm — half the
24 mm hurdle bar. Interpolation is cut in `τ`, the servo path in sub-steps (rounded to MuJoCo's 2 ms
steps). The sweep actually achieved between consecutive checked configurations is measured by FK on every
final check and recorded (`metrics["path_check"]["max_sweep_mm"]`) — measured, not assumed.

**Which rows.** Every row `linearize` would consider (candidate, plane, ESDF), at every intermediate
configuration. The QP gets a fixed budget per control row (`reduction.sweep_rows_per_step`, tightest
first across all of that row's configurations), so its sparsity is fixed by the configuration alone
(row → control row `k` → blocks `Q[:, k-1..k]` or `Q[:, 0..k]`) and never by the scene. The full
check (`violation`) evaluates every row of every configuration, as `full_violation` does for the
waypoints.

**Not the grasp target.** Robot-sphere ESDF rows whose nearest surface is the object being
manipulated (E1's `|d − d_object| ≤ voxel`, only where the distance includes that object) are left
out: reaching the object is the task, and the waypoint rows keep E1's margin on it. Everything else
stays — obstacles, the table, the destination, and the held object's own query points.

**The floor — not deeper than now.** An intermediate row demands ``h(X) ≥ min(0, h(q_now))``: the
same clearance as a waypoint row, **except** where the robot already starts inside it, where it may
not get any deeper. Without it, a robot standing 9.7 mm from the bar under a 10 mm obstacle margin
(T41 b, C1807 t=216) could never pass any path check — every path starts at `q_now` — and the escape
that physically worked there would be refused, the failure `--rest-start` had. Waypoint rows keep
their zero floor (unchanged).
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

from benchmark.trajopt.config import TrajOptConfig

#: configurations are padded to a multiple of this before FK, so the number of distinct CasADi maps
#: stays small while `m_k` adapts per chunk.
_BUCKET = 8
#: grid points per control row on which a row's path length is measured (`PathChecker.configs`).
_GRID = 8


@dataclasses.dataclass
class PathConfigs:
    full: np.ndarray            # (nq_model, n)
    params: Optional[np.ndarray]  # (n_params, n) or None
    segment: np.ndarray         # (n,) control row k
    where: np.ndarray           # (n,) τ in (0, 1) (interp) or sub-step index (servo)
    W: np.ndarray               # (nq_opt, n, H)  ∂X/∂Q
    pieces: np.ndarray          # (H,) m_k
    path_m: np.ndarray         # (H,) path length per row on the grid (0 without sweep_check)
    cap_bound: tuple[int, ...]
    #: q_now and the row ends from the grid FK (`None` without sweep_check) — the interpolation
    #: chain needs the waypoints themselves.
    ends_centres: Optional[np.ndarray]    # (H + 1, Q, 3)
    x_end: Optional[np.ndarray] = None  # (nq_opt, H) servo row-end prediction

    @property
    def n(self) -> int:
        return int(self.segment.shape[0])


@dataclasses.dataclass
class PathRows:
    value: np.ndarray      # (H, R) clearance above the floor; +inf unused
    gradient: np.ndarray   # (H, R, H, nq_opt) with respect to Q[:, i]
    used: np.ndarray       # (H, R) bool
    config: np.ndarray     # (H, R) index into PathConfigs
    query: np.ndarray      # (H, R)
    n_configs: int


class PathBlock:
    """Fixed sparsity of the path rows: row → control row `k` → its blocks, plus its own slack."""

    #: how the path rows share slack: `row` (one per row, as the waypoint rows — the exact L1 penalty
    #: `problem.build_problem` uses), `step` (one per control row) or `one` (one for every path row).
    #: Measured on T40 C1807 t=216 and T41 a C1807 t=272 (`outputs/impl/T43Q/p/diag_slack.py`): without
    #: the stay start (`sqp._stay_start`) none of the three ends feasible at t=216; with it `row` and
    #: `one` both do. `row` keeps the waypoint rows' convention.
    SLACK = "row"

    def __init__(self, horizon: int, rows_per_step: int, nq_opt: int, mode: str):
        self.horizon, self.rows_per_step, self.nq_opt = int(horizon), int(rows_per_step), int(nq_opt)
        self.mode = mode
        self.n_rows = self.horizon * self.rows_per_step
        self.blocks = [self.pattern(k) for k in range(self.horizon)]
        self.n_slack = {"row": self.n_rows, "step": self.horizon, "one": 1}[self.SLACK]

    def slack_of(self, k: int, r: int) -> int:
        return {"row": k * self.rows_per_step + r, "step": k, "one": 0}[self.SLACK]

    def pattern(self, k: int) -> list[int]:
        if self.mode == "servo":
            return list(range(0, k + 1))
        return [i for i in (k - 1, k) if i >= 0]

    def matrix(self, rows: PathRows, n_col: int, slack_offset: int) -> sp.csr_matrix:
        """The rows with this iteration's gradients; the pattern depends only on the shape."""
        H, R, nq = self.horizon, self.rows_per_step, self.nq_opt
        n_q = H * nq
        indptr = [0]
        indices: list[np.ndarray] = []
        data: list[np.ndarray] = []
        for k in range(H):
            cols = np.concatenate([np.arange(i * nq, (i + 1) * nq) for i in self.blocks[k]])
            for r in range(R):
                g = rows.gradient[k, r][self.blocks[k]].reshape(-1)
                indices.append(np.concatenate([cols, [n_q + slack_offset + self.slack_of(k, r)]]))
                data.append(np.concatenate([g if rows.used[k, r] else np.zeros_like(g), [1.0]]))
                indptr.append(indptr[-1] + cols.size + 1)
        A = sp.csr_matrix((np.concatenate(data), np.concatenate(indices).astype(np.int32),
                           np.asarray(indptr, np.int32)), shape=(self.n_rows, int(n_col)))
        return A


class PathChecker:
    """Builds, linearizes and checks the intermediate configurations of one chunk."""

    def __init__(self, linearizer, layout, config: TrajOptConfig, servo=None):
        self.lin = linearizer
        self.layout = layout
        self.config = config
        self.servo = servo
        self.mode = "servo" if servo is not None else "interp"
        red = config.reduction
        self.sweep = bool(getattr(red, "sweep_check", False))
        self.max_sweep = float(getattr(red, "sweep_max_m", 0.012))
        self.max_pieces = int(getattr(red, "sweep_max_samples", 16))
        self.rows_per_step = int(getattr(red, "sweep_rows_per_step", 24))
        self.horizon = int(linearizer.horizon)
        self.block = PathBlock(self.horizon, self.rows_per_step, layout.nq_opt, self.mode)
        self._cfg = dataclasses.replace(
            config, reduction=dataclasses.replace(red, rows_per_step=self.rows_per_step,
                                                  temporal_stride=1))
        self._floor: Optional[np.ndarray] = None
        self._state = None
        self._q_now = None
        self.timing = {"configs": 0.0, "fk": 0.0, "rows": 0.0, "check": 0.0}

    def warm(self) -> float:
        """Build every FK map a chunk can ask for (one per bucket size, with and without the
        Jacobian) so no chunk pays a CasADi map construction (up to ~100 ms each, measured).
        Returns the seconds it took. Called once when the optimizer is built."""
        mark = time.perf_counter()
        most = self.horizon * (self.max_pieces if self.sweep else 1)
        for n in range(_BUCKET, max(_BUCKET, int(math.ceil(most / _BUCKET)) * _BUCKET) + 1, _BUCKET):
            self.lin._config_maps(n, True)
            self.lin._config_maps(n, False)
        return time.perf_counter() - mark

    @property
    def n_rows(self) -> int:
        return self.block.n_rows

    @property
    def n_slack(self) -> int:
        return self.block.n_slack

    # ------------------------------------------------------------------------------------------
    def begin(self, q_now: np.ndarray, scene, state=None) -> None:
        """Once per solve: the servo state (servo mode) and the row floors at `q_now`."""
        self._q_now = np.asarray(q_now, np.float64).reshape(-1)
        self._state = state
        self.timing = {"configs": 0.0, "fk": 0.0, "rows": 0.0, "check": 0.0}
        #: per-solve memo, keyed by the trajectory's bytes: the SQP evaluates the same trajectory
        #: more than once (its merit, then its linearization as the next iterate, then the final
        #: check of the returned one). `begin` clears it — the state and `q_now` change per chunk.
        self._memo: dict = {}
        params = self._params()
        centres, _ = self.lin.config_states(self._q_now[:, None],
                                            None if params is None else params[:, :1],
                                            with_jac=False)
        self._centres_now = centres[0]
        h0 = self.lin.flat_clearances(centres, scene)[0]
        self._floor = np.minimum(0.0, np.where(np.isfinite(h0), h0, 0.0))
        self.floor_rows = int(np.sum(self._floor < 0.0))
        self.floor_min_m = float(self._floor.min(initial=0.0))

    def _params(self) -> Optional[np.ndarray]:
        return None if self.lin._p_sym is None else np.asarray(self.lin.param_path(), np.float64)

    # ------------------------------------------------------------------------------------------
    def configs(self, trajectory: np.ndarray) -> PathConfigs:
        key = ("configs", np.ascontiguousarray(trajectory, np.float64).tobytes())
        memo = getattr(self, "_memo", None)
        if memo is not None and key in memo:
            return memo[key]
        out = self._configs(trajectory)
        if memo is not None:
            if len(memo) > 64:
                memo.clear()
            memo[key] = out
        return out

    def _configs(self, trajectory: np.ndarray) -> PathConfigs:
        """The intermediate configurations of `trajectory`, and their affine maps.

        With `sweep_check` each row's path is first measured on a grid of `_GRID` points (FK,
        positions only): `L_k = Σ max_query |Δp|` over the grid cells. By the triangle inequality no
        query sphere moves more than the `L` gained between two configurations, so cutting row `k`
        at equal steps of `L_k / m_k` with `m_k = ceil(L_k / sweep_max_m)` keeps every sphere within
        `sweep_max_m` between consecutive checks (up to the motion inside one grid cell, and the
        whole sub-step rounding in servo mode). The achieved number is measured in `violation`.
        """
        mark = time.perf_counter()
        Q = np.asarray(trajectory, np.float64)
        H, nq = Q.shape[1], Q.shape[0]
        q_opt = self._q_now[self.layout.q_indices]
        params = self._params()

        x_end = X_rows = W_all = X_all = None
        if self.mode == "servo":
            S = self.servo.substeps
            W_all, Fq, Eq = self.servo.affine_path(H, list(range(1, S + 1)))
            x0 = np.stack([q_opt, np.asarray(self._state.v, np.float64)], axis=1)
            offset_all = np.einsum("ncj,nj->nc", Fq, x0) + Eq * np.asarray(self._state.tau)[:, None]
            X_all = offset_all + np.einsum("nci,ni->nc", W_all, Q)        # (nq, H*S)
            X_rows = X_all.reshape(nq, H, S)
            x_end = X_rows[:, :, -1]
            grid = np.unique(np.clip(np.round(np.arange(1, _GRID + 1) * S / _GRID), 1, S)).astype(int)
        else:
            grid = np.arange(1, _GRID + 1) / _GRID                         # τ

        path_len = np.zeros(H)
        cum = None
        ends_centres = None
        if self.sweep:
            G = len(grid)
            cols, prm = [], []
            for k in range(H):
                for g in grid:
                    if self.mode == "servo":
                        cols.append(X_rows[:, k, int(g) - 1])
                        prm.append(None if params is None
                                   else self._interp_params(params, k, float(g) / S))
                    else:
                        a = q_opt if k == 0 else Q[:, k - 1]
                        cols.append((1.0 - g) * a + g * Q[:, k])
                        prm.append(None if params is None else self._interp_params(params, k, g))
            P = None if params is None else np.stack(prm, axis=1)
            centres, _ = self._states(self._full(np.stack(cols, axis=1)), P, with_jac=False)
            chain = np.concatenate([self._centres_now[None], centres], axis=0)
            step = np.linalg.norm(np.diff(chain, axis=0), axis=2).max(axis=1).reshape(H, G)
            cum = np.concatenate([np.zeros((H, 1)), np.cumsum(step, axis=1)], axis=1)  # (H, G+1)
            path_len = cum[:, -1]
            ends_centres = np.concatenate([self._centres_now[None], centres[G - 1::G]], axis=0)
            want = np.maximum(1, np.ceil(path_len / self.max_sweep - 1e-9)).astype(int)
        else:
            want = np.ones(H, int)
        pieces = np.minimum(want, self.max_pieces)
        cap = tuple(int(k) for k in np.flatnonzero(want > self.max_pieces))

        seg, where, Ws, fulls, prm = [], [], [], [], []
        for k in range(H):
            m = int(pieces[k])
            # positions along the row, in the row's own parameter (sub-step or τ), at equal L
            if cum is not None and path_len[k] > 0.0:
                knots = np.concatenate([[0.0], grid.astype(float)])
                targets = np.arange(1, m + 1) * path_len[k] / m
                params_at = np.interp(targets, cum[k], knots)
            else:
                params_at = np.arange(1, m + 1) / m * (self.servo.substeps if self.mode == "servo"
                                                       else 1.0)
            if self.mode == "servo":
                S = self.servo.substeps
                subs: list[int] = []
                for value in params_at:
                    sub = int(min(max(int(round(value)), 1), S))
                    if not subs or sub > subs[-1]:
                        subs.append(sub)
                if subs[-1] != S:
                    subs.append(S)
                for sub in subs:
                    c = k * S + sub - 1
                    seg.append(k)
                    where.append(float(sub))
                    Ws.append(W_all[:, c, :])
                    fulls.append(X_all[:, c])
                    # the fingers move through the row too (their own servo); without this the
                    # finger spheres jump at the row boundary and no sub-step can split the jump
                    prm.append(None if params is None else self._interp_params(params, k, sub / S))
            else:
                a = q_opt if k == 0 else Q[:, k - 1]
                for tau in params_at[:-1]:                                 # interior only
                    tau = float(tau)
                    if not 0.0 < tau < 1.0:
                        continue
                    w = np.zeros((nq, H))
                    if k > 0:
                        w[:, k - 1] = 1.0 - tau
                    w[:, k] = tau
                    seg.append(k)
                    where.append(tau)
                    Ws.append(w)
                    fulls.append((1.0 - tau) * a + tau * Q[:, k])
                    prm.append(None if params is None else self._interp_params(params, k, tau))
        n = len(seg)
        W = np.stack(Ws, axis=1) if n else np.zeros((nq, 0, H))
        full = self._full(np.stack(fulls, axis=1)) if n else np.zeros((len(self._q_now), 0))
        P = None if params is None or not n else np.stack(prm, axis=1)
        self.timing["configs"] += (time.perf_counter() - mark) * 1000.0
        return PathConfigs(full=full, params=P, segment=np.asarray(seg, int),
                           where=np.asarray(where, float), W=W, pieces=pieces, path_m=path_len,
                           cap_bound=cap, ends_centres=ends_centres, x_end=x_end)

    @staticmethod
    def _interp_params(params: np.ndarray, k: int, tau: float) -> np.ndarray:
        """Finger joints between rows `k-1` and `k` (row 0 starts from the measured opening)."""
        p0 = params[:, 0] if k == 0 else params[:, k - 1]
        return (1.0 - tau) * p0 + tau * params[:, k]

    def _full(self, opt_cols: np.ndarray) -> np.ndarray:
        full = np.tile(self._q_now[:, None], (1, opt_cols.shape[1]))
        full[self.layout.q_indices, :] = opt_cols
        return full

    def _states(self, full, params, *, with_jac: bool):
        """FK at `full` padded to a bucket; results trimmed back."""
        mark = time.perf_counter()
        n = full.shape[1]
        size = max(_BUCKET, int(math.ceil(n / _BUCKET)) * _BUCKET)
        if size != n:
            pad = size - n
            full = np.concatenate([full, np.repeat(full[:, -1:], pad, axis=1)], axis=1)
            if params is not None:
                params = np.concatenate([params, np.repeat(params[:, -1:], pad, axis=1)], axis=1)
        centres, jac = self.lin.config_states(full, params, with_jac=with_jac)
        self.timing["fk"] += (time.perf_counter() - mark) * 1000.0
        return centres[:n], (None if jac is None else jac[:n])

    # ------------------------------------------------------------------------------------------
    def linearize(self, trajectory: np.ndarray, scene) -> PathRows:
        cfgs = self.configs(trajectory)
        H, R, nq = self.horizon, self.rows_per_step, self.layout.nq_opt
        value = np.full((H, R), np.inf)
        gradient = np.zeros((H, R, H, nq))
        used = np.zeros((H, R), bool)
        config = np.zeros((H, R), int)
        query = np.zeros((H, R), int)
        if not cfgs.n:
            return PathRows(value, gradient, used, config, query, 0)
        centres, jac = self._states(cfgs.full, cfgs.params, with_jac=True)
        mark = time.perf_counter()
        saved = dict(self.lin.last_target_free_tiers)
        floor = self._floor[None, :].repeat(cfgs.n, axis=0)
        target = self._target_rows(centres, scene)
        if target is not None:
            floor = np.where(target, -np.inf, floor)                     # never selected
        rows = self.lin.linearize(np.zeros((nq, cfgs.n)), self._q_now, scene, self._cfg,
                                  states=(centres, jac), row_floor=floor)
        self.lin.last_target_free_tiers = saved
        for k in range(H):
            mine = np.flatnonzero(cfgs.segment == k)
            if not mine.size:
                continue
            vals = rows.value[mine].reshape(-1)
            ok = rows.used[mine].reshape(-1)
            order = np.argsort(np.where(ok, vals, np.inf), kind="stable")[:R]
            for r, flat in enumerate(order):
                if not ok[flat]:
                    break
                c = int(mine[flat // rows.value.shape[1]])
                j = int(flat % rows.value.shape[1])
                g = rows.gradient[c, j]                       # (nq,) on X_c
                value[k, r] = vals[flat]
                gradient[k, r] = (g[None, :] * cfgs.W[:, c, :].T)  # (H, nq)
                used[k, r] = True
                config[k, r] = c
                query[k, r] = int(rows.sphere[c, j])
        self.timing["rows"] += (time.perf_counter() - mark) * 1000.0
        return PathRows(value, gradient, used, config, query, cfgs.n)

    def violation(self, trajectory: np.ndarray, scene, *, record: bool = False):
        """`(worst clearance above floor over every row of every configuration, record | None)`."""
        cfgs = self.configs(trajectory)
        if not cfgs.n:
            return np.inf, (self._record(cfgs, np.inf, None, None) if record else None)
        key = ("centres", np.ascontiguousarray(trajectory, np.float64).tobytes())
        memo = getattr(self, "_memo", None)
        if memo is not None and key in memo:
            centres = memo[key]
        else:
            centres, _ = self._states(cfgs.full, cfgs.params, with_jac=False)
            if memo is not None:
                memo[key] = centres
        mark = time.perf_counter()
        flat = self.lin.flat_clearances(centres, scene) - self._floor[None, :]
        target = self._target_rows(centres, scene)
        self.target_rows_excluded = 0 if target is None else int(target.sum())
        if target is not None:
            flat = np.where(target, np.inf, flat)
        self.timing["check"] += (time.perf_counter() - mark) * 1000.0
        idx = int(np.argmin(flat))
        worst = float(flat.reshape(-1)[idx])
        if not record:
            return worst, None
        c, col = divmod(idx, flat.shape[1])
        return worst, self._record(cfgs, worst, (c, col, centres), flat)

    def _target_rows(self, centres: np.ndarray, scene) -> Optional[np.ndarray]:
        """`(n, width)` — ESDF rows of **robot spheres** whose nearest field surface is the object
        being manipulated (E1's own test, ``|d − d_object| ≤ voxel``), in rows whose distance
        includes that object (not the target-free layer). `None` when nothing is manipulated.

        Those rows are left out of the path check. Touching the object being grasped is the task;
        the waypoint rows keep E1's phase margin on it as before. The path rows guard against
        everything else — obstacles, the table, the destination — and the held object's own query
        points are always kept (they are what hits a hurdle while carried, T43 task item 5).
        Measured why (B1968 seq 18, `outputs/impl/T43Q/`): the fingers' predicted path entered the
        20 mm APPROACH margin to the apple by 12 mm on a chunk of a run that grasped successfully.
        """
        if scene.esdf is None:
            return None
        n, n_query = centres.shape[0], centres.shape[1]
        pts = centres.reshape(-1, 3)
        d_object = self.lin._manipulated_distance(pts, scene)
        if d_object is None:
            return None
        d = np.asarray(scene.esdf.distance(pts), np.float64).reshape(n, n_query)
        is_target = np.abs(d - d_object.reshape(n, n_query)) <= float(scene.esdf.grid.voxel_size)
        free = self.lin._target_free_rows(scene, n_query)
        if free is not None:
            is_target &= ~free[None, :]
        is_target[:, self.lin.n_spheres:] = False
        width = self._floor.shape[0]
        out = np.zeros((n, width), bool)
        out[:, width - n_query:] = is_target
        return out if out.any() else None

    def _record(self, cfgs: PathConfigs, worst: float, where, flat) -> dict[str, Any]:
        out: dict[str, Any] = {
            "mode": self.mode,
            "sweep_check": self.sweep,
            "n_configs": int(cfgs.n),
            "pieces_per_step": [int(v) for v in cfgs.pieces],
            "path_mm_per_step": ([round(float(v) * 1e3, 2) for v in cfgs.path_m]
                                 if self.sweep else None),
            "sweep_target_mm": round(self.max_sweep * 1e3, 3) if self.sweep else None,
            "cap_bound_steps": list(cfgs.cap_bound),
            "clearance_m": float(worst) if np.isfinite(worst) else None,
            "floor_rows": int(getattr(self, "floor_rows", 0)),
            "target_rows_excluded": int(getattr(self, "target_rows_excluded", 0)),
            "floor_min_m": float(getattr(self, "floor_min_m", 0.0)),
            "worst": None,
            "max_sweep_mm": None,
        }
        if self._state is not None:
            out["servo_state"] = self._state.record()
        if where is None:
            return out
        c, col, centres = where
        n_query = centres.shape[1]
        width = flat.shape[1]
        esdf_start = width - n_query
        q = col - esdf_start if col >= esdf_start else None
        out["worst"] = {
            "step": int(cfgs.segment[c]),
            "at": float(cfgs.where[c]),
            "at_kind": "substep" if self.mode == "servo" else "tau",
            "column": int(col),
            "query": None if q is None else int(q),
            "link": None if q is None else self.lin._query_name(int(q)),
            "floor_m": float(self._floor[col]),
            "point_m": None if q is None else [float(v) for v in centres[c, q]],
        }
        # achieved sweep: q_now, then per row its configurations in time order, row end included
        chain = [self._centres_now]
        for k in range(self.horizon):
            for c2 in np.flatnonzero(cfgs.segment == k):
                chain.append(centres[c2])
            if self.mode != "servo":
                chain.append(cfgs.ends_centres[k + 1])           # the waypoint `Q[:, k]` itself
        chain = np.stack(chain)
        out["max_sweep_mm"] = round(float(np.linalg.norm(np.diff(chain, axis=0), axis=2).max(
            initial=0.0)) * 1e3, 3)
        return out


def append_path_rows(problem, rows: PathRows, iterate: np.ndarray, block: PathBlock,
                     slack_offset: int, backoff: float = 0.0):
    """Write the path rows below whatever `problem` already holds (same form as
    `linearize.append_collision_rows`): ``Σ_i g_i·Q_i + s ≥ Σ_i g_i·Q_i^k − value + backoff``."""
    Qk = np.asarray(iterate, np.float64)
    H, R = rows.value.shape
    at_iterate = np.einsum("krin,ni->kr", rows.gradient, Qk)
    lower = np.where(rows.used, at_iterate - rows.value + float(backoff), 0.0).reshape(-1)
    A_block = block.matrix(rows, problem.A.shape[1], slack_offset)
    A = sp.vstack([problem.A, A_block], format="csc")
    start = problem.A.shape[0]
    row_blocks = dict(problem.row_blocks)
    row_blocks["path"] = (start, start + block.n_rows)
    return dataclasses.replace(problem, A=A, l=np.concatenate([problem.l, lower]),
                               u=np.concatenate([problem.u, np.full(block.n_rows, np.inf)]),
                               row_blocks=row_blocks)


__all__ = ["PathBlock", "PathChecker", "PathConfigs", "PathRows", "append_path_rows"]
