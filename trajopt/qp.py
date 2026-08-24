"""The quadratic subproblem, and which solver runs it.

Four backends sit behind one call because the right choice is a measurement, not a preference, and
because the measurement should be repeatable on the machine that will run the robot. `--benchmark`
times every available backend on a problem the same size as the real one and prints the table the
README quotes.

**Why not an interior-point NLP solver for the whole thing.** IPOPT is already wired into this
repository (`tests/ag3s/test_to_contract.py`) and it converges, so the objection is not that it fails
— it is that it cannot be given a deadline.

* Interior-point iterates follow a central path parameterised by a barrier weight. A good
  collision-free trajectory hugs its safety margin, which is exactly where the previous solution sits
  *close to the constraint boundary* — and that is a poor starting point for the next barrier
  subproblem, so the solver re-anneals rather than resuming. Active-set and operator-splitting
  methods have no such pathology; warm-starting them is just handing them the previous answer.
* The iteration count is not bounded. IPOPT runs to a tolerance and may enter a restoration phase.
  Real-time control needs a fixed budget per cycle, which is what the real-time iteration scheme
  provides: take one or a few Newton-type steps and let the feedback loop finish the job.

**What the measurement said.** On the RB-Y1 subproblem — 1,884 variables, 4,116 rows, 16,980
nonzeros, H=32 with 24 rows per step:

| backend | first solve | re-solve | verdict                                     |
|---------|-------------|----------|---------------------------------------------|
| `osqp`  | 96.7 ms     | 2.7 ms   | **default**                                  |
| `highs` | 337.7 ms    | 332.8 ms | correct, 120x slower, no warm-start benefit  |
| `proxqp`| 5,723 ms    | 5,739 ms | correct and unusable here                    |
| `qrqp`  | > 200 s     | —        | excluded                                     |
| `ipqp`  | —           | —        | **aborted the interpreter** (heap corruption)|

The last row is why `ipqp` is not in `QpConfig.SOLVERS`. A backend that can take the control loop
down with it is not a candidate whatever its timing would have been.

The re-solve column is the optimistic case: the *same* problem handed back, where the warm start
lands on the answer. Inside the SQP loop each iteration linearizes somewhere new, and the honest
figure is the per-chunk total in the README (28.4 ms), not this column.

**Why qpOASES is absent from the list.** Not preference — structure. It is a dense online active-set
method whose cost grows superlinearly with the constraint count, and this problem has thousands of
rows. It would be the right answer for a QP an order of magnitude smaller.

**Why not acados, HPIPM or cuRobo.** They are not installed, and the repository's dependency policy
is to use what is present. acados needs C code generation and a build toolchain; cuRobo needs CUDA.
If the measured budget is missed after the reductions in `linearize`, that is the moment to revisit
them — with a number to justify the dependency.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

from benchmark.trajopt.config import QpConfig
from benchmark.trajopt.problem import QpProblem

#: Bounds this large are treated as infinite. OSQP wants a finite-ish number; CasADi does not care.
_INF = 1e20


@dataclasses.dataclass(frozen=True)
class QpSolution:
    """One QP solve. `x` is meaningful only when `success` is True."""

    x: np.ndarray
    y: Optional[np.ndarray]  # dual variables, for warm-starting the next solve
    success: bool
    status: str
    iterations: int
    solve_time_ms: float

    @property
    def n_var(self) -> int:
        return int(self.x.shape[0])


class QpSolver:
    """A QP backend, kept alive across solves so it can reuse whatever it is able to reuse.

    Instances are stateful on purpose. OSQP factorizes its KKT matrix once and reuses it while the
    sparsity pattern holds; a fresh object per iteration would throw that away and turn the single
    biggest advantage of the method into overhead.
    """

    def __init__(self, config: Optional[QpConfig] = None):
        self.config = config or QpConfig()
        self._backend = None
        self._signature: Optional[tuple] = None
        self._last_x: Optional[np.ndarray] = None
        self._last_y: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Drop the cached factorization. Call between episodes, not between iterations."""
        self._backend = None
        self._signature = None
        self._last_x = None
        self._last_y = None

    def solve(self, problem: QpProblem, *, warm_start: bool = True) -> QpSolution:
        started = time.perf_counter()
        name = self.config.solver
        if name == "osqp":
            solution = self._solve_osqp(problem, warm_start)
        else:
            solution = self._solve_casadi(problem, name, warm_start)
        elapsed = (time.perf_counter() - started) * 1000.0
        if solution.success:
            self._last_x, self._last_y = solution.x, solution.y
        return dataclasses.replace(solution, solve_time_ms=elapsed)

    # --- OSQP -----------------------------------------------------------------------------
    def _solve_osqp(self, problem: QpProblem, warm_start: bool) -> QpSolution:
        import osqp

        signature = _signature(problem)
        l = np.clip(problem.l, -_INF, _INF)
        u = np.clip(problem.u, -_INF, _INF)

        if self._backend is None or self._signature != signature:
            self._backend = osqp.OSQP()
            self._backend.setup(
                P=sp.triu(problem.P, format="csc"),
                q=problem.q,
                A=problem.A,
                l=l,
                u=u,
                verbose=self.config.verbose,
                eps_abs=self.config.eps_abs,
                eps_rel=self.config.eps_rel,
                max_iter=self.config.max_iter,
                polish=self.config.polish,
                adaptive_rho=self.config.adaptive_rho,
                warm_starting=warm_start,
            )
            self._signature = signature
        else:
            # Same pattern: only the numbers move. This is the path the whole fixed-sparsity design
            # exists to reach — no re-factorization, no re-setup.
            self._backend.update(
                Px=sp.triu(problem.P, format="csc").data,
                Ax=problem.A.data,
                q=problem.q,
                l=l,
                u=u,
            )
            if warm_start and self._last_x is not None and self._last_x.shape[0] == problem.n_var:
                self._backend.warm_start(x=self._last_x, y=self._last_y)

        result = self._backend.solve()
        status = str(getattr(result.info, "status", "unknown"))
        success = status in ("solved", "solved inaccurate")
        return QpSolution(
            x=np.asarray(result.x, np.float64) if success else np.zeros(problem.n_var),
            y=np.asarray(result.y, np.float64) if success else None,
            # `bool(...)` rather than the numpy scalar `np.all` returns: a `np.bool_` leaking into
            # the public result makes `isinstance(x, bool)` false for a caller who reasonably assumed
            # a flag is a flag.
            success=bool(success and np.all(np.isfinite(result.x))),
            status=status,
            iterations=int(getattr(result.info, "iter", 0)),
            solve_time_ms=0.0,
        )

    # --- CasADi conic backends -------------------------------------------------------------
    def _solve_casadi(self, problem: QpProblem, name: str, warm_start: bool) -> QpSolution:
        import casadi as ca

        signature = _signature(problem)
        P = _to_dm(problem.P)
        A = _to_dm(problem.A)
        if self._backend is None or self._signature != signature:
            options: dict[str, Any] = {"print_time": False}
            if not self.config.verbose:
                options.update(_quiet_options(name))
            self._backend = ca.conic(
                f"trajopt_{name}", name, {"h": P.sparsity(), "a": A.sparsity()}, options
            )
            self._signature = signature

        kwargs: dict[str, Any] = {
            "h": P,
            "g": problem.q,
            "a": A,
            "lba": np.clip(problem.l, -np.inf, np.inf),
            "uba": np.clip(problem.u, -np.inf, np.inf),
        }
        if warm_start and self._last_x is not None and self._last_x.shape[0] == problem.n_var:
            kwargs["x0"] = self._last_x
        try:
            result = self._backend(**kwargs)
        except RuntimeError as exc:
            return QpSolution(np.zeros(problem.n_var), None, False, f"error: {exc}", 0, 0.0)

        x = np.asarray(result["x"], np.float64).reshape(-1)
        stats = self._backend.stats()
        success = bool(stats.get("success", True)) and bool(np.all(np.isfinite(x)))
        return QpSolution(
            x=x,
            y=np.asarray(result.get("lam_a", np.zeros(0)), np.float64).reshape(-1),
            success=success,
            status=str(stats.get("return_status", "solved" if success else "failed")),
            iterations=int(stats.get("iter_count", 0)),
            solve_time_ms=0.0,
        )


def _signature(problem: QpProblem) -> tuple:
    """What must stay the same for a solver to reuse its setup: shapes and sparsity, never values."""
    return (
        problem.P.shape,
        problem.A.shape,
        problem.P.nnz,
        problem.A.nnz,
        problem.P.indptr.tobytes(),
        problem.P.indices.tobytes(),
        problem.A.indptr.tobytes(),
        problem.A.indices.tobytes(),
    )


def _to_dm(matrix: sp.spmatrix):
    import casadi as ca

    coo = matrix.tocoo()
    return ca.DM(
        ca.Sparsity.triplet(coo.shape[0], coo.shape[1], coo.row.tolist(), coo.col.tolist()),
        coo.data.tolist(),
    )


def _quiet_options(name: str) -> dict[str, Any]:
    return {
        "proxqp": {"proxqp": {"verbose": False}},
        "qrqp": {"print_iter": False, "print_header": False},
        "ipqp": {"print_iter": False, "print_header": False},
        "highs": {"highs": {"output_flag": False}},
    }.get(name, {})


def available_solvers() -> list[str]:
    """Which backends this build can actually construct. Probed, not assumed."""
    import casadi as ca

    found: list[str] = []
    try:
        import osqp  # noqa: F401

        found.append("osqp")
    except ImportError:
        pass
    probe_h = ca.DM(np.eye(2)).sparsity()
    probe_a = ca.DM(np.ones((1, 2))).sparsity()
    for name in ("proxqp", "highs"):
        try:
            # Several backends announce themselves on construction regardless of their options, so
            # the probe is muted at the file-descriptor level — CasADi prints from C++, where a
            # Python `redirect_stdout` does not reach.
            with _muted():
                ca.conic("probe", name, {"h": probe_h, "a": probe_a},
                         {"print_time": False, **_quiet_options(name)})
            found.append(name)
        except Exception:
            continue
    return found


@contextlib.contextmanager
def _muted():
    """Silence C++-level stdout for the duration of the block."""
    import os

    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _bench_one(name: str, horizon: int, rows_per_step: int, repeats: int) -> None:
    """Time one backend and print a single line. Runs as its own process — see `main`."""
    import dataclasses as dc
    import time

    from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS, load_rby1
    from benchmark.trajopt.config import TrajOptConfig
    from benchmark.trajopt.limits import build_limits
    from benchmark.trajopt.linearize import append_collision_rows
    from benchmark.trajopt.problem import build_problem
    from benchmark.trajopt.sqp import TrajectoryOptimizer
    from benchmark.trajopt.types import ChunkLayout
    from tests.trajopt.fixtures import Q_HOME, near_miss_scene, sweep_reference

    config = TrajOptConfig.from_dict({
        "horizon": {"horizon": horizon, "execution_length": 8, "plan_horizon": None},
        "reduction": {"rows_per_step": rows_per_step},
    })
    robot = load_rby1()
    layout = ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)
    limits = build_limits(robot, layout, dt=config.horizon.dt, config=config.limits)
    optimizer = TrajectoryOptimizer(robot, layout, limits, config)
    reference = sweep_reference(layout, horizon)
    scene = near_miss_scene(optimizer.linearizer, reference, Q_HOME)
    rows = optimizer.linearizer.linearize(reference, Q_HOME, scene, config)
    problem = append_collision_rows(
        build_problem(reference, reference, limits, config,
                      q_now=Q_HOME[layout.q_indices], n_slack=optimizer.block.n_rows),
        rows, reference, optimizer.block, config.reduction.linearization_backoff,
    )
    solver = QpSolver(dc.replace(QpConfig(), solver=name))
    started = time.perf_counter()
    first = solver.solve(problem)
    elapsed = (time.perf_counter() - started) * 1000.0
    if not first.success:
        print(f"BENCH {name} - - - {first.status}")
        return
    warm = [solver.solve(problem).solve_time_ms for _ in range(repeats)]
    checksum = float(np.abs(problem.split(first.x)[0]).sum())
    print(f"BENCH {name} {elapsed:.1f} {float(np.median(warm)):.1f} {first.iterations} "
          f"{first.status} {checksum:.6f}")


def main(argv: Optional[list[str]] = None) -> int:
    """Time every available backend on a problem the size of the real one.

    The point is that the default is a measurement taken on the machine that will run the robot, not
    a number copied out of a paper.

    Each backend runs in **its own process**. Two of the CasADi conic plugins abort this problem at
    the C level (`double free or corruption`), and a benchmark that a candidate can crash is a
    benchmark that only ever reports the survivors. A crash is a result, and it is reported as one.
    """
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(description="QP backend benchmark for the RB-Y1 subproblem")
    parser.add_argument("--benchmark", action="store_true", help="run the comparison")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--rows-per-step", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per backend")
    parser.add_argument("--only", default=None, help=argparse.SUPPRESS)  # used by the child process
    args = parser.parse_args(argv)

    if args.only:
        _bench_one(args.only, args.horizon, args.rows_per_step, args.repeats)
        return 0
    if not args.benchmark:
        parser.print_help()
        return 0

    print(f"{'solver':>8s} {'first(ms)':>10s} {'warm(ms)':>9s} {'iters':>7s}  status")
    reference = None
    for name in available_solvers():
        command = [
            sys.executable, "-m", "benchmark.trajopt.qp", "--only", name,
            "--horizon", str(args.horizon), "--rows-per-step", str(args.rows_per_step),
            "--repeats", str(args.repeats),
        ]
        try:
            done = subprocess.run(
                command, capture_output=True, text=True, timeout=args.timeout, check=False
            )
        except subprocess.TimeoutExpired:
            print(f"{name:>8s} {'-':>10s} {'-':>9s} {'-':>7s}  excluded: over {args.timeout:.0f} s")
            continue
        line = next(
            (l for l in done.stdout.splitlines() if l.startswith("BENCH ")), None
        )
        if line is None:
            reason = "crashed" if done.returncode != 0 else "no result"
            print(f"{name:>8s} {'-':>10s} {'-':>9s} {'-':>7s}  excluded: {reason} "
                  f"(exit {done.returncode})")
            continue
        _, _, first, warm, iterations, status, *rest = line.split()
        note = ""
        if rest:
            # Sum of |Q| over every joint and step. Reported as a difference rather than judged: at a
            # solver tolerance of 1e-3 across ~400 entries a spread of this order is expected, and
            # labelling it a disagreement would be reading precision into the third decimal of a
            # number the solvers were never asked to agree on. A large gap here means something real.
            checksum = float(rest[0])
            if reference is None:
                reference = checksum
            else:
                per_entry = abs(checksum - reference) / max(args.horizon * 12, 1)
                note = f"  checksum d={per_entry:.1e}/entry"
        print(f"{name:>8s} {first:>10s} {warm:>9s} {iterations:>7s}  {status}{note}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["QpSolution", "QpSolver", "available_solvers", "main"]
