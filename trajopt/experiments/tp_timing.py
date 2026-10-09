"""T43 TP — controlled TO timing: the same recorded chunks re-solved under several settings, per stage.

Reuses the T43 TA reconstruction (`ta_offline.Run`: server flags → config, field, scene, history)
and adds what that harness does not measure: **where the time goes inside one solve**, and the
solve the way the server runs it (one optimizer per setting kept across the run's chunks, so the
OSQP factorization and warm start carry over as they do online — `--mode server`; `--mode reset`
= `ta_offline`'s per-chunk reset).

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work OMP_NUM_THREADS=1 \\
        .venv-openpi-live/bin/python -m benchmark.trajopt.experiments.tp_timing \\
        --index outputs/verify/T43/V8/analysis/analysis.json --key V8/H/V8_ep1807_s18071 \\
        --variants V8 V11AB --out outputs/impl/T43TP/timing/V8_H_V8_ep1807_s18071.json

## Stages (exclusive wall ms per solve, a stack of timers patched onto the classes for this process)

| stage | what |
|---|---|
| `fk` | `CollisionLinearizer.sphere_states` (+ the held probes) — CasADi FK + Jacobian over the plan |
| `esdf` | every field read: `distance` · `gradient` · `label` · `target_free_*` · `answer_tier` · the harness' destination label |
| `clearance` | `_esdf_clearance` minus the field reads: margins, classification (`_obstacle_rows`, `_lookahead_keep`, `_manipulated_distance`) |
| `linearize` | `CollisionLinearizer.linearize` minus the above: row selection, directions × Jacobian |
| `qp_build` | `build_problem` + `append_collision_rows` |
| `osqp` | `QpSolver.solve` (setup or update + solve) |
| `objective` | the cost part of every merit |
| `other` | the rest of `TrajectoryOptimizer.solve` (gate bookkeeping, `_finish` records, …) |

Also per solve: wall (`wall_ms` = server-equivalent: the harness label time removed, `wall_raw_ms` as measured) and CPU (`time.process_time`) totals, the call counts of `fk` / `esdf_clearance`
/ `osqp`, the returned trajectory's SHA-1 (bit-identity checks between settings) and the status.
Pod load: `/proc/loadavg` and the cgroup's `cpu.stat` throttling before and after each run.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np

STAGES = ("fk", "esdf", "clearance", "linearize", "qp_build", "osqp", "objective", "other")


class StageClock:
    """Exclusive wall time per stage: a stage's own time excludes the stages nested inside it."""

    def __init__(self):
        self.stack: list[list] = []
        self.acc: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.enabled = False

    def reset(self):
        self.acc = {}
        self.calls = {}

    def wrap(self, stage: str, fn, count_as: str | None = None):
        clock = self

        @functools.wraps(fn)
        def inner(*a, **k):
            if not clock.enabled:
                return fn(*a, **k)
            t0 = time.perf_counter()
            clock.stack.append([stage, 0.0])
            try:
                return fn(*a, **k)
            finally:
                _, child = clock.stack.pop()
                dt = time.perf_counter() - t0
                clock.acc[stage] = clock.acc.get(stage, 0.0) + (dt - child)
                name = count_as or stage
                clock.calls[name] = clock.calls.get(name, 0) + 1
                if clock.stack:
                    clock.stack[-1][1] += dt

        inner.__tp_wrapped__ = True
        return inner


CLOCK = StageClock()


def _patch(owner, name: str, stage: str, count_as: str | None = None) -> None:
    fn = getattr(owner, name)
    if getattr(fn, "__tp_wrapped__", False):
        return
    setattr(owner, name, CLOCK.wrap(stage, fn, count_as))


def _patch_destination_label(cls) -> None:
    """The harness' destination label (`ta_offline._DestinationLabels.is_label`: nearest surface
    `p − d∇d` in the destination box = one distance + one gradient query) is **not** what the server
    pays: its field answers a label with one lattice pass and a gather of the label grid
    (`CuroboEsdfField.label`). The answer is still the harness' (the solve must be the TA one), but
    its time goes to `harness` (excluded from `wall_adj`), and the server's cost is reproduced on
    the same points and charged to `esdf`: `CuroboEsdfField.label` on layers that carry a label grid
    (`prepare_field`)."""
    fn = cls.is_label
    if getattr(fn, "__tp_wrapped__", False):
        return

    def inner(self, points, name):
        if not CLOCK.enabled:
            return fn(self, points, name)
        t0 = time.perf_counter()
        CLOCK.enabled = False                  # the harness' own field reads are not stages
        try:
            out = fn(self, points, name)
        finally:
            CLOCK.enabled = True
            dt = time.perf_counter() - t0
            CLOCK.acc["harness"] = CLOCK.acc.get("harness", 0.0) + dt
            if CLOCK.stack:
                CLOCK.stack[-1][1] += dt
        # the server's label read (`prepare_field` gave every layer a grid); `label` is an `esdf` stage
        self._field.label(points)
        return out

    inner.__tp_wrapped__ = True
    cls.is_label = inner

    def fused(self, points, name):
        # `sqp.fast_eval`: the server reads distance and label in one pass — that is what is timed;
        # the answer's label is still the harness' (untimed, `harness`).
        if not CLOCK.enabled:
            return self._field.distance(points), fn(self, points, name)
        d, _ = self._field.distance_and_label(points, name)
        t0 = time.perf_counter()
        CLOCK.enabled = False
        try:
            out = fn(self, points, name)
        finally:
            CLOCK.enabled = True
            dt = time.perf_counter() - t0
            CLOCK.acc["harness"] = CLOCK.acc.get("harness", 0.0) + dt
            if CLOCK.stack:
                CLOCK.stack[-1][1] += dt
        return d, out

    cls.distance_and_label = fused


def install() -> None:
    from benchmark.ag3s.fields.curobo_field import CuroboEsdfField
    from benchmark.trajopt import sqp
    from benchmark.trajopt.experiments import ta_offline
    from benchmark.trajopt.linearize import CollisionLinearizer
    from benchmark.trajopt.qp import QpSolver

    for name in ("distance", "gradient", "label", "target_free_distance", "target_free_gradient",
                 "answer_tier", "distance_and_label"):
        if hasattr(CuroboEsdfField, name):
            _patch(CuroboEsdfField, name, "esdf")
    _patch_destination_label(ta_offline._DestinationLabels)
    _patch(CollisionLinearizer, "sphere_states", "fk")
    _patch(CollisionLinearizer, "_esdf_clearance", "clearance", "esdf_clearance")
    if hasattr(CollisionLinearizer, "_esdf_clearance_impl"):
        # computed (not memo hits) — `sqp.fast_eval`
        _patch(CollisionLinearizer, "_esdf_clearance_impl", "clearance", "esdf_clearance_computed")
    _patch(CollisionLinearizer, "linearize", "linearize")
    _patch(sqp, "build_problem", "qp_build")
    if hasattr(sqp, "rebox_problem"):
        _patch(sqp, "rebox_problem", "qp_build")
    _patch(sqp, "append_collision_rows", "qp_build")
    _patch(QpSolver, "solve", "osqp")
    _patch(sqp.TrajectoryOptimizer, "_objective", "objective")


# ======================================================================================
# settings
# ======================================================================================

#: V11AB on top of the V8 server flags (`ta_offline` overrides; HM forced off as in TA).
V11AB = {"horizon": {"plan_horizon": 16, "lookahead": "obstacles", "lookahead_gate": 0.02},
         "qp": {"eps_abs": 1e-4, "eps_rel": 1e-4}, "cost": {"w_deflection_rate": 20.0}}


def _merge(*parts: dict) -> dict:
    out: dict = {}
    for p in parts:
        for sec, kv in p.items():
            out.setdefault(sec, {}).update(kv)
    return out


FE = {"sqp": {"fast_eval": True}}

#: Setting name → overrides on the run's server config. `V8` = the recorded server as is.
SETTINGS: dict[str, dict] = {"V8": {}, "V11AB": V11AB, "V8fe": FE, "V11ABfe": _merge(V11AB, FE)}


def register(settings: dict[str, dict]) -> None:
    SETTINGS.update(settings)


# ======================================================================================
# one run
# ======================================================================================

def _cgroup_cpu() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        for line in open("/sys/fs/cgroup/cpu.stat"):
            k, v = line.split()
            out[k] = int(v)
    except OSError:
        pass
    try:
        out["loadavg"] = [float(x) for x in open("/proc/loadavg").read().split()[:3]]
    except OSError:
        pass
    out["epoch"] = time.time()
    return out


def prepare_field(scene, device: str):
    """The scene's field as the server holds it, for timing.

    * every layer gets an all-zero label grid, so the destination label is read as the server reads it
      (one lattice pass + a gather, `_patch_destination_label`); no label id is named, so `_identify`
      still answers `None` exactly as on a field without labels;
    * `device == "cuda"`: the layers become `DeviceEsdfField`s (the grids stay on the GPU and only
      the queried corners come back — the server's T38 B3 path, bit-identical values); `"cpu"`:
      the host `EsdfField`s, whose whole-grid gradient cache is built here, untimed (the device
      field has none — it differentiates the corners a query uses).
    Returns `(scene, prewarm_ms)`."""
    import dataclasses

    from benchmark.ag3s.fields.curobo_field import CuroboEsdfField
    from benchmark.trajopt.experiments import ta_offline as TA

    wrapper = scene.esdf if isinstance(scene.esdf, TA._DestinationLabels) else None
    field = wrapper._field if wrapper is not None else scene.esdf
    if field is None:
        return scene, 0.0
    w0 = time.perf_counter()
    if device == "cuda":
        import torch

        from benchmark.ag3s.fields.device_field import layer_from_device

        def dev(layer, labels: bool):
            values = torch.as_tensor(np.asarray(layer.distance_grid, np.float32), device="cuda")
            lab = (torch.zeros(tuple(layer.grid.shape), dtype=torch.int32, device="cuda")
                   if labels else None)
            return layer_from_device(values, layer.grid.origin, layer.grid.voxel_size, labels=lab,
                                     outside_distance=layer.outside_distance)

        new = CuroboEsdfField(tuple(dev(f, True) for f in field.layers),
                              outside_distance=field.outside_distance,
                              target_free_layers=tuple(dev(f, False)
                                                       for f in field.target_free_layers))
        torch.cuda.synchronize()
    else:
        for layer in field.layers:
            if layer.label_grid is None:
                layer.label_grid = np.zeros(tuple(layer.grid.shape), np.int32)
        for layer in tuple(field.layers) + tuple(field.target_free_layers):
            layer.gradient(np.zeros((1, 3)))
        new = field
    if wrapper is not None:
        w = object.__new__(TA._DestinationLabels)
        w.__dict__.update(wrapper.__dict__)
        w.__dict__["_field"] = new
        new = w
    return dataclasses.replace(scene, esdf=new), (time.perf_counter() - w0) * 1e3


def run_timing(key: str, entry: dict, cli) -> dict[str, Any]:
    from benchmark.trajopt.experiments import ta_offline as TA
    from benchmark.trajopt.refiner import TrajOptChunkRefiner

    install()
    ns = argparse.Namespace(hm=False, no_plane_rows=False)
    run = TA.Run(key, entry, ns)
    names = list(cli.variants)
    cfgs = {}
    extra_of = {}
    for v in names:
        if v in SETTINGS:
            over = {sec: dict(kv) for sec, kv in SETTINGS[v].items()}
            over.setdefault("collision", {})["held_obstacle_margin"] = False
            cfgs[v] = run.base_cfg.with_overrides(over)
            extra_of[v] = ()
        else:                                   # a `ta_offline` variant (T43 TP candidates)
            cfgs[v] = TA.variant_config(run.base_cfg, v, False)
            extra_of[v] = TA.variant_server_args(v)
    alts = {e: TA.Run(key, entry, ns, extra_server_args=e) for e in set(extra_of.values()) if e}
    solver_run = {v: alts.get(extra_of[v], run) for v in names}
    holder: dict[str, Any] = {}
    own_prev: dict[str, Any] = {}
    refiners = {v: TrajOptChunkRefiner(solver_run[v].robot, solver_run[v].layout,
                                       lambda _c: holder["value"], cfgs[v]) for v in names}
    out: dict[str, Any] = {"key": key, "tool": "benchmark.trajopt.experiments.tp_timing",
                           "mode": cli.mode, "field": cli.field,
                           "variants": {v: cfgs[v].to_dict() for v in names},
                           "server_args": {v: list(extra_of[v]) for v in names if extra_of[v]},
                           "load_before": _cgroup_cpu(), "chunks": []}
    seqs = sorted(run.chunks)
    if cli.max_chunks:
        seqs = seqs[cli.first_chunk:cli.first_chunk + cli.max_chunks]
    prev_seq = None
    for ci, seq in enumerate(seqs):
        inp, why = run.inputs(seq)
        if inp is None:
            out["chunks"].append({"seq": seq, "skipped": why})
            prev_seq = None
            if cli.mode == "server":
                for r in refiners.values():
                    r.reset()
            continue
        s, q_now, fb = inp["summary"], inp["q_now"], inp["feedback"]
        scene, prewarm_ms = prepare_field(inp["scene"], cli.field)
        alt_inp = {}
        for e, alt in alts.items():
            a, _ = alt.inputs(seq)
            alt_inp[e] = (a, prepare_field(a["scene"], cli.field)[0])
        arm = np.asarray(fb.get("applied_arm"), np.float64) if fb.get("applied_arm") is not None \
            else None
        hist = np.ascontiguousarray(arm[-2:].T) if arm is not None and arm.ndim == 2 \
            and arm.shape[0] >= 2 else None
        dhist = None
        prev_chunk = None
        n_exec = fb.get("n_exec") if fb.get("available") else None
        if prev_seq is not None and prev_seq == seq - 1:
            with np.load(run.chunks[prev_seq], allow_pickle=True) as pz:
                prev_chunk = np.asarray(pz["refined_chunk"], np.float64)
                prev_ref = np.asarray(pz["reference_chunk"], np.float64)
            ex = list(fb.get("executed") or [])
            if arm is not None and ex and ex[-1]:
                dhist = arm[-1] - run.layout.chunk_to_trajectory(prev_ref)[:, len(ex) - 1]
        ctx0: dict[str, Any] = {"t_step": seq, "previous_physical_chunk": prev_chunk}
        if n_exec is not None:
            ctx0["previous_executed_steps"] = int(n_exec)
        g = s.get("grasp") or {}
        rec: dict[str, Any] = {"seq": seq, "grasp_state": g.get("state"),
                               "held": inp["held"] is not None,
                               "recorded_to_ms": (s.get("to") or {}).get("solve_ms"),
                               "prewarm_ms": prewarm_ms,
                               "variants": {}}
        order = names[ci % len(names):] + names[:ci % len(names)]
        for rep in range(cli.repeat):
            for v in (order if rep % 2 == 0 else list(reversed(order))):
                ref = refiners[v]
                if cli.mode == "reset":
                    ref.reset()
                cfg = cfgs[v]
                lin = ref.optimizer.linearizer
                lin.set_joint_parameter_path(solver_run[v].finger_path(s, inp["ref_chunk"],
                                                                       cfg.horizon.planned))
                holder["value"] = ((alt_inp[extra_of[v]][1] if extra_of[v] else scene), q_now, True)
                ctx = dict(ctx0)
                if hist is not None and cfg.limits.servo_accel != "off":
                    ctx["command_history"] = hist
                if dhist is not None and cfg.cost.w_deflection_rate > 0:
                    ctx["deflection_history"] = dhist
                if (cli.mode == "server" and own_prev.get(v) is not None
                        and ctx.get("previous_physical_chunk") is not None):
                    # online, the continuity reference is this setting's own last chunk
                    ctx["previous_physical_chunk"] = own_prev[v]
                CLOCK.reset()
                CLOCK.enabled = True
                c0 = time.process_time()
                w0 = time.perf_counter()
                own = ref.refine(inp["ref_chunk"], ctx)
                wall = (time.perf_counter() - w0) * 1e3
                cpu = (time.process_time() - c0) * 1e3
                CLOCK.enabled = False
                if cli.mode == "server" and rep == 0:
                    own_prev[v] = np.asarray(own, np.float64)
                res = ref.last_result
                stages = {k: v_ * 1e3 for k, v_ in CLOCK.acc.items()}
                harness = stages.pop("harness", 0.0)
                # the server-equivalent label cost was added on top of the solve's own wall time
                wall_adj = wall - harness
                stages["other"] = max(0.0, wall_adj - sum(stages.values()))
                entry_v = rec["variants"].setdefault(v, {"wall_ms": [], "cpu_ms": [], "stages": [],
                                                         "calls": [], "wall_raw_ms": [],
                                                         "harness_ms": []})
                entry_v["wall_ms"].append(wall_adj)
                entry_v["wall_raw_ms"].append(wall)
                entry_v["harness_ms"].append(harness)
                entry_v["cpu_ms"].append(cpu - harness)
                entry_v["stages"].append({k: round(x, 4) for k, x in stages.items()})
                entry_v["calls"].append(dict(CLOCK.calls))
                if rep == 0:
                    traj = np.ascontiguousarray(np.asarray(res.trajectory, np.float64))
                    m = res.metrics
                    la = m.get("lookahead") or {}
                    entry_v.update(
                        sha1=hashlib.sha1(traj.tobytes()).hexdigest(),
                        status=res.status.value, max_violation_mm=float(res.max_violation) * 1e3,
                        solve_ms=float(res.solve_time_ms), gate=la.get("gate"),
                        qp_iterations=m.get("qp_iterations"),
                        sqp_iterations=m.get("sqp_iterations"),
                        traj8=traj[:, :8].tolist() if cli.keep_traj else None)
        out["chunks"].append(rec)
        inp["z"].close()
        for a, _ in alt_inp.values():
            a["z"].close()
        prev_seq = seq
        if cli.progress:
            print(f"{key} seq {seq} " + " ".join(
                f"{v}={np.median(rec['variants'][v]['wall_ms']):.0f}" for v in names), flush=True)
    out["load_after"] = _cgroup_cpu()
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=["V8", "V11AB"])
    ap.add_argument("--mode", choices=("server", "reset"), default="server",
                    help="server: one optimizer per setting kept across the run (as online); "
                         "reset: reset before every solve (ta_offline)")
    ap.add_argument("--field", choices=("cpu", "cuda"), default="cpu",
                    help="cpu: host EsdfField layers (default, CPU only); cuda: DeviceEsdfField "
                         "layers as on the server (one GPU process, < 1 GB)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="solve every chunk this many times per setting (alternating order)")
    ap.add_argument("--settings-module", default=None,
                    help="python module whose SETTINGS dict is merged in (candidate settings)")
    ap.add_argument("--keep-traj", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--first-chunk", type=int, default=0)
    cli = ap.parse_args(argv)
    if cli.mode == "server" and cli.repeat != 1:
        raise SystemExit("--repeat > 1 needs --mode reset: a server-mode repeat would warm-start from "
                         "its own answer to the same chunk")
    if cli.settings_module:
        import importlib

        register(importlib.import_module(cli.settings_module).SETTINGS)
    from benchmark.trajopt.experiments.ta_offline import VARIANTS

    unknown = [v for v in cli.variants if v not in SETTINGS and v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown settings {unknown}; known: {sorted(SETTINGS) + sorted(VARIANTS)}")
    entry = json.load(open(cli.index))["runs"][cli.key]
    started = time.time()
    res = run_timing(cli.key, entry, cli)
    res["wall_s"] = round(time.time() - started, 1)
    out = pathlib.Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(out)
    print(f"{cli.key}: {len(res['chunks'])} chunks, {res['wall_s']} s -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
