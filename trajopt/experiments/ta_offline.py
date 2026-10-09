"""T43 TA — offline re-solve of recorded closed-loop runs under TO path-shape variants (A · B · C).

The scene and the policy chunk of every recorded chunk are held fixed; only the TO is solved again,
once per variant, from the same reconstructed input. Unlike `fixed_input_ablation` (T24), which skips
every chunk with a held object, this harness **includes the held query rows** (the grasped apple,
with the T43 HM classification), so carry and crate-entry chunks are re-solved too.

    MUJOCO_GL=osmesa JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/dev/work \\
        .venv-openpi-live/bin/python -m benchmark.trajopt.experiments.ta_offline \\
        --index outputs/verify/T43/V8/analysis/analysis.json --key V8/H/V8_ep1807_s18071 \\
        --contacts benchmark/ag3s/docs/handoff/T43HX.verify.json \\
        --out outputs/impl/T43TA/raw/V8_H_V8_ep1807_s18071.json

## Reconstruction (per chunk = one server record `chunk_*.npz` + the client planning row)

| input | from |
|---|---|
| server flags → `TrajOptConfig`, robot model | the run's `server_proc.txt` command line parsed by **`serve_safe.build_parser`**, then `serve_safe.trajopt_config_from_args` · `build_constraint_robot_model` (+ `sphere_options` · `gripper_cover_options` · `finger_cover_options`) — the server's own functions |
| `q_now` | planning row `qpos` → `DEFAULT_RBY1_JOINTS` (`fixed_input_ablation.q_now_from_qpos`) |
| field | `esdf_coarse_*` · `esdf_*` · `esdf_free_*` → `fixed_input_ablation.build_field` |
| destination label | **not recorded** → approximated: the nearest field surface point `p − d∇d` inside the record's destination AABB (+1 fine voxel) is the destination (`_DestinationLabels`) |
| contact policy | `ClearancePolicy` (AG3S default) × (manifest phase, manipulators): `manipulated_link_margin`, `target_field_exclude` |
| manipulated object | not held: `target_points`; held: the held query spheres placed by FK at `q_now` (`clearance.manipulated_object`) |
| held query spheres | `grasp.attached.held.body_cover`: `lifted_parent_frame` when `frame.near_support`, else `body_parent_frame`; no body cover: `spheres_parent_frame` (T43 DX rule); parent link `held.frame_link` |
| support plane | `held.support.plane` when recorded; else the run's first recorded one (the table is static); else a plane fitted to `filtered_cloud[support_mask]`. Used for the classification **and** as the plane's own row (`offset + support safety_margin`, as `SupportSurface.to_halfspace`; `--no-plane-rows` = the first TA pass, which had none) |
| capture volume (HM rows, A's tail) | `held.fit_spheres` (capture pose; the runtime also adds the pre-grasp place — not recorded) |
| finger joints | `finger_joints.opening_norm` + the reference chunk's gripper columns → `FingerJointMap.path` (T29) |
| continuity | previous record's `refined_chunk`, aligned by `exec_feedback.n_exec` (as `SafePolicy`) |
| B history | last applied row − previous reference at that row (only if that row executed) |
| C history | `exec_feedback.applied_arm[-2:]` |
| servo state (evaluator only) | `ServoObserver` propagated through the recorded `exec_feedback` in chunk order (`servo.py`) |
| target (outward, evaluator only) | `target_centroid` of chunks with a target and nothing held |

Open loop: every chunk starts from the recorded state with the recorded (V8) history — a variant's
own earlier output does not feed its next chunk (continuity · B · C history · servo state).

Fidelity is measured before any variant: the recorded refined trajectory is re-measured on the
reconstructed scene (`worst_row` over the executed window) and compared with the record's
`to.metrics.clearance_m`; the baseline re-solve is compared with the recorded refined chunk.

## Per chunk and variant (JSON; the aggregation lives in `outputs/impl/T43TA/p/`)

status · window violation · tail record (A) · verdict rows (`safe_policy.classify_violations` over the
executed window: collision rows = a HOLD candidate) · deviation from the policy (joint °, palm mm) ·
boundary sharpness (command second difference across the chunk start, palm-direction turn) ·
obstacle clearance per group (palm, fingers, held) for the commanded rows and for the servo-predicted
states (4 sub-steps per row): TO view (`d − r − margin`) and geometric (`d − r`, obstacle class,
nearest surface outside the destination AABB = "hurdle") · contact moments (HX list) · outward
(finger-midpoint distance to the target at row K−1, variant − policy) · CPU time
(`time.process_time`, variants interleaved in a rotating order, every `--repeat-every`-th chunk
timed twice, the second time in reversed order).
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import shlex
import sys
import time
from typing import Any, Optional

import numpy as np

TOOL = "benchmark.trajopt.experiments.ta_offline"
K_EXEC = 8
SERVO_SAMPLES = (8, 16, 25, 33)

#: OSQP accuracy of every SQP subproblem (`--qp-eps`). The server default is 1e-3; at 1e-3 the
#: 16-step look-ahead subproblem can return a first step far from its own optimum that the merit still
#: accepts (V8 H ep1967 s19672 chunk 17: 0.10 rad, palm 88 mm off the policy at the grasp; 1e-4 and
#: 1e-5 both give 0.019 rad), so A is compared at 1e-4 too, and so is the base (`Q4`).
Q4 = {"qp": {"eps_abs": 1e-4, "eps_rel": 1e-4}}


def _a(h: int = 16, gate: Optional[float] = 0.02) -> dict:
    return {"plan_horizon": h, "lookahead": "obstacles", "lookahead_gate": gate}


def _c(mode: str = "relaxed", deg: float = 1.0, weight: float = 0.01) -> dict:
    return {"servo_accel": mode, "servo_accel_tolerance": float(np.radians(deg)),
            "servo_accel_weight": weight}


def _v(h: Optional[int] = None, gate: Optional[float] = 0.02, q4: bool = True, b: float = 0.0,
       c: Optional[dict] = None) -> dict:
    out: dict = {}
    if h is not None:
        out["horizon"] = _a(h, gate)
    if q4:
        out.update(Q4)
    if b:
        out["cost"] = {"w_deflection_rate": float(b)}
    if c:
        out["limits"] = c
    return out


#: Variant name → overrides on top of the run's server config (`held_obstacle_margin` forced to
#: `--hm`, default off). Grouped A → A+B → A+B+C as the task orders them. `q` = `--qp-eps 1e-4`,
#: `u` = no gate; every other A is gated at 20 mm. C: `r`/`s` = relaxed/strict, the digit = ε in
#: degrees; slack weight 0.01 × w_slack unless `w1` (0.1). At 0.01, 0.01 rad/row² over the bound
#: costs as much as 0.1 mm of collision and 1000× a 0.01-rad tracking error.
VARIANTS: dict[str, dict] = {
    "base": {},
    "Q4": _v(),
    "A16": _v(16, q4=False),
    "A16q": _v(16),
    "A16uq": _v(16, None),
    "A24q": _v(24),
    "A16qB5": _v(16, b=5.0),
    "A16qB20": _v(16, b=20.0),
    "A16qB50": _v(16, b=50.0),
    "A16qB20Cr1": _v(16, b=20.0, c=_c("relaxed", 1.0, 0.01)),
    "A16qB20Cr1w1": _v(16, b=20.0, c=_c("relaxed", 1.0, 0.1)),
    "A16qB20Cr2": _v(16, b=20.0, c=_c("relaxed", 2.0, 0.01)),
    "A16qB20Cs1": _v(16, b=20.0, c=_c("strict", 1.0, 0.01)),
    "A16qB50Cr1": _v(16, b=50.0, c=_c("relaxed", 1.0, 0.01)),
}


def _tp(*parts: dict) -> dict:
    """T43 TP — V11AB (= `A16qB20`) plus the given sections (later parts win per key)."""
    out: dict = {k: dict(v) for k, v in _v(16, b=20.0).items()}
    for part in parts:
        for sec, kv in part.items():
            if sec.startswith("_"):
                out[sec] = kv
            else:
                out.setdefault(sec, {}).update(kv)
    return out


_FE = {"sqp": {"fast_eval": True}}
_WI = {"qp": {"warm_start_iterate": True}}
_E3N = {"qp": {"eps_after_first": 1e-3}}
_CAP = lambda n: {"qp": {"max_iter": int(n), "accept_max_iter": True}}  # noqa: E731
_IT2 = {"sqp": {"max_iterations": 2, "min_iterations": 2}}
_ILP = {"_server_args": ["--finger-cover-solver", "ilp"]}
_HM = {"collision": {"esdf_host_mirror": True}}

#: T43 TP — compute-time candidates, each on top of V11AB (`A16qB20`). `fe` (`sqp.fast_eval`) is
#: bit-identical, so it is in every candidate below `TPfe` only to make the offline pass faster.
#: `_server_args`: extra server flags for that variant's own robot model (`Run(extra_server_args=)`)
#: — its trajectory is still measured on the recorded server's model.
VARIANTS.update({
    "TPfe": _tp(_FE),
    "TPwi": _tp(_FE, _WI),
    "TPe3n": _tp(_FE, _E3N),
    "TPcap1k": _tp(_FE, _CAP(1000)),
    "TPcap500": _tp(_FE, _CAP(500)),
    "TPg10": _tp(_FE, {"horizon": {"lookahead_gate": 0.01}}),
    "TPg5": _tp(_FE, {"horizon": {"lookahead_gate": 0.005}}),
    "TPh12": _tp(_FE, {"horizon": {"plan_horizon": 12}}),
    "TPs2": _tp(_FE, {"horizon": {"lookahead_stride": 2}}),
    "TPs4": _tp(_FE, {"horizon": {"lookahead_stride": 4}}),
    "TPit2": _tp(_FE, _IT2),
    "TPwl2": _tp(_FE, _IT2, {"sqp": {"warm_start_lookahead": True}}),
    "TPilp": _tp(_FE, _ILP),
    "TPwi_e3n": _tp(_FE, _WI, _E3N),
    "TPwi_cap1k": _tp(_FE, _WI, _CAP(1000)),
    "TPwi_e3n_cap1k": _tp(_FE, _WI, _E3N, _CAP(1000)),
    "TPwi_e3n_cap1k_s2": _tp(_FE, _WI, _E3N, _CAP(1000), {"horizon": {"lookahead_stride": 2}}),
    "TPwi_e3n_cap1k_h12": _tp(_FE, _WI, _E3N, _CAP(1000), {"horizon": {"plan_horizon": 12}}),
    # with the look-ahead warm start (+ 2 SQP iterations): needs `--keep-state` to mean anything
    "TPwi_e3n_wl2": _tp(_FE, _WI, _E3N, _IT2, {"sqp": {"warm_start_lookahead": True}}),
    "TPwi_e3n_cap1k_wl2": _tp(_FE, _WI, _E3N, _CAP(1000), _IT2,
                              {"sqp": {"warm_start_lookahead": True}}),
    # GPU field read from a per-chunk host copy: changes nothing on the host fields used here; it is
    # a timing setting for `tp_timing --field cuda` (the server's `DeviceEsdfField`)
    "TPhm": _tp(_FE, _HM),
    "TPwi_e3n_cap1k_hm": _tp(_FE, _WI, _E3N, _CAP(1000), _HM),
    "TPwi_e3n_cap1k_wl2_hm": _tp(_FE, _WI, _E3N, _CAP(1000), _IT2, _HM,
                                 {"sqp": {"warm_start_lookahead": True}}),
})


GROUPS = ("palm", "fingers", "held")


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


# ======================================================================================
# server settings
# ======================================================================================

def server_args(chunk_dir: pathlib.Path, extra: tuple = ()):
    """The server's argparse namespace, from `server_proc.txt` next to its record directory.
    `extra` (T43 TP): flags appended to the recorded command line (a variant's own robot model)."""
    from benchmark.trajopt import serve_safe

    proc = chunk_dir.parent.parent / "server_proc.txt"
    line = next(x for x in proc.read_text().splitlines() if "benchmark.trajopt.serve_safe" in x)
    tokens = shlex.split(line)
    argv = tokens[tokens.index("benchmark.trajopt.serve_safe") + 1:] + list(extra)
    return serve_safe.build_parser().parse_args(argv), " ".join(argv)


def build_robot(args):
    import mujoco

    from benchmark.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene
    from benchmark.trajopt import serve_safe

    mj = mujoco.MjModel.from_xml_path(str(pathlib.Path(args.model_xml).resolve()))
    scene = TransportScene.attach(mj, mujoco.MjData(mj))
    cover = {}
    gc = serve_safe.gripper_cover_options(args)
    if gc:
        cover["gripper_cover"] = gc
    fc = serve_safe.finger_cover_options(args)
    if fc:
        cover["finger_cover"] = fc
    robot = build_constraint_robot_model(
        scene, link_filter=serve_safe.constraint_link_filter(args.links),
        sphere_options=serve_safe.sphere_options(args), **cover)
    qadr = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i): int(mj.jnt_qposadr[i])
            for i in range(mj.njnt)}
    return robot, qadr


def variant_server_args(name: str) -> tuple:
    """T43 TP — extra server flags of a variant (its own robot model), `()` for the recorded one."""
    return tuple(VARIANTS[name].get("_server_args", ()))


def variant_config(base_cfg, name: str, hm: bool):
    over = {k: dict(v) for k, v in VARIANTS[name].items() if not k.startswith("_")}
    # T43 TA baseline = V8 flags (lead, 2026-10-07: HM on hold) — HM is forced off unless asked,
    # also for records of a server that ran with it (V10).
    over.setdefault("collision", {})["held_obstacle_margin"] = bool(hm)
    return base_cfg.with_overrides(over) if over else base_cfg


# ======================================================================================
# scene
# ======================================================================================

class _DestinationLabels:
    """The recorded field + an approximate destination label (the label grid is not recorded):
    a point is `destination` when its nearest field surface point lies in the destination AABB."""

    def __init__(self, field, aabb_min, aabb_max, label: str):
        self._field = field
        pad = float(field.grid.voxel_size)
        self._lo = np.asarray(aabb_min, np.float64) - pad
        self._hi = np.asarray(aabb_max, np.float64) + pad
        self._label = str(label)

    def __getattr__(self, name):
        return getattr(self._field, name)

    has_labels = True

    def is_label(self, points, name):
        p = np.asarray(points, np.float64).reshape(-1, 3)
        if str(name) != self._label:
            return np.zeros(len(p), bool)
        surf = nearest_surface(self._field, p)
        return np.all((surf >= self._lo) & (surf <= self._hi), axis=1)

    def distance_and_label(self, points, name):
        """T43 TP (`sqp.fast_eval`): the field's distance and **this** label (explicit, so the
        attribute fallback never hands out the field's own label grid instead)."""
        return self._field.distance(points), self.is_label(points, name)


def nearest_surface(field, p: np.ndarray) -> np.ndarray:
    d = np.asarray(field.distance(p), np.float64).reshape(-1)
    g = np.asarray(field.gradient(p), np.float64).reshape(-1, 3)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return p - d[:, None] * np.where(n > 1e-12, g / np.maximum(n, 1e-12), 0.0)


def held_set(summary: dict):
    """`(parent_link, centres (N,3), radii (N,), fit (K,4) | None, plane (4,) | None)` or None."""
    g = summary.get("grasp") or {}
    if not g.get("attached_in_constraints"):
        return None
    att = lit(g.get("attached"))
    if not isinstance(att, dict):
        return None
    held = lit(att.get("held")) or {}
    bc = lit(held.get("body_cover"))
    if isinstance(bc, dict) and bc:
        near = bool((bc.get("frame") or {}).get("near_support"))
        src = bc.get("lifted_parent_frame" if near else "body_parent_frame")
    else:
        src = held.get("spheres_parent_frame")
    if not src:
        return None
    c = np.asarray([x["centre_m"] for x in src], np.float64).reshape(-1, 3)
    r = np.asarray([x["radius_mm"] for x in src], np.float64) * 1e-3
    fit = held.get("fit_spheres") or []
    fit = (np.asarray([[*x["centre_m"], x["radius_mm"] * 1e-3] for x in fit], np.float64)
           if fit else None)
    sup = (held.get("support") or {}).get("plane") or {}
    plane = None
    if sup.get("normal") is not None and sup.get("offset_m") is not None:
        n = np.asarray(sup["normal"], np.float64)
        plane = np.asarray([*(n / np.linalg.norm(n)), float(sup["offset_m"])])
    link = held.get("frame_link") or att.get("parent_link")
    return str(link), c, r, fit, plane


def fitted_plane(z) -> Optional[np.ndarray]:
    if "support_mask" not in z.files or "filtered_cloud" not in z.files:
        return None
    m = np.asarray(z["support_mask"], bool)
    pts = np.asarray(z["filtered_cloud"], np.float64)
    if m.sum() < 50 or m.shape[0] != pts.shape[0]:
        return None
    p = pts[m]
    c = p.mean(axis=0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
    n = n if n[2] > 0 else -n
    off = float(np.median(p @ n))
    return np.asarray([*n, off])


# ======================================================================================
# one run
# ======================================================================================

class Run:
    def __init__(self, key: str, entry: dict, args_cli, extra_server_args: tuple = ()):
        from benchmark.ag3s.config import AG3SConfig
        from benchmark.ag3s.constraints.clearance import ClearancePolicy
        from benchmark.ag3s.types import ContactPolicyContext, SourceType
        from benchmark.trajopt import serve_safe
        from benchmark.trajopt.limits import build_limits
        from benchmark.trajopt.servo import ServoModel, ServoObserver, servo_params_from_xml
        from benchmark.trajopt.types import ChunkLayout

        self.key = key
        self.rd = pathlib.Path(entry["run_dir"])
        seg = lit(entry["server_segment"])
        first = pathlib.Path(seg["first"])
        i0 = int(first.stem.split("_")[1])
        self.chunks: dict[int, pathlib.Path] = {}
        for n in range(int(seg["n"])):
            p = first.parent / f"chunk_{i0 + n:05d}.npz"
            with np.load(p, allow_pickle=True) as z:
                self.chunks[int(z["t_step"])] = p
        self.sargs, self.cmdline = server_args(first.parent, tuple(extra_server_args))
        # the table plane AG3S recorded at attach (static over the run); chunks without one use it
        self.table_plane = None
        for path in self.chunks.values():
            with np.load(path, allow_pickle=True) as z:
                hs = held_set(json.loads(str(z["summary_json"])))
            if hs is not None and hs[4] is not None:
                self.table_plane = hs[4]
                break
        self.base_cfg = serve_safe.trajopt_config_from_args(self.sargs)
        self.hm = bool(args_cli.hm)
        self.robot, self.qadr = build_robot(self.sargs)
        self.layout = ChunkLayout.rby1(self.robot.joint_names)
        self.limits = build_limits(self.robot, self.layout, dt=self.base_cfg.horizon.dt,
                                   config=self.base_cfg.limits)
        rows = [json.loads(x) for x in open(self.rd / "frames.jsonl") if x.strip()]
        self.plan = {int(r["seq"]): r for r in rows if r["kind"] == "planning"}
        self.ctrl = {int(r["t_step"]): r for r in rows if r["kind"] == "control"}
        self.manifest = json.load(open(self.rd / "manifest.json"))
        pol = self.manifest.get("policy") or {}
        self.phase = str(pol.get("safe_phase") or "approach")
        self.manipulators = [str(m) for m in (pol.get("safe_manipulators") or ["left"])]
        self.hand = self.manipulators[0]
        cfg = AG3SConfig.from_dict({"collision_backend": "esdf"})
        self.policy = ClearancePolicy.from_config(cfg.contact, cfg.geometry, cfg.support_surface)
        self.ctx = ContactPolicyContext.make(self.phase, self.manipulators)
        self.authorized = sorted(self.policy.authorized_links(self.ctx))
        self.dest_margin = float(self.policy.full_margin(SourceType.DESTINATION))
        self.support_margin = float(self.policy.support_margin)
        self.plane_rows = not bool(getattr(args_cli, "no_plane_rows", False))
        self.outside = (float(cfg.esdf.max_distance) if cfg.esdf.unknown_policy == "free"
                        else -float(cfg.esdf.max_distance))
        names = [str(n) for n in self.robot.sphere_link_names]
        self.names = names
        self.margin_vec = self.policy.margin_matrix(names, [SourceType.TARGET], context=self.ctx,
                                                    target_grounded=True)[:, 0]
        fl = "l" if self.hand == "left" else "r"
        self.idx = {"palm": np.asarray([i for i, n in enumerate(names) if n == f"ee_{self.hand}"]),
                    "fingers": np.asarray([i for i, n in enumerate(names)
                                           if n in (f"ee_finger_{fl}1", f"ee_finger_{fl}2")])}
        self.palm_link = f"ee_{self.hand}"
        # finger joints (T29)
        from benchmark.ag3s.robot_models.gripper_state import FingerJointMap

        self.fmap = FingerJointMap(self.robot) if getattr(self.robot, "param_joint_names", ()) \
            else None
        # servo (evaluator only)
        jn = [str(self.robot.joint_names[int(i)]) for i in self.layout.q_indices]
        sp = servo_params_from_xml(str(self.sargs.model_xml), jn,
                                   control_hz=self.base_cfg.horizon.control_hz)
        self.servo = ServoModel(sp)
        self.observer = ServoObserver(self.servo)
        self.servo_summary = sp.summary()
        self.meters: dict[int, Any] = {}

    # ---------------------------------------------------------------------------------
    def meter(self, horizon: int):
        from benchmark.trajopt.linearize import CollisionLinearizer

        if horizon not in self.meters:
            self.meters[horizon] = CollisionLinearizer(self.robot, self.layout, horizon)
        return self.meters[horizon]

    def finger_path(self, summary: dict, chunk: np.ndarray, horizon: int):
        if self.fmap is None:
            return None
        on = (summary.get("finger_joints") or {}).get("opening_norm") or {}
        measured = {h: float(v) for h, v in on.items() if v is not None}
        if not measured:
            return None
        left, right = 7, 15
        commanded = {"left": chunk[:, left], "right": chunk[:, right]}
        P, _ = self.fmap.path(measured, commanded, horizon, default=self.robot.param_vector())
        return P

    def inputs(self, seq: int):
        """Reconstructed TO input of chunk `seq`, or `(None, reason)`."""
        from benchmark.trajopt.experiments import fixed_input_ablation as FIA
        from benchmark.trajopt.linearize import SceneSnapshot

        if seq not in self.plan:
            return None, "no planning row"
        z = np.load(self.chunks[seq], allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        row = self.plan[seq]
        ref_chunk = np.asarray(z["reference_chunk"], np.float64)
        rec_chunk = np.asarray(z["refined_chunk"], np.float64)
        if np.abs(np.asarray(row["actions"], np.float64) - rec_chunk).max() > 1e-5:
            return None, "record and planning row are not the same chunk"
        q_now = FIA.q_now_from_qpos(row["qpos"], self.qadr)
        if "esdf_distance" not in z.files:
            return None, "no field recorded"
        field = FIA.build_field(z, outside_distance=self.outside)
        dest = s.get("destination") or {}
        label = None
        if dest.get("aabb_min") is not None and dest.get("label"):
            label = str(dest["label"])
            field = _DestinationLabels(field, dest["aabb_min"], dest["aabb_max"], label)
        held = held_set(s)
        plane = held[4] if held is not None and held[4] is not None else self.table_plane
        if plane is None:
            plane = fitted_plane(z)
        cfg = self.base_cfg
        kw: dict[str, Any] = {}
        radii = self.meter(K_EXEC).robot_radii
        # manipulated object + contact policy (to_adapter.build_constraint_set)
        if held is not None:
            link, c_loc, r_loc, fit, _ = held
            T = np.asarray(self.robot.link_pose(q_now, link), np.float64)
            kw.update(manipulated_spheres=(c_loc @ T[:3, :3].T + T[:3, 3]),
                      manipulated_sphere_radii=r_loc, manipulated_link_margin=self.margin_vec,
                      attached_points=c_loc, attached_parent_link=link, attached_radii=r_loc)
        else:
            fit = None
            tp = z["target_points"] if "target_points" in z.files else None
            if bool(s.get("has_target")) and tp is not None and len(tp):
                kw.update(manipulated_points=np.asarray(tp, np.float64),
                          manipulated_link_margin=self.margin_vec)
                if (str(cfg_policy(self.sargs)) != "relax"
                        and getattr(field, "target_free_layers", ())):
                    if cfg_policy(self.sargs) == "exclude_all":
                        mask = np.ones(len(self.names), bool)
                    else:
                        mask = np.asarray([n in self.authorized for n in self.names], bool)
                    if mask.any():
                        kw["target_free_mask"] = mask
        if label is not None:
            kw.update(destination_label=label, destination_margin=self.dest_margin)
        om = float(cfg.collision.obstacle_margin)
        extra = {"obstacle_margin": om, "support_planes": None if plane is None else plane[None]}
        if held is not None and fit is not None:
            # read only by the HM rows and the look-ahead tail (`_held_rows_class`) — inert otherwise
            extra["held_capture_spheres"] = fit
        # the support plane's own rows (`scene_from_constraint_set` keeps them under every backend:
        # `SupportSurface.to_halfspace` = offset + safety margin), unless `--no-plane-rows`
        rows_on = plane is not None and self.plane_rows and bool(
            getattr(cfg.collision, "use_support_planes", True))
        pn = plane[None, :3] if rows_on else np.zeros((0, 3))
        po = (np.asarray([plane[3] + self.support_margin]) if rows_on else np.zeros(0))
        scene = SceneSnapshot(
            candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
            candidate_active=np.zeros(0, bool), d_safe=np.zeros((radii.size, 0)),
            plane_normal=pn, plane_offset=po,
            plane_active=np.ones(po.shape[0], bool), robot_radii=radii,
            candidate_ids=np.zeros(0, np.int64), esdf=field,
            esdf_margin=float(cfg.collision.esdf_margin), **kw, **extra)
        hm_scene = None
        if self.hm or bool(cfg.collision.held_obstacle_margin):
            import dataclasses

            hm_scene = dataclasses.replace(
                scene, held_obstacle_margin=True,
                held_capture_spheres=(fit if held is not None else None))
        tc = (np.asarray(z["target_centroid"], np.float64).reshape(3)
              if "target_centroid" in z.files and held is None and bool(s.get("has_target"))
              else None)
        fb = s.get("exec_feedback") or row.get("exec_feedback") or {}
        fb = dict(fb)
        if isinstance(fb.get("applied_arm"), str):
            fb["applied_arm"] = lit(fb["applied_arm"])
        return {"seq": seq, "z": z, "summary": s, "row": row, "q_now": q_now, "scene": scene,
                "hm_scene": hm_scene, "ref_chunk": ref_chunk, "rec_chunk": rec_chunk,
                "held": held, "feedback": fb, "plane": plane, "dest": dest,
                "target_centroid": tc}, None


def cfg_policy(sargs) -> str:
    from benchmark.trajopt import serve_safe

    return str(serve_safe.resolve_target_field_policy(sargs.target_field_policy))


# ======================================================================================
# measurements
# ======================================================================================

def states_of(run: Run, traj: np.ndarray, q_now, scene, fingers: Optional[np.ndarray]):
    """Centres `(H, Q, 3)` with the held spheres, through a meter linearizer of that horizon."""
    lin = run.meter(traj.shape[1])
    lin.set_joint_parameter_path(fingers)
    lin.set_attached(getattr(scene, "attached_points", None),
                     getattr(scene, "attached_parent_link", None))
    return lin, lin.sphere_states(traj, q_now)[0]


def group_clearance(run: Run, lin, centres, scene, dest_box) -> dict[str, Any]:
    """Per group, per row: TO view (`d − r − margin`) and geometric hurdle-ish (`d − r`, obstacle
    class, nearest surface outside the destination box)."""
    to = lin._esdf_clearance(centres, scene)[..., 0]                      # (H, Q)
    cls = getattr(lin, "_last_obstacle_class", None)
    H, Q = to.shape
    flat = centres.reshape(-1, 3)
    d = np.asarray(scene.esdf.distance(flat), np.float64).reshape(H, Q)
    geo = d - np.asarray(lin.query_radii, np.float64)[None, :Q]
    hurdle = np.full((H, Q), np.inf)
    near = geo < 0.08
    if cls is not None:
        near &= (np.asarray(cls) == 0)
    if near.any():
        pts = flat[near.reshape(-1)]
        surf = nearest_surface(scene.esdf, pts)
        inside = np.zeros(len(pts), bool)
        if dest_box is not None:
            lo, hi = dest_box
            inside = np.all((surf >= lo) & (surf <= hi), axis=1)
        vals = geo.reshape(-1)[near.reshape(-1)]
        tmp = hurdle.reshape(-1)
        idx = np.flatnonzero(near.reshape(-1))
        tmp[idx[~inside]] = vals[~inside]
        hurdle = tmp.reshape(H, Q)
    n = lin.n_spheres
    out = {}
    for g in GROUPS:
        cols = (np.arange(n, Q) if g == "held" else run.idx[g])
        if len(cols) == 0:
            out[g] = None
            continue
        out[g] = {"to_mm": (to[:, cols].min(axis=1) * 1e3).tolist(),
                  "hurdle_mm": [None if not np.isfinite(v) else v
                                for v in (hurdle[:, cols].min(axis=1) * 1e3).tolist()]}
    return out


def finger_mid(run: Run, traj, q_now, row: int) -> np.ndarray:
    """Midpoint of the active hand's two finger link origins at `row` (URDF names; the midpoint is
    the same whichever finger is called l1 — the MJCF/URDF left-finger swap does not matter here)."""
    full = run.layout.full_q(traj[:, row:row + 1], q_now)[:, 0]
    fl = "l" if run.hand == "left" else "r"
    return 0.5 * sum(np.asarray(run.robot.link_pose(full, f"ee_finger_{fl}{i}"))[:3, 3]
                     for i in (1, 2))


def palm_path(run: Run, traj, q_now) -> np.ndarray:
    full = run.layout.full_q(traj, q_now)
    return np.stack([np.asarray(run.robot.link_pose(full[:, k], run.palm_link))[:3, 3]
                     for k in range(traj.shape[1])])


def boundary_metrics(run: Run, traj, q_now, hist: Optional[np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    Q = np.asarray(traj, np.float64)[:, :K_EXEC]
    inner = np.abs(np.diff(Q, 2, axis=1)).max() if Q.shape[1] > 2 else 0.0
    out["acc_inner_max_rad"] = float(inner)
    if hist is None:
        return out
    ext = np.concatenate([hist, Q[:, :2]], axis=1)                      # u-2, u-1, Q0, Q1
    acc = np.abs(np.diff(ext, 2, axis=1))
    out["acc_boundary_max_rad"] = float(acc.max())
    step = np.asarray(run.limits.max_step, np.float64)
    out["vel_boundary_ratio"] = float(np.max(np.abs(Q[:, 0] - hist[:, -1]) / step))
    out["vel_inner_ratio"] = float(np.max(np.abs(np.diff(Q, axis=1)) / step[:, None]))
    P = palm_path(run, ext, q_now)
    v = np.diff(P, axis=0)                                               # 3 displacements
    turns = []
    for a, b in ((v[0], v[1]), (v[1], v[2])):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        turns.append(float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))
                     if na > 2e-3 and nb > 2e-3 else None)
    out["palm_turn_deg"] = turns
    out["palm_reversal"] = any(t is not None and t > 90.0 for t in turns)
    acc_e = np.diff(P, 2, axis=0)
    out["palm_acc_boundary_mm"] = float(np.linalg.norm(acc_e, axis=1).max() * 1e3)
    return out


# ======================================================================================
# main loop
# ======================================================================================

def phase_of(s: dict, state: dict) -> str:
    g = s.get("grasp") or {}
    st = g.get("state")
    if g.get("attached_in_constraints"):
        state["held_seen"] = True
        return "carry"
    if st == "closing":
        return "closing"
    if state.get("held_seen") or st == "placed":
        return "return"
    return "approach"


def run_one(key: str, entry: dict, cli) -> dict[str, Any]:
    from benchmark.trajopt.refiner import TrajOptChunkRefiner
    from benchmark.trajopt.safe_policy import classify_violations

    run = Run(key, entry, cli)
    variants = list(cli.variants)
    cfgs = {v: variant_config(run.base_cfg, v, run.hm) for v in variants}
    # T43 TP — a variant with its own server flags (robot model) solves on its own `Run`; every
    # measurement below stays on the recorded server's model (`run`).
    alts: dict[tuple, Run] = {}
    for v in variants:
        extra = variant_server_args(v)
        if extra and extra not in alts:
            alts[extra] = Run(key, entry, cli, extra_server_args=extra)
    solver_run = {v: alts.get(variant_server_args(v), run) for v in variants}
    keep_state = bool(getattr(cli, "keep_state", False))
    ref_variant = getattr(cli, "ref_variant", None)
    holder: dict[str, Any] = {}
    own_prev: dict[str, Any] = {}
    refiners = {v: TrajOptChunkRefiner(solver_run[v].robot, solver_run[v].layout,
                                       lambda _c: holder["value"], cfgs[v])
                for v in variants}
    contacts = {}
    if cli.contacts:
        for c in json.load(open(cli.contacts))["numbers"]["contacts22"]:
            if c["key"] == key:
                contacts[int(c["t"])] = c
    out: dict[str, Any] = {
        "key": key, "tool": TOOL, "server_cmdline": run.cmdline, "hm": run.hm,
        **({"keep_state": True} if keep_state else {}),
        **({"ref_variant": ref_variant} if ref_variant is not None else {}),
        "plane_rows": run.plane_rows, "support_margin_m": run.support_margin,
        "variants": {v: cfgs[v].to_dict() for v in variants},
        "servo": run.servo_summary, "n_spheres": int(run.robot.n_spheres),
        "grasp_t": entry.get("grasp_t"), "place_t": entry.get("place_t"),
        "contacts": list(contacts.values()), "chunks": []}
    state: dict[str, Any] = {}
    prev_seq = None
    seqs = sorted(run.chunks)
    if cli.max_chunks:
        seqs = seqs[cli.first_chunk:cli.first_chunk + cli.max_chunks]
    for ci, seq in enumerate(seqs):
        inp, why = run.inputs(seq)
        if inp is None:
            out["chunks"].append({"seq": seq, "skipped": why})
            prev_seq = None
            continue
        alt_inp = {}
        for extra, alt in alts.items():
            alt_inp[extra], why_alt = alt.inputs(seq)
            if alt_inp[extra] is None:
                raise RuntimeError(f"{key} seq {seq}: the {extra} model's input failed ({why_alt}) "
                                   "where the recorded model's did not")
        s, q_now, fb = inp["summary"], inp["q_now"], inp["feedback"]
        q14 = q_now[run.layout.q_indices]
        sstate = run.observer.update(q14, fb if fb.get("available") else None, seq)
        phase = phase_of(s, state)
        g = s.get("grasp") or {}
        rec: dict[str, Any] = {
            "seq": seq, "t0": int(inp["row"]["t_step"]), "phase": phase,
            "grasp_state": g.get("state"), "attach": bool(g.get("attach")),
            "recorded_status": (s.get("to") or {}).get("status"),
            "recorded_clearance_mm": None, "servo_source": sstate.source, "variants": {}}
        dest = inp["dest"]
        dest_box = None
        if dest.get("aabb_min") is not None:
            dest_box = (np.asarray(dest["aabb_min"]) - 0.005, np.asarray(dest["aabb_max"]) + 0.005)
        if inp["held"] is not None and dest_box is not None:
            T = np.asarray(run.robot.link_pose(q_now, inp["held"][0]))
            centre = (inp["held"][1] @ T[:3, :3].T + T[:3, 3]).mean(axis=0)
            lo, hi = dest_box
            gap = np.maximum(np.maximum(lo - centre, centre - hi), 0.0)
            if float(np.linalg.norm(gap[:2])) <= 0.08 and centre[2] <= hi[2] + 0.08:
                rec["phase"] = phase = "crate"
        scene = inp["hm_scene"] if run.hm else inp["scene"]
        rec_traj = run.layout.chunk_to_trajectory(inp["rec_chunk"])[:, :K_EXEC]
        ref_traj_full = run.layout.chunk_to_trajectory(inp["ref_chunk"])
        # fidelity — the recorded refined chunk on the reconstructed (server-flag) scene
        met = (s.get("to") or {}).get("metrics") or {}
        server_scene = (inp["hm_scene"] if run.base_cfg.collision.held_obstacle_margin
                        else inp["scene"])
        if met.get("clearance_m") is not None:
            lin8 = run.meter(K_EXEC)
            lin8.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], K_EXEC))
            lin8.set_attached(inp["scene"].attached_points, inp["scene"].attached_parent_link)
            mine, pair = lin8.worst_row(rec_traj, q_now, server_scene)
            recp = lit(met.get("max_violation_pair")) or {}
            rec["recorded_clearance_mm"] = float(met["clearance_m"]) * 1e3
            rec["fidelity"] = {"recon_clearance_mm": float(mine) * 1e3,
                               "abs_diff_mm": abs(float(mine) - float(met["clearance_m"])) * 1e3,
                               "same_row": bool(pair and recp and int(recp.get("step", -1))
                                                == int(pair["step"]) and int(recp.get("query", -1))
                                                == int(pair["query"]))}
        # history (B, C) and continuity input, as SafePolicy would build them
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
        context_base: dict[str, Any] = {"t_step": seq, "previous_physical_chunk": prev_chunk}
        if n_exec is not None:
            context_base["previous_executed_steps"] = int(n_exec)
        order = variants[ci % len(variants):] + variants[:ci % len(variants)]
        timing_pass = [order] + ([list(reversed(order))] if ci % max(1, cli.repeat_every) == 0
                                 and not keep_state else [])
        # T43 TP `--keep-state`: optimizers carry over between consecutive chunks (as online).
        carry = keep_state and prev_seq is not None and prev_seq == seq - 1
        results: dict[str, Any] = {}
        times: dict[str, list] = {v: [] for v in variants}
        for pass_i, seq_order in enumerate(timing_pass):
            for v in seq_order:
                ref = refiners[v]
                if not carry:
                    ref.reset()
                cfg = cfgs[v]
                H = cfg.horizon.planned
                lin = ref.optimizer.linearizer
                extra = variant_server_args(v)
                if extra:
                    a_inp = alt_inp[extra]
                    v_scene = a_inp["hm_scene"] if run.hm else a_inp["scene"]
                    lin.set_joint_parameter_path(solver_run[v].finger_path(s, inp["ref_chunk"], H))
                else:
                    v_scene = scene
                    lin.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], H))
                # certified = True: geometry certification is the same for every variant and is
                # not what is compared here (the verdict rows are, `classify_violations`).
                holder["value"] = (v_scene, q_now, True)
                ctx = dict(context_base)
                if hist is not None and cfg.limits.servo_accel != "off":
                    ctx["command_history"] = hist
                if dhist is not None and cfg.cost.w_deflection_rate > 0:
                    ctx["deflection_history"] = dhist
                if carry and own_prev.get(v) is not None:
                    # `--keep-state`: the continuity reference is this variant's own last chunk, as
                    # online (the refiner aligns it by the executed steps, `n_exec`)
                    ctx["previous_physical_chunk"] = own_prev[v]
                t0 = time.process_time()
                w0 = time.perf_counter()
                own = ref.refine(inp["ref_chunk"], ctx)
                if keep_state and pass_i == 0:
                    own_prev[v] = np.asarray(own, np.float64)
                times[v].append({"cpu_ms": (time.process_time() - t0) * 1e3,
                                 "wall_ms": (time.perf_counter() - w0) * 1e3})
                if pass_i == 0:
                    results[v] = ref.last_result
        base_traj = None
        trajs: dict[str, np.ndarray] = {}
        for v in variants:
            res = results[v]
            traj = np.asarray(res.trajectory, np.float64)
            cfg = cfgs[v]
            H = traj.shape[1]
            ref_traj = ref_traj_full[:, :H]
            ex = traj[:, :K_EXEC]
            if v == variants[0]:
                base_traj = ex
            trajs[v] = ex
            fingers8 = run.finger_path(s, inp["ref_chunk"], K_EXEC)
            lin8, cen = states_of(run, ex, q_now, scene, fingers8)
            gc = group_clearance(run, lin8, cen, scene, dest_box)
            # servo-predicted states (evaluator) — 4 sub-steps per row
            ro = run.servo.rollout(sstate.q, sstate.v, ex, sstate.tau, samples=SERVO_SAMPLES)
            qs = ro["q_samples"].reshape(ex.shape[0], -1)                # (nq, 8*4)
            fs = None if fingers8 is None else np.repeat(fingers8, len(SERVO_SAMPLES), axis=1)
            lin32, cen32 = states_of(run, qs, q_now, scene, fs)
            gs = group_clearance(run, lin32, cen32, scene, dest_box)
            cls = classify_violations(
                lin8, ex, q_now, scene, authorized_links=run.authorized,
                tolerance=float(cfg.safety.violation_tolerance),
                manipulated=s.get("manipulated") or {}, steps=K_EXEC)
            pr = palm_path(run, ex, q_now)
            pref = palm_path(run, ref_traj[:, :K_EXEC], q_now)
            m = res.metrics
            vr = {
                "status": res.status.value,
                "window_violation_mm": float(res.max_violation) * 1e3,
                "n_collision_rows": int(cls["n_collision"]),
                "n_allowed_rows": int(cls["n_allowed"]),
                "collision_worst": None if cls["collision_worst"] is None else {
                    k: cls["collision_worst"].get(k) for k in ("link", "step", "clearance_m",
                                                               "why", "label")},
                "dev_joint_deg_max": float(np.degrees(np.abs(ex - ref_traj[:, :K_EXEC]).max())),
                "dev_palm_mm": (np.linalg.norm(pr - pref, axis=1) * 1e3).tolist(),
                "vs_base_joint_deg_max": float(np.degrees(np.abs(ex - base_traj).max())),
                "vs_recorded_joint_deg_max": float(np.degrees(np.abs(ex - rec_traj).max())),
                # the base at `--qp-eps 1e-4` (variant `Q4`), when it ran before this one
                "vs_q4_joint_deg_max": (float(np.degrees(np.abs(ex - trajs["Q4"]).max()))
                                        if "Q4" in trajs else None),
                "boundary": boundary_metrics(run, ex, q_now, hist),
                "clear_cmd": gc, "clear_servo": gs,
                "cpu_ms": [t["cpu_ms"] for t in times[v]],
                "wall_ms": [t["wall_ms"] for t in times[v]],
                "solve_ms": float(res.solve_time_ms),
                "sqp_iterations": m.get("sqp_iterations"),
                "qp_failures": m.get("qp_failures"), "returned": m.get("returned"),
                "lookahead": m.get("lookahead"),
                "servo_accel": None if m.get("servo_accel") is None else {
                    k: m["servo_accel"].get(k) for k in ("overshoot_max_rad", "n_over",
                                                         "boundary", "boundary_velocity_ratio",
                                                         "boundary_velocity_overshoot_rad")},
                "deflection_rate": m.get("deflection_rate"),
            }
            tc = inp["target_centroid"]
            if tc is not None:
                # T43 W focus rule on the target AG3S tracks: > 0 = the fingers end farther from it
                # than the policy's row 7 (row K-1), the V9 grasp-loss signature
                d_v = float(np.linalg.norm(finger_mid(run, ex, q_now, K_EXEC - 1) - tc))
                d_p = float(np.linalg.norm(finger_mid(run, ref_traj[:, :K_EXEC], q_now,
                                                      K_EXEC - 1) - tc))
                vr["finger_target_mm"] = d_v * 1e3
                vr["outward_mm"] = (d_v - d_p) * 1e3
            if v == variants[0]:
                vr["exec_palm_path_m"] = pr.tolist()
            rec["variants"][v] = vr
        if ref_variant is not None and ref_variant in trajs:
            # T43 TP — every variant against the reference variant (V11AB), rows 0-7
            for v in variants:
                rec["variants"][v]["vs_ref_joint_deg_max"] = float(
                    np.degrees(np.abs(trajs[v] - trajs[ref_variant]).max()))
        for t, c in contacts.items():
            crow = run.ctrl.get(t)
            if crow is not None and int(crow["chunk_seq"]) == seq:
                rec.setdefault("contact_rows", []).append(
                    {"t": t, "k": int(crow["step_in_chunk"]), "group": c["group"]})
        out["chunks"].append(rec)
        inp["z"].close()
        for a_inp in alt_inp.values():
            if a_inp is not None:
                a_inp["z"].close()
        prev_seq = seq
        if cli.progress:
            print(f"{key} seq {seq} {phase} done", flush=True)
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="analysis json with runs[key].run_dir / "
                                                   "server_segment (T43 V8 / V10 analysis)")
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contacts", default=None, help="T43HX.verify.json (numbers.contacts22)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--hm", action="store_true",
                    help="add --held-obstacle-margin to every variant (default: off for every "
                         "variant — the T43 TA baseline is the V8 flag set)")
    ap.add_argument("--no-plane-rows", action="store_true",
                    help="leave the support plane's own rows out of the scene (the first TA pass "
                         "did; the server keeps them)")
    ap.add_argument("--repeat-every", type=int, default=3,
                    help="time every N-th chunk twice (forward and reversed variant order)")
    ap.add_argument("--keep-state", action="store_true",
                    help="T43 TP: keep every variant's optimizer across consecutive chunks (warm "
                         "starts carry over, as online) and give it its own previous chunk as the "
                         "continuity reference; no repeated timing pass")
    ap.add_argument("--ref-variant", default=None,
                    help="T43 TP: also record each variant's max joint change vs this variant "
                         "(`vs_ref_joint_deg_max`, rows 0-7)")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--max-chunks", type=int, default=0, help="debug: only this many chunks")
    ap.add_argument("--first-chunk", type=int, default=0, help="debug: start index")
    cli = ap.parse_args(argv)
    if cli.variants[0] != "base":
        cli.variants = ["base"] + [v for v in cli.variants if v != "base"]
    entry = json.load(open(cli.index))["runs"][cli.key]
    started = time.time()
    res = run_one(cli.key, entry, cli)
    res["wall_s"] = round(time.time() - started, 1)
    out = pathlib.Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(_jsonable(res)))
    tmp.replace(out)
    print(f"{cli.key}: {len(res['chunks'])} chunks, {res['wall_s']} s -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
