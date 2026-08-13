"""P2b — run the KNOWS safety filter over recorded episodes, offline.

Replays a collected rollout and, at every step, builds the obstacle set and pushes the policy's
nominal action through the CBF-QP. Nothing is fed back into the simulator, so the trajectory is
unchanged; what we measure is how much the filter *would* intervene and whether that intervention
is sensible.

Target source is a switch so attention error and filter behaviour stay separable:
  gt        -- exclude the BDDL phase-appropriate object (perfect target identification)
  none      -- exclude nothing, every object is an obstacle (what Eq. (4) does when the top-1/top-2
               gap falls below delta, i.e. the conservative fallback)
  keep-all  -- like `none`, but also keeps the target: the upper bound on how conservative the
               filter can get

    src/openpi/.venv/bin/python -m benchmark.knows_vla.offline_filter
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from benchmark.knows_vla.cbf.ellipsoid import Ellipsoid, barrier, rotate_shape
from benchmark.knows_vla.cbf.filter import CbfParams, SafetyFilter
from benchmark.knows_vla.perception.ellipsoid_fit import CameraModel, ObjectTracker, fit_objects

# robosuite OSC_POSE, control_delta=true, input_max=1:
#   output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
# so the policy's [-1, 1] action maps to metres / radians by these factors.
# Confirmed in docs/10-p2-perception.md §4.
POS_SCALE = 0.05
ROT_SCALE = 0.5

# Franka hand only -- not the wrist link. See gripper_ellipsoid_shape. [ASSUMPTION]
DEFAULT_GRIPPER_SEMI_AXES = (0.04, 0.04, 0.07)

ROBOT_TOKENS = ("Panda", "Mount", "Gripper", "Robot")
GRIPPER_TOKEN = "Gripper"


def _axisangle_to_mat(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def gripper_ellipsoid_shape(d_seg, d_dep, cam, gid, eef_axisangle, semi_axes=None):
    """Q_R in the gripper's own frame.

    The paper only says the end-effector semi-axes are "calibrated offline" and gives no procedure
    (OPEN-QUESTIONS #3). Fitting the `PandaGripper0` segmentation mask -- the obvious data-driven
    stand-in -- is WRONG: that instance covers the whole hand-plus-wrist assembly and yields a
    7.2 x 12.7 x 38.5 cm ellipsoid, i.e. 77 cm long, while LIBERO tabletop objects sit 13-18 cm
    apart. The barrier is then negative everywhere and the filter simply flees (P3a: 0/9 success).

    So default to an explicit calibration of the Franka hand itself. Still an [ASSUMPTION], but a
    physically defensible one, and `--gripper-from-mask` keeps the old behaviour for comparison.
    """
    if semi_axes is not None:
        return np.diag(np.asarray(semi_axes, np.float64) ** 2)
    fits = fit_objects(d_seg, d_dep, cam, [gid]) if gid is not None else {}
    if gid not in fits:
        return np.eye(3) * 0.04**2
    R0 = _axisangle_to_mat(eef_axisangle)
    return R0.T @ fits[gid].Q @ R0


def run_episode(path: pathlib.Path, cams: dict, params: CbfParams, target_mode: str,
                semi_axes=DEFAULT_GRIPPER_SEMI_AXES) -> dict:
    d = np.load(path, allow_pickle=True)
    key = f"{d['suite']}_task{int(d['task_id'])}"
    if key not in cams:
        raise KeyError(f"no camera params for {key}")
    cam = CameraModel.from_json_entry(cams[key])

    id2name = {int(s.split(":", 1)[0]): s.split(":", 1)[1] for s in d["id2name"]}
    obj_ids = [i for i, n in id2name.items() if i != 0 and not any(t in n for t in ROBOT_TOKENS)]
    gid = next((i for i, n in id2name.items() if GRIPPER_TOKEN in n), None)

    # Shapes frozen at t=0 (Sec. 3.2); centroids tracked per step.
    shapes = fit_objects(d["seg_full"][0], d["depth_full"][0], cam, obj_ids)
    tracker = ObjectTracker(cam, shapes)
    Q_local = gripper_ellipsoid_shape(d["seg_full"][0], d["depth_full"][0], cam, gid,
                                      d["state"][0, 3:6], semi_axes)

    # Phase-appropriate ground-truth target, same rule as probe_p0b.
    def _resolve(nm):
        nm = str(nm)
        cands = [i for i in obj_ids if nm.startswith(id2name[i])]
        return max(cands, key=lambda i: len(id2name[i])) if cands else None

    reach = {j for n in d["manipulated"] if (j := _resolve(n)) is not None}
    place = {j for n in d["destinations"] if (j := _resolve(n)) is not None} or reach
    grip = d["action"][:, 6]
    closed = np.flatnonzero(grip > 0)
    grasp_t = int(closed[0]) if closed.size else len(grip)

    filt = SafetyFilter(params)
    rec = {k: [] for k in ("dpos", "drot", "stop", "min_h", "n_obs", "infeasible")}

    for t in range(len(d["t"])):
        obstacles = tracker.update(d["seg_full"][t], d["depth_full"][t])
        if target_mode == "gt":
            tgt = reach if t < grasp_t else place
            obstacles = {k: v for k, v in obstacles.items() if k not in tgt}
        elif target_mode in ("none", "keep-all"):
            pass
        else:
            raise ValueError(target_mode)

        R_t = _axisangle_to_mat(d["state"][t, 3:6])
        robot = Ellipsoid(d["state"][t, 0:3], R_t @ Q_local @ R_t.T)

        dc_nom = d["action"][t, 0:3].astype(np.float64) * POS_SCALE
        dth_nom = d["action"][t, 3:6].astype(np.float64) * ROT_SCALE
        res = filt(robot, obstacles, dc_nom, dth_nom)

        rec["dpos"].append(float(np.linalg.norm(res.delta_c - dc_nom)))
        rec["drot"].append(float(np.linalg.norm(res.delta_theta - dth_nom)))
        rec["stop"].append(bool(res.emergency_stop))
        rec["infeasible"].append(not res.feasible)
        rec["n_obs"].append(len(obstacles))
        rec["min_h"].append(float(res.h.min()) if res.h.size else np.inf)

    out = {k: np.asarray(v) for k, v in rec.items()}
    out["name"] = path.name
    out["n_steps"] = len(d["t"])
    return out


def summarize(results: list[dict], target_mode: str, params: CbfParams) -> None:
    dpos = np.concatenate([r["dpos"] for r in results])
    drot = np.concatenate([r["drot"] for r in results])
    stop = np.concatenate([r["stop"] for r in results])
    minh = np.concatenate([r["min_h"] for r in results])
    nobs = np.concatenate([r["n_obs"] for r in results])
    touched = dpos > 1e-6

    lim = ("unbounded (Eq. 12 literal)" if params.max_delta_pos is None
           else f"|delta_c|<={params.max_delta_pos} m")
    print(f"\n--- target={target_mode}  gamma_h={params.gamma_h} eps={params.eps_normal} "
          f"W={params.W}  action limit: {lim} ---")
    print(f"  episodes {len(results)}, steps {len(dpos)}, obstacles/step {nobs.mean():.1f}")
    print(f"  filter intervened      {touched.mean() * 100:5.1f}% of steps")
    print(f"  |d(delta_c)|  (cm)     mean {dpos.mean() * 100:6.3f}  p50 {np.median(dpos) * 100:6.3f}"
          f"  p95 {np.percentile(dpos, 95) * 100:6.3f}  max {dpos.max() * 100:6.3f}")
    print(f"  |d(delta_th)| (rad)    mean {drot.mean():6.4f}  p95 {np.percentile(drot, 95):6.4f}")
    print(f"  emergency stops        {stop.mean() * 100:5.1f}%")
    finite = minh[np.isfinite(minh)]
    if finite.size:
        print(f"  min h (cm)             p5 {np.percentile(finite, 5) * 100:6.2f}"
              f"  p50 {np.median(finite) * 100:6.2f}   h<0 in {(finite < 0).mean() * 100:4.1f}% of steps")
    # A 5 cm nominal step is the OSC limit, so express intervention against that.
    print(f"  intervention vs the {POS_SCALE * 100:.0f} cm action limit: "
          f"p95 = {np.percentile(dpos, 95) / POS_SCALE * 100:.1f}%")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", nargs="+", default=sorted(
        str(q) for q in pathlib.Path("data/knows_p0b").glob("*.npz")))
    p.add_argument("--camera-params", default="benchmark/knows_vla/camera_params.json")
    p.add_argument("--target", nargs="+", default=["gt", "none"])
    p.add_argument("--gamma-h", type=float, default=0.5)
    p.add_argument("--eps-normal", type=float, default=0.05)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0, help="use only the first N episodes")
    p.add_argument("--max-delta-pos", type=float, default=None,
                   help="NOT in Eq. (12): bound |delta_c|_inf, e.g. 0.05 for the OSC limit")
    p.add_argument("--max-delta-rot", type=float, default=None, help="NOT in Eq. (12)")
    p.add_argument("--gripper-semi-axes", type=float, nargs=3, default=list(DEFAULT_GRIPPER_SEMI_AXES))
    p.add_argument("--gripper-from-mask", action="store_true",
                   help="fit Q_R from the PandaGripper mask (too large; see P3a)")
    a = p.parse_args()

    cams = json.loads(pathlib.Path(a.camera_params).read_text())
    eps = [pathlib.Path(x) for x in a.episodes]
    if a.limit:
        eps = eps[: a.limit]
    params = CbfParams(gamma_h=a.gamma_h, W=a.w, eps_normal=a.eps_normal,
                       max_delta_pos=a.max_delta_pos, max_delta_rot=a.max_delta_rot)
    semi_axes = None if a.gripper_from_mask else tuple(a.gripper_semi_axes)

    print("=" * 78)
    print(f"P2b offline filter — {len(eps)} episodes")
    print("=" * 78)
    for mode in a.target:
        results = []
        print(f"\ntarget={mode}", flush=True)
        for i, path in enumerate(eps, 1):
            try:
                results.append(run_episode(path, cams, params, mode, semi_axes))
            except KeyError as exc:
                print(f"  skip {path.name}: {exc}", flush=True)
            if i % 5 == 0 or i == len(eps):
                print(f"  ... {i}/{len(eps)} episodes", flush=True)
        if results:
            summarize(results, mode, params)


if __name__ == "__main__":
    main()
