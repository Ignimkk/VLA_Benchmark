"""Do the D2/D3 changes actually change behaviour? — A/B over recorded LIBERO rollouts.

The unit tests say the maths is right. They cannot say whether the epsilon loophole matters on real
trajectories, which is the claim docs/15-development-plan.md makes. This replays the recorded
episodes -- no policy server, no simulator, no checkpoint -- and runs every filter variant over
*identical* perception output, so differences are attributable to the filter alone.

The metric that matters is `cbf_viol`. Every variant satisfies its own linearized constraint by
construction; the question is whether the step it returns satisfies the discrete CBF condition on
the **true** barrier,

    h*(after) >= (1 - gamma_h) h*(before),        h*(E) = max_n h(n)

which is what "safe" was supposed to mean. A filter can pass Eq. (8) and fail this, by tilting the
separating plane instead of moving the robot -- exactly what the epsilon box permits.

    src/openpi/.venv/bin/python -m benchmark.knows_vla.compare_variants --limit 6
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time

import numpy as np

from benchmark.knows_vla.cbf.ellipsoid import Ellipsoid, barrier, optimal_normal
from benchmark.knows_vla.cbf.filter import CbfParams, RobotBody, SafetyFilter
from benchmark.knows_vla.perception.held import HeldObjectTracker, promote
from benchmark.knows_vla.offline_filter import (
    DEFAULT_GRIPPER_SEMI_AXES,
    GRIPPER_TOKEN,
    POS_SCALE,
    ROBOT_TOKENS,
    ROT_SCALE,
    _axisangle_to_mat,
    gripper_ellipsoid_shape,
)
from benchmark.knows_vla.perception.ellipsoid_fit import CameraModel, ObjectTracker, fit_objects

# The OSC action limit. Used both as the delta_c bound and as the yardstick for "how big is this
# correction", since a correction larger than the controller can execute is not a correction.
OSC_LIMIT = POS_SCALE


def variants() -> dict[str, CbfParams]:
    """Eq. (12) as written, then one change at a time so each effect is attributable."""
    paper = CbfParams()  # relaxed normals, eps=0.05, no bound on delta_c, no slack
    return {
        "paper": paper,
        "paper+limit": dataclasses.replace(paper, max_delta_pos=OSC_LIMIT, max_delta_rot=0.5),
        "eps=0": dataclasses.replace(paper, eps_normal=0.0),
        "fixed": dataclasses.replace(paper, normals="fixed"),
        "fixed+limit": dataclasses.replace(paper, normals="fixed",
                                           max_delta_pos=OSC_LIMIT, max_delta_rot=0.5),
        "fixed+limit+slack": dataclasses.replace(paper, normals="fixed", max_delta_pos=OSC_LIMIT,
                                                 max_delta_rot=0.5, slack_weight=1e3),
        "fixedpoint+limit+slack": dataclasses.replace(paper, normals="fixedpoint",
                                                      max_delta_pos=OSC_LIMIT, max_delta_rot=0.5,
                                                      slack_weight=1e3),
    }


def perceive(path: pathlib.Path, cams: dict, semi_axes, promote_held: bool = False) -> list[dict]:
    """Everything upstream of the QP, once: obstacles, robot ellipsoid and nominal action per step.

    Doing this per variant instead would dominate the runtime and, worse, let perception noise
    differ between arms of the comparison.
    """
    d = np.load(path, allow_pickle=True)
    key = f"{d['suite']}_task{int(d['task_id'])}"
    if key not in cams:
        raise KeyError(f"no camera params for {key}")
    cam = CameraModel.from_json_entry(cams[key])

    id2name = {int(s.split(":", 1)[0]): s.split(":", 1)[1] for s in d["id2name"]}
    obj_ids = [i for i, n in id2name.items() if i != 0 and not any(t in n for t in ROBOT_TOKENS)]
    gid = next((i for i, n in id2name.items() if GRIPPER_TOKEN in n), None)

    shapes = fit_objects(d["seg_full"][0], d["depth_full"][0], cam, obj_ids)
    tracker = ObjectTracker(cam, shapes)
    Q_local = gripper_ellipsoid_shape(d["seg_full"][0], d["depth_full"][0], cam, gid,
                                      d["state"][0, 3:6], semi_axes)

    def _resolve(nm):
        cands = [i for i in obj_ids if str(nm).startswith(id2name[i])]
        return max(cands, key=lambda i: len(id2name[i])) if cands else None

    reach = {j for n in d["manipulated"] if (j := _resolve(n)) is not None}
    place = {j for n in d["destinations"] if (j := _resolve(n)) is not None} or reach
    closed = np.flatnonzero(d["action"][:, 6] > 0)
    grasp_t = int(closed[0]) if closed.size else len(d["action"])

    steps = []
    held_trk = HeldObjectTracker() if promote_held else None
    n_held = 0
    for t in range(len(d["t"])):
        obstacles = tracker.update(d["seg_full"][t], d["depth_full"][t])
        tgt = reach if t < grasp_t else place  # privileged target, as in P2b/P3a
        obstacles = {k: v for k, v in obstacles.items() if k not in tgt}
        R_t = _axisangle_to_mat(d["state"][t, 3:6])
        robot = Ellipsoid(d["state"][t, 0:3], R_t @ Q_local @ R_t.T)
        if held_trk is not None:
            key = held_trk.update(robot.c, bool(d["action"][t, 6] > 0), obstacles)
            robot, obstacles = promote(RobotBody.single(robot), obstacles, key, robot.c)
            n_held += key is not None
        # The pre-step barrier is a property of the scene, not of the filter, so compute it once
        # here and let every variant share it -- and reuse its normals to warm-start the post-step
        # evaluation, which is otherwise the dominant cost of this script.
        h_pre, n_pre = true_min_barrier(robot, obstacles)
        steps.append({
            "robot": robot,
            "obstacles": obstacles,
            "dc_nom": d["action"][t, 0:3].astype(np.float64) * POS_SCALE,
            "dth_nom": d["action"][t, 3:6].astype(np.float64) * ROT_SCALE,
            "h_pre": h_pre,
            "n_pre": n_pre,
        })
    return steps


def _parts(robot) -> dict:
    """Uniform view over a bare ellipsoid and a multi-part body."""
    if isinstance(robot, RobotBody):
        return {k: robot.world(k) for k in robot.parts}
    return {"eef": robot}


def _moved(robot, delta_c: np.ndarray):
    if isinstance(robot, RobotBody):
        return RobotBody(robot.origin + delta_c, robot.parts)
    return Ellipsoid(robot.c + delta_c, robot.Q)


def true_min_barrier(robot, obstacles: dict, init: dict | None = None):
    """(min barrier over obstacles at each one's *optimal* normal, the normals used).

    The honest gap: unlike the filter's own linearized rows, this maximises over n, so it is what
    "are they actually separated" means.
    """
    if not obstacles:
        return np.inf, {}
    normals, best = {}, np.inf
    for pname, part in _parts(robot).items():
        for k, o in obstacles.items():
            key = (pname, k)
            n = optimal_normal(part, o, iters=400 if init is None else 80,
                               init=None if init is None else init.get(key))
            normals[key] = n
            best = min(best, barrier(n, part, o))
    return best, normals


def run_variant(steps: list[dict], params: CbfParams) -> dict:
    filt = SafetyFilter(params)
    rec = {k: [] for k in ("dpos", "stop", "h_pre", "cbf_viol", "ms")}
    for s in steps:
        t0 = time.perf_counter()
        res = filt(s["robot"], s["obstacles"], s["dc_nom"], s["dth_nom"])
        rec["ms"].append((time.perf_counter() - t0) * 1e3)

        h_pre = s["h_pre"]
        after = _moved(s["robot"], res.delta_c)
        h_post, _ = true_min_barrier(after, s["obstacles"], init=s["n_pre"])
        # Discrete CBF condition on the true barrier, with a 1 mm tolerance for the linearization.
        viol = np.isfinite(h_pre) and h_post < (1.0 - params.gamma_h) * h_pre - 1e-3

        rec["dpos"].append(float(np.linalg.norm(res.delta_c - s["dc_nom"])))
        rec["stop"].append(bool(res.emergency_stop))
        rec["h_pre"].append(h_pre)
        rec["cbf_viol"].append(bool(viol))
    return {k: np.asarray(v) for k, v in rec.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", nargs="+", default=sorted(
        str(q) for q in pathlib.Path("data/knows_p0b").glob("*.npz")))
    p.add_argument("--camera-params", default="benchmark/knows_vla/camera_params.json")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", nargs="*", help="subset of variant names")
    p.add_argument("--gripper-semi-axes", type=float, nargs=3, default=list(DEFAULT_GRIPPER_SEMI_AXES))
    p.add_argument("--promote-held", action="store_true",
                   help="A3: move the object in the hand from the obstacle set into the robot body")
    a = p.parse_args()

    cams = json.loads(pathlib.Path(a.camera_params).read_text())
    paths = [pathlib.Path(x) for x in a.episodes][: a.limit or None]
    chosen = {k: v for k, v in variants().items() if not a.only or k in a.only}

    acc: dict[str, list[dict]] = {k: [] for k in chosen}
    for i, path in enumerate(paths, 1):
        try:
            steps = perceive(path, cams, tuple(a.gripper_semi_axes), a.promote_held)
        except KeyError as exc:
            print(f"  skip {path.name}: {exc}", flush=True)
            continue
        for name, params in chosen.items():
            acc[name].append(run_variant(steps, params))
        print(f"  {i}/{len(paths)}  {path.name}  ({len(steps)} steps)", flush=True)

    print("\n" + "=" * 100)
    print(f"{'variant':<24}{'개입률':>8}{'보정 p95':>10}{'보정 max':>10}"
          f"{'e-stop':>8}{'CBF 위반':>10}{'h<0':>8}{'QP ms':>8}")
    print("-" * 100)
    for name in chosen:
        if not acc[name]:
            continue
        d = {k: np.concatenate([r[k] for r in acc[name]]) for k in acc[name][0]}
        finite = d["h_pre"][np.isfinite(d["h_pre"])]
        print(f"{name:<24}{(d['dpos'] > 1e-6).mean() * 100:7.1f}%"
              f"{np.percentile(d['dpos'], 95) * 100:9.2f}cm"
              f"{d['dpos'].max() * 100:9.1f}cm"
              f"{d['stop'].mean() * 100:7.1f}%"
              f"{d['cbf_viol'].mean() * 100:9.1f}%"
              f"{(finite < 0).mean() * 100:7.1f}%"
              f"{d['ms'].mean():8.2f}")
    print("=" * 100)
    print("개입률 = 명목 액션이 바뀐 스텝 비율 | 보정 = |delta_c - nominal|, OSC 한계는 5 cm")
    print("CBF 위반 = 실제 배리어가 h_post >= (1-gamma) h_pre 를 어긴 스텝 —"
          " 선형화된 제약은 만족해도 진짜 조건은 깨질 수 있다")
    print("h<0 = 필터를 걸기 전 이미 겹쳐 있던 스텝 (OPEN-Q 19: 쥔 물체가 장애물로 남는 문제)")


if __name__ == "__main__":
    main()
