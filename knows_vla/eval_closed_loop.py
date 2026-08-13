"""P3a — closed-loop LIBERO with the safety filter actually in the control loop.

Unlike ``offline_filter.py``, the filtered action is what gets executed, so this is the first run
where the filter changes the trajectory. Two conditions on identical seeds and initial states:

    --filter off   the stock policy (baseline)
    --filter on    the same policy with the CBF-QP applied to every action

Target source is the privileged simulator state, not attention. That deliberately isolates filter
viability from target-identification error, and mirrors the oracle-target condition the paper uses
for its Naive baseline. Attention in the loop is P3b and needs the policy server to return it.

Implements Eq. (12) **literally** -- no bound on delta_c. See docs/11-p2b-results.md for why that
matters; ``--max-delta-pos`` opts into the non-paper action limit for comparison.

Server (openpi venv, GPU box):
    .venv/bin/python scripts/serve_policy.py --env LIBERO

Client (LIBERO venv):
    LIBERO_CONFIG_PATH=$PWD/benchmark/knows_vla/libero_config \
    PYTHONPATH=$PWD/src/openpi/third_party/libero:$PWD MUJOCO_GL=egl \
    src/openpi/examples/libero/.venv/bin/python -m benchmark.knows_vla.eval_closed_loop \
        --filter off on --suite libero_spatial --task-ids 0 1 --episodes 3 --save-video
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from benchmark.knows_vla.cbf.ellipsoid import Ellipsoid
from benchmark.knows_vla.cbf.filter import CbfParams, SafetyFilter
from benchmark.knows_vla.libero_env import instance_id_to_name, make_env, parse_goal_objects
from benchmark.knows_vla.offline_filter import DEFAULT_GRIPPER_SEMI_AXES, gripper_ellipsoid_shape
from benchmark.knows_vla.perception.ellipsoid_fit import CameraModel, ObjectTracker, fit_objects

POS_SCALE, ROT_SCALE = 0.05, 0.5
ROBOT_TOKENS = ("Panda", "Mount", "Gripper", "Robot")
RESIZE, REPLAN_STEPS, NUM_STEPS_WAIT = 224, 5, 10
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def _axisangle_to_mat(v):
    v = np.asarray(v, np.float64).reshape(3)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _quat2axisangle(quat):
    quat = np.asarray(quat, np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    return np.zeros(3) if np.isclose(den, 0.0) else (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def run_condition(args, use_filter: bool, cams: dict) -> dict:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as _wcp

    client = _wcp.WebsocketClientPolicy(args.host, args.port)
    params = CbfParams(gamma_h=args.gamma_h, eps_normal=args.eps_normal, W=args.w,
                       max_delta_pos=args.max_delta_pos)
    video_dir = pathlib.Path(args.video_out)
    video_dir.mkdir(parents=True, exist_ok=True)

    stats = collections.defaultdict(list)
    stats_minh_name: list[str] = []
    for task_id in args.task_ids:
        env, task, suite = make_env(args.suite, task_id, seed=args.seed)
        init_states = suite.get_task_init_states(task_id)
        id2name = instance_id_to_name(env)
        obj_ids = [i for i, n in id2name.items() if i != 0 and not any(t in n for t in ROBOT_TOKENS)]
        gid = next((i for i, n in id2name.items() if "Gripper" in n), None)
        goal = parse_goal_objects(env)
        cam = CameraModel.from_json_entry(cams[f"{args.suite}_task{task_id}"])

        def _resolve(nm, _ids=obj_ids, _n=id2name):
            c = [i for i in _ids if str(nm).startswith(_n[i])]
            return max(c, key=lambda i: len(_n[i])) if c else None

        reach = {j for n in goal["manipulated"] if (j := _resolve(n)) is not None}
        place = {j for n in goal["destinations"] if (j := _resolve(n)) is not None} or reach
        max_steps = MAX_STEPS.get(args.suite, 300)

        for ep in range(args.episodes):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            plan: collections.deque = collections.deque()
            filt = SafetyFilter(params)
            tracker, Q_local = None, None
            frames, grasped, n_stop, n_touch, done = [], False, 0, 0, False
            t = NUM_STEPS_WAIT

            for t in range(max_steps + NUM_STEPS_WAIT):
                if t < NUM_STEPS_WAIT:
                    obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    continue

                seg = obs["agentview_segmentation_instance"][::-1, ::-1, 0]
                dep = obs["agentview_depth"][::-1, ::-1, 0]
                rgb = obs["agentview_image"][::-1, ::-1]
                if tracker is None:  # freeze shapes at the first controlled step (Sec. 3.2)
                    tracker = ObjectTracker(cam, fit_objects(seg, dep, cam, obj_ids))
                    Q_local = gripper_ellipsoid_shape(
                        seg, dep, cam, gid, _quat2axisangle(obs["robot0_eef_quat"]), args.semi_axes)

                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, RESIZE, RESIZE))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1], RESIZE, RESIZE))
                state = np.concatenate((obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"]))
                if not plan:
                    r = client.infer({"observation/image": img, "observation/wrist_image": wrist,
                                      "observation/state": state, "prompt": str(task.language)})
                    plan.extend(r["actions"][:REPLAN_STEPS])
                action = np.asarray(plan.popleft(), np.float64)

                if use_filter:
                    tracked = tracker.update(seg, dep)
                    tgt = place if grasped else reach
                    obstacles = {k: v for k, v in tracked.items() if k not in tgt}
                    if args.no_obstacles:  # plumbing control: the filter must become the identity
                        obstacles = {}
                    if obstacles:
                        j = min(obstacles, key=lambda k: float(np.linalg.norm(state[0:3] - obstacles[k].c)))
                        stats_minh_name.append(id2name[j])
                    R_t = _axisangle_to_mat(state[3:6])
                    robot = Ellipsoid(state[0:3], R_t @ Q_local @ R_t.T)
                    res = filt(robot, obstacles, action[0:3] * POS_SCALE, action[3:6] * ROT_SCALE)
                    n_stop += int(res.emergency_stop)
                    n_touch += int(np.linalg.norm(res.delta_c - action[0:3] * POS_SCALE) > 1e-6)
                    # back to action units, clipped to the controller's input range
                    action = np.concatenate([np.clip(res.delta_c / POS_SCALE, -1, 1),
                                             np.clip(res.delta_theta / ROT_SCALE, -1, 1),
                                             action[6:7]])
                if action[6] > 0:
                    grasped = True
                if args.save_video:
                    frames.append(img)

                obs, _, done, _ = env.step(action.tolist())
                if done:
                    break

            tag = "on" if use_filter else "off"
            n = max(t - NUM_STEPS_WAIT + 1, 1)
            stats["success"].append(bool(done))
            stats["steps"].append(n)
            stats["stop_rate"].append(n_stop / n)
            stats["touch_rate"].append(n_touch / n)
            print(f"    [{tag}] task{task_id} ep{ep}: success={bool(done)} steps={n}"
                  f" stop={stats['stop_rate'][-1]:.1%} intervene={stats['touch_rate'][-1]:.1%}",
                  flush=True)
            if args.save_video and frames:
                import imageio.v2 as imageio
                imageio.mimwrite(video_dir / f"{args.suite}_task{task_id}_ep{ep}_filter-{tag}.mp4",
                                 frames, fps=20, quality=8, macro_block_size=1)
        env.close()
    return {k: np.asarray(v) for k, v in stats.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-ids", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--filter", nargs="+", default=["off", "on"], choices=["off", "on"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--camera-params", default="benchmark/knows_vla/camera_params.json")
    p.add_argument("--gamma-h", type=float, default=0.5)
    p.add_argument("--eps-normal", type=float, default=0.05)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--max-delta-pos", type=float, default=None, help="NOT in Eq. (12)")
    p.add_argument("--gripper-semi-axes", type=float, nargs=3, default=list(DEFAULT_GRIPPER_SEMI_AXES),
                   help="EEF ellipsoid semi-axes in m; the paper only says 'calibrated offline'")
    p.add_argument("--gripper-from-mask", action="store_true",
                   help="fit Q_R from the PandaGripper mask instead (77 cm long; fails, see P3a)")
    p.add_argument("--no-obstacles", action="store_true",
                   help="control: filter on but obstacle set empty -- must reproduce the baseline")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-out", default="data/knows_videos/closed_loop")
    a = p.parse_args()

    a.semi_axes = None if a.gripper_from_mask else tuple(a.gripper_semi_axes)
    cams = json.loads(pathlib.Path(a.camera_params).read_text())
    summary = {}
    for cond in a.filter:
        print(f"\n=== filter {cond} ===", flush=True)
        summary[cond] = run_condition(a, cond == "on", cams)

    print("\n" + "=" * 66)
    print(f"{'condition':<10} {'success':>9} {'steps':>8} {'e-stop':>9} {'intervene':>10}")
    print("=" * 66)
    for cond, s in summary.items():
        print(f"{cond:<10} {s['success'].mean() * 100:8.1f}% {s['steps'].mean():8.1f}"
              f" {s['stop_rate'].mean() * 100:8.1f}% {s['touch_rate'].mean() * 100:9.1f}%")
    print(f"\n(n = {len(next(iter(summary.values()))['success'])} episodes per condition)")
    print(f"gripper semi-axes: {'from mask' if a.gripper_from_mask else a.gripper_semi_axes}"
          f"{'   [no-obstacles control]' if a.no_obstacles else ''}")


if __name__ == "__main__":
    main()
