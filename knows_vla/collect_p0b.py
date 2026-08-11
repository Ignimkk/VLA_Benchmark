"""P0b stage 1 — roll out LIBERO under the real policy and dump what the probe needs.

Runs in the LIBERO venv and talks to an openpi policy server over websocket, exactly like
``examples/libero/main.py``. On top of the normal rollout it records, per control step, the
ground-truth instance segmentation and the BDDL-derived target labels, so stage 2 can score the
attention signal without any hand annotation.

Why a rollout and not stored frames: the paper's claim is about the object the policy *is
currently acting toward*. That only means something along a trajectory the policy itself drove.

Start the server first (openpi venv):
    uv run scripts/serve_policy.py --env LIBERO

Then (LIBERO venv):
    LIBERO_CONFIG_PATH=$PWD/benchmark/knows_vla/libero_config \
    PYTHONPATH=$PWD/src/openpi/third_party/libero:$PWD MUJOCO_GL=egl \
    src/openpi/examples/libero/.venv/bin/python -m benchmark.knows_vla.collect_p0b --max-steps 60
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import pathlib

import numpy as np

from benchmark.knows_vla.libero_env import (
    LIBERO_ENV_RESOLUTION,
    instance_id_to_name,
    make_env,
    parse_goal_objects,
)

# Instances that are never candidate objects for target identification: they are the robot itself.
ROBOT_INSTANCE_PREFIXES = ("MountedPanda", "RethinkMount", "PandaGripper", "Panda")

RESIZE = 224  # examples/libero/main.py::Args.resize_size -- what the policy actually sees
REPLAN_STEPS = 5  # examples/libero/main.py::Args.replan_steps
NUM_STEPS_WAIT = 10  # let objects settle; these steps are recorded but flagged


@dataclasses.dataclass
class Args:
    suite: str = "libero_spatial"
    task_ids: tuple[int, ...] = (0,)
    episodes: int = 1
    max_steps: int = 220
    seed: int = 0
    host: str = "0.0.0.0"
    port: int = 8000
    out: str = "data/knows_p0b"


def _is_robot(name: str) -> bool:
    return name.startswith(ROBOT_INSTANCE_PREFIXES)


def collect(args: Args) -> None:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as _wcp

    client = _wcp.WebsocketClientPolicy(args.host, args.port)
    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for task_id in args.task_ids:
        env, task, task_suite = make_env(args.suite, task_id, seed=args.seed)
        init_states = task_suite.get_task_init_states(task_id)
        id2name = instance_id_to_name(env)
        goal = parse_goal_objects(env)
        candidates = {i: n for i, n in id2name.items() if i != 0 and not _is_robot(n)}

        print(f"\n=== {args.suite} task {task_id}: {task.language}")
        print(f"    manipulated={goal['manipulated']}  destination={goal['destinations']}")
        print(f"    candidate objects: {sorted(candidates.values())}")

        for ep in range(args.episodes):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            action_plan: collections.deque = collections.deque()
            steps: list[dict] = []

            for t in range(args.max_steps + NUM_STEPS_WAIT):
                agent_rgb = obs["agentview_image"][::-1, ::-1]  # LIBERO renders upside-down
                wrist_rgb = obs["robot0_eye_in_hand_image"][::-1, ::-1]
                seg = obs["agentview_segmentation_instance"][::-1, ::-1, 0]
                depth = obs["agentview_depth"][::-1, ::-1, 0]

                if t < NUM_STEPS_WAIT:
                    # Dummy action from examples/libero/main.py: hold still, gripper open.
                    obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    continue

                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(agent_rgb, RESIZE, RESIZE))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_rgb, RESIZE, RESIZE))
                state = np.concatenate(
                    (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                )

                if not action_plan:
                    result = client.infer(
                        {
                            "observation/image": img,
                            "observation/wrist_image": wrist,
                            "observation/state": state,
                            "prompt": str(task.language),
                        }
                    )
                    action_plan.extend(result["actions"][:REPLAN_STEPS])
                    replanned = True
                else:
                    replanned = False

                action = action_plan.popleft()
                steps.append(
                    {
                        "t": t - NUM_STEPS_WAIT,
                        "image": img,                    # 224x224x3 uint8, exactly what the policy saw
                        "wrist_image": wrist,
                        "state": state.astype(np.float32),
                        "action": np.asarray(action, np.float32),
                        "seg_full": seg.astype(np.int16),  # 256x256, GT instance ids (pre-resize)
                        "depth_full": depth.astype(np.float32),
                        "replanned": replanned,
                    }
                )

                obs, _, done, _ = env.step(action.tolist())
                if done:
                    break

            out = out_root / f"{args.suite}_task{task_id}_ep{ep}.npz"
            np.savez_compressed(
                out,
                prompt=str(task.language),
                suite=args.suite,
                task_id=task_id,
                episode=ep,
                success=bool(done),
                id2name=np.array([f"{k}:{v}" for k, v in sorted(id2name.items())]),
                candidate_ids=np.array(sorted(candidates)),
                manipulated=np.array(goal["manipulated"]),
                destinations=np.array(goal["destinations"]),
                resize=RESIZE,
                env_resolution=LIBERO_ENV_RESOLUTION,
                **{k: np.stack([s[k] for s in steps]) for k in
                   ("t", "image", "wrist_image", "state", "action", "seg_full", "depth_full", "replanned")},
            )
            print(f"    ep{ep}: {len(steps)} steps, success={bool(done)} -> {out}")

        env.close()


def _quat2axisangle(quat):
    """Copied from examples/libero/main.py (which copied it from robosuite)."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default=Args.suite)
    p.add_argument("--task-ids", type=int, nargs="+", default=list(Args.task_ids))
    p.add_argument("--episodes", type=int, default=Args.episodes)
    p.add_argument("--max-steps", type=int, default=Args.max_steps)
    p.add_argument("--seed", type=int, default=Args.seed)
    p.add_argument("--host", default=Args.host)
    p.add_argument("--port", type=int, default=Args.port)
    p.add_argument("--out", default=Args.out)
    a = p.parse_args()
    collect(Args(a.suite, tuple(a.task_ids), a.episodes, a.max_steps, a.seed, a.host, a.port, a.out))


if __name__ == "__main__":
    main()
