"""In-process LIBERO rollout shared by the baseline/SEAM evaluators.

This mirrors ``examples/libero/main.py`` but runs the policy in-process (no websocket) so SEAM has
direct access to the model-space chunk and per-episode state. It imports ``libero`` lazily, so this
module can be imported even where the LIBERO simulator is not installed (e.g. for --help / smoke).

The real environment rollout requires the LIBERO container dependencies (``libero``, ``robosuite``,
``mujoco``); see ``scripts/SETUP_CONTAINER.md``.
"""

from __future__ import annotations

import collections
import dataclasses
import math
from typing import Any, Callable

import numpy as np

LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_STEPS_WAIT = 10

_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def quat2axisangle(quat):
    """robosuite-compatible quaternion -> axis-angle (matches examples/libero/main.py)."""
    quat = np.asarray(quat)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def make_env(task, resolution, seed):
    """Build a LIBERO OffScreenRenderEnv (lazy import of libero)."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import os

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def build_observation(obs, task_description, resize_size=224):
    """Convert a LIBERO obs dict into the policy input dict (matches main.py preprocessing)."""
    from openpi_client import image_tools

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": str(task_description),
    }


@dataclasses.dataclass
class EpisodeResult:
    success: bool
    executed_actions: np.ndarray  # [T, 7]
    num_chunks: int
    per_chunk_correction_norm: list
    episode_length: int


def run_episode(
    env,
    task_description,
    predict_chunk_fn: Callable[[dict], tuple],
    execution_length: int,
    max_steps: int,
    init_state=None,
) -> EpisodeResult:
    """Run one episode. ``predict_chunk_fn(obs_dict) -> (physical_chunk[H,7], diagnostics)``."""
    from benchmark.seam_vla.rollout.chunk_executor import ChunkExecutor

    obs = env.reset()
    if init_state is not None:
        obs = env.set_init_state(init_state)

    executor = ChunkExecutor(predict_chunk_fn, execution_length)
    executed = []
    corr_norms = []
    success = False
    t = 0
    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue
        obs_dict = build_observation(obs, task_description)
        action = executor.act(obs_dict)
        if executor.last_diagnostics is not None and executor.num_chunks > len(corr_norms):
            corr_norms.append(float(getattr(executor.last_diagnostics, "correction_norm", 0.0)))
        executed.append(np.asarray(action, dtype=np.float32))
        obs, _, done, _ = env.step(np.asarray(action).tolist())
        t += 1
        if done:
            success = True
            break

    return EpisodeResult(
        success=success,
        executed_actions=np.asarray(executed, dtype=np.float32),
        num_chunks=executor.num_chunks,
        per_chunk_correction_norm=corr_norms,
        episode_length=len(executed),
    )


def max_steps_for(task_suite_name: str) -> int:
    return _MAX_STEPS.get(task_suite_name, 520)
