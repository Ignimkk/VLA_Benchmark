"""Dump LIBERO agentview camera parameters, per task, for depth back-projection.

Needed by the perception stage (§3.2 of the paper: back-project the masked depth into 3D through
the known camera intrinsics and extrinsics). ``collect_p0b.py`` did not record these, and they are
cheap to recover -- creating the env is enough, no policy involved.

They are per-task, not global: the depth buffer is normalized and converting it to metres needs
``near``/``far``, which scale with ``sim.model.stat.extent`` -- a property of the scene, not the
camera. The agentview pose also differs between LIBERO scenes.

Run (LIBERO venv):
    LIBERO_CONFIG_PATH=$PWD/benchmark/knows_vla/libero_config \
    PYTHONPATH=$PWD/src/openpi/third_party/libero:$PWD MUJOCO_GL=egl \
    src/openpi/examples/libero/.venv/bin/python -m benchmark.knows_vla.dump_camera_params
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from benchmark.knows_vla.libero_env import LIBERO_ENV_RESOLUTION, make_env


def params_for(suite: str, task_id: int, resolution: int = LIBERO_ENV_RESOLUTION) -> dict:
    from robosuite.utils import camera_utils

    env, task, _ = make_env(suite, task_id, resolution=resolution)
    env.reset()
    sim = env.env.sim

    K = camera_utils.get_camera_intrinsic_matrix(sim, "agentview", resolution, resolution)
    T_cam_to_world = camera_utils.get_camera_extrinsic_matrix(sim, "agentview")
    extent = float(sim.model.stat.extent)
    out = {
        "suite": suite,
        "task_id": task_id,
        "language": str(task.language),
        "resolution": resolution,
        "intrinsic": np.asarray(K, float).tolist(),
        "cam_to_world": np.asarray(T_cam_to_world, float).tolist(),
        # get_real_depth_map: near / (1 - d * (1 - near/far))
        "znear": float(sim.model.vis.map.znear) * extent,
        "zfar": float(sim.model.vis.map.zfar) * extent,
        "extent": extent,
        # Our stored arrays are rotated 180 deg to match training preprocessing
        # (examples/libero/main.py), so pixel (r, c) here is (H-1-r, W-1-c) in camera coordinates.
        "stored_rot180": True,
    }
    env.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suites", nargs="+",
                   default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    p.add_argument("--task-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default="data/knows_p0b/camera_params.json")
    a = p.parse_args()

    table = {}
    for suite in a.suites:
        for tid in a.task_ids:
            try:
                info = params_for(suite, tid)
            except Exception as exc:  # a task id may not exist in a suite
                print(f"  skip {suite}/{tid}: {exc}")
                continue
            table[f"{suite}_task{tid}"] = info
            print(f"  {suite}_task{tid}: znear={info['znear']:.4f} zfar={info['zfar']:.2f} "
                  f"fx={info['intrinsic'][0][0]:.1f}")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    print(f"\nwrote {len(table)} entries -> {out}")


if __name__ == "__main__":
    main()
