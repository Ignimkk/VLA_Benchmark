"""LIBERO environment helpers for the KNOWS P0b probe.

Runs in the LIBERO venv (``src/openpi/examples/libero/.venv``), NOT the openpi venv -- LIBERO
pins numpy 1.22 / robosuite 1.4.1 / mujoco 3.2.3, which are incompatible with the JAX stack.

Required environment:
    LIBERO_CONFIG_PATH=benchmark/knows_vla/libero_config   # keeps ~/.libero untouched
    PYTHONPATH=$PWD/src/openpi/third_party/libero
    MUJOCO_GL=egl
"""

from __future__ import annotations

import pathlib

import numpy as np

# LIBERO ships its init-state files as pickled torch tensors. torch >= 2.6 defaults
# `weights_only=True`, which refuses the numpy globals inside them. These files come from the
# LIBERO repo we vendored, so loading them fully is safe here.
import torch as _torch

if not getattr(_torch.load, "_knows_patched", False):
    _orig_load = _torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _load._knows_patched = True
    _torch.load = _load


LIBERO_ENV_RESOLUTION = 256  # matches examples/libero/main.py -- the resolution training data used


def make_env(suite_name: str, task_id: int, *, seed: int = 0, resolution: int = LIBERO_ENV_RESOLUTION):
    """Create a LIBERO env with ground-truth instance segmentation and depth enabled.

    Mirrors examples/libero/main.py::_get_libero_env, plus the two extra observation channels the
    KNOWS pipeline needs: `camera_segmentations` supplies the object masks that Eq. (2) integrates
    attention over, and `camera_depths` supplies the back-projection input for ellipsoid fitting.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
        camera_segmentations="instance",
        camera_depths=True,
    )
    env.seed(seed)
    return env, task, task_suite


def instance_id_to_name(env) -> dict[int, str]:
    """Map rendered instance-segmentation values to object names.

    Exact inverse of robosuite's own encoding in
    ``robosuite/environments/robot_env.py::_create_segementation_sensor``::

        name2id = {inst: i for i, inst in enumerate(model.instances_to_ids.keys())}
        seg     = mapping.get(geom_id, -1) + 1

    so a rendered value ``v`` means ``instances_to_ids.keys()[v - 1]``, and ``v == 0`` is
    unmapped (background: table, walls, floor).
    """
    names = list(env.env.model.instances_to_ids.keys())
    mapping = {i + 1: name for i, name in enumerate(names)}
    mapping[0] = "__background__"
    return mapping


def parse_goal_objects(env) -> dict[str, list[str]]:
    """Read the task's goal predicates to recover which objects the task is actually about.

    LIBERO states goals in BDDL, e.g. ``(On akita_black_bowl_1 plate_1)``. That gives the
    manipulated object and its destination without any guessing from the language string -- which
    matters because several LIBERO scenes contain two instances of the same object type and the
    instruction disambiguates them only spatially.
    """
    problem = env.env.parsed_problem
    goal_states = problem.get("goal_state", [])
    manipulated: list[str] = []
    destinations: list[str] = []
    fixtures = set(problem.get("fixtures", {}))
    objects = set(problem.get("objects", {}))
    known = {n.lower() for n in list(fixtures) + list(objects)}

    for pred in goal_states:
        # pred looks like ['on', 'akita_black_bowl_1', 'plate_1'] (predicate first, then args)
        args = [a for a in pred[1:] if a.lower() in known or True]
        if len(args) >= 1:
            manipulated.append(args[0])
        if len(args) >= 2:
            destinations.append(args[1])
    return {"goal_predicates": goal_states, "manipulated": manipulated, "destinations": destinations}


def describe_observation(obs: dict) -> str:
    lines = []
    for k in sorted(obs):
        v = obs[k]
        if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 2:
            lines.append(f"  {k:42s} {str(v.shape):18s} {v.dtype} [{np.min(v)}, {np.max(v)}]")
    return "\n".join(lines)
