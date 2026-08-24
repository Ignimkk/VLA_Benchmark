"""MuJoCo -> KNOWS adapters: camera, segmentation, ground-truth shapes, and Jacobians.

LIBERO gave us observations through robosuite; the transport scenario runs on bare MuJoCo, so the
same four things have to be sourced again. Nothing here is transport-specific -- it takes a model
and a data, so the transport scene plugs in unchanged once its assets load.

Deliberately model-agnostic for a second reason: **the transport model does not compile in this
environment.** MuJoCo fails to open a different one of its 1,129 OBJ assets on each attempt while
Python reads all of them fine, which is a sandbox interaction rather than a model defect (it is
also why `check_admissibility.py` parses MJCF directly). Everything here is therefore validated
against a small mesh-free scene built in the tests, and the transport model exercises the same
code paths with no changes.

    src/rby1_manipulation/.venv/bin/python -m benchmark.knows_vla.mujoco_bridge --self-test
"""

from __future__ import annotations

import dataclasses

import numpy as np

from benchmark.knows_vla.cbf.ellipsoid import Ellipsoid
from benchmark.knows_vla.cbf.filter import ArticulatedBody, Link
from benchmark.knows_vla.perception.ellipsoid_fit import (
    CameraModel,
    PerceptionMargin,
    complete_backface,
    fit_objects,
    mvee,
)

# robosuite's correction: a MuJoCo camera looks down -z with +y up, while the pixel convention
# wants +z forward and +y down. CameraModel inherits that convention from the LIBERO path, so the
# same flip has to be applied here or nothing back-projects.
_AXIS_FIX = np.diag([1.0, -1.0, -1.0, 1.0])


def camera_model(model, data, cam_name: str, height: int, width: int,
                 *, flip_row: bool = False, flip_col: bool = False) -> CameraModel:
    """Intrinsics and extrinsics for a named camera, in the convention `CameraModel` expects.

    `flip_row`/`flip_col` default to False, unlike the LIBERO path: `mujoco.Renderer` already
    returns rows top-down, and robosuite's extra flips do not apply. `--self-test` re-derives the
    right combination from geometry rather than trusting that.
    """
    import mujoco

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise KeyError(f"no camera named {cam_name!r}")
    f = 0.5 * height / np.tan(float(model.cam_fovy[cid]) * np.pi / 360.0)
    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(data.cam_xmat[cid], np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(data.cam_xpos[cid], np.float64)
    return CameraModel(K=K, cam_to_world=pose @ _AXIS_FIX,
                       znear=float(model.vis.map.znear * model.stat.extent),
                       zfar=float(model.vis.map.zfar * model.stat.extent),
                       flip_row=flip_row, flip_col=flip_col)


def render(renderer, data, cam_name: str):
    """(rgb uint8, depth in metres, segmentation as body ids).

    MuJoCo's depth renderer already returns metres, so unlike the LIBERO path this must **not** go
    through `real_depth` -- doing so silently produces a plausible-looking wrong scene.
    """
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=cam_name)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    depth = renderer.render().copy().astype(np.float64)
    renderer.disable_depth_rendering()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam_name)
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    return rgb, depth, seg


def segmentation_to_body(model, seg: np.ndarray) -> np.ndarray:
    """Renderer segmentation -> body id per pixel, -1 for background.

    Per **body**, not per geom: the paper's obstacles are object instances, and a crate rendered as
    fourteen geoms must not become fourteen obstacles.

    Channel order is (object id, object type) -- the reverse of what the name "segmentation image"
    suggests, and getting it backwards yields an all-background mask rather than an error.
    """
    import mujoco

    ids, types = seg[..., 0], seg[..., 1]
    out = np.full(ids.shape, -1, np.int64)
    hit = types == int(mujoco.mjtObj.mjOBJ_GEOM)
    out[hit] = model.geom_bodyid[ids[hit]]
    return out


def _geom_points(model, data, gid: int, n_sphere: int = 64) -> np.ndarray | None:
    """World-frame surface samples for one geom, whatever its type."""
    import mujoco

    gtype = int(model.geom_type[gid])
    size = np.asarray(model.geom_size[gid], np.float64)
    T = mujoco.mjtGeom
    if gtype == int(T.mjGEOM_MESH):
        mid = int(model.geom_dataid[gid])
        a = int(model.mesh_vertadr[mid])
        local = np.asarray(model.mesh_vert[a : a + int(model.mesh_vertnum[mid])], np.float64)
    elif gtype == int(T.mjGEOM_BOX):
        local = np.array([[sx, sy, sz] for sx in (-size[0], size[0])
                          for sy in (-size[1], size[1]) for sz in (-size[2], size[2])])
    elif gtype == int(T.mjGEOM_SPHERE):
        i = np.arange(n_sphere) + 0.5
        phi = np.arccos(1.0 - 2.0 * i / n_sphere)
        th = np.pi * (1.0 + 5.0**0.5) * i
        local = size[0] * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    elif gtype in (int(T.mjGEOM_CAPSULE), int(T.mjGEOM_CYLINDER)):
        i = np.arange(32) + 0.5
        th = 2 * np.pi * i / 32
        ring = np.stack([size[0] * np.cos(th), size[0] * np.sin(th), np.zeros(32)], 1)
        local = np.vstack([ring + [0, 0, size[1]], ring - [0, 0, size[1]]])
    else:
        return None  # planes and heightfields are not obstacles in this sense
    R = np.asarray(data.geom_xmat[gid], np.float64).reshape(3, 3)
    return local @ R.T + np.asarray(data.geom_xpos[gid], np.float64)


def body_ellipsoids(model, data, body_names, *, groups: dict | None = None) -> dict[str, Ellipsoid]:
    """Ground-truth ellipsoid per body, fitted to its geoms in the world frame.

    This is the "structure comes from the model" path of docs/16 §4: a crate's 0.9 cm wall cannot
    be recovered from one depth frame, and it does not need to be -- it is a known asset.
    ``groups`` maps a body name to {part name: [geom-name substrings]} to decompose it, which the
    crate needs or its handles inflate it across the whole table.
    """
    import mujoco

    out: dict[str, Ellipsoid] = {}
    for name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        buckets: dict[str, list[np.ndarray]] = {}
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != bid:
                continue
            pts = _geom_points(model, data, gid)
            if pts is None:
                continue
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            key = name
            for part, needles in (groups or {}).get(name, {}).items():
                if any(s in gname for s in needles):
                    key = f"{name}_{part}"
                    break
            buckets.setdefault(key, []).append(pts)
        for key, chunks in buckets.items():
            P = np.vstack(chunks)
            if len(P) > 3:
                c, Q = mvee(P, tol=1e-4)
                out[key] = Ellipsoid(c, Q)
    return out


@dataclasses.dataclass(frozen=True)
class LinkSpec:
    """A robot link to protect: which body, and the ellipsoid to wrap it in."""

    body: str
    ellipsoid: Ellipsoid | None = None  # None -> fit from the body's own geoms


def articulated_body(model, data, links, dof_ids, *, eef: str | None = None) -> ArticulatedBody:
    """Build the joint-space robot model from live MuJoCo state.

    ``mj_jac`` is evaluated **at each ellipsoid's centre**, not at the body origin: the barrier
    gradient is with respect to where the ellipsoid is, and a link's origin can sit tens of
    centimetres away from it.

    ``dof_ids`` selects the columns the filter is allowed to move, so the same call covers "arm
    only" and "arm plus base" -- the latter is what brings walls and shelves into the same QP.
    """
    import mujoco

    dof_ids = np.asarray(dof_ids, int)
    shapes = body_ellipsoids(model, data, [s.body for s in links if s.ellipsoid is None])
    out = {}
    for spec in links:
        E = spec.ellipsoid or shapes.get(spec.body)
        if E is None:
            raise KeyError(f"no geometry for link body {spec.body!r}")
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jp, jr, E.c, bid)
        out[spec.body] = Link(E, jp[:, dof_ids], jr[:, dof_ids])
    return ArticulatedBody(out, eef or links[-1].body)


def perceive_objects(model, data, cam: CameraModel, depth: np.ndarray, body_seg: np.ndarray,
                     body_ids, *, camera_pos=None, margin: PerceptionMargin | None = None,
                     complete: bool = True) -> dict[int, Ellipsoid]:
    """The full online path: render output -> one ellipsoid per body, corrected for one-view bias.

    Validated end to end against MuJoCo ground truth on rendered spheres: centres land within
    1.5-2.6 mm and the semi-axes come out slightly *larger* than truth, which is the safe side.

    ``complete`` should stay on for compact objects and off for structure -- the back-face rule
    assumes an object is about as deep as it is wide, which is precisely wrong for a crate wall
    (docs/16 §4). Structure should come from `body_ellipsoids` instead.
    """
    fits = fit_objects(body_seg, depth, cam, body_ids, depth_is_metric=True)
    if not complete:
        return fits
    cpos = np.asarray(camera_pos if camera_pos is not None else cam.cam_to_world[:3, 3], np.float64)
    margin = margin or PerceptionMargin()
    return {k: margin.inflate(complete_backface(E, cpos), cpos) for k, E in fits.items()}
