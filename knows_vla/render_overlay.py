"""Render what the safety filter sees and what it would do, as a video.

Draws, on the frames the policy actually saw:
  * every tracked object's fitted ellipsoid, projected to the image
      green  = the phase-appropriate target (excluded from the obstacle set)
      red    = obstacle
      grey   = occluded this step, so its centroid is frozen
  * the gripper ellipsoid
  * the nominal action (white arrow) and the filtered action (cyan arrow)
  * min h, the intervention size, and whether the QP fell back to an emergency stop

IMPORTANT: this is the *offline* filter. Its output is never fed back to the simulator, so the
trajectory shown is the unfiltered one -- the cyan arrow is what the filter *would* have commanded,
not what happened. A genuine before/after comparison needs closed-loop execution (P3).

    src/openpi/.venv/bin/python -m benchmark.knows_vla.render_overlay \
        --episodes data/knows_p0b/libero_spatial_task0_ep0.npz --out data/knows_videos
"""

from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import imageio.v2 as imageio
import numpy as np

from benchmark.knows_vla.cbf.ellipsoid import Ellipsoid
from benchmark.knows_vla.cbf.filter import CbfParams, SafetyFilter
from benchmark.knows_vla.offline_filter import (
    GRIPPER_TOKEN,
    POS_SCALE,
    ROBOT_TOKENS,
    ROT_SCALE,
    _axisangle_to_mat,
    _gripper_shape,
)
from benchmark.knows_vla.perception.ellipsoid_fit import CameraModel, ObjectTracker, fit_objects

TARGET_BGR = (90, 220, 90)
OBSTACLE_BGR = (70, 70, 235)
FROZEN_BGR = (150, 150, 150)
ROBOT_BGR = (235, 200, 60)
NOMINAL_BGR = (255, 255, 255)
FILTERED_BGR = (235, 220, 60)


def _project(cam: CameraModel, pts_world: np.ndarray, h_img: int, w_img: int, scale: float):
    """World points -> pixel coords in the *stored* (rot-180) frame, scaled for display."""
    P = cam.world_to_pixel()
    hom = np.concatenate([pts_world, np.ones((len(pts_world), 1))], axis=1)
    uv = hom @ P.T
    z = np.clip(uv[:, 2:3], 1e-6, None)
    raw = uv[:, :2] / z  # [col, row] in raw-render indexing
    cols_raw, rows_raw = raw[:, 0], raw[:, 1]
    cols = (w_img - 1 - cols_raw) if cam.flip_col else cols_raw
    rows = (h_img - 1 - rows_raw) if cam.flip_row else rows_raw
    return np.stack([cols * scale, rows * scale], axis=1), z[:, 0]


def _ellipsoid_hull(cam, E: Ellipsoid, h_img, w_img, scale, n=180):
    """Outline of a projected ellipsoid: project surface samples, take the 2D convex hull."""
    u = np.random.default_rng(0).normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    try:
        L = np.linalg.cholesky(E.Q + np.eye(3) * 1e-12)
    except np.linalg.LinAlgError:
        return None
    pts = E.c + u @ L.T
    uv, z = _project(cam, pts, h_img, w_img, scale)
    uv = uv[z > 1e-3]
    if len(uv) < 3:
        return None
    hull = cv2.convexHull(uv.astype(np.float32))
    return hull.astype(np.int32)


def _arrow(img, cam, start_world, delta_world, colour, h_img, w_img, scale, gain=6.0):
    """Draw a world-space delta as an arrow, exaggerated so a 1 cm step is visible."""
    pts = np.stack([start_world, start_world + delta_world * gain])
    uv, z = _project(cam, pts, h_img, w_img, scale)
    if np.any(z <= 1e-3):
        return
    p0, p1 = uv[0].astype(int), uv[1].astype(int)
    if np.linalg.norm(p1 - p0) < 1.5:
        cv2.circle(img, tuple(p0), 3, colour, -1, cv2.LINE_AA)
    else:
        cv2.arrowedLine(img, tuple(p0), tuple(p1), colour, 2, cv2.LINE_AA, tipLength=0.3)


def render(path: pathlib.Path, cams: dict, params: CbfParams, out_dir: pathlib.Path,
           *, scale: float = 3.0, fps: int = 10, target_mode: str = "gt") -> pathlib.Path:
    d = np.load(path, allow_pickle=True)
    key = f"{d['suite']}_task{int(d['task_id'])}"
    cam = CameraModel.from_json_entry(cams[key])
    id2name = {int(s.split(":", 1)[0]): s.split(":", 1)[1] for s in d["id2name"]}
    obj_ids = [i for i, n in id2name.items() if i != 0 and not any(t in n for t in ROBOT_TOKENS)]
    gid = next((i for i, n in id2name.items() if GRIPPER_TOKEN in n), None)

    shapes = fit_objects(d["seg_full"][0], d["depth_full"][0], cam, obj_ids)
    tracker = ObjectTracker(cam, shapes)
    Q_local = _gripper_shape(d, cam, gid) if gid is not None else np.eye(3) * 0.05**2

    def _resolve(nm):
        nm = str(nm)
        c = [i for i in obj_ids if nm.startswith(id2name[i])]
        return max(c, key=lambda i: len(id2name[i])) if c else None

    reach = {j for n in d["manipulated"] if (j := _resolve(n)) is not None}
    place = {j for n in d["destinations"] if (j := _resolve(n)) is not None} or reach
    grip = d["action"][:, 6]
    closed = np.flatnonzero(grip > 0)
    grasp_t = int(closed[0]) if closed.size else len(grip)

    filt = SafetyFilter(params)
    seg_h, seg_w = d["seg_full"].shape[1:3]
    frames = []

    for t in range(len(d["t"])):
        tracked = tracker.update(d["seg_full"][t], d["depth_full"][t])
        tgt = reach if t < grasp_t else place
        obstacles = {k: v for k, v in tracked.items() if k not in tgt} if target_mode == "gt" else dict(tracked)

        R_t = _axisangle_to_mat(d["state"][t, 3:6])
        eef = d["state"][t, 0:3].astype(np.float64)
        robot = Ellipsoid(eef, R_t @ Q_local @ R_t.T)
        dc_nom = d["action"][t, 0:3].astype(np.float64) * POS_SCALE
        dth_nom = d["action"][t, 3:6].astype(np.float64) * ROT_SCALE
        res = filt(robot, obstacles, dc_nom, dth_nom)

        img = cv2.cvtColor(d["image"][t], cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, None, fx=scale * seg_w / img.shape[1], fy=scale * seg_h / img.shape[0],
                         interpolation=cv2.INTER_NEAREST)
        H, W = img.shape[:2]

        for oid, E in sorted(tracked.items()):
            colour = (FROZEN_BGR if oid in tracker.frozen
                      else TARGET_BGR if oid in tgt else OBSTACLE_BGR)
            hull = _ellipsoid_hull(cam, E, seg_h, seg_w, scale)
            if hull is not None:
                cv2.polylines(img, [hull], True, colour, 2, cv2.LINE_AA)
        hull = _ellipsoid_hull(cam, robot, seg_h, seg_w, scale)
        if hull is not None:
            cv2.polylines(img, [hull], True, ROBOT_BGR, 2, cv2.LINE_AA)

        _arrow(img, cam, eef, dc_nom, NOMINAL_BGR, seg_h, seg_w, scale)
        _arrow(img, cam, eef, res.delta_c, FILTERED_BGR, seg_h, seg_w, scale)

        phase = "reach" if t < grasp_t else "place"
        dpos = float(np.linalg.norm(res.delta_c - dc_nom))
        min_h = float(res.h.min()) if res.h.size else float("nan")
        lines = [
            f"t={t:3d}  {phase}  obstacles={len(obstacles)}",
            f"min h = {min_h * 100:+6.1f} cm" if res.h.size else "min h = n/a",
            f"|dv| = {dpos * 100:5.2f} cm" + ("  EMERGENCY STOP" if res.emergency_stop else ""),
        ]
        cv2.rectangle(img, (0, 0), (W, 16 + 18 * len(lines)), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        legend = "white=nominal  cyan=filtered  green=target  red=obstacle  grey=occluded"
        cv2.putText(img, legend, (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1,
                    cv2.LINE_AA)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_overlay.mp4"
    imageio.mimwrite(out, frames, fps=fps, quality=8, macro_block_size=1)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--camera-params", default="benchmark/knows_vla/camera_params.json")
    p.add_argument("--out", default="data/knows_videos")
    p.add_argument("--target", default="gt", choices=["gt", "none"])
    p.add_argument("--fps", type=int, default=10)
    a = p.parse_args()

    cams = json.loads(pathlib.Path(a.camera_params).read_text())
    params = CbfParams()
    for ep in a.episodes:
        out = render(pathlib.Path(ep), cams, params, pathlib.Path(a.out),
                     fps=a.fps, target_mode=a.target)
        print(f"  wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
