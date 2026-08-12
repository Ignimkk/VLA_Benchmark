"""Pin down the depth back-projection convention empirically, then sanity-check the MVEE fits.

Frame errors here are silent and fatal: a mirrored image still produces plausible-looking
ellipsoids, but they sit in the wrong place, and the CBF filter would then push the arm the wrong
way. So rather than reason about robosuite's IMAGE_CONVENTION plus our own 180-degree rotation, we
measure: the gripper's own segmentation mask must back-project onto ``robot0_eef_pos``, which the
episodes already record.

    src/openpi/.venv/bin/python -m benchmark.knows_vla.validate_backprojection
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from benchmark.knows_vla.cbf.ellipsoid import Ellipsoid
from benchmark.knows_vla.perception.ellipsoid_fit import (
    CameraModel,
    centroid_of,
    fit_objects,
    real_depth,
)

GRIPPER_TOKENS = ("Gripper",)


def _gripper_id(id2name: dict[int, str]) -> int | None:
    for i, name in id2name.items():
        if any(tok in name for tok in GRIPPER_TOKENS):
            return i
    return None


def check_conventions(episode: pathlib.Path, cams: dict) -> None:
    d = np.load(episode, allow_pickle=True)
    key = f"{d['suite']}_task{int(d['task_id'])}"
    if key not in cams:
        print(f"  {episode.name}: no camera params for {key}")
        return
    id2name = {int(s.split(":", 1)[0]): s.split(":", 1)[1] for s in d["id2name"]}
    gid = _gripper_id(id2name)
    if gid is None:
        print(f"  {episode.name}: no gripper instance")
        return

    eef = d["state"][:, 0:3]
    print(f"\n{episode.name}  ({id2name[gid]}, {len(d['t'])} steps)")
    print(f"  {'flip_row':>8} {'flip_col':>8} | {'mean err (m)':>12} {'median':>9} {'p90':>9}")
    best = None
    for flip_row in (False, True):
        for flip_col in (False, True):
            cam = CameraModel.from_json_entry(cams[key], flip_row=flip_row, flip_col=flip_col)
            errs = []
            for t in range(0, len(d["t"]), max(1, len(d["t"]) // 12)):
                c = centroid_of(d["seg_full"][t], d["depth_full"][t], cam, gid)
                if c is not None:
                    errs.append(float(np.linalg.norm(c - eef[t])))
            if not errs:
                continue
            e = np.asarray(errs)
            print(f"  {flip_row!s:>8} {flip_col!s:>8} | {e.mean():12.4f} {np.median(e):9.4f} {np.percentile(e, 90):9.4f}")
            if best is None or e.mean() < best[0]:
                best = (float(e.mean()), flip_row, flip_col)
    if best:
        print(f"  -> best: flip_row={best[1]} flip_col={best[2]}  (mean err {best[0]:.4f} m)")


def inspect_fits(episode: pathlib.Path, cams: dict, *, flip_row: bool, flip_col: bool) -> None:
    """Fitted ellipsoid semi-axes should be physically plausible for LIBERO tabletop objects."""
    d = np.load(episode, allow_pickle=True)
    key = f"{d['suite']}_task{int(d['task_id'])}"
    if key not in cams:
        return
    id2name = {int(s.split(":", 1)[0]): s.split(":", 1)[1] for s in d["id2name"]}
    cam = CameraModel.from_json_entry(cams[key], flip_row=flip_row, flip_col=flip_col)
    # Recompute candidates: the stored `candidate_ids` came from collection time, when the robot
    # filter still used name prefixes and let NullMount0 / OnTheGroundPanda0 through.
    robot_tokens = ("Panda", "Mount", "Gripper", "Robot")
    ids = [i for i, n in id2name.items() if i != 0 and not any(t in n for t in robot_tokens)]

    fits = fit_objects(d["seg_full"][0], d["depth_full"][0], cam, ids)
    print(f"\n{episode.name}: MVEE fits at t=0")
    print(f"  {'object':<34} {'semi-axes (cm)':>26} {'centre z (m)':>12}")
    for oid, E in sorted(fits.items()):
        semi = np.sqrt(np.clip(np.linalg.eigvalsh(E.Q), 0, None)) * 100.0
        print(f"  {id2name[oid]:<34} {semi[0]:7.1f} {semi[1]:7.1f} {semi[2]:7.1f}   {E.c[2]:12.3f}")

    dm = real_depth(d["depth_full"][0], cam.znear, cam.zfar)
    print(f"  depth range: {dm.min():.3f} .. {dm.max():.3f} m")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", nargs="+",
                   default=["data/knows_p0b/libero_spatial_task0_ep0.npz",
                            "data/knows_p0b/libero_object_task0_ep0.npz"])
    p.add_argument("--camera-params", default="benchmark/knows_vla/camera_params.json")
    a = p.parse_args()

    cams = json.loads(pathlib.Path(a.camera_params).read_text())
    print("=" * 78)
    print("Convention search: gripper mask centroid vs recorded robot0_eef_pos")
    print("=" * 78)
    for ep in a.episodes:
        check_conventions(pathlib.Path(ep), cams)

    print("\n" + "=" * 78)
    print("MVEE fits with flip_row=False flip_col=True")
    print("=" * 78)
    for ep in a.episodes:
        inspect_fits(pathlib.Path(ep), cams, flip_row=False, flip_col=True)


if __name__ == "__main__":
    main()
