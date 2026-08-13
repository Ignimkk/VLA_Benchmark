"""Is a scene geometrically compatible with an ellipsoid-based CBF filter? — pre-flight check.

P3a taught this the expensive way: on LIBERO the filter failed 9/9 not because of attention error
or tuning, but because reaching the target already violated a *neighbouring* object's ellipsoid.
Gripper 7.0 cm + plate 8.1 cm = 15.1 cm of required clearance against a 12.7 cm target-to-plate
distance -- geometrically impossible, and no choice of gamma_h or epsilon can rescue it
(docs/12-p3a-results.md §3).

So before writing any integration code for a new scene, check it here. The test places the gripper
ellipsoid at each grasp target in turn and evaluates the paper's own barrier, Eq. (6), against every
non-target object:

    h_j = max_n  n.(c_tau - c_j) - sqrt(n' Q_R n) - sqrt(n' Q_j n)

h_j <= 0 for any obstacle j means the grasp pose itself lies inside that obstacle's ellipsoid.

Reads the MJCF and its OBJ meshes directly rather than compiling the model, so it needs no
measuring, stays in sync with the scene, and does not care whether the model currently compiles
(the transport model's keyframes are mid-refactor -- see TRANSPORT_SCENARIO_KO.md §8).

    src/openpi/.venv/bin/python -m benchmark.knows_vla.check_admissibility
"""

from __future__ import annotations

import argparse
import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np

from benchmark.knows_vla.cbf.ellipsoid import Ellipsoid, barrier, optimal_normal

# RB-Y1 gripper. [ASSUMPTION] -- the paper only says "calibrated offline" (OPEN-QUESTIONS #3), and
# docs/12-p3a-results.md §2 shows that fitting the gripper's segmentation mask swallows the wrist.
DEFAULT_GRIPPER_SEMI_AXES = (0.04, 0.04, 0.07)


def _parse_mjcf(path: pathlib.Path) -> ET.Element:
    """Parse an MJCF fragment.

    MuJoCo's XML reader tolerates `--` inside comments; ElementTree does not, and the transport
    scene has `--self-check` in one. Strip comments before parsing rather than editing the model.
    """
    text = re.sub(r"<!--.*?-->", "", path.read_text(), flags=re.S)
    return ET.fromstring(f"<mjcfroot>{text}</mjcfroot>")


def _floats(node, attr, default=None):
    v = node.get(attr)
    if v is None:
        return default
    return np.array([float(x) for x in v.split()], dtype=np.float64)


def _obj_extent(path: pathlib.Path) -> np.ndarray | None:
    """Half-extent of an OBJ's vertices about their own centre."""
    if not path.exists():
        return None
    v = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
    if not v:
        return None
    P = np.asarray(v)
    return 0.5 * (P.max(axis=0) - P.min(axis=0))


def collect_bodies(scene_xml: pathlib.Path, mesh_root: pathlib.Path, mesh_files: dict[str, str]):
    """Body name -> (world centre, half-extent), from geoms declared under each body.

    Deliberately one axis-aligned box per body, then one ellipsoid over it: the same coarse,
    single-convex representation the paper uses. That is the representation under test.
    """
    tree = _parse_mjcf(scene_xml)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for body in tree.iter("body"):
        name = body.get("name")
        if not name:
            continue
        bpos = _floats(body, "pos", np.zeros(3))
        lo, hi = None, None
        for geom in body.findall("geom"):
            gpos = _floats(geom, "pos", np.zeros(3))
            gtype = geom.get("type", "sphere")
            size = _floats(geom, "size")
            if gtype == "mesh":
                fname = mesh_files.get(geom.get("mesh", ""))
                half = _obj_extent(mesh_root / fname) if fname else None
            elif gtype == "box" and size is not None:
                half = size[:3]
            elif gtype == "sphere" and size is not None:
                half = np.full(3, size[0])
            elif gtype in ("capsule", "cylinder") and size is not None:
                half = np.array([size[0], size[0], size[1] + size[0]])
            else:
                half = None
            if half is None:
                continue
            a, b = gpos - half, gpos + half
            lo = a if lo is None else np.minimum(lo, a)
            hi = b if hi is None else np.maximum(hi, b)
        if lo is None:
            continue
        centre_local = 0.5 * (lo + hi)
        out[name] = (bpos + centre_local, np.maximum(0.5 * (hi - lo), 1e-4))
    return out


def to_ellipsoid(centre, half) -> Ellipsoid:
    """Circumscribed ellipsoid of an axis-aligned box: semi-axes = sqrt(3) x half-extent.

    Using the half-extent itself would leave the box corners outside the ellipsoid, i.e. the
    obstacle would be under-approximated -- the dangerous direction for a safety filter.
    """
    return Ellipsoid(centre, np.diag((np.asarray(half) * np.sqrt(3.0)) ** 2))


def main() -> None:
    root = pathlib.Path("src/rby1_description/models/rby1a/mujoco")
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=str(root / "scenes" / "scene_transport.xml"))
    p.add_argument("--assets", default=str(root / "scenes" / "transport_prop_assets.xml"))
    p.add_argument("--mesh-root", default=str(root))
    p.add_argument("--targets", nargs="+", default=["apple", "banana", "orange", "pear", "crate"])
    p.add_argument("--gripper-semi-axes", type=float, nargs=3, default=list(DEFAULT_GRIPPER_SEMI_AXES))
    a = p.parse_args()

    mesh_files = {}
    asset_path = pathlib.Path(a.assets)
    if asset_path.exists():
        for m in _parse_mjcf(asset_path).iter("mesh"):
            if m.get("name") and m.get("file"):
                mesh_files[m.get("name")] = m.get("file")

    bodies = collect_bodies(pathlib.Path(a.scene), pathlib.Path(a.mesh_root), mesh_files)
    ell = {n: to_ellipsoid(c, h) for n, (c, h) in bodies.items()}
    Q_R = np.diag(np.asarray(a.gripper_semi_axes, np.float64) ** 2)

    print("=" * 78)
    print(f"Admissibility check — {pathlib.Path(a.scene).name}")
    print(f"gripper semi-axes: {np.asarray(a.gripper_semi_axes) * 100} cm")
    print("=" * 78)
    print(f"\n{'body':<14} {'semi-axes (cm)':>24}   centre (m)")
    for n in sorted(ell):
        semi = np.sqrt(np.linalg.eigvalsh(ell[n].Q)) * 100
        print(f"{n:<14} {semi[0]:7.1f} {semi[1]:7.1f} {semi[2]:7.1f}   {np.round(ell[n].c, 3)}")

    targets = [t for t in a.targets if t in ell]
    ok_all = True
    for tgt in targets:
        print(f"\n=== 파지 타깃: {tgt} ===")
        print(f"  {'obstacle':<14} {'centre dist':>12} {'h at grasp':>12}   verdict")
        robot = Ellipsoid(ell[tgt].c, Q_R)  # gripper at the target: the pose a grasp requires
        for obs in sorted(ell):
            if obs == tgt:
                continue
            E = ell[obs]
            d = float(np.linalg.norm(robot.c - E.c))
            if d > 1.5:  # far away; not a candidate obstacle for this grasp
                continue
            h = barrier(optimal_normal(robot, E), robot, E)
            bad = h <= 0.0
            ok_all &= not bad
            print(f"  {obs:<14} {d * 100:11.1f} {h * 100:11.1f}   {'BLOCKED' if bad else 'ok'}")

    print("\n" + "=" * 78)
    print("RESULT:", "ADMISSIBLE" if ok_all else
          "NOT ADMISSIBLE — 타원체 표현으로 성립하지 않는 파지가 있음")
    print("=" * 78)
    if not ok_all:
        print("BLOCKED = 파지 자세 자체가 그 장애물 타원체 안에 있다는 뜻이다.")
        print("gamma_h / epsilon 조정으로 해결되지 않는다 — docs/12-p3a-results.md §3.")


if __name__ == "__main__":
    main()
