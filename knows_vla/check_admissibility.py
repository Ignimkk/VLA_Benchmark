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

The verdict depends on three modelling choices, so each is a flag rather than a constant:

  --fit mvee|box     how tightly an object's ellipsoid hugs it. `box` circumscribes the
                     axis-aligned bounding box (semi = sqrt(3) x half-extent) and is the cheap
                     conservative bound; `mvee` is the paper's own [35] fitted to the mesh.
  --grasp-offset     where the gripper ellipsoid sits relative to the target centre. Zero means
                     concentric, which no real top-down grasp is.
  --include-static   whether support surfaces count as obstacles. The paper segments
                     "manipulable objects" (§3.2), so by default they do not.

    src/openpi/.venv/bin/python -m benchmark.knows_vla.check_admissibility
"""

from __future__ import annotations

import argparse
import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np

from benchmark.knows_vla.cbf.ellipsoid import Ellipsoid, barrier, optimal_normal
from benchmark.knows_vla.perception.ellipsoid_fit import (
    PerceptionMargin,
    complete_backface,
    mvee,
    visible_half,
)

# RB-Y1 head camera, approximate. [ASSUMPTION] -- the transport MJCF declares no camera, and only
# the ray *direction* at each object matters here, which is insensitive to a few cm of camera pose.
DEFAULT_CAMERA = (0.05, 0.0, 1.45)

# RB-Y1 gripper. [ASSUMPTION] -- the paper only says "calibrated offline" (OPEN-QUESTIONS #3), and
# docs/12-p3a-results.md §2 shows that fitting the gripper's segmentation mask swallows the wrist.
DEFAULT_GRIPPER_SEMI_AXES = (0.04, 0.04, 0.07)

# Bodies that are support surfaces or room structure rather than manipulable objects.
STATIC_BODIES = ("table", "office", "shelf", "floor", "wall")

# Left and right handle assemblies, separately. Grouping both into one "handles" ellipsoid is worse
# than not splitting at all: the two bars sit at y = +-0.20 with nothing between them, so the MVEE
# that encloses both spans 36 cm of mostly empty air.
HANDLE_SPLIT = {"crate": [(r"^handle_(vis_)?\w*_l[pn]?$", "handle_l"),
                          (r"^handle_(vis_)?\w*_r[pn]?$", "handle_r")]}

# Full shell decomposition: one ellipsoid per face instead of one for the whole crate. A container
# is the case where a single convex hull is worst -- it fills the very volume the task needs.
CRATE_SHELL = {"crate": [
    (r"^handle_(vis_)?\w*_l[pn]?$", "handle_l"),
    (r"^handle_(vis_)?\w*_r[pn]?$", "handle_r"),
    (r"^crate_(vis_)?floor$", "floor"),
    (r"^crate_(vis_)?(wall|rim)_px$", "wall_px"),
    (r"^crate_(vis_)?(wall|rim)_nx$", "wall_nx"),
    (r"^crate_(vis_)?(wall|rim|slat)_py(_\d)?$", "wall_py"),
    (r"^crate_(vis_)?(wall|rim|slat)_ny(_\d)?$", "wall_ny"),
]}


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


def _obj_vertices(path: pathlib.Path) -> np.ndarray | None:
    if not path.exists():
        return None
    v = [[float(x) for x in ln.split()[1:4]] for ln in path.read_text().splitlines() if ln.startswith("v ")]
    return np.asarray(v, np.float64) if v else None


def _sphere_points(radius: float, n: int = 64) -> np.ndarray:
    """Fibonacci sphere. A sphere has no corners to enumerate, so sample its surface instead."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return radius * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def _geom_points(geom, mesh_root: pathlib.Path, mesh_files: dict[str, str]) -> np.ndarray | None:
    """Surface points of one geom, in its body's frame.

    Points rather than a bounding box, because an MVEE over the union of a body's geoms is what
    makes the fit tight -- boxing each geom first would throw the tightness away before fitting.
    """
    gpos = _floats(geom, "pos", np.zeros(3))
    gtype = geom.get("type", "sphere")
    size = _floats(geom, "size")
    if gtype == "mesh":
        fname = mesh_files.get(geom.get("mesh", ""))
        pts = _obj_vertices(mesh_root / fname) if fname else None
    elif gtype == "box" and size is not None:
        s = size[:3]
        pts = np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
    elif gtype == "sphere" and size is not None:
        pts = _sphere_points(size[0])
    elif gtype in ("capsule", "cylinder") and size is not None:
        r, hl = size[0], size[1]
        cap = _sphere_points(r, 32)
        pts = np.vstack([cap + [0, 0, hl], cap - [0, 0, hl]])
    else:
        return None
    return None if pts is None else pts + gpos


def collect_bodies(
    scene_xml: pathlib.Path,
    mesh_root: pathlib.Path,
    mesh_files: dict[str, str],
    split_geoms: dict[str, list[tuple[str, str]]] | None = None,
) -> dict[str, np.ndarray]:
    """Body name -> world-frame surface point cloud.

    `split_geoms` maps a body name to [(geom-name regex, part name)]; matching geoms move into a
    separate pseudo-body `<body>_<part>`, first match winning. Per-instance segmentation would
    return the crate whole, so any split is an engineering adaptation to be justified, not a default.
    """
    tree = _parse_mjcf(scene_xml)
    out: dict[str, np.ndarray] = {}
    for body in tree.iter("body"):
        name = body.get("name")
        if not name:
            continue
        bpos = _floats(body, "pos", np.zeros(3))
        groups: dict[str, list[np.ndarray]] = {}
        rules = (split_geoms or {}).get(name, [])
        for geom in body.findall("geom"):
            pts = _geom_points(geom, mesh_root, mesh_files)
            if pts is None:
                continue
            gname = geom.get("name") or ""
            key = next((f"{name}_{part}" for pat, part in rules if re.match(pat, gname)), name)
            groups.setdefault(key, []).append(pts + bpos)
        for key, chunks in groups.items():
            out[key] = np.vstack(chunks)
    return out


def fit_ellipsoid(points: np.ndarray, method: str) -> Ellipsoid:
    """Wrap a point cloud in one ellipsoid -- the paper's single-convex object representation.

    `box` circumscribes the axis-aligned bounding box: semi-axes = sqrt(3) x half-extent, since the
    half-extent alone would leave the box corners outside, i.e. under-approximate the obstacle,
    which is the dangerous direction for a safety filter. It is a bound, not a fit.

    `mvee` is the paper's own choice ([35], Khachiyan) applied to the mesh, so it can rotate with
    the object and does not pay for the corners of a box that is not there.
    """
    if method == "mvee":
        c, Q = mvee(points, tol=1e-4)
        return Ellipsoid(c, Q)
    lo, hi = points.min(0), points.max(0)
    half = np.maximum(0.5 * (hi - lo), 1e-4)
    return Ellipsoid(0.5 * (lo + hi), np.diag((half * np.sqrt(3.0)) ** 2))


def fit_as_perceived(points: np.ndarray, camera, margin: PerceptionMargin) -> Ellipsoid:
    """What the perception stack would actually produce from one viewpoint, not the GT mesh fit.

    Every earlier verdict in docs/14 fitted the *complete* mesh, which no camera ever sees. This
    keeps only the near-facing points, fits those, then corrects the missing back face and adds the
    residual margin -- the same three steps the online pipeline has to perform.
    """
    vis = visible_half(points, camera)
    if len(vis) <= 3:
        vis = points
    c, Q = mvee(vis, tol=1e-4)
    return margin.inflate(complete_backface(Ellipsoid(c, Q), camera), camera)


def axis_extent(E: Ellipsoid) -> np.ndarray:
    """How far the ellipsoid reaches along x, y, z: sqrt(Q_ii), its support in each axis.

    Reporting sorted eigenvalues instead is what hid the crate's real shape: its 36 cm semi-axis is
    the y one -- the handles reach y = +-0.21 -- and reading it as z blamed the handle *height*.
    Support per axis is also the quantity that decides whether two objects at a known separation
    can both be enclosed, so it is what belongs in the table.
    """
    return np.sqrt(np.diag(E.Q))


def block_gripper(semi_axes) -> dict[str, Ellipsoid]:
    """The one-ellipsoid end-effector the paper assumes, centred on the EEF site."""
    return {"gripper": Ellipsoid(np.zeros(3), np.diag(np.asarray(semi_axes, np.float64) ** 2))}


def rby1_gripper(assets: pathlib.Path, opening: float = 0.037) -> dict[str, Ellipsoid]:
    """RB-Y1 hand as measured parts, in the `right_ee` site frame.

    Measured, not assumed -- and it matters: the hand housing is 7.6 x 4.5 x 5.8 cm and reaches
    11.3 cm above the site, so it is *larger* than the 4 x 4 x 7 cm block we had been using. The
    fingers (1.2 x 2.1 x 4.9 cm) never dominate; the housing always does.

    Frame chain from rby1.xml: EE_BODY_R at wrist z=-0.1548, finger bodies at z=-0.2278, the
    `right_ee` site at z=-0.2548. `opening` is the finger slide plus its 3 mm mount offset, and the
    fingers close along the site's y axis (the banana's long axis is kept across it -- see the
    scene comment on why).

    Deliberately one ellipsoid for the housing rather than its 25 collision pieces. Splitting a
    *convex* body makes the fit worse, not better: each piece pays its own MVEE inflation and can
    poke outside the parent ellipsoid where that ellipsoid narrows (measured: orange -1.0 -> -1.6 cm).
    Decompose concave bodies only -- see `CRATE_SHELL`.
    """
    def load(pattern: str) -> np.ndarray:
        files = sorted((assets / pattern.split("/")[0]).glob(pattern.split("/")[1]))
        return np.vstack([_obj_vertices(f) for f in files])

    ch, Qh = mvee(load("EE_BODY/EE_BODY_collision_*.obj") + [0, 0, 0.100], tol=1e-6)
    cf, Qf = mvee(load("EE_FINGER/EE_FINGER_collision_*.obj") + [0, 0, 0.027], tol=1e-6)
    parts = {"hand": Ellipsoid(ch, Qh)}
    for sign, tag in ((+1.0, "finger_p"), (-1.0, "finger_n")):
        parts[tag] = Ellipsoid(cf + sign * opening * np.array([0.0, 1.0, 0.0]), Qf)
    return parts


def evaluate(ell: dict[str, Ellipsoid], targets, robot: dict[str, Ellipsoid],
             offset: np.ndarray, max_dist: float):
    """h at the grasp pose for every (target, obstacle) pair, minimised over robot parts.

    Rows carry the dominating part so a bad verdict says *what* is in the way -- with a multi-part
    hand that is the whole diagnosis.
    """
    rows = []
    for tgt in targets:
        origin = ell[tgt].c + offset
        for obs in sorted(ell):
            if obs == tgt:
                continue
            E = ell[obs]
            d = float(np.linalg.norm(origin - E.c))
            if d > max_dist:
                continue
            hs = {}
            for pname, part in robot.items():
                moved = Ellipsoid(origin + part.c, part.Q)
                hs[pname] = barrier(optimal_normal(moved, E), moved, E)
            worst = min(hs, key=hs.get)
            rows.append((tgt, obs, d, hs[worst], worst))
    return rows, all(r[3] > 0.0 for r in rows)


def critical_offset(ell: dict[str, Ellipsoid], tgt: str, robot: dict[str, Ellipsoid],
                    max_dist: float) -> float | None:
    """Smallest vertical gripper offset that clears every obstacle for this target, or None.

    A grasp is not concentric with its object, so h at zero offset is a worst case rather than the
    operating point. This says how much lift the method needs -- and if the answer exceeds what the
    task allows (the fruit must still be reachable), the scene is out regardless.
    """
    lo, hi = 0.0, 0.30
    if evaluate(ell, [tgt], robot, np.array([0.0, 0.0, hi]), max_dist)[1] is False:
        return None
    if evaluate(ell, [tgt], robot, np.zeros(3), max_dist)[1]:
        return 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if evaluate(ell, [tgt], robot, np.array([0.0, 0.0, mid]), max_dist)[1]:
            hi = mid
        else:
            lo = mid
    return hi


def main() -> None:
    root = pathlib.Path("src/rby1_description/models/rby1a/mujoco")
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=str(root / "scenes" / "scene_transport.xml"))
    p.add_argument("--assets", default=str(root / "scenes" / "transport_prop_assets.xml"))
    p.add_argument("--mesh-root", default=str(root))
    p.add_argument("--targets", nargs="+", default=["apple", "banana", "orange", "pear", "crate"])
    p.add_argument("--fit", choices=["mvee", "box"], default="mvee")
    p.add_argument("--gripper", choices=["rby1", "block"], default="rby1",
                   help="rby1 = hand + two fingers measured from the meshes; block = the paper's single ellipsoid")
    p.add_argument("--gripper-semi-axes", type=float, nargs=3, default=list(DEFAULT_GRIPPER_SEMI_AXES),
                   help="--gripper block only")
    p.add_argument("--gripper-opening", type=float, default=0.037,
                   help="--gripper rby1 only: finger centre offset from the EEF axis, metres")
    p.add_argument("--grasp-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help="EEF site relative to the target centre, metres")
    p.add_argument("--include-static", action="store_true",
                   help="count support surfaces and room structure as obstacles (the paper does not)")
    p.add_argument("--split-handles", action="store_true",
                   help="fit each of the crate's two handle assemblies as its own ellipsoid")
    p.add_argument("--shell", action="store_true",
                   help="decompose the crate into faces (4 walls + floor + 2 handles)")
    p.add_argument("--max-dist", type=float, default=1.5, help="ignore obstacles beyond this range")
    p.add_argument("--perception", choices=["gt", "single-view"], default="gt",
                   help="gt fits the whole mesh (optimistic); single-view fits only what a camera sees")
    p.add_argument("--camera", type=float, nargs=3, default=list(DEFAULT_CAMERA))
    p.add_argument("--margin-ray", type=float, default=0.005,
                   help="single-view only: residual along-ray padding after back-face completion")
    p.add_argument("--margin-lateral", type=float, default=0.003,
                   help="single-view only: mask-boundary padding across the ray")
    p.add_argument("--model-known", nargs="*", default=["crate", "shelf"],
                   help="body prefixes whose geometry comes from the model, not from perception")
    a = p.parse_args()

    mesh_files = {}
    asset_path = pathlib.Path(a.assets)
    if asset_path.exists():
        for m in _parse_mjcf(asset_path).iter("mesh"):
            if m.get("name") and m.get("file"):
                mesh_files[m.get("name")] = m.get("file")

    bodies = collect_bodies(
        pathlib.Path(a.scene), pathlib.Path(a.mesh_root), mesh_files,
        split_geoms=CRATE_SHELL if a.shell else HANDLE_SPLIT if a.split_handles else None,
    )
    if not a.include_static:
        # Prefix match, not substring: `crate_wall_px` is a piece of a manipulable crate, and a
        # substring test on "wall" would silently drop it along with the room's walls.
        bodies = {n: p_ for n, p_ in bodies.items()
                  if not any(n == s or n.startswith(s + "_") for s in STATIC_BODIES)}
    if a.perception == "single-view":
        # Structure (crate, shelf) comes from its model, not from a depth frame: it is a known
        # tracked asset, and single-view back-face completion is meaningless for a thin plate --
        # it would refill the crate interior that decomposing it opened. Perception is applied to
        # exactly the objects that have to be discovered.
        margin = PerceptionMargin(a.margin_ray, a.margin_lateral)
        ell = {n: (fit_ellipsoid(pts, a.fit)
                   if any(n.startswith(m) for m in a.model_known)
                   else fit_as_perceived(pts, a.camera, margin))
               for n, pts in bodies.items()}
    else:
        ell = {n: fit_ellipsoid(pts, a.fit) for n, pts in bodies.items()}
    robot = (rby1_gripper(pathlib.Path(a.mesh_root) / "assets", a.gripper_opening)
             if a.gripper == "rby1" else block_gripper(a.gripper_semi_axes))
    offset = np.asarray(a.grasp_offset, np.float64)

    print("=" * 78)
    print(f"Admissibility check — {pathlib.Path(a.scene).name}")
    print(f"fit={a.fit}  gripper={a.gripper}  grasp offset={offset * 100} cm  "
          f"static obstacles={'on' if a.include_static else 'off'}")
    print(f"perception={a.perception}"
          + (f"  camera={a.camera}  margin ray/lat={a.margin_ray * 100:.1f}/{a.margin_lateral * 100:.1f} cm"
             if a.perception == "single-view" else "  (전체 메시 — 낙관적)"))
    print("=" * 78)
    print(f"\n{'robot part':<16} {'reach x,y,z (cm)':>24}   centre rel. EEF (cm)")
    for n in sorted(robot):
        print(f"{n:<16} {np.array2string(axis_extent(robot[n]) * 100, precision=1, floatmode='fixed'):>24}"
              f"   {np.round(robot[n].c * 100, 1)}")
    print(f"\n{'body':<16} {'reach x,y,z (cm)':>24}   centre (m)")
    for n in sorted(ell):
        print(f"{n:<16} {np.array2string(axis_extent(ell[n]) * 100, precision=1, floatmode='fixed'):>24}"
              f"   {np.round(ell[n].c, 3)}")

    targets = [t for t in a.targets if t in ell]
    rows, ok_all = evaluate(ell, targets, robot, offset, a.max_dist)
    for tgt in targets:
        print(f"\n=== 파지 타깃: {tgt} ===")
        print(f"  {'obstacle':<16} {'centre dist':>12} {'h at grasp':>12}   {'part':<10} verdict")
        for t, obs, d, h, part in rows:
            if t == tgt:
                print(f"  {obs:<16} {d * 100:11.1f} {h * 100:11.1f}   {part:<10} "
                      f"{'BLOCKED' if h <= 0 else 'ok'}")
        crit = critical_offset(ell, tgt, robot, a.max_dist)
        print(f"  -> 필요한 최소 수직 오프셋: "
              f"{'없음 (오프셋 0에서 통과)' if crit == 0.0 else 'z=30 cm에서도 통과 못함' if crit is None else f'{crit * 100:.1f} cm'}")

    print("\n" + "=" * 78)
    print("RESULT:", "ADMISSIBLE" if ok_all else
          "NOT ADMISSIBLE — 타원체 표현으로 성립하지 않는 파지가 있음")
    print("=" * 78)
    if not ok_all:
        print("BLOCKED = 파지 자세 자체가 그 장애물 타원체 안에 있다는 뜻이다.")
        print("gamma_h / epsilon 조정으로 해결되지 않는다 — docs/12-p3a-results.md §3.")


if __name__ == "__main__":
    main()
