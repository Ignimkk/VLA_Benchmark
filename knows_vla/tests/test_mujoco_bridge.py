"""MuJoCo adapter verification — camera, segmentation, Jacobians, and the whole chain.

Runs on a small scene built here rather than the transport model, because the transport model does
not compile in this environment: MuJoCo fails to open a different one of its 1,129 OBJ assets on
each attempt while Python reads all of them, which is a sandbox interaction rather than a model
defect. The code paths are identical, so the transport scene substitutes without changes.

    src/openpi/.venv/bin/python -m pytest benchmark/knows_vla/tests/test_mujoco_bridge.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from benchmark.knows_vla.cbf.ellipsoid import Ellipsoid, barrier, optimal_normal  # noqa: E402
from benchmark.knows_vla.cbf.filter import CbfParams, SafetyFilter  # noqa: E402
from benchmark.knows_vla.mujoco_bridge import (  # noqa: E402
    LinkSpec,
    articulated_body,
    body_ellipsoids,
    camera_model,
    perceive_objects,
    render,
    segmentation_to_body,
)

SCENE = """<mujoco>
 <visual><global offwidth="640" offheight="480"/></visual>
 <worldbody>
  <light pos="0 0 3"/>
  <camera name="head" pos="0.10 -0.70 1.45" xyaxes="1 0 0  0 0.62 0.79"/>
  <geom name="table" type="box" size="0.5 0.4 0.02" pos="0.4 0 0.80" rgba=".7 .6 .4 1"/>
  <body name="apple"  pos="0.30 -0.12 0.86"><geom type="sphere" size="0.040" rgba="1 0 0 1"/></body>
  <body name="orange" pos="0.46  0.10 0.87"><geom type="sphere" size="0.050" rgba="0 1 0 1"/></body>
  <body name="pear"   pos="0.55 -0.05 0.855"><geom type="sphere" size="0.035" rgba="0 0 1 1"/></body>
  <body name="crate" pos="0.40 0.25 0.86">
    <geom name="crate_floor"   type="box" size="0.09 0.12 0.005" pos="0 0 -0.05"/>
    <geom name="crate_wall_px" type="box" size="0.005 0.12 0.05" pos=" 0.085 0 0"/>
    <geom name="crate_wall_nx" type="box" size="0.005 0.12 0.05" pos="-0.085 0 0"/>
    <geom name="handle_l"      type="box" size="0.04 0.008 0.01" pos="0 0.25 0.03"/>
  </body>
  <body name="l0" pos="0 0 0.9"><joint name="j0" type="hinge" axis="0 0 1"/>
    <geom name="g0" type="capsule" size="0.03 0.10" pos="0.10 0 0" euler="0 90 0"/>
    <body name="l1" pos="0.22 0 0"><joint name="j1" type="hinge" axis="0 1 0"/>
      <geom name="g1" type="capsule" size="0.03 0.08" pos="0.08 0 0" euler="0 90 0"/>
      <body name="l2" pos="0.18 0 0"><joint name="j2" type="hinge" axis="1 0 0"/>
        <geom name="g2" type="box" size="0.04 0.04 0.06"/>
      </body></body></body>
 </worldbody></mujoco>"""


ARM_CLEAR = [-1.2, -0.25, 0.1]  # swung away from the fruits
ARM_IN_FRONT = [0.15, -0.25, 0.1]  # forearm covers roughly half the orange


@pytest.fixture(scope="module")
def sim():
    m = mujoco.MjModel.from_xml_string(SCENE)
    d = mujoco.MjData(m)
    return m, d


def _pose(sim, qpos):
    m, d = sim
    d.qpos[:3] = qpos
    mujoco.mj_forward(m, d)
    return m, d


def _visible_fraction(m, d, cam, bodies, name, radius):
    """Rendered pixel count over what an unoccluded disc of that radius would cover."""
    bid = _bid(m, name)
    dist = float(np.linalg.norm(np.asarray(d.xpos[bid]) - cam.cam_to_world[:3, 3]))
    return float((bodies == bid).sum()) / (np.pi * (radius * cam.K[0, 0] / dist) ** 2)


def _bid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


# ------------------------------------------------------------------ perception
def test_perceived_ellipsoids_match_ground_truth(sim):
    """The whole chain: render -> segmentation -> back-project -> trim -> MVEE -> back-face fix.

    Ground truth is exact here (spheres of known radius at known poses), so this pins the camera
    convention, the metric-depth handling and the bias correction in one assertion. Measured with
    nothing in the way: centres land within 1.2-1.9 mm and no semi-axis falls below the true
    radius, which is the only acceptable direction of error for an obstacle.
    """
    m, d = _pose(sim, ARM_CLEAR)
    r = mujoco.Renderer(m, 480, 640)
    _, depth, seg = render(r, d, "head")
    cam = camera_model(m, d, "head", 480, 640)
    bodies = segmentation_to_body(m, seg)
    ids = {n: _bid(m, n) for n in ("apple", "orange", "pear")}
    fits = perceive_objects(m, d, cam, depth, bodies, list(ids.values()))

    for name, radius in (("apple", 0.040), ("orange", 0.050), ("pear", 0.035)):
        assert _visible_fraction(m, d, cam, bodies, name, radius) > 0.9, f"{name} unexpectedly hidden"
        E = fits[ids[name]]
        assert np.linalg.norm(E.c - np.asarray(d.xpos[ids[name]])) < 0.003, name
        semi = np.sqrt(np.diag(E.Q))
        assert semi.min() >= radius - 1e-3, (name, semi)  # never smaller than the real object
        assert semi.max() < radius + 0.010, (name, semi)  # nor absurdly padded


def test_occlusion_makes_the_fit_under_estimate(sim):
    """What half a view costs -- and it costs it in the dangerous direction.

    With the forearm across the orange the centre error grows from 1.2 mm to ~13 mm and, worse,
    the smallest semi-axis drops *below* the true radius: the filter would believe the obstacle is
    smaller than it is. The back-face correction cannot help, since it assumes the visible surface
    is the object's near face, and here part of that face is missing too.

    Recorded rather than fixed: it needs occlusion-aware fitting (or a second view), which the
    paper does not describe either. See docs/OPEN-QUESTIONS.md.
    """
    m, d = _pose(sim, ARM_IN_FRONT)
    r = mujoco.Renderer(m, 480, 640)
    _, depth, seg = render(r, d, "head")
    cam = camera_model(m, d, "head", 480, 640)
    bodies = segmentation_to_body(m, seg)
    bid = _bid(m, "orange")

    assert 0.3 < _visible_fraction(m, d, cam, bodies, "orange", 0.050) < 0.8
    E = perceive_objects(m, d, cam, depth, bodies, [bid])[bid]
    assert np.linalg.norm(E.c - np.asarray(d.xpos[bid])) > 0.006
    assert np.sqrt(np.diag(E.Q)).min() < 0.050  # under-estimated: the direction that collides


def test_depth_must_be_declared_metric(sim):
    """MuJoCo already returns metres; treating it as a normalized buffer fails quietly, not loudly."""
    from benchmark.knows_vla.perception.ellipsoid_fit import fit_objects

    m, d = _pose(sim, ARM_CLEAR)
    r = mujoco.Renderer(m, 240, 320)
    _, depth, seg = render(r, d, "head")
    cam = camera_model(m, d, "head", 240, 320)
    bodies = segmentation_to_body(m, seg)
    oid = _bid(m, "orange")
    good = fit_objects(bodies, depth, cam, [oid], depth_is_metric=True)[oid]
    bad = fit_objects(bodies, depth, cam, [oid], depth_is_metric=False)[oid]
    truth = np.asarray(d.xpos[oid])
    assert np.linalg.norm(good.c - truth) < 0.03
    assert np.linalg.norm(bad.c - truth) > 0.20  # wrong, and silently so


def test_segmentation_is_per_body_not_per_geom(sim):
    """A crate drawn as four geoms is one obstacle, not four."""
    m, d = _pose(sim, ARM_CLEAR)
    r = mujoco.Renderer(m, 480, 640)
    _, _, seg = render(r, d, "head")
    bodies = segmentation_to_body(m, seg)
    assert (bodies == _bid(m, "crate")).sum() > 200
    assert set(np.unique(bodies)) <= {-1, 0} | {_bid(m, n) for n in
                                                ("apple", "orange", "pear", "crate", "l0", "l1", "l2")}


def test_structure_comes_from_the_model_and_can_be_decomposed(sim):
    """The crate's walls are 5 mm thick; only the model knows that (docs/16 §4)."""
    m, d = _pose(sim, ARM_CLEAR)
    whole = body_ellipsoids(m, d, ["crate"])["crate"]
    parts = body_ellipsoids(m, d, ["crate"], groups={"crate": {"handle": ["handle"]}})
    assert set(parts) == {"crate", "crate_handle"}
    # Splitting the handle off shrinks the body's reach along the axis the handle sticks out on.
    assert np.sqrt(parts["crate"].Q[1, 1]) < np.sqrt(whole.Q[1, 1]) - 0.02


# ------------------------------------------------------------------ kinematics
def test_mujoco_jacobians_match_finite_differenced_forward_kinematics(sim):
    """`mj_jac` at the ellipsoid centre, differenced against mj_forward -- not against itself."""
    m, d = _pose(sim, ARM_IN_FRONT)
    links = [LinkSpec("l0"), LinkSpec("l1"), LinkSpec("l2")]
    dofs = [0, 1, 2]
    body = articulated_body(m, d, links, dofs)
    obstacle = Ellipsoid.from_semi_axes([0.42, 0.02, 0.95], [0.05, 0.05, 0.05])

    q0 = d.qpos[:3].copy()
    for name in body.links:
        part = body.world(name)
        n = optimal_normal(part, obstacle)
        L = body.links[name]
        analytic = n @ L.Jv + np.cross(n, part.Q @ n) / np.sqrt(n @ part.Q @ n) @ L.Jw
        for i in range(3):
            eps = 1e-6
            hs = []
            for s in (+1, -1):
                d.qpos[:3] = q0
                d.qpos[i] += s * eps
                mujoco.mj_forward(m, d)
                moved = articulated_body(m, d, links, dofs).world(name)
                hs.append(barrier(n, moved, obstacle))
            d.qpos[:3] = q0
            mujoco.mj_forward(m, d)
            fd = (hs[0] - hs[1]) / (2 * eps)
            assert analytic[i] == pytest.approx(fd, rel=2e-3, abs=1e-7), (name, i)


def test_filter_runs_on_a_live_mujoco_state(sim):
    """End to end: perceived obstacles + MuJoCo Jacobians -> a joint command the robot can take."""
    m, d = _pose(sim, ARM_IN_FRONT)
    r = mujoco.Renderer(m, 240, 320)
    _, depth, seg = render(r, d, "head")
    cam = camera_model(m, d, "head", 240, 320)
    bodies = segmentation_to_body(m, seg)
    ids = [_bid(m, n) for n in ("apple", "orange", "pear")]
    obstacles = perceive_objects(m, d, cam, depth, bodies, ids)
    assert obstacles

    body = articulated_body(m, d, [LinkSpec("l0"), LinkSpec("l1"), LinkSpec("l2")], [0, 1, 2])
    p = CbfParams(normals="fixed", max_delta_q=0.05, slack_weight=1e3)
    res = SafetyFilter(p)(body, obstacles, delta_q_nom=np.array([0.05, 0.05, 0.05]))
    assert res.feasible and res.delta_q is not None
    assert np.max(np.abs(res.delta_q)) <= 0.05 + 1e-7
