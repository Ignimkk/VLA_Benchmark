"""MuJoCo -> AG3S input adapter for the RB-Y1 transport scene.

Turns a MuJoCo camera into exactly what `AG3S.process` wants: metric depth, a pinhole `K`, a
`T_base_cam` in the robot base frame, and the 20-vector the URDF collision model expects. Ground
truth comes free — MuJoCo's segmentation buffer names the body behind every pixel, so candidates can
be scored rather than eyeballed.

Four conventions had to be pinned down by measurement rather than by reading. Each one is silent
when wrong, which is exactly the class of bug `benchmark/knows_vla/validate_backprojection.py` was
written about:

**The model must be loaded from an absolute path.** With a relative one, mesh loading fails on a
*different* `assets/base/base_N.obj` on each run — a file that is present, readable and valid. The
nondeterminism is the tell: it is path resolution during MuJoCo's parallel mesh load, not a corrupt
asset. `transport_scene.py` already carries the same note for its own reasons.

**Depth is already metric.** MuJoCo 3.x `enable_depth_rendering()` returns metres, unlike the
normalized buffer robosuite hands back (which is why `knows_vla` needs its `real_depth()` helper).
Unhit pixels come back at the far plane, ~679 m here, and are dropped by `pointcloud.depth_max`.

**The segmentation buffer is (object id, object type), not the reverse.** Reading it the other way
around silently yields zero geoms and an empty ground truth.

**MuJoCo cameras look down -z with +y up; AG3S/OpenCV uses +z forward and +y down.** The fix is a
`diag(1, -1, -1)` on the rotation. Verified by back-projecting the floor: it comes back flat at
z = 0.006 m with a 1.3 mm spread.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Optional, Sequence

import numpy as np

TRANSPORT_MODEL = pathlib.Path(
    "src/rby1_description/models/rby1a/mujoco/model_transport.xml"
)

#: The 20 joints `benchmark.ag3s.robot_models.load_rby1` expects, in its order.
from benchmark.ag3s.runtime.asset_path import resolve_asset  # noqa: E402
from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS  # noqa: E402

FREE_JOINTS = ("crate_free", "apple_free", "banana_free", "orange_free", "pear_free")

#: Which URDF link each camera is bolted to, and the `CameraID` it plays in a fused frame. Read off
#: `src/rby1_description/models/rby1a/mujoco/rby1.xml`: `wrist_cam_body.xml` is included inside
#: `link_right_arm_6`, `wrist_cam_body_l.xml` inside `link_left_arm_6`, and `zed_body.xml` inside
#: `link_head_2`. All three names exist unchanged in the URDF, which is what makes capture-time FK
#: possible without a separate extrinsics file.
CAMERA_MOUNTS: dict[str, tuple[str, str]] = {
    "zed_left": ("link_head_2", "head"),
    "zed_right": ("link_head_2", "head"),
    "wrist_cam_l": ("link_left_arm_6", "left_wrist"),
    "wrist_cam_r": ("link_right_arm_6", "right_wrist"),
}

#: Head joints, which `DEFAULT_RBY1_JOINTS` leaves out of `q` because they carry no collision
#: capsule. The head camera's pose depends on them, so they have to be pinned into the URDF chain as
#: fixed values or its FK tracks a head that is always level.
HEAD_JOINTS = ("head_0", "head_1")

#: Bodies that are the robot itself. Used only to score the self-filter against ground truth — the
#: filter itself works from FK and never sees these names.
ROBOT_BODY_PREFIXES = (
    "base", "link_", "wheel", "ee_", "EE_", "FT_", "d435i", "wrist_bracket", "zed",
)


def is_robot_body(name: Optional[str]) -> bool:
    return bool(name) and name.startswith(ROBOT_BODY_PREFIXES)


@dataclasses.dataclass(frozen=True)
class CameraFrame:
    """One camera's capture, in the form AG3S consumes."""

    name: str
    depth: np.ndarray  # (H, W) float64 metres
    camera_intrinsics: np.ndarray  # (3, 3)
    T_base_cam: np.ndarray  # (4, 4) camera -> robot base
    body_ids: np.ndarray  # (H, W) int32 ground-truth body id, -1 where nothing was hit
    body_names: dict[int, str]
    robot_state: np.ndarray  # (20,)
    #: (H, W) int32 ground-truth *geom* id, -1 where nothing was hit. Body ids are too coarse for
    #: some questions -- a table is one body but five geoms, and fitting a plane to all of them
    #: fits the legs as well as the top.
    geom_ids: Optional[np.ndarray] = None

    @property
    def hw(self) -> tuple[int, int]:
        return int(self.depth.shape[0]), int(self.depth.shape[1])

    def label_map(self) -> np.ndarray:
        """Body id per pixel — what `pixel_labels` indexes into."""
        return self.body_ids

    def name_of(self, body_id: int) -> str:
        return self.body_names.get(int(body_id), "?")


class TransportScene:
    """The RB-Y1 crate-transport scene, posed for manipulation.

    The pose is built in two steps because the scene needs both a settled world and a useful arm
    configuration: free bodies are dropped and simulated until they rest on the table, then the
    articulated joints are overwritten from the model's own `teleop` keyframe. Applying the keyframe
    wholesale instead would also reset the crate and fruits to the origin, collapsing the scene.
    """

    def __init__(
        self,
        model_path: str | pathlib.Path = TRANSPORT_MODEL,
        *,
        keyframe: str = "teleop",
        settle_steps: int = 400,
        height: int = 480,
        width: int = 640,
    ):
        import mujoco

        self.mujoco = mujoco
        # 기록의 옛 절대경로도 여기서 이 머신의 자산으로 해석된다 — `ag3s.asset_path` 참고.
        path = resolve_asset(model_path, what="transport model")
        # Absolute, always — see the module docstring.
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.height, self.width = int(height), int(width)

        self._joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self._qadr = {
            name: int(self.model.jnt_qposadr[i])
            for i, name in enumerate(self._joint_names)
            if name
        }
        self._body_names = {
            i: (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}")
            for i in range(self.model.nbody)
        }
        self.reset(keyframe=keyframe, settle_steps=settle_steps)
        self._renderer = None

    @classmethod
    def attach(cls, model, data, *, height: int = 480, width: int = 640) -> "TransportScene":
        """이미 돌고 있는 시뮬레이션의 `model`/`data` 에 **붙는다**. 새로 로드하지 않는다.

        `__init__` 은 XML 을 읽어 자기 `MjData` 를 만든다. 재생 실험에서는 그것이 맞지만 live
        제어 루프에서는 치명적이다 — 로봇이 움직이는 씬과 AG3S 가 보는 씬이 갈라져, 제약이
        설명하는 자세와 실행되는 자세가 다른 것을 아무도 눈치채지 못한다. 여기서는 같은 객체를
        가리키므로 `reset`/`settle` 도 하지 않는다: 이 씬의 상태는 제어 루프가 소유한다.
        """
        import mujoco

        self = cls.__new__(cls)
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.height, self.width = int(height), int(width)
        self._joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
        ]
        self._qadr = {
            name: int(model.jnt_qposadr[i])
            for i, name in enumerate(self._joint_names) if name
        }
        self._body_names = {
            i: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}")
            for i in range(model.nbody)
        }
        self._renderer = None
        return self

    # --- posing -------------------------------------------------------------------------
    def reset(self, *, keyframe: str = "teleop", settle_steps: int = 400) -> None:
        mujoco = self.mujoco
        mujoco.mj_resetData(self.model, self.data)
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
        if keyframe:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if key_id < 0:
                raise KeyError(f"no keyframe {keyframe!r} in the model")
            target = self.model.key_qpos[key_id]
            for name in self._joint_names:
                if name and name not in FREE_JOINTS:
                    self.data.qpos[self._qadr[name]] = target[self._qadr[name]]
        mujoco.mj_forward(self.model, self.data)

    def set_joint(self, name: str, value: float) -> None:
        self.data.qpos[self._qadr[name]] = float(value)
        self.mujoco.mj_forward(self.model, self.data)

    # --- state --------------------------------------------------------------------------
    def robot_state(self) -> np.ndarray:
        """The 20-vector `load_rby1()` consumes, read straight out of `qpos`."""
        return np.asarray(
            [self.data.qpos[self._qadr[j]] for j in DEFAULT_RBY1_JOINTS], np.float64
        )

    def body_pose(self, name: str) -> np.ndarray:
        bid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"no body {name!r}")
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[bid].reshape(3, 3)
        T[:3, 3] = self.data.xpos[bid]
        return T

    def body_position(self, name: str) -> np.ndarray:
        return self.body_pose(name)[:3, 3].copy()

    def body_position_in_base(self, name: str) -> np.ndarray:
        T = np.linalg.inv(self.body_pose("base")) @ self.body_pose(name)
        return T[:3, 3].copy()

    # --- capture ------------------------------------------------------------------------
    def _renderer_for(self):
        if self._renderer is None:
            self._renderer = self.mujoco.Renderer(self.model, self.height, self.width)
        return self._renderer

    def intrinsics(self, camera: str) -> np.ndarray:
        """Pinhole `K` from the camera's vertical field of view. Square pixels, centred principal point."""
        cid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cid < 0:
            raise KeyError(f"no camera {camera!r}")
        fovy = float(self.model.cam_fovy[cid])
        f = (self.height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        return np.array(
            [[f, 0.0, self.width / 2.0], [0.0, f, self.height / 2.0], [0.0, 0.0, 1.0]],
            np.float64,
        )

    def T_base_cam(self, camera: str) -> np.ndarray:
        """Camera pose in the robot base frame, in the OpenCV convention AG3S uses."""
        cid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cid < 0:
            raise KeyError(f"no camera {camera!r}")
        T_world_cam = np.eye(4)
        # MuJoCo: -z forward, +y up. OpenCV: +z forward, +y down.
        T_world_cam[:3, :3] = self.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T_world_cam[:3, 3] = self.data.cam_xpos[cid]
        return np.linalg.inv(self.body_pose("base")) @ T_world_cam

    def capture(self, camera: str) -> CameraFrame:
        mujoco = self.mujoco
        renderer = self._renderer_for()

        renderer.disable_depth_rendering()
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=camera)
        depth = np.asarray(renderer.render(), np.float64)

        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(self.data, camera=camera)
        seg = np.asarray(renderer.render())
        renderer.disable_segmentation_rendering()

        # (object id, object type) — not the reverse. See the module docstring.
        object_id, object_type = seg[..., 0], seg[..., 1]
        is_geom = object_type == int(mujoco.mjtObj.mjOBJ_GEOM)
        body_ids = np.full(object_id.shape, -1, np.int32)
        geom_ids = object_id[is_geom]
        body_ids[is_geom] = self.model.geom_bodyid[geom_ids]

        geom_map = np.full(object_id.shape, -1, np.int32)
        geom_map[is_geom] = geom_ids

        return CameraFrame(
            name=camera,
            depth=depth,
            camera_intrinsics=self.intrinsics(camera),
            T_base_cam=self.T_base_cam(camera),
            body_ids=body_ids,
            body_names=dict(self._body_names),
            robot_state=self.robot_state(),
            geom_ids=geom_map,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def pixel_labels(frame: CameraFrame, uv: np.ndarray) -> np.ndarray:
    """Ground-truth body id for each `(u, v)` a reconstructed point came from."""
    uv = np.asarray(uv, np.int64)
    return frame.body_ids[uv[:, 1], uv[:, 0]]


def gaussian_attention(
    frame: CameraFrame,
    target_position_base: np.ndarray,
    *,
    grid: int = 16,
    sigma_patches: float = 1.3,
    floor: float = 0.02,
) -> np.ndarray:
    """A **stand-in** for VLA attention: a Gaussian on the target's projected centroid.

    This is not π0.5 attention. That checkpoint is trained on LIBERO's agent view and takes a
    224x224 RGB observation, so pointing it at an RB-Y1 ZED frame would produce a number with no
    meaning attached. What this experiment can honestly test is everything downstream of attention —
    lifting, 3D grounding, candidate generation, constraint emission — given a plausible attention
    blob in a known place. Replacing this function is the single change needed once a policy that
    actually looks through these cameras exists.

    `floor` keeps every patch slightly non-zero, so a pipeline that (wrongly) used attention to
    *select geometry* would still see the whole scene rather than accidentally looking correct.
    """
    height, width = frame.hw
    K, T = frame.camera_intrinsics, frame.T_base_cam
    point_cam = (np.asarray(target_position_base, np.float64).reshape(3) - T[:3, 3]) @ T[:3, :3]
    depth = max(float(point_cam[2]), 1e-6)
    u = K[0, 0] * point_cam[0] / depth + K[0, 2]
    v = K[1, 1] * point_cam[1] / depth + K[1, 2]

    gy, gx = np.mgrid[0:grid, 0:grid]
    px, py = u / width * grid, v / height * grid
    blob = np.exp(-(((gx + 0.5) - px) ** 2 + ((gy + 0.5) - py) ** 2) / (2.0 * sigma_patches**2))
    blob = blob + floor
    return (blob / blob.max()).astype(np.float32)


# --------------------------------------------------------- filling the URDF's collision gaps


def _body_vertices(model, body_id: int, *, visual_ok: bool = True) -> np.ndarray:
    """Every mesh vertex of a body's geoms, expressed in the body frame.

    Visual geoms are included by default and that is the point. The self-filter's job is to delete
    what the *camera sees*, and a link with no collision geometry still has a visible surface — the
    RB-Y1 base and wheels never self-collide, so neither the URDF nor the MuJoCo model gives them a
    collision volume, yet they fill a third of the head camera's view.
    """
    import mujoco

    vertices = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body_id:
            continue
        if not visual_ok and model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0:
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        local = np.asarray(model.mesh_vert[start : start + count], np.float64).reshape(-1, 3)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        vertices.append(local @ rotation.reshape(3, 3).T + model.geom_pos[geom])
    return np.vstack(vertices) if vertices else np.zeros((0, 3))


def bounding_capsules(
    model,
    body_name: str,
    *,
    segments: int = 3,
    subsample: int = 4000,
) -> list:
    """Approximate a body's visible surface with a short chain of `UrdfCapsule`s.

    One capsule per body would be badly loose for a wide, flat shape: a single capsule containing the
    RB-Y1 base needs a ~30 cm radius, which would delete the floor around the robot along with the
    robot. Splitting the vertices into slabs along the principal axis and fitting one capsule per
    slab keeps each radius near the local half-width instead.

    Reuses `geometry.fit_capsule`, so the containment guarantee is the same one the rest of AG3S
    relies on: radius is the maximum distance from the segment, not an average.
    """
    import mujoco

    from benchmark.ag3s.stages.geometry import fit_capsule
    from benchmark.ag3s.robot_models.urdf_sphere_chain import UrdfCapsule

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"no body {body_name!r}")
    points = _body_vertices(model, body_id)
    if points.shape[0] < 8:
        return []
    if points.shape[0] > subsample:  # deterministic stride, as everywhere else in AG3S
        points = points[np.unique(np.linspace(0, points.shape[0] - 1, subsample).astype(np.int64))]

    centred = points - points.mean(axis=0)
    cov = (centred.T @ centred) / max(points.shape[0] - 1, 1)
    axis = np.linalg.eigh(cov)[1][:, 2]
    projection = centred @ axis
    edges = np.linspace(projection.min(), projection.max(), max(int(segments), 1) + 1)

    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        selected = (projection >= lo) & (projection <= hi if i == len(edges) - 2 else projection < hi)
        slab = points[selected]
        if slab.shape[0] < 4:
            continue
        capsule = fit_capsule(slab)
        origin = np.eye(4)
        origin[:3, :3] = capsule.orientation
        origin[:3, 3] = capsule.center
        out.append(
            UrdfCapsule(
                link=body_name,
                origin=origin,
                radius=float(capsule.dimensions[0]),
                length=float(2.0 * capsule.dimensions[1]),
            )
        )
    return out


#: Links the RB-Y1 URDF gives no collision capsule for, but which a camera on the robot sees.
#: `link_head_*` is excluded on purpose: the ZED is mounted on it, so it is never in its own view.
UNCOVERED_LINKS = (
    "base", "wheel_r", "wheel_l", "link_torso_3",
    "link_right_arm_6", "link_left_arm_6",
    "ee_finger_r1", "ee_finger_r2", "ee_finger_l1", "ee_finger_l2",
)


def gap_filling_capsules(model, links: Sequence[str] = UNCOVERED_LINKS, **kwargs) -> list:
    """`extra_capsules` for `UrdfSphereChain`, covering what the URDF's arm-only model omits."""
    out = []
    for link in links:
        try:
            out.extend(bounding_capsules(model, link, **kwargs))
        except KeyError:
            continue  # a link the simulator does not model separately
    return out


def camera_observation(
    scene: "TransportScene",
    camera: str,
    robot_model,
    *,
    timestamp: float = 0.0,
    attention_map=None,
):
    """A `CameraObservation` placed by **forward kinematics at capture time**, not by a fixed pose.

    The hand-eye transform is solved rather than assumed:

        T_link_cam = inv(T_base_link_urdf(q)) @ T_base_cam_mujoco(q)

    Deriving it from the URDF's own link pose — not MuJoCo's — is what makes the result
    self-consistent: `resolve_T_base_cam` will later compose it with the same URDF FK, so at the
    capture configuration it reproduces MuJoCo exactly, and at any other configuration it follows the
    kinematics AG3S actually has. Any disagreement between the URDF and MuJoCo chains is absorbed
    into the calibration, which is precisely what a real hand-eye calibration does; `link_pose_error`
    measures that disagreement separately so it is visible rather than hidden.
    """
    from benchmark.ag3s.types import CameraID, CameraObservation

    if camera not in CAMERA_MOUNTS:
        raise KeyError(f"no mount recorded for camera {camera!r}; have {sorted(CAMERA_MOUNTS)}")
    mount_link, camera_id = CAMERA_MOUNTS[camera]
    frame = scene.capture(camera)
    T_base_link = robot_model.link_pose(frame.robot_state, mount_link)
    return CameraObservation(
        camera_id=CameraID.parse(camera_id),
        depth=frame.depth,
        camera_intrinsics=frame.camera_intrinsics,
        mount_link=mount_link,
        T_link_cam=np.linalg.inv(T_base_link) @ frame.T_base_cam,
        robot_state=frame.robot_state,
        timestamp=float(timestamp),
        attention_map=attention_map,
        image_hw=frame.hw,
    ), frame


def link_pose_error(scene: "TransportScene", robot_model, links: Sequence[str]) -> dict[str, float]:
    """Position disagreement between the URDF chain and MuJoCo's, per link, in metres.

    Worth reporting rather than assuming zero. The two models are built from the same robot but not
    from the same file, and a centimetre of disagreement in a camera mount is a centimetre of
    registration error in that camera's cloud once the arm leaves the calibration pose.
    """
    q = scene.robot_state()
    out: dict[str, float] = {}
    for link in links:
        try:
            mujoco_pose = scene.body_pose(link)
        except KeyError:
            continue
        base = np.linalg.inv(scene.body_pose("base")) @ mujoco_pose
        out[link] = float(np.linalg.norm(robot_model.link_pose(q, link)[:3, 3] - base[:3, 3]))
    return out


__all__ = [
    "CAMERA_MOUNTS",
    "CameraFrame",
    "HEAD_JOINTS",
    "camera_observation",
    "link_pose_error",
    "UNCOVERED_LINKS",
    "bounding_capsules",
    "gap_filling_capsules",
    "DEFAULT_RBY1_JOINTS",
    "TRANSPORT_MODEL",
    "TransportScene",
    "gaussian_attention",
    "is_robot_body",
    "pixel_labels",
]
