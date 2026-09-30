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
import logging
import pathlib
from typing import Optional, Sequence

import numpy as np

_LOG = logging.getLogger(__name__)

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
    link: Optional[str] = None,
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

    **Nothing here is a hand-chosen dimension.** Radius and length come out of the body's own mesh
    vertices, so a capsule is as big as the surface the camera actually sees and no bigger. The only
    knob is `segments`, and it is left at 3 everywhere: at 4 the slab boundaries on `EE_BODY_L` land
    inside the gripper's wide cheeks and one slab's radius jumps from ~33 mm to 64 mm, which is
    exactly the over-deletion this split-into-slabs scheme exists to avoid.

    `link` names the **URDF** link the capsule is attached to, when the two models spell the same
    part differently. `body_name` is always the MuJoCo name (that is what gets measured); `link`
    defaults to it and only has to be given for the aliased bodies in `MJCF_BODY_ALIASES`.
    """
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"no body {body_name!r}")
    link_name = body_name if link is None else link
    points = _body_vertices(model, body_id)
    return _capsules_from_points(points, link_name, segments=segments, subsample=subsample)


def _capsules_from_points(points: np.ndarray, link_name: str, *, segments: int = 3,
                          subsample: int = 4000) -> list:
    """`bounding_capsules` 의 몸통 — 점(이미 link frame)을 PCA 주축 slab 으로 나눠 slab 마다 capsule.

    T29 가 이 부분을 떼어 냈다: 부모 URDF link 의 frame 으로 옮긴 **다른 body 의** 정점(손목 카메라
    bracket · D435i)에도 같은 규칙을 쓰기 위해서다. 동작은 예전과 글자 그대로 같다.
    """
    from benchmark.ag3s.stages.geometry import fit_capsule
    from benchmark.ag3s.robot_models.urdf_sphere_chain import UrdfCapsule

    points = np.asarray(points, np.float64).reshape(-1, 3)
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
                link=link_name,
                origin=origin,
                radius=float(capsule.dimensions[0]),
                length=float(2.0 * capsule.dimensions[1]),
            )
        )
    return out


#: MuJoCo body name for the links the two robot files spell differently, keyed by the **URDF** name.
#:
#: The URDF and the MJCF were exported by different tools and do not agree on every name. The
#: gripper palm is `ee_left`/`ee_right` in the URDF and `EE_BODY_L`/`EE_BODY_R` in the MJCF, and both
#: spellings have to be right at the same time, for two different reasons:
#:
#: * `bounding_capsules` measures the mesh, so it needs the **MuJoCo** name — `mj_name2id` returns
#:   -1 for `ee_left`.
#: * `UrdfSphereChain` attaches the capsule by FK and rejects an `extra_capsules` entry whose `link`
#:   is not a URDF link (`robot_models/urdf_sphere_chain.py:393-395`) — `EE_BODY_L` is not one.
#:
#: Getting it wrong in the MuJoCo direction used to fail **silently**: `gap_filling_capsules`
#: swallows the `KeyError` below, so the link simply produced no capsule and the self-filter kept a
#: hole where nobody thought there was one. That is how `ee_left`/`ee_right` stayed missing.
#: **The swallow stays** — a different simulator may legitimately not model a link — but it no longer
#: keeps quiet: `gap_filling_capsules` warns, by name, for every listed link that yields nothing.
#:
#: Same trap, not yet triggered: URDF `FT_sensor_L`/`FT_sensor_R` are `FT_SENSOR_L`/`FT_SENSOR_R` in
#: the MJCF. Neither is in `UNCOVERED_LINKS`, so nothing depends on it today — and **that is exactly
#: the case the warning is for**: the moment someone adds `FT_sensor_L` to `UNCOVERED_LINKS` without
#: an alias here, the first `gap_filling_capsules` call says so instead of handing back a self-filter
#: that is quietly one body short.
#: **왼손가락 둘은 이름이 서로 바뀌어 있다** (2026-09-28, T11). 이름 불일치이고 기하 불일치가
#: 아니다 — 측정이 그것을 결정했다:
#:
#:     URDF ee_finger_l1  vs  MJCF ee_finger_l1 :  6.0000 mm | 180.000 deg
#:     URDF ee_finger_l1  vs  MJCF ee_finger_l2 :  0.0005 mm |   0.000 deg   <- 정상 잔차
#:
#: `link_*_arm_6` 기준으로 URDF 는 양팔 모두 `finger1` 을 `x = +3 mm` · 회전 0° 에 두는데,
#: MJCF 는 **왼손만** 거울이다 (`l1` 이 `x = −3 mm` · z 축 180°). 6.000 mm 는 `|+3 − (−3)|` 다.
#: 오른손은 일치하고 손바닥도 일치한다.
#:
#: **어느 파일도 틀리지 않았다.** 관절 축·리밋까지 양쪽이 각자 일관되고 (finger1 = axis `−x` ·
#: `[−0.05, 0]`, finger2 = axis `+x` · `[0, 0.05]`), frame 과 축이 함께 뒤집혀 있어 **물리적 개폐
#: 동작이 같다.** 그래서 고칠 기하가 없다 — 고칠 것은 두 이름표의 짝이고, 이 표가 정확히 그
#: 일을 하는 곳이다 (`ee_left → EE_BODY_L` 이 이미 같은 일을 한다).
#:
#: URDF 쪽 joint origin 을 MJCF 에 맞추는 길(그쪽이 (B) 였다)을 **안 고른 이유**: vendored robot
#: description 을 고치면 FK 가 바뀌어 **제약 모델의 구 위치가 전부 움직이고 회귀 기준선이 또
#: 이동한다.** 이름표 두 줄로 끝나는 일에 그 대가를 낼 이유가 없다.
MJCF_BODY_ALIASES: dict[str, str] = {
    "ee_left": "EE_BODY_L",
    "ee_right": "EE_BODY_R",
    # 왼손가락 둘 — 위 주석. **두 줄이 서로를 가리킨다**: URDF l1 의 mesh 는 MJCF l2 에서 재야
    # 하고, 그 반대도 같다.
    "ee_finger_l1": "ee_finger_l2",
    "ee_finger_l2": "ee_finger_l1",
}

#: Links the RB-Y1 URDF gives no collision capsule for, but which a camera on the robot sees.
#: Named the way the **URDF** names them; `MJCF_BODY_ALIASES` translates where the MJCF disagrees.
#: `link_head_*` is excluded on purpose: the ZED is mounted on it, so it is never in its own view.
#:
#: `ee_left`/`ee_right` (the gripper palm, `EE_BODY_L`/`EE_BODY_R` in the MJCF) were missing until
#: 2026-09-25. The fingers were listed but the lump they hang off was not, and a wrist camera is
#: bolted to the same wrist link, so it stares straight at its own palm on every frame of every
#: episode — measured against ground-truth segmentation, `EE_BODY_L`/`EE_BODY_R` leaked 90,690 and
#: 90,688 px, 100 % of it into `wrist_cam_l`/`wrist_cam_r` respectively and 0 elsewhere.
#:
#: Raising `self_filter_inflation` does not reach them. Inflation grows spheres that exist; these
#: bodies have none, because the URDF gives `ee_*`, `link_*_arm_6`, `base` and `wheel_*` a visual
#: element and zero `<collision>` elements. Filling the gap from the simulator's meshes is the only
#: route, which is what this tuple is.
UNCOVERED_LINKS = (
    "base", "wheel_r", "wheel_l", "link_torso_3",
    "link_right_arm_6", "link_left_arm_6",
    "ee_left", "ee_right",
    "ee_finger_r1", "ee_finger_r2", "ee_finger_l1", "ee_finger_l2",
)


#: `(link, body, reason)` already warned about in this process. `gap_filling_capsules` is a
#: setup-time call, but sweeps rebuild the robot model once per configuration
#: (`experiments/live/sweep_attached_threshold.py:110`), so an undeduplicated warning would
#: repeat once per sweep iteration and train the reader to skip it. Tests that assert on the
#: warning clear this.
_WARNED_NO_CAPSULES: set[tuple[str, str, str]] = set()


def _warn_no_capsules(link: str, body: str, reason: str, *, empty: int, listed: int) -> None:
    """**삼키되 크게 말한다.** 캡슐 0 개는 조용히 지나가면 안 되는 사건이다.

    Swallowing is still right — `links` is written for *this* robot and another simulator may not
    model, say, a wheel as its own body, so raising would break scenes that are merely different.
    What was wrong is that the swallow said nothing: a link listed in `UNCOVERED_LINKS` is there
    because someone measured a leak through it, so zero capsules means the leak is back and the
    self-filter has a hole nobody is looking at. That is how `ee_left`/`ee_right` — 90,690 and
    90,688 px straight into the wrist cameras — stayed missing.

    `warning`, not `error`, and for the same reason the swallow stays: a legitimately absent link
    must not read as a fault. It matches `trajopt/serve_safe.py:79`, which is the same shape — a
    configuration that will silently produce wrong numbers rather than a crash.

    The message has to carry **which spelling was tried**. That is the whole diagnosis: looked up
    under the MJCF name and still missing means the body is really absent; looked up under the URDF
    name means `MJCF_BODY_ALIASES` is what needs the entry.
    """
    key = (link, body, reason)
    if key in _WARNED_NO_CAPSULES:
        return
    _WARNED_NO_CAPSULES.add(key)
    spelling = (
        f"MJCF spelling {body!r} from MJCF_BODY_ALIASES"
        if body != link
        else f"URDF spelling {body!r} verbatim — no MJCF_BODY_ALIASES entry"
    )
    _LOG.warning(
        "self-filter gap NOT filled for link %r: %s, %s. "
        "%d of %d listed links produced no capsule. "
        "The link stays in UNCOVERED_LINKS and gets no sphere, so the self-filter keeps a "
        "hole there and everything the cameras see of it survives into the obstacle cloud. "
        "If the body exists in the MJCF under another spelling, add it to MJCF_BODY_ALIASES "
        "(`experiments/sources/mujoco_source.py` — `FT_sensor_L`/`FT_SENSOR_L` is the next "
        "one waiting).",
        link, spelling, reason, empty, listed,
    )


def gap_filling_capsules(model, links: Sequence[str] = UNCOVERED_LINKS, **kwargs) -> list:
    """`extra_capsules` for `UrdfSphereChain`, covering what the URDF's arm-only model omits.

    A link that yields **no** capsule is swallowed, as before, but warned about by name — see
    `_warn_no_capsules` for why that is the right pair. Both silent-zero routes are covered: a
    `KeyError` (no MuJoCo body under that name) and an empty return from `bounding_capsules` (the
    body is there but carries no mesh to measure).
    """
    links = tuple(links)
    out: list = []
    empty: list[tuple[str, str, str]] = []
    for link in links:
        body = MJCF_BODY_ALIASES.get(link, link)
        try:
            capsules = bounding_capsules(model, body, link=link, **kwargs)
        except KeyError:  # a link the simulator does not model separately
            capsules = []
            reason = "mj_name2id found no such body"
        else:
            reason = "the body has no mesh geoms to measure (too few vertices to fit a capsule)"
        if capsules:
            out.extend(capsules)
        else:
            empty.append((link, body, reason))
    for link, body, reason in empty:
        _warn_no_capsules(link, body, reason, empty=len(empty), listed=len(links))
    return out


#: T29 — MJCF bodies whose visible surface the self-filter spheres **did not cover** at
#: `self_filter_inflation = 0` (T28 P1-3b, GT segmentation, every planning row of T14 / T17, px):
#:
#: | body | leak T14 / T17 | why |
#: |---|---:|---|
#: | `link_left_arm_2` | 24,697 / 22,034 | URDF 에 capsule 이 둘뿐이다 — `link_*_arm_1` 의 위팔 capsule (축 z ∈ [−0.2, 0], r 50) 과 `link_*_arm_2` 의 팔꿈치 capsule (y 축, 중심 x +26 · z −276 mm, r 50, L 80). **그 사이 z −200…−280 mm 의 위팔 아래쪽이 비어 있다** — mesh 는 x −45…+71 · y ±67 mm 인데 팔꿈치 capsule 은 x +26 에 치우쳐 x < −24 mm 를 못 덮는다. 정점 11 % 가 구 밖, 최대 14.0 mm (T14/T17 자세 전부) |
#: | `wrist_bracket_l/r` | 2,661 / 6,879 · 3,920 / 9,405 | URDF link 이 아니다 (MJCF 전용 카메라 bracket, `link_*_arm_6` 아래) — 자기 구가 없다. 3.1 % · 9.2 mm |
#: | `d435i_body_l/r` | 1,777 / 3,648 · 2,468 / 6,881 | 같은 이유 (D435i, bracket 아래). 34 % · 10.7 mm |
#: | `link_left_arm_3` | 429 / 4,581 | URDF capsule (x −31 mm, r 40) 보다 mesh 가 넓다 (x −68 mm). 2.9 % · 9.5 mm |
#: | `link_left_arm_4` | 26 / 241 | 0.2 mm — 경계 잔차 |
#:
#: 오른팔의 같은 link 은 T14/T17 에 안 보였지만 거울 기하라 같이 덮는다 (다른 에피소드에서 보인다).
#: `link_torso_5` 도 정점 28 % 가 구 밖이지만 (최대 50 mm) 머리 카메라가 그 위에 달려 있어 어느
#: 카메라에도 안 보였다 (누수 0 px) — 넣지 않는다.
#:
#: **`link_*_arm_5` 는 더하는 것이 아니라 바꾼다** (`SELF_FILTER_MESH_LINKS`). T17 에서 사과가 통째로
#: 지워진 17 프레임 (seq 12–28) 의 사과 px 는 **전부 `link_left_arm_5` 의 구 안**이었다
#: (seq 12: 17,277 / 17,277 px; 손가락 구는 665 px, 손바닥 236 px — 손가락 관절을 실제 개도로
#: 놓아도 이 17 프레임은 그대로 남았다). 그 URDF capsule 은 r 75 mm · L 250 mm, 축 z ∈
#: [−0.225, +0.025] 로 손목·손바닥·손가락과 **그 사이의 물체**까지 삼킨다 (`CapsuleTrim` 머리말).
#: 자기 mesh 는 z ≥ −82 mm 에서 끝난다.
#:
#: 그것을 빼자 **그 capsule 이 가리고 있던 구멍**이 드러났다: 손바닥 (`EE_BODY_L/R`, 누수 T14
#: 140,675 / 148,929 px) 과 손가락 (1,420–2,032 px) 의 `gap_filling_capsules` 는 **정점**에 맞춘
#: slab capsule (4000 점 솎음) 이라 slab 경계를 가로지르는 삼각형의 안쪽 면이 최대 7 mm 밖에 있다
#: (wrist_cam_l 에서 손바닥 depth 점의 45 % 가 구 밖, 그 점들은 mesh 표면에서 ≤ 1.6 mm). 그래서
#: 손바닥 · 손가락 · `link_*_arm_6` · FT sensor 도 목록에 있다. 이들의 gap-filler 는 **그대로 두고**
#: (제약 모델과 T26 max opening 이 그것을 쓴다) 덮개를 더한다.
#:
#: 이름은 **MJCF body** 이름이다. capsule 이 붙는 URDF link 은 `static_urdf_ancestor` 가 찾는다
#: (`EE_BODY_L` → `ee_left`, MJCF `ee_finger_l1` → URDF `ee_finger_l2` (이름 뒤바뀜),
#: `wrist_bracket_l` · `d435i_body_l` · `FT_SENSOR_L` → `link_left_arm_6`).
#:
#: **T30 (F5) — `base` 추가.** base mesh 표면 점 2,116,268 개 중 31,730 개(1.5 %)가 URDF 구 14 개
#: 밖에 있다(최대 53.6 mm). 회귀 기준선 `run_16d_ep1800` 15 프레임 × 3 카메라의 GT 에서는 그 자리가
#: 안 보여서 누수가 0 px 이었다(base 33,123 px 보임, 0 px 샘). 그래도 넣는 이유는 다른 자세·카메라에서
#: 보일 수 있기 때문이다. 기준선에서 inflation 0 과 0.05 사이에 갈리는 px 는 base 가 아니라 **crate** 다
#: (frame0 wrist_cam_l 2,787 · wrist_cam_r 569 px). 0.05 가 왼손 옆의 실제 crate 를 지우고 있었던
#: 것이므로 덮개로 되돌릴 대상이 아니다(T30b.impl).
#: `wheel_*` 는 표면이 전부 구 안이다(최대 −0.9 mm). `link_torso_0` 은 3,933 점이 최대 3.2 mm 밖으로
#: 나오지만 누수가 0 이라 넣지 않았다.
SELF_FILTER_COVER_BODIES = (
    "base",
    "link_left_arm_2", "link_right_arm_2",
    "link_left_arm_3", "link_right_arm_3",
    "link_left_arm_4", "link_right_arm_4",
    "link_left_arm_5", "link_right_arm_5",
    "link_left_arm_6", "link_right_arm_6",
    "FT_SENSOR_L", "FT_SENSOR_R",
    "wrist_bracket_l", "wrist_bracket_r",
    "d435i_body_l", "d435i_body_r",
    "EE_BODY_L", "EE_BODY_R",
    "ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2",
)

#: URDF capsule 을 **빼고** mesh capsule 만 쓰는 link (자기 필터 모델만) — 위 표.
SELF_FILTER_MESH_LINKS = ("link_left_arm_5", "link_right_arm_5")

#: 덮개 capsule 을 맞출 때 mesh 삼각형 표면을 이 간격(m)으로 채운다. 정점만 쓰면 slab 경계를
#: 가로지르는 큰 삼각형의 안쪽이 두 capsule 어디에도 안 들어간다 (bracket 이 정점 637 개라 실측
#: 95 / 236 px 가 그렇게 샜고, 손바닥은 140 k px).
SURFACE_SAMPLE_SPACING = 0.004

#: 덮개 capsule 반지름에 더하는 여유 (m). **inflation 이 아니다** — 덮개 구에만 붙는 기하의 일부이고
#: 표본 간격(4 mm 격자의 삼각형 안쪽 최대 ≈ 1.2 mm)과 depth 양자화(1 mm, `wire.DEPTH_SCALE_MM`)를
#: 흡수한다. 다른 구는 그대로다.
COVER_PAD = 0.002

#: 덮이지 않은 표면 점을 이 거리(m) 안에서 이어진 덩어리로 나눠 덩어리마다 capsule 을 맞춘다.
#: 한 body 의 양 끝에 흩어진 구멍을 capsule 하나로 이으면 그 사이 (손가락 사이 = 사과 자리) 까지
#: 덮는다.
COVER_CLUSTER_EPS = 0.010

#: T30 (F5): 덮개가 **자기 mesh 상자 밖으로 나가면 안 되는** body. base 에서 구 밖에 남는 표면은
#: 앞뒤 두 줄 구 사이의 띠 하나다(x −12…147 · y ±230 · z 36…175 mm, 연결 덩어리 1 개).
#: slab 3 개로 맞추면 capsule 이 r 134–141 mm 가 되어 base 옆으로 ~170 mm 튀어나간다
#: (y −432 mm, base 상자는 ±263). 그 자리의 바닥이나 물체를 로봇으로 지우게 된다. 그래서 이 body 들은
#: 덮개 capsule 의 AABB 가 body 표면 AABB(+`COVER_PAD`) 안에 들 때까지 점을 주축으로 반씩 나눠
#: 다시 맞춘다(`_fit_confined`). 숫자 상한이 아니라 기하 규칙이다. T29 의 22 body 는
#: 누수 표로 검증된 덮개라 바꾸지 않는다.
COVER_CONFINED_BODIES = ("base",)
#: `_fit_confined` 의 최대 분할 깊이 (2^8 = 256 조각). 그 깊이에서도 상자를 넘으면 그대로 둔다
#: (덮개가 빠지는 것보다 조금 튀어나오는 쪽이 낫다 — 담는 것이 먼저다).
COVER_CONFINE_MAX_DEPTH = 8


def _body_surface_points(model, body_id: int, spacing: float = SURFACE_SAMPLE_SPACING
                         ) -> np.ndarray:
    """Mesh vertices **and** points on every triangle (barycentric grid ≤ `spacing`), body frame.

    Visual geoms included, like `_body_vertices` — the camera sees them.
    """
    import mujoco

    out = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body_id or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[geom])
        v0, nv = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        f0, nf = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        verts = np.asarray(model.mesh_vert[v0:v0 + nv], np.float64).reshape(-1, 3)
        faces = np.asarray(model.mesh_face[f0:f0 + nf], np.int64).reshape(-1, 3)
        pts = [verts]
        if faces.size:
            a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
            edge = np.max(np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                                    np.linalg.norm(a - c, axis=1)], 1), axis=1)
            n = np.ceil(edge / float(spacing)).astype(np.int64)
            for k in np.unique(n[n >= 2]):
                sel = n == k
                ij = np.array([(i, j) for i in range(k + 1) for j in range(k + 1 - i)], np.float64) / k
                w = np.column_stack([1.0 - ij.sum(1), ij])                   # (m, 3)
                tri = np.stack([a[sel], b[sel], c[sel]], 1)                   # (t, 3, 3)
                pts.append(np.einsum("mk,tkj->tmj", w, tri).reshape(-1, 3))
        local = np.vstack(pts)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        out.append(local @ rotation.reshape(3, 3).T + model.geom_pos[geom])
    return np.vstack(out) if out else np.zeros((0, 3))


def _body_to_parent(model, body_id: int) -> np.ndarray:
    """4×4 constant pose of `body_id` in its parent's frame (`body_pos` · `body_quat`)."""
    import mujoco

    T = np.eye(4)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, model.body_quat[body_id])
    T[:3, :3] = R.reshape(3, 3)
    T[:3, 3] = model.body_pos[body_id]
    return T


def static_urdf_ancestor(model, body_name: str, urdf_links: Sequence[str]) -> tuple[str, np.ndarray]:
    """`(URDF link, T_link_body)` — the nearest ancestor that the URDF knows, **rigidly** attached.

    Walks up the MJCF tree from `body_name` until a body whose name (through `MJCF_BODY_ALIASES`,
    reversed) is a URDF link. Every body on the way must have **no joint** — otherwise the offset
    is not constant and a capsule fixed in the ancestor's frame would be wrong in every pose but
    one, so this raises instead. A URDF link itself returns `(link, I)`.
    """
    import mujoco

    reverse = {mj: urdf for urdf, mj in MJCF_BODY_ALIASES.items()}
    known = set(urdf_links)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise KeyError(f"no body {body_name!r}")
    T = np.eye(4)
    cursor = bid
    while True:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cursor) or ""
        urdf = reverse.get(name, name)
        if urdf in known:
            return urdf, T
        if int(model.body_jntnum[cursor]) != 0:
            raise ValueError(
                f"{body_name!r}: ancestor {name!r} has a joint before any URDF link is reached — "
                "a capsule fixed in a URDF link frame would not follow it")
        parent = int(model.body_parentid[cursor])
        if parent == cursor or parent == 0:
            raise ValueError(f"{body_name!r}: no URDF link among its MJCF ancestors")
        T = _body_to_parent(model, cursor) @ T
        cursor = parent


#: 덮개 capsule 캐시 — 표면 표본 + 솎지 않은 fit 이 body 당 0.3–0.8 s 라 (D435i 정점 49.7 만 개)
#: 모델을 짓는 곳마다 다시 하면 테스트·sweep 가 느려진다. 키는 body 의 **mesh 내용 digest** 와
#: 조상 변환이므로 다른 모델의 같은 이름 body 가 옛 값을 받지 않는다.
_COVER_CACHE: dict = {}


def _body_mesh_digest(model, body_id: int) -> str:
    import hashlib

    import mujoco

    h = hashlib.sha1()
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body_id or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[geom])
        v0, nv = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        f0, nf = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        h.update(np.ascontiguousarray(model.mesh_vert[v0:v0 + nv]).tobytes())
        h.update(np.ascontiguousarray(model.mesh_face[f0:f0 + nf]).tobytes())
        h.update(np.ascontiguousarray(model.geom_pos[geom]).tobytes())
        h.update(np.ascontiguousarray(model.geom_quat[geom]).tobytes())
    return h.hexdigest()


def _clusters(points: np.ndarray, eps: float) -> list[np.ndarray]:
    """Connected components of `points` on an `eps` voxel grid (26-neighbourhood), index arrays.

    A grid rather than an `eps`-ball graph: the D435i surface alone is ~10^5 points and the pair
    list of a ball graph at 10 mm is quadratic in the local density.
    """
    from scipy import ndimage

    n = points.shape[0]
    if n == 0:
        return []
    cell = np.floor((points - points.min(axis=0)) / float(eps)).astype(np.int64)
    grid = np.zeros(tuple(cell.max(axis=0) + 1), bool)
    grid[cell[:, 0], cell[:, 1], cell[:, 2]] = True
    labels, count = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
    point_label = labels[cell[:, 0], cell[:, 1], cell[:, 2]]
    order = np.argsort(point_label, kind="stable")
    bounds = np.searchsorted(point_label[order], np.arange(1, count + 2))
    return [order[bounds[k]:bounds[k + 1]] for k in range(count) if bounds[k + 1] > bounds[k]]


def _capsule_distance(points: np.ndarray, capsule) -> np.ndarray:
    """Signed distance from `points` to a `UrdfCapsule` (link frame; negative = inside)."""
    a = capsule.origin[:3, 3] - capsule.origin[:3, 2] * (capsule.length / 2.0)
    b = capsule.origin[:3, 3] + capsule.origin[:3, 2] * (capsule.length / 2.0)
    ab = b - a
    denom = float(ab @ ab)
    t = np.zeros(points.shape[0]) if denom <= 1e-18 else np.clip((points - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1) - float(capsule.radius)


def _fit_cover(points: np.ndarray, link: str, segments: int = 3) -> list:
    """PCA slabs like `_capsules_from_points`, but **no slab is dropped** (a slab of 1–3 points gets
    its own `fit_capsule`/sphere) — a skipped slab is an uncovered patch."""
    from benchmark.ag3s.robot_models.urdf_sphere_chain import UrdfCapsule
    from benchmark.ag3s.stages.geometry import fit_capsule

    def one(pts):
        prim = fit_capsule(pts, min_radius=0.0)
        origin = np.eye(4)
        origin[:3, :3] = prim.orientation
        origin[:3, 3] = prim.center
        half = float(prim.dimensions[1]) if prim.type.value == "capsule" else 0.0
        return UrdfCapsule(link=link, origin=origin, radius=float(prim.dimensions[0]),
                           length=2.0 * half)

    if points.shape[0] < 8:
        return [one(points)]
    centred = points - points.mean(axis=0)
    axis = np.linalg.eigh((centred.T @ centred) / max(points.shape[0] - 1, 1))[1][:, 2]
    proj = centred @ axis
    edges = np.linspace(proj.min(), proj.max(), max(int(segments), 1) + 1)
    out = []
    for i in range(len(edges) - 1):
        sel = (proj >= edges[i]) & (proj <= edges[i + 1] if i == len(edges) - 2 else proj < edges[i + 1])
        if sel.any():
            out.append(one(points[sel]))
    return out


def _capsule_aabb(capsule) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box of a `UrdfCapsule` in its link frame."""
    half = capsule.origin[:3, 2] * (capsule.length / 2.0)
    a, b = capsule.origin[:3, 3] - half, capsule.origin[:3, 3] + half
    return np.minimum(a, b) - capsule.radius, np.maximum(a, b) + capsule.radius


def _fit_confined(points: np.ndarray, link: str, box: tuple[np.ndarray, np.ndarray], *,
                  slack: float, depth: int = 0) -> list:
    """`_fit_cover`, then every capsule whose AABB leaves `box` (± `slack`) is refit on its points
    split in half along their principal axis — recursively, up to `COVER_CONFINE_MAX_DEPTH` (T30 F5).
    Still containing: each piece is a max-distance fit of its own points."""
    lo, hi = np.asarray(box[0]) - slack, np.asarray(box[1]) + slack
    out = []
    for cap in _fit_cover(points, link, segments=1):
        c_lo, c_hi = _capsule_aabb(cap)
        if (np.all(c_lo >= lo) and np.all(c_hi <= hi)) or points.shape[0] < 8 \
                or depth >= COVER_CONFINE_MAX_DEPTH:
            out.append(cap)
            continue
        centred = points - points.mean(axis=0)
        axis = np.linalg.eigh(centred.T @ centred)[1][:, 2]
        proj = centred @ axis
        cut = float(np.median(proj))
        left, right = proj <= cut, proj > cut
        if not left.any() or not right.any():
            out.append(cap)
            continue
        out += _fit_confined(points[left], link, box, slack=slack, depth=depth + 1)
        out += _fit_confined(points[right], link, box, slack=slack, depth=depth + 1)
    return out


def _cover_capsules_for(points: np.ndarray, link: str, *, pad: float, eps: float,
                        box: Optional[tuple[np.ndarray, np.ndarray]] = None) -> list:
    """Capsules that contain **every** point: per `eps`-cluster, `_fit_cover`; then any point still
    outside (it should not happen, but a cover that silently misses is the failure T29 fixes) is
    covered on its own. Radius + `pad`, role `COVER_ROLE`.

    `box` (T30, `COVER_CONFINED_BODIES`): the body's own surface AABB in the link frame — the
    capsules are split until they stay inside it (`_fit_confined`)."""
    from benchmark.ag3s.robot_models.urdf_sphere_chain import COVER_ROLE, UrdfCapsule

    def fit(pts):
        return (_fit_cover(pts, link) if box is None
                else _fit_confined(pts, link, box, slack=pad))

    out = []
    for idx in _clusters(points, eps):
        cluster = points[idx]
        caps = fit(cluster)
        for _ in range(3):
            d = np.min(np.stack([_capsule_distance(cluster, c) for c in caps]), axis=0)
            missed = cluster[d > 1e-12]
            if not missed.shape[0]:
                break
            caps += [c for sub in _clusters(missed, eps) for c in fit(missed[sub])]
        out.extend(UrdfCapsule(link=c.link, origin=c.origin, radius=float(c.radius) + pad,
                               length=float(c.length), role=COVER_ROLE) for c in caps)
    return out


def self_filter_covering_capsules(model, urdf_links: Sequence[str],
                                  bodies: Sequence[str] = SELF_FILTER_COVER_BODIES,
                                  *, existing=None,
                                  spacing: float = SURFACE_SAMPLE_SPACING,
                                  pad: float = COVER_PAD,
                                  eps: float = COVER_CLUSTER_EPS) -> list:
    """`extra_capsules` that make the **self-filter** model cover `bodies` (T29). Role
    `COVER_ROLE` — `gripper_openings` does not read them.

    For each MJCF body:

    1. **surface** points (`_body_surface_points`: vertices + every triangle every `spacing`,
       visual geoms included — the camera sees them), moved into the frame of its nearest
       rigidly-attached URDF link (`static_urdf_ancestor`);
    2. with `existing` (a `UrdfSphereChain` that already has the link's own spheres), only the points
       **outside** that link's spheres are kept — this is gap filling, so a link the URDF already
       covers gets nothing and the cover stays where the holes are;
    3. the rest is split into `eps`-connected clusters and each cluster gets `fit_capsule`-style
       capsules (PCA slabs, max distance — they contain every point), radius + `pad`;
    4. (T30) a body in `COVER_CONFINED_BODIES` has its capsules split until each stays inside the
       body's own surface AABB (+ `pad`) — `_fit_confined`.

    Missing bodies are warned about and skipped, as in `gap_filling_capsules`.
    """
    import mujoco

    out: list = []
    empty: list[tuple[str, str, str]] = []
    for body in tuple(bodies):
        try:
            link, T = static_urdf_ancestor(model, body, urdf_links)
        except KeyError:
            empty.append((body, body, "mj_name2id found no such body"))
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        centres, radii = (np.zeros((0, 3)), np.zeros(0)) if existing is None \
            else existing.local_spheres(link)
        confined = body in COVER_CONFINED_BODIES
        key = (body, link, T.tobytes(), float(spacing), float(pad), float(eps),
               centres.tobytes(), radii.tobytes(), _body_mesh_digest(model, bid), confined)
        capsules = _COVER_CACHE.get(key)
        if capsules is None:
            pts = _body_surface_points(model, bid, spacing)
            if pts.shape[0]:
                pts = pts @ T[:3, :3].T + T[:3, 3]
            box = (pts.min(axis=0), pts.max(axis=0)) if confined and pts.shape[0] else None
            if pts.shape[0] and radii.size:
                from scipy.spatial import cKDTree

                inside = np.zeros(pts.shape[0], bool)
                tree = cKDTree(pts)
                for c, r in zip(centres, radii):
                    hits = tree.query_ball_point(c, float(r))
                    if hits:
                        inside[hits] = True
                pts = pts[~inside]
            capsules = (_cover_capsules_for(pts, link, pad=pad, eps=eps, box=box)
                        if pts.shape[0] else [])
            _COVER_CACHE[key] = capsules
        if capsules:
            out.extend(capsules)
        elif existing is None:
            empty.append((body, body, "the body has no mesh geoms to measure"))
    for link, body, reason in empty:
        _warn_no_capsules(link, body, reason, empty=len(empty), listed=len(tuple(bodies)))
    return out


def set_finger_joints_from_scene(scene: "TransportScene", robot_model) -> Optional[dict]:
    """Put the scene's **current** finger opening into `robot_model` (T29) and return the values.

    Read the client's way (`gripper_state.openings_from_mujoco` = `rby1_state()`), inverted to the
    URDF finger joints (`FingerJointMap`). A model without finger parameters is left alone (None).
    `camera_observation` calls this, so every offline tool that builds observations from a MuJoCo
    scene gets a self-filter whose fingers are where the rendered fingers are — the same thing
    `SafePolicy` does on the server from the request's 16D state.
    """
    if not tuple(getattr(robot_model, "param_joint_names", ()) or ()):
        return None
    from benchmark.ag3s.robot_models.gripper_state import FingerJointMap, openings_from_mujoco

    fmap = getattr(robot_model, "_t29_finger_map", None)
    if fmap is None:
        fmap = FingerJointMap(robot_model)
        robot_model._t29_finger_map = fmap
    values = fmap.joint_values(openings_from_mujoco(scene.model, scene.data, fmap.convention))
    robot_model.set_joint_parameters(values)
    return values


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
    set_finger_joints_from_scene(scene, robot_model)
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
    unmeasured: list[str] = []
    for link in links:
        # **alias 를 여기서도 쓴다** (2026-09-28). 안 쓰면 `ee_left`/`ee_right` 가 `KeyError` 로
        # 조용히 건너뛰어졌다 — 이 함수가 "frame 이 맞는가" 를 묻는 유일한 도구인데 **손바닥
        # 둘을 한 번도 안 재고 있었다.** 조용히 건너뛰는 것은 조용히 실패하는 것과 같다.
        try:
            mujoco_pose = scene.body_pose(MJCF_BODY_ALIASES.get(link, link))
        except KeyError:
            # 여전히 건너뛴다 (다른 시뮬레이터가 그 link 을 body 로 안 모델링할 수 있다). 다만
            # **못 쟀다고 말한다** — 반환 dict 에 키가 없는 것과 0 mm 는 완전히 다른 뜻인데,
            # 읽는 쪽은 둘을 구별할 근거가 없었다. 반환 타입은 그대로 `dict[str, float]` 다.
            unmeasured.append(link)
            continue
        base = np.linalg.inv(scene.body_pose("base")) @ mujoco_pose
        out[link] = float(np.linalg.norm(robot_model.link_pose(q, link)[:3, 3] - base[:3, 3]))
    if unmeasured:
        logging.getLogger(__name__).warning(
            "link_pose_error: %s 는 MJCF body 를 못 찾아 **재지 않았습니다** (반환 dict 에 키가 "
            "없습니다 — 0 mm 가 아닙니다). MJCF 이름이 다르면 MJCF_BODY_ALIASES 에 넣으십시오.",
            ", ".join(unmeasured))
    return out


__all__ = [
    "CAMERA_MOUNTS",
    "CameraFrame",
    "HEAD_JOINTS",
    "MJCF_BODY_ALIASES",
    "camera_observation",
    "link_pose_error",
    "UNCOVERED_LINKS",
    "SELF_FILTER_COVER_BODIES",
    "SELF_FILTER_MESH_LINKS",
    "COVER_PAD",
    "COVER_CONFINED_BODIES",
    "bounding_capsules",
    "gap_filling_capsules",
    "self_filter_covering_capsules",
    "set_finger_joints_from_scene",
    "static_urdf_ancestor",
    "DEFAULT_RBY1_JOINTS",
    "TRANSPORT_MODEL",
    "TransportScene",
    "gaussian_attention",
    "is_robot_body",
    "pixel_labels",
]
