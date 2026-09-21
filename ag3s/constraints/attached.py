"""The object in the gripper.

Once a grasp closes, the target stops being an obstacle in the world and becomes part of the moving
robot. The thing that must not hit the table is now the crate, and the crate's pose is a function of
the joints:

    T_base_object(q) = T_base_parent(q) @ T_parent_object

Three decisions here are safety properties rather than conveniences.

**Existence does not depend on perception.** The geometry is a *snapshot*, taken at the instant an
external caller confirmed the grasp. After that, occlusion, a grounding failure or a dropped camera
cannot remove it — the object is still in the hand whatever the cameras can see. Only an explicit
detach removes it. A system that dropped the attached geometry when tracking failed would forget it
was carrying something at exactly the moment it could no longer see.

**AG3S does not decide when a grasp succeeded.** There is no force threshold here, no gripper-width
heuristic, no state machine. `attach` is called by whoever knows.

**Contact is an allowlist.** The held object overlaps the fingers holding it, so those specific links
are excluded by name. The forearm, the torso and the opposite arm stay constrained, because an object
swinging into the robot's own elbow is a real collision. A link name nobody recognises is not in the
allowlist, so it stays constrained — the direction that fails closed.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Optional, Sequence

import numpy as np

from benchmark.ag3s.stages.geometry import to_spheres
from benchmark.ag3s.types import (
    AttachedCollisionGeometry,
    Primitive,
    RobotCollisionModel,
    TargetGeometry,
)


def _link_pose_numeric(robot_model: RobotCollisionModel, q: np.ndarray, link: str) -> np.ndarray:
    pose_fn = getattr(robot_model, "link_pose", None) or getattr(
        robot_model, "link_pose_numeric", None
    )
    if pose_fn is None:
        raise ValueError(
            "the injected robot model cannot compute link poses, so nothing can be attached to it"
        )
    return np.asarray(pose_fn(np.asarray(q, np.float64).reshape(-1), link), np.float64)


def attached_points_in_base(attached, *, robot_model: RobotCollisionModel,
                            robot_state: np.ndarray) -> Optional[np.ndarray]:
    """쥔 물체의 점을 **지금** base 좌표계 어디에 있는지로. 없으면 `None`.

    점은 parent link 프레임에 스냅샷돼 있으므로 (`AttachedCollisionGeometry.points`), 현재 자세의
    FK 한 번이면 지금 자리가 나온다 — 물체가 손에 **강체**로 붙어 있다는 것이 그 전제이고,
    그 전제는 실측으로 확인됐다 (손 319 mm 이동 동안 표면 어긋남 중앙 0.8 mm, E3).

    쓰는 곳이 둘이다: 최적화기가 여기에 거리를 묻고(`sphere_states`), 필드가 여기를 **파낸다**
    (A2). 같은 점 집합이어야 "질의점으로는 있고 장애물로는 없다" 가 성립한다.
    """
    if attached is None:
        return None
    pts = getattr(attached, "points", None)
    if pts is None or not len(pts):
        return None
    T = _link_pose_numeric(robot_model, robot_state, attached.parent_link)
    return np.asarray(pts, np.float64) @ T[:3, :3].T + T[:3, 3]


def attach_from_target(
    target: TargetGeometry,
    *,
    robot_model: RobotCollisionModel,
    robot_state: np.ndarray,
    parent_link: str,
    allowed_contact_links: Iterable[str] = (),
    label: str = "attached_object",
    timestamp: float = 0.0,
    extra_primitives: Sequence[Primitive] = (),
    point_voxel: float = 0.010,
    point_trim_radius: float = 0.020,
    point_trim_min_neighbors: int = 3,
) -> AttachedCollisionGeometry:
    """Snapshot a grounded target as geometry riding on `parent_link`.

    The transform is inverted out of the current state at the moment of the call:

        T_parent_object = inv(T_base_parent(q_grasp)) @ T_base_object

    which is why the caller has to pass the configuration the grasp closed in. Using a later `q`
    would bake in however far the arm had moved since, placing the object at an offset that then
    travels with the hand forever.

    **Both the points and the primitives are snapshotted** (F19, `docs/AG3S_REVIEW_LOG.md`). The
    earlier contract kept only primitives, on the reasoning that points are evidence and a primitive
    is the conservative shape to constrain against. Measurement overturned that: fitting one sphere
    to an apple's observed cloud gives r = 57.4 mm, because the cloud includes the stem and the leaf,
    and that eats 13.2 mm of clearance on average -- 23.7 mm at worst -- against the basket it is
    being placed into. In one frame the sphere reports -1.2 mm where the points report +9.3 mm: a
    collision that is not there. So the points are what the distance field is queried with, and the
    primitives stay for consumers that need a closed volume.

    `points` are downsampled to `point_voxel` before being stored, because the raw cloud is ~15,000
    points and the optimizer evaluates them at every step of the horizon. 10 mm leaves 64-89 points
    and changes the reported clearance by less than 1 mm.
    """
    T_base_parent = _link_pose_numeric(robot_model, robot_state, parent_link)
    T_parent_base = np.linalg.inv(T_base_parent)

    primitives: list[Primitive] = []
    for primitive in [target.bounding_geometry, *extra_primitives]:
        centre = T_parent_base[:3, :3] @ primitive.center + T_parent_base[:3, 3]
        orientation = T_parent_base[:3, :3] @ primitive.orientation
        primitives.append(dataclasses.replace(primitive, center=centre, orientation=orientation))

    points_parent = None
    observed = getattr(target, "points", None)
    if observed is not None and len(observed):
        pts = np.asarray(observed, np.float64).reshape(-1, 3)
        pts = pts @ T_parent_base[:3, :3].T + T_parent_base[:3, 3]
        points_parent = voxel_unique(pts, point_voxel)
        keep = trim_outliers(points_parent, point_trim_radius, point_trim_min_neighbors)
        if keep.all() or int(keep.sum()) >= 4:
            # 4 점 미만만 남는 필터는 적용하지 않는다 — 물체가 통째로 사라지는 것보다
            # 잡티를 남기는 쪽이 안전하다.
            points_parent = points_parent[keep]

    return AttachedCollisionGeometry(
        points=points_parent,
        parent_link=str(parent_link),
        # Identity, because the primitives above are already expressed in the parent frame. The field
        # is kept rather than folded away so a caller can instead attach a known CAD model in its own
        # frame and supply the transform directly.
        T_parent_object=np.eye(4),
        primitives=primitives,
        allowed_contact_links=frozenset(str(link) for link in allowed_contact_links),
        source_candidate_id=int(target.id),
        attached_at=float(timestamp),
        label=label,
    )


def voxel_unique(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """One representative point per occupied voxel, first in input order.

    Deliberately the same rule as `reconstruction.voxel_downsample` -- first occurrence, not the
    centroid -- so the two never disagree about which point survives. Duplicated here rather than
    imported because `reconstruction` pulls in `config`, and this module is on the path that
    `types`-only consumers import.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if voxel_size <= 0.0 or pts.shape[0] == 0:
        return pts
    keys = np.floor(pts / float(voxel_size)).astype(np.int64)
    keys -= keys.min(axis=0)
    extent = keys.max(axis=0) + 1
    packed = (keys[:, 0] * extent[1] + keys[:, 1]) * extent[2] + keys[:, 2]
    _, first = np.unique(packed, return_index=True)
    first.sort()
    return pts[first]


def trim_outliers(points: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    """`radius` 안에 이웃이 `min_neighbors` 개 미만인 점을 버린다 — 잡티와 가는 돌출부.

    **이것은 여유거리를 벌려고 넣은 것이 아니다.** 사과의 꼭지·잎을 버리면 무엇이 달라지는지
    쟀더니 `run_0004` 담는 구간 세 프레임에서 **여유거리가 정확히 0.0 mm 바뀌었다.** 꼭지는
    위를 향하고 구속은 수평(바구니 내벽)이라 애초에 가장 가까운 점이 아니었기 때문이다. 버린
    점들은 언제나 최소 여유보다 멀었다 (39.7~71.6 mm 대 33.2~49.0 mm).

    꼭지가 실제로 비쌌던 것은 **단일 구를 맞출 때**였다 — 꼭지까지 덮느라 반지름이 57.4 mm 가
    됐다 (F19). 점 기반으로 옮긴 순간 그 비용은 이미 사라졌다. 여기 남기는 것은 depth 잡티를
    거르기 위해서고, 그래서 기본값이 보수적이다 (`min_neighbors=3` 은 위 측정에서 **한 점도**
    버리지 않았다 — 진짜 고립점만 걸린다).

    **버리는 것은 안전하지 않은 방향이다.** 가는 돌출부가 정말로 가장 가까운 점인 씬 — 벽을
    향한 가느다란 주둥이 같은 것 — 에서는 이것이 위험을 숨긴다. 그래서 몇 점을 버렸는지
    호출자가 볼 수 있게 남긴다.

    **전환 신호**: 버린 점의 최소 여유가 남긴 점의 최소 여유보다 작아지는 것.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if min_neighbors <= 0 or radius <= 0.0 or pts.shape[0] <= min_neighbors:
        return np.ones(pts.shape[0], bool)
    from scipy.spatial import cKDTree

    counts = cKDTree(pts).query_ball_point(pts, float(radius), return_length=True) - 1
    return counts >= int(min_neighbors)


def base_frame_points(
    attached: AttachedCollisionGeometry,
    robot_model: RobotCollisionModel,
    robot_state: np.ndarray,
) -> np.ndarray:
    """`(N, 3)` -- the held object's snapshot points in the base frame at configuration `q`.

    Empty `(0, 3)` when nothing was snapshotted, so a caller can concatenate unconditionally.
    """
    if attached.points is None or len(attached.points) == 0:
        return np.zeros((0, 3))
    R = attached.T_parent_object[:3, :3]
    t = attached.T_parent_object[:3, 3]
    local = np.asarray(attached.points, np.float64) @ R.T + t
    T = _link_pose_numeric(robot_model, robot_state, attached.parent_link)
    return local @ T[:3, :3].T + T[:3, 3]


def attached_spheres(attached: AttachedCollisionGeometry) -> list[tuple[np.ndarray, float]]:
    """`[(centre_in_parent_frame, radius)]` — what the constraint form actually consumes.

    Composition happens here rather than in the symbolic graph so the graph never has to carry a
    rotation matrix as a parameter: every sphere reduces to a point plus a radius in the parent
    frame, and the symbolic side needs only `R_parent(q) @ c + t_parent(q)`. Nothing is lost, because
    the constraint is sphere-versus-sphere either way.
    """
    R = attached.T_parent_object[:3, :3]
    t = attached.T_parent_object[:3, 3]
    out: list[tuple[np.ndarray, float]] = []
    for primitive in attached.primitives:
        for centre, radius in to_spheres(primitive):
            out.append((R @ np.asarray(centre, np.float64) + t, float(radius)))
    return out


def base_frame_spheres(
    attached: AttachedCollisionGeometry,
    robot_model: RobotCollisionModel,
    robot_state: np.ndarray,
) -> list[tuple[np.ndarray, float]]:
    """The held object's spheres in the base frame at configuration `q`. For diagnostics and plots."""
    T = _link_pose_numeric(robot_model, robot_state, attached.parent_link)
    return [
        (T[:3, :3] @ centre + T[:3, 3], radius) for centre, radius in attached_spheres(attached)
    ]


def rigid_spheres(
    attached: AttachedCollisionGeometry,
    robot_model: RobotCollisionModel,
    robot_state: np.ndarray,
    free_joints: Sequence[int],
    *,
    trials: int = 24,
    scale: float = 0.30,
    tolerance: float = 1e-6,
    seed: int = 0,
) -> np.ndarray:
    """`(S,)` bool — **최적화가 쥔 물체와의 거리를 바꿀 수 없는** 로봇 구.

    쥔 물체를 로봇 자신과도 충돌 검사하려면 (cuRoboV2 §6.2 의 map-reduce self-collision) 어떤
    쌍을 빼야 하는지부터 정해야 한다. `self_collision_mask` 의 손끝 허용목록만으로는 **부족하다**
    — 실측(`run_0004`, RB-Y1): 쥔 사과가 `link_left_arm_5` 의 구 안에 **-56.1 mm** 들어가 있고,
    그 값이 파지 구간 9 프레임 내내 **한 번도 변하지 않는다.** 손목 전체가 물체와 함께 움직이기
    때문이다.

    그런 쌍에 제약을 걸면 **영원히 못 푸는 행**이 된다. 실측으로 그런 구가 **27 개**(손끝 22 +
    `link_left_arm_5` 5)이고 그중 **24 개**가 50 mm 를 못 지킨다.

    그래서 허용목록을 손으로 쓰지 않고 **기구학에서 뽑는다**: 최적화가 만질 수 있는 관절만
    흔들어 보고, 거리가 전혀 변하지 않는 구를 뺀다. 이것은 MoveIt 의 Allowed Collision Matrix 를
    이름이 아니라 **자유도**로 유도하는 것이고, 링크 이름이 바뀌거나 그리퍼가 교체되어도
    따라온다.

    `free_joints` 는 `ChunkLayout.q_indices` — 최적화가 실제로 바꾸는 관절이다. 고정 관절
    (RB-Y1 의 `arm_6` 과 토르소)을 함께 흔들면 움직일 수 없는 쌍을 움직인다고 잘못 판정한다.
    처음에 그렇게 재서 `link_left_arm_5` 를 "움직인다" 로 분류했고, 실제 롤아웃의 상수값이
    그것을 잡아냈다.
    """
    q0 = np.asarray(robot_state, np.float64).reshape(-1)
    free = np.asarray(free_joints, np.int64).reshape(-1)
    if attached.points is None or len(attached.points) == 0 or free.size == 0:
        return np.zeros(int(robot_model.n_spheres), bool)

    def gaps(q: np.ndarray) -> np.ndarray:
        centres, radii = robot_model.sphere_centers_numeric(q)
        pts = base_frame_points(attached, robot_model, q)
        return np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=2).min(axis=0) - radii

    rng = np.random.default_rng(seed)
    samples = [gaps(q0)]
    for _ in range(int(trials)):
        q = q0.copy()
        q[free] += rng.normal(scale=float(scale), size=free.size)
        samples.append(gaps(q))
    stack = np.asarray(samples)
    return (stack.max(axis=0) - stack.min(axis=0)) < float(tolerance)


def self_collision_mask(
    attached: AttachedCollisionGeometry, sphere_link_names: Sequence[str]
) -> np.ndarray:
    """`(S,)` in {0, 1}: which robot spheres the held object must still be kept away from.

    Zero only for links the caller explicitly allowlisted — the fingers actually holding the object.
    Everything else is one, including any link name the allowlist does not mention, which is how an
    unknown or misspelled link fails closed rather than being quietly excused.
    """
    allowed = attached.allowed_contact_links
    return np.asarray(
        [0.0 if str(name) in allowed else 1.0 for name in sphere_link_names], np.float64
    )


__all__ = [
    "attach_from_target",
    "attached_spheres",
    "base_frame_points",
    "base_frame_spheres",
    "rigid_spheres",
    "self_collision_mask",
    "trim_outliers",
    "voxel_unique",
]
