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
from typing import Iterable, Sequence

import numpy as np

from benchmark.ag3s.geometry import to_spheres
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
) -> AttachedCollisionGeometry:
    """Snapshot a grounded target as geometry riding on `parent_link`.

    The transform is inverted out of the current state at the moment of the call:

        T_parent_object = inv(T_base_parent(q_grasp)) @ T_base_object

    which is why the caller has to pass the configuration the grasp closed in. Using a later `q`
    would bake in however far the arm had moved since, placing the object at an offset that then
    travels with the hand forever.

    Only the *primitives* are snapshotted, not the point cloud. Points are evidence about where the
    object was; primitives are the conservative shape a constraint can be written against, and the
    shape is what has to keep moving with the hand.
    """
    T_base_parent = _link_pose_numeric(robot_model, robot_state, parent_link)
    T_parent_base = np.linalg.inv(T_base_parent)

    primitives: list[Primitive] = []
    for primitive in [target.bounding_geometry, *extra_primitives]:
        centre = T_parent_base[:3, :3] @ primitive.center + T_parent_base[:3, 3]
        orientation = T_parent_base[:3, :3] @ primitive.orientation
        primitives.append(dataclasses.replace(primitive, center=centre, orientation=orientation))

    return AttachedCollisionGeometry(
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
    "base_frame_spheres",
    "self_collision_mask",
]
