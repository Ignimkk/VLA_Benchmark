"""Concrete `RobotCollisionModel` implementations.

These are *adapters*, not part of AG3S's algorithm. AG3S depends only on the
`benchmark.ag3s.types.RobotCollisionModel` Protocol; anything satisfying it — Pinocchio, MuJoCo, a
hand-written chain — can be injected instead. Nothing here is imported by the pipeline.
"""

from benchmark.ag3s.robot_models.urdf_sphere_chain import (
    DEFAULT_RBY1_JOINTS,
    RBY1_URDF,
    CoverageShortfall,
    UrdfCapsule,
    UrdfSphereChain,
    load_rby1,
    parse_urdf,
)

__all__ = [
    "DEFAULT_RBY1_JOINTS",
    "RBY1_URDF",
    "CoverageShortfall",
    "UrdfCapsule",
    "UrdfSphereChain",
    "load_rby1",
    "parse_urdf",
]
