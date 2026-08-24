"""AG3S — Attention-Guided 3D Safety Scene.

Attention decides **what the target is**. 3D geometry decides **what can collide**. Geometry with
low or zero attention is still a collision candidate if it physically exists; an attention score is
never used as an obstacle-detection threshold.

AG3S turns (depth or point cloud, camera intrinsics, extrinsics, robot state, attention map, phase)
into a `CollisionConstraintSet` — a CasADi-compatible, fixed-structure constraint specification a
trajectory optimizer can consume directly:

    VLA -> Action Chunk -> SEAM -> Reference Trajectory --.
                                                          +-> [TO: not implemented] -> Safe Chunk
    AG3S -> CollisionConstraintSet ----------------------'

The TO itself is deliberately out of scope here; AG3S's job ends at the interface contract, which
`tests/ag3s/test_to_contract.py` verifies against a minimal CasADi/IPOPT harness.

Imports here stay cheap and dependency-free (numpy only) so that `import benchmark.ag3s` cannot fail
for environment reasons. CasADi, scipy and matplotlib are imported lazily by the stages that need
them.
"""

from benchmark.ag3s.config import AG3SConfig, AG3SConfigError
from benchmark.ag3s.types import (
    AttentionAdapter,
    AttentionPointCloud,
    CollisionCandidate,
    CollisionConstraintSet,
    ConstraintSpec,
    GroundingStatus,
    Phase,
    PipelineStatus,
    PointCloud,
    Primitive,
    PrimitiveType,
    RobotCollisionModel,
    SourceType,
    SupportSurface,
    TargetGeometry,
)

__all__ = [
    "AG3SConfig",
    "AG3SConfigError",
    "AttentionAdapter",
    "AttentionPointCloud",
    "CollisionCandidate",
    "CollisionConstraintSet",
    "ConstraintSpec",
    "GroundingStatus",
    "Phase",
    "PipelineStatus",
    "PointCloud",
    "Primitive",
    "PrimitiveType",
    "RobotCollisionModel",
    "SourceType",
    "SupportSurface",
    "TargetGeometry",
]
