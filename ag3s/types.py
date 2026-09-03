"""AG3S data contracts.

The whole point of AG3S is a separation the type signatures are supposed to *enforce*, not merely
document:

    attention decides WHAT THE TARGET IS.
    3D geometry decides WHAT CAN COLLIDE.

So `TargetGeometry` carries attention scores and `CollisionCandidate` does not. Geometry with zero
attention is still a candidate if it physically exists; an attention score is never a detection
threshold. Stages downstream of grounding never receive the attention array at all — see
`collision_candidates.py`, whose entry point takes point indices and not an attention map.

Everything numeric is float64 in the **robot base frame** unless a field says otherwise. This module
imports numpy and nothing else: `config`, `casadi`, `scipy` and matplotlib all stay out so that
`from benchmark.ag3s.types import ...` can never fail for environment reasons.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

# --------------------------------------------------------------------------------------- enums


class Phase(str, enum.Enum):
    """Manipulation phase — an **injected external input**, never inferred by AG3S.

    AG3S deliberately contains no transition logic. Whoever runs the policy knows whether it is
    transiting, approaching, pre-grasping or grasping; encoding a guess here would put a second,
    disagreeing state machine in the safety path.
    """

    TRANSIT = "transit"
    APPROACH = "approach"
    PRE_GRASP = "pre_grasp"
    GRASP = "grasp"

    @classmethod
    def parse(cls, value: "Phase | str") -> "Phase":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"unknown phase {value!r}; expected one of {[p.value for p in cls]}") from exc


class SourceType(str, enum.Enum):
    """Where a collision candidate came from.

    "Collision candidate", not "obstacle": the term is the design. Anything that could touch the
    robot along the current or future trajectory belongs here, including the target itself, which is
    governed by phase rules rather than by deletion.
    """

    OBJECT = "object"
    SUPPORT_SURFACE = "support_surface"
    RESTRICTED_REGION = "restricted_region"
    UNKNOWN_GEOMETRY = "unknown_geometry"
    TARGET = "target"
    OVERFLOW = "overflow"

    @classmethod
    def parse(cls, value: "SourceType | str") -> "SourceType":
        """Unknown names raise. A source type that cannot be named cannot be given a margin."""
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class Manipulator(str, enum.Enum):
    """Which arm/gripper an external caller has authorized to make contact.

    **Injected, never inferred.** `Phase.GRASP` says a grasp is happening; it does not say which hand
    is doing it, and guessing "the one nearest the target" would hand contact clearance to whichever
    arm drifted closest. A bimanual grasp is expressed as both members of the set, so the contract
    does not have to change when the second arm joins.
    """

    LEFT = "left"
    RIGHT = "right"

    @classmethod
    def parse(cls, value: "Manipulator | str") -> "Manipulator":
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class PrimitiveType(str, enum.Enum):
    """TO-friendly shapes, in the priority order §3.5 asks for."""

    SPHERE = "sphere"
    CAPSULE = "capsule"
    BOX = "box"
    ELLIPSOID = "ellipsoid"


class GroundingStatus(str, enum.Enum):
    """Why grounding produced (or failed to produce) a target.

    A failure is reported, never papered over: if no seed survives or the best cluster scores below
    threshold, the target is ``None`` and candidate generation must fall back to treating all
    geometry conservatively. Fabricating a target would silently disable the target's own collision
    constraint.
    """

    OK = "ok"
    NO_GEOMETRY = "no_geometry"  # nothing survived reconstruction / self-filtering
    NO_ATTENTION = "no_attention"  # attention map absent or degenerate (all-equal)
    NO_SEED = "no_seed"  # threshold/percentile left no seed point
    NO_CLUSTER = "no_cluster"  # seeds present but no cluster met min_points
    LOW_SCORE = "low_score"  # best cluster scored below target_score_threshold

    @property
    def ok(self) -> bool:
        return self is GroundingStatus.OK


class PipelineStatus(str, enum.Enum):
    """Overall outcome of one `AG3S.process` call."""

    OK = "ok"
    NO_TARGET = "no_target"  # ran fully, but grounding declined to name a target (conservative mode)
    NO_GEOMETRY = "no_geometry"  # nothing to constrain; TO should treat this as sensor failure
    DEGRADED = "degraded"  # produced constraints, but some stage hit a configured cap
    GEOMETRY_INCOMPLETE = "geometry_incomplete"  # some physical geometry could not be represented


class ConstraintValidity(str, enum.Enum):
    """Whether this frame's constraint set can be certified geometry-complete.

    This is not a diagnostic string. It rides on `CollisionConstraintSet` so a TO cannot consume a
    frame whose geometry AG3S could not account for without first having seen the fact. The
    distinction that matters is between `INCOMPLETE` and an ordinary empty set: an empty `VALID` set
    means "AG3S looked and there is nothing there", while `INCOMPLETE` means "AG3S cannot tell you
    what is there". Rendering the second as the first is how a robot drives into an unmodelled wall.

    AG3S does not decide what to do about it. Stop, hold, re-use the last certified scene, reject the
    frame — all of that is the caller's policy, and putting it here would be a second safety state
    machine disagreeing with the first.
    """

    VALID = "valid"  # every valid physical geometry element is represented
    DEGRADED = "degraded"  # represented, but conservatively aggregated or partially observed
    INCOMPLETE = "incomplete"  # geometry existed that could not be represented at all

    @property
    def certified(self) -> bool:
        """True only for `VALID`. `DEGRADED` is usable; `INCOMPLETE` is not certified."""
        return self is ConstraintValidity.VALID

    @classmethod
    def worst(cls, *values: "ConstraintValidity") -> "ConstraintValidity":
        """The most pessimistic of the given validities — the direction that fails closed."""
        order = {cls.VALID: 0, cls.DEGRADED: 1, cls.INCOMPLETE: 2}
        return max(values, key=lambda v: order[v]) if values else cls.VALID


class CameraID(str, enum.Enum):
    """Which physical camera an observation came from.

    Roles (head = global backbone, wrists = local refinement and occlusion recovery) are a statement
    about *coverage*, never a licence to drop geometry: every camera's valid points are equally
    eligible to become collision candidates. The enum exists so provenance survives fusion, not so
    the pipeline can treat one view as more real than another.
    """

    HEAD = "head"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"

    @classmethod
    def parse(cls, value: "CameraID | str") -> "CameraID":
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


@dataclasses.dataclass(frozen=True)
class ContactPolicyContext:
    """Everything outside AG3S that decides which contact is currently permitted.

    Both fields are injected. `phase` says how far along the manipulation is; `active_manipulators`
    says which arm is authorized to touch the target. Neither is inferred, and the empty set is a
    perfectly valid value meaning "nobody may touch anything" — which is also what an invalid or
    missing manipulator collapses to, so a caller that forgets to fill this in gets full clearance
    everywhere rather than silent permission.
    """

    phase: Phase = Phase.TRANSIT
    active_manipulators: frozenset[Manipulator] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", Phase.parse(self.phase))
        parsed: set[Manipulator] = set()
        for m in self.active_manipulators or ():
            try:
                parsed.add(Manipulator.parse(m))
            except ValueError:
                # Fail closed: an unparseable manipulator grants nothing rather than everything.
                continue
        object.__setattr__(self, "active_manipulators", frozenset(parsed))

    @classmethod
    def make(
        cls,
        phase: "Phase | str" = Phase.TRANSIT,
        active_manipulators: "Any" = None,
    ) -> "ContactPolicyContext":
        """Build from loose input — a phase name and any iterable (or single value) of manipulators."""
        if active_manipulators is None:
            manipulators: frozenset = frozenset()
        elif isinstance(active_manipulators, (str, Manipulator)):
            manipulators = frozenset({active_manipulators})
        else:
            manipulators = frozenset(active_manipulators)
        return cls(Phase.parse(phase), manipulators)

    def authorizes(self, manipulator: "Manipulator | str") -> bool:
        try:
            return Manipulator.parse(manipulator) in self.active_manipulators
        except ValueError:
            return False


# ----------------------------------------------------------------------------------- geometry


@dataclasses.dataclass(frozen=True)
class PointCloud:
    """Points in `frame_id`, with the originating pixel of each point where it is known.

    ``uv`` is what makes 2D->3D attention lifting possible at all, so it is carried through
    reconstruction, voxel downsampling and self-filtering rather than being recomputed. It is
    ``None`` only when the caller supplied a raw point cloud with no image behind it.

    Not validated on every construction beyond shapes: this object is created once per stage on the
    hot path, and `select` is used far more often than the constructor.
    """

    points: np.ndarray  # (N, 3) float64
    uv: Optional[np.ndarray] = None  # (N, 2) int32, column-then-row (u, v)
    frame_id: str = "base"

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {pts.shape}")
        object.__setattr__(self, "points", pts)
        if self.uv is not None:
            uv = np.asarray(self.uv, np.int32)
            if uv.shape != (pts.shape[0], 2):
                raise ValueError(f"uv must be ({pts.shape[0]}, 2), got {uv.shape}")
            object.__setattr__(self, "uv", uv)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.points.shape[0] == 0

    def select(self, index: np.ndarray) -> "PointCloud":
        """Index-select points and their uv together, so the two can never drift apart."""
        index = np.asarray(index)
        return PointCloud(
            points=self.points[index],
            uv=None if self.uv is None else self.uv[index],
            frame_id=self.frame_id,
        )

    @staticmethod
    def empty(frame_id: str = "base") -> "PointCloud":
        return PointCloud(np.zeros((0, 3), np.float64), None, frame_id)


@dataclasses.dataclass(frozen=True)
class AttentionPointCloud:
    """A `PointCloud` plus a per-point attention value, `P_A = {(X, Y, Z, A)}`.

    Every point of the input cloud is present. Low-attention points are **never** dropped here —
    they are exactly the geometry that must still become a collision candidate. Attention selects
    seeds in `target_grounding`; it does not select geometry.

    Two attention arrays, on purpose. `attention` is normalized and is what thresholds and scores
    compare against; `raw_attention` is the adapter's output before normalization, and exists because
    normalization and peak-finding want incompatible things from the same signal:

    * a threshold wants **comparable values**, so percentile normalization clips the top 1% to
      guard against a single hot patch crushing the rest of the map;
    * `distance_from_attention_peak` wants an **unambiguous ranking**, and that same clip creates
      hundreds of exact ties. Measured on the default fixture, clipping at p99 ties 418 points at
      1.0, 154 of them on the table, so `argmax` lands on a table point metres from the object.

    Moving the percentile cannot satisfy both — at p100 the ties vanish but a lone 1000x outlier
    squashes the real range to 0.001. So the peak is read off `raw_attention` and the thresholds off
    `attention`, and neither has to compromise.
    """

    cloud: PointCloud
    attention: np.ndarray  # (N,) float32, normalization per config
    raw_attention: Optional[np.ndarray] = None  # (N,) float32, pre-normalization

    def __post_init__(self) -> None:
        att = np.asarray(self.attention, np.float32).reshape(-1)
        if att.shape[0] != len(self.cloud):
            raise ValueError(f"attention has {att.shape[0]} values for {len(self.cloud)} points")
        object.__setattr__(self, "attention", att)
        if self.raw_attention is not None:
            raw = np.asarray(self.raw_attention, np.float32).reshape(-1)
            if raw.shape[0] != att.shape[0]:
                raise ValueError(f"raw_attention has {raw.shape[0]} values for {att.shape[0]} points")
            object.__setattr__(self, "raw_attention", raw)

    def __len__(self) -> int:
        return len(self.cloud)

    @property
    def points(self) -> np.ndarray:
        return self.cloud.points

    @property
    def peak_signal(self) -> np.ndarray:
        """The array to take the attention peak from — raw when available, normalized otherwise."""
        return self.attention if self.raw_attention is None else self.raw_attention

    def select(self, index: np.ndarray) -> "AttentionPointCloud":
        index = np.asarray(index)
        return AttentionPointCloud(
            self.cloud.select(index),
            self.attention[index],
            None if self.raw_attention is None else self.raw_attention[index],
        )


@dataclasses.dataclass(frozen=True)
class CameraObservation:
    """One camera's capture, with everything needed to place it in the base frame *at capture time*.

    The timestamp fields are the point of this type. A wrist camera moves with the arm, so a frame
    captured 80 ms ago belongs at the pose the arm held 80 ms ago — transforming all three cameras
    with one `q_now` smears the wrist clouds into ghost geometry and, worse, misaligns the
    self-filter, leaving robot points in the scene to cluster into a phantom obstacle welded to the
    gripper. So each observation carries its own `robot_state`.

    The camera pose arrives one of two ways:

    * `T_base_cam` directly, for a caller that has already done the transform (a simulator, or a
      head camera whose mount is effectively fixed);
    * `mount_link` plus `T_link_cam` — the hand-eye calibration — which AG3S composes with forward
      kinematics at `robot_state`. This is the route for the wrists, and the direction of
      `T_link_cam` is named in the field so it cannot be applied backwards silently.

    `attention_map` is optional per camera. A camera without one contributes zero to target scoring
    and its **full geometry** to the collision scene, which is the correct asymmetry: attention names
    the target, geometry decides what can collide.
    """

    camera_id: CameraID
    depth: Optional[np.ndarray] = None
    pointcloud: Optional[np.ndarray] = None
    camera_intrinsics: Optional[np.ndarray] = None
    T_base_cam: Optional[np.ndarray] = None  # (4, 4) camera -> base, if already known
    mount_link: Optional[str] = None  # otherwise: the link this camera is bolted to
    T_link_cam: Optional[np.ndarray] = None  # (4, 4) mount link frame -> camera frame
    robot_state: Optional[np.ndarray] = None  # q at *this* camera's capture instant
    timestamp: float = 0.0  # when the image was captured
    state_timestamp: Optional[float] = None  # when `robot_state` was sampled; defaults to timestamp
    attention_map: Any = None
    image_hw: Optional[tuple[int, int]] = None
    uv: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_id", CameraID.parse(self.camera_id))
        for name in ("T_base_cam", "T_link_cam"):
            value = getattr(self, name)
            if value is not None:
                T = np.asarray(value, np.float64)
                if T.shape != (4, 4):
                    raise ValueError(f"{name} must be (4, 4), got {T.shape}")
                object.__setattr__(self, name, T)
        if self.robot_state is not None:
            object.__setattr__(
                self, "robot_state", np.asarray(self.robot_state, np.float64).reshape(-1)
            )
        if (self.depth is None) == (self.pointcloud is None):
            raise ValueError(
                f"{self.camera_id.value}: provide exactly one of depth= or pointcloud="
            )
        if self.T_base_cam is None and (self.mount_link is None or self.T_link_cam is None):
            raise ValueError(
                f"{self.camera_id.value}: provide T_base_cam, or mount_link together with T_link_cam"
            )

    @property
    def state_time(self) -> float:
        return self.timestamp if self.state_timestamp is None else float(self.state_timestamp)

    @property
    def has_attention(self) -> bool:
        return self.attention_map is not None

    def resolve_T_base_cam(self, robot_model: Optional["RobotCollisionModel"] = None) -> np.ndarray:
        """``T_base_cam`` at this observation's own capture time.

        Composition order is ``FK(q_capture, mount_link) @ T_link_cam``: the link's pose in the base
        frame, then the fixed offset from the link to the optical frame. Writing it the other way
        round produces a camera that orbits the base instead of riding the wrist, which looks wrong
        immediately in a plot and is why the direction is checked by
        `tests/ag3s/test_multicamera.py` rather than left to convention.
        """
        if self.T_base_cam is not None:
            return self.T_base_cam
        if robot_model is None:
            raise ValueError(
                f"{self.camera_id.value}: mount_link={self.mount_link!r} needs a robot model to "
                "compute forward kinematics; none was injected"
            )
        pose_fn = getattr(robot_model, "link_pose", None) or getattr(
            robot_model, "link_pose_numeric", None
        )
        if pose_fn is None:
            raise ValueError(
                f"the injected robot model cannot compute link poses, so camera "
                f"{self.camera_id.value} cannot be placed from mount_link={self.mount_link!r}"
            )
        if self.robot_state is None:
            raise ValueError(
                f"{self.camera_id.value}: mount_link placement needs the robot_state captured with "
                "the image; reusing a single q_now for every camera is exactly what this avoids"
            )
        return np.asarray(pose_fn(self.robot_state, self.mount_link), np.float64) @ self.T_link_cam


@dataclasses.dataclass(frozen=True)
class FusedPointCloud:
    """One geometric scene, assembled from several cameras, with provenance kept alongside.

    Two things are deliberately separate here, because collapsing them is how multi-view fusion
    usually loses information:

    **Geometry** is `points` — one representative per occupied base-frame voxel, and always an
    *actually observed* point rather than a synthetic centroid. A centroid has no pixel, so it has no
    attention value and no camera to attribute it to; averaging three observations would destroy the
    correspondence at the exact moment several cameras finally agree about something.

    **Provenance** is the CSR block: `obs_offset[i]:obs_offset[i+1]` indexes every observation that
    landed in point `i`'s voxel, each carrying its camera, its pixel, its capture time and its
    attention. A Python object per point would be unaffordable at 60k points a frame; three flat
    arrays and an offset vector are not.

    Downstream stages take `as_pointcloud()` and never learn that fusion happened, which is what
    keeps support-surface fitting, grounding, candidate generation and constraint building unchanged.
    """

    points: np.ndarray  # (N, 3) base frame
    obs_offset: np.ndarray  # (N + 1,) int64, CSR row pointers
    obs_camera: np.ndarray  # (K,) int16, index into `cameras`
    obs_uv: np.ndarray  # (K, 2) int32
    obs_time: np.ndarray  # (K,) float64
    obs_attention: np.ndarray  # (K,) float32, normalized per camera before fusion
    representative: np.ndarray  # (N,) int64, which observation each point's coordinates came from
    cameras: tuple[CameraID, ...] = ()
    frame_id: str = "base"

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", np.asarray(self.points, np.float64).reshape(-1, 3))
        object.__setattr__(self, "obs_offset", np.asarray(self.obs_offset, np.int64).reshape(-1))
        object.__setattr__(self, "representative", np.asarray(self.representative, np.int64).reshape(-1))
        n, k = self.points.shape[0], self.obs_camera.shape[0]
        if self.obs_offset.shape[0] != n + 1:
            raise ValueError(f"obs_offset must have {n + 1} entries, got {self.obs_offset.shape[0]}")
        if int(self.obs_offset[-1]) != k:
            raise ValueError(f"obs_offset ends at {self.obs_offset[-1]} but there are {k} observations")

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.points.shape[0] == 0

    @property
    def n_observations(self) -> int:
        return int(self.obs_camera.shape[0])

    def as_pointcloud(self) -> "PointCloud":
        """The plain cloud every downstream stage consumes, carrying the representative's pixel."""
        uv = None if self.is_empty else self.obs_uv[self.representative]
        return PointCloud(self.points, uv, self.frame_id)

    def fused_attention(self) -> np.ndarray:
        """`max` over each point's observations — the required default aggregation.

        Max rather than mean because attention is evidence *for* the target, and a wrist camera that
        sees the cup clearly should not have its testimony averaged away by two cameras that barely
        see it. Cameras with no attention map contribute zero, which under max is exactly "no
        opinion" rather than "votes against".
        """
        out = np.zeros(len(self), np.float32)
        for i in range(len(self)):
            lo, hi = int(self.obs_offset[i]), int(self.obs_offset[i + 1])
            if hi > lo:
                out[i] = self.obs_attention[lo:hi].max()
        return out

    def observation_slice(self, index: int) -> slice:
        return slice(int(self.obs_offset[index]), int(self.obs_offset[index + 1]))

    def cameras_of(self, index: int) -> set["CameraID"]:
        """Which cameras saw point `index`. The provenance query traceability actually needs."""
        block = self.obs_camera[self.observation_slice(index)]
        return {self.cameras[int(c)] for c in block}

    def select(self, index: np.ndarray) -> "FusedPointCloud":
        """Index-select points, carrying their observation blocks with them."""
        index = np.asarray(index).reshape(-1)
        if index.dtype == bool:
            index = np.nonzero(index)[0]
        blocks = [
            np.arange(self.obs_offset[i], self.obs_offset[i + 1], dtype=np.int64) for i in index
        ]
        counts = np.asarray([b.size for b in blocks], np.int64)
        flat = np.concatenate(blocks) if blocks else np.zeros(0, np.int64)
        offset = np.zeros(index.size + 1, np.int64)
        np.cumsum(counts, out=offset[1:])
        # `representative` is an index into the observation table, so it has to be renumbered into
        # the compacted table rather than carried across.
        remap = np.full(self.n_observations, -1, np.int64)
        remap[flat] = np.arange(flat.size, dtype=np.int64)
        return FusedPointCloud(
            points=self.points[index],
            obs_offset=offset,
            obs_camera=self.obs_camera[flat],
            obs_uv=self.obs_uv[flat],
            obs_time=self.obs_time[flat],
            obs_attention=self.obs_attention[flat],
            representative=remap[self.representative[index]],
            cameras=self.cameras,
            frame_id=self.frame_id,
        )

    @staticmethod
    def empty(frame_id: str = "base") -> "FusedPointCloud":
        return FusedPointCloud(
            points=np.zeros((0, 3), np.float64),
            obs_offset=np.zeros(1, np.int64),
            obs_camera=np.zeros(0, np.int16),
            obs_uv=np.zeros((0, 2), np.int32),
            obs_time=np.zeros(0, np.float64),
            obs_attention=np.zeros(0, np.float32),
            representative=np.zeros(0, np.int64),
            cameras=(),
            frame_id=frame_id,
        )


@dataclasses.dataclass(frozen=True)
class Primitive:
    """One convex shape a TO can write a closed-form distance against.

    `center` and `orientation` are in `frame_id`; `dimensions` is shape-specific and always the
    *half* extent, so that a sphere's `dimensions[0]` is its radius rather than its diameter:

        SPHERE     (r, r, r)              orientation ignored
        CAPSULE    (r, half_length, r)    segment along the local +z axis of `orientation`
        BOX        (hx, hy, hz)           half extents along the local axes
        ELLIPSOID  (a, b, c)              semi-axes along the local axes

    `safety_margin` is *not* baked into `dimensions`. Keeping it separate is what lets the phase
    rules relax clearance on the target without refitting geometry, and lets TO expose the margin as
    a tunable parameter.
    """

    type: PrimitiveType
    center: np.ndarray  # (3,)
    dimensions: np.ndarray  # (3,) half-extents, see above
    orientation: np.ndarray = dataclasses.field(  # (3, 3) rotation, local -> frame_id
        default_factory=lambda: np.eye(3)
    )
    semantic_role: str = "unknown"
    candidate_id: int = -1
    safety_margin: float = 0.0
    collision_enabled: bool = True
    contact_permission: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", PrimitiveType(self.type))
        object.__setattr__(self, "center", np.asarray(self.center, np.float64).reshape(3))
        object.__setattr__(self, "dimensions", np.asarray(self.dimensions, np.float64).reshape(3))
        object.__setattr__(self, "orientation", np.asarray(self.orientation, np.float64).reshape(3, 3))

    @property
    def bounding_radius(self) -> float:
        """Radius of the smallest sphere at `center` containing the primitive.

        This is what the sphere-vs-sphere constraint in `constraint_builder` consumes. For a sphere
        it is exact; for the others it is the conservative direction, which is the only safe one.
        """
        if self.type is PrimitiveType.SPHERE:
            return float(self.dimensions[0])
        if self.type is PrimitiveType.CAPSULE:
            return float(self.dimensions[1] + self.dimensions[0])
        return float(np.linalg.norm(self.dimensions))


@dataclasses.dataclass(frozen=True)
class SupportSurface:
    """A fitted plane (table, floor, shelf), kept out of the voxel/cluster path on purpose.

    A table is not a blob. Approximating it with primitives either misses most of it or produces an
    enormous bounding shape; as a half-space it is *exact* and costs the TO a single linear row.
    The free side is ``n . p >= d + margin`` with ``n`` a unit normal.
    """

    id: int
    normal: np.ndarray  # (3,) unit, pointing into free space
    offset: float  # d, so the plane is n . p = d
    point_indices: np.ndarray  # inliers, indices into the cloud it was fitted on
    point_count: int
    rms_error: float
    safety_margin: float = 0.0
    frame_id: str = "base"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, np.float64).reshape(3)
        nrm = float(np.linalg.norm(n))
        object.__setattr__(self, "normal", n / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0]))
        object.__setattr__(self, "point_indices", np.asarray(self.point_indices, np.int64).reshape(-1))

    def to_halfspace(self) -> tuple[np.ndarray, float]:
        """``(n, d + margin)`` for the constraint ``n . p >= d + margin``."""
        return self.normal.copy(), float(self.offset + self.safety_margin)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """``n . p - (d + margin)``; negative means inside the keep-out side."""
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        return pts @ self.normal - (self.offset + self.safety_margin)


@dataclasses.dataclass(frozen=True)
class TargetGeometry:
    """The object attention says the policy is acting on.

    Carrying `attention_score` here and nowhere else is the type-level statement of the core
    principle. `confidence` is the grounding score, which mixes attention and spatial consistency;
    `attention_score` is the attention part alone, so an ablation can report them separately.
    """

    id: int
    points: np.ndarray  # (M, 3) the cluster's own points
    point_indices: np.ndarray  # (M,) indices into the cloud grounding ran on
    centroid: np.ndarray  # (3,)
    bounding_geometry: Primitive
    attention_score: float
    confidence: float
    timestamp: float = 0.0
    frame_id: str = "base"
    metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    # --- multi-view provenance (§14), kept deliberately small -----------------------------
    #: Which camera carried the attention peak that seeded this target, and which cameras contributed
    #: any of its points. Enough to answer "why did AG3S think *that* was the cup" after the fact
    #: without storing a per-point camera list a second time — the fused cloud already has that, and
    #: duplicating it here would double the memory of the one object most likely to be logged.
    seed_camera: Optional["CameraID"] = None
    supporting_cameras: tuple["CameraID", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", np.asarray(self.points, np.float64).reshape(-1, 3))
        object.__setattr__(self, "point_indices", np.asarray(self.point_indices, np.int64).reshape(-1))
        object.__setattr__(self, "centroid", np.asarray(self.centroid, np.float64).reshape(3))


@dataclasses.dataclass
class CollisionCandidate:
    """Any 3D geometry TO must consider as a collision-constraint candidate.

    Mutable, unlike the rest of this module: `CandidateTracker` rewrites `id`, `track_age` and the
    phase-derived fields in place as frames arrive, and copying the whole record per frame for the
    sake of immutability would be pure waste on the hot path.

    `id` is **stable across frames** via centroid nearest-neighbour association. TO warm-starting
    depends on that: if ids were reassigned every frame the solver's previous solution would be
    matched against the wrong constraint slot.
    """

    id: int
    source_type: SourceType
    semantic_role: str
    geometry: list[Primitive]
    pose: np.ndarray  # (4, 4) candidate frame -> frame_id
    dimensions: np.ndarray  # (3,) overall half-extents, for logging/visualization
    confidence: float
    safety_margin: float
    collision_enabled: bool
    contact_permission: bool
    phase_rule: str
    timestamp: float
    frame_id: str = "base"
    # --- tracking / provenance (not in the §3.4 field list, but required by it) ---
    track_age: int = 0
    centroid: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    point_indices: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0, np.int64))
    point_count: int = 0

    def __post_init__(self) -> None:
        self.source_type = SourceType(self.source_type)
        self.pose = np.asarray(self.pose, np.float64).reshape(4, 4)
        self.dimensions = np.asarray(self.dimensions, np.float64).reshape(3)
        self.centroid = np.asarray(self.centroid, np.float64).reshape(3)
        self.point_indices = np.asarray(self.point_indices, np.int64).reshape(-1)


@dataclasses.dataclass(frozen=True)
class AttachedCollisionGeometry:
    """An object the robot is holding, riding on a parent link.

    Once grasped, the target stops being a fixed obstacle and becomes part of the moving robot: the
    thing that must not hit the table is now the *crate*, not just the gripper. So it is modelled as
    primitives rigidly attached to `parent_link`, and its pose follows symbolic FK:

        T_base_object(q) = T_base_parent(q) @ T_parent_object

    `T_parent_object` is fixed at attach time and named for its direction — parent link frame to
    object frame — because getting it backwards produces an object that swings the wrong way and
    still looks plausible in a plot. `tests/ag3s/test_attached.py` pins the direction numerically.

    Two properties are deliberate:

    **Existence does not depend on perception.** The geometry is a *snapshot* taken when an external
    caller confirmed the grasp. After that, occlusion, grounding failure or a dropped camera cannot
    remove it — the object is still in the gripper whatever the cameras can see. Only an explicit
    detach event removes it.

    **Contact is an allowlist, not a global disable.** The held object overlaps the fingers that hold
    it, so those pairs are excluded by name. Everything else — the forearm, the torso, the opposite
    arm — stays constrained, because an object swinging into the robot's own elbow is a real
    collision. An unknown link name is not in the allowlist, so it stays constrained: fail closed.
    """

    parent_link: str
    T_parent_object: np.ndarray  # (4, 4) parent link frame -> object frame
    primitives: list[Primitive]
    allowed_contact_links: frozenset[str] = frozenset()
    source_candidate_id: int = -1
    attached_at: float = 0.0
    label: str = "attached_object"

    def __post_init__(self) -> None:
        T = np.asarray(self.T_parent_object, np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T_parent_object must be (4, 4), got {T.shape}")
        object.__setattr__(self, "T_parent_object", T)
        object.__setattr__(self, "primitives", list(self.primitives))
        object.__setattr__(self, "allowed_contact_links", frozenset(self.allowed_contact_links))

    def permits_contact_with(self, link: str) -> bool:
        return str(link) in self.allowed_contact_links


# --------------------------------------------------------------------------------- constraints


@dataclasses.dataclass(frozen=True)
class ConstraintSpec:
    """A CasADi-compatible constraint specification with a **fixed structure**.

    The structure (number of slots, horizon, robot sphere count) is built once. Each frame updates
    only `parameter_values`. This is the whole reason `max_candidates` exists: if the NLP were
    rebuilt every frame, IPOPT would re-do symbolic setup and warm-starting would be impossible, and
    "real-time" would be off the table.

    Attributes:
        expr_factory: ``(q_syms, p_sym) -> casadi.SX`` returning the stacked constraint vector
            ``h(q) >= 0``. Held as a callable rather than a materialized expression so a caller can
            build it against its own decision variables.
        parameter_symbol: the CasADi ``SX`` parameter vector the factory closes over.
        parameter_values: the current numeric value of that vector, shape ``(n_params,)``.
        active_mask: ``(max_candidates,)`` in {0, 1}. Slots at 0 are disabled by making their row
            trivially satisfied, so the row count never changes.
        lower_bound / upper_bound: ``lbg`` / ``ubg`` for the stacked vector; ``h >= 0`` means
            ``lbg = 0``, ``ubg = +inf``.
        candidate_ids: which candidate occupies each slot, ``-1`` for empty. Lets TO carry a warm
            start across frames slot-by-slot.
    """

    expr_factory: Callable[..., Any]
    parameter_symbol: Any
    parameter_values: np.ndarray
    active_mask: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    candidate_ids: np.ndarray
    horizon: int
    max_candidates: int
    n_robot_spheres: int
    squared_distance: bool
    layout: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def n_constraints(self) -> int:
        return int(self.lower_bound.shape[0])


@dataclasses.dataclass(frozen=True)
class CollisionConstraintSet:
    """The final AG3S output — everything a TO needs and nothing it has to re-derive.

    `frame_id` is the *coordinate frame* the geometry lives in (all of it, uniformly). The per-frame
    sequence counter is `frame_index`, kept separate so the two are never confused.
    """

    constraints: Optional[ConstraintSpec]
    candidates: list[CollisionCandidate]
    target: Optional[TargetGeometry]
    support_surfaces: list[SupportSurface]
    robot_state: np.ndarray
    phase: Phase
    timestamp: float
    frame_id: str
    status: PipelineStatus
    grounding_status: GroundingStatus = GroundingStatus.OK
    frame_index: int = 0
    profile: dict[str, float] = dataclasses.field(default_factory=dict)
    notes: list[str] = dataclasses.field(default_factory=list)
    # --- fail-closed contract -----------------------------------------------------------
    validity: ConstraintValidity = ConstraintValidity.VALID
    contact_context: ContactPolicyContext = dataclasses.field(default_factory=ContactPolicyContext)
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    attached: Optional["AttachedCollisionGeometry"] = None
    #: Optional ESDF collision field, present when `collision_backend` is `esdf` or `both`.
    #: Typed as `Any` on purpose: `benchmark.ag3s.esdf` imports scipy, and `types` must stay
    #: importable on a machine that has only numpy — the same rule `__init__` follows.
    #:
    #: It sits **beside** `candidates`, never instead of them. The target/obstacle separation is a
    #: property of the candidate list, and an ESDF cannot express "this geometry is the target" —
    #: it only answers "how far is the nearest surface". Dropping the candidates to save memory
    #: would delete the one thing the clearance policy reads.
    esdf: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "robot_state", np.asarray(self.robot_state, np.float64).reshape(-1))
        object.__setattr__(self, "phase", Phase.parse(self.phase))

    @property
    def has_target(self) -> bool:
        return self.target is not None

    @property
    def geometry_certified(self) -> bool:
        """Whether a TO may treat this frame's geometry as complete.

        The one question a caller must ask before planning through empty space. `False` does not say
        what to do about it — that is the caller's policy.
        """
        return self.validity.certified

    def enabled_candidates(self) -> list[CollisionCandidate]:
        return [c for c in self.candidates if c.collision_enabled]


# ----------------------------------------------------------------------------------- protocols


@runtime_checkable
class RobotCollisionModel(Protocol):
    """The robot side of the collision pair. **AG3S does not own this.**

    AG3S knows the scene; it does not know the robot's kinematics or its collision geometry. That
    belongs to whoever integrates it — a Pinocchio model, a MuJoCo model, or the URDF sphere chain
    in `robot_models/`. Injecting it behind a Protocol is what keeps AG3S from growing a
    robot-specific dependency, and what lets the TO-contract tests run against a two-link analytic
    mock with no solver-visible difference.

    `sphere_centers_symbolic` must return CasADi expressions in `q`; `sphere_centers_numeric` the
    same chain evaluated numerically, which is what the self-filter uses. Implementations are
    expected to agree — `tests/ag3s/test_robot_models.py` checks that they do.
    """

    @property
    def nq(self) -> int:
        """Number of configuration variables the model consumes."""
        ...

    @property
    def sphere_link_names(self) -> Sequence[str]:
        """One link name per collision sphere, in **exactly the sphere order** both FK paths use.

        This is what makes a link-wise clearance policy possible at all: without it a constraint row
        knows the robot sphere's index but not which part of the robot it is, and "the right
        fingertip may touch the target" is not expressible. The alignment is load-bearing — an
        off-by-one here would grant fingertip clearance to the forearm — so
        `tests/ag3s/test_robot_models.py` checks length and ordering against both FK
        implementations rather than trusting the convention.
        """
        ...

    def sphere_centers_symbolic(self, q: Any) -> Sequence[tuple[Any, float]]:
        """``q`` (casadi SX/MX, shape (nq, 1)) -> ``[(center_SX_3x1, radius), ...]``."""
        ...

    def sphere_centers_numeric(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``q`` (nq,) -> ``(centers (S, 3), radii (S,))``."""
        ...


@runtime_checkable
class AttentionAdapter(Protocol):
    """Turns whatever a VLA calls "attention" into a dense pixel-space map.

    π0.5 hands back a 16x16 patch grid per (layer, head); another model might hand back a dense map,
    a token-to-patch matrix, or nothing at all. AG3S is not allowed to care. Adapters live at the
    boundary and everything downstream sees only ``(H, W) float32``.
    """

    def to_pixel_map(self, raw: Any, out_hw: tuple[int, int]) -> np.ndarray:
        """Return an ``(H, W)`` float32 attention map for the requested image size."""
        ...


__all__ = [
    "AttachedCollisionGeometry",
    "AttentionAdapter",
    "AttentionPointCloud",
    "CameraID",
    "CameraObservation",
    "CollisionCandidate",
    "CollisionConstraintSet",
    "ConstraintSpec",
    "ConstraintValidity",
    "ContactPolicyContext",
    "FusedPointCloud",
    "GroundingStatus",
    "Manipulator",
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
