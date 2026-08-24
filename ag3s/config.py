"""AG3S configuration.

Every ablation the study needs must be reachable **by config alone** — no code edits, no forked
branches. Concretely: attention normalization, attention layer/head, seed threshold, 2D-attention-
only vs 3D grounding, clustering method, self-filter on/off, primitive type, safety margin, and
phase-aware target constraints on/off. Each of those maps to exactly one field below, and
`test_pipeline.py` asserts each is reachable.

The YAML schema is the one in the AG3S spec §9, extended only where a stage genuinely needs a knob
the spec did not spell out (camera depth range, RANSAC budget, tracking radius). Extended fields all
carry defaults that reproduce the spec's behaviour, so a spec-exact YAML loads unchanged.

Unknown keys are a hard error, not a warning. A silently ignored `seed_percentile: 95` under the
wrong parent is an ablation that reports the default's numbers under the variant's name.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Mapping

from benchmark.ag3s.types import CameraID, Manipulator, Phase, PrimitiveType


class AG3SConfigError(ValueError):
    """Raised when a configuration is malformed or internally inconsistent."""


# ------------------------------------------------------------------------------------ sections


@dataclasses.dataclass(frozen=True)
class AttentionConfig:
    """How a VLA's attention becomes a per-point scalar.

    `layer`/`head` are here rather than in the adapter because they are ablation axes: the KNOWS
    probe found the paper's (12, 3) cell is not obviously the best one on this checkpoint
    (`benchmark/knows_vla/docs/08-p0b-results.md`), so which cell AG3S consumes has to be swappable
    from the config file.
    """

    normalization: str = "percentile"  # percentile | minmax | softmax | none
    seed_percentile: float = 95.0
    temperature: float = 1.0  # softmax normalization only
    percentile_range: tuple[float, float] = (5.0, 99.0)  # percentile normalization only
    seed_threshold: float | None = None  # absolute cut; overrides seed_percentile when set
    interpolation: str = "bilinear"  # bilinear | nearest, for attention-res != depth-res
    layer: int | None = None  # adapter hint; None = adapter's own default
    head: int | None = None

    NORMALIZATIONS = ("percentile", "minmax", "softmax", "none")
    INTERPOLATIONS = ("bilinear", "nearest")

    def validate(self) -> None:
        if self.normalization not in self.NORMALIZATIONS:
            raise AG3SConfigError(
                f"attention.normalization must be one of {self.NORMALIZATIONS}, got {self.normalization!r}"
            )
        if self.interpolation not in self.INTERPOLATIONS:
            raise AG3SConfigError(
                f"attention.interpolation must be one of {self.INTERPOLATIONS}, got {self.interpolation!r}"
            )
        if not 0.0 <= self.seed_percentile <= 100.0:
            raise AG3SConfigError(f"attention.seed_percentile must be in [0, 100], got {self.seed_percentile}")
        if self.temperature <= 0.0:
            raise AG3SConfigError(f"attention.temperature must be > 0, got {self.temperature}")
        lo, hi = self.percentile_range
        if not 0.0 <= lo < hi <= 100.0:
            raise AG3SConfigError(f"attention.percentile_range must satisfy 0 <= lo < hi <= 100, got {(lo, hi)}")


@dataclasses.dataclass(frozen=True)
class ClusteringConfig:
    """Target-grounding clustering.

    `use_3d_connectivity=False` is the "2D attention only" ablation: seeds are taken straight as the
    target with no geometric growth, which is what the KNOWS-style pipeline effectively does. It
    exists to be measured against, not to be used.
    """

    method: str = "dbscan"  # dbscan | region_growing
    eps: float = 0.03
    min_points: int = 20
    use_3d_connectivity: bool = True
    # None means "use `eps`", which is the default and the only setting that makes
    # `clustering.method` a clean ablation: same neighbourhood radius, different density rule. Set a
    # number only to deliberately vary the radius, and report it as a separate axis — a method
    # comparison run at two different radii measures the radius.
    region_growing_radius: float | None = None
    target_score_threshold: float = 0.25
    w_attention: float = 0.7
    w_geometry: float = 0.3
    max_seed_points: int = 4000  # cap on seeds fed to the connectivity search, for latency
    # Length scales for the two spatial-consistency terms, in metres. `compactness_scale` is set to
    # a graspable object's RMS radius (a 4 cm sphere's visible cap measures ~2 cm, a table fragment
    # tens of cm); `peak_scale` to how far a cluster centroid may sit from the attention peak before
    # it stops plausibly being the thing attended to.
    compactness_scale: float = 0.05
    peak_scale: float = 0.10

    METHODS = ("dbscan", "region_growing")

    def validate(self) -> None:
        if self.method not in self.METHODS:
            raise AG3SConfigError(f"clustering.method must be one of {self.METHODS}, got {self.method!r}")
        if self.eps <= 0.0:
            raise AG3SConfigError(f"clustering.eps must be > 0, got {self.eps}")
        if self.min_points < 1:
            raise AG3SConfigError(f"clustering.min_points must be >= 1, got {self.min_points}")
        if self.w_attention < 0.0 or self.w_geometry < 0.0:
            raise AG3SConfigError("clustering weights must be non-negative")
        if self.w_attention + self.w_geometry <= 0.0:
            raise AG3SConfigError("clustering.w_attention + w_geometry must be > 0")
        if self.compactness_scale <= 0.0 or self.peak_scale <= 0.0:
            raise AG3SConfigError("clustering.compactness_scale and peak_scale must be > 0")
        if self.region_growing_radius is not None and self.region_growing_radius <= 0.0:
            raise AG3SConfigError(
                f"clustering.region_growing_radius must be > 0 or null, got {self.region_growing_radius}"
            )


@dataclasses.dataclass(frozen=True)
class PointCloudConfig:
    """Reconstruction and preprocessing.

    `max_points` is not an optimization, it is a correctness requirement for real-time viability: a
    640x480 depth frame is 307k points and every KD-tree downstream is superlinear in that. The
    subsample must be **deterministic** so an ablation rerun produces the same clusters.
    """

    voxel_size: float = 0.005
    self_filter: bool = True
    max_points: int = 60000
    depth_min: float = 0.05  # metres; below this the sensor is not trustworthy
    depth_max: float = 3.0
    # Optional **radial** range gate, in metres. `depth_min`/`depth_max` bound the pinhole z-depth,
    # which is the convention a depth sensor reports and the one `backproject` implements. On a wide
    # field of view those are very different quantities: at fovy 90 degrees on a 4:3 frame the corner
    # ray sits ~58 degrees off the optical axis, so a z-depth cut of 2.2 m still admits points 4.2 m
    # away (measured on the RB-Y1 wrist camera). Set this to bound the workspace by true distance.
    range_max: float | None = None
    depth_scale: float = 1.0  # multiply raw depth by this to get metres (e.g. 0.001 for uint16 mm)
    self_filter_inflation: float = 0.02  # metres added to each robot sphere before rejecting points
    subsample_seed: int = 0  # only used by the deterministic stride/permutation
    # How `max_points` is enforced. "voxel" grows the voxel size until the cloud fits, so every
    # occupied voxel still contributes a representative and no region larger than the final voxel
    # goes unobserved; the cost is a coarser cloud, reported in the stats. "stride" is the old
    # `linspace` selection, kept as an ablation — it samples the whole index range but makes no
    # spatial guarantee at all, and geometry that is never sampled cannot become a candidate.
    cap_strategy: str = "voxel"  # voxel | stride
    # Multiplier per growth step. 2 ** (1/3) roughly halves the occupied-voxel count each time, so
    # the loop converges in a handful of iterations from any realistic starting point.
    cap_voxel_growth: float = 1.26
    cap_max_iterations: int = 12

    CAP_STRATEGIES = ("voxel", "stride")

    def validate(self) -> None:
        if self.voxel_size < 0.0:
            raise AG3SConfigError(f"pointcloud.voxel_size must be >= 0, got {self.voxel_size}")
        if self.max_points < 1:
            raise AG3SConfigError(f"pointcloud.max_points must be >= 1, got {self.max_points}")
        if not 0.0 <= self.depth_min < self.depth_max:
            raise AG3SConfigError(
                f"pointcloud must satisfy 0 <= depth_min < depth_max, got {(self.depth_min, self.depth_max)}"
            )
        if self.depth_scale <= 0.0:
            raise AG3SConfigError(f"pointcloud.depth_scale must be > 0, got {self.depth_scale}")
        if self.range_max is not None and self.range_max <= 0.0:
            raise AG3SConfigError(f"pointcloud.range_max must be > 0 or null, got {self.range_max}")
        if self.cap_strategy not in self.CAP_STRATEGIES:
            raise AG3SConfigError(
                f"pointcloud.cap_strategy must be one of {self.CAP_STRATEGIES}, got {self.cap_strategy!r}"
            )
        if self.cap_voxel_growth <= 1.0:
            raise AG3SConfigError(
                f"pointcloud.cap_voxel_growth must be > 1 (it has to shrink the cloud), "
                f"got {self.cap_voxel_growth}"
            )
        if self.cap_max_iterations < 1:
            raise AG3SConfigError(
                f"pointcloud.cap_max_iterations must be >= 1, got {self.cap_max_iterations}"
            )


@dataclasses.dataclass(frozen=True)
class SupportSurfaceConfig:
    """RANSAC plane fitting for tables and floors.

    Managed separately from object clustering because a plane converted to a half-space costs the TO
    one linear row, while the same surface voxelized costs thousands of sphere rows and still
    approximates it badly.
    """

    enabled: bool = True
    max_planes: int = 2
    distance_threshold: float = 0.008
    min_inliers: int = 500
    min_inlier_ratio: float = 0.05
    max_iterations: int = 200
    normal_reference: tuple[float, float, float] = (0.0, 0.0, 1.0)
    max_normal_angle_deg: float = 25.0  # reject planes too far from `normal_reference`
    safety_margin: float = 0.01
    seed: int = 0

    def validate(self) -> None:
        if self.max_planes < 0:
            raise AG3SConfigError(f"support_surface.max_planes must be >= 0, got {self.max_planes}")
        if self.distance_threshold <= 0.0:
            raise AG3SConfigError(
                f"support_surface.distance_threshold must be > 0, got {self.distance_threshold}"
            )
        if self.max_iterations < 1:
            raise AG3SConfigError(f"support_surface.max_iterations must be >= 1, got {self.max_iterations}")


@dataclasses.dataclass(frozen=True)
class CollisionCandidateConfig:
    """Residual clustering and the conservative-unknown policy.

    `preserve_unknown=True` is the safe default and the reason the pipeline never calls anything
    "background": points that cluster too small to be an object are still mass the robot can hit, so
    they become `UNKNOWN_GEOMETRY` candidates rather than being discarded.
    """

    cluster_method: str = "dbscan"  # dbscan | region_growing
    preserve_unknown: bool = True
    eps: float = 0.03
    min_points: int = 20
    unknown_min_points: int = 3  # below this even the conservative mode has no shape to fit
    max_clusters: int = 64
    # Clusters beyond `max_clusters`, and residue below `unknown_min_points`, are folded into this
    # many conservative aggregates rather than discarded. One aggregate over scattered clutter would
    # span the workspace and forbid every trajectory; a few local ones stay contained without being
    # useless. Set to 0 to get the old drop-it behaviour, which reports GEOMETRY_INCOMPLETE.
    overflow_groups: int = 4
    track_association_radius: float = 0.08  # metres, centroid nearest-neighbour gate
    track_max_missed: int = 3  # frames a track survives without a match

    METHODS = ("dbscan", "region_growing")

    def validate(self) -> None:
        if self.cluster_method not in self.METHODS:
            raise AG3SConfigError(
                f"collision_candidate.cluster_method must be one of {self.METHODS}, got {self.cluster_method!r}"
            )
        if self.eps <= 0.0:
            raise AG3SConfigError(f"collision_candidate.eps must be > 0, got {self.eps}")
        if self.track_association_radius <= 0.0:
            raise AG3SConfigError("collision_candidate.track_association_radius must be > 0")
        if self.overflow_groups < 0:
            raise AG3SConfigError(
                f"collision_candidate.overflow_groups must be >= 0, got {self.overflow_groups}"
            )


@dataclasses.dataclass(frozen=True)
class GeometryConfig:
    """Primitive fitting. Priority is sphere -> capsule -> box -> ellipsoid (spec §3.5)."""

    primitive: str = "sphere"  # sphere | capsule | box | ellipsoid
    safety_margin: float = 0.05
    max_candidates: int = 32
    max_primitives_per_candidate: int = 1
    min_radius: float = 0.005  # floor, so a 3-point cluster still has a usable shape
    # One budget for every perception error that makes the fitted shape smaller than the real one:
    # depth noise, voxel quantization, camera registration, hand-eye calibration. It is added to the
    # **primitive radius** and to nothing else. Adding it to `d_safe` as well would count the same
    # centimetre twice and quietly double the robot's detours.
    perception_uncertainty: float = 0.0

    def validate(self) -> None:
        try:
            PrimitiveType(self.primitive)
        except ValueError as exc:
            raise AG3SConfigError(
                f"geometry.primitive must be one of {[p.value for p in PrimitiveType]}, got {self.primitive!r}"
            ) from exc
        if self.safety_margin < 0.0:
            raise AG3SConfigError(f"geometry.safety_margin must be >= 0, got {self.safety_margin}")
        if self.max_candidates < 1:
            raise AG3SConfigError(f"geometry.max_candidates must be >= 1, got {self.max_candidates}")
        if self.perception_uncertainty < 0.0:
            raise AG3SConfigError(
                f"geometry.perception_uncertainty must be >= 0, got {self.perception_uncertainty}"
            )

    @property
    def primitive_type(self) -> PrimitiveType:
        return PrimitiveType(self.primitive)


@dataclasses.dataclass(frozen=True)
class PhaseRule:
    """What one phase does to the clearance between the **contact-authorized links** and the target.

    Read the scope carefully, because it is the safety property. A phase rule never applies to the
    whole robot: it applies to the links an external `ContactPolicyContext` has authorized to touch
    the target, and to nothing else. The torso, the forearm and the opposite arm keep the full
    margin in every phase, `GRASP` included.

    `collision_enabled` is retained for API compatibility and is **always True for the target now**.
    It used to be False at `GRASP`, which deleted the target's constraint rows outright — the robot
    could put its elbow through the object it was grasping. Contact is expressed as a margin of
    `contact_margin` (0 by default) on the permitted links: touching is allowed, penetrating is not,
    and the row stays in the graph either way.
    """

    collision_enabled: bool
    contact_permission: bool
    margin_scale: float

    def validate(self, name: str) -> None:
        if self.margin_scale < 0.0:
            raise AG3SConfigError(f"contact.phase_rules.{name}.margin_scale must be >= 0")
        if not self.collision_enabled:
            raise AG3SConfigError(
                f"contact.phase_rules.{name}.collision_enabled must be True. Disabling a candidate "
                "removes its constraint rows entirely; allowed contact is expressed as "
                "contact_margin on the authorized links, not by deleting the constraint."
            )


DEFAULT_TARGET_PHASE_RULES: dict[str, PhaseRule] = {
    # Scales below apply **only to the contact-authorized links** (see `ContactConfig.contact_links`).
    # TRANSIT     -> the target is just another obstacle while moving past it
    Phase.TRANSIT.value: PhaseRule(collision_enabled=True, contact_permission=False, margin_scale=1.0),
    # APPROACH    -> the authorized gripper may come closer
    Phase.APPROACH.value: PhaseRule(collision_enabled=True, contact_permission=False, margin_scale=0.4),
    # PRE_GRASP   -> controlled approach, still no contact
    Phase.PRE_GRASP.value: PhaseRule(collision_enabled=True, contact_permission=False, margin_scale=0.1),
    # GRASP       -> contact permitted for the authorized links, at `contact_margin`
    Phase.GRASP.value: PhaseRule(collision_enabled=True, contact_permission=True, margin_scale=0.0),
}

#: Links the RB-Y1 URDF names for each gripper. Only these may ever be relaxed against the target,
#: and only when an external context names their manipulator. The forearm links are deliberately
#: absent: a forearm pressing into the object it is grasping is a collision, not a grasp.
DEFAULT_CONTACT_LINKS: dict[str, tuple[str, ...]] = {
    "right": ("ee_right", "ee_finger_r1", "ee_finger_r2"),
    "left": ("ee_left", "ee_finger_l1", "ee_finger_l2"),
}


@dataclasses.dataclass(frozen=True)
class ContactConfig:
    """Phase- and link-conditioned target clearance.

    `phase_aware=False` is the ablation: the target keeps the full margin on every link in every
    phase, which makes grasping impossible and is exactly the point of measuring it.

    `contact_margin` is the surface distance permitted between an authorized link and the target
    once `contact_permission` holds. Zero means "may touch"; it does **not** mean "unconstrained" —
    the row is still there and still forbids penetration. A negative value would authorize
    penetration and is rejected.
    """

    phase_aware: bool = True
    contact_margin: float = 0.0
    contact_links: dict[str, tuple[str, ...]] = dataclasses.field(
        default_factory=lambda: {k: tuple(v) for k, v in DEFAULT_CONTACT_LINKS.items()}
    )
    phase_rules: dict[str, PhaseRule] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_TARGET_PHASE_RULES)
    )

    def validate(self) -> None:
        for name, rule in self.phase_rules.items():
            Phase.parse(name)  # raises on an unknown phase name
            rule.validate(name)
        missing = {p.value for p in Phase} - set(self.phase_rules)
        if missing:
            raise AG3SConfigError(f"contact.phase_rules is missing {sorted(missing)}")
        if self.contact_margin < 0.0:
            raise AG3SConfigError(
                f"contact.contact_margin must be >= 0 (a negative margin authorizes penetration), "
                f"got {self.contact_margin}"
            )
        for name in self.contact_links:
            Manipulator.parse(name)  # raises on an unknown manipulator name

    def rule_for(self, phase: Phase) -> PhaseRule:
        if not self.phase_aware:
            return PhaseRule(collision_enabled=True, contact_permission=False, margin_scale=1.0)
        return self.phase_rules[Phase.parse(phase).value]

    def links_for(self, manipulator: "Manipulator | str") -> tuple[str, ...]:
        """Contact-authorized links for one manipulator; empty for an unknown one (fail closed)."""
        try:
            key = Manipulator.parse(manipulator).value
        except ValueError:
            return ()
        return tuple(self.contact_links.get(key, ()))


@dataclasses.dataclass(frozen=True)
class ConstraintConfig:
    """Constraint emission.

    `squared_distance` defaults to True because ``||.||`` is non-differentiable at the origin and an
    interior-point method will happily walk a robot sphere straight through a candidate centre while
    the gradient is undefined. The non-squared form is kept only so the difference can be measured.
    """

    squared_distance: bool = True
    horizon: int = 20
    include_support_surfaces: bool = True
    include_target: bool = True
    # Slots held back from normal candidates to carry conservative aggregates when the scene has
    # more constraint spheres than slots. Two rather than one because a single aggregate over
    # spatially scattered clutter spans the workspace and forbids every trajectory; two local ones
    # stay contained without being useless. Set to 0 only in an ablation that wants the old
    # drop-the-smallest behaviour, and expect `GEOMETRY_INCOMPLETE` when it binds.
    reserved_overflow_slots: int = 2
    # Fixed attached-object primitive slots. Reserved whether or not anything is held, so attaching
    # and detaching never rebuilds the graph.
    max_attached_primitives: int = 4

    def validate(self) -> None:
        if self.horizon < 1:
            raise AG3SConfigError(f"constraint.horizon must be >= 1, got {self.horizon}")
        if self.reserved_overflow_slots < 0:
            raise AG3SConfigError(
                f"constraint.reserved_overflow_slots must be >= 0, got {self.reserved_overflow_slots}"
            )
        if self.max_attached_primitives < 0:
            raise AG3SConfigError(
                f"constraint.max_attached_primitives must be >= 0, got {self.max_attached_primitives}"
            )


@dataclasses.dataclass(frozen=True)
class TimingConfig:
    """Freshness limits for multi-camera input, and the fusion voxel.

    Three separate ages because they fail differently. A stale **image** shows the world as it was; a
    stale **robot state** puts a fresh image at the wrong pose, which is worse because the geometry
    looks plausible and is in the wrong place; and **skew** between cameras means the three views
    disagree about a scene that moved between them.

    The defaults are deliberately loose (100 ms), because the numbers that matter come from the real
    robot's transport and control rates and nobody has measured them here yet. What matters is that
    they are configured rather than hard-coded, and that exceeding them degrades the frame's validity
    instead of being silently absorbed.
    """

    max_state_age_sec: float = 0.1  # gap between an image and the joint state used to place it
    max_transform_age_sec: float = 0.1  # gap between an image and its extrinsics
    max_camera_skew_sec: float = 0.1  # spread across the cameras of one fused frame
    # Voxel used to decide that two cameras are looking at the same piece of the world. Coarser than
    # `pointcloud.voxel_size` on purpose: registration error between cameras is larger than the
    # quantization within one, so a fusion voxel as fine as the per-camera one would leave the same
    # surface duplicated as two parallel sheets a few millimetres apart.
    fusion_voxel_size: float = 0.01
    #: Cameras this deployment expects. An empty tuple means "whatever arrived", which is right for
    #: a bench test and wrong for a robot — set it, and a dropped camera becomes visible.
    expected_cameras: tuple[str, ...] = ()

    def validate(self) -> None:
        for name in ("max_state_age_sec", "max_transform_age_sec", "max_camera_skew_sec"):
            if getattr(self, name) < 0.0:
                raise AG3SConfigError(f"timing.{name} must be >= 0, got {getattr(self, name)}")
        if self.fusion_voxel_size <= 0.0:
            raise AG3SConfigError(
                f"timing.fusion_voxel_size must be > 0, got {self.fusion_voxel_size}"
            )
        for name in self.expected_cameras:
            CameraID.parse(name)  # raises on an unknown camera name


@dataclasses.dataclass(frozen=True)
class ProfilingConfig:
    enabled: bool = True
    warmup_frames: int = 0

    def validate(self) -> None:
        if self.warmup_frames < 0:
            raise AG3SConfigError(f"profiling.warmup_frames must be >= 0, got {self.warmup_frames}")


# -------------------------------------------------------------------------------------- root


_SECTIONS: dict[str, type] = {
    "attention": AttentionConfig,
    "clustering": ClusteringConfig,
    "pointcloud": PointCloudConfig,
    "support_surface": SupportSurfaceConfig,
    "collision_candidate": CollisionCandidateConfig,
    "geometry": GeometryConfig,
    "contact": ContactConfig,
    "constraint": ConstraintConfig,
    "timing": TimingConfig,
    "profiling": ProfilingConfig,
}


@dataclasses.dataclass(frozen=True)
class AG3SConfig:
    """Root config. Construct with `AG3SConfig()` for the spec §9 defaults."""

    attention: AttentionConfig = dataclasses.field(default_factory=AttentionConfig)
    clustering: ClusteringConfig = dataclasses.field(default_factory=ClusteringConfig)
    pointcloud: PointCloudConfig = dataclasses.field(default_factory=PointCloudConfig)
    support_surface: SupportSurfaceConfig = dataclasses.field(default_factory=SupportSurfaceConfig)
    collision_candidate: CollisionCandidateConfig = dataclasses.field(
        default_factory=CollisionCandidateConfig
    )
    geometry: GeometryConfig = dataclasses.field(default_factory=GeometryConfig)
    contact: ContactConfig = dataclasses.field(default_factory=ContactConfig)
    constraint: ConstraintConfig = dataclasses.field(default_factory=ConstraintConfig)
    timing: TimingConfig = dataclasses.field(default_factory=TimingConfig)
    profiling: ProfilingConfig = dataclasses.field(default_factory=ProfilingConfig)
    frame_id: str = "base"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Per-section validation. Cross-section relations are deliberately *not* errors.

        In particular `geometry.max_candidates < collision_candidate.max_clusters` is legal and
        common: slots are a hard cap, clusters beyond it are dropped by score, and the pipeline
        reports `DEGRADED` when that happens. Making it an error would forbid the very configuration
        a latency ablation wants.
        """
        for name in _SECTIONS:
            getattr(self, name).validate()

    # --- construction -------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "AG3SConfig":
        data = dict(data or {})
        unknown = set(data) - set(_SECTIONS) - {"frame_id"}
        if unknown:
            raise AG3SConfigError(
                f"unknown top-level config key(s): {sorted(unknown)}; expected {sorted(set(_SECTIONS) | {'frame_id'})}"
            )
        kwargs: dict[str, Any] = {}
        if "frame_id" in data:
            kwargs["frame_id"] = str(data["frame_id"])
        for name, section_cls in _SECTIONS.items():
            if name in data:
                kwargs[name] = _build_section(section_cls, name, data[name])
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "AG3SConfig":
        import yaml  # local: keeps `import config` free of a PyYAML dependency

        text = pathlib.Path(path).read_text()
        return cls.from_dict(yaml.safe_load(text) or {})

    def replace(self, **overrides: Any) -> "AG3SConfig":
        """Section-level override, e.g. ``cfg.replace(geometry=dataclasses.replace(cfg.geometry, ...))``."""
        return dataclasses.replace(self, **overrides)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "AG3SConfig":
        """Merge a partial dict (same shape as the YAML) onto this config.

        This is the ablation entry point: ``cfg.with_overrides({"geometry": {"primitive": "capsule"}})``
        touches one field and leaves everything else, including nested sections, alone.
        """
        merged = self.to_dict()
        for key, value in (overrides or {}).items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return AG3SConfig.from_dict(merged)

    # --- serialization ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"frame_id": self.frame_id}
        for name in _SECTIONS:
            out[name] = _section_to_dict(getattr(self, name))
        return out


def _build_section(section_cls: type, name: str, raw: Any) -> Any:
    if not isinstance(raw, Mapping):
        raise AG3SConfigError(f"config section {name!r} must be a mapping, got {type(raw).__name__}")
    fields = {f.name: f for f in dataclasses.fields(section_cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise AG3SConfigError(f"unknown key(s) in {name!r}: {sorted(unknown)}; expected {sorted(fields)}")
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if name == "contact" and key == "phase_rules":
            kwargs[key] = {
                str(phase): (rule if isinstance(rule, PhaseRule) else PhaseRule(**dict(rule)))
                for phase, rule in dict(value).items()
            }
        elif name == "contact" and key == "contact_links":
            kwargs[key] = {str(m): tuple(links) for m, links in dict(value).items()}
        elif key == "expected_cameras" and value is not None:
            kwargs[key] = tuple(str(v) for v in value)
        elif key in ("percentile_range", "normal_reference") and value is not None:
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return section_cls(**kwargs)


def _section_to_dict(section: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in dataclasses.fields(section):
        value = getattr(section, f.name)
        if isinstance(value, dict) and value and isinstance(next(iter(value.values())), PhaseRule):
            out[f.name] = {k: dataclasses.asdict(v) for k, v in value.items()}
        elif isinstance(value, dict) and value and isinstance(next(iter(value.values())), tuple):
            out[f.name] = {k: list(v) for k, v in value.items()}
        elif isinstance(value, tuple):
            out[f.name] = list(value)
        else:
            out[f.name] = value
    return out


__all__ = [
    "AG3SConfig",
    "AG3SConfigError",
    "AttentionConfig",
    "ClusteringConfig",
    "CollisionCandidateConfig",
    "ConstraintConfig",
    "ContactConfig",
    "DEFAULT_CONTACT_LINKS",
    "DEFAULT_TARGET_PHASE_RULES",
    "GeometryConfig",
    "PhaseRule",
    "PointCloudConfig",
    "ProfilingConfig",
    "SupportSurfaceConfig",
    "TimingConfig",
]
