"""AG3S orchestration. **No algorithm lives here.**

Every stage is implemented and unit-tested in its own module; this file only decides what runs in
what order, times it, and assembles the result. That constraint is worth stating because it is easy
to erode: the first "just a small special case" added here is the point at which a stage stops being
independently testable.

The order differs from the spec's section numbering in one place, deliberately. Spec §3.4 introduces
support-surface fitting after §3.3's target grounding, but the plane fit does not depend on the
target, and grounding *does* depend on the plane: objects stand on the table within any usable `eps`,
so Euclidean connectivity floods from the target across the whole surface unless the plane is already
identified (measured: one 41,745-point cluster swallowing the entire scene). So planes are fitted
first and the inlier mask is handed to grounding. Nothing about the method changes — only when it
runs.

Full order:

    1. scene reconstruction     depth or point cloud -> filtered cloud in the base frame
    2. robot self-filter        remove the robot's own body via the injected collision model
    3. support surface          RANSAC planes; the mask feeds stages 5 and 6
    4. attention lifting        A(u, v) -> per-point attention, nothing dropped
    5. target grounding         seeds -> 3D connectivity -> clusters -> score, or a failure status
    6. collision candidates     residual clustering, unknown geometry, phase rules, tracking
    7. primitive fitting        (inside 6; timed separately, see `collision_candidates`)
    8. constraint generation    fixed-slot CasADi parameters

Phase is an injected argument at every level. AG3S never infers it.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from benchmark.ag3s.stages.attention_lifting import lift, make_adapter
from benchmark.ag3s.constraints.clearance import ClearancePolicy
from benchmark.ag3s.stages.collision_candidates import CandidateTracker, generate_candidates
from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.runtime.profiler import StageProfiler
from benchmark.ag3s.stages.reconstruction import reconstruct
from benchmark.ag3s.stages.robot_filter import (
    filter_robot_points,
    inflation_table,
    resolve_self_filter_inflation,
    self_filter_mask,
)
from benchmark.ag3s.runtime.multiview import fuse_observations
from benchmark.ag3s.stages.support_surface import fit_support_surfaces
from benchmark.ag3s.stages.admissibility import (
    DestinationGeometry,
    ExclusionInvariantViolation,
    MaxOpening,
    assess,
    destination_overlap,
    hand_spheres,
    resolve_max_opening,
)
from benchmark.ag3s.stages.target_grounding import (
    LOST,
    GroundingResult,
    ManipulatedIdentity,
    TargetConfirm,
    ground_target,
)
from benchmark.ag3s.constraints.to_adapter import build_constraint_set
from benchmark.ag3s.runtime.degradation import ensure_reason, reason
from benchmark.ag3s.robot_models.held_object import HELD_FILTER_SLIP_M
from benchmark.ag3s.types import (
    DESTINATION_LABEL,
    TARGET_LABEL,
    AttachedCollisionGeometry,
    AttentionPointCloud,
    CameraObservation,
    TargetGeometry,
    CollisionConstraintSet,
    ConstraintValidity,
    ContactPolicyContext,
    GroundingStatus,
    Phase,
    PipelineStatus,
    PointCloud,
    RobotCollisionModel,
    SourceType,
)


@dataclasses.dataclass(frozen=True)
class _HeldTarget:
    """What `attach_from_target` reads off a target (T32 H2): the primary held sphere replaces the
    old centroid + max-distance `bounding_geometry`."""

    id: int
    points: Any
    bounding_geometry: Any


class AG3S:
    """The pipeline object. One instance per camera/robot session.

    State held across frames, and why:

    * `CandidateTracker` — stable candidate ids, which TO warm-starting depends on.
    * `ConstraintBuilder` — the symbolic NLP structure, built once. Rebuilding it per frame would
      make warm-starting impossible, which is the whole reason for fixed slots.
    * `StageProfiler` — accumulated latency.

    `robot_model` is optional. Without it the self-filter is skipped and no constraints are emitted
    (there is no robot to write them against); the rest of the pipeline still runs, which is what
    makes perception debuggable on a machine with no robot description.

    `constraint_robot_model` exists because the two uses of a robot model want different things, and
    on a real robot they diverge sharply:

    * the **self-filter** must cover everything a camera can see of the robot — base, wheels,
      grippers, the lot — or the leftovers cluster into a phantom obstacle attached to the robot;
    * the **constraints** only need the links that move and can hit something, and every extra
      sphere multiplies the row count by `horizon * max_candidates`.

    On RB-Y1 the full visible body is 194 spheres and the arm chain is 61, which at H=20 and 32 slots
    is the difference between ~132k rows and ~41k. Defaults to `robot_model`, so a caller who does
    not care sees no change.
    """

    def __init__(
        self,
        config: AG3SConfig | None = None,
        *,
        robot_model: Optional[RobotCollisionModel] = None,
        constraint_robot_model: Optional[RobotCollisionModel] = None,
        attention_adapter: Any = None,
        max_support_surfaces: Optional[int] = None,
        attached_parent_links: Sequence[str] = (),
        strict_invariants: Optional[bool] = None,
    ):
        self.config = config or AG3SConfig()
        self.robot_model = robot_model
        self.constraint_robot_model = constraint_robot_model or robot_model
        self.attention_adapter = attention_adapter
        self.tracker = CandidateTracker(self.config.collision_candidate)
        #: T26: raise `ExclusionInvariantViolation` instead of refusing + recording. `None` reads
        #: `AG3S_STRICT_INVARIANTS` (the unit tests set it to 1 in `tests/conftest.py`).
        self.strict_invariants = (os.environ.get("AG3S_STRICT_INVARIANTS", "") == "1"
                                  if strict_invariants is None else bool(strict_invariants))
        #: T26: the widest object the gripper can close around — the fingers' collision geometry at
        #: the finger joints' open limit, **from the injected robot model** (self-filter model first:
        #: it is the whole body and never thinned by `--sphere-*`), or `clustering.
        #: gripper_max_opening`. Unknown → nothing is graspable → no manipulated object (fail-closed).
        self._max_opening: MaxOpening = resolve_max_opening(
            self.config.clustering.gripper_max_opening, robot_model, self.constraint_robot_model)
        _log = logging.getLogger(__name__)
        if self._max_opening.known:
            _log.info("AG3S gripper max opening %.1f mm (source %s) — manipulated candidates must "
                      "have a narrowest principal extent at or below it",
                      self._max_opening.value_m * 1000.0, self._max_opening.source)
        else:
            _log.warning("AG3S gripper max opening UNKNOWN (%s) — no cluster is graspable, so no "
                         "manipulated object, no exclusion and no contact permission will be "
                         "granted. Give a robot model with finger geometry or "
                         "clustering.gripper_max_opening", self._max_opening.detail)
        #: The only cross-frame state grounding has (T5e). `ground_target` still decides from one
        #: frame; this decides whether a *different* object has led long enough to take the target
        #: over. It lives here because the episode lives here — `reset()` is the episode boundary.
        #: Since T26 it also owns admissibility (only graspable, non-destination clusters compete)
        #: and the destination registry.
        self._target_confirm = TargetConfirm(
            self.config.clustering,
            max_opening=self._max_opening.value_m if self._max_opening.known else None,
            overlap_distance=self._overlap_distance())
        #: T26: the manipulated object as of the last frame **that passed the exclusion gate**
        #: (`_exclusion_gate`) — what the T19 guard is centred on. None when the gate refused.
        self._gated_manipulated: Optional[ManipulatedIdentity] = None
        #: T26: a grasp is in progress (closing … held) — set by the caller that owns the grasp
        #: (`SafePolicy`). Challengers do not count while it is set, and while an object is attached.
        self._grasp_active = False
        #: SUBTASK-c: this request's subtask probabilities (`set_subtask`), handed to
        #: `TargetConfirm.note_subtask` by the next grounding call and consumed by it.
        self._subtask_p: Optional[dict[str, float]] = None
        #: The **manipulated object** after the latest frame (T20): id, observation state
        #: (`visible | occluded | lost`) and last observed geometry, as `TargetConfirm` left it.
        #: Contact permission (`manipulated_link_margin`), target exclusion (`target_field_exclude`,
        #: target-free layer, `_target_ball`), the fine-window centre, the `TARGET_LABEL` points and
        #: the T19 self-filter guard all read *this*, not the frame's attention target. Before a
        #: grasp the two coincide; they part when the object is hidden or attention moves on.
        self._manipulated: Optional[ManipulatedIdentity] = None
        #: The self-filter margin, resolved **once** from `pointcloud.self_filter_inflation[_by_link]`
        #: and the self-filter model's link names (T19). A float when there is no per-link override
        #: (the pre-T19 value itself), else one value per sphere. The cloud filter and the depth
        #: robot mask are both handed *this object*, so they cannot disagree on the robot.
        #: Resolving here also means a mistyped link name stops construction, not frame 1.
        self._self_filter_inflation = self._resolve_self_filter_inflation()
        #: This frame's target-guard centroid (`None` = no guard), fixed at the start of `_run`
        #: from the **manipulated object's** centroid *before* this frame's grounding (T20; it was
        #: `TargetConfirm.held_centroid`, which is the same point), and read by `_robot_mask_for`
        #: later in the same frame — after grounding may already have moved it.
        self._frame_guard_centre: Optional[np.ndarray] = None
        #: This frame's **effective** guard radius (T21, user ruling 2026-09-28): the configured
        #: `self_filter_target_guard_radius` when it is non-zero, otherwise the manipulated object's
        #: radius + one fine voxel (`_guard_radius_for`). 0.0 = no guard this frame.
        self._frame_guard_radius: float = 0.0
        #: `self.config` with `pointcloud.self_filter_target_guard_radius` set to the effective
        #: radius — handed to the self-filter call sites (cloud filter, fused path, depth robot
        #: mask) so `stages/robot_filter` reads one radius without being changed. It *is*
        #: `self.config` whenever the two radii agree.
        self._frame_filter_config: AG3SConfig = self.config
        #: Pixels the guard kept out of this frame's depth robot masks, one entry per mask built.
        self._mask_guard_px: list[int] = []
        #: Kept across frames so the ESDF's local update has something to be incremental against.
        self._esdf_builder = None
        self.profiler = StageProfiler(
            enabled=self.config.profiling.enabled,
            warmup_frames=self.config.profiling.warmup_frames,
        )
        self.frame_index = 0
        # Set only by an explicit `attach()` and cleared only by an explicit `detach()`. Perception
        # failure never touches it: the object is still in the gripper whatever the cameras can see.
        self._attached: Optional[AttachedCollisionGeometry] = None
        #: T32 H2: how the held object's spheres were made at the last `attach()` (fit, spheres,
        #: frame link) — `metrics["held_object"]`. None when nothing is held.
        self._held_record: Optional[dict[str, Any]] = None
        #: T32 H2: the robot state each recent frame was captured at, keyed by the grounding frame
        #: (`ManipulatedIdentity.frame` / `last_seen_frame`). `attach()` snapshots the manipulated
        #: geometry with the state **it was observed at**, not the attach request's — see `attach`.
        self._state_by_frame: dict[int, np.ndarray] = {}
        #: T32 H3: the last accepted surface fit of the manipulated object before a grasp
        #: (`_target_ball`) — `(centre, radius)`, base frame. Its place on the table is one of the
        #: held object's old traces cleared at attach.
        self._last_object_fit: Optional[tuple[np.ndarray, float]] = None
        #: T32 H3: extra spheres (base frame) whose TSDF voxels are set free on the first held
        #: frames — the object's pre-grasp place and its pose at the geometry's capture — and how
        #: many held frames still get them.
        self._held_first_free: list[tuple[np.ndarray, float]] = []
        self._held_first_free_frames: int = 0
        #: T34 J1: the last processed frame's support surfaces and ESDF field. `attach()` runs
        #: before its frame is processed (T22 order (a)), so these are what it can see of the table:
        #: the plane the held query spheres must not cross, and the field's surface band over it.
        self._last_support_surfaces: list = []
        self._last_esdf_field = None
        #: T34 J1: the held object's **self-filter** spheres (parent frame) — the fit spheres before
        #: the support lift. The filter copy is left as T32 made it (fit + pad + slip); only the
        #: query spheres are lifted. None when nothing is held (or a prebuilt attachment).
        self._held_filter_local: Optional[list[tuple[np.ndarray, float]]] = None
        #: T34 J2: what the held spheres are compared with after attach — the attach frame index,
        #: the unlifted primary centre (parent frame), its base position at the attach request, and
        #: the attached geometry's centroid (base).
        self._held_check_ref: Optional[dict[str, Any]] = None
        #: T34 J1: clearance of the lifted query spheres over the support plane (m) — `None` =
        #: measure the field's surface band at attach (`held_object.field_support_band`), else
        #: this value. A switch for measurement/ablation.
        self.held_support_clearance_m: Optional[float] = None
        #: T34 J1: lift the held query spheres off the support at attach. Ablation switch.
        self.held_support_lift: bool = True
        #: T32 H3: free the held object's traces in the TSDF (cuRobo backend). Switch for ablation.
        self.free_held_traces: bool = True
        #: T32 H1: the held object's spheres join the **self-filter** sphere set while attached
        #: (cuRobo attach semantics) — its pixels are removed like the robot's and never reach the
        #: TSDF. A public switch for ablation/measurement only; the live path keeps it on.
        self.self_filter_held_object: bool = True
        #: T32 H1: extra radius (m) the held spheres get **in the self-filter only** — the measured
        #: in-hand slip (`held_object.HELD_FILTER_SLIP_M`, 20 mm). The query spheres keep fit + one
        #: fine voxel (together fit + 25 mm). 0 = the filter uses exactly the query spheres.
        self.held_filter_slip_m: float = HELD_FILTER_SLIP_M
        #: This frame's self-filter model and inflation (T32 H1): `self.robot_model` / the resolved
        #: inflation, or — while attached — the model wrapped with the held spheres
        #: (`HeldSphereFilterModel`) and the inflation extended with 0 for them. Fixed at the start
        #: of `_run` so the cloud filter, the fused path and the depth robot mask see one robot.
        self._frame_filter_model: Optional[RobotCollisionModel] = robot_model
        self._frame_filter_inflation = None
        #: Depth pixels inside the held spheres this frame, one entry per depth robot mask (T32 H1).
        self._mask_held_px: list[int] = []
        #: T29: the finger joints the models use this frame and where they came from
        #: (`set_finger_joints`) — `metrics["finger_joints"]`. Until a caller sets them, whatever the
        #: models were built with (RB-Y1 default: 0 = closed, the pre-T29 behaviour).
        self._finger_record: dict[str, Any] = {"source": "model_default",
                                               "applied": self._finger_values_applied()}
        self._max_support_surfaces = (
            self.config.support_surface.max_planes
            if max_support_surfaces is None
            else int(max_support_surfaces)
        )
        self._builder = None
        if self.constraint_robot_model is not None:
            from benchmark.ag3s.constraints.constraint_builder import ConstraintBuilder

            self._builder = ConstraintBuilder(
                self.constraint_robot_model,
                self.config.constraint,
                self.config.geometry,
                max_support_surfaces=self._max_support_surfaces,
                clearance_policy=ClearancePolicy.from_config(
                    self.config.contact, self.config.geometry, self.config.support_surface
                ),
                attached_parent_links=attached_parent_links,
            )

    @property
    def builder(self):
        """The `ConstraintBuilder`, or None when no robot model was injected."""
        return self._builder

    # ------------------------------------------------------------------ self-filter settings
    def _resolve_self_filter_inflation(self):
        cfg = self.config.pointcloud
        if self.robot_model is None:
            return None
        names = (getattr(self.robot_model, "sphere_link_names", None)
                 if cfg.self_filter_inflation_by_link else None)
        return resolve_self_filter_inflation(names, cfg)

    def self_filter_settings(self) -> dict[str, Any]:
        """What the self-filter will **actually** apply — for the start log and the run record.

        `inflation_effective` is the per-link table after group expansion and precedence, not the
        config as written: the point is to show the value that reaches `robot_sphere_mask`, which is
        what guide §3.2 found missing (a YAML value that never reached the live path).
        """
        cfg = self.config.pointcloud
        model = self.robot_model
        names = None if model is None else getattr(model, "sphere_link_names", None)
        n_spheres = None
        if model is not None:
            n_spheres = getattr(model, "n_spheres", None)
            if n_spheres is None and names is not None:
                n_spheres = len(names)
        infl = self._self_filter_inflation
        return {
            "enabled": bool(cfg.self_filter and model is not None),
            "n_spheres": None if n_spheres is None else int(n_spheres),
            "inflation_default": float(cfg.self_filter_inflation),
            "inflation_by_link": dict(cfg.self_filter_inflation_by_link),
            "inflation_effective": ({} if infl is None else inflation_table(names, infl)),
            "target_guard_radius": float(cfg.self_filter_target_guard_radius),
            # T21: 0 no longer means "off" — it means "per object" (`_guard_radius_for`).
            "target_guard_rule": ("fixed" if cfg.self_filter_target_guard_radius > 0.0
                                  else "manipulated_radius_plus_voxel"),
            "target_guard_pad_m": self._guard_pad(),
        }

    def reset(self, destination_points: Optional[np.ndarray] = None) -> None:
        """Clear cross-frame state. Call between episodes, never mid-episode.

        This *does* drop an attached object, because an episode boundary is an explicit statement
        that the previous task is over. Mid-episode there is no path from perception to detachment.

        `destination_points` (T26): a caller that knows where the object goes (the crate) names it
        here — `source="injected"`, and it takes precedence over perception's registration for the
        whole episode. Without it the destination is registered from perception
        (`stages/admissibility.DestinationRegistry`).
        """
        self.tracker.reset()
        self.profiler.reset()
        self._target_confirm.reset(destination_points)
        self._manipulated = None
        self._gated_manipulated = None
        self._grasp_active = False
        self._subtask_p = None
        self.frame_index = 0
        self._attached = None
        self._held_record = None
        self._state_by_frame = {}
        self._last_object_fit = None
        self._held_first_free = []
        self._held_first_free_frames = 0
        self._held_filter_local = None
        self._held_check_ref = None
        self._last_support_surfaces = []
        self._last_esdf_field = None
        self._esdf_builder = None

    @property
    def manipulated(self) -> Optional[ManipulatedIdentity]:
        """The manipulated object after the latest frame (T20), or None before one is confirmed."""
        return self._manipulated

    @property
    def destination(self) -> Optional[DestinationGeometry]:
        """Where the manipulated object is being taken (T26): injected at `reset`/`process`, or
        registered from perception. Never excluded from any field; never admissible as the
        manipulated object. `SafePolicy` reads this for the latch and for `placed_fn`."""
        return self._target_confirm.destination

    @property
    def max_opening(self) -> MaxOpening:
        """The gripper's max inner opening and where it came from (T26)."""
        return self._max_opening

    def set_grasp_active(self, active: bool) -> None:
        """The caller that owns the grasp says a grasp is in progress (T26 §4).

        From the closing command to the release, the manipulated object must not change — a switch
        there would put the apple back into the gripper's field between closing and attach. While
        set (and whenever an object is attached) challengers do not count toward a switch.
        """
        self._grasp_active = bool(active)

    def set_subtask(self, p: Optional[Mapping[str, Any]]) -> None:
        """This request's subtask label probabilities (SUBTASK-c), as `set_grasp_active` is given.

        `p`: `{"pick": .., "place": .., "home": ..}` — or the whole `result["subtask"]` record
        (`{"p": {...}, "argmax": .., "probe": ..}`) — or None (no label this request). The next
        grounding call hands it to `TargetConfirm.note_subtask` (debounced there) and consumes it;
        a request without a call counts as None. Only `clustering.subtask_gate` lets the confirmed
        label change a decision: under `place` / `home` no new manipulated object is taken.
        """
        if p is not None and isinstance(p.get("p"), Mapping):
            p = p["p"]
        self._subtask_p = None if p is None else {str(k): float(v) for k, v in p.items()}

    @property
    def subtask_label(self) -> Optional[str]:
        """The confirmed subtask label after the latest grounding call (None before one)."""
        return self._target_confirm.subtask_label

    def set_placed(self, placed: bool) -> None:
        """The caller that owns the grasp latch says it is in PLACED (SUBTASK-c B3), given where
        `set_grasp_active` is. Sticky for the episode (`reset()` clears it). With
        `clustering.subtask_gate` on and a confirmed `home` label, the placed manipulated object is
        then released — nothing is carved from it on. A revoked false attach is not PLACED."""
        self._target_confirm.note_placed(bool(placed))

    def set_finger_joints(self, values: Optional[Mapping[str, float]], *,
                          record: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """The finger joints **at capture time** for both robot models (T29).

        Before T29 every finger joint was pinned at 0 (closed) because it is not in `q`, so with the
        gripper open the model fingers sat in the middle of the object between the real fingers: the
        self-filter deleted the apple there and the constraints saw the apple inside the fingers.
        The caller that knows the measured opening (`SafePolicy`: the 16D state's gripper columns
        through `robot_models/gripper_state`) sets it here before `process*`; every numeric FK in
        this frame — self-filter, depth robot mask, fine window, attach, reach — then uses it.

        `values = None` restores each model's construction defaults (closed). Models without
        finger parameters (`set_joint_parameters`) are skipped. `record` is stored verbatim with the
        applied values in `metrics["finger_joints"]` from the next frame on. Returns that record.
        """
        seen: set[int] = set()
        for model in (self.robot_model, self.constraint_robot_model):
            setter = getattr(model, "set_joint_parameters", None)
            if model is None or not callable(setter) or id(model) in seen:
                continue
            seen.add(id(model))
            names = tuple(getattr(model, "param_joint_names", ()) or ())
            setter(None if values is None
                   else {k: float(v) for k, v in values.items() if k in names})
        self._finger_record = {
            "source": "model_default" if values is None else "caller",
            **dict(record or {}),
            "set_by_caller": values is not None,
            "applied": self._finger_values_applied(),
        }
        return dict(self._finger_record)

    def _finger_values_applied(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for role, model in (("self_filter", self.robot_model),
                            ("constraint", self.constraint_robot_model)):
            values = getattr(model, "joint_parameters", None) if model is not None else None
            if isinstance(values, dict):
                out[role] = {k: float(v) for k, v in values.items()}
        return out

    @property
    def finger_record(self) -> dict[str, Any]:
        """What `metrics["finger_joints"]` carries (T29)."""
        return self._finger_frame_record()

    def _finger_frame_record(self) -> dict[str, Any]:
        """This frame's finger joints, **read back from the models** (T29).

        `source`: whatever `set_finger_joints` was told (`set_by_caller: true`; SafePolicy says
        `state[7],state[15]` or `exec_feedback...`);
        `model_state` = someone set the models directly (offline: `camera_observation` from the
        MuJoCo scene); `model_default` = nobody did — the fingers are at the construction default
        (closed), which leaks open fingers through the self-filter now that `link_*_arm_5`'s
        oversized URDF capsule no longer hides them. That case is warned about once.
        """
        rec = dict(self._finger_record)
        rec["applied"] = self._finger_values_applied()
        if not rec.get("set_by_caller"):
            defaults = [getattr(m, "joint_parameters_are_default", True)
                        for m in (self.robot_model, self.constraint_robot_model)
                        if m is not None and getattr(m, "param_joint_names", ())]
            rec["source"] = "model_default" if all(defaults) else "model_state"
            if defaults and all(defaults) and not getattr(self, "_warned_finger_default", False):
                self._warned_finger_default = True
                logging.getLogger(__name__).warning(
                    "AG3S: finger joints are at the model default (closed) — nobody set the "
                    "measured opening (AG3S.set_finger_joints / SafePolicy / camera_observation). "
                    "With open hands the finger points survive the self-filter and the constraint "
                    "fingers sit between the real ones (T29)")
        return rec

    def _overlap_distance(self) -> float:
        """`clustering.destination_overlap_distance`, or one cloud voxel (T26)."""
        cfg = self.config
        if cfg.clustering.destination_overlap_distance is not None:
            return float(cfg.clustering.destination_overlap_distance)
        voxel = float(cfg.pointcloud.voxel_size)
        return voxel if voxel > 0.0 else float(cfg.clustering.eps)

    def admissibility_of(self, points: np.ndarray):
        """`Admissibility` of an arbitrary point set against this AG3S's max opening and its
        current destination — the same rule grounding applies to clusters (T26)."""
        cfg = self.config.clustering
        return assess(np.asarray(points, np.float64).reshape(-1, 3), extents=None,
                      max_opening_m=self._max_opening.value_m if self._max_opening.known else None,
                      destination=self.destination, overlap_distance=self._overlap_distance(),
                      overlap_fraction=float(cfg.destination_overlap_fraction))

    # ------------------------------------------------------- explicit grasp state, injected
    @property
    def attached(self) -> Optional[AttachedCollisionGeometry]:
        """The object AG3S believes is in the gripper, or None."""
        return self._attached

    def attach(
        self,
        geometry: AttachedCollisionGeometry | TargetGeometry,
        *,
        robot_state: Optional[np.ndarray] = None,
        parent_link: Optional[str] = None,
        allowed_contact_links: Any = (),
        label: str = "attached_object",
        timestamp: Optional[float] = None,
        reach: Optional[float] = None,
        hold_link: Optional[str] = None,
        geometry_state: Optional[np.ndarray] = None,
    ) -> AttachedCollisionGeometry:
        """Record an externally confirmed grasp. **AG3S never calls this itself.**

        **T32 H2 — a `TargetGeometry` is attached as fitted spheres on the palm.** The observed
        points are fitted with a sphere (`robot_models.held_object.fit_surface_sphere`: the centre
        of a convex object from its one visible side, radius ≤ half the max opening); the primary
        sphere is that fit + one fine voxel of pad, and observed points still outside it (stem,
        leaf) get a small cover sphere each (`covering_spheres`, ≤ `max_attached_primitives` in
        all). The snapshot rides `hold_link`: `None` = the palm the grasping finger hangs from
        (`palm_link_for`) when the constraint builder reserved it, else `parent_link` itself. The
        reach gate below is still measured from `parent_link` (the grasping finger, T22). These
        spheres are the held object from here on: self-filter spheres (H1), query spheres for the
        optimizer, the field's seed exclusion. `held_record` says how they were made.

        **The snapshot uses the state the geometry was captured at** (`geometry_state`; `None` =
        looked up for the manipulated object's `last_seen_frame`, else `robot_state`). The attach
        request comes one request (8 steps) or more after the geometry was observed (T22 order (a),
        T30 F4 `settled`), and by then the hand has moved — by the lift, in the recordings: the
        pre-T32 snapshot at the attach request's q put the held points 12–46 mm off the apple
        (T31-diag D3); at the capture q the fitted centre is 1.5–3.5 mm off (T32a §2). The reach gate
        still uses `robot_state` (where the hand is now).

        Pass an `AttachedCollisionGeometry` to attach a known shape, or a `TargetGeometry` together
        with `robot_state` and `parent_link` to snapshot what perception last fitted. Either way the
        result is a snapshot: from here on the object's existence is a fact about the gripper, not a
        claim about what the cameras can see.

        There is deliberately no grasp detector behind this. AG3S cannot tell a closed gripper
        holding a cup from a closed gripper holding nothing, and guessing would mean either an
        invisible object or a phantom one attached to the hand.

        **It does refuse the impossible (T22).** `reach` (m) bounds the distance between the parent
        link's origin and the object's centroid; beyond it the call raises `AttachRejected` with the
        measured distance and changes nothing. This is the last gate behind whoever decided the
        grasp: T14 seq 26 snapshotted a cluster 636 mm from `ee_finger_l1` and the phantom then
        violated −3.1 mm at the grasp instant. `SafePolicy` always passes `LatchConfig.reach`
        (default `attached.DEFAULT_ATTACH_REACH_M`, provisional — see there).

        `reach=None` (the default for direct callers) skips the check, so existing callers keep
        their behaviour: `trajopt/experiments/grasp_damage.py` attaches MuJoCo's true apple to
        `link_left_arm_6`, whose origin is ~270 mm from the apple (T17 FK), and a hand-sized default
        would refuse it. A limit only means something relative to a parent link.

        The attach frame is the given `parent_link`. With a finger link (the default grasp links are
        `ee_finger_l1`/`ee_finger_r1`) the snapshot rides on that finger, so any later change of the
        gripper opening moves the object with that finger rather than with the hand — callers should
        attach only once the opening has settled (the T22 latch does).
        """
        from benchmark.ag3s.constraints.attached import (
            AttachRejected,
            _link_pose_numeric,
            attach_from_target,
            attached_reach,
            parent_reach,
        )

        # **T26 — only an admissible object is ever held.** Attach grants contact permission to the
        # gripper and carves the object out of the field; the crate must never get either. Same
        # rule as grounding (narrowest principal extent ≤ max opening, not the destination), on the
        # points being attached. Extents are rotation-invariant, so parent-frame points serve.
        pts = getattr(geometry, "points", None)
        if (pts is None or not len(np.asarray(pts).reshape(-1, 3))) \
                and isinstance(geometry, AttachedCollisionGeometry):
            pts = self._primitive_surface_points(geometry)
        if pts is None or not len(np.asarray(pts).reshape(-1, 3)):
            raise AttachRejected(
                "inadmissible: the geometry has no points, so it cannot be shown to be graspable",
                parent_link=None if parent_link is None else str(parent_link))
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        if isinstance(geometry, AttachedCollisionGeometry):
            T = np.asarray(geometry.T_parent_object, np.float64)
            # For the destination overlap the points must be in the base frame; without a pose the
            # extent test still applies and the overlap is checked on the base-frame path only.
            adm = self.admissibility_of(pts @ T[:3, :3].T + T[:3, 3]) if robot_state is None \
                else self.admissibility_of(self._attached_in_base(geometry, robot_state, pts))
        else:
            adm = self.admissibility_of(pts)
        if not adm.admissible:
            raise AttachRejected(
                f"inadmissible ({adm.reason}): narrowest extent "
                f"{'?' if adm.narrowest_extent_m is None else f'{adm.narrowest_extent_m * 1000.0:.1f}'}"
                f" mm vs max opening "
                f"{'?' if adm.max_opening_m is None else f'{adm.max_opening_m * 1000.0:.1f}'} mm, "
                f"destination overlap {adm.destination_overlap:.2f}",
                parent_link=None if parent_link is None else str(parent_link))

        if isinstance(geometry, AttachedCollisionGeometry):
            if reach is not None:
                dist = attached_reach(geometry)
                if dist is not None and dist > float(reach):
                    raise AttachRejected(
                        f"reach: attached geometry centroid is {dist * 1000.0:.1f} mm from "
                        f"{geometry.parent_link!r}'s origin, over the {float(reach) * 1000.0:.0f} mm "
                        "limit", reach_m=dist, limit_m=float(reach),
                        parent_link=geometry.parent_link)
            self._attached = geometry
            # A prebuilt attachment keeps its own primitives (they are the held object's spheres
            # for the self-filter and the optimizer); nothing is fitted.
            self._held_record = {"mode": "prebuilt", "frame_link": geometry.parent_link,
                                 "n_spheres": len(geometry.primitives)}
            return self._attached
        if robot_state is None or parent_link is None:
            raise ValueError(
                "attaching a TargetGeometry needs the robot_state the grasp closed in and the "
                "parent_link it closed on; a later state would bake in however far the arm has moved"
            )
        if self.constraint_robot_model is None:
            raise ValueError("attaching needs a robot model to invert the parent link's pose")

        if reach is not None:
            centroid = getattr(geometry, "centroid", None)
            if centroid is None and getattr(geometry, "points", None) is not None \
                    and len(geometry.points):
                centroid = np.asarray(geometry.points, np.float64).reshape(-1, 3).mean(axis=0)
            if centroid is None:
                raise AttachRejected("reach: the target has neither a centroid nor points, so "
                                     "its distance to the hand cannot be checked",
                                     limit_m=float(reach), parent_link=str(parent_link))
            dist = parent_reach(self.constraint_robot_model, robot_state, str(parent_link),
                                centroid)
            if dist > float(reach):
                raise AttachRejected(
                    f"reach: target centroid is {dist * 1000.0:.1f} mm from {parent_link!r}'s "
                    f"origin, over the {float(reach) * 1000.0:.0f} mm limit",
                    reach_m=dist, limit_m=float(reach), parent_link=str(parent_link))

        frame_link = self._resolve_hold_link(str(parent_link), hold_link)
        primitives, held_record = self._held_primitives(pts)
        snap_state, snap_source = self._snapshot_state(geometry, robot_state, geometry_state)
        T_snap = _link_pose_numeric(self.constraint_robot_model, snap_state, frame_link)
        T_now = _link_pose_numeric(self.constraint_robot_model, robot_state, frame_link)
        T_parent = np.linalg.inv(T_snap)
        # T34 J1: the fit spheres as T32 made them are the self-filter copy (and the first-frame
        # TSDF free spheres at the capture pose); the query spheres are lifted off the support.
        fit_primitives = list(primitives)
        primitives, support_record = self._lift_off_support(primitives, T_now @ T_parent)
        # `attach_from_target` reads `id`, `points` and `bounding_geometry` only; the primary sphere
        # stands in for the old centroid + max-distance bounding sphere, the covers ride along.
        snapshot_of = _HeldTarget(id=int(getattr(geometry, "id", -1)), points=geometry.points,
                                  bounding_geometry=primitives[0])
        allowed = frozenset(str(link) for link in allowed_contact_links) | {frame_link}
        self._attached = attach_from_target(
            snapshot_of,
            robot_model=self.constraint_robot_model,
            robot_state=snap_state,
            parent_link=frame_link,
            allowed_contact_links=allowed,
            label=label,
            timestamp=time.time() if timestamp is None else float(timestamp),
            extra_primitives=primitives[1:],
        )

        def to_parent(p):
            return T_parent[:3, :3] @ np.asarray(p.center, np.float64) + T_parent[:3, 3]

        self._held_filter_local = [(to_parent(p), float(p.dimensions[0])) for p in fit_primitives]
        g_pts = np.asarray(geometry.points, np.float64).reshape(-1, 3)
        self._held_check_ref = {
            "frame": int(self.frame_index),
            "primary_fit_local": self._held_filter_local[0][0].copy(),
            "primary_fit_radius": float(self._held_filter_local[0][1]),
            "primary_fit_at_attach": (T_now[:3, :3] @ self._held_filter_local[0][0] + T_now[:3, 3]),
            "object_centroid_at_attach": (np.asarray(geometry.centroid, np.float64).reshape(3)
                                          if getattr(geometry, "centroid", None) is not None
                                          else g_pts.mean(axis=0)),
        }
        if support_record.get("applied"):
            held_record["fit_spheres"] = held_record.get("spheres")
            held_record["spheres"] = [
                {"centre_m": [round(float(v), 6) for v in p.center],
                 "radius_mm": round(float(p.dimensions[0]) * 1000.0, 3)} for p in primitives]
        self._held_record = {
            **held_record,
            "grasp_link": str(parent_link),
            "frame_link": frame_link,
            "snapshot_state": snap_source,
            # T34 J1: how the query spheres were lifted off the support (plane, band, per sphere).
            "support": support_record,
            "filter_spheres_parent_frame": [
                {"centre_m": [round(float(v), 6) for v in c], "radius_mm": round(r * 1000.0, 3)}
                for c, r in self._held_filter_local],
            "first_free_spheres": len(self._arm_first_free(fit_primitives, snap_state)),
            "hand_shift_since_capture_mm": round(
                float(np.linalg.norm(T_now[:3, 3] - T_snap[:3, 3])) * 1000.0, 2),
            "spheres_parent_frame": [
                {"centre_m": [round(float(v), 6) for v in (T_parent[:3, :3] @ p.center
                                                          + T_parent[:3, 3])],
                 "radius_mm": round(float(p.dimensions[0]) * 1000.0, 3)} for p in primitives],
        }
        return self._attached

    def _lift_off_support(self, primitives, T_rel) -> tuple[list, dict]:
        """T34 J1 — the held **query** spheres do not cross the support surface at attach.

        The object rests on the support when it is grasped, so a sphere reaching below the plane is
        fit error (T33: all 9 attaches put the primary sphere's bottom 1.8–10.1 mm under the table
        top; every E3b HOLD chunk was that held sphere against the table). Each sphere's centre is
        raised along the plane normal until its bottom clears the support at **both** instants the
        object was on it: at the geometry's capture pose (the snapshot, base frame of `primitives`)
        it clears the plane itself, and at the attach request's pose (`T_rel` = T_now · T_snap⁻¹;
        the hand may have pressed down since the capture — T33 E3b ep1800 r2: 10 mm) it clears the
        plane + the field's surface band, because from that pose on the optimizer reads the table
        through the field, which puts d = 0 up to ~8.8 mm above the plane (T33, 9 of 9 attaches).
        Radii are kept (`held_object.lift_off_support` says why lifting, not capping).

        The plane is the last processed frame's (`_last_support_surfaces`, the rule of
        `_support_plane_z` under the primary sphere's centre); the band is measured in the last
        field (`held_object.field_support_band`) unless `held_support_clearance_m` is set. No plane
        → nothing changes (recorded). Returns `(primitives, record)`.
        """
        from benchmark.ag3s.robot_models.held_object import (
            HELD_SUPPORT_BAND_FALLBACK_M,
            field_support_band,
            lift_off_support,
            support_plane_under,
        )
        from benchmark.ag3s.types import Primitive

        rec: dict[str, Any] = {"applied": False}
        if not self.held_support_lift:
            rec["reason"] = "switched_off"
            return primitives, rec
        primary = np.asarray(primitives[0].center, np.float64)
        plane = support_plane_under(self._last_support_surfaces, primary)
        if plane is None:
            rec["reason"] = "no_support_plane_under_the_object"
            return primitives, rec
        normal, offset, surface = plane
        rec["plane"] = {"normal": [round(float(v), 6) for v in normal],
                        "offset_m": round(float(offset), 6),
                        "surface_id": int(getattr(surface, "id", -1))}
        if self.held_support_clearance_m is not None:
            clearance, source = float(self.held_support_clearance_m), "given"
        else:
            band, band_rec = (None, {"reason": "no_field"})
            if self._last_esdf_field is not None:
                band, band_rec = field_support_band(self._last_esdf_field, normal, offset, primary)
            rec["band"] = band_rec
            if band is not None:
                clearance, source = band, "field_band"
            elif self._last_esdf_field is not None:
                clearance, source = HELD_SUPPORT_BAND_FALLBACK_M, "fallback"
            else:
                clearance, source = 0.0, "no_field"
        rec["clearance_source"] = source
        spheres = [(np.asarray(p.center, np.float64), float(p.dimensions[0])) for p in primitives]
        # Capture pose: the plane itself (geometry — the object was not below its support). Attach
        # request pose: the plane + the field's band (what the optimizer's held↔table row reads from
        # here on). Measured on T33: lifts 7.9–29.3 mm (`T34.impl.md` §1).
        lifted, lift_rec = lift_off_support(spheres, normal, offset, clearance=clearance,
                                            relative_poses=[T_rel], snapshot_clearance=0.0)
        rec.update(lift_rec)
        rec["applied"] = True
        out = [dataclasses.replace(p, center=np.asarray(c, np.float64))
               if isinstance(p, Primitive) else p for p, (c, _) in zip(primitives, lifted)]
        return out, rec

    def held_observation(self) -> Optional[dict[str, Any]]:
        """T34 J2 — the latest frame's view of the held object against the held spheres, or None.

        None when nothing is held, when the manipulated object was **not observed** in the latest
        frame (`last_seen_frame != frame`: while really held its pixels are erased by the
        self-filter, so it is normally unobserved — nothing to compare), or when that observation
        is not after the attach. Otherwise the measurement of
        `held_object.held_observation_check` at the state that frame was captured at, plus the
        held primary sphere's self-filter radius (fit + pad) and the frame. The decision is
        `trajopt.grasp_latch`'s (`LatchConfig.revoke_*`).
        """
        ref = self._held_check_ref
        manip = self._manipulated
        if self._attached is None or ref is None or manip is None:
            return None
        frame = int(manip.frame)
        if int(manip.last_seen_frame) != frame or frame < int(ref["frame"]):
            return None
        q = self._state_by_frame.get(frame)
        if q is None or self.constraint_robot_model is None:
            return None
        from benchmark.ag3s.constraints.attached import _link_pose_numeric
        from benchmark.ag3s.robot_models.held_object import held_observation_check

        T = _link_pose_numeric(self.constraint_robot_model, q, self._attached.parent_link)
        centre = T[:3, :3] @ ref["primary_fit_local"] + T[:3, 3]
        out = held_observation_check(
            observed_centroid=manip.centroid, held_centre=centre,
            held_centre_at_attach=ref["primary_fit_at_attach"],
            object_centroid_at_attach=ref["object_centroid_at_attach"])
        out.update({"frame": frame, "attach_frame": int(ref["frame"]),
                    "held_radius_m": float(ref["primary_fit_radius"]),
                    "n_points": (0 if getattr(manip, "geometry", None) is None
                                 else int(len(np.asarray(manip.geometry.points).reshape(-1, 3)))),
                    "state": str(manip.state)})
        return out

    def _arm_first_free(self, primitives, snap_state) -> list:
        """T32 H3: the spheres (base) freed in the TSDF on the first held frames — the held spheres
        at the geometry's capture pose (where the object was when last integrated) and the
        manipulated object's last pre-grasp fit + one fine voxel (its place on the table)."""
        extra = [(np.asarray(p.center, np.float64).copy(), float(p.dimensions[0]))
                 for p in primitives]                 # base frame at `snap_state`
        if self._last_object_fit is not None:
            c, r = self._last_object_fit
            extra.append((np.asarray(c, np.float64).copy(), float(r) + self._guard_pad()))
        self._held_first_free = extra
        self._held_first_free_frames = 2
        return extra

    def _snapshot_state(self, geometry, robot_state, geometry_state):
        """`(q, source)` to snapshot `geometry` with (T32 H2): explicit > the frame the manipulated
        object's geometry was last observed in > the attach request's `robot_state`."""
        if geometry_state is not None:
            return np.asarray(geometry_state, np.float64), "given"
        manip = self._manipulated
        if manip is not None and getattr(manip, "geometry", None) is geometry:
            q = self._state_by_frame.get(int(manip.last_seen_frame))
            if q is not None and len(q) == len(np.asarray(robot_state).reshape(-1)):
                return q, f"frame {int(manip.last_seen_frame)} (manipulated last seen)"
        return np.asarray(robot_state, np.float64), "attach_request"

    def _resolve_hold_link(self, parent_link: str, hold_link: Optional[str]) -> str:
        """The link the held object's snapshot rides (T32 H2). See `attach`."""
        reserved = tuple(getattr(self._builder, "attached_parent_links", ()) or ())
        if hold_link is not None:
            if reserved and str(hold_link) not in reserved:
                raise ValueError(
                    f"hold_link={hold_link!r} has no reserved attached slots (reserved: "
                    f"{list(reserved)}); pass it in attached_parent_links= when building AG3S")
            return str(hold_link)
        from benchmark.ag3s.robot_models.held_object import palm_link_for

        palm = palm_link_for(self.constraint_robot_model, parent_link)
        if palm is not None and (not reserved or palm in reserved):
            return palm
        return parent_link

    def _held_primitives(self, points) -> tuple[list, dict]:
        """Observed points (base) → `[primary, *covers]` sphere `Primitive`s (base) + record (H2)."""
        from benchmark.ag3s.constraints.attached import trim_outliers
        from benchmark.ag3s.robot_models.held_object import covering_spheres, fit_surface_sphere
        from benchmark.ag3s.types import Primitive, PrimitiveType

        pts = np.asarray(points, np.float64).reshape(-1, 3)
        keep = trim_outliers(pts, 0.020, 3)
        if int(keep.sum()) >= 4:
            pts = pts[keep]
        max_r = 0.5 * self._max_opening.value_m if self._max_opening.known else None
        fit = fit_surface_sphere(pts, max_radius=max_r)
        slots = int(getattr(self._builder, "max_attached", 0) or 0) or 4
        spheres, rec = covering_spheres(pts, fit, pad=self._guard_pad(), max_spheres=slots)
        from benchmark.ag3s.robot_models.held_object import HELD_ROLE

        prims = [Primitive(type=PrimitiveType.SPHERE, center=c, dimensions=np.full(3, float(r)),
                           semantic_role=HELD_ROLE) for c, r in spheres]
        rec["max_radius_mm"] = None if max_r is None else round(max_r * 1000.0, 3)
        return prims, rec

    @property
    def held_record(self) -> Optional[dict[str, Any]]:
        """How the held object's spheres were made at the last `attach()` (T32 H2), or None."""
        return None if self._attached is None or self._held_record is None \
            else dict(self._held_record)

    @staticmethod
    def _primitive_surface_points(geometry) -> Optional[np.ndarray]:
        """Six extreme points per bounding sphere of a points-less attachment (object frame →
        parent frame through `T_parent_object`), so its extent can be tested like a cluster's."""
        from benchmark.ag3s.stages.geometry import to_spheres

        out = []
        for primitive in geometry.primitives:
            for local, radius in to_spheres(primitive):
                c = np.asarray(local, np.float64).reshape(3)
                out.extend(c + float(radius) * d for d in np.vstack([np.eye(3), -np.eye(3)]))
        if not out:
            return None
        return np.asarray(out, np.float64)

    def _attached_in_base(self, geometry, robot_state, local_points) -> np.ndarray:
        """A prebuilt attachment's points in the base frame at `robot_state` (for the overlap)."""
        from benchmark.ag3s.constraints.attached import _link_pose_numeric

        try:
            T_base_parent = _link_pose_numeric(self.constraint_robot_model, robot_state,
                                               geometry.parent_link)
        except Exception:  # noqa: BLE001 — no pose: fall back to the parent frame (extent still holds)
            T_base_parent = np.eye(4)
        T = T_base_parent @ np.asarray(geometry.T_parent_object, np.float64)
        return local_points @ T[:3, :3].T + T[:3, 3]

    def detach(self) -> Optional[AttachedCollisionGeometry]:
        """Record an externally confirmed release. Returns what was let go, or None.

        The only way an attached object leaves. Grounding failure, occlusion and camera dropout all
        deliberately have no path here: the robot does not put an object down because it stopped
        being able to see it.
        """
        released, self._attached = self._attached, None
        self._held_record = None
        self._held_first_free = []
        self._held_first_free_frames = 0
        self._held_filter_local = None
        self._held_check_ref = None
        return released

    # ------------------------------------------------------------------------------------
    def process(self, **kwargs: Any) -> CollisionConstraintSet:
        """Run one frame. Returns the `CollisionConstraintSet` a TO consumes.

        Exactly one of `depth` or `pointcloud` is required. `attention_map` may be omitted, in which
        case grounding reports `NO_ATTENTION` and every piece of geometry is treated as a
        conservative collision candidate — which is the correct degradation, not an error.
        """
        return self._run(**kwargs)[0]

    def process_multi(
        self,
        observations: Sequence[CameraObservation],
        **kwargs: Any,
    ) -> CollisionConstraintSet:
        """Run one frame from several cameras, fused into a single base-frame scene.

        Each observation carries its own capture timestamp and its own `robot_state`, and is placed
        with the kinematics of that instant. There is no `robot_state` argument here on purpose: a
        single current `q` applied to three cameras is the bug this method exists to prevent.

        `robot_state` for the *constraints* — the configuration the TO plans from — is taken from the
        newest observation, since that is the freshest thing AG3S has been told about the robot.
        """
        return self.process_multi_debug(observations, **kwargs)[0]

    def process_multi_debug(
        self,
        observations: Sequence[CameraObservation],
        **kwargs: Any,
    ) -> tuple[CollisionConstraintSet, dict[str, Any]]:
        """`process_multi`, plus the fusion result and the intermediate clouds."""
        observations = list(observations)
        kwargs.setdefault(
            "robot_state",
            max(observations, key=lambda o: o.timestamp).robot_state if observations else None,
        )
        kwargs.setdefault(
            "timestamp", max((o.timestamp for o in observations), default=None) or None
        )
        return self._run(observations=observations, **kwargs)

    def process_debug(self, **kwargs: Any) -> tuple[CollisionConstraintSet, dict[str, Any]]:
        """`process`, plus the intermediate artifacts.

        Exists for `visualization` and for tests that need to see a stage's input as well as its
        output. Kept separate so the production path never pays to retain clouds it will not use.
        """
        return self._run(**kwargs)

    def _run(
        self,
        *,
        attention_map: Any = None,
        depth: Optional[np.ndarray] = None,
        pointcloud: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[np.ndarray] = None,
        T_base_cam: Optional[np.ndarray] = None,
        robot_state: Optional[np.ndarray] = None,
        phase: Phase | str = Phase.TRANSIT,
        image_hw: Optional[tuple[int, int]] = None,
        uv: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        active_manipulators: Any = None,
        contact_context: Optional[ContactPolicyContext] = None,
        observations: Optional[Sequence[CameraObservation]] = None,
        destination_points: Optional[np.ndarray] = None,
        static_geometry: Optional[Sequence[Any]] = None,
        execution_path: Optional[np.ndarray] = None,
    ) -> tuple[CollisionConstraintSet, dict[str, Any]]:
        """The single implementation behind `process`, `process_debug` and `process_multi`.

        `execution_path` (T21): the robot states — same vector as `robot_state`, one row per step —
        the policy's reference chunk will pass through during the **executed** window (K steps).
        Only the fine-window placement reads it (`_build_esdf`); without it the window is placed
        from `robot_state` alone and `esdf.stats["window"]["hand_source"]` says `q_now`.

        With `observations`, stages 1, 2 and 4 are replaced by the multi-camera front end in
        `multiview`; stages 3, 5, 6 and 8 are byte-for-byte the same code operating on the fused
        cloud. That is the point of `FusedPointCloud.as_pointcloud()` — fusion is a change to where
        the points come from, not to what happens to them.
        """
        cfg = self.config
        phase = Phase.parse(phase)
        # Phase alone does not say which hand is grasping, and AG3S must not guess. With no
        # manipulator named, the context authorizes nobody and every link keeps full clearance.
        context = contact_context or ContactPolicyContext.make(phase, active_manipulators)
        ts = time.time() if timestamp is None else float(timestamp)
        profiler = self.profiler
        notes: list[str] = []
        # T19 target guard: the manipulated object's centroid *before* this frame (T20 — the object
        # being grasped, not whatever attention led with), frozen for the whole frame so the cloud
        # filter (stage 2) and the depth robot mask (stage 8, after grounding) use the same one.
        # Any state, `lost` included: the guard asserts nothing about the object, it only keeps
        # the fingers' margin from erasing it where it was last seen, which is how it can re-appear.
        # T21 (user ruling 2026-09-28): the radius is the configured one when non-zero, otherwise
        # this object's radius + one fine voxel — set per object, no number baked in.
        # T26: only a manipulated object that passed the exclusion gate last frame is guarded.
        self._frame_guard_radius = self._guard_radius_for(self._gated_manipulated)
        self._frame_guard_centre = (
            self._gated_manipulated.centroid.copy()
            if self._frame_guard_radius > 0.0 and self._gated_manipulated is not None else None)
        self._frame_filter_config = self._config_with_guard_radius(self._frame_guard_radius)
        self._mask_guard_px = []
        # T32 H1: while an object is held, its spheres are part of the robot for the self-filter.
        self._prepare_frame_filter()

        fusion = None
        if observations is not None:
            # 1 + 2 + 4, fused. Each camera is reconstructed and self-filtered at its own capture
            # instant, then merged on a base-frame voxel grid with provenance retained.
            with profiler.stage("scene_reconstruction"):
                fusion = fuse_observations(
                    observations,
                    # `cfg` with this frame's effective guard radius (T21) — identical otherwise.
                    self._frame_filter_config,
                    robot_model=self._frame_filter_model,
                    attention_adapter=self.attention_adapter,
                    now=timestamp,
                    self_filter_inflation=self._frame_filter_inflation,
                    self_filter_guard_centre=self._frame_guard_centre,
                )
            cloud = fusion.pointcloud
            recon_stats = self._fused_recon_stats(fusion)
            filter_enabled = cfg.pointcloud.self_filter and self.robot_model is not None
            filter_stats = {
                "n_in": fusion.metrics["n_points_before_fusion"] + fusion.metrics["n_self_filtered"],
                "n_removed": fusion.metrics["n_self_filtered"],
                "n_out": fusion.metrics["n_points_before_fusion"],
                "enabled": filter_enabled,
                **self._guard_stats(
                    filter_enabled,
                    sum(fusion.metrics.get("n_self_filter_guard_protected", {}).values())),
            }
            profiler.record("robot_self_filter", 0.0)  # timed inside the fusion stage
            notes.extend(fusion.notes)
        else:
            # 1. scene reconstruction -----------------------------------------------------
            with profiler.stage("scene_reconstruction"):
                cloud, recon_stats = reconstruct(
                    depth=depth,
                    pointcloud=pointcloud,
                    camera_intrinsics=camera_intrinsics,
                    T_base_cam=T_base_cam,
                    uv=uv,
                    config=cfg.pointcloud,
                    frame_id=cfg.frame_id,
                )
        validity = ConstraintValidity.VALID if fusion is None else fusion.validity
        if recon_stats["capped"]:
            notes.append(reason(
                "pointcloud_capped",
                f"point cloud capped at max_points={cfg.pointcloud.max_points} "
                f"(from {recon_stats['n_voxel']}); voxel grown to "
                f"{recon_stats['final_voxel_size'] * 1000:.1f} mm in "
                f"{recon_stats['n_growth_steps']} step(s)",
            ))
            # Coarser is not incomplete: growing the voxel keeps a representative in every occupied
            # cell, so nothing physical went unobserved. Index selection makes no such promise.
            validity = ConstraintValidity.worst(validity, ConstraintValidity.DEGRADED)
        if not recon_stats["covered"]:
            notes.append(
                "the point cloud could not be reduced to max_points while preserving spatial "
                "coverage; some geometry is unrepresented"
            )
            validity = ConstraintValidity.worst(validity, ConstraintValidity.INCOMPLETE)

        # 2. robot self-filter ------------------------------------------------------------
        raw_cloud = cloud
        if fusion is None:
            with profiler.stage("robot_self_filter"):
                cloud, filter_stats = filter_robot_points(
                    cloud, self._frame_filter_model, robot_state,
                    self._frame_filter_config.pointcloud,
                    inflation=self._frame_filter_inflation,
                    guard_centre=self._frame_guard_centre,
                )

        # 3. support surfaces -------------------------------------------------------------
        with profiler.stage("support_surface"):
            surfaces, support_mask = fit_support_surfaces(cloud, cfg.support_surface, timestamp=ts)
        self._last_support_surfaces = list(surfaces)   # T34 J1: the plane `attach()` lifts off

        # 4. attention lifting ------------------------------------------------------------
        with profiler.stage("attention_lifting"):
            if fusion is not None:
                # Already lifted per camera and aggregated by max over each fused point's
                # observations. Rebuilding it here from a single map would need a single camera.
                attention_cloud = AttentionPointCloud(cloud, fusion.attention, fusion.raw_attention)
            else:
                # `image_hw` 를 주지 않으면 `lift` 는 **걸러지고 남은 클라우드의 uv 최댓값**으로
                # 해상도를 추정한다. 클라우드는 깊이 유효성·`range_max`·로봇 자기 필터를 이미
                # 거친 뒤라 이미지 경계에 못 닿는 일이 흔하고, 그러면 attention 맵이 틀린 크기로
                # 펼쳐진다 (F9). 실측에서 최대 57 픽셀 — attention 셀 한 칸(40 px)이 넘는다 —
                # 어긋났고, run_0004 의 두 프레임에서 **target 물체 자체가 바뀌었다**.
                #
                # 그런데 그 참값은 여기 이미 있다. `depth` 가 바로 attention 맵이 대응하는 그
                # 이미지다. 걸러진 점들에서 되짚는 것보다 언제나 옳다.
                hw = image_hw
                if hw is None and depth is not None:
                    hw = tuple(int(x) for x in np.asarray(depth).shape[:2])
                attention_cloud = self._lift(
                    cloud, attention_map, hw, camera_intrinsics, T_base_cam
                )

        # 5. target grounding -------------------------------------------------------------
        # T26: a per-call destination is an injection (sticky until `reset`), and it must be known
        # *before* grounding — it is part of what makes a cluster inadmissible.
        if destination_points is not None and len(destination_points):
            self._target_confirm.inject_destination(destination_points)
        # T26 §4: no switch while a grasp is in progress or an object is attached.
        self._target_confirm.freeze(self._grasp_active or self._attached is not None)
        # SUBTASK-c: this request's subtask label (consumed here; None when the caller gave none).
        self._target_confirm.note_subtask(self._subtask_p)
        self._subtask_p = None
        # T32b S3: the hands at this frame's state — challengers do not count while the manipulated
        # object is hidden by the hand (`TargetConfirm.note_hand`, consumed by this grounding call).
        self._target_confirm.note_hand(self._hand_for_grounding(robot_state))
        with profiler.stage("target_grounding"):
            grounding = ground_target(
                attention_cloud,
                cfg.clustering,
                seed_percentile=cfg.attention.seed_percentile,
                seed_threshold=cfg.attention.seed_threshold,
                exclude_mask=support_mask,
                min_radius=cfg.geometry.min_radius,
                timestamp=ts,
                confirm=self._target_confirm,
            )
        if fusion is not None and grounding.target is not None:
            grounding = dataclasses.replace(
                grounding,
                target=self._with_camera_provenance(grounding.target, fusion, attention_cloud),
            )
        # T20: what the consumers below refer to. `geometry` is this frame's observation when
        # `visible`, the last one when `occluded`; `lost` (or nothing confirmed yet) means none.
        self._manipulated = manipulated = self._target_confirm.manipulated
        if manipulated is not None and robot_state is not None:
            # T32 H2: which q this frame's geometry was captured at (for `attach`'s snapshot).
            self._state_by_frame[int(manipulated.frame)] = np.asarray(robot_state, np.float64).copy()
            for old_frame in sorted(self._state_by_frame)[:-16]:
                del self._state_by_frame[old_frame]
        # T26 §3: **exclusion geometry only from an admissible manipulated object.** The gate
        # re-checks the evidence (narrowest extent ≤ max opening, not the destination); on failure
        # nothing is excluded, nothing is permitted, and `invariant_violation` is recorded (raised
        # in strict mode). `_target_ball`, the target-free layer, the fine-window object box,
        # `TARGET_LABEL`, `manipulated_link_margin` / `target_field_exclude` and the next frame's
        # guard all read `manipulated_geometry` from here.
        manipulated_geometry, invariant = self._exclusion_gate(manipulated)
        self._gated_manipulated = manipulated if manipulated_geometry is not None else None
        if invariant is not None:
            notes.append(reason("invariant_violation", invariant["detail"]))
            if grounding.target is not None:
                # The same object must not reach the primitive path's TARGET relaxation either.
                grounding = dataclasses.replace(grounding, target=None,
                                                status=GroundingStatus.NO_ADMISSIBLE)
        destination = self._target_confirm.destination
        destination_points = None if destination is None else destination.points
        if self._target_confirm.destinations.last_event == "registered":
            notes.append(
                f"destination registered from perception: id={destination.id}, "
                f"{len(destination.points)} point(s), extents "
                f"{[round(float(v) * 1000.0) for v in destination.extents_m]} mm (not graspable, "
                f"led {self._target_confirm.frames} frame(s) at score ≥ "
                f"{self._target_confirm.switch_min_score}); it stays in every field")
        decision = self._target_confirm.last
        if (decision is not None and decision.reason == "inadmissible_match"
                and manipulated is not None):
            adm = decision.admissibility
            notes.append(
                f"manipulated object id={manipulated.id}: the cluster at its place is inadmissible "
                f"({'?' if adm is None else adm.reason}; narrowest "
                f"{'?' if adm is None or adm.narrowest_extent_m is None else round(adm.narrowest_extent_m * 1000.0, 1)}"
                " mm) — merged with the fingers or the destination; the last admissible geometry "
                "stands in (occluded)")
        if manipulated is not None and manipulated.state == LOST:
            # The reason code T20 asks for, distinct from `no_seed`: *which* object and for how long.
            notes.append(
                f"lost(id={manipulated.id}, age={manipulated.age_frames}): the manipulated object "
                f"has been unobserved for {manipulated.age_frames} frame(s) "
                f"(> clustering.target_lost_frames={self._target_confirm.lost_frames}); "
                "no challenger is named in its place"
            )
        if self._target_confirm.subtask_released is not None:
            notes.append(
                f"subtask gate: confirmed 'home' after PLACED — manipulated id="
                f"{self._target_confirm.subtask_released} released (no exclusion from it on)")
        if self._target_confirm.subtask_blocked is not None:
            notes.append(
                f"subtask gate: confirmed label {self._target_confirm.subtask_label!r} — "
                f"{self._target_confirm.subtask_blocked} blocked (no new manipulated object while "
                "placing / homing; the current one is kept)")
        if grounding.target is None:
            notes.append(
                f"no target ({grounding.status.value}); all geometry held at full clearance"
                + (f" — none of {len(grounding.clusters)} cluster(s) is admissible (graspable at "
                   f"max opening {self._max_opening.record()['value_mm']} mm and off the "
                   "destination)" if grounding.status is GroundingStatus.NO_ADMISSIBLE
                   and invariant is None else "")
            )
            if manipulated_geometry is not None:
                notes.append(
                    f"manipulated object id={manipulated.id} {manipulated.state} "
                    f"({manipulated.age_frames} frame(s) unobserved): contact permission and "
                    "target exclusion keep its last observed geometry"
                )
        else:
            decision = self._target_confirm.last
            if decision is not None and decision.mode in ("hold", "switch", "occluded"):
                # Said out loud only when the hysteresis did something, because a frame where it
                # overruled the highest-scoring cluster is otherwise indistinguishable from one
                # where it agreed. Not a `reason(...)`: nothing is degraded — the target is named
                # with full geometry either way.
                notes.append(
                    f"target {decision.mode} (rank {decision.rank}, challenger at "
                    f"{decision.streak}/{decision.frames} frames"
                    + (f"; id={decision.manipulated_id} unobserved {decision.age_frames} "
                       "frame(s), last observed geometry stands in"
                       if decision.mode == "occluded" else "")
                    + ")"
                )

        # 6 + 7. collision candidates and primitive fitting -------------------------------
        # Skipped entirely when the field is the only thing the optimizer will read. See
        # `EsdfConfig.emit_candidates` for why that is the default rather than an optimisation.
        primitive_path = (cfg.collision_backend != "esdf") or cfg.esdf.emit_candidates
        if not primitive_path:
            candidates, candidate_stats = [], {}
            notes.append(
                "collision_backend=esdf: the primitive candidate set was not built; the collision "
                "constraint comes from the ESDF and the target is named by grounding "
                "(esdf.emit_candidates turns it back on)"
            )
        if primitive_path:
              with profiler.stage("collision_candidates"):
                candidates, candidate_stats = generate_candidates(
                    cloud,
                    phase=phase,
                    target=grounding.target,
                    support_surfaces=surfaces,
                    support_mask=support_mask,
                    config=cfg.collision_candidate,
                    geometry_config=cfg.geometry,
                    contact_config=cfg.contact,
                    tracker=self.tracker,
                    timestamp=ts,
                )
        if primitive_path:
            profiler.record("primitive_fitting", candidate_stats.get("primitive_fit_ms", 0.0))
        if candidate_stats.get("n_overflow"):
            notes.append(reason(
                "candidate_overflow",
                f"{candidate_stats['n_overflow_points']} point(s) beyond "
                f"max_clusters={cfg.collision_candidate.max_clusters} / "
                f"unknown_min_points={cfg.collision_candidate.unknown_min_points} were folded into "
                f"{candidate_stats['n_overflow']} conservative aggregate(s)",
            ))
            validity = ConstraintValidity.worst(validity, ConstraintValidity.DEGRADED)
        if candidate_stats.get("n_unassigned_points"):
            # The only path that actually loses points, and only under `overflow_groups: 0`.
            notes.append(
                f"{candidate_stats['n_unassigned_points']} point(s) were discarded: "
                "collision_candidate.overflow_groups is 0, so there was no aggregate to hold them"
            )
            validity = ConstraintValidity.worst(validity, ConstraintValidity.INCOMPLETE)

        # 8. constraint generation --------------------------------------------------------
        with profiler.stage("constraint_generation"):
            # 8b. ESDF backend ------------------------------------------------------------
            # Built here, before the constraint set is assembled, because this is the only scope
            # that still has the raw depth. It runs *after* candidate generation on purpose:
            # whether the target is carved out of the field follows the same contact rule the
            # clearance policy uses, and that rule needs the grounded target to already exist.
            esdf_field = None
            if cfg.collision_backend in ("esdf", "both"):
                cameras = self._depth_cameras_from(
                    observations, depth, camera_intrinsics, T_base_cam, robot_state)
                with profiler.stage("esdf"):
                    esdf_field, esdf_notes = self._build_esdf(
                        # T20: the manipulated object's geometry (last observed when occluded) —
                        # fine-window centre, target-free layer, `_target_ball`, `TARGET_LABEL`.
                        cameras, manipulated_geometry,
                        # 지원면은 **테이블에 구멍을 내지 않기 위해** 필요하다 (T13). AG3S 가
                        # 이미 뽑아 둔 평면을 쓴다 — 여기서 z 를 추측하지 않는다.
                        support_surfaces=surfaces,
                        support_points=(cloud.points[support_mask]
                                        if cfg.esdf.exclude_support_surfaces
                                        and support_mask is not None else None),
                        destination_points=destination_points,
                        static_geometry=static_geometry,
                        robot_state=robot_state,
                        # T21: the fine window follows the authorized links' swept volume.
                        contact_context=context,
                        execution_path=execution_path,
                        # age 의 기준은 **적분한 관측 중 가장 최신의 촬영 시각**이다.
                        # `q_now` 를 고르는 규칙(`safe_policy._scene_fn`)과 같게 둔다 —
                        # 둘이 어긋나면 "이 자세에서 이 필드" 라는 짝이 깨진다.
                        observed_at=(max(float(o.timestamp) for o in observations)
                                     if observations else ts),
                        frame_id=self.config.frame_id,
                        frame_index=self.frame_index)
                notes.extend(esdf_notes)
                self._last_esdf_field = esdf_field     # T34 J1: the field band `attach()` measures
                if esdf_field is not None:
                    # G2 (`docs/AG3S_REVIEW_LOG.md` Step 4): an unobserved field used to produce a
                    # *note* and nothing else, so a frame whose collision field was entirely empty
                    # still reported `validity: valid` / `geometry_certified: True` — and in
                    # `collision_backend: esdf` the field is the only obstacle source. This is
                    # verbatim the failure `ConstraintValidity`'s own docstring names: "an empty
                    # VALID set means AG3S looked and there is nothing there, while INCOMPLETE means
                    # AG3S cannot tell you what is there. Rendering the second as the first is how a
                    # robot drives into an unmodelled wall."
                    stats = esdf_field.stats
                    # block-sparse backend 에서 "아무것도 안 봤다" 의 직접 신호는 할당된
                    # 블록이 0 개인 것이다. dense 의 `n_free`/`n_occupied` 는 그쪽에 없다.
                    nothing_observed = int(stats.get("n_active_blocks", -1)) == 0
                    if nothing_observed or (int(stats.get("n_free", 0)) == 0
                                            and int(stats.get("n_occupied", 0)) == 0):
                        # Not one voxel was observed either way: the integration produced no
                        # information at all. A genuinely empty workspace still yields FREE voxels
                        # along the rays that passed through it, so this only fires on failure —
                        # wrong depth units, wrong extrinsics, a bounds box the cameras cannot see.
                        notes.append(
                            "the ESDF integrated no observations at all (every voxel unknown); the "
                            "collision field is empty and cannot be certified — check the depth "
                            "units (pointcloud.depth_scale), the extrinsics and esdf.bounds_*"
                        )
                        validity = ConstraintValidity.worst(
                            validity, ConstraintValidity.INCOMPLETE)
                    elif esdf_field.unknown_fraction is None:
                        # **"모른다" 를 0 으로 읽지 않는다.** block-sparse TSDF 는 "보고 지나간
                        # 자유 공간" 과 "한 번도 안 본 곳" 을 구분하지 못하므로 legacy 의
                        # per-voxel UNKNOWN 비율에 대응하는 값이 없다 (F16 — 블록-스파스가
                        # 만드는 새 안전 질문). 지금은 보수적으로 `DEGRADED` 로 두고 그 이유를
                        # 노트에 싣는다. **이 정책은 미정이다** — 매 프레임 인증이 내려가면
                        # 안전 게이트가 통째로 막히므로, 대안(절두체 기반 관측 부피 측정 /
                        # 질의점 단위 판정)을 재고 정한다.
                        # **노트만 남기고 인증은 내리지 않는다** (2026-09-22 판정). 전역
                        # 비율은 block-sparse 에 대응하는 값이 없고, 있어도 이 씬에서는 격자의
                        # 대부분이 팔이 닿지 못하는 허공이라 판정 기준이 되기에 부적절하다.
                        # 인증은 아래 `_esdf_coverage` 가 **로봇 구와 쥔 물체가 지나가는 곳**의
                        # 관측 여부로 한다.
                        fo = stats.get("frustum_observation") or {}
                        seen = fo.get("observed_fraction")
                        notes.append(
                            "this backend has no dense per-voxel UNKNOWN count (a block-sparse "
                            "TSDF leaves 'seen through, free' and 'never looked at' both "
                            f"unallocated); {int(stats.get('n_active_blocks', -1))} block(s) "
                            "allocated"
                            + ("" if seen is None else
                               f", frustum estimate {seen:.1%} of the workspace box observed")
                            + ". Certification comes from the swept-volume observation check "
                              "instead of a global fraction"
                        )
                    elif esdf_field.unknown_fraction >= cfg.esdf.unknown_report_threshold:
                        notes.append(reason(
                            "esdf_unknown_fraction",
                            f"{esdf_field.unknown_fraction:.1%} of the ESDF volume was never "
                            f"observed (threshold "
                            f"{cfg.esdf.unknown_report_threshold:.1%}) and is treated as "
                            f"{cfg.esdf.unknown_policy} by esdf.unknown_policy",
                        ))
                        # "represented, but ... partially observed" — the enum's own words for
                        # DEGRADED. Still usable; no longer certified.
                        validity = ConstraintValidity.worst(
                            validity, ConstraintValidity.DEGRADED)
                    # G3/G4 — the two *partial* failures G2 cannot see, because a field that is
                    # wrong is not the same as a field that is empty.
                    cover_notes, cover_validity = self._esdf_coverage(esdf_field, robot_state)
                    notes.extend(cover_notes)
                    validity = ConstraintValidity.worst(validity, cover_validity)

            constraint_set = self._constraints(
                esdf=esdf_field,
                destination_points=destination_points,
                # Planes still live in the spec, so it is built whenever a support surface survived.
                # In pure-field mode there is neither a candidate nor a plane to pack.
                build_spec=primitive_path or bool(surfaces),
                candidates=candidates,
                surfaces=surfaces,
                grounding=grounding,
                manipulated_geometry=manipulated_geometry,
                robot_state=robot_state,
                phase=phase,
                timestamp=ts,
                notes=notes,
                context=context,
                validity=validity,
                metrics={
                    **(fusion.metrics if fusion is not None else {}),
                    "n_points_raw": recon_stats["n_raw"],
                    "n_points_voxel": recon_stats["n_voxel"],
                    "n_points_final": recon_stats["n_final"],
                    "final_voxel_size_m": recon_stats["final_voxel_size"],
                    **({} if esdf_field is None else
                       {"esdf": {k: v for k, v in esdf_field.stats.items()
                                 if k != "per_camera"}}),
                    "n_points_self_filtered": filter_stats["n_removed"],
                    "self_filter_enabled": filter_stats["enabled"],
                    "self_filter": self._self_filter_frame_stats(filter_stats),
                    "n_support_points": int(support_mask.sum()),
                    "target_grounding_status": grounding.status.value,
                    # T20 §3: `{id, state, age_frames, centroid, radius, n_points,
                    # last_seen_frame, switched_from, frame}` or None. The only way to tell,
                    # after the fact, whether contact permission was alive on an occluded frame.
                    "manipulated": None if manipulated is None else manipulated.record(),
                    # T26: this frame's admissibility (per cluster, score order), the destination
                    # and the exclusion-gate verdict. `exclusion` is added after the set is built.
                    "admissibility": self._admissibility_frame_record(),
                    "destination": (None if destination is None else {
                        **destination.record(),
                        "event": self._target_confirm.destinations.last_event}),
                    "gripper_max_opening": {"value_mm": self._max_opening.record()["value_mm"],
                                            "source": self._max_opening.source},
                    # T29: the finger joints both models used this frame (self-filter and
                    # constraint FK) and their provenance (`set_finger_joints`).
                    "finger_joints": self._finger_frame_record(),
                    # SUBTASK-c: the subtask label this frame and what the gate did with it.
                    "subtask": self._subtask_frame_record(),
                    # T32 H2: the held object's spheres (fit, pad, frame link) or None.
                    "held_object": self.held_record,
                    "invariant_violation": invariant,
                    "target_confidence": (
                        0.0 if grounding.target is None else float(grounding.target.confidence)
                    ),
                    "n_object": candidate_stats.get("n_object", 0),
                    "n_unknown": candidate_stats.get("n_unknown", 0),
                    "n_overflow": candidate_stats.get("n_overflow", 0),
                    "n_overflow_points": candidate_stats.get("n_overflow_points", 0),
                    "n_unassigned_points": candidate_stats.get("n_unassigned_points", 0),
                    "overflow_containment_rate": candidate_stats.get(
                        "overflow_containment_rate", 1.0
                    ),
                    "overflow_excess_volume_ratio": candidate_stats.get(
                        "overflow_excess_volume_ratio", 0.0
                    ),
                    **self._clearance_metrics(context, grounding.target is not None),
                },
            )

        # T26 §4: which mechanism keeps the manipulated object out of the gripper's field this frame.
        constraint_set.metrics["exclusion"] = self._exclusion_record(
            constraint_set, manipulated_geometry, context, robot_state)
        profiler.end_frame()
        self.frame_index += 1
        object.__setattr__(constraint_set, "profile", profiler.last_frame_ms())
        object.__setattr__(
            constraint_set,
            "notes",
            constraint_set.notes + self._stat_notes(recon_stats, filter_stats, candidate_stats),
        )
        debug = {
            "raw_cloud": raw_cloud,
            "filtered_cloud": cloud,
            "attention_cloud": attention_cloud,
            "seed_indices": grounding.seed_indices,
            "support_mask": support_mask,
            "grounding": grounding,
            "reconstruction_stats": recon_stats,
            "self_filter_stats": filter_stats,
            "candidate_stats": candidate_stats,
            "fusion": fusion,
        }
        return constraint_set, debug

    # --- T26: the exclusion gate and its record ---------------------------------------------
    def _exclusion_gate(self, manipulated: Optional[ManipulatedIdentity]):
        """`(geometry | None, invariant_violation record | None)` — exclusion only from an
        admissible manipulated object (T26 §3).

        Not a limit: an inadmissible object should never have become the manipulated object
        (`TargetConfirm` admits only graspable, off-destination clusters). Arriving here without
        the evidence is a bug, so it is refused **and** recorded (raised in strict mode). Checked:
        the identity carries `admissibility`; it was `admissible`; its narrowest extent is at or
        under *this* AG3S's max opening; and its geometry does not overlap the destination as it
        stands **now** (the destination may have been registered after the object was adopted).
        """
        if manipulated is None or not manipulated.usable:
            return None, None
        adm = manipulated.admissibility
        cfg = self.config.clustering
        opening = self._max_opening.value_m if self._max_opening.known else None
        problem = None
        overlap_now = None
        if adm is None:
            problem = "no admissibility evidence on the manipulated object"
        elif not adm.admissible:
            problem = f"the manipulated object's last observation was inadmissible ({adm.reason})"
        elif opening is None or adm.narrowest_extent_m is None:
            problem = "the gripper's max opening or the object's extent is unknown"
        elif adm.narrowest_extent_m > opening + 1e-12:
            problem = (f"narrowest extent {adm.narrowest_extent_m * 1000.0:.1f} mm exceeds the max "
                       f"opening {opening * 1000.0:.1f} mm")
        else:
            overlap_now = destination_overlap(manipulated.points, self.destination,
                                              self._overlap_distance())
            if (self.destination is not None
                    and overlap_now >= float(cfg.destination_overlap_fraction)):
                problem = (f"it overlaps the destination id={self.destination.id} "
                           f"({overlap_now:.2f} of its points)")
        if problem is None:
            return manipulated.geometry, None
        record = {
            "detail": (f"exclusion geometry refused for manipulated id={manipulated.id}: {problem}; "
                       "no ball, no target-free layer, no guard, no contact permission this frame"),
            "manipulated_id": int(manipulated.id),
            "admissibility": None if adm is None else adm.record(),
            "max_opening_mm": None if opening is None else round(opening * 1000.0, 2),
            "destination_overlap_now": overlap_now,
            "destination_id": None if self.destination is None else int(self.destination.id),
        }
        if self.strict_invariants:
            raise ExclusionInvariantViolation(record["detail"], record)
        logging.getLogger(__name__).error("invariant_violation: %s", record["detail"])
        return None, record

    def _subtask_frame_record(self) -> dict[str, Any]:
        """`metrics["subtask"]` — `{p, argmax, label, gate, blocked, would_block, streak, frames}`.

        `label` is the confirmed (debounced) label; `blocked` what the gate blocked this frame
        (`first` | `switch` | None, always None with the gate off); `would_block` the same question
        answered with the gate on — the shadow record for a gate-off run.
        """
        tc = self._target_confirm
        rec = tc.subtask.record()
        return {"p": rec["p"], "argmax": rec["argmax"], "label": rec["label"],
                "gate": bool(tc.subtask_gate), "blocked": tc.subtask_blocked,
                "would_block": tc.subtask_would_block, "streak": rec["streak"],
                "frames": rec["frames"],
                # B3: released the placed manipulated object this frame (and which id).
                "released": tc.subtask_released is not None,
                "released_id": tc.subtask_released, "placed_seen": bool(tc.placed_seen)}

    def _admissibility_frame_record(self) -> Optional[dict]:
        """This frame's clusters' admissibility, score order — the evidence behind `first`/`switch`."""
        adm = self._target_confirm.last_admissibility
        decision = self._target_confirm.last
        if adm is None:
            return None
        return {
            "max_opening_mm": self._max_opening.record()["value_mm"],
            "n_clusters": len(adm),
            "n_admissible": sum(1 for a in adm if a.admissible),
            "clusters": [a.record() for a in adm],
            "decision": None if decision is None else {
                "mode": decision.mode, "rank": decision.rank, "reason": decision.reason,
                "frozen": decision.frozen},
            "destination_streak": self._target_confirm.destinations.streak,
        }

    def _exclusion_record(self, constraint_set, manipulated_geometry, context, robot_state):
        """`metrics["exclusion"]` — `{active, source, mechanism, ...}` (T26 §4).

        `source`: `attached` once an object is attached (it is carved out of the field and rides the
        hand as robot geometry), `manipulated` before that (target-free layer and/or contact margin
        on the authorized links), `none` when neither holds. `active` says the mechanism actually
        applies to at least one authorized link this frame. The grasp is continuous iff no frame
        between the closing command and the release reads `none`.
        """
        authorized = sorted(self._authorized_links(context))
        cs_manip = getattr(constraint_set, "manipulated", None)
        mechanism: list[str] = []
        if self._attached is not None:
            source = "attached"
            if cs_manip is not None and getattr(cs_manip, "source", None) == "attached":
                mechanism.append("attached_query")
            esdf = getattr(constraint_set, "esdf", None)
            if (robot_state is not None and self.constraint_robot_model is not None
                    and esdf is not None):
                # T32 H3: `held_free` = the held volume is set free in the TSDF (cuRobo);
                # `attached_carve` = its seeds are excluded (legacy / points-only attachments).
                freed = ((getattr(esdf, "stats", None) or {}).get("held_free") or {})
                mechanism.append("held_free" if freed.get("n_points") else "attached_carve")
            if int(getattr(self._frame_filter_model, "n_held", 0) or 0):
                # T32 H1: its pixels are removed by the self-filter like the robot's.
                mechanism.append("self_filter")
            active = bool(mechanism)
        elif manipulated_geometry is not None:
            source = "manipulated"
            if getattr(constraint_set, "target_field_exclude", None) is not None:
                mechanism.append("target_free_layer")
            if getattr(constraint_set, "manipulated_link_margin", None) is not None:
                mechanism.append("contact_margin")
            active = bool(mechanism) and bool(authorized)
        else:
            source, active = "none", False
        manip = self._manipulated
        return {
            "active": bool(active),
            "source": source,
            "mechanism": mechanism,
            "authorized_links": authorized,
            "manipulated_id": None if manip is None else int(manip.id),
            "manipulated_state": None if manip is None else str(manip.state),
            "attached_label": None if self._attached is None else self._attached.label,
            "grasp_active": bool(self._grasp_active),
        }

    # --- helpers, kept thin on purpose ---------------------------------------------------
    def _lift(self, cloud, attention_map, image_hw, camera_intrinsics, T_base_cam):
        if attention_map is None or cloud.is_empty:
            # A frame with no attention still has geometry, and that geometry still has to be
            # avoided. Zero attention makes grounding report NO_ATTENTION and stage 6 fall back to
            # treating everything conservatively.
            return AttentionPointCloud(cloud, np.zeros(len(cloud), np.float32))
        adapter = self.attention_adapter or make_adapter(attention_map, self.config.attention)
        return lift(
            cloud,
            attention_map,
            self.config.attention,
            adapter=adapter,
            image_hw=image_hw,
            camera_intrinsics=camera_intrinsics,
            T_base_cam=T_base_cam,
        )

    def _constraints(
        self, *, candidates, surfaces, grounding, robot_state, phase, timestamp, notes, context,
        validity=ConstraintValidity.VALID, metrics=None, esdf=None, build_spec=True,
        destination_points=None, manipulated_geometry=None,
    ):
        # 목적지 마진은 `ClearancePolicy` 에서 나온다 — 마진이 나오는 곳은 하나여야 하고,
        # 소비 쪽(trajopt)은 그 값을 다시 계산하지 않고 읽기만 한다.
        has_destination = destination_points is not None and len(destination_points) > 0
        destination_label = DESTINATION_LABEL if has_destination else None
        destination_margin = None
        if has_destination:
            policy = getattr(self._builder, "clearance_policy", None)
            if policy is None:
                policy = ClearancePolicy.from_config(
                    self.config.contact, self.config.geometry, self.config.support_surface)
            destination_margin = float(policy.full_margin(SourceType.DESTINATION))
        if self._builder is None:
            return CollisionConstraintSet(
                constraints=None,
                candidates=list(candidates),
                target=grounding.target,
                support_surfaces=list(surfaces),
                robot_state=np.zeros(0) if robot_state is None else robot_state,
                phase=phase,
                timestamp=timestamp,
                frame_id=self.config.frame_id,
                status=(
                    PipelineStatus.NO_GEOMETRY
                    if not candidates and not surfaces
                    else PipelineStatus.DEGRADED
                ),
                grounding_status=grounding.status,
                frame_index=self.frame_index,
                # `build_constraint_set` 을 거치지 않는 유일한 경로이므로 마지막 관문도
                # 여기서 한 번 더 지난다 — 관문이 한 곳뿐이면 이 분기가 그 밖에 남는다.
                notes=ensure_reason(
                    notes + [reason(
                        "no_robot_model",
                        f"no robot model injected; {len(list(candidates))} candidate(s) and "
                        f"{len(list(surfaces))} support surface(s) were found but no constraints "
                        "were generated against them")],
                    degraded=True),
                # There is geometry but nothing to write constraints against. That is not a certified
                # scene, and calling it VALID would let a caller read "no constraints" as "clear".
                validity=ConstraintValidity.worst(validity, ConstraintValidity.DEGRADED),
                contact_context=context,
                metrics=dict(metrics or {}),
                attached=self._attached,
                esdf=esdf,
                destination_label=destination_label,
                destination_margin=destination_margin,
            )
        return build_constraint_set(
            self._builder,
            candidates=candidates,
            support_surfaces=surfaces,
            target=grounding.target,
            robot_state=np.zeros(self._builder.nq) if robot_state is None else robot_state,
            phase=phase,
            grounding_status=grounding.status,
            frame_id=self.config.frame_id,
            frame_index=self.frame_index,
            timestamp=timestamp,
            notes=notes,
            contact_context=context,
            validity=validity,
            metrics=metrics,
            attached=self._attached,
            esdf=esdf,
            build_spec=build_spec,
            destination_label=destination_label,
            destination_margin=destination_margin,
            manipulated_geometry=manipulated_geometry,
        )

    def _esdf_coverage(self, field, robot_state):
        """`(notes, validity)` — does the field actually cover what it is asked about?

        G2 catches a field that observed *nothing*. These two catch a field that is populated but
        does not cover the question being asked, which reports as a perfectly healthy frame
        (`docs/AG3S_REVIEW_LOG.md` Step 4).

        **G3 — a camera that contributed nothing.** Measured on a healthy RB-Y1 frame every camera
        writes 13k-150k voxels, so `n_updated == 0` is not a quiet frame, it is a broken one: wrong
        extrinsics, a dead driver, depth entirely outside `esdf.depth_min/max`. This check only
        became possible once E5 was fixed — before that all three cameras collapsed onto the key
        `"camera"` and overwrote each other, so a dead camera was not merely unreported but
        unobservable.

        **G4 — a robot sphere outside the grid.** `EsdfField.distance` answers `outside_distance`
        for any query beyond the grid, so a sphere out there reads as free space no matter what is
        actually next to it (E4). AG3S knows the constraint model and this frame's `q`, so it can
        say so *before* the optimizer plans, which is worth more than the query-time counter
        `EsdfField.outside_query_fraction` that only trajopt would ever see. Measured over the whole
        `run_0004` rollout: the `arms` constraint model (120 spheres, the default) never leaves the
        default grid, so this does not fire on the healthy path; the whole-body model has 25 spheres
        permanently outside — `base`, `wheel_l`, `wheel_r`, `link_torso_0`, all below the grid floor
        — which is true, is the documented reason `--constraint-links arms` is the default, and is
        exactly the sort of thing that should be said out loud rather than left implicit.
        """
        notes: list[str] = []
        validity = ConstraintValidity.VALID

        per_camera = field.stats.get("per_camera", {}) or {}
        dead = sorted(n for n, s in per_camera.items() if int(s.get("n_updated", 0)) == 0)
        if dead and len(dead) < len(per_camera):
            notes.append(reason(
                "esdf_dead_camera",
                f"camera(s) {', '.join(dead)} contributed no voxels to the ESDF this frame "
                f"({len(dead)} of {len(per_camera)}); the field is built from the remaining "
                "view(s) only — check their extrinsics and depth range",
            ))
            validity = ConstraintValidity.worst(validity, ConstraintValidity.DEGRADED)

        model = self.constraint_robot_model
        if model is not None and robot_state is not None:
            centres, radii = model.sphere_centers_numeric(np.asarray(robot_state, np.float64))
            centres = np.asarray(centres, np.float64).reshape(-1, 3)
            radii = np.asarray(radii, np.float64).reshape(-1)
            # **관측 판정은 여기서 하지 않는다 — 지평이 여기 없다.**
            #
            # 2026-09-22 판정은 "로봇 구와 attached object 가 실제로 지나가는 swept volume 의
            # 관측 여부를 안전 게이트로 쓴다" 였다. 그 판정은 맞지만 **이 자리가 아니다.**
            # AG3S 는 현재 프레임의 `q` 만 알고, 현재 자세의 구 중심은 **정의상 로봇 안에**
            # 있다. 그것을 관측 판정하면 미지 장애물 부피가 아니라 self-occlusion 을 잰다.
            #
            # 실측 (seed 101, 구 120 개): 구 중심을 판정하면 63 개가 미관측으로 나오는데,
            # 그 구들은 첫 표면보다 **중앙 87.2 mm 뒤**에 있고 반지름은 중앙 26 mm 다 —
            # 자기 몸이나 다른 로봇 부위에 가려진 것이다 (관측된 구는 뒤 깊이 중앙 28.5 mm).
            # 마스킹한 depth 로 재면 113/120 까지 갔는데 그것은 더 심한 동어반복이었다
            # (로봇이 자기를 지웠으니 자기가 안 보인다).
            #
            # 그래서 필드는 `observation` probe 를 **들고 나가고**, 판정은 지평의 미래 스텝
            # 구 위치에서 한다 — 그 자리는 trajopt 이고, T4 의 합격 조건 "전체 계획 지평의
            # 로봇 구와 attached point 가 coarse coverage 안에 있다" 와 같은 곳이다.
            # **거친 계층의 범위를 본다.** 2계층 필드에서 `field.grid` 는 *가장 미세한* 계층
            # 이고 (그쪽이 E1 의 target 판정 허용오차에 맞는 선택이다), 미세 창은 관심 영역만
            # 덮으므로 팔 구 대부분이 당연히 그 밖에 있다. 그것을 G4(로봇 구가 격자 밖이면
            # 무조건 자유로 읽히는데 아무도 경고 안 한다)로 세면 **매 프레임 degraded** 가
            # 나고 안전 게이트가 통째로 막힌다 — F14(하네스가 만든 가짜 카메라 지연)와 같은
            # 모양의 오진이다. `CuroboEsdfField` 가 이 경로를 위해 `coverage_grid` 를 따로
            # 두고 있고, 여기가 그것을 읽는 자리다.
            cover = getattr(field, "coverage_grid", None) or field.grid
            lower, upper = cover.origin, cover.upper
            outside = np.any(
                (centres - radii[:, None] < lower) | (centres + radii[:, None] > upper), axis=1
            )
            if outside.any():
                names = getattr(model, "sphere_link_names", None)
                where = (f" ({', '.join(sorted(set(str(names[i]) for i in np.flatnonzero(outside))))})"
                         if names is not None and len(names) == outside.size else "")
                notes.append(reason(
                    "spheres_outside_grid",
                    f"{int(outside.sum())} of {outside.size} constraint sphere(s) lie outside the "
                    f"ESDF grid{where}; the field answers "
                    f"{field.outside_distance:+.2f} m for them regardless of what is there, so they "
                    "are not constrained by it — widen esdf.bounds_lower/upper or exclude those "
                    "links",
                ))
                validity = ConstraintValidity.worst(validity, ConstraintValidity.DEGRADED)
        return notes, validity

    def _robot_mask_for(self, depth, K, T_base_cam, robot_state):
        """`(H, W)` True where a depth pixel is the robot itself, or `None` when unknowable.

        The point cloud is already self-filtered by `robot_filter`; the ESDF integrates raw depth
        and so needs the same exclusion applied in image space. Without it the arm is written into
        the obstacle field and every frame reports the robot colliding with itself — measured at
        130 of 194 spheres in violation before this existed.

        The masked rays become **unobserved**, not free. What is behind the arm this frame is not
        something the camera saw.
        """
        from benchmark.ag3s.stages.reconstruction import backproject

        model = self._frame_filter_model
        if model is None or robot_state is None:
            return None
        cfg = self.config.pointcloud
        if not cfg.self_filter:
            return None
        cloud = backproject(depth, K, T_base_cam, cfg)
        if cloud.uv is None or len(cloud) == 0:
            return None
        # Same decision as the cloud filter (T19): the inflation object resolved at construction
        # and this frame's guard centroid, both through `self_filter_mask`. T32 H1: the model
        # carries the held spheres while attached (`_prepare_frame_filter`).
        q = np.asarray(robot_state, np.float64)
        inside, guard = self_filter_mask(
            cloud.points, model, q, self._frame_filter_config.pointcloud,
            inflation=self._frame_filter_inflation, guard_centre=self._frame_guard_centre)
        self._mask_guard_px.append(int(guard["n_guard_protected"]))
        held_fn = getattr(model, "held_spheres", None)
        if callable(held_fn):
            from benchmark.ag3s.stages.robot_filter import robot_sphere_mask

            hc, hr = held_fn(q)
            self._mask_held_px.append(int(robot_sphere_mask(cloud.points, hc, hr, 0.0).sum()))
        mask = np.zeros(np.asarray(depth).shape, bool)
        uv = cloud.uv[inside]
        mask[uv[:, 1], uv[:, 0]] = True
        return mask

    def _hand_for_grounding(self, robot_state):
        """`HandSpheres` for T32b S3 at `robot_state`, or None (S3 then suppresses nothing).

        **The constraint model first**, then the bare self-filter model. T32b calibrated the 30 mm
        reach on the recorded constraint spheres (`sphere_centres/radii`: palm + fingers, 6.2 / 3.3
        mm); the self-filter model's palm carries T29 cover capsules up to ~37 mm, which would make
        the hand ~30 mm "thicker" than the reach was measured with. Never the held-object wrapper
        (`_frame_filter_model`): a held sphere is not the hand.

        Finger joints: both models carry this frame's measured opening (`set_finger_joints`, T29 —
        `SafePolicy` sets them from the request's `state` before `process*`), and
        `sphere_centers_numeric` uses them. `robot_state` is the one the constraints are built at
        (the newest observation's, `process_multi_debug`).
        """
        if robot_state is None:
            return None
        for model in (self.constraint_robot_model, self.robot_model):
            hand = hand_spheres(model, robot_state)
            if hand is not None:
                return hand
        return None

    def _prepare_frame_filter(self) -> None:
        """This frame's self-filter model + inflation (T32 H1, cuRobo attach semantics).

        Nothing held (or the switch off, or no self-filter model): the model and the inflation
        resolved at construction — the pre-T32 arithmetic exactly. Held: the model wrapped with the
        held object's spheres (`attached_spheres`, parent frame; FK'd per capture `q` with the
        **constraint** model — the one `attach()` snapshotted with), and the inflation extended with
        **0** for them: the spheres cover the object by construction (fit + pad), so inflating them
        would only erase what is next to the object.
        """
        self._mask_held_px = []
        self._frame_filter_model = self.robot_model
        self._frame_filter_inflation = self._self_filter_inflation
        held = self._attached
        if held is None or self.robot_model is None or not self.self_filter_held_object:
            return
        from benchmark.ag3s.constraints.attached import attached_spheres
        from benchmark.ag3s.robot_models.held_object import (
            HeldSphereFilterModel,
            extend_inflation,
        )

        spheres = attached_spheres(held)
        if not spheres:
            return
        from benchmark.ag3s.robot_models.held_object import held_spheres_of

        if held_spheres_of(held) and self._held_filter_local:
            # T34 J1: the self-filter copy is the fit as T32 made it — the support lift is for
            # the query spheres only.
            spheres = [(np.asarray(c, np.float64).copy(), float(r))
                       for c, r in self._held_filter_local]
        if held_spheres_of(held) and float(self.held_filter_slip_m) > 0.0:
            # T32: the fitted held spheres + the measured in-hand slip (filter only).
            spheres = [(c, float(r) + float(self.held_filter_slip_m)) for c, r in spheres]
        model = HeldSphereFilterModel(self.robot_model, spheres, held.parent_link,
                                      pose_model=self.constraint_robot_model or self.robot_model)
        n_robot = int(model.n_spheres) - model.n_held
        self._frame_filter_model = model
        self._frame_filter_inflation = extend_inflation(
            self._self_filter_inflation, n_robot, model.n_held)

    def _guard_pad(self) -> float:
        """One **fine** voxel (m) — the same voxel `_target_ball` pads with (T21 guard rule)."""
        esdf = self.config.esdf
        return float(getattr(esdf, "fine_voxel_size", None) or esdf.voxel_size)

    def _guard_radius_for(self, manipulated: Optional[ManipulatedIdentity]) -> float:
        """The T19 guard radius for this frame (T21, user ruling 2026-09-28).

        `pointcloud.self_filter_target_guard_radius` when it is non-zero; when it is 0, the
        manipulated object's radius (`ManipulatedIdentity.radius`, bounding sphere of its last
        observation) + one fine voxel, so it is set **per object** rather than as a number.
        No manipulated object (or one with no geometry) → 0.0, no guard.
        """
        configured = float(self.config.pointcloud.self_filter_target_guard_radius)
        if configured > 0.0:
            return configured
        if manipulated is None or manipulated.geometry is None:
            return 0.0
        radius = float(manipulated.radius)
        if not np.isfinite(radius) or radius <= 0.0:
            return 0.0
        return radius + self._guard_pad()

    def _config_with_guard_radius(self, radius: float) -> AG3SConfig:
        """`self.config` with the effective guard radius, or `self.config` itself if unchanged."""
        pc = self.config.pointcloud
        if float(pc.self_filter_target_guard_radius) == float(radius):
            return self.config
        return dataclasses.replace(
            self.config,
            pointcloud=dataclasses.replace(pc, self_filter_target_guard_radius=float(radius)))

    def _guard_stats(self, enabled: bool, n_protected: int) -> dict[str, Any]:
        """The guard fields of `filter_robot_points`' stats, for the fused path that has no call."""
        radius = float(self._frame_guard_radius)
        active = bool(enabled and radius > 0.0 and self._frame_guard_centre is not None)
        return {
            "guard_enabled": radius > 0.0,
            "guard_active": active,
            "guard_radius": radius,
            "guard_centroid": (None if not active
                               else [float(x) for x in self._frame_guard_centre.reshape(3)]),
            "n_guard_protected": int(n_protected) if active else 0,
        }

    def _self_filter_frame_stats(self, filter_stats: dict[str, Any]) -> dict[str, Any]:
        """`metrics["self_filter"]`: the margin applied and what the target guard did this frame.

        `n_guard_protected_points` counts cloud points (per camera before fusion), and
        `n_guard_protected_depth_px` depth pixels kept out of the TSDF robot masks — two views of
        the same decision, reported separately because they are counted on different inputs.
        """
        cfg = self.config.pointcloud
        return {
            "inflation_default": float(cfg.self_filter_inflation),
            "inflation_by_link": dict(cfg.self_filter_inflation_by_link),
            "guard_enabled": bool(filter_stats.get("guard_enabled", False)),
            "guard_active": bool(filter_stats.get("guard_active", False)),
            "guard_radius": float(filter_stats.get("guard_radius", 0.0)),
            # T21: where the radius came from — "fixed" (config) or "manipulated" (radius + voxel).
            "guard_radius_rule": ("fixed" if cfg.self_filter_target_guard_radius > 0.0
                                  else "manipulated_radius_plus_voxel"),
            "guard_centroid": filter_stats.get("guard_centroid"),
            "n_guard_protected_points": int(filter_stats.get("n_guard_protected", 0)),
            "n_depth_masks": len(self._mask_guard_px),
            "n_guard_protected_depth_px": int(sum(self._mask_guard_px)),
            # T32 H1: the held object's spheres in this frame's self-filter (0 = nothing held) and
            # the depth pixels inside them (removed from the TSDF like the robot's).
            "n_held_spheres": int(getattr(self._frame_filter_model, "n_held", 0) or 0),
            "held_filter_slip_m": (float(self.held_filter_slip_m)
                                   if getattr(self._frame_filter_model, "n_held", 0) else None),
            "n_held_depth_px": int(sum(self._mask_held_px)),
        }

    def _depth_cameras_from(self, observations, depth, camera_intrinsics, T_base_cam,
                            robot_state=None):
        """`CameraDepth` list from whichever front end this frame used. Empty when depth is absent.

        A caller that supplied a bare point cloud has no depth image, and the projective TSDF needs
        one — it decides *free* space from what a ray passed through, which a point cloud cannot
        say. That case returns nothing and the pipeline notes it rather than silently building a
        field out of surface points alone, which would mark everything else unobserved.

        **`pointcloud.depth_scale` is applied here** (G1, `docs/AG3S_REVIEW_LOG.md` Step 4). The
        depth image's units are a property of the *input*, not of whoever consumes it, so both
        consumers must decode them the same way: `reconstruction.backproject` scales, and until this
        was fixed the TSDF did not. With the documented `depth_scale: 0.001` for uint16-millimetre
        depth — what a real sensor and `--record-depth` both deliver — the cloud came out in metres
        while the field integrated the raw millimetre numbers, every ray fell outside
        `esdf.depth_max`, and the collision field ended up **entirely empty while the frame still
        reported `status: ok`**. `_robot_mask_for` is deliberately given the *unscaled* depth: it
        goes through `backproject`, which applies the scale itself.
        """
        from benchmark.ag3s.fields.esdf import CameraDepth

        scale = float(self.config.pointcloud.depth_scale)
        out = []
        if observations:
            for obs in observations:
                d = getattr(obs, "depth", None)
                if d is None:
                    continue
                K = np.asarray(obs.camera_intrinsics, np.float64)
                # `self.robot_model` 를 반드시 넘겨야 한다. 손목 카메라는 `T_base_cam` 대신
                # `mount_link` + `T_link_cam` 으로 오고, 그 조합은 FK 없이는 풀리지 않는다.
                # 인자를 빼면 head/wrist 관측에서 ESDF 경로 전체가 예외로 죽는다.
                T = (np.asarray(obs.resolve_T_base_cam(self.robot_model), np.float64)
                     if hasattr(obs, "resolve_T_base_cam")
                     else np.asarray(obs.T_base_cam, np.float64))
                d = np.asarray(d, np.float64)
                mask = getattr(obs, "robot_mask", None)
                if mask is None:
                    # Each observation carries the `q` it was captured at, so the mask is built
                    # with *that* configuration — the same rule the fused cloud follows.
                    mask = self._robot_mask_for(d, K, T, getattr(obs, "robot_state", None))
                # E5 (`docs/AG3S_REVIEW_LOG.md` Step 2): `CameraObservation` names itself via
                # `camera_id`, not `camera`/`name` — those two never existed on it, so all three
                # real cameras fell back to the same literal `"camera"` and silently overwrote each
                # other in `EsdfField.stats["per_camera"]`. `camera_id` is tried first now; the old
                # fallbacks stay for any duck-typed observation that only has those.
                cam_id = getattr(obs, "camera_id", None)
                name = (cam_id.value if cam_id is not None
                        else str(getattr(obs, "camera", getattr(obs, "name", "camera"))))
                out.append(CameraDepth(
                    name=name, depth=d * scale, camera_intrinsics=K, T_base_cam=T,
                    robot_mask=mask))
        elif depth is not None and camera_intrinsics is not None and T_base_cam is not None:
            d = np.asarray(depth, np.float64)
            K = np.asarray(camera_intrinsics, np.float64)
            T = np.asarray(T_base_cam, np.float64)
            out.append(CameraDepth("camera", d * scale, K, T,
                                   robot_mask=self._robot_mask_for(d, K, T, robot_state)))
        return out

    def _build_esdf(self, depth_cameras, target, *, support_surfaces=(), support_points=None,
                    destination_points=None, static_geometry=None, robot_state=None,
                    observed_at=None, frame_id="", frame_index=-1, contact_context=None,
                    execution_path=None):
        """Integrate this frame into the ESDF and return `(field, notes)`.

        `target` is the **manipulated object's** geometry since T20 (`AG3S.manipulated`; the last
        observed one while it is `occluded`, None when `lost`), not the frame's attention target.
        It decides what the target-free layer removes (`_target_ball`) — and only that.

        **Where** the fine / target-free window sits is decided by the hand (T21, guide §5.1): the
        contact-authorized links' spheres at `robot_state` and along `execution_path`, padded by
        radius + margin + interpolation voxels, unioned with the manipulated object's box when that
        fits (`_fine_window_request` → `curobo_builder.place_fine_window`). The result, and which
        tier answered the authorized links' query points, is in `field.stats["window"]`.

        The builder is kept on the instance rather than rebuilt: that is the only way the local
        update means anything, since a fresh grid has nothing to be incremental against.
        """
        cfg = self.config.esdf
        if not depth_cameras:
            return None, ["esdf backend requested but no camera depth was supplied; "
                          "the field was not built and the primitive candidates stand alone"]
        if self._esdf_builder is None:
            # **필드 구현만 갈아탄다.** 아래의 정책은 backend 와 무관하게 그대로다 — E1 이
            # 금지한 target carving, 목적지 라벨(F18), 쥔 물체(A2), 해석적 기하(N2) 의 결정은
            # 여기서 내려지고 builder 는 그것을 받는 쪽이다.
            #
            # `curobo` 일 때 `EsdfBuilder` 를 **만들지 않는다.** 만들면 legacy 가 조용히
            # 도는 것이고, `AG3S_TOTAL_TEST_Prompt.md` 의 T0 은 그것을 즉시 실패 조건으로
            # 둔다. 조용히 흐르는 것을 막으려면 갈림길이 여기 하나여야 한다.
            if cfg.backend == "curobo":
                from benchmark.ag3s.fields.curobo_builder import CuroboFieldBuilder
                self._esdf_builder = CuroboFieldBuilder(cfg)
            else:
                from benchmark.ag3s.fields.esdf import EsdfBuilder
                self._esdf_builder = EsdfBuilder(cfg)

        if cfg.backend == "curobo":
            # **프레임마다 본다.** 위 갈림길은 첫 프레임에 한 번만 지나가므로, 거기서만 확인하면
            # 그 뒤에 다른 경로가 legacy 를 만들어도 모른다. 출처 도장은 *쓰인* 필드가 무엇인지
            # 말해 주지만 만들어만 놓은 것은 안 잡는다 — 그래서 생성 횟수를 따로 센다.
            from benchmark.ag3s.fields.esdf import EsdfBuilder
            if EsdfBuilder.instances_created:
                raise RuntimeError(
                    f"esdf.backend='curobo' 인데 legacy EsdfBuilder 가 이 프로세스에서 "
                    f"{EsdfBuilder.instances_created} 번 만들어졌다. T0 의 즉시 실패 조건이다 — "
                    "누가 만들었는지 찾아 없앤다. 두 backend 가 한 프로세스에 공존하면 어느 "
                    "필드가 판정에 쓰였는지 기록만 보고는 되짚을 수 없다.")

        # 예전엔 여기서 `contact.rule_for(phase).contact_permission`을 보고 target을 필드에서
        # 통째로 carve() 했다 (E1, `docs/AG3S_REVIEW_LOG.md` Step 2). 문제: 필드는 익명이라
        # "누가 묻는지"를 모른다 — GRASP phase가 되면 그 물체는 잡는 손끝뿐 아니라 몸통·전완·
        # 반대팔에게도 사라졌다. 대신 필드는 **항상** target을 그대로 담아 두고,
        # `to_adapter.build_constraint_set`가 채우는 `manipulated_link_margin` (구별 마진, 권한 있는
        # 링크만 완화)이 trajopt의 `_esdf_clearance`에서 그 구별을 한다 — MoveIt의 Allowed
        # Collision Matrix, cuRobo의 attached-object 패턴과 같은 "지우지 말고 질의 쪽에서
        # 봐준다" 방식. `cfg.exclude_target`는 이제 이 경로에서 쓰이지 않는다 (config.py 참고).
        points = None if target is None else np.asarray(target.points, np.float64)
        # 목적지는 **주입**이다. AG3S 는 과제를 모르므로 어느 물체가 목적지인지 알 수 없다 —
        # phase 와 같은 계약이다. 라벨을 달아 두면 필드가 "가장 가까운 표면이 목적지인가" 를
        # 답할 수 있고, 그래야 그 표면에만 얇은 마진을 걸 수 있다 (F18).
        labelled = None
        if destination_points is not None and len(destination_points):
            labelled = {DESTINATION_LABEL: np.asarray(destination_points, np.float64)}
        # **target 도 라벨로 싣는다** (T8b) — `destination` 과 같은 규약이다. 라벨은 target 을
        # 필드에서 빼지 **않는다.** "지금 가장 가까운 표면이 target 인가" 를 물을 수 있게 할
        # 뿐이고, 그것은 진단용이다 (어느 로봇 구가 사과 때문에 막혀 있는지 — T8a 가 묻는 질문).
        # 완화 판정 자체는 라벨에 기대지 않는다: 거친 격자에서 값(삼선형)과 라벨(최근접 격자점)이
        # 어긋나는 자리가 하필 파지하는 자리다 (`CuroboEsdfField.target_free_distance`).
        #
        # **쥔 것이 없을 때만 싣는다.** 쥔 뒤에는 그 물체가 attached 경로로 필드에서 파이고
        # (아래 `attached_points`) 로봇 쪽 질의점이 된다 — 그래서 제외의 근거가 끊기지 않는다
        # (T26 §4: 파지 전 = manipulated, 파지 후 = attached, `metrics["exclusion"]`).
        # T26 전에는 "쥔 뒤의 target 은 목적지(crate)" 라는 이유를 적었다. 이제 target 은 파지
        # 뒤에도 사과이고, crate 는 admissible 이 아니라 이 인자로 들어올 수 없다.
        nothing_held = self._attached is None
        if (nothing_held and not self._grasp_active and points is not None and len(points)
                and self.free_held_traces):
            # T32 H3: where the object sits before the grasp (its surface fit, last frame before the
            # closing started — while closing the fingers cut the surface) — the TSDF voxels there
            # are set free once it is attached (`_arm_first_free`).
            from benchmark.ag3s.robot_models.held_object import fit_surface_sphere

            mo = self._max_opening
            fit = fit_surface_sphere(np.asarray(points, np.float64),
                                     max_radius=0.5 * mo.value_m if mo.known else None)
            if fit.accepted:
                self._last_object_fit = (fit.centre.copy(), float(fit.radius))
        if nothing_held and points is not None and len(points):
            labelled = dict(labelled or {})
            labelled[TARGET_LABEL] = np.asarray(points, np.float64)
        # target 없는 계층은 **정책이 요구할 때만** 만든다. 기본(`relax`)에서는 이 인자가
        # `None` 이고 프레임 비용이 예전과 같다.
        policy = self.config.constraint.target_field_policy
        target_free_points = None
        if (self.config.constraint.excludes_target_from_field and nothing_held
                and points is not None and len(points)):
            if cfg.backend != "curobo":
                raise ValueError(
                    f"constraint.target_field_policy={policy!r} 는 target 없는 계층을 요구하고, "
                    "그 계층은 cuRobo backend 만 만듭니다 (`CuroboFieldBuilder`). legacy "
                    "EsdfBuilder 로는 정책을 켰다고 믿은 채 아무 일도 일어나지 않으므로 여기서 "
                    "멈춥니다 — esdf.backend='curobo' 로 두거나 정책을 'relax' 로 두십시오")
            if not getattr(cfg, "fine_voxel_size", None):
                raise ValueError(
                    f"constraint.target_field_policy={policy!r} 는 미세 계층을 요구합니다 "
                    "(target 없는 계층은 그 한 겹으로 만들어집니다). "
                    "esdf.fine_voxel_size 가 비어 있어 단일 계층으로 돌고 있으므로, 켠 정책이 "
                    "아무 일도 하지 않습니다 — 값을 주거나 정책을 'relax' 로 두십시오")
            target_free_points = np.asarray(points, np.float64)
            # **점이 아니라 기하 범위로 지운다** (T13, 사용자 판정 "중심 + 반지름, 다만 테이블 면
            # 위로만"). 점으로는 사과를 **한 voxel도 지우지 못했다** — 표본점 112 개(그중 seed
            # 72 개)에 표면이 2,203 voxel 이고, 거리장은 성긴 제거에 꿈쩍하지 않는다.
            target_free_ball = self._target_ball(
                target_free_points, support_surfaces,
                voxel=float(getattr(cfg, "fine_voxel_size", 0.0) or cfg.voxel_size))
        # 쥔 물체는 **양쪽에 동시에 있으면 안 된다** (A2). 파지가 닫히는 순간 그 물체는
        # 로봇 쪽 질의점이 되므로 (E3 — 쥔 물체가 optimizer 에 도달하지 않는다), 장애물 쪽에서는
        # 빠져야 한다. 안 빼면 자기 자신에게 부딪히고 그 행은 **어떤 해로도 못 푼다**.
        #
        # **E1(조작 대상을 필드에서 파내면 손끝뿐 아니라 전신에게 사라진다)과 다른 경우다.**
        # E1 이 금지한 것은 *아직 안 쥔* target 을 파내는 것이었다. 쥔 뒤에는 로봇의 일부이고,
        # 로봇을 depth 에서 지우는 것과 같은 처리다. 그래서 `attach()` 가 불린 뒤에만 돈다.
        # T32 H3: with the cuRobo backend the held object's volume is set **free in the TSDF**
        # (`_held_free_points`) — the TSDF then holds no surface there to seed, so the per-frame seed
        # exclusion of that same volume is **retired** (measured: E3b ep1807 r1 · E3a r3, freeing
        # with vs without seed exclusion gave identical trails, table spot, held-sphere minimum and
        # crate voxels; T32a §H3). The seed exclusion stays for the paths that do not free: legacy
        # backend, `free_held_traces=False`, pre-T32 (points-only) attachments.
        held_free = None
        if cfg.backend == "curobo" and self._attached is not None and robot_state is not None \
                and self.free_held_traces and self.constraint_robot_model is not None:
            held_free = self._held_free_points(robot_state, support_surfaces, destination_points)
        attached_points = None
        if (held_free is None and self._attached is not None and robot_state is not None
                and self.constraint_robot_model is not None):
            from benchmark.ag3s.constraints.attached import attached_points_in_base
            # **`attach()` 가 쓴 것과 같은 모델이어야 한다.** 점은 그 모델의 parent link 프레임에
            # 스냅샷돼 있고, 다른 모델로 되돌리면 물체가 조용히 엉뚱한 자리에서 파인다.
            attached_points = self._held_seed_points(robot_state, support_surfaces)
            if attached_points is None:
                attached_points = attached_points_in_base(
                    self._attached, robot_model=self.constraint_robot_model,
                    robot_state=robot_state)
        # **창의 배치 (T21)** — 손이 정한다. 제외(`target_free_ball`)는 위에서 manipulated 가
        # 정했고 여기서 넓히지 않는다 (지침 §5.3-3: 창 확대 ≠ 제외 영역 확대).
        window_request = None
        if cfg.backend == "curobo":
            window_request = self._fine_window_request(
                robot_state, execution_path, contact_context, points)
        field = self._esdf_builder.update(depth_cameras, target_points=points,
                                          exclude_target=False,
                                          **({} if window_request is None
                                             else {"fine_window": window_request}),
                                          # 파내는 것이 아니라 **한 겹 더 만드는 것**이다 (T8b).
                                          # 본 계층은 한 복셀도 바뀌지 않는다.
                                          **({} if target_free_points is None else {
                                              "target_free_points": target_free_points,
                                              "target_free_label": TARGET_LABEL,
                                              "target_free_ball": target_free_ball}),
                                          support_points=support_points,
                                          attached_points=attached_points,
                                          # T32 H3: the held object's old traces → free.
                                          **({} if held_free is None
                                             else {"held_free_points": held_free}),
                                          labelled_points=labelled,
                                          # 아는 정적 기하 — **주입**이다. AG3S 는 무엇이 벽이고
                                          # 무엇이 선반인지 알 수 없다 (phase·목적지와 같은 계약).
                                          static_geometry=static_geometry,
                                          # 출처. age 의 기준은 **관측 시각**이다 — 서버
                                          # monotonic 은 프로세스마다 원점이 달라 클라이언트가
                                          # 자기 시계와 견줄 수 없다.
                                          observed_at=observed_at,
                                          frame_id=frame_id, frame_index=frame_index)
        notes = []
        if window_request is not None:
            notes = self._window_query_report(
                field, window_request,
                target_free=target_free_points is not None
                and bool(getattr(field, "has_target_free", False)),
                margin=self._fine_window_pads()[0])
        return field, notes

    def _held_seed_points(self, robot_state, support_surfaces) -> Optional[np.ndarray]:
        """The held spheres' **volume** as fine-voxel grid points (base) — what the field's seed
        exclusion removes (T32 H1), or None when the attachment has no spheres (points-only).

        Before T32 the exclusion used the 25–33 radius-0 snapshot points, so only *their* voxels
        lost their seeds and the rest of the apple surface stayed an obstacle (T31-diag: fine value
        at the apple centre down to −24 mm). The spheres are what the optimizer now queries with,
        so the field must not hold a surface inside them — "a query, not an obstacle" (A2) on the
        same set. Why it is still needed after H1: the TSDF surfaces integrated **before** attach
        (the apple on the table, the closing/settle requests) sit exactly inside the spheres at
        attach, and no ray can clear them any more — the self-filter now masks those very pixels
        (a masked ray is unobserved, not free). Per frame and non-destructive (seeds only, the TSDF
        is untouched). Points below the support plane under the sphere + one fine voxel are
        dropped — the `TargetBall` rule — so a sphere resting on the table does not erase the table.
        """
        from benchmark.ag3s.constraints.attached import _link_pose_numeric
        from benchmark.ag3s.robot_models.held_object import held_spheres_of, sphere_volume_points

        held = self._attached
        local = None if held is None else held_spheres_of(held)
        if not local:
            return None
        T = _link_pose_numeric(self.constraint_robot_model, np.asarray(robot_state, np.float64),
                               held.parent_link)
        spheres = [(T[:3, :3] @ c + T[:3, 3], r) for c, r in local]
        spacing = self._guard_pad()
        z_plane = self._support_plane_z(support_surfaces, spheres[0][0])
        z_min = None if z_plane is None else float(z_plane) + spacing
        pts = sphere_volume_points(spheres, spacing, z_min=z_min)
        return pts if len(pts) else None

    def _held_free_points(self, robot_state, support_surfaces, destination_points):
        """Points whose TSDF voxels are set **free** this frame (T32 H3), or None.

        The held spheres at `robot_state` (every held frame), plus on the first held frames the
        spheres at the geometry's capture pose and the object's pre-grasp place
        (`_arm_first_free`). Measured need (T32a §H3): H1 stops new integration, but the table spot
        and the frames between closing and attach stay in the TSDF — nothing clears them, since
        the rays through them are masked or the hand still occludes them.

        Only voxels the held object can occupy: points below the support plane + one fine voxel
        are dropped (the `TargetBall` rule — never a hole in the table), and so are points within
        two fine voxels of the destination (the crate the object is lowered into is never eroded).
        Sampled at half a TSDF voxel so every voxel whose centre is inside is hit.
        """
        from benchmark.ag3s.constraints.attached import _link_pose_numeric
        from benchmark.ag3s.robot_models.held_object import held_spheres_of, sphere_volume_points

        held = self._attached
        local = held_spheres_of(held)
        if not local:
            return None
        T = _link_pose_numeric(self.constraint_robot_model, np.asarray(robot_state, np.float64),
                               held.parent_link)
        spheres = [(T[:3, :3] @ c + T[:3, 3], r) for c, r in local]
        if self._held_first_free_frames > 0:
            spheres = spheres + list(self._held_first_free)
            self._held_first_free_frames -= 1
        cfg = self.config.esdf
        tsdf_voxel = float(getattr(cfg, "tsdf_voxel_size", None) or cfg.voxel_size)
        pad = self._guard_pad()
        z_plane = self._support_plane_z(support_surfaces, spheres[0][0])
        pts = sphere_volume_points(spheres, 0.5 * tsdf_voxel,
                                   z_min=None if z_plane is None else float(z_plane) + pad)
        if len(pts) and destination_points is not None and len(destination_points):
            from scipy.spatial import cKDTree

            d, _ = cKDTree(np.asarray(destination_points, np.float64).reshape(-1, 3)).query(
                pts, distance_upper_bound=2.0 * pad)
            pts = pts[~np.isfinite(d)]
        return pts if len(pts) else None

    def _authorized_links(self, contact_context) -> frozenset:
        """Contact-authorized links for this frame — the **same** set the target-free mask and the
        contact margin use (`ClearancePolicy.authorized_links`), so "the hand the window follows"
        and "the links that may pass through the object" cannot part."""
        if contact_context is None:
            return frozenset()
        if self._builder is not None:
            return frozenset(self._builder.clearance_policy.authorized_links(contact_context))
        links: set[str] = set()
        for manipulator in contact_context.active_manipulators:
            links.update(self.config.contact.links_for(manipulator))
        return frozenset(links)

    def _fine_window_pads(self) -> tuple[float, float]:
        """`(margin, interpolation pad)` in metres for the fine window (T21).

        margin = `esdf.fine_window_margin`, or `geometry.safety_margin` when that is null — the
        clearance the ESDF rows demand, so no second number; interpolation pad =
        `esdf.fine_window_pad_voxels` × the fine voxel.
        """
        cfg = self.config.esdf
        fine = float(getattr(cfg, "fine_voxel_size", None) or cfg.voxel_size)
        margin = (float(self.config.geometry.safety_margin) if cfg.fine_window_margin is None
                  else float(cfg.fine_window_margin))
        return margin, float(cfg.fine_window_pad_voxels) * fine

    def _fine_window_request(self, robot_state, execution_path, contact_context, points):
        """`FineWindowRequest` for this frame (T21). Pure numpy — FK through
        `constraint_robot_model.sphere_centers_numeric`, no cuRobo.

        Hand = the contact-authorized links' spheres at `robot_state` (step 0) then at each row
        of `execution_path` (priority order: now → executed steps). Missing any of robot state,
        model or authorized links leaves the hand empty and the window falls back to the
        manipulated centroid (`basis="manipulated"`) — recorded, not silent.

        The hand is computed under `placement="target_centroid"` too: the old rule ignores it for
        placement, but the record still says where the hand was relative to the window — that is
        the before/after comparison T21 is judged on.
        """
        from benchmark.ag3s.fields.curobo_builder import FineWindowRequest

        cfg = self.config.esdf
        margin, interp = self._fine_window_pads()
        placement = str(cfg.fine_window_placement)
        base = dict(manipulated_points=points, manipulated_pad=interp, placement=placement,
                    hand_pad=margin + interp)
        model = self.constraint_robot_model
        links = self._authorized_links(contact_context)
        if model is None or robot_state is None or not links:
            return FineWindowRequest(**base, hand_source="none", links=tuple(sorted(links)))
        names = [str(n) for n in getattr(model, "sphere_link_names", ()) or ()]
        idx = np.asarray([i for i, n in enumerate(names) if n in links], np.int64)
        if not idx.size:
            return FineWindowRequest(**base, hand_source="none", links=tuple(sorted(links)))
        nq = int(model.nq)
        states = [np.asarray(robot_state, np.float64).reshape(-1)]
        if execution_path is not None:
            path = np.asarray(execution_path, np.float64)
            path = path.reshape(-1, path.shape[-1]) if path.size else path.reshape(0, nq)
            states.extend(path)
        for i, q in enumerate(states):
            if q.shape[0] != nq:
                # 조용히 버리면 창이 손을 놓친 채 "hand_swept" 로 기록된다.
                raise ValueError(
                    f"fine window: {'robot_state' if i == 0 else f'execution_path[{i - 1}]'} has "
                    f"{q.shape[0]} entries but the constraint model expects nq={nq}")
        centres, radii = [], None
        for q in states:
            c, r = model.sphere_centers_numeric(q)
            centres.append(np.asarray(c, np.float64)[idx])
            if radii is None:
                radii = np.asarray(r, np.float64).reshape(-1)[idx]
        return FineWindowRequest(
            **base, hand_centres=np.stack(centres), hand_radii=radii,
            hand_source="q_now" if len(states) == 1 else "q_now+path",
            links=tuple(names[i] for i in idx))

    @staticmethod
    def _window_query_report(field, request, *, target_free: bool,
                             margin: float = 0.0) -> list[str]:
        """Which tier answered the authorized links' query points (T21 §2·§3). Returns notes.

        Adds `field.stats["window"]["authorized_query"]`. A query point answered by `coarse` or
        `outside` is **not** a fine-resolution answer: it is recorded and noted, never read as
        free space or as a successful contact permission (guide §5.3). A `window_edge` answer is
        a lower bound (`b`, distance to the window boundary); it settles the row only when the
        bound alone already clears `radius + margin` (`n_bounded`), otherwise it is unresolved too.
        """
        from benchmark.ag3s.fields.curobo_field import count_tiers, unresolved_rows

        stats = getattr(field, "stats", None)
        window = None if stats is None else stats.get("window")
        # 미세 계층이 없는 설정(단일 계층)에는 "창 밖" 이 없다 — 보고할 것이 없다.
        if window is None or window.get("basis") is None or request.hand_centres is None:
            return []
        tier_of = getattr(field, "answer_tier", None)
        if tier_of is None:
            return []
        centres = request.hand_centres
        n_steps, n_spheres = centres.shape[:2]
        pts = centres.reshape(-1, 3)
        tiers = np.asarray(tier_of(pts, target_free=target_free))
        inside = np.zeros(len(pts), bool)
        if window.get("lower_m") is not None:
            lo = np.asarray(window["lower_m"], np.float64)
            hi = np.asarray(window["upper_m"], np.float64)
            inside = np.all((pts >= lo) & (pts <= hi), axis=1)
        value = (field.target_free_distance(pts) if target_free else field.distance(pts))
        radii = np.tile(np.asarray(request.hand_radii, np.float64), n_steps)
        unresolved, bounded = unresolved_rows(tiers, np.asarray(value) - radii - float(margin))
        now = slice(0, n_spheres)
        window["authorized_query"] = {
            "n_points": int(len(pts)),
            "n_steps": int(n_steps),
            "n_spheres": int(n_spheres),
            "target_free": bool(target_free),
            "fraction_in_window": float(inside.mean()) if len(pts) else None,
            "fraction_in_window_now": float(inside[now].mean()) if n_spheres else None,
            "by_tier": count_tiers(tiers),
            "by_tier_now": count_tiers(tiers[now]),
            "n_unresolved": int(unresolved.sum()),
            "n_bounded": int(bounded.sum()),
            "margin_m": float(margin),
        }
        notes = []
        if window.get("basis") == "overflow":
            notes.append(
                f"fine window overflow ({window.get('overflow')}): the hand's swept box ∪ the "
                f"manipulated box does not fit {window.get('extent_m')} m; placed hand-first, "
                f"{window.get('n_steps_covered')}/{window.get('n_steps')} step(s) covered, "
                f"manipulated covered={window.get('manipulated_covered')}")
        if unresolved.any():
            by = {k: v for k, v in count_tiers(tiers[unresolved]).items() if v}
            notes.append(
                f"fine window: {int(unresolved.sum())}/{len(pts)} authorized-link query point(s) "
                f"({request.hand_source}, links {', '.join(request.links)}) are answered by "
                f"{by} — not at fine resolution"
                + (" and not by the target-free layer" if target_free else "")
                + "; these rows are unverified, not free space and not a contact permission")
        return notes

    def _target_ball(self, points, support_surfaces, *, voxel: float):
        """target 점구름 → `TargetBall`, 또는 **`None`** (안전하게 물러난다).

        사용자 판정이 *"중심 + 반지름, 다만 테이블 면 위로만"* 이다. 이 함수는 그 셋을 **재서**
        정한다 — 숫자를 박지 않는다. 사과·오렌지·배가 다 다르고, 무엇을 썼는지는 시작 로그와
        기록(`stats["target_free"]["ball"]`)에 그대로 실린다.

        | 값 | 어떻게 |
        |---|---|
        | 중심 | **관측 표면에 맞춘 구의 중심** (T32 S1, `fit_surface_sphere`); 맞춤이 물러나면 centroid |
        | 반지름 | `max(중심까지 거리의 target_ball_quantile 분위수, 맞춘 반지름)` + 여유 `pad × voxel` |
        | `z_min` | **지원면** 평면 z (centroid 의 x·y 에서) + 여유 `pad × voxel` |

        **`z_min` 을 못 얻으면 `None` 을 돌려준다.** 그러면 호출자는 지금 동작(점 제외)으로
        물러난다 — 사과는 안 지워지지만 **테이블에 구멍이 나지 않는다.** 조용히 물러나지 않는다:
        builder 의 시작 로그가 "제외 방식 = 점" 을 크게 적고, 기록의 `mode` 가 `"points"` 로 남는다.
        테이블을 지우는 쪽이 최악이기 때문에 이 방향으로 닫는다 — 팔이 테이블을 통과해도 아무도
        막지 않는 것보다, 사과를 못 잡는 것이 낫다.

        지원면을 **평면으로** 읽는 이유: AG3S 는 테이블을 반평면으로 뽑는다 (`SupportSurface`,
        `n · p = d`). 점구름의 최소 z 를 쓰면 사과 밑면의 관측 노이즈가 그대로 절단면이 된다.
        """
        from benchmark.ag3s.fields.curobo_builder import TargetBall

        from benchmark.ag3s.robot_models.held_object import fit_surface_sphere

        pts = np.asarray(points, np.float64).reshape(-1, 3)
        if not len(pts):
            return None
        cfg = self.config.esdf
        quantile = float(getattr(cfg, "target_ball_quantile", 1.0))
        pad = float(getattr(cfg, "target_ball_pad_voxels", 1.0)) * float(voxel)
        # T32 S1: the centre is the **sphere fitted to the observed surface** (a convex fruit's
        # centre from its one visible side), not the centroid of that side — the centroid sits on
        # the seen face and the ball missed the apple's back by a median 4.4 mm once T31b split
        # merged clusters. The radius covers both the observed points (as before) and the fitted
        # sphere (the unseen back). The fit's radius is capped at half the max opening (nothing
        # graspable is larger); a surface that is not a sphere falls back to the centroid rule.
        mo = getattr(self, "_max_opening", None)
        fit = fit_surface_sphere(
            pts, max_radius=(0.5 * mo.value_m if mo is not None and mo.known else None))
        centre = fit.centre
        spread = np.linalg.norm(pts - centre[None, :], axis=1)
        raw = float(np.quantile(spread, np.clip(quantile, 0.0, 1.0)))
        if fit.accepted:
            raw = max(raw, float(fit.radius))
        radius = raw + pad
        plane_z = self._support_plane_z(support_surfaces, centre)
        if plane_z is None or not np.isfinite(radius) or radius <= 0.0:
            return None
        if plane_z >= centre[2] + radius:
            # 평면이 공보다 위에 있다 = 그 평면은 이 target 의 받침이 아니다. 지울 것이 없으므로
            # 물러난다 — 엉뚱한 평면으로 자르면 공 전체가 살거나 전체가 지워진다.
            return None
        return TargetBall(
            centre=centre, radius=radius, z_min=float(plane_z) + pad,
            provenance={
                "n_points": int(len(pts)),
                "radius_quantile": quantile,
                "radius_raw_mm": round(raw * 1000.0, 3),
                # T32 S1: how the centre was found (`surface_fit` | `centroid` + reason).
                "centre_method": fit.method,
                "fit": fit.record(),
                "pad_mm": round(pad * 1000.0, 3),
                "support_plane_z_m": float(plane_z),
                "voxel_mm": round(float(voxel) * 1000.0, 3),
            })

    @staticmethod
    def _support_plane_z(support_surfaces, point) -> Optional[float]:
        """`point` 의 x·y 에서 **그 점 아래** 가장 높은 수평 지원면의 z. 없으면 `None`.

        수평 판정은 법선의 z 성분으로 한다 (`|n_z| > 0.9`) — 벽이나 선반 옆면을 받침으로 쓰면
        절단면이 엉뚱한 곳에 생긴다. 여러 개면 **가장 높은 것**을 고른다: 테이블 위에 놓인
        물건의 받침은 바닥이 아니라 테이블이다.
        """
        p = np.asarray(point, np.float64).reshape(3)
        best = None
        for surface in support_surfaces or ():
            n = np.asarray(getattr(surface, "normal", ()), np.float64).reshape(-1)
            if n.size != 3 or abs(float(n[2])) <= 0.9:
                continue
            nz = float(n[2])
            # `n · p = d` 를 z 로 풀어 (x, y) 에서의 평면 높이를 얻는다.
            z = (float(getattr(surface, "offset", 0.0)) - float(n[0]) * p[0]
                 - float(n[1]) * p[1]) / nz
            if not np.isfinite(z) or z > p[2]:
                continue  # 점 위에 있는 면은 받침이 아니다
            best = z if best is None else max(best, z)
        return best

    @staticmethod
    def _with_camera_provenance(target, fusion, attention_cloud):
        """Record which cameras saw the target, and which one carried its attention peak.

        The peak is read off the pre-normalization signal for the reason `AttentionPointCloud`
        explains — percentile clipping ties hundreds of points at 1.0, and `argmax` over ties picks
        an arbitrary one, which here would credit an arbitrary camera.
        """
        fused = fusion.cloud
        indices = np.asarray(target.point_indices, np.int64)
        supporting: set = set()
        for i in indices:
            supporting |= fused.cameras_of(int(i))

        seed_camera = None
        if indices.size:
            peak_signal = attention_cloud.peak_signal[indices]
            peak_point = int(indices[int(np.argmax(peak_signal))])
            block = fused.observation_slice(peak_point)
            local = fused.obs_attention[block]
            if local.size:
                seed_camera = fused.cameras[int(fused.obs_camera[block][int(np.argmax(local))])]

        return dataclasses.replace(
            target,
            seed_camera=seed_camera,
            supporting_cameras=tuple(sorted(supporting, key=lambda c: c.value)),
        )

    @staticmethod
    def _fused_recon_stats(fusion) -> dict[str, Any]:
        """Present the fused front end in the shape the single-camera stats block already uses.

        Keeping one shape means the notes, the metrics and `_stat_notes` need no branch, and a reader
        comparing a fused frame against a single-camera one is comparing the same fields.
        """
        capped = any(r.stats.get("capped", False) for r in fusion.per_camera)
        return {
            "n_raw": sum(int(r.stats.get("n_raw", 0)) for r in fusion.per_camera),
            "n_voxel": fusion.metrics["n_points_before_fusion"],
            "n_final": fusion.metrics["n_points_fused"],
            "capped": capped,
            "covered": all(r.stats.get("covered", True) for r in fusion.per_camera),
            "final_voxel_size": max(
                (float(r.stats.get("final_voxel_size", 0.0)) for r in fusion.per_camera),
                default=0.0,
            ),
            "n_growth_steps": max(
                (int(r.stats.get("n_growth_steps", 0)) for r in fusion.per_camera), default=0
            ),
        }

    def _clearance_metrics(self, context, target_grounded: bool) -> dict[str, Any]:
        """What the contact policy authorized this frame, flattened for logs.

        Recorded per frame rather than derived on demand because "which links were relaxed, against
        what" is the question any incident review starts from, and the context that produced it is
        gone by then.
        """
        if self._builder is None:
            return {}
        record = self._builder.clearance_policy.describe(context, target_grounded=target_grounded)
        return {f"clearance_{k}": v for k, v in record.items()}

    @staticmethod
    def _stat_notes(recon_stats, filter_stats, candidate_stats) -> list[str]:
        notes = [
            f"points: {recon_stats['n_raw']} raw -> {recon_stats['n_voxel']} voxel -> "
            f"{filter_stats['n_out']} after self-filter"
            + ("" if filter_stats["enabled"] else " (self-filter off)"),
        ]
        # `candidate_stats` is empty when the primitive path did not run at all. Report nothing
        # rather than a row of zeros, which would read as "clustering found nothing" — a very
        # different and much more alarming statement than "clustering was not asked to run".
        if candidate_stats:
            notes.append(
                f"candidates: {candidate_stats['n_object']} object, "
                f"{candidate_stats['n_unknown']} unknown, "
                f"{candidate_stats['n_target']} target, {candidate_stats['n_support']} support"
            )
        return notes


# ------------------------------------------------------------------------------------ CLI


def main(argv: Optional[list[str]] = None) -> int:
    """Smoke-run the pipeline on the synthetic fixture and print the latency table.

    The fixture lives in `tests/ag3s/` because it is test data, so this entry point imports it
    lazily and only for the demo. Nothing in `benchmark/ag3s/` depends on it.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="AG3S end-to-end smoke run")
    parser.add_argument("--config", default="benchmark/ag3s/configs/default.yaml")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--phase", default="approach")
    parser.add_argument("--profile", action="store_true", help="print the per-stage latency table")
    parser.add_argument("--json", default=None, help="write the constraint-set summary here")
    args = parser.parse_args(argv)

    try:
        from tests.ag3s.fixtures import DEFAULT_ARM_Q, PlanarTwoLinkArm, make_scene
    except ImportError as exc:  # pragma: no cover - demo path
        print(f"the demo scene lives in tests/ag3s/fixtures.py; run from the workspace root ({exc})")
        return 1

    from benchmark.ag3s.constraints.to_adapter import summary

    config = AG3SConfig.from_yaml(args.config)
    # One warm-up frame: the first pass pays for scipy's KD-tree import and CasADi's first symbolic
    # build, which is not steady-state latency.
    config = config.with_overrides({"profiling": {"enabled": True, "warmup_frames": 1}})

    arm = PlanarTwoLinkArm()
    ag3s = AG3S(config, robot_model=arm)
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)

    constraint_set = None
    for _ in range(max(args.frames, 2)):
        constraint_set = ag3s.process(
            attention_map=scene.attention_grid,
            depth=scene.depth,
            camera_intrinsics=scene.camera_intrinsics,
            T_base_cam=scene.T_base_cam,
            robot_state=DEFAULT_ARM_Q,
            phase=args.phase,
            image_hw=scene.hw,
        )

    digest = summary(constraint_set)
    print(json.dumps(digest, indent=2, default=str))
    for note in constraint_set.notes:
        print(f"  note: {note}")
    if args.profile:
        print()
        print(ag3s.profiler.to_table())
    if args.json:
        import pathlib

        pathlib.Path(args.json).write_text(json.dumps(digest, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["AG3S", "main"]
