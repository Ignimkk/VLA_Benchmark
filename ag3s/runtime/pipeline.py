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
import time
from typing import Any, Optional, Sequence

import numpy as np

from benchmark.ag3s.stages.attention_lifting import lift, make_adapter
from benchmark.ag3s.constraints.clearance import ClearancePolicy
from benchmark.ag3s.stages.collision_candidates import CandidateTracker, generate_candidates
from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.runtime.profiler import StageProfiler
from benchmark.ag3s.stages.reconstruction import reconstruct
from benchmark.ag3s.stages.robot_filter import filter_robot_points
from benchmark.ag3s.runtime.multiview import fuse_observations
from benchmark.ag3s.stages.support_surface import fit_support_surfaces
from benchmark.ag3s.stages.target_grounding import GroundingResult, TargetConfirm, ground_target
from benchmark.ag3s.constraints.to_adapter import build_constraint_set
from benchmark.ag3s.runtime.degradation import ensure_reason, reason
from benchmark.ag3s.types import (
    DESTINATION_LABEL,
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
    ):
        self.config = config or AG3SConfig()
        self.robot_model = robot_model
        self.constraint_robot_model = constraint_robot_model or robot_model
        self.attention_adapter = attention_adapter
        self.tracker = CandidateTracker(self.config.collision_candidate)
        #: The only cross-frame state grounding has (T5e). `ground_target` still decides from one
        #: frame; this decides whether a *different* object has led long enough to take the target
        #: over. It lives here because the episode lives here — `reset()` is the episode boundary.
        self._target_confirm = TargetConfirm(self.config.clustering)
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

    def reset(self) -> None:
        """Clear cross-frame state. Call between episodes, never mid-episode.

        This *does* drop an attached object, because an episode boundary is an explicit statement
        that the previous task is over. Mid-episode there is no path from perception to detachment.
        """
        self.tracker.reset()
        self.profiler.reset()
        self._target_confirm.reset()
        self.frame_index = 0
        self._attached = None
        self._esdf_builder = None

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
    ) -> AttachedCollisionGeometry:
        """Record an externally confirmed grasp. **AG3S never calls this itself.**

        Pass an `AttachedCollisionGeometry` to attach a known shape, or a `TargetGeometry` together
        with `robot_state` and `parent_link` to snapshot what perception last fitted. Either way the
        result is a snapshot: from here on the object's existence is a fact about the gripper, not a
        claim about what the cameras can see.

        There is deliberately no grasp detector behind this. AG3S cannot tell a closed gripper
        holding a cup from a closed gripper holding nothing, and guessing would mean either an
        invisible object or a phantom one attached to the hand.
        """
        if isinstance(geometry, AttachedCollisionGeometry):
            self._attached = geometry
            return self._attached
        if robot_state is None or parent_link is None:
            raise ValueError(
                "attaching a TargetGeometry needs the robot_state the grasp closed in and the "
                "parent_link it closed on; a later state would bake in however far the arm has moved"
            )
        if self.constraint_robot_model is None:
            raise ValueError("attaching needs a robot model to invert the parent link's pose")

        from benchmark.ag3s.constraints.attached import attach_from_target

        self._attached = attach_from_target(
            geometry,
            robot_model=self.constraint_robot_model,
            robot_state=robot_state,
            parent_link=parent_link,
            allowed_contact_links=allowed_contact_links,
            label=label,
            timestamp=time.time() if timestamp is None else float(timestamp),
        )
        return self._attached

    def detach(self) -> Optional[AttachedCollisionGeometry]:
        """Record an externally confirmed release. Returns what was let go, or None.

        The only way an attached object leaves. Grounding failure, occlusion and camera dropout all
        deliberately have no path here: the robot does not put an object down because it stopped
        being able to see it.
        """
        released, self._attached = self._attached, None
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
    ) -> tuple[CollisionConstraintSet, dict[str, Any]]:
        """The single implementation behind `process`, `process_debug` and `process_multi`.

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

        fusion = None
        if observations is not None:
            # 1 + 2 + 4, fused. Each camera is reconstructed and self-filtered at its own capture
            # instant, then merged on a base-frame voxel grid with provenance retained.
            with profiler.stage("scene_reconstruction"):
                fusion = fuse_observations(
                    observations,
                    cfg,
                    robot_model=self.robot_model,
                    attention_adapter=self.attention_adapter,
                    now=timestamp,
                )
            cloud = fusion.pointcloud
            recon_stats = self._fused_recon_stats(fusion)
            filter_stats = {
                "n_in": fusion.metrics["n_points_before_fusion"] + fusion.metrics["n_self_filtered"],
                "n_removed": fusion.metrics["n_self_filtered"],
                "n_out": fusion.metrics["n_points_before_fusion"],
                "enabled": cfg.pointcloud.self_filter and self.robot_model is not None,
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
                    cloud, self.robot_model, robot_state, cfg.pointcloud
                )

        # 3. support surfaces -------------------------------------------------------------
        with profiler.stage("support_surface"):
            surfaces, support_mask = fit_support_surfaces(cloud, cfg.support_surface, timestamp=ts)

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
        if grounding.target is None:
            notes.append(
                f"no target ({grounding.status.value}); all geometry held at full clearance"
            )
        else:
            decision = self._target_confirm.last
            if decision is not None and decision.mode in ("hold", "switch", "unobserved"):
                # Said out loud only when the hysteresis did something, because a frame where it
                # overruled the highest-scoring cluster is otherwise indistinguishable from one
                # where it agreed. Not a `reason(...)`: nothing is degraded — the target is named
                # with full geometry either way.
                notes.append(
                    f"target {decision.mode} (rank {decision.rank}, challenger at "
                    f"{decision.streak}/{decision.frames} frames)"
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
                        cameras, grounding.target,
                        support_points=(cloud.points[support_mask]
                                        if cfg.esdf.exclude_support_surfaces
                                        and support_mask is not None else None),
                        destination_points=destination_points,
                        static_geometry=static_geometry,
                        robot_state=robot_state,
                        # age 의 기준은 **적분한 관측 중 가장 최신의 촬영 시각**이다.
                        # `q_now` 를 고르는 규칙(`safe_policy._scene_fn`)과 같게 둔다 —
                        # 둘이 어긋나면 "이 자세에서 이 필드" 라는 짝이 깨진다.
                        observed_at=(max(float(o.timestamp) for o in observations)
                                     if observations else ts),
                        frame_id=self.config.frame_id,
                        frame_index=self.frame_index)
                notes.extend(esdf_notes)
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
                    "n_support_points": int(support_mask.sum()),
                    "target_grounding_status": grounding.status.value,
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
        destination_points=None,
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
        from benchmark.ag3s.stages.robot_filter import robot_sphere_mask

        model = self.robot_model
        if model is None or robot_state is None:
            return None
        cfg = self.config.pointcloud
        if not cfg.self_filter:
            return None
        cloud = backproject(depth, K, T_base_cam, cfg)
        if cloud.uv is None or len(cloud) == 0:
            return None
        centres, radii = model.sphere_centers_numeric(np.asarray(robot_state, np.float64))
        inside = robot_sphere_mask(cloud.points, centres, radii, cfg.self_filter_inflation)
        mask = np.zeros(np.asarray(depth).shape, bool)
        uv = cloud.uv[inside]
        mask[uv[:, 1], uv[:, 0]] = True
        return mask

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

    def _build_esdf(self, depth_cameras, target, *, support_points=None,
                    destination_points=None, static_geometry=None, robot_state=None,
                    observed_at=None, frame_id="", frame_index=-1):
        """Integrate this frame into the ESDF and return `(field, notes)`.

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
        # 쥔 물체는 **양쪽에 동시에 있으면 안 된다** (A2). 파지가 닫히는 순간 그 물체는
        # 로봇 쪽 질의점이 되므로 (E3 — 쥔 물체가 optimizer 에 도달하지 않는다), 장애물 쪽에서는
        # 빠져야 한다. 안 빼면 자기 자신에게 부딪히고 그 행은 **어떤 해로도 못 푼다**.
        #
        # **E1(조작 대상을 필드에서 파내면 손끝뿐 아니라 전신에게 사라진다)과 다른 경우다.**
        # E1 이 금지한 것은 *아직 안 쥔* target 을 파내는 것이었다. 쥔 뒤에는 로봇의 일부이고,
        # 로봇을 depth 에서 지우는 것과 같은 처리다. 그래서 `attach()` 가 불린 뒤에만 돈다.
        attached_points = None
        if (self._attached is not None and robot_state is not None
                and self.constraint_robot_model is not None):
            from benchmark.ag3s.constraints.attached import attached_points_in_base
            # **`attach()` 가 쓴 것과 같은 모델이어야 한다.** 점은 그 모델의 parent link 프레임에
            # 스냅샷돼 있고, 다른 모델로 되돌리면 물체가 조용히 엉뚱한 자리에서 파인다.
            attached_points = attached_points_in_base(
                self._attached, robot_model=self.constraint_robot_model,
                robot_state=robot_state)
        field = self._esdf_builder.update(depth_cameras, target_points=points,
                                          exclude_target=False,
                                          support_points=support_points,
                                          attached_points=attached_points,
                                          labelled_points=labelled,
                                          # 아는 정적 기하 — **주입**이다. AG3S 는 무엇이 벽이고
                                          # 무엇이 선반인지 알 수 없다 (phase·목적지와 같은 계약).
                                          static_geometry=static_geometry,
                                          # 출처. age 의 기준은 **관측 시각**이다 — 서버
                                          # monotonic 은 프로세스마다 원점이 달라 클라이언트가
                                          # 자기 시계와 견줄 수 없다.
                                          observed_at=observed_at,
                                          frame_id=frame_id, frame_index=frame_index)
        return field, []

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
