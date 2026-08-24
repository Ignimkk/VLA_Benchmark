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

from benchmark.ag3s.attention_lifting import lift, make_adapter
from benchmark.ag3s.clearance import ClearancePolicy
from benchmark.ag3s.collision_candidates import CandidateTracker, generate_candidates
from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.profiler import StageProfiler
from benchmark.ag3s.reconstruction import reconstruct
from benchmark.ag3s.robot_filter import filter_robot_points
from benchmark.ag3s.multiview import fuse_observations
from benchmark.ag3s.support_surface import fit_support_surfaces
from benchmark.ag3s.target_grounding import GroundingResult, ground_target
from benchmark.ag3s.to_adapter import build_constraint_set
from benchmark.ag3s.types import (
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
            from benchmark.ag3s.constraint_builder import ConstraintBuilder

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
        self.frame_index = 0
        self._attached = None

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

        from benchmark.ag3s.attached import attach_from_target

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
            notes.append(
                f"point cloud capped at max_points={cfg.pointcloud.max_points} "
                f"(from {recon_stats['n_voxel']}); voxel grown to "
                f"{recon_stats['final_voxel_size'] * 1000:.1f} mm in "
                f"{recon_stats['n_growth_steps']} step(s)"
            )
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
                attention_cloud = self._lift(
                    cloud, attention_map, image_hw, camera_intrinsics, T_base_cam
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

        # 6 + 7. collision candidates and primitive fitting -------------------------------
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
        profiler.record("primitive_fitting", candidate_stats.get("primitive_fit_ms", 0.0))
        if candidate_stats.get("n_overflow"):
            notes.append(
                f"{candidate_stats['n_overflow_points']} point(s) beyond "
                f"max_clusters={cfg.collision_candidate.max_clusters} / "
                f"unknown_min_points={cfg.collision_candidate.unknown_min_points} were folded into "
                f"{candidate_stats['n_overflow']} conservative aggregate(s)"
            )
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
            constraint_set = self._constraints(
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
                    "n_points_self_filtered": filter_stats["n_removed"],
                    "self_filter_enabled": filter_stats["enabled"],
                    "n_support_points": int(support_mask.sum()),
                    "target_grounding_status": grounding.status.value,
                    "target_confidence": (
                        0.0 if grounding.target is None else float(grounding.target.confidence)
                    ),
                    "n_object": candidate_stats["n_object"],
                    "n_unknown": candidate_stats["n_unknown"],
                    "n_overflow": candidate_stats["n_overflow"],
                    "n_overflow_points": candidate_stats["n_overflow_points"],
                    "n_unassigned_points": candidate_stats["n_unassigned_points"],
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
        validity=ConstraintValidity.VALID, metrics=None,
    ):
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
                notes=notes + ["no robot model injected; constraints not generated"],
                # There is geometry but nothing to write constraints against. That is not a certified
                # scene, and calling it VALID would let a caller read "no constraints" as "clear".
                validity=ConstraintValidity.worst(validity, ConstraintValidity.DEGRADED),
                contact_context=context,
                metrics=dict(metrics or {}),
                attached=self._attached,
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
        )

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
        return [
            f"points: {recon_stats['n_raw']} raw -> {recon_stats['n_voxel']} voxel -> "
            f"{filter_stats['n_out']} after self-filter"
            + ("" if filter_stats["enabled"] else " (self-filter off)"),
            f"candidates: {candidate_stats['n_object']} object, {candidate_stats['n_unknown']} unknown, "
            f"{candidate_stats['n_target']} target, {candidate_stats['n_support']} support",
        ]


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

    from benchmark.ag3s.to_adapter import summary

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
