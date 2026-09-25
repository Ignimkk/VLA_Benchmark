"""AG3S on the RB-Y1 crate-transport scene, through its ZED and two D435i cameras — **fused**.

The earlier version of this script ran one `AG3S` per camera and printed three reports. Every one of
them looked reasonable and together they were unusable: the crate was candidate `0` in the head view
and the table was candidate `0` in a wrist view, so `id=0` meant a different object depending on
which report you were reading. A trajectory optimizer cannot consume that, and neither can a person.

What runs now is one pipeline over one scene. Each camera is reconstructed and self-filtered with the
robot state captured alongside its own image, placed in the base frame by forward kinematics through
its mount link, and merged on a base-frame voxel grid with provenance retained. Support surfaces,
grounding, candidates and constraints then see a single cloud, so an object has one identity.

What this exercises that the synthetic fixture cannot:

* **A real robot in the frame.** The head ZED looks down its own arms, so the self-filter has to
  remove tens of thousands of robot pixels without touching the table or the crate.
* **Genuine cross-view overlap.** The head at 1.45 m and the wrists at ~0.5 m see the same crate from
  three angles, which is the only way to test whether fusion merges what it should.
* **Kinematics that actually differ per camera.** The wrist cameras hang off `link_left_arm_6` and
  `link_right_arm_6`; their poses come from FK, not from a table of constants.
* **Clutter nobody tuned on.** Office walls, a shelf, a crate with handles, four fruits.

**Attention is a stand-in.** Pi-0.5 is trained on LIBERO's 224x224 agent view; pointing it at an
RB-Y1 ZED frame yields a number with no meaning. `gaussian_attention` puts a plausible blob on a
chosen object instead, so everything downstream of attention is under test and attention itself is
not. What that costs is stated in the report rather than left implicit.

The output stops where the spec says AG3S stops: a `CollisionConstraintSet`, the candidate index
behind it, and the figures. Nothing here optimizes a trajectory.

    MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.reports.rby1_transport
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from typing import Optional, Sequence

import numpy as np

from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.experiments.sources.mujoco_source import (
    CAMERA_MOUNTS,
    HEAD_JOINTS,
    CameraFrame,
    TransportScene,
    camera_observation,
    gap_filling_capsules,
    gaussian_attention,
    is_robot_body,
    link_pose_error,
    pixel_labels,
)
from benchmark.ag3s.runtime.pipeline import AG3S
from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
from benchmark.ag3s.constraints.to_adapter import summary
from benchmark.ag3s.types import CameraID, Manipulator, SourceType

#: The three cameras of a fused frame. `zed_right` exists but adds a 12 cm baseline on the same
#: mount, which is stereo rather than coverage.
DEFAULT_CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")

#: The body each camera's attention blob is aimed at. The head is shown the crate it is about to
#: lift; each wrist is shown the fruit on its own side. They deliberately disagree, so the fused
#: grounding has to resolve three opinions rather than three copies of one.
DEFAULT_TARGETS = {
    "zed_left": "crate",
    "wrist_cam_l": "pear",
    "wrist_cam_r": "orange",
}

#: Capture instants, in seconds. Spread across 30 ms so the freshness machinery has something real to
#: report — this is roughly what three USB3 cameras on one host actually deliver.
DEFAULT_TIMESTAMPS = (0.000, 0.015, 0.030)

#: Which manipulator is authorized to touch the target. Injected, as the contract requires: the phase
#: says a grasp is happening and this says which hand is doing it.
DEFAULT_MANIPULATORS = (Manipulator.RIGHT,)

#: Tabletop working volume plus the fusion grid.
SCENE_OVERRIDES = {
    # `range_max` rather than `depth_max` alone: these are 90-degree lenses, so a z-depth cut of
    # 2.2 m still admits the office wall 4.2 m away at the frame corners. 1.6 m radial covers the
    # table, the crate, all four fruits and the floor under the robot, and stops at the room.
    "pointcloud": {"depth_max": 2.5, "range_max": 1.6, "voxel_size": 0.006, "max_points": 60000},
    "support_surface": {"max_planes": 2, "min_inliers": 800},
    "clustering": {"eps": 0.03, "min_points": 25},
    "collision_candidate": {"eps": 0.03, "min_points": 25},
    "timing": {
        # Coarser than the per-camera voxel on purpose: registration error *between* cameras is
        # larger than quantization *within* one, so a fusion voxel as fine as 6 mm would leave the
        # same table surface as three parallel sheets a few millimetres apart.
        "fusion_voxel_size": 0.012,
        "expected_cameras": ["head", "left_wrist", "right_wrist"],
        "max_camera_skew_sec": 0.05,
    },
}


@dataclasses.dataclass
class FusedRun:
    """One fused frame, with per-camera ground truth kept for scoring."""

    constraint_set: object
    debug: dict
    frames: dict  # CameraID -> CameraFrame
    labels: np.ndarray  # GT body id per *fused* point
    body_names: dict
    targets: dict  # camera name -> the body its attention blob was aimed at
    link_errors: dict


def _purity(labels: np.ndarray, names: dict) -> tuple[str, float]:
    """Dominant ground-truth body in a candidate, and what fraction of it that body is."""
    if labels.size == 0:
        return "-", 0.0
    values, counts = np.unique(labels, return_counts=True)
    top = int(values[int(np.argmax(counts))])
    return names.get(top, "?"), float(counts.max() / labels.size)


def fused_labels(run_debug: dict, frames: dict) -> np.ndarray:
    """Ground-truth body id per fused point, taken from its representative observation.

    The representative is an actual observed pixel in an actual camera, so its label is a real
    lookup rather than a vote — which is the practical payoff of not synthesising centroids.
    """
    fusion = run_debug["fusion"]
    cloud = fusion.cloud
    out = np.full(len(cloud), -1, np.int32)
    for i in range(len(cloud)):
        obs = int(cloud.representative[i])
        camera = cloud.cameras[int(cloud.obs_camera[obs])]
        u, v = cloud.obs_uv[obs]
        frame = frames[camera]
        if 0 <= v < frame.body_ids.shape[0] and 0 <= u < frame.body_ids.shape[1]:
            out[i] = frame.body_ids[int(v), int(u)]
    return out


def run_fused(
    scene: TransportScene,
    cameras: Sequence[str],
    *,
    config: AG3SConfig,
    phase: str,
    robot_models: dict,
    manipulators: Sequence[Manipulator] = DEFAULT_MANIPULATORS,
    timestamps: Sequence[float] = DEFAULT_TIMESTAMPS,
) -> FusedRun:
    """Three cameras, one scene, one candidate index."""
    filter_model = robot_models["filter"]
    observations, frames, targets = [], {}, {}
    for camera, timestamp in zip(cameras, timestamps):
        target_body = DEFAULT_TARGETS.get(camera, "crate")
        # Built twice on purpose: once to get the frame for the attention blob and the ground truth,
        # once inside `camera_observation` for the depth AG3S consumes. The scene is static between
        # them, so the two captures are identical.
        observation, frame = camera_observation(
            scene, camera, filter_model, timestamp=timestamp
        )
        observation = dataclasses.replace(
            observation,
            attention_map=gaussian_attention(frame, scene.body_position_in_base(target_body)),
        )
        observations.append(observation)
        frames[observation.camera_id] = frame
        targets[camera] = target_body

    ag3s = AG3S(
        config,
        # Two models on purpose. The filter has to account for everything the cameras see of the
        # robot, including the base and wheels the URDF's arm-only collision model omits; the
        # constraints only need the chain that moves, and every extra sphere costs
        # horizon * slots rows.
        robot_model=filter_model,
        constraint_robot_model=robot_models["constraint"],
    )
    constraint_set, debug = ag3s.process_multi_debug(
        observations, phase=phase, active_manipulators=list(manipulators)
    )
    return FusedRun(
        constraint_set=constraint_set,
        debug=debug,
        frames=frames,
        labels=fused_labels(debug, frames),
        body_names=next(iter(frames.values())).body_names,
        targets=targets,
        link_errors=link_pose_error(
            scene, filter_model, sorted({m for m, _ in CAMERA_MOUNTS.values()})
        ),
    )


def candidate_index(run: FusedRun) -> list[dict]:
    """The obstacle/collision index — one row per candidate of the **single fused scene**."""
    rows = []
    for candidate in sorted(
        run.constraint_set.candidates, key=lambda c: (-c.point_count, c.id)
    ):
        gt_name, purity = _purity(run.labels[candidate.point_indices], run.body_names)
        primitive = candidate.geometry[0] if candidate.geometry else None
        cloud = run.debug["fusion"].cloud
        cameras = set()
        for i in candidate.point_indices[:400]:  # a sample; the full set is thousands of points
            cameras |= cloud.cameras_of(int(i))
        rows.append(
            {
                "id": candidate.id,
                "source_type": candidate.source_type.value,
                "gt_body": gt_name,
                "gt_purity": round(purity, 4),
                "points": candidate.point_count,
                "seen_by": sorted(c.value for c in cameras),
                "centroid_base": [round(float(v), 4) for v in candidate.centroid],
                "radius_m": None if primitive is None else round(float(primitive.dimensions[0]), 4),
                "safety_margin_m": round(float(candidate.safety_margin), 4),
                "collision_enabled": bool(candidate.collision_enabled),
                "contact_permission": bool(candidate.contact_permission),
                "phase_rule": candidate.phase_rule,
                "track_age": candidate.track_age,
            }
        )
    return rows


def self_filter_report(run: FusedRun) -> dict:
    """How much of the robot the self-filter removed, per camera, and what it cost the scene."""
    out = {}
    for result in run.debug["fusion"].per_camera:
        frame = run.frames[result.camera_id]
        kept = pixel_labels(frame, result.cloud.uv)
        robot_after = sum(1 for label in kept if is_robot_body(frame.body_names.get(int(label))))
        out[result.camera_id.value] = {
            "points_kept": int(len(result.cloud)),
            "robot_points_removed": int(result.stats.get("n_self_filtered", 0)),
            "robot_points_remaining": int(robot_after),
        }
    return out


def target_report(run: FusedRun) -> dict:
    """Did fused grounding land on a body some camera's blob was aimed at — and which camera won?"""
    constraint_set = run.constraint_set
    out = {
        "requested": dict(run.targets),
        "grounding_status": constraint_set.grounding_status.value,
    }
    target = constraint_set.target
    if target is None:
        return out
    gt_name, purity = _purity(run.labels[target.point_indices], run.body_names)
    out.update(
        {
            "grounded_body": gt_name,
            "purity": round(purity, 4),
            "points": int(target.point_indices.size),
            "confidence": round(float(target.confidence), 4),
            "centroid_base": [round(float(v), 4) for v in target.centroid],
            "seed_camera": None if target.seed_camera is None else target.seed_camera.value,
            "supporting_cameras": [c.value for c in target.supporting_cameras],
        }
    )
    return out


def cross_view_identity(run: FusedRun) -> dict:
    """The measurement the rewrite is actually for: one candidate, many cameras.

    Under the per-camera pipeline a surface seen by all three views produced three candidates in
    three unrelated id spaces. The property that replaced it is `n_multi_camera_candidates`: a single
    candidate whose points came from more than one camera. That number was structurally zero before
    and is what proves the views were merged rather than concatenated.

    `candidates_per_body` is reported alongside and is **not** the same claim. A body split across
    several candidates is a clustering outcome, not a fusion failure — a crate is a hollow box whose
    walls are not Euclidean-connected across its opening, so any `eps` small enough to separate the
    fruit will separate the walls. The distinction matters because conflating them would make a
    clustering property look like a registration bug.
    """
    cloud = run.debug["fusion"].cloud
    by_body: dict[str, list[int]] = {}
    multi = 0
    camera_counts: dict[int, int] = {}
    for candidate in run.constraint_set.candidates:
        if candidate.source_type is SourceType.SUPPORT_SURFACE:
            continue
        cameras = set()
        for i in candidate.point_indices[:400]:
            cameras |= cloud.cameras_of(int(i))
        camera_counts[candidate.id] = len(cameras)
        multi += len(cameras) > 1
        if candidate.point_count >= 50:
            gt_name, purity = _purity(run.labels[candidate.point_indices], run.body_names)
            if purity >= 0.5:
                by_body.setdefault(gt_name, []).append(candidate.id)
    return {
        "n_candidates": len(camera_counts),
        "n_multi_camera_candidates": multi,
        "cameras_per_candidate": camera_counts,
        "candidates_per_body": {name: sorted(ids) for name, ids in sorted(by_body.items())},
        "bodies_with_one_candidate": sum(1 for ids in by_body.values() if len(ids) == 1),
        "bodies_split_by_clustering": sum(1 for ids in by_body.values() if len(ids) > 1),
    }


def format_report(run: FusedRun) -> str:
    constraint_set = run.constraint_set
    digest = summary(constraint_set)
    metrics = constraint_set.metrics
    lines: list[str] = ["=" * 100]
    lines.append(
        f"FUSED SCENE   cameras={', '.join(metrics.get('cameras', []))}   "
        f"phase={digest['phase']}   status={digest['status']}   validity={digest['validity']}"
    )
    lines.append("=" * 100)

    lines.append(
        f"  fusion      : {metrics['n_points_before_fusion']} points from "
        f"{metrics['n_cameras']} cameras -> {metrics['n_points_fused']} fused "
        f"({metrics['fusion_compression'] * 100:.1f}% kept) at "
        f"{metrics['fusion_voxel_size_m'] * 1000:.0f} mm; skew "
        f"{metrics['camera_skew_sec'] * 1000:.0f} ms"
    )
    per_camera = ", ".join(
        f"{name} {n}" for name, n in sorted(metrics["n_points_per_camera"].items())
    )
    lines.append(f"  per camera  : {per_camera}")

    sf = self_filter_report(run)
    for name, stats in sorted(sf.items()):
        lines.append(
            f"  self-filter : {name:<12} removed {stats['robot_points_removed']:6d} robot points, "
            f"{stats['robot_points_remaining']:5d} remaining in {stats['points_kept']} kept"
        )
    lines.append(
        "  FK check    : URDF vs MuJoCo link position, "
        + ", ".join(f"{k} {v * 1000:.1f} mm" for k, v in sorted(run.link_errors.items()))
    )

    tr = target_report(run)
    if "grounded_body" in tr:
        lines.append(
            f"  grounding   : {tr['grounded_body']} purity={tr['purity']:.4f} n={tr['points']} "
            f"conf={tr['confidence']:.3f} seed={tr['seed_camera']} "
            f"seen_by={tr['supporting_cameras']}"
        )
    else:
        lines.append(f"  grounding   : no target ({tr['grounding_status']})")
    lines.append(f"  asked for   : {tr['requested']}")

    for surface in constraint_set.support_surfaces:
        normal, offset = surface.to_halfspace()
        lines.append(
            f"  plane {surface.id}     : n={np.round(normal, 3).tolist()} "
            f"d={offset:.4f} m  inliers={surface.point_count}  rms={surface.rms_error:.2e} m"
        )

    lines.append("")
    lines.append(
        f"  {'id':>5} {'source_type':<17} {'gt body':<16} {'pur':>5} {'pts':>7} "
        f"{'r(m)':>7} {'margin':>7} {'touch':>6}  seen by"
    )
    lines.append(f"  {'-' * 96}")
    for row in candidate_index(run):
        radius = "-" if row["radius_m"] is None else f"{row['radius_m']:7.4f}"
        lines.append(
            f"  {row['id']:>5} {row['source_type']:<17} {row['gt_body']:<16} "
            f"{row['gt_purity']:5.2f} {row['points']:7d} {radius:>7} "
            f"{row['safety_margin_m']:7.3f} {str(row['contact_permission']):>6.5}  "
            f"{','.join(row['seen_by'])}"
        )

    identity = cross_view_identity(run)
    lines.append("")
    lines.append(
        f"  fusion gain : {identity['n_multi_camera_candidates']}/{identity['n_candidates']} "
        "candidates carry points from more than one camera — structurally impossible before, when "
        "each view had its own id space"
    )
    lines.append(
        f"  clustering  : {identity['bodies_with_one_candidate']} bodies map to one candidate, "
        f"{identity['bodies_split_by_clustering']} are split by Euclidean clustering "
        "(a hollow crate's walls are not connected across its opening; not a fusion problem)"
    )
    for name, ids in identity["candidates_per_body"].items():
        if len(ids) > 1:
            lines.append(f"                {name} -> candidates {ids}")

    clearance = {k[10:]: v for k, v in metrics.items() if k.startswith("clearance_")}
    if clearance:
        lines.append(
            f"  clearance   : phase={clearance.get('phase')} "
            f"manipulators={clearance.get('active_manipulators')} "
            f"authorized={clearance.get('authorized_links')} "
            f"target margin {clearance.get('target_margin_authorized')} on authorized links vs "
            f"{clearance.get('target_margin_other')} elsewhere"
        )
    lines.append(
        f"  constraints : {digest['n_constraints']} rows, "
        f"{digest['n_active_slots']}/{digest['max_candidates']} slots active, "
        f"horizon={digest['horizon']}, squared={digest['squared_distance']}"
    )
    for note in constraint_set.notes:
        lines.append(f"  note        : {note}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AG3S on the RB-Y1 transport scene, fused")
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--phase", default="approach")
    parser.add_argument("--config", default=None, help="YAML config; defaults to the tuned preset")
    parser.add_argument(
        "--images",
        default="benchmark/ag3s/asset/image/rby1_transport",
        help="directory for the stage dumps ('none' to skip)",
    )
    parser.add_argument("--json", default=None, help="write the candidate index here")
    args = parser.parse_args(argv)

    config = (
        AG3SConfig.from_yaml(args.config) if args.config else AG3SConfig()
    ).with_overrides(SCENE_OVERRIDES)

    scene = TransportScene()
    urdf = parse_urdf(RBY1_URDF)
    extra = gap_filling_capsules(scene.model)
    # The head camera hangs off `link_head_2`, whose pose depends on two joints `DEFAULT_RBY1_JOINTS`
    # leaves out of `q` (they carry no collision capsule). Pinning them at the scene's actual values
    # is what keeps the head camera's FK from tracking a head that is always level.
    head_values = {
        name: float(scene.data.qpos[scene._qadr[name]]) for name in HEAD_JOINTS
        if name in scene._qadr
    }
    robot_models = {
        "filter": UrdfSphereChain(
            urdf, extra_capsules=extra, fixed_joint_values=head_values
        ),
        # Fingertip capsules injected deliberately. RB-Y1's URDF carries collision geometry for the
        # torso and arm links 0-5 only, so a constraint model built from it alone has no sphere on
        # any contact-authorized link — and a clearance policy with nothing to relax cannot express a
        # grasp at all. See `tests/ag3s/test_robot_models.py`, which pins that fact.
        #
        # `ee_finger_` and not `ee_`: since 2026-09-25 `UNCOVERED_LINKS` also carries the gripper
        # *palm* (`ee_left`/`ee_right`), which is there for the self-filter — the wrist cameras see
        # it — and is not a contact-authorized link. A bare `ee_` prefix would have quietly moved
        # palm spheres into the constraint set along with it.
        "constraint": UrdfSphereChain(
            urdf,
            extra_capsules=[c for c in extra if c.link.startswith("ee_finger_")],
            fixed_joint_values=head_values,
        ),
    }
    print(
        f"robot models: self-filter {robot_models['filter'].n_spheres} spheres "
        f"(URDF {len(urdf.capsules)} capsules + {len(extra)} gap-filling), "
        f"constraints {robot_models['constraint'].n_spheres} spheres "
        f"(contact links: "
        f"{sorted(set(robot_models['constraint'].sphere_link_names) & {'ee_finger_r1', 'ee_finger_r2', 'ee_finger_l1', 'ee_finger_l2'})})\n"
    )

    run = run_fused(
        scene, args.cameras, config=config, phase=args.phase, robot_models=robot_models
    )
    print(format_report(run))

    if args.images and args.images.lower() != "none":
        from benchmark.ag3s.runtime.visualization import dump_from_pipeline

        out = pathlib.Path(args.images)
        written = dump_from_pipeline(out, None, run.constraint_set, run.debug, prefix="fused")
        print(f"\n  {len(written)} figures -> {out}")

    if args.json:
        payload = {
            "fused": {
                "target": target_report(run),
                "self_filter": self_filter_report(run),
                "identity": cross_view_identity(run),
                "link_pose_error_m": run.link_errors,
                "summary": summary(run.constraint_set),
                "candidates": candidate_index(run),
            }
        }
        pathlib.Path(args.json).write_text(json.dumps(payload, indent=2, default=str))
        print(f"  candidate index -> {args.json}")

    scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FusedRun",
    "candidate_index",
    "cross_view_identity",
    "format_report",
    "main",
    "run_fused",
]
