"""Three cameras, one scene.

RB-Y1 carries a ZED on its head and a D435i on each wrist. The head sees the workspace; the wrists
see what the head cannot, which on a robot reaching into a crate is most of what matters. Running
them independently — one `AG3S` per camera, as `experiments/rby1_transport.py` originally did —
produces three unrelated answers: the same crate gets a different candidate id in each view, and
`id=0` means the crate in one and the table in another. Nothing downstream can consume that.

Fusion happens **after** per-camera reconstruction and self-filtering and **before** everything else.
That ordering is forced by two facts:

* Self-filtering needs the camera's own extrinsics and the robot state *at that camera's capture
  instant*. It has to happen per camera, before the clouds are merged and provenance is what tells
  them apart.
* Support-surface fitting, grounding, candidate generation and constraint building want one scene.
  Running them per camera and merging the *results* would mean reconciling three sets of clusters,
  three plane fits and three target hypotheses — a harder problem than merging points, and one whose
  failure modes are much less obvious.

**Capture-time kinematics.** Each observation is placed with its own `q`. A wrist camera moving at
1 rad/s and a 50 ms state lag put the cloud 2.5 cm from where it belongs — enough to smear a crate
edge and, worse, to misalign the self-filter so that the arm's own points survive and cluster into a
phantom obstacle attached to the gripper. `CameraObservation.resolve_T_base_cam` composes
`FK(q_capture, mount_link) @ T_link_cam`; there is deliberately no `q_now` parameter anywhere in this
module to fall back on.

**Provenance survives.** A fused point keeps every observation that produced it — camera, pixel,
timestamp, attention — in a CSR block. That is what makes the attention `max` meaningful (it is a max
over *identified* cameras, not over an anonymous pile) and what lets a candidate be traced back to
the view it came from when something goes wrong.

**Degradation is reported, not handled.** A dropped camera, a stale transform, a skewed frame: each
lowers `ConstraintValidity` and lands in the metrics. What to do about it — stop, hold, re-use the
last certified scene — belongs to the caller. A partially observed scene must never be indistinguishable
from a fully observed empty one.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

import numpy as np

from benchmark.ag3s.stages.attention_lifting import lift, make_adapter, normalize_attention
from benchmark.ag3s.config import AG3SConfig, AttentionConfig
from benchmark.ag3s.runtime.degradation import reason
from benchmark.ag3s.stages.reconstruction import reconstruct
from benchmark.ag3s.stages.robot_filter import filter_robot_points
from benchmark.ag3s.types import (
    CameraID,
    CameraObservation,
    ConstraintValidity,
    FusedPointCloud,
    PointCloud,
    RobotCollisionModel,
)


@dataclasses.dataclass
class CameraResult:
    """One camera's contribution, after reconstruction, self-filtering and attention lifting."""

    camera_id: CameraID
    cloud: PointCloud  # base frame, robot points removed
    attention: np.ndarray  # (N,) normalized; zeros when this camera had no attention map
    raw_attention: np.ndarray  # (N,) pre-normalization, for unambiguous peak finding
    timestamp: float
    T_base_cam: np.ndarray
    had_attention: bool
    stats: dict[str, Any] = dataclasses.field(default_factory=dict)
    notes: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class FusionResult:
    """The fused scene plus everything a caller needs to judge how much to trust it."""

    cloud: FusedPointCloud
    attention: np.ndarray  # (N,) fused per point
    raw_attention: np.ndarray  # (N,) fused pre-normalization signal
    per_camera: list[CameraResult]
    validity: ConstraintValidity
    notes: list[str]
    metrics: dict[str, Any]

    @property
    def pointcloud(self) -> PointCloud:
        return self.cloud.as_pointcloud()


# ------------------------------------------------------------------------- per-camera stage


def process_observation(
    observation: CameraObservation,
    config: AG3SConfig,
    *,
    robot_model: Optional[RobotCollisionModel] = None,
    attention_adapter: Any = None,
) -> CameraResult:
    """Reconstruct, self-filter and lift attention for one camera, in its own capture instant.

    The self-filter is given `observation.robot_state`, not some global current state. On a wrist
    camera those are different poses, and using the wrong one deletes free space while leaving the
    arm behind — the two failure directions at once.
    """
    T_base_cam = observation.resolve_T_base_cam(robot_model)
    cloud, recon_stats = reconstruct(
        depth=observation.depth,
        pointcloud=observation.pointcloud,
        camera_intrinsics=observation.camera_intrinsics,
        T_base_cam=T_base_cam,
        uv=observation.uv,
        config=config.pointcloud,
        frame_id=config.frame_id,
    )
    cloud, filter_stats = filter_robot_points(
        cloud, robot_model, observation.robot_state, config.pointcloud
    )

    notes: list[str] = []
    if observation.has_attention and not cloud.is_empty:
        adapter = attention_adapter or make_adapter(observation.attention_map, config.attention)
        # `lift()` normalizes over *this camera's* points, so the value it returns is only
        # comparable within this camera. **That is why `fuse` re-normalizes the fused raw values
        # instead of using this array** (F2, `AG3S_REVIEW_LOG.md` Step 5) — otherwise the fusion
        # `max` picks the most generously scaled camera rather than the strongest evidence.
        # The per-camera array is still carried in the CSR block as provenance.
        attention_cloud = lift(
            cloud,
            observation.attention_map,
            config.attention,
            adapter=adapter,
            image_hw=observation.image_hw,
            camera_intrinsics=observation.camera_intrinsics,
            T_base_cam=T_base_cam,
        )
        attention = np.asarray(attention_cloud.attention, np.float32)
        # Both signals are carried through fusion for the reason `AttentionPointCloud` documents:
        # percentile clipping makes the normalized map comparable and, in doing so, creates hundreds
        # of exact ties that ruin `argmax`. Thresholds read the normalized array; the peak reads the
        # raw one. Fusing only the normalized array would reintroduce the tie at the fused level.
        raw = attention_cloud.raw_attention
        raw_attention = attention if raw is None else np.asarray(raw, np.float32)
    else:
        # No attention map is not an error and not a reason to drop geometry. Zero is "no opinion",
        # which under the `max` aggregation neither helps nor hurts this camera's points.
        attention = np.zeros(len(cloud), np.float32)
        raw_attention = attention
        if not observation.has_attention:
            notes.append(f"{observation.camera_id.value}: no attention map; geometry retained")

    return CameraResult(
        camera_id=observation.camera_id,
        cloud=cloud,
        attention=attention,
        raw_attention=raw_attention,
        timestamp=float(observation.timestamp),
        T_base_cam=T_base_cam,
        had_attention=observation.has_attention,
        stats={**recon_stats, "n_self_filtered": filter_stats["n_removed"]},
        notes=notes,
    )


# ------------------------------------------------------------------------------ freshness


def check_freshness(
    observations: Sequence[CameraObservation], config: AG3SConfig, *, now: Optional[float] = None
) -> tuple[ConstraintValidity, list[str], dict[str, Any]]:
    """Compare each observation's ages against the configured limits.

    The reference instant is the newest capture rather than wall-clock `time.time()`, so replaying a
    recorded episode does not report every frame as hours stale. Pass `now` explicitly for a live
    system where "how old is the newest camera" is itself the question.
    """
    timing = config.timing
    if not observations:
        return ConstraintValidity.INCOMPLETE, ["no camera observations"], {"n_cameras": 0}

    times = np.asarray([obs.timestamp for obs in observations], np.float64)
    reference = float(times.max()) if now is None else float(now)
    skew = float(times.max() - times.min())

    validity = ConstraintValidity.VALID
    notes: list[str] = []
    stale_state = stale_transform = 0

    for obs in observations:
        state_age = abs(obs.timestamp - obs.state_time)
        if state_age > timing.max_state_age_sec:
            stale_state += 1
            notes.append(reason(
                "camera_state_stale",
                f"{obs.camera_id.value}: robot state is {state_age * 1000:.0f} ms from the image "
                f"(limit {timing.max_state_age_sec * 1000:.0f} ms); its cloud may be mis-placed",
            ))
        transform_age = abs(reference - obs.timestamp)
        if transform_age > timing.max_transform_age_sec:
            stale_transform += 1
            notes.append(reason(
                "camera_transform_stale",
                f"{obs.camera_id.value}: image is {transform_age * 1000:.0f} ms behind the frame "
                f"(limit {timing.max_transform_age_sec * 1000:.0f} ms)",
            ))

    if skew > timing.max_camera_skew_sec:
        notes.append(reason(
            "camera_skew",
            f"cameras span {skew * 1000:.0f} ms (limit {timing.max_camera_skew_sec * 1000:.0f} ms); "
            "the views may disagree about a scene that moved between them",
        ))
    if stale_state or stale_transform or skew > timing.max_camera_skew_sec:
        # Degraded, not incomplete. The geometry is all here; some of it is in the wrong place, and
        # the caller has to decide whether that is tolerable for what it is doing.
        validity = ConstraintValidity.DEGRADED

    present = {obs.camera_id for obs in observations}
    expected = {CameraID.parse(c) for c in timing.expected_cameras}
    missing = sorted(c.value for c in expected - present)
    if missing:
        notes.append(reason(
            "camera_missing",
            f"camera(s) {missing} did not report; the scene is only partially observed"))
        validity = ConstraintValidity.worst(validity, ConstraintValidity.DEGRADED)

    return validity, notes, {
        "n_cameras": len(observations),
        "cameras": sorted(c.value for c in present),
        "missing_cameras": missing,
        "camera_skew_sec": skew,
        "n_stale_state": stale_state,
        "n_stale_transform": stale_transform,
    }


# --------------------------------------------------------------------------------- fusion


def fuse(
    results: Sequence[CameraResult],
    *,
    voxel_size: float,
    frame_id: str = "base",
    attention_config: Optional[AttentionConfig] = None,
) -> tuple[FusedPointCloud, np.ndarray, np.ndarray]:
    """Merge per-camera clouds on a base-frame voxel grid. Returns `(fused, attention)`.

    One representative per occupied voxel, and it is **an observed point**, not the group's centroid.
    A centroid belongs to no pixel, so it has no attention value and no camera to attribute it to;
    synthesising one at the moment several cameras finally agree would throw away the correspondence
    that agreement represents. The representative is the first observation in input order, which
    makes the whole function a deterministic function of its input — an ablation rerun compares like
    with like.

    Every observation that landed in the voxel is kept in the CSR block, so the point knows it was
    seen twice and by whom.
    """
    clouds = [r for r in results if not r.cloud.is_empty]
    if not clouds:
        empty = np.zeros(0, np.float32)
        return FusedPointCloud.empty(frame_id), empty, empty

    cameras = tuple(r.camera_id for r in results)
    camera_index = {r.camera_id: i for i, r in enumerate(results)}

    points = np.vstack([r.cloud.points for r in clouds])
    obs_camera = np.concatenate([
        np.full(len(r.cloud), camera_index[r.camera_id], np.int16) for r in clouds
    ])
    obs_uv = np.vstack([
        r.cloud.uv if r.cloud.uv is not None else np.full((len(r.cloud), 2), -1, np.int32)
        for r in clouds
    ]).astype(np.int32)
    obs_time = np.concatenate([np.full(len(r.cloud), r.timestamp, np.float64) for r in clouds])
    obs_attention = np.concatenate([r.attention for r in clouds]).astype(np.float32)
    obs_raw = np.concatenate([r.raw_attention for r in clouds]).astype(np.float32)

    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    keys -= keys.min(axis=0)
    extent = keys.max(axis=0) + 1
    if float(extent[0]) * float(extent[1]) * float(extent[2]) >= 2.0**62:
        raise ValueError(
            f"fusion voxel grid {tuple(extent)} overflows int64; "
            f"timing.fusion_voxel_size={voxel_size} is too small for this scene"
        )
    packed = (keys[:, 0] * extent[1] + keys[:, 1]) * extent[2] + keys[:, 2]

    # Sorting by (voxel, input order) gives contiguous observation blocks, which is exactly the CSR
    # layout, and makes "first in input order" the representative for free.
    order = np.argsort(packed, kind="stable")
    sorted_keys = packed[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    offsets = np.r_[starts, sorted_keys.size].astype(np.int64)

    representative = starts.astype(np.int64)  # index into the *sorted* observation table
    fused = FusedPointCloud(
        points=points[order][representative],
        obs_offset=offsets,
        obs_camera=obs_camera[order],
        obs_uv=obs_uv[order],
        obs_time=obs_time[order],
        obs_attention=obs_attention[order],
        representative=representative,
        cameras=cameras,
        frame_id=frame_id,
    )

    # Vectorized max per block: `np.maximum.reduceat` over the CSR boundaries, which beats a Python
    # loop over 60k points by two orders of magnitude and gives the identical answer.
    raw_attention = np.maximum.reduceat(obs_raw[order], starts).astype(np.float32)

    # **F2 — normalization happens here, after fusion, not per camera** (`AG3S_REVIEW_LOG.md`
    # Step 5). `lift()` normalizes over the points of *one* camera, so each camera stretched its
    # own maximum to 1.0 and this `max` was choosing the most generously scaled camera rather than
    # the strongest evidence — exactly what the comment in `process_observation` warns about.
    # Measured on RB-Y1 `run_0004`: the normalization ceiling differed 2.7x across cameras, rank
    # correlation between the two schemes was 0.693, and the grounded target changed in 4 of 15
    # frames (three of them a different object entirely, up to 307.5 mm).
    #
    # Normalizing the fused *raw* values gives every camera one scale. Order is preserved by
    # `normalize_attention`, and a monotone map commutes with `max`, so this is the same ranking as
    # normalizing every camera against a shared lo/hi before fusion — with no change to `lift()`.
    # With a single camera the two schemes coincide by definition, so nothing on that path moves.
    if attention_config is not None:
        attention = normalize_attention(raw_attention, attention_config).astype(np.float32)
    else:
        # No config: fall back to the per-camera values. Kept so `fuse` stays callable on its own
        # (tests, ablations) rather than silently producing an unnormalized array.
        attention = np.maximum.reduceat(fused.obs_attention, starts).astype(np.float32)
    return fused, attention, raw_attention


def fuse_observations(
    observations: Sequence[CameraObservation],
    config: AG3SConfig,
    *,
    robot_model: Optional[RobotCollisionModel] = None,
    attention_adapter: Any = None,
    now: Optional[float] = None,
) -> FusionResult:
    """The whole multi-camera front end: per-camera processing, freshness checks, fusion."""
    validity, notes, metrics = check_freshness(observations, config, now=now)
    results = [
        process_observation(
            obs, config, robot_model=robot_model, attention_adapter=attention_adapter
        )
        for obs in observations
    ]
    for result in results:
        notes.extend(result.notes)

    fused, attention, raw_attention = fuse(
        results, voxel_size=config.timing.fusion_voxel_size, frame_id=config.frame_id,
        attention_config=config.attention,
    )

    n_input = sum(len(r.cloud) for r in results)
    metrics.update({
        "n_points_per_camera": {r.camera_id.value: len(r.cloud) for r in results},
        "n_points_before_fusion": n_input,
        "n_points_fused": len(fused),
        "n_observations": fused.n_observations,
        "fusion_voxel_size_m": float(config.timing.fusion_voxel_size),
        # How much the cameras actually overlap. Near 1.0 means they are looking at disjoint parts of
        # the scene, which is not wrong but is worth knowing before trusting a cross-view agreement.
        "fusion_compression": (float(len(fused)) / n_input) if n_input else 1.0,
        "cameras_with_attention": sorted(
            r.camera_id.value for r in results if r.had_attention
        ),
        "n_self_filtered": sum(int(r.stats.get("n_self_filtered", 0)) for r in results),
    })

    if any(not r.stats.get("covered", True) for r in results):
        notes.append(
            "at least one camera could not be reduced to max_points while preserving coverage"
        )
        validity = ConstraintValidity.worst(validity, ConstraintValidity.INCOMPLETE)
    elif any(r.stats.get("capped", False) for r in results):
        # **2026-09-25 까지 이 분기에만 노트가 없었다.** `validity` 만 내리고 조용히 지나가서,
        # 로컬은 `degraded` 를 받고도 어느 단계가 원인인지 알 길이 없었다 (첫 live smoke 에서
        # 2 청크 중 1 개가 그렇게 HOLD 로 갔다). 어느 카메라가 몇 점에서 걸렸는지까지 적는다 —
        # 사유에 수치가 없으면 다음 사람이 다시 서버에 들어가서 재야 한다.
        capped = {r.camera_id.value: (int(r.stats.get("n_voxel", 0)),
                                      float(r.stats.get("final_voxel_size", 0.0)))
                  for r in results if r.stats.get("capped", False)}
        notes.append(reason(
            "fused_pointcloud_capped",
            "camera(s) "
            + ", ".join(f"{cam} ({n} pts -> voxel {v * 1000:.1f} mm)"
                        for cam, (n, v) in sorted(capped.items()))
            + f" hit pointcloud.max_points={config.pointcloud.max_points} and had their voxel "
              "grown; the fused cloud is coarser than configured (nothing was dropped — the "
              "path that drops points reports INCOMPLETE instead)"))
        validity = ConstraintValidity.worst(validity, ConstraintValidity.DEGRADED)

    return FusionResult(
        cloud=fused,
        attention=attention,
        raw_attention=raw_attention,
        per_camera=results,
        validity=validity,
        notes=notes,
        metrics=metrics,
    )


__all__ = [
    "CameraResult",
    "FusionResult",
    "check_freshness",
    "fuse",
    "fuse_observations",
    "process_observation",
]
