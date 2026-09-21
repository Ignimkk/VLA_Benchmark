"""새 MuJoCo 씬을 캡처해 AG3S와 cuRobo의 한 회 테스트 입력을 만든다.

기존 ``run_XXXX``나 attention dump를 읽지 않는다. 프로세스를 시작할 때 transport 씬을 새로
초기화하고 세 카메라를 현재 시뮬레이션 시각에서 캡처한다. attention은 정책 검증용이 아니라
attention 이후 AG3S 기능을 분리해 확인하는 합성 Gaussian이며, 출력에 그 사실을 기록한다.

이 스크립트는 ``.venv-ag3s``에서 실행한다. 출력은 같은 테스트 세션의 다음 단계가
``.venv-curobo``에서 즉시 소비하는 프로세스 경계 자료다. 과거 실행의 replay 입력이 아니다.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
IDS = ("head", "left_wrist", "right_wrist")


def _render_rgb(scene, camera: str) -> np.ndarray:
    renderer = scene._renderer_for()
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    renderer.update_scene(scene.data, camera=camera)
    return np.ascontiguousarray(renderer.render()).astype(np.uint8)


def _jitter_target(scene, target: str, seed: int) -> np.ndarray:
    """대상을 테이블 위에서 작게 옮겨 매 실행의 씬을 명시적으로 새로 만든다."""
    joint = f"{target}_free"
    if joint not in scene._qadr:
        raise KeyError(f"target {target!r} has no free joint {joint!r}")
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-0.012, 0.012, size=2)
    qadr = scene._qadr[joint]
    scene.data.qpos[qadr:qadr + 2] += delta
    scene.mujoco.mj_forward(scene.model, scene.data)
    return delta


def main() -> None:
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS,
        build_constraint_robot_model,
        build_robot_model,
    )
    from benchmark.ag3s.experiments.sources.mujoco_source import (
        TRANSPORT_MODEL,
        TransportScene,
        gaussian_attention,
    )
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.ag3s.types import CameraID, CameraObservation

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="현재 세션의 frame.npz")
    ap.add_argument("--summary", required=True, help="현재 세션의 AG3S JSON 요약")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    output = pathlib.Path(args.out)
    summary_path = pathlib.Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    scene = TransportScene(
        TRANSPORT_MODEL,
        settle_steps=400,
        height=args.height,
        width=args.width,
    )
    try:
        delta = _jitter_target(scene, args.target, args.seed)
        target_true = scene.body_position_in_base(args.target)
        filter_robot = build_robot_model(scene)
        constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)

        frames = {camera: scene.capture(camera) for camera in CAMS}
        rgbs = {camera: _render_rgb(scene, camera) for camera in CAMS}
        captured_at = time.monotonic()
        camera_ids = {
            "zed_left": CameraID.HEAD,
            "wrist_cam_l": CameraID.LEFT_WRIST,
            "wrist_cam_r": CameraID.RIGHT_WRIST,
        }
        observations = []
        for camera in CAMS:
            frame = frames[camera]
            observations.append(CameraObservation(
                camera_id=camera_ids[camera],
                depth=frame.depth,
                camera_intrinsics=frame.camera_intrinsics,
                T_base_cam=frame.T_base_cam,
                robot_state=frame.robot_state,
                timestamp=captured_at,
                state_timestamp=captured_at,
                attention_map=gaussian_attention(frame, target_true),
                image_hw=frame.hw,
            ))

        # T1은 AG3S 앞단만 시험한다. legacy EsdfBuilder가 조용히 끼어들지 않도록 primitive
        # backend로 실행하고, 거리장은 다음 프로세스에서 실제 cuRobo Mapper가 만든다.
        config = AG3SConfig.from_dict({
            "collision_backend": "primitive",
            "pointcloud": {"range_max": 2.0},
        })
        ag3s = AG3S(
            config,
            robot_model=filter_robot,
            constraint_robot_model=constraint_robot,
        )
        constraint_set, debug = ag3s.process_multi_debug(
            observations,
            phase="approach",
            active_manipulators=["left"],
        )
        if ag3s._esdf_builder is not None:
            raise RuntimeError("fresh T1 unexpectedly instantiated legacy EsdfBuilder")
        if constraint_set.target is None:
            raise RuntimeError(
                f"fresh scene target grounding failed: status={constraint_set.status.value}, "
                f"grounding={constraint_set.grounding_status.value}, notes={constraint_set.notes}"
            )

        q = np.asarray(frames[CAMS[0]].robot_state, np.float64)
        centres, radii = constraint_robot.sphere_centers_numeric(q)
        target_est = np.asarray(constraint_set.target.centroid, np.float64)
        payload: dict[str, np.ndarray] = {
            "source_is_fresh": np.asarray(True),
            "synthetic_attention": np.asarray(True),
            "seed": np.asarray(args.seed, np.int64),
            "model_path": np.asarray(str(TRANSPORT_MODEL)),
            "qpos": np.asarray(scene.data.qpos, np.float64).copy(),
            "target_true": target_true,
            "target_centroid": target_est,
            "target_jitter_xy": delta,
            "sphere_centers": np.asarray(centres, np.float64).reshape(-1, 3),
            "sphere_radii": np.asarray(radii, np.float64).reshape(-1),
            "sphere_link_names": np.asarray(constraint_robot.sphere_link_names),
            "target_body_id": np.asarray(scene.mujoco.mj_name2id(
                scene.model, scene.mujoco.mjtObj.mjOBJ_BODY, args.target), np.int32),
            "body_names_json": np.asarray(json.dumps(frames[CAMS[0]].body_names)),
        }

        mask_stats = {}
        for camera, cid in zip(CAMS, IDS):
            frame = frames[camera]
            depth = np.asarray(frame.depth, np.float32)
            intrinsics = np.asarray(frame.camera_intrinsics, np.float32)
            transform = np.asarray(frame.T_base_cam, np.float32)
            mask = ag3s._robot_mask_for(
                depth.astype(np.float64),
                intrinsics.astype(np.float64),
                transform.astype(np.float64),
                frame.robot_state,
            )
            if mask is None:
                raise RuntimeError(f"{cid}: robot mask was not produced")
            payload[f"rgb_{cid}"] = rgbs[camera]
            payload[f"depth_{cid}"] = depth
            payload[f"depth_masked_{cid}"] = np.where(mask, np.float32(0.0), depth)
            payload[f"robot_mask_{cid}"] = np.asarray(mask, bool)
            payload[f"attention_{cid}"] = np.asarray(observations[len(mask_stats)].attention_map)
            payload[f"K_{cid}"] = intrinsics
            payload[f"T_{cid}"] = transform
            payload[f"body_ids_{cid}"] = np.asarray(frame.body_ids, np.int32)
            if frame.geom_ids is not None:
                payload[f"geom_ids_{cid}"] = np.asarray(frame.geom_ids, np.int32)
            mask_stats[cid] = {
                "pixels": int(mask.sum()),
                "fraction": float(mask.mean()),
            }

        for name in ("raw_cloud", "filtered_cloud"):
            cloud = debug.get(name)
            if cloud is not None:
                payload[name] = np.asarray(cloud.points, np.float32)
        attention_cloud = debug.get("attention_cloud")
        if attention_cloud is not None:
            payload["attention_cloud"] = np.asarray(attention_cloud.points, np.float32)
            payload["attention_values"] = np.asarray(attention_cloud.attention, np.float32)
        payload["target_points"] = np.asarray(constraint_set.target.points, np.float32)

        np.savez_compressed(output, **payload)
        summary = {
            "source": "fresh_mujoco_scene",
            "used_saved_run": False,
            "synthetic_attention": True,
            "seed": args.seed,
            "target": args.target,
            "target_jitter_xy_m": delta.tolist(),
            "target_true_m": target_true.tolist(),
            "target_estimated_m": target_est.tolist(),
            "target_centroid_error_mm": float(np.linalg.norm(target_est - target_true) * 1000.0),
            "pipeline_status": constraint_set.status.value,
            "grounding_status": constraint_set.grounding_status.value,
            "geometry_validity": constraint_set.validity.value,
            "n_target_points": int(len(constraint_set.target.points)),
            "n_candidates": int(len(constraint_set.candidates)),
            "notes": list(constraint_set.notes),
            "metrics": {
                key: (value.item() if isinstance(value, np.generic) else value)
                for key, value in constraint_set.metrics.items()
                if isinstance(value, (str, int, float, bool, np.generic)) or value is None
            },
            "robot_mask": mask_stats,
            "profile_ms": {k: float(v) for k, v in constraint_set.profile.items()},
            "legacy_esdf_builder_created": False,
            "elapsed_ms": float((time.perf_counter() - started) * 1000.0),
            "frame_npz": str(output),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

        print(f"wrote {output}")
        print(f"wrote {summary_path}")
        print(f"  source fresh={bool(payload['source_is_fresh'])}, saved run used=False")
        print(f"  target true {np.round(target_true, 4)}")
        print(f"  target AG3S {np.round(target_est, 4)}")
        print(f"  centroid error {summary['target_centroid_error_mm']:.2f} mm")
        print(f"  status {constraint_set.status.value}/{constraint_set.grounding_status.value}")
        for cid in IDS:
            stat = mask_stats[cid]
            print(f"  mask {cid:<12} {stat['pixels']:>7,} px ({100.0*stat['fraction']:.2f} %)")
    finally:
        scene.close()


if __name__ == "__main__":
    main()
