"""cuRobo 검증용 입력 프레임을 npz 로 내보낸다. **openpi venv 에서** 돌린다.

curobo 는 격리된 venv 에 있어서 우리 benchmark 코드를 임포트할 수 없다. 그래서 MuJoCo 재생과
AG3S 실행은 이쪽에서 하고, 결과만 npz 로 넘긴다.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \\
        /mnt/dev/work/.venv-openpi-live/bin/python -m \\
        benchmark.ag3s.experiments.curobo.export_frame \\
        --records outputs/live_test/20260924_long16d/run_0000 --step 9 \\
        --out outputs/verify/R/rby1_frame_16d.npz

`--records` 와 `--out` 에 기본값을 두지 않는다. 예전 기본값(`run_0004`, `/tmp/rby1_frame.npz`)
때문에 16D 재측정이 14D 프레임을 말없이 다시 읽었다 (2026-09-25).
"""
import argparse
import pathlib

import numpy as np

CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
IDS = ("head", "left_wrist", "right_wrist")


def main() -> None:
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import gaussian_attention
    from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
    from benchmark.ag3s.runtime.pipeline import AG3S

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True,
                    help="어느 기록에서 프레임을 뽑을지. 기본값 없음 (위 머리말 참고)")
    ap.add_argument("--step", type=int, default=9)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--out", required=True,
                    help="쓸 npz 경로. 기본값 없음. /tmp 는 컨테이너 재시작에 날아가므로 "
                         "재측정본은 outputs/ 아래에 둔다")
    args = ap.parse_args()

    run = load_run(args.records)
    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    pose_scene(scene, run.steps[args.step])
    frames = {c: scene.capture(c) for c in CAMS}
    head = frames["zed_left"]

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": 0.02, "max_distance": 0.4},
    })
    ag = AG3S(cfg, robot_model=filter_robot, constraint_robot_model=constraint_robot)
    cs = ag.process(
        depth=head.depth, camera_intrinsics=head.camera_intrinsics,
        T_base_cam=head.T_base_cam,
        attention_map=gaussian_attention(head, scene.body_position_in_base(args.target)),
        robot_state=head.robot_state, phase="grasp", active_manipulators=["right"])
    if cs.target is None:
        raise SystemExit(f"target {args.target!r} 을 잡지 못했다 — 다른 step 을 고르라")

    q = np.asarray(head.robot_state, np.float64)
    centres, radii = constraint_robot.sphere_centers_numeric(q)
    out = {
        "target_centroid": np.asarray(cs.target.centroid, np.float64),
        "sphere_centers": np.asarray(centres, np.float64).reshape(-1, 3),
        "sphere_radii": np.asarray(radii, np.float64).reshape(-1),
    }
    n_masked = {}
    for cam, cid in zip(CAMS, IDS):
        f = frames[cam]
        depth = np.asarray(f.depth, np.float32)
        K = np.asarray(f.camera_intrinsics, np.float32)
        T = np.asarray(f.T_base_cam, np.float32)
        # 로봇 자기 관측 제거. AG3S 가 자기 ESDF 에 하는 것과 **같은 계산**을 그대로 부른다
        # (`pipeline.py:_robot_mask_for`). 이것 없이 넣으면 팔이 장애물로 적분되어
        # 구 120 개 중 105 개가 자기 자신과 충돌한다고 나온다 — AG3S 쪽 주석이 적어둔
        # "130 of 194 spheres in violation" 과 같은 현상이다.
        # 마스크된 픽셀은 0.0 으로 둔다. 그러면 우리 TSDF 도 (`esdf.py:146`) cuRobo 도
        # (`builder_camera_integrate.py:146`, depth < depth_min 이면 return) 그 광선을
        # 통째로 건너뛴다 — 즉 **자유가 아니라 미관측**이 된다.
        mask = ag._robot_mask_for(depth.astype(np.float64), K.astype(np.float64),
                                  T.astype(np.float64), f.robot_state)
        if mask is None:
            raise SystemExit(f"{cid}: 로봇 마스크를 만들지 못했다 — self_filter 설정을 확인하라")
        out[f"depth_{cid}"] = depth
        out[f"depth_masked_{cid}"] = np.where(mask, np.float32(0.0), depth)
        out[f"robot_mask_{cid}"] = np.asarray(mask, bool)
        out[f"K_{cid}"] = K
        out[f"T_{cid}"] = T
        n_masked[cid] = int(mask.sum())

    # 출처 도장 — 이 프레임이 어느 기록에서 나왔는지 파일이 스스로 말하게 한다.
    # verify_two_tier.py 가 이것을 읽어 찍는다.
    from benchmark.ag3s.experiments.sources.policy_record import provenance_arrays
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **provenance_arrays(run, 1), **out)
    print(f"wrote {args.out}")
    print(f"  target centroid {np.round(out['target_centroid'], 3)}")
    print(f"  제약 구 {len(out['sphere_radii'])}  카메라 {len(CAMS)}")
    for cid, n in n_masked.items():
        tot = out[f"robot_mask_{cid}"].size
        print(f"  로봇 마스크 {cid:<12} {n:>7,} / {tot:,} px ({100.0*n/tot:.2f} %)")


if __name__ == "__main__":
    main()
