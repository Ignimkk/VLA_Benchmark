"""서버 단독 replay — 기록된 MuJoCo 관측으로 `SafePolicy` 경로 전체를 돌린다.

로컬을 붙이기 **전에** 서버 쪽만 검증하는 단계다. 둘을 동시에 켜고 문제가 나면 그것이 지각인지
최적화인지 네트워크인지 알 수 없다. 여기서는 네트워크가 없고 정책도 기록된 청크로 대신하므로,
남는 것은 AG3S+TO 와 배선뿐이다.

    MUJOCO_GL=osmesa python -m benchmark.trajopt.experiments.safe_replay \\
        --records outputs/.../ag3s_records/run_0002 --frames 10

`--checkpoint` 를 주면 진짜 π0.5 를 쓴다 (GPU 서버에서). 안 주면 기록된 청크를 그대로 되먹이는
스텁 정책을 쓴다 — 그래도 AG3S·TO·안전 판정·응답 형식은 전부 실제 경로를 탄다.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, help="`PolicyRecordWriter` 가 쓴 run 디렉터리")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--cameras", nargs="*", default=None)
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--esdf-margin", type=float, default=0.05)
    ap.add_argument("--links", choices=("arms", "all"), default="arms")
    ap.add_argument("--allow-uncertified", action="store_true")
    ap.add_argument("--drop-camera", default=None, metavar="CAM",
                    help="이 카메라를 요청에서 빼서 누락 시나리오를 시험한다")
    ap.add_argument("--out-json", default="benchmark/trajopt/asset/safe_replay.json")
    args = ap.parse_args()

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene
    from benchmark.ag3s.pipeline import AG3S
    from benchmark.trajopt import wire
    from benchmark.trajopt.config import TrajOptConfig
    from benchmark.trajopt.safe_policy import SafePolicy

    run = load_run(args.records)
    scene = replay_scene(run)
    cameras = tuple(args.cameras or wire.DEFAULT_CAMERAS)

    filter_robot = build_robot_model(scene)
    constraint_robot = build_constraint_robot_model(
        scene, link_filter=None if args.links == "all" else ARM_LINKS)
    ag3s = AG3S(
        AG3SConfig.from_dict({
            "collision_backend": "esdf",
            "pointcloud": {"range_max": args.range_max},
            "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                     "exclude_support_surfaces": False},
        }),
        robot_model=filter_robot, constraint_robot_model=constraint_robot)

    steps = run.steps[:args.frames]

    class _RecordedPolicy:
        """기록된 청크를 되먹이는 스텁. 정책이 아니라 **배선**을 시험하는 것이 목적이다."""

        def __init__(self):
            self.i = 0

        def infer(self, _obs, **_):
            chunk = np.asarray(steps[min(self.i, len(steps) - 1)].actions, np.float64)
            self.i += 1
            return {"actions": chunk}

        def reset(self):
            self.i = 0

        @property
        def metadata(self):
            return {"stub": "recorded chunks"}

    safe = SafePolicy(
        _RecordedPolicy(), ag3s=ag3s,
        to_config=TrajOptConfig.from_dict({
            "collision": {"backend": "esdf", "esdf_margin": args.esdf_margin,
                          "use_support_planes": False},
            "safety": {"require_certified_geometry": not args.allow_uncertified},
        }))

    print(f"제약 구 {safe.linearizer.n_spheres}  카메라 {cameras}"
          + (f"  (누락 시험: {args.drop_camera})" if args.drop_camera else ""))
    print(f"{'seq':>4} {'AG3S':>10} {'인증':>5} {'TO':>10} {'위반mm':>8} "
          f"{'safe':>5} {'전체ms':>8} {'그리퍼':>7}")

    rows = []
    for i, step in enumerate(steps):
        pose_scene(scene, step)
        depth, K, T, state, stamps = {}, {}, {}, {}, {}
        for cam in cameras:
            if cam == args.drop_camera:
                continue
            frame = scene.capture(cam)
            depth[cam] = np.asarray(frame.depth, np.float64)
            K[cam] = np.asarray(frame.camera_intrinsics, np.float64)
            T[cam] = np.asarray(frame.T_base_cam, np.float64)
            state[cam] = np.asarray(frame.robot_state, np.float64)
            stamps[cam] = time.monotonic()
        sent = tuple(c for c in cameras if c != args.drop_camera)

        request = wire.pack_request(
            {"state": np.asarray(step.state, np.float64)},
            cameras=sent, depth=depth, intrinsics=K, extrinsics=T,
            robot_state=state, stamps=stamps, phase="approach",
            reset=(i == 0), seq=i + 1)

        policy_chunk = np.asarray(step.actions, np.float64)
        result = safe.infer(request)
        actions = np.asarray(result["actions"])

        # 그리퍼 열이 정책 값 그대로인지 매 프레임 확인한다. 이 두 열이 조용히 바뀌면 손이
        # 엉뚱한 순간에 열리고, 궤적 오차와 달리 눈에 띄지 않는다.
        grippers_held = all(
            np.allclose(actions[:, c], policy_chunk[:, c], atol=1e-6)
            for c in wire.GRIPPER_COLUMNS)
        shape_ok = actions.shape == policy_chunk.shape

        v = result["max_violation_m"]
        print(f"{result['seq']:>4} {result['ag3s_status']:>10} "
              f"{str(result['geometry_certified']):>5} {result['trajopt_status']:>10} "
              f"{v * 1000 if np.isfinite(v) else float('inf'):>8.1f} "
              f"{str(result['safe']):>5} {result['timing_ms']['total']:>8.0f} "
              f"{'ok' if grippers_held and shape_ok else 'BROKEN':>7}")
        rows.append({
            "seq": int(result["seq"]),
            "ag3s_status": result["ag3s_status"],
            "geometry_certified": bool(result["geometry_certified"]),
            "trajopt_status": result["trajopt_status"],
            "max_violation_mm": float(v * 1000.0) if np.isfinite(v) else None,
            "safe": bool(result["safe"]),
            "timing_ms": result["timing_ms"],
            "grippers_held": bool(grippers_held),
            "shape_ok": bool(shape_ok),
            "delta_max_rad": float(np.abs(actions - policy_chunk).max()),
        })

    n_safe = sum(r["safe"] for r in rows)
    total = np.array([r["timing_ms"]["total"] for r in rows])
    print(f"\n{len(rows)}청크  safe {n_safe}  "
          f"전체 중앙 {np.median(total):.0f} ms  최대 {total.max():.0f} ms")
    broken = [r["seq"] for r in rows if not (r["grippers_held"] and r["shape_ok"])]
    print("그리퍼·형태 불변: " + ("전부 유지" if not broken else f"깨진 청크 {broken}"))

    out = pathlib.Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "records": args.records, "cameras": list(cameras),
        "drop_camera": args.drop_camera, "links": args.links,
        "voxel": args.voxel, "esdf_margin": args.esdf_margin,
        "constraint_spheres": int(safe.linearizer.n_spheres),
        "frames": rows,
    }, indent=2, ensure_ascii=False))
    print(f"wrote {out}")
    scene.close()


if __name__ == "__main__":
    main()
