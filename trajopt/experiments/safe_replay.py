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
    ap.add_argument("--placed-fn", choices=("none", "destination"), default="destination",
                    help="잠금의 **풀기**에 성공 판정을 주입할지 (A3). destination(기본) 은 "
                         "`trajopt/placed.py` 의 참조 구현 — 쥔 물체가 목적지 라벨의 수평 "
                         "범위 안이고 테두리 아래인가. none 이면 그리퍼 폴백만 돈다(예전 동작). "
                         "**둘 다 그리퍼가 닫혀 있는 동안은 안 푼다**")
    ap.add_argument("--below-rim", type=float, default=0.060,
                    help="성공 판정: 쥔 물체가 목적지 테두리보다 이만큼 아래여야 한다 (m)")
    ap.add_argument("--attention", default=None, metavar="NPZ",
                    help="1단계 attention .npz. **이것이 없으면 통합 경로의 하류가 통째로 "
                         "안 돈다** — target 이 없으면 잠금도 attach 도 목적지도 없다 "
                         "(실측: 24 청크 전부 no_target · safe 0). none 이면 합성 블롭을 쓰고 "
                         "그 사실을 결과에 적는다")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/archive/step-verification-20260904/step-01-attention.json")
    ap.add_argument("--no-attention", action="store_true",
                    help="attention 을 **아예 안 넘긴다** — 고치기 전 상태의 대조군. 실측: "
                         "target 이 한 번도 안 잡혀 잠금·attach·목적지·성공 판정이 전부 "
                         "안 돌고 safe 가 0 이 된다")
    ap.add_argument("--static-geometry", default="none", metavar="none|auto|PATH",
                    help="아는 고정 기하를 해석적 채널에 싣는다 (A1). `serve_safe` 와 같은 계약")
    ap.add_argument("--phase-boundaries", type=int, nargs=3, default=(24, 56, 72),
                    metavar=("TRANSIT", "APPROACH", "PRE_GRASP"),
                    help="제어 스텝 -> 단계. 고정 phase 를 쓰면 접촉 권한(E1)이 불활성이다")
    ap.add_argument("--esdf-backend", choices=("legacy", "curobo"), default="legacy",
                    help="AG3S 가 live 로 쓸 필드 구현. `curobo` 는 cuRobo Mapper 가 필요하므로 "
                         "`.venv-openpi-live` 에서 돌려야 한다")
    ap.add_argument("--fine-voxel", type=float, default=0.005,
                    help="`--esdf-backend curobo` 의 미세 계층 복셀. 0 이면 단일 계층")
    ap.add_argument("--tsdf-voxel", type=float, default=0.005)
    ap.add_argument("--attached-sign-threshold", type=float, default=1.5,
                    help="쥔 물체 복셀에서 부호를 양수로 강제할지 가르는 문턱 (복셀 단위). "
                         "cuRobo backend 에서만 쓰인다")
    ap.add_argument("--out-json", default="benchmark/trajopt/asset/safe_replay.json")
    args = ap.parse_args()

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.trajopt import wire
    from benchmark.trajopt.config import TrajOptConfig
    from benchmark.trajopt.experiments.esdf_rollout import phase_for
    from benchmark.trajopt.safe_policy import SafePolicy, grasp_parent_links

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
                     "exclude_support_surfaces": False,
                     "backend": args.esdf_backend,
                     "fine_voxel_size": (args.fine_voxel
                                         if args.esdf_backend == "curobo"
                                         and args.fine_voxel else None),
                     "tsdf_voxel_size": (args.tsdf_voxel
                                         if args.esdf_backend == "curobo" else None),
                     "attached_sign_threshold_voxels": args.attached_sign_threshold},
        }),
        robot_model=filter_robot, constraint_robot_model=constraint_robot,
        # **attached 슬롯을 여기서 예약한다.** 슬롯은 생성 시점에 고정되고 `attach()` 가
        # 나중에 늘릴 수 없다. 안 넘기면 파지 다음 프레임부터 파이프라인이 멈춘다
        # (2026-09-18 실측). `SafePolicy` 생성자가 이것을 다시 확인한다.
        attached_parent_links=grasp_parent_links())

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

    # 성공 판정은 **주입**이다 — 바구니의 내경과 테두리가 어디인지는 과제를 아는 쪽만 안다
    # (`phase`·목적지·정적 기하와 같은 계약). 참조 구현은 거리장의 **라벨 층**에서 목적지의
    # 범위를 읽으므로 따로 기하를 받지 않는다.
    placed_fn = None
    if args.placed_fn == "destination":
        from benchmark.trajopt.placed import destination_placement
        placed_fn = destination_placement(robot_model=constraint_robot,
                                          below_rim=args.below_rim)

    # --- 배선 ① attention -------------------------------------------------------------
    # 이것이 없으면 target 이 안 잡히고, target 이 없으면 잠금·attach·목적지·성공 판정이
    # **전부** 안 돈다 (실측: 24 청크 전부 `no_target`, safe 0).
    attention_kind, att_block, cell, aidx = "없음", None, None, None
    if args.attention and pathlib.Path(args.attention).exists():
        blob = np.load(args.attention, allow_pickle=False)
        cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
        att_block = np.asarray(blob["attention"], np.float32)
        aidx = ([int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"])),
                [str(a) for a in blob["aggregations"]].index(str(cell["agg"])),
                [str(c) for c in blob["cameras"]].index("cam_high"))
        attention_kind = "실측"
    else:
        from benchmark.ag3s.experiments.reports.attention_report import target_from_prompt
        from benchmark.ag3s.experiments.sources.mujoco_source import gaussian_attention
        target_name = target_from_prompt(run.prompt)
        attention_kind = "합성 블롭"

    frame_index = {"i": 0}

    def attention_fn(_obs, _res):
        i = frame_index["i"]
        if att_block is not None:
            di, ai, ci = aidx
            return {"zed_left": att_block[i, di, ai, cell["layer"], cell["head"], ci]}
        head = scene.capture("zed_left")
        return {"zed_left": gaussian_attention(
            head, scene.body_position_in_base(target_name))}

    # --- 배선 ② 정적 기하 (A1) --------------------------------------------------------
    static_geometry = None
    if args.static_geometry != "none":
        from benchmark.ag3s.fields import static_scene
        if args.static_geometry == "auto":
            static_geometry, sg = static_scene.from_mujoco(scene.model, scene.data)
            print(f"[static] {sg.summary()}")
        else:
            static_geometry = static_scene.load(args.static_geometry)

    if args.no_attention:
        attention_fn, attention_kind = None, "없음 (대조군)"

    safe = SafePolicy(
        _RecordedPolicy(), ag3s=ag3s, attention_fn=attention_fn,
        static_geometry=static_geometry,
        to_config=TrajOptConfig.from_dict({
            "collision": {"backend": "esdf", "esdf_margin": args.esdf_margin,
                          "use_support_planes": False},
            "safety": {"require_certified_geometry": not args.allow_uncertified},
        }),
        placed_fn=placed_fn)

    print(f"제약 구 {safe.linearizer.n_spheres}  카메라 {cameras}  attention {attention_kind}"
          f"  정적기하 {len(static_geometry or ())}"
          + (f"  (누락 시험: {args.drop_camera})" if args.drop_camera else ""))
    # **모듈이 실제로 돌았는지**를 청크마다 보인다. 상태 한 줄만 찍으면 "안전하지 않다" 는
    # 결과는 보여도 그것이 지각이 멈춰서인지 위반이 있어서인지 구별할 수 없다.
    print(f"{'seq':>4} {'AG3S':>10} {'인증':>5} {'TO':>10} {'위반mm':>8} "
          f"{'safe':>5} {'ms':>6} {'그리퍼':>7} | "
          f"{'단계':>9} {'target':>6} {'잠금':>9} {'쥔점':>5} {'파냄':>5} {'목적지':>6}")

    rows = []
    for i, step in enumerate(steps):
        pose_scene(scene, step)
        depth, K, T, state, stamps = {}, {}, {}, {}, {}
        # 카메라 셋은 같은 시뮬 순간이다 — 하나의 `qpos` 에서 렌더한다. 카메라마다
        # `time.monotonic()` 을 찍으면 렌더 시간이 촬영 지연으로 읽혀 F14(자세 지연)가
        # 매 프레임 DEGRADED 를 내고, 그러면 기하 인증과 안전 게이트가 통째로 막힌다
        # (실측 296 ms, 한계 100 ms). 하네스가 만든 가짜 지연이므로 한 번만 찍는다.
        stamp = time.monotonic()
        for cam in cameras:
            if cam == args.drop_camera:
                continue
            frame = scene.capture(cam)
            depth[cam] = np.asarray(frame.depth, np.float64)
            K[cam] = np.asarray(frame.camera_intrinsics, np.float64)
            T[cam] = np.asarray(frame.T_base_cam, np.float64)
            state[cam] = np.asarray(frame.robot_state, np.float64)
            stamps[cam] = stamp
        sent = tuple(c for c in cameras if c != args.drop_camera)

        # --- 배선 ③ 단계 · ④ 조작 손 ---------------------------------------------------
        # 고정 phase 를 쓰면 접촉 권한(E1 — 손끝은 target 을 만져도 되지만 팔꿈치는 안 된다)이
        # 통째로 불활성이다. 실제 시스템에서 단계는 과제 계층이 주므로 재생에서는 스텝 경계로
        # 흉내 내고, 그 경계를 인자로 두어 가정이 코드에 숨지 않게 한다.
        phase = phase_for(step.t_step, args.phase_boundaries)
        # 손은 **쥔 뒤에만** 권한을 받는다. 잠금이 HELD 가 아니면 아무도 아무것도 못 만진다.
        manipulators = ("left",) if safe._latch.holding else ()
        frame_index["i"] = i

        request = wire.pack_request(
            {"state": np.asarray(step.state, np.float64)},
            cameras=sent, depth=depth, intrinsics=K, extrinsics=T,
            robot_state=state, stamps=stamps, phase=phase,
            active_manipulators=manipulators,
            reset=(i == 0), seq=i + 1)

        policy_chunk = np.asarray(step.actions, np.float64)
        result = safe.infer(request)
        actions = np.asarray(result["actions"])

        # 그리퍼 열이 정책 값 그대로인지 매 프레임 확인한다. 이 두 열이 조용히 바뀌면 손이
        # 엉뚱한 순간에 열리고, 궤적 오차와 달리 눈에 띄지 않는다.
        # 열은 서버가 쓰는 것과 **같은 자리**에서 온다 (`SafePolicy.gripper_columns`, 레이아웃에서
        # 유도). 여기서 모듈 상수를 쓰면 14D 재생에서 손목 열을 검사하고 그리퍼는 안 본다.
        grippers_held = all(
            np.allclose(actions[:, c], policy_chunk[:, c], atol=1e-6)
            for c in safe.gripper_columns)
        shape_ok = actions.shape == policy_chunk.shape

        v = result["max_violation_m"]
        cs = safe._last_constraint_set
        att = safe.ag3s.attached
        mods = {
            "phase": phase,
            "target": bool(cs is not None and cs.has_target),
            "latch": safe._latch.phase.name,
            "held_points": 0 if att is None or att.points is None else int(len(att.points)),
            "carved": (0 if cs is None or cs.esdf is None
                       else int(cs.esdf.stats.get("n_attached_voxels_carved", 0)
                                or cs.esdf.stats.get("n_attached_seeds_excluded", 0))),
            "attached_sign_correction": (None if cs.esdf is None
                                         else cs.esdf.stats.get("attached_sign_correction")),
            "esdf_backend": (None if cs.esdf is None
                             else cs.esdf.stats.get("backend", "legacy")),
            "destination": bool(getattr(cs, "destination_label", None)),
            "scene_failure": getattr(safe.refiner, "last_failure", None),
        }
        print(f"{result['seq']:>4} {result['ag3s_status']:>10} "
              f"{str(result['geometry_certified']):>5} {result['trajopt_status']:>10} "
              f"{v * 1000 if np.isfinite(v) else float('inf'):>8.1f} "
              f"{str(result['safe']):>5} {result['timing_ms']['total']:>6.0f} "
              f"{'ok' if grippers_held and shape_ok else 'BROKEN':>7} | "
              f"{mods['phase']:>9} {str(mods['target']):>6} {mods['latch']:>9} "
              f"{mods['held_points']:>5} {mods['carved']:>5} {str(mods['destination']):>6}")
        if mods["scene_failure"]:
            print(f"      !! 씬 실패: {mods['scene_failure']}")
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
            "modules": mods,
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
