"""풀 시나리오 — 관측에서 ESDF 충돌 제약까지, 그리고 TO 가 그것으로 청크를 고치는 데까지.

지금까지의 검증은 단계를 하나씩 떼어 봤다. 이 스크립트는 **끊지 않고** 돌린다:

    기록된 관측 (depth 3대 + 정책 청크)
        -> AG3S  `collision_backend: esdf`   : TSDF -> ESDF, target 파냄, 지지면 평면
        -> TO    `collision.backend: esdf`   : d_esdf(p(q)) - r - margin >= 0
        -> 안전한 청크 + 상태

**primitive 는 쓰지 않는다.** AG3S 는 backend 와 무관하게 candidate 를 내놓지만
`scene_from_constraint_set` 이 `esdf` 에서 슬롯을 전부 끄므로, 이 실행에서 로봇을 막는 것은
거리장과 지지면 평면뿐이다.

참조 궤적은 **정책이 실제로 낸 청크**다. 합성 궤적을 쓰면 "TO 가 고칠 수 있는 위반"을 내가
만들어 넣는 셈이라 아무것도 증명하지 못한다. 기록된 청크가 실제로 위반하지 않으면 그것도 결과다.

실행:
    MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.trajopt.experiments.esdf_rollout \
        --records outputs/.../ag3s_records/run_0002
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.pipeline import AG3S
from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS
from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.limits import build_limits
from benchmark.trajopt.linearize import CollisionLinearizer, scene_from_constraint_set
from benchmark.trajopt.sqp import TrajectoryOptimizer
from benchmark.trajopt.types import ChunkLayout

CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def phase_for(t_step: int, boundaries) -> str:
    """제어 스텝 -> 단계. 접촉 허가와 target 파냄이 여기에 달려 있다.

    실제 시스템에서 단계는 과제 계층이 정한다 (`AG3S.attach()` 의 docstring 이 "AG3S never calls
    this itself" 라고 못박는 것과 같은 이유다). 여기서는 재생이므로 스텝 경계로 흉내 내고,
    그 경계를 인자로 두어 **가정이 코드에 숨지 않게** 한다.
    """
    transit, approach, pre_grasp = boundaries
    if t_step < transit:
        return "transit"
    if t_step < approach:
        return "approach"
    if t_step < pre_grasp:
        return "pre_grasp"
    return "grasp"


def main() -> None:
    import mujoco

    from benchmark.ag3s.experiments.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.mujoco_source import gaussian_attention, is_robot_body
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--attention", default=None,
                    help="1단계 .npz. 없으면 합성 블롭으로 대신하며 그 사실을 결과에 적는다")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/step-01-attention.json")
    ap.add_argument("--target", default=None)
    ap.add_argument("--frames", type=int, default=0, help="0이면 전부")
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--esdf-margin", type=float, default=0.05)
    ap.add_argument("--constraint-links", choices=("arms", "all"), default="arms",
                    help="충돌 제약을 어느 링크에 걸지. `arms` 는 양팔 링크와 손끝만 — 결정 "
                         "변수가 양팔 12관절뿐이므로 바퀴·베이스·토르소 구는 어떤 해에서도 "
                         "같은 값을 내고, 고칠 수 없는 위반을 상수로 깔아 실제 신호를 묻는다. "
                         "`all` 은 전신(이전 동작). 자기 필터 모델은 어느 쪽이든 전신이다")
    ap.add_argument("--support-surfaces", default="field", choices=("plane", "field"),
                    help="지지면을 어떻게 다룰지. `plane` 은 half-space 행(여유 10 mm)으로 쓰고 "
                         "필드에서는 파낸다. `field` 는 표면을 거리장에 남기고 평면 행을 읽지 "
                         "않는다 — **최종 제약을 전부 ESDF 가 만들게 하려면 이쪽**이라 기본값이다. "
                         "어느 쪽이든 평면 *추출* 은 켜져 있다: grounding 이 그 마스크 없이는 "
                         "테이블을 타고 번져 target 을 못 찾는다")
    ap.add_argument("--phase-boundaries", type=int, nargs=3, default=(24, 56, 72),
                    metavar=("TRANSIT", "APPROACH", "PRE_GRASP"))
    ap.add_argument("--out-json", default="benchmark/trajopt/asset/esdf_rollout.json")
    ap.add_argument("--out-doc", default="benchmark/ag3s/docs/esdf-full-scenario.md")
    args = ap.parse_args()

    run = load_run(args.records, limit=args.frames or None)
    from benchmark.ag3s.experiments.attention_report import target_from_prompt
    target = args.target or target_from_prompt(run.prompt)

    cell = None
    attention_block = None
    if args.attention:
        blob = np.load(args.attention, allow_pickle=False)
        cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
        attention_block = np.asarray(blob["attention"], np.float32)
        di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
        ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
        ci = [str(c) for c in blob["cameras"]].index("cam_high")

    scene = replay_scene(run)
    # 두 모델은 **일부러 다르다**. 자기 필터는 전신이어야 바퀴·베이스 점이 클라우드에서 지워지고,
    # 제약은 최적화기가 실제로 움직일 수 있는 구에만 걸려야 의미가 있다.
    filter_robot = build_robot_model(scene)
    robot = build_constraint_robot_model(
        scene, link_filter=None if args.constraint_links == "all" else ARM_LINKS)
    names = {i: (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
             for i in range(scene.model.nbody)}
    robot_ids = [i for i, n in names.items() if is_robot_body(n)]

    # 평면 **추출**은 어느 모드에서도 켜 둔다. `ground_target` 이 그 마스크를 exclude_mask 로
    # 받지 못하면 영역 성장이 테이블을 타고 번져 씬 전체가 한 클러스터가 되고, target 을 아예
    # 못 찾는다 (`no_target`). 모드가 정하는 것은 그 다음 두 가지뿐이다:
    #   - 표면을 필드에서 파낼 것인가 (`exclude_support_surfaces`)
    #   - 최적화기가 평면 행을 제약으로 읽을 것인가 (`use_support_planes`)
    plane_mode = args.support_surfaces == "plane"
    ag_cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": plane_mode},
    })
    ag = AG3S(ag_cfg, robot_model=filter_robot, constraint_robot_model=robot)

    layout = ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)
    to_cfg = TrajOptConfig.from_dict({
        "collision": {"backend": "esdf", "esdf_margin": args.esdf_margin,
                      "use_support_planes": plane_mode},
    })
    limits = build_limits(robot, layout, dt=to_cfg.horizon.dt, config=to_cfg.limits)
    planned = to_cfg.with_overrides({"horizon": {"horizon": to_cfg.horizon.planned,
                                                 "execution_length": min(
                                                     to_cfg.horizon.execution_length,
                                                     to_cfg.horizon.planned),
                                                 "plan_horizon": None}})
    optimizer = TrajectoryOptimizer(robot, layout, limits, planned)
    linearizer = CollisionLinearizer(robot, layout, planned.horizon.horizon)

    # `--constraint-links arms` 면 제약 모델에 바퀴·베이스·토르소 구가 아예 없으므로 두 마스크가
    # 같아진다. `all` 로 되돌리면 다시 갈라지고, 그때는 두 줄을 나란히 보는 것이 의미가 있다 —
    # 이동 베이스는 바닥에 **닿아 있는 것이 정상**이라 고칠 수 없는 위반이 상수로 깔린다.
    link_names = list(robot.sphere_link_names)
    arm_only = np.array([not n.startswith(("base", "wheel")) for n in link_names])
    print(f"로봇 모델: 자기필터 {filter_robot.n_spheres}구 / 제약 {robot.n_spheres}구"
          f"  ({args.constraint_links})")

    def violation(traj, q, snap, mask=None):
        cand, plane, _ = linearizer._clearances_from(
            linearizer.sphere_states(traj, q)[0], snap)
        esdf = linearizer._esdf_clearance(linearizer.sphere_states(traj, q)[0], snap)
        worst = np.inf
        for block in (cand, plane, esdf):
            if block.size == 0:
                continue
            b = block if mask is None else block[:, mask, :]
            if b.size:
                worst = min(worst, float(np.min(b)))
        return worst

    print(f"target={target}  attention={'실측' if attention_block is not None else '합성'}  "
          f"복셀={args.voxel*1000:.0f}mm  esdf_margin={args.esdf_margin*1000:.0f}mm  "
          f"계획 지평={planned.horizon.horizon}")

    rows = []
    previous = None
    for i, step in enumerate(run):
        pose_scene(scene, step)
        frames = {c: scene.capture(c) for c in CAMERAS}
        head = frames["zed_left"]
        if attention_block is not None:
            att = attention_block[i, di, ai, cell["layer"], cell["head"], ci]
        else:
            att = gaussian_attention(head, scene.body_position_in_base(target))
        phase = phase_for(step.t_step, args.phase_boundaries)

        t0 = time.time()
        cs = ag.process(depth=head.depth, camera_intrinsics=head.camera_intrinsics,
                        T_base_cam=head.T_base_cam, attention_map=att,
                        robot_state=head.robot_state, phase=phase)
        ag_ms = (time.time() - t0) * 1000.0

        snap = scene_from_constraint_set(cs, linearizer.robot_radii, to_cfg)
        reference = layout.chunk_to_trajectory(step.actions)[:, :planned.horizon.horizon]
        q_now = np.asarray([scene.data.qpos[scene._qadr[j]] for j in DEFAULT_RBY1_JOINTS], float)

        before = linearizer.full_violation(reference, q_now, snap)
        before_arm = violation(reference, q_now, snap, arm_only)
        t0 = time.time()
        result = optimizer.solve(reference, q_now, snap, previous_chunk=previous,
                                 template=step.actions,
                                 geometry_certified=cs.geometry_certified)
        to_ms = (time.time() - t0) * 1000.0
        after = linearizer.full_violation(result.trajectory, q_now, snap)
        after_arm = violation(result.trajectory, q_now, snap, arm_only)
        # 연속성 항은 관절 궤적끼리 비교한다. chunk 는 (H, action_dim) 이라 모양이 맞지 않는다.
        previous = result.trajectory

        rows.append({
            "i": i, "t_step": int(step.t_step), "phase": phase,
            "ag3s_ms": ag_ms, "to_ms": to_ms,
            "n_candidates": len(cs.candidates),
            "target": cs.has_target,
            "esdf_unknown": float(cs.esdf.unknown_fraction),
            "esdf_occupied": int(cs.esdf.stats["n_occupied"]),
            "target_voxels_carved": int(cs.esdf.stats.get("n_target_voxels_carved", 0)),
            "clearance_before_mm": float(before) * 1000.0,
            "clearance_after_mm": float(after) * 1000.0,
            "arm_before_mm": float(before_arm) * 1000.0,
            "arm_after_mm": float(after_arm) * 1000.0,
            "status": result.status.value,
            "safe": bool(result.safe),
            "iterations": int(result.iterations),
            "reference_deviation": float(result.reference_deviation),
        })
        if i % 5 == 0 or i == len(run) - 1:
            print(f"  [{i:3d}] t={step.t_step:4d} {phase:9s} AG3S {ag_ms:6.0f}ms  TO {to_ms:6.0f}ms  "
                  f"전체 {before*1000:+8.1f}→{after*1000:+8.1f}  "
                  f"팔만 {before_arm*1000:+7.1f}→{after_arm*1000:+7.1f} mm  {result.status.value}")
    scene.close()

    out = pathlib.Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "records": str(run.path), "prompt": run.prompt, "target": target,
        "attention": "measured" if attention_block is not None else "synthetic",
        "ag3s": {"collision_backend": "esdf", "voxel_size": args.voxel,
                 "range_max": args.range_max,
                 "constraint_links": args.constraint_links,
                 "n_filter_spheres": filter_robot.n_spheres,
                 "n_constraint_spheres": robot.n_spheres},
        "trajopt": {"backend": "esdf", "esdf_margin": args.esdf_margin,
                    "planned_horizon": planned.horizon.horizon,
                    "support_surfaces": args.support_surfaces},
        "frames": rows,
    }, indent=2, ensure_ascii=False))

    def summarise(before_key, after_key, label):
        viol = [r for r in rows if r[before_key] < 0]
        fixed = [r for r in viol if r[after_key] >= 0]
        imp = [r for r in rows if r[after_key] > r[before_key] + 1e-6]
        print(f"  {label}: 위반으로 시작 {len(viol)}/{len(rows)}  해소 {len(fixed)}  개선 {len(imp)}")

    print(f"\n{len(rows)}청크")
    summarise("clearance_before_mm", "clearance_after_mm", "전체 구  ")
    summarise("arm_before_mm", "arm_after_mm", "팔 구만  ")
    print(f"  AG3S {np.mean([r['ag3s_ms'] for r in rows]):.0f} ms  "
          f"TO {np.mean([r['to_ms'] for r in rows]):.0f} ms  "
          f"합계 {np.mean([r['ag3s_ms']+r['to_ms'] for r in rows]):.0f} ms / 예산 66.7 ms")
    print(f"  상태: " + ", ".join(f"{k} {sum(1 for r in rows if r['status']==k)}"
                                  for k in sorted({r['status'] for r in rows})))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
