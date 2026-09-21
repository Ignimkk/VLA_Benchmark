"""실패 케이스가 있는 롤아웃을 **각본으로** 뜬다 (2026-09-18).

`run_0004` 는 정상 동작만 담고 있어서 지금 열려 있는 두 질문을 못 재운다:

* **B1** — 감쇠(F20: 사라진 물체의 잔상이 8 프레임 뒤에도 남는다)를 **시야를 완전히 벗어나는
  장애물**에서 재야 권고값 `a_t 0.99 · a_f 0.8` 을 켤 수 있다. `run_0004` 는 머리 카메라가
  테이블을 내내 봐서 그런 장애물이 없다.
* **B·C 선택** — 지금 배포된 완료 판정(A)과 제안(B·C)이 `run_0004` 에서 **똑같이 프레임 19**
  라 구별되지 않는다. 차이는 **그리퍼가 거짓말하는 경우에만** 드러난다.

### 왜 각본인가 — 그리고 무엇을 포기하는가

정책 체크포인트 없이 뜬다. `run_0004` 의 **관절 궤적을 본보기로** 삼아 위치 액추에이터로
같은 자세를 따라가게 하고, 시나리오마다 한 군데만 바꾼다. 물리는 MuJoCo 가 푼다 — 그래야
사과가 실제로 떨어지고 빈손이 닫힌다.

**포기하는 것은 attention 이다.** 정책이 없으므로 정책 이미지는 렌더한 것이고 attention 맵은
없다. 이 기록으로 재려는 것(감쇠·파지 실패 판정)은 attention 을 안 쓰므로 성립하지만,
**grounding·attention 검증에는 쓰면 안 된다.** `meta.json` 의 `scripted: true` 와
`policy_model: null` 이 그 표시다.

### 시나리오

| 이름 | 무엇을 바꾸나 | 무엇을 재려고 |
|---|---|---|
| `baseline` | 아무것도 (본보기 재현) | 각본 재생이 원본과 같은지 확인하는 대조군 |
| `empty_grasp` | 사과를 옆으로 옮겨 그리퍼가 **빈손으로** 닫히게 | A 는 attach 하고 B 는 안 한다 |
| `slip` | 옮기는 도중 그리퍼를 잠깐 연다 — 사과가 **떨어진다** | A 는 계속 쥔 줄 알고, B 는 Detached |
| `head_away` | 도중에 머리를 돌려 테이블을 **시야에서 뺀다** | 감쇠의 침식 위험 (B1) |

실행:
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -m \
        benchmark.ag3s.experiments.sources.record_scripted --template run_0004 \
        --scenario slip --out records
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

SCENARIOS = ("baseline", "empty_grasp", "slip", "head_away")
#: 본보기에서 파지가 닫히는 스텝 (실측: `run_0004` 프레임 10 에서 그립 폭 90 -> 66 mm).
GRASP_STEP = 10
#: 본보기의 그립 폭이 이보다 좁으면 "쥐려는 중" 으로 보고 **완전 닫힘**을 명령한다.
GRIP_CLOSED_WIDTH_MM = 80.0
#: `slip` 이 그리퍼를 여는 스텝. 파지(10)와 놓기(19) 사이 한가운데.
SLIP_STEP = 14
#: `head_away` 가 머리를 돌리기 시작하는 스텝.
HEAD_AWAY_STEP = 12
#: 머리를 어디로 돌리나 `(head_0 요, head_1 피치)` [rad]. **둘 다 필요하다** — 요만 돌리면
#: `head_0` 의 관절 한계가 0.523 rad(30 도)이고 카메라 시야가 90 도라 테이블이 안 빠진다
#: (실측: 테이블 24,816 -> 15,700 px 로 줄 뿐이고 과일은 오히려 늘었다). 요와 피치를 함께
#: 한계까지 주면 테이블·상자·과일이 **전부 0 px** 이 된다.
HEAD_AWAY_POSE = (0.523, -0.35)


def build_actuator_map(model, mujoco):
    """`액추에이터 인덱스 -> qpos 주소`. 위치 액추에이터만 다룬다.

    본보기의 `qpos` 를 그대로 목표로 주려면 이 대응이 있어야 한다. 이름으로 찾지 않고
    `actuator_trnid` 로 찾는 이유는 액추에이터 이름과 관절 이름이 다르기 때문이다
    (`left_arm_1_act` -> `left_arm_0`).
    """
    out = {}
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        j = int(model.actuator_trnid[a][0])
        out[a] = int(model.jnt_qposadr[j])
    return out


def main() -> None:
    import mujoco

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--template", default="run_0004", help="관절 궤적의 본보기가 될 기록")
    ap.add_argument("--scenario", choices=SCENARIOS, required=True)
    ap.add_argument("--out", default="records", help="run_XXXX 를 만들 부모 디렉터리")
    ap.add_argument("--frames", type=int, default=0, help="0 이면 본보기 전부")
    ap.add_argument("--ctrl-hz", type=float, default=15.0)
    ap.add_argument("--open-loop-horizon", type=int, default=8)
    ap.add_argument("--depth-hw", type=int, nargs=2, default=(480, 640))
    args = ap.parse_args()

    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene
    from benchmark.ag3s.experiments.sources.policy_record import (
        CAMERA_BINDINGS, POLICY_CAMERA_NAMES, load_run)

    tmpl = load_run(args.template, limit=args.frames or None)
    scene = TransportScene(tmpl.model_xml, height=args.depth_hw[0], width=args.depth_hw[1],
                           settle_steps=0)
    model, data = scene.model, scene.data
    amap = build_actuator_map(model, mujoco)

    # 본보기의 첫 자세에서 출발한다 — 물리를 돌리기 전 한 번만 직접 놓는다.
    data.qpos[:] = tmpl.steps[0].qpos
    data.qvel[:] = 0.0

    held_j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "apple_free")
    held_q = int(model.jnt_qposadr[held_j])
    grip_l_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_l_act")
    head0_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_0_act")
    head1_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_1_act")
    # **부호에 주의.** `gripper_finger_l1` 의 범위는 [-0.05, 0] 이고 폭은 `l2 - l1` 이므로
    # **더 음수인 쪽이 열림**(100 mm), 0 이 닫힘(0 mm)이다. 반대로 잡으면 파지 구간에서
    # 그리퍼가 활짝 열린다 (실측으로 잡았다).
    grip_open = float(min(model.actuator_ctrlrange[grip_l_a]))
    grip_shut = float(max(model.actuator_ctrlrange[grip_l_a]))

    if args.scenario == "empty_grasp":
        # 사과를 옆으로 80 mm 옮긴다. 팔은 본보기 궤적을 그대로 따라가므로 **빈손으로** 닫힌다.
        data.qpos[held_q + 1] += 0.08
    mujoco.mj_forward(model, data)

    sim_per_step = max(int(round(args.open_loop_horizon / args.ctrl_hz / model.opt.timestep)), 1)
    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for k in range(1_000_000):
        run_dir = out_root / f"run_{k:04d}"
        try:
            run_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            continue

    renderer = mujoco.Renderer(model, 224, 224)
    prev_ctrl = np.array([data.qpos[amap[a]] for a in sorted(amap)], float)
    print(f"[{args.scenario}] {len(tmpl.steps)} 스텝, 스텝당 {sim_per_step} 물리 스텝 "
          f"-> {run_dir}")
    print(f"{'i':>3} {'그립폭mm':>9} {'사과z':>8} {'손-사과mm':>10} {'head_0':>8} {'head_1':>8}")

    order = sorted(amap)
    gi = order.index(grip_l_a)
    for i, step in enumerate(tmpl.steps):
        target = np.array([step.qpos[amap[a]] for a in order], float)

        # **그리퍼는 달성값이 아니라 명령값을 줘야 한다.** 본보기의 `qpos` 는 손가락이 사과에
        # 막혀 멈춘 **결과**(66 mm)이고, 그것을 목표로 주면 쥐는 힘이 0 이라 사과가 안 들린다
        # (실측: 사과 z 가 0.850 에서 안 움직였다). 본보기가 닫힌 구간에서는 **완전 닫힘**을
        # 명령해 사과가 손가락을 멈추게 한다 — 실제 파지와 같은 상태다.
        width_mm = (step.qpos[amap[order[gi]] + 1] - step.qpos[amap[order[gi]]]) * 1000.0
        if width_mm < GRIP_CLOSED_WIDTH_MM:
            target[gi] = grip_shut

        # --- 시나리오: 목표를 한 군데만 바꾼다 ------------------------------------------
        if args.scenario == "slip" and i in (SLIP_STEP, SLIP_STEP + 1):
            target[gi] = grip_open                                # 잠깐 놓는다
        if args.scenario == "head_away" and i >= HEAD_AWAY_STEP:
            target[order.index(head0_a)] = HEAD_AWAY_POSE[0]      # 테이블에서 눈을 뗀다
            target[order.index(head1_a)] = HEAD_AWAY_POSE[1]

        # 한 스텝 안에서 목표를 선형으로 옮긴다 — 계단 입력을 주면 액추에이터가 튄다.
        for s in range(sim_per_step):
            w = (s + 1) / sim_per_step
            ctrl = (1.0 - w) * prev_ctrl + w * target
            for c, a in enumerate(order):
                data.ctrl[a] = ctrl[c]
            mujoco.mj_step(model, data)
        prev_ctrl = target

        # --- 기록 ------------------------------------------------------------------------
        payload = {
            "t_step": np.int64(step.t_step),
            "qpos": np.asarray(data.qpos, np.float64).copy(),
            "qvel": np.asarray(data.qvel, np.float64).copy(),
            "state": np.asarray(step.state, np.float64).copy(),
            # 본보기의 청크를 그대로 둔다 — 팔은 실제로 그 목표를 따라갔다. 그리퍼 열(6)만
            # 우리가 실제로 명령한 값으로 덮는다. 안 덮으면 `slip` 이 기록에서 안 보인다.
            "actions": np.asarray(step.actions, np.float32).copy(),
            "infer_ms": np.float64(0.0),
        }
        if args.scenario == "slip" and i in (SLIP_STEP, SLIP_STEP + 1):
            payload["actions"][:, 6] = np.float32(grip_open)
        for key in POLICY_CAMERA_NAMES:
            cam = CAMERA_BINDINGS[key][0]
            renderer.update_scene(data, camera=cam)
            payload[f"image_{key}"] = np.ascontiguousarray(renderer.render()).astype(np.uint8)
        for cam in ("zed_left", "wrist_cam_l", "wrist_cam_r"):
            frame = scene.capture(cam)
            payload[f"depth_{cam}"] = np.clip(
                np.rint(np.asarray(frame.depth) * 1000.0), 0, 65535).astype(np.uint16)
            payload[f"K_{cam}"] = np.asarray(frame.camera_intrinsics, np.float64)
            payload[f"T_base_cam_{cam}"] = np.asarray(frame.T_base_cam, np.float64)
        np.savez_compressed(run_dir / f"step_{i:05d}.npz", **payload)

        hand = 0.5 * (data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_finger_l1")]
                      + data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                                    "ee_finger_l2")])
        apple = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "apple")]
        gl = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                 "gripper_finger_l1")]
        gl2 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                  "gripper_finger_l2")]
        print(f"{i:>3} {(data.qpos[gl2]-data.qpos[gl])*1000:>9.1f} {apple[2]:>8.3f} "
              f"{np.linalg.norm(hand-apple)*1000:>10.1f} "
              f"{data.qpos[scene._qadr['head_0']]:>8.3f} "
              f"{data.qpos[scene._qadr['head_1']]:>8.3f}")

    renderer.close()
    meta = dict(tmpl.meta)
    meta.update({
        "n_steps": len(tmpl.steps),
        # **각본이라는 표시.** 이 기록에는 정책도 attention 도 없다 — grounding·attention 검증에
        # 쓰면 안 된다. 지우면 나중에 이 기록이 실측 롤아웃인 척하게 된다.
        "scripted": True,
        "scripted_scenario": args.scenario,
        "scripted_template": str(args.template),
        "policy_model": None,
        "attention": None,
        "ctrl_hz": args.ctrl_hz,
        "open_loop_horizon": args.open_loop_horizon,
        "sim_steps_per_record": sim_per_step,
    })
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ag3s] wrote {len(tmpl.steps)} scripted observations to {run_dir}")


if __name__ == "__main__":
    main()
