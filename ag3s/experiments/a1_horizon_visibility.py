"""정적 기하가 이 과제에 **필요한가** — 계획 궤적이 시야를 벗어나는가 (2026-09-16 질문).

사용자 지적: *"사과를 바구니에 담는 작업은 정적 기하를 몰라도 된다. 손목·머리 카메라가 못 보는
경로로 궤적을 그릴 일이 거의 없다."*

`a1_static_geometry_effect.py` 가 그 주장을 뒷받침했지만 **구멍이 하나 있었다** — 거기서는
각 프레임의 **실행된 자세 하나**(`q_now`)만 봤다. 최적화기가 제약을 거는 것은 앞으로
**32 스텝을 내다본 계획 궤적 전체**다 (32 × 120 = 3,840 행/청크). 실행 궤적이 시야 안에
있어도 계획 궤적이 시야 밖을 지나면, 최적화기는 "거기가 비었다"는 답을 보고 그쪽으로 민다.

그래서 여기서는 **계획 지평 전체**의 구 위치를 잰다. 셋을 비교한다:

* `q_now` — 실행된 자세 하나 (앞선 측정이 본 것)
* `reference` — 정책이 낸 원래 청크의 32 스텝 전부
* `optimized` — SQP 가 고친 청크의 32 스텝 전부 ← **실제로 실행될 것**

각 구 중심에서 `필드가 답한 거리 − 아는 정적 기하까지의 참 거리` 를 잰다. **양수면 필드가
아는 것보다 넓다고 말한 것**이고, 그것만이 위험하다.

실행:
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -m \
        benchmark.ag3s.experiments.a1_horizon_visibility --records run_0004 --frames 15
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures/a1-horizon-visibility.png")
CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments import figstyle
    figstyle.use_korean()
    return figstyle


def main() -> None:
    import mujoco

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--attention", default="attention_step1_run0004.npz")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/step-01-attention.json")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--esdf-margin", type=float, default=0.05)
    ap.add_argument("--cameras", choices=("head", "all"), default="head",
                    help="head = 회귀 기준선과 같은 조건(가장 불리한 쪽). all = 서빙 경로")
    ap.add_argument("--phase-boundaries", type=int, nargs=3, default=(24, 56, 72))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--out-json",
                    default="benchmark/ag3s/docs/figures/a1-horizon-visibility.json")
    args = ap.parse_args()

    from benchmark.ag3s import static_scene
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.esdf import analytic_distance
    from benchmark.ag3s.experiments.attention_report import target_from_prompt
    from benchmark.ag3s.experiments.grounding_report import (
        build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.mujoco_source import camera_observation, gaussian_attention
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene
    from benchmark.ag3s.pipeline import AG3S
    from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS
    from benchmark.trajopt.config import TrajOptConfig
    from benchmark.trajopt.experiments.esdf_rollout import phase_for
    from benchmark.trajopt.limits import build_limits
    from benchmark.trajopt.linearize import CollisionLinearizer, scene_from_constraint_set
    from benchmark.trajopt.sqp import TrajectoryOptimizer
    from benchmark.trajopt.types import ChunkLayout

    run = load_run(args.records, limit=args.frames or None)
    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    robot = build_constraint_robot_model(scene)

    pose_scene(scene, run.steps[0])
    shapes, survey = static_scene.from_mujoco(scene.model, scene.data)
    print(f"[static] {survey.summary()}")

    cell = attention_block = None
    if args.attention and pathlib.Path(args.attention).exists():
        blob = np.load(args.attention, allow_pickle=False)
        cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
        attention_block = np.asarray(blob["attention"], np.float32)
        di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
        ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
        ci = [str(c) for c in blob["cameras"]].index("cam_high")
    target_name = target_from_prompt(run.prompt)

    ag_cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False},
    })
    # **정적 기하 없이** 돌린다. 질문이 "없어도 되는가" 이므로 없는 쪽이 측정 대상이다.
    ag = AG3S(ag_cfg, robot_model=filter_robot, constraint_robot_model=robot)

    layout = ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)
    to_cfg = TrajOptConfig.from_dict({
        "collision": {"backend": "esdf", "esdf_margin": args.esdf_margin,
                      "use_support_planes": False}})
    limits = build_limits(robot, layout, dt=to_cfg.horizon.dt, config=to_cfg.limits)
    planned = to_cfg.with_overrides({"horizon": {
        "horizon": to_cfg.horizon.planned,
        "execution_length": min(to_cfg.horizon.execution_length, to_cfg.horizon.planned),
        "plan_horizon": None}})
    optimizer = TrajectoryOptimizer(robot, layout, limits, planned)
    linearizer = CollisionLinearizer(robot, layout, planned.horizon.horizon)

    S = int(robot.n_spheres)
    kinds = ("q_now", "reference", "optimized")
    opt = {k: [] for k in kinds}          # 프레임별 낙관 오차 배열
    sat = {k: 0 for k in kinds}
    n_pts = {k: 0 for k in kinds}
    previous = None

    for i, step in enumerate(run):
        pose_scene(scene, step)
        frames = {c: scene.capture(c) for c in CAMERAS}
        head = frames["zed_left"]
        if attention_block is not None:
            att = attention_block[i, di, ai, cell["layer"], cell["head"], ci]
        else:
            att = gaussian_attention(head, scene.body_position_in_base(target_name))
        phase = phase_for(step.t_step, args.phase_boundaries)

        if args.cameras == "all":
            observations = [
                camera_observation(scene, c, filter_robot, timestamp=float(i),
                                   attention_map=(att if c == "zed_left" else None))[0]
                for c in CAMERAS]
            cs = ag.process_multi(observations, phase=phase)
        else:
            cs = ag.process(depth=head.depth, camera_intrinsics=head.camera_intrinsics,
                            T_base_cam=head.T_base_cam, attention_map=att,
                            robot_state=head.robot_state, phase=phase)

        snap = scene_from_constraint_set(cs, linearizer.robot_radii, to_cfg)
        reference = layout.chunk_to_trajectory(step.actions)[:, :planned.horizon.horizon]
        q_now = np.asarray([scene.data.qpos[scene._qadr[j]] for j in DEFAULT_RBY1_JOINTS], float)
        result = optimizer.solve(reference, q_now, snap, previous_chunk=previous,
                                 template=step.actions,
                                 geometry_certified=cs.geometry_certified)
        previous = result.trajectory

        # `sphere_states` 는 `(H, Q, 3)` — 계획 지평 **전체**의 구 중심. 앞의 S 개가 로봇 구다
        # (뒤는 쥔 물체의 점, 반지름 0).
        pts = {
            "q_now": robot.sphere_centers_numeric(q_now)[0],
            "reference": linearizer.sphere_states(reference, q_now)[0][:, :S, :].reshape(-1, 3),
            "optimized": linearizer.sphere_states(result.trajectory, q_now)[0][:, :S, :]
                         .reshape(-1, 3),
        }
        for k, p in pts.items():
            d = cs.esdf.distance(p)
            truth = analytic_distance(p, shapes)
            opt[k].append(d - truth)
            sat[k] += int(np.isclose(d, cs.esdf.max_distance, atol=1e-9).sum())
            n_pts[k] += len(p)
        print(f"  [{i:3d}] {phase:9s} "
              + "  ".join(f"{k} max{float(np.max(opt[k][-1]))*1000:+8.1f}mm" for k in kinds))

    summary = {"records": args.records, "frames": len(run.steps), "cameras": args.cameras,
               "horizon": int(planned.horizon.horizon), "spheres": S, "n_shapes": len(shapes)}
    for k in kinds:
        v = np.concatenate(opt[k]) * 1000.0
        summary[k] = {"n_points": n_pts[k],
                      "max_optimism_mm": float(v.max()),
                      "n_optimistic": int((v > 0.05).sum()),
                      "frac_optimistic": float((v > 0.05).mean()),
                      "saturated": sat[k],
                      "median_mm": float(np.median(v))}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------ 그림
    fs = _style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 1.0], wspace=0.30)
    colours = {"q_now": fs.CATEGORICAL[2], "reference": fs.CATEGORICAL[3],
               "optimized": fs.CATEGORICAL[0]}
    label = {"q_now": "실행된 자세 하나 (앞선 측정)",
             "reference": f"정책 청크 {planned.horizon.horizon} 스텝",
             "optimized": f"SQP 가 고친 청크 (실행될 것)"}

    ax = fig.add_subplot(gs[0, 0])
    allv = np.concatenate([np.concatenate(opt[k]) for k in kinds]) * 1000.0
    bins = np.linspace(float(allv.min()), max(float(allv.max()), 50.0), 80)
    for k in kinds:
        ax.hist(np.concatenate(opt[k]) * 1000.0, bins=bins, histtype="step", lw=1.7,
                color=colours[k], label=label[k])
    ax.axvline(0, color=fs.INK, lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("낙관 오차  필드 − 참(정적 기하) [mm]  — 0 오른쪽이 문제")
    ax.set_ylabel("질의 수 (log)")
    ax.set_title("① 분포 — 계획 지평까지 넣어도 0 오른쪽이 있는가",
                 fontsize=10.5, color=fs.INK)
    ax.legend(fontsize=8.2, frameon=False, loc="upper left")

    ax2 = fig.add_subplot(gs[0, 1])
    n = len(opt["q_now"])
    for k in kinds:
        ax2.plot(range(n), [float(np.max(x)) * 1000.0 for x in opt[k]], "-o", ms=4,
                 color=colours[k], label=label[k])
    ax2.axhline(0, color=fs.INK, lw=1.4)
    ax2.set_xlabel("청크"); ax2.set_ylabel("그 청크의 최대 낙관 [mm]")
    ax2.set_title("② 청크별 최대 낙관 — 0 선을 넘는 청크가 있는가",
                  fontsize=10.5, color=fs.INK)
    ax2.legend(fontsize=8.2, frameon=False, loc="lower left")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    # matplotlib 은 별표를 굵게 렌더하지 않는다 — 제목·라벨에 쓰지 말 것 (CLAUDE.md).
    ax3.set_title(f"③ 결과 — {args.cameras} 카메라, 정적 기하 없이",
                  fontsize=10.5, color=fs.INK)
    rows = [["무엇", "질의 수", "최대 낙관", "낙관 질의"]]
    for k in kinds:
        rows.append([label[k].split(" (")[0], f"{summary[k]['n_points']:,}",
                     f"{summary[k]['max_optimism_mm']:+,.1f} mm",
                     f"{summary[k]['n_optimistic']:,}"])
    t = ax3.table(cellText=rows, colWidths=[0.42, 0.19, 0.22, 0.17],
                  loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(8.4); t.scale(1, 1.6)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if r == 0:
            cell.set_facecolor("#ecebe7"); cell.set_text_props(weight="bold")

    fs.style_axes(fig, [ax, ax2])
    fig.suptitle("정적 기하가 이 과제에 필요한가 — 계획 지평까지 포함해 낙관을 다시 잰다"
                 f"   ({args.records}, {len(run.steps)} 청크, 지평 {planned.horizon.horizon}, "
                 f"{S} 구)", fontsize=12.5, color=fs.INK)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
