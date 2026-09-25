"""A3 — 잠금의 **풀기**를 그리퍼로만 하고 있다. 성공 판정을 주입하면 무엇이 달라지는가.

`GraspLatch` 의 풀기는 두 갈래다:

* `placed` — **외부가 주는 성공 판정**. `SafePolicy(placed_fn=...)` 로 받도록 열려 있다.
* `_open_streak >= release_frames` — 그리퍼가 2 프레임 연속 열려 있는 것. **폴백**이다.

지금 `placed_fn` 을 넘기는 호출자가 없어 **폴백만 돈다.**

### 왜 지금 재는가 — A2 가 비용을 바꿨다

A2(쥔 물체를 필드에서 파낸다) 전에는 detach 가 늦어도 손해가 크지 않았다. 쥔 물체가 질의점
으로 남을 뿐이었다. **A2 이후로는 늦은 detach 가 필드를 계속 파낸다** — 그 자리에 이미
놓인 사과가 있는데도. 그러면 팔이 "거기는 비었다" 는 답을 받는다. 낙관이고 위험한 쪽이다.

그래서 세 가지를 잰다:

1. **두 신호가 언제 발동하는가** — 그리퍼 폴백 대 성공 판정.
2. **늦은 detach 가 얼마를 낳는가** — 놓은 뒤에도 파내기를 계속했을 때 팔 구의 낙관.
3. **성공 판정이 흔들리는가** — 한 번 참이 된 뒤 깜빡이면 detach 가 왔다 갔다 한다.

성공 판정식은 기록에 있다 (`AG3S_REVIEW_LOG.md` "성공 판정"):

    (사과–바구니 수평거리 <= 82 mm) AND (사과가 테두리 60 mm 아래) AND (손–사과 > 80 mm)

**참값 자세로 잰다.** 실행 시에는 관측으로 해야 하지만 판정의 **형태**는 같고, 여기서 재려는
것은 판정식의 정확도가 아니라 **두 신호의 시점 차이와 그 사이의 손해**다.

실행:
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -m \
        benchmark.ag3s.experiments.studies.a3_release_signal --records run_0004
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/archive/14d-era-20260923/archive/14d-era-20260923/figures/a3-release-signal.png")
PARENT_LINK = "link_left_arm_6"
CONTACT_LINKS = frozenset({"ee_finger_l1", "ee_finger_l2"})
CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
GRIP_CLOSED_MM = 80.0
HAND_NEAR_MM = 70.0
#: 성공 판정의 세 문턱 (기록의 실측값). 헐겁게 잡혀 있다 — 실측 16.1 mm 대 82 mm 등.
PLACED_RADIAL_MM = 82.0
PLACED_BELOW_RIM_MM = 60.0
PLACED_HAND_GAP_MM = 80.0
#: `LatchConfig.release_frames` 와 같은 값. 그리퍼 폴백이 몇 프레임 열려야 푸는가.
RELEASE_FRAMES = 2


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments.common import figstyle
    figstyle.use_korean()
    return figstyle


def backproject(depth, K, T_base_cam, mask):
    vs, us = np.nonzero(mask)
    z = np.asarray(depth, np.float64)[vs, us]
    good = np.isfinite(z) & (z > 1e-6)
    vs, us, z = vs[good], us[good], z[good]
    if z.size == 0:
        return np.zeros((0, 3))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam = np.column_stack([(us - cx) / fx * z, (vs - cy) / fy * z, z])
    return cam @ np.asarray(T_base_cam, np.float64)[:3, :3].T + T_base_cam[:3, 3]


def main() -> None:
    import mujoco

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--held", default="apple")
    ap.add_argument("--destination", default="crate")
    ap.add_argument("--frames", type=int, default=0, help="0 이면 전부")
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--esdf-margin", type=float, default=0.05)
    ap.add_argument("--cameras", choices=("head", "all"), default="all")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--out-json", default="benchmark/ag3s/docs/archive/14d-era-20260923/archive/14d-era-20260923/figures/a3-release-signal.json")
    args = ap.parse_args()

    from benchmark.ag3s.constraints.attached import attach_from_target, attached_points_in_base
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.fields.esdf import CameraDepth, EsdfBuilder
    from benchmark.ag3s.experiments.reports.grounding_report import (
        build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS
    from benchmark.ag3s.types import (
        DESTINATION_LABEL, Primitive, PrimitiveType, TargetGeometry)
    from benchmark.trajopt.placed import destination_placement

    run = load_run(args.records, limit=args.frames or None)
    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    robot = build_constraint_robot_model(scene)

    held_b = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, args.held)
    dest_b = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, args.destination)
    fl = [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, x)
          for x in ("ee_finger_l1", "ee_finger_l2")]
    gadr = {n: scene.model.jnt_qposadr[
                mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ("gripper_finger_l1", "gripper_finger_l2")}
    rim_g = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "crate_wall_px")
    # 참조 구현(`trajopt/placed.py`). 참값이 아니라 **관측**으로 돈다 — 라벨 층에서 목적지의
    # 범위를 읽고 쥔 물체의 질의점과 비교한다.
    ref_placed_fn = destination_placement(robot_model=robot, below_rim=0.060)

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False},
    })
    ag = AG3S(cfg, robot_model=filter_robot, constraint_robot_model=robot)
    # 필드 둘. 같은 관측을 쓰고 **파내기를 언제 그만두는가**만 다르다.
    #   `truth`  — 성공 판정이 참이 되는 순간 파내기를 멈춘다 (옳은 동작)
    #   `stale`  — 끝까지 파낸다 (detach 가 영영 안 오는 최악)
    builders = {"truth": EsdfBuilder(cfg.esdf), "stale": EsdfBuilder(cfg.esdf)}

    cam_names = ("zed_left",) if args.cameras == "head" else CAMERAS
    attached = None
    placed_since = None
    open_streak = 0
    fired = {"gripper": None, "placed": None, "reference": None}
    rows = []

    print(f"{'i':>3} {'그립폭':>7} {'쥠':>3} {'수평mm':>7} {'테두리아래':>10} "
          f"{'손-사과':>8} {'참값':>5} {'참조':>5} {'그리퍼해제':>10} {'팔낙관mm':>9}")

    for i, step in enumerate(run.steps):
        pose_scene(scene, step)
        q = np.asarray([scene.data.qpos[scene._qadr[j]] for j in DEFAULT_RBY1_JOINTS], float)
        caps = {c: scene.capture(c) for c in cam_names}
        head = caps["zed_left"]

        cams = []
        held_px = dest_px = None
        for c, fr in caps.items():
            Kc = np.asarray(fr.camera_intrinsics, np.float64)
            Tc = np.asarray(fr.T_base_cam, np.float64)
            dc = np.asarray(fr.depth, np.float64)
            m = ag._robot_mask_for(dc, Kc, Tc, fr.robot_state)
            m = np.zeros(dc.shape, bool) if m is None else np.asarray(m, bool)
            cams.append(CameraDepth(c, dc, Kc, Tc, robot_mask=m))
            if c == "zed_left":
                ids = np.asarray(fr.body_ids, np.int64)
                held_px, dest_px = ids == held_b, ids == dest_b

        # --- 파지와 두 해제 신호 -------------------------------------------------------
        wl = (scene.data.qpos[gadr["gripper_finger_l2"]]
              - scene.data.qpos[gadr["gripper_finger_l1"]]) * 1000
        hand = 0.5 * (scene.data.xpos[fl[0]] + scene.data.xpos[fl[1]])
        held_c = np.asarray(scene.data.xpos[held_b], np.float64)
        dest_c = np.asarray(scene.data.xpos[dest_b], np.float64)
        rim_z = float(scene.data.geom_xpos[rim_g][2] + scene.model.geom_size[rim_g][2])
        closed = bool(wl < GRIP_CLOSED_MM)
        grasped = bool(closed and np.linalg.norm(hand - held_c) * 1000 < HAND_NEAR_MM)

        radial = float(np.linalg.norm((held_c - dest_c)[:2])) * 1000.0
        below = (rim_z - float(held_c[2])) * 1000.0
        hand_gap = float(np.linalg.norm(hand - held_c)) * 1000.0
        placed = bool(radial <= PLACED_RADIAL_MM and below >= PLACED_BELOW_RIM_MM
                      and hand_gap > PLACED_HAND_GAP_MM)

        # 그리퍼 폴백 — `GraspLatch` 와 같은 규칙 (연속 N 프레임 열림).
        open_streak = open_streak + 1 if (attached is not None and not closed) else 0
        gripper_release = attached is not None and open_streak >= RELEASE_FRAMES

        if placed and placed_since is None:
            placed_since = i
        if placed and fired["placed"] is None:
            fired["placed"] = i
        if gripper_release and fired["gripper"] is None:
            fired["gripper"] = i

        # --- attach / detach ---------------------------------------------------------
        if grasped and attached is None and held_px is not None and held_px.any():
            pts = backproject(head.depth, np.asarray(head.camera_intrinsics, np.float64),
                              np.asarray(head.T_base_cam, np.float64), held_px)
            if len(pts):
                c = pts.mean(axis=0)
                sphere = Primitive(
                    type=PrimitiveType.SPHERE, center=c, orientation=np.eye(3),
                    dimensions=np.array([float(np.linalg.norm(pts - c, axis=1).max()), 0.0, 0.0]))
                attached = attach_from_target(
                    TargetGeometry(id=0, points=pts,
                                   point_indices=np.arange(len(pts), dtype=np.int64),
                                   centroid=c, bounding_geometry=sphere,
                                   attention_score=1.0, confidence=1.0),
                    robot_state=q, robot_model=robot, parent_link=PARENT_LINK,
                    allowed_contact_links=CONTACT_LINKS, label="held_apple")

        a_base = (None if attached is None else
                  attached_points_in_base(attached, robot_model=robot, robot_state=q))
        # `truth` 는 성공 판정이 참이 되면 파내기를 그만둔다. `stale` 은 계속한다.
        carve_pts = {"truth": (None if placed_since is not None else a_base),
                     "stale": a_base}
        # 목적지 라벨을 실제로 채운다 — `SafePolicy` 가 잠금이 목적지를 확정한 뒤 넘기는 것과
        # 같은 자리다. 참조 판정이 읽는 것이 이 라벨이다.
        K0 = np.asarray(head.camera_intrinsics, np.float64)
        T0 = np.asarray(head.T_base_cam, np.float64)
        dest_pts = (backproject(head.depth, K0, T0, dest_px)
                    if dest_px is not None and dest_px.any() else None)
        labelled = ({DESTINATION_LABEL: dest_pts}
                    if dest_pts is not None and len(dest_pts) else None)
        fields = {t: builders[t].update(cams, attached_points=carve_pts[t],
                                        labelled_points=labelled)
                  for t in builders}

        # 참조 구현을 **그 프레임의 관측으로** 돌린다. 잠금이 그리퍼와 AND 로 묶으므로
        # 여기서도 같은 규칙을 쓴다 — 기하만으로는 쥔 채로 풀리면 안 된다.
        class _Stub:
            attached = None
            esdf = None
            destination_label = None
            robot_state = q
        stub = _Stub()
        stub.attached, stub.esdf = attached, fields["truth"]
        stub.destination_label = DESTINATION_LABEL if labelled else None
        ref_geom = bool(attached is not None and ref_placed_fn({}, stub))
        ref_placed = bool(ref_geom and not closed)

        # --- 늦은 detach 의 값 — 팔 구가 얼마나 낙관적이 되는가 ---------------------------
        arm_c, arm_r = robot.sphere_centers_numeric(q)
        clear = {t: np.asarray(f.distance(arm_c), np.float64) - arm_r - args.esdf_margin
                 for t, f in fields.items()}
        arm_opt = float((clear["stale"] - clear["truth"]).max()) * 1000.0

        if ref_placed and fired.get("reference") is None:
            fired["reference"] = i

        rows.append(dict(i=i, t_step=int(step.t_step), grip_mm=float(wl), grasped=grasped,
                         ref_geom=ref_geom, ref_placed=ref_placed,
                         attached=attached is not None, radial_mm=radial, below_mm=below,
                         hand_gap_mm=hand_gap, placed=placed,
                         gripper_release=gripper_release, arm_optimism_mm=arm_opt,
                         arm_worst_truth_mm=float(clear["truth"].min()) * 1000.0,
                         arm_worst_stale_mm=float(clear["stale"].min()) * 1000.0))
        print(f"{i:>3} {wl:>7.1f} {'O' if grasped else '.':>3} {radial:>7.1f} "
              f"{below:>10.1f} {hand_gap:>8.1f} {'O' if placed else '.':>5} "
              f"{'O' if ref_placed else '.':>5} "
              f"{'O' if gripper_release else '.':>10} {arm_opt:>9.1f}")

    # 깜빡임 — 한 번 참이 된 뒤 거짓으로 돌아가는가.
    after = [r for r in rows if fired["placed"] is not None and r["i"] >= fired["placed"]]
    flicker = sum(1 for r in after if not r["placed"])
    summary = {
        "records": args.records, "frames": len(rows), "cameras": args.cameras,
        "placed_first_frame": fired["placed"], "gripper_release_frame": fired["gripper"],
        "reference_placed_frame": fired.get("reference"),
        "reference_vs_truth_frames": (
            None if fired.get("reference") is None or fired["placed"] is None
            else fired["reference"] - fired["placed"]),
        "reference_flicker_frames": sum(
            1 for r in rows
            if fired.get("reference") is not None and r["i"] >= fired["reference"]
            and not r["ref_placed"]),
        "lag_frames": (None if fired["gripper"] is None or fired["placed"] is None
                       else fired["gripper"] - fired["placed"]),
        "placed_flicker_frames": flicker,
        "placed_holds_for": len(after),
        "arm_optimism_max_mm": max((r["arm_optimism_mm"] for r in rows), default=0.0),
        "arm_optimism_after_placed_mm": max(
            (r["arm_optimism_mm"] for r in after), default=0.0),
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     indent=2, ensure_ascii=False))
    pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------ 그림
    fs = _style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.6, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.06, 1.06, 0.92], wspace=0.30, hspace=0.44)
    idx = [r["i"] for r in rows]

    def _marks(a):
        if fired["placed"] is not None:
            a.axvline(fired["placed"], color=fs.CATEGORICAL[2], lw=1.6, ls="-",
                      label=f"성공 판정 {fired['placed']}")
        if fired["gripper"] is not None:
            a.axvline(fired["gripper"], color=fs.CATEGORICAL[1], lw=1.6, ls="--",
                      label=f"그리퍼 폴백 {fired['gripper']}")

    # ① 세 조건
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(idx, [r["radial_mm"] for r in rows], "-o", ms=3.5, color=fs.CATEGORICAL[0],
            label="사과–바구니 수평거리")
    ax.axhline(PLACED_RADIAL_MM, color=fs.CATEGORICAL[0], ls=":", lw=1.2)
    ax.plot(idx, [r["below_mm"] for r in rows], "-s", ms=3.5, color=fs.CATEGORICAL[3],
            label="테두리 아래 깊이")
    ax.axhline(PLACED_BELOW_RIM_MM, color=fs.CATEGORICAL[3], ls=":", lw=1.2)
    ax.plot(idx, [r["hand_gap_mm"] for r in rows], "-^", ms=3.5, color=fs.CATEGORICAL[5],
            label="손–사과 거리")
    ax.axhline(PLACED_HAND_GAP_MM, color=fs.CATEGORICAL[5], ls=":", lw=1.2)
    _marks(ax)
    ax.set_xlabel("프레임"); ax.set_ylabel("[mm]")
    ax.set_title("① 성공 판정의 세 조건 (점선 = 문턱)", fontsize=10.5, color=fs.INK)
    ax.legend(fontsize=7.8, frameon=False, ncol=2)

    # ② 두 신호
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.step(idx, [1 if r["attached"] else 0 for r in rows], where="mid",
             color=fs.CATEGORICAL[6], lw=1.8, label="attach 상태")
    ax2.step(idx, [0.9 if r["placed"] else 0 for r in rows], where="mid",
             color=fs.CATEGORICAL[2], lw=1.6, label="성공 판정")
    ax2.step(idx, [0.85 if r["ref_placed"] else 0 for r in rows], where="mid",
             color=fs.CATEGORICAL[0], lw=1.6, label="참조 구현 (관측)")
    ax2.step(idx, [0.8 if r["gripper_release"] else 0 for r in rows], where="mid",
             color=fs.CATEGORICAL[1], lw=1.6, ls="--", label="그리퍼 폴백")
    ax2.set_ylim(-0.1, 1.3); ax2.set_yticks([])
    ax2.set_xlabel("프레임")
    ax2.set_title("② 세 신호는 언제 켜지는가", fontsize=10.5, color=fs.INK)
    ax2.legend(fontsize=8.0, frameon=False, loc="upper left")

    # ③ 늦은 detach 의 값
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(idx, [r["arm_optimism_mm"] for r in rows], "-o", ms=4,
             color=fs.CATEGORICAL[7])
    ax3.axhline(0, color=fs.INK, lw=1.0)
    _marks(ax3)
    ax3.set_xlabel("프레임"); ax3.set_ylabel("팔 구의 낙관 [mm]")
    ax3.set_title("③ detach 가 안 오면 — 파내기가 실제 장애물을 계속 지운다",
                  fontsize=10.0, color=fs.INK)
    ax3.legend(fontsize=8.0, frameon=False)

    # ④ 팔 최악 여유거리
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(idx, [r["arm_worst_truth_mm"] for r in rows], "-o", ms=3.5,
             color=fs.CATEGORICAL[2], label="성공 판정에 맞춰 detach")
    ax4.plot(idx, [r["arm_worst_stale_mm"] for r in rows], "--s", ms=3.5,
             color=fs.CATEGORICAL[7], label="detach 안 함")
    ax4.axhline(0, color=fs.INK, lw=1.0)
    _marks(ax4)
    ax4.set_xlabel("프레임"); ax4.set_ylabel("팔 구 최악 여유거리 [mm]")
    ax4.set_title("④ 팔이 받는 답이 달라진다", fontsize=10.0, color=fs.INK)
    ax4.legend(fontsize=8.0, frameon=False)

    # ⑤ 표
    ax5 = fig.add_subplot(gs[:, 2])
    ax5.axis("off")
    ax5.set_title("⑤ 결과", fontsize=10.5, color=fs.INK)
    trows = [["무엇", "값"],
             ["기록", args.records],
             ["프레임", f"{len(rows)}"],
             ["성공 판정 첫 참 (참값)", f"{summary['placed_first_frame']}"],
             ["참조 구현 (관측)", f"{summary['reference_placed_frame']}"],
             ["  참값 대비", f"{summary['reference_vs_truth_frames']} 프레임"],
             ["  깜빡임", f"{summary['reference_flicker_frames']} 프레임"],
             ["그리퍼 폴백 발동", f"{summary['gripper_release_frame']}"],
             ["차이", f"{summary['lag_frames']} 프레임"],
             ["성공 판정 깜빡임", f"{summary['placed_flicker_frames']} / "
                              f"{summary['placed_holds_for']} 프레임"],
             ["detach 없을 때 팔 낙관", f"{summary['arm_optimism_max_mm']:+,.1f} mm"],
             ["  성공 판정 이후만", f"{summary['arm_optimism_after_placed_mm']:+,.1f} mm"]]
    t = ax5.table(cellText=trows, colWidths=[0.56, 0.44], loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(8.6); t.scale(1, 1.6)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if r == 0:
            cell.set_facecolor("#ecebe7"); cell.set_text_props(weight="bold")

    fs.style_axes(fig, [ax, ax2, ax3, ax4])
    fig.suptitle("A3 — 잠금의 풀기: 그리퍼 폴백 대 성공 판정 주입"
                 f"   ({args.records}, {len(rows)} 프레임, {args.cameras} 카메라)",
                 fontsize=12.5, color=fs.INK)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
