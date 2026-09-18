"""쥔 물체가 장애물로 남아 있어서 생기는 피해 규모 — 관측만 한다, 고치지 않는다.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.trajopt.experiments.grasp_damage --frames 20

## 무엇을 재는가

`attached.py` 는 "파지가 닫히면 물체는 월드의 장애물이기를 그치고 움직이는 로봇의 일부가 된다"
고 쓰여 있는데, 그 배선이 비어 있다 (누구도 `attach()` 를 부르지 않고, trajopt 은 attached 를
읽지 않으며, ESDF 는 쥔 물체를 모른다). 그래서 **쥐고 있는 사과가 여전히 완전 여유거리를
요구하는 장애물**이다.

그 피해가 몇 mm 인지를 잰다. 고치기 전에 규모를 아는 것이 목적이다.

## 어떻게

파지 구간은 **기록된 그리퍼 폭**으로 판정한다 — 왼쪽 그리퍼가 90 mm 에서 66 mm 로 닫히고
(사과 지름 66.6 mm) 스텝 19 에 다시 열린다. AG3S 는 일부러 파지를 감지하지 않으므로
(`attach()` 의 "AG3S never calls this itself"), 이 판정은 **실험 하네스 쪽**에 둔다.

각 스텝에서 제약 모델의 구마다:

* `d_esdf`  — 최적화기가 실제로 읽는 거리장 값
* `d_apple` — 사과 표면까지의 해석적 거리 (구 근사, r = 33 mm)
* **사과가 이 구의 최근접 장애물인가** — `d_esdf >= d_apple - voxel`.
  `linearize._esdf_clearance` 가 target 을 식별할 때 쓰는 것과 **같은 판정**이다: 익명 필드를
  해석적 거리와 대조해 "지금 이 값을 만든 것이 그 물체인가" 를 본다.
* 적용된 마진 — E1 의 `target_link_margin` (attention target 기준) 또는 평 `esdf_margin`
* 잔차 `d_esdf - r_sphere - margin` — 음수면 위반

사과가 최근접 장애물이면서 잔차가 음수인 구가 **피해**다. 그 구들은 사과를 쥐고 있어서 가까운
것이지 충돌하고 있는 것이 아니다.
"""

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")
#: 사과 메시의 bounding half-extent 는 [31.1, 33.3, 33.3] mm. 구로 근사한다.
APPLE_R = 0.0333
#: 왼쪽 그리퍼가 이 폭 아래로 닫히고 손이 사과 곁에 있으면 쥔 것으로 본다.
GRIP_CLOSED_MM = 80.0
HAND_NEAR_MM = 70.0


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments import figstyle
    figstyle.use_korean()


def main() -> None:
    _style()
    import mujoco

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.pipeline import AG3S
    from benchmark.ag3s.experiments.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene
    from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS
    from benchmark.trajopt.config import TrajOptConfig
    from benchmark.trajopt.linearize import CollisionLinearizer, scene_from_constraint_set
    from benchmark.trajopt.types import ChunkLayout

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--attention", default="attention_step1_run0004.npz")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/step-01-attention.json")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--esdf-margin", type=float, default=0.05)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--held", default="apple")
    ap.add_argument("--phase-boundaries", type=int, nargs=3, default=(24, 56, 72))
    ap.add_argument("--attach", action="store_true",
                    help="그리퍼가 닫히면 `AG3S.attach()` 를 부르고 왼팔을 manipulator 로 지정한다. "
                         "이것이 F11 수정을 실제로 발동시키는 스위치다 — 없으면 조작 대상이 "
                         "주목 대상과 같아 예전 동작 그대로다.")
    ap.add_argument("--out-npz", default=None, help="스텝별 잔차/귀속을 npz 로 남긴다 (그림용)")
    args = ap.parse_args()

    run = load_run(args.records, limit=args.frames)
    blob = np.load(args.attention, allow_pickle=False)
    cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    att_block = np.asarray(blob["attention"], np.float32)
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    ci = [str(c) for c in blob["cameras"]].index("cam_high")

    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    ag = AG3S(AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False},
    }), robot_model=filter_robot, constraint_robot_model=robot,
        # 슬롯은 **생성 시점에** 예약된다 — `attach()` 가 나중에 늘릴 수 없다. primitive backend 의
        # attached 행이 쓰는 것이고 ESDF 경로는 그 행을 읽지 않지만, 예약이 없으면 `attach()` 자체가
        # 거절된다.
        attached_parent_links=(PARENT_LINK,) if args.attach else ())
    to_cfg = TrajOptConfig.from_dict({
        "collision": {"backend": "esdf", "esdf_margin": args.esdf_margin,
                      "use_support_planes": False}})
    lin = CollisionLinearizer(robot, ChunkLayout.rby1(DEFAULT_RBY1_JOINTS), 1)
    links = list(robot.sphere_link_names)

    held_b = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, args.held)
    fl = [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, x)
          for x in ("ee_finger_l1", "ee_finger_l2")]
    gadr = {n: scene.model.jnt_qposadr[
                mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ("gripper_finger_l1", "gripper_finger_l2")}

    from benchmark.ag3s.types import (
        Manipulator, Primitive, PrimitiveType, TargetGeometry)
    from benchmark.ag3s.experiments.attention_report import target_from_prompt
    from benchmark.trajopt.experiments.esdf_rollout import phase_for
    tgt_name = target_from_prompt(run.prompt)

    print("=" * 96)
    print(f"쥔 물체가 장애물로 남아서 생기는 피해 — {args.records}, 쥔 물체 = {args.held!r}")
    print(f"제약 모델 {robot.n_spheres} 구 (arms), 여유거리 {args.esdf_margin*1000:.0f} mm, "
          f"복셀 {args.voxel*1000:.0f} mm")
    print("=" * 96)
    print(f"{'i':>2} {'그립폭':>7} {'쥠':>3} {'target':>7} {'필드에':>6} {'d@중심':>8} "
          f"{'귀속구':>6} {'그중위반':>8} {'최악잔차':>9} {'최악링크':<20}")

    rec = []
    for i, step in enumerate(run.steps):
        pose_scene(scene, step)
        q = np.asarray([scene.data.qpos[scene._qadr[j]] for j in DEFAULT_RBY1_JOINTS], float)
        wl = (scene.data.qpos[gadr["gripper_finger_l2"]]
              - scene.data.qpos[gadr["gripper_finger_l1"]]) * 1000
        hand = 0.5 * (scene.data.xpos[fl[0]] + scene.data.xpos[fl[1]])
        held_c = np.asarray(scene.data.xpos[held_b], np.float64)
        grasped = bool(wl < GRIP_CLOSED_MM
                       and np.linalg.norm(hand - held_c) * 1000 < HAND_NEAR_MM)

        # 파지 신호는 **하네스가** 준다. AG3S 는 일부러 파지를 감지하지 않으므로
        # (`attach()` 의 "AG3S never calls this itself"), 재생에서는 기록된 그리퍼 폭으로
        # 판정하고 쥔 물체의 형상은 MuJoCo 참값에서 만든다. 실제 시스템에서는 무엇을 쥐었는지
        # 아는 과제 계층이 같은 자리에서 같은 호출을 한다. **관측보다 먼저** 불러야 그 프레임의
        # 제약이 이미 쥔 상태를 반영한다.
        if args.attach:
            _sync_attachment(ag, grasped, held_c, q, robot,
                             Primitive, PrimitiveType, TargetGeometry)

        head = scene.capture("zed_left")
        att = att_block[i, di, ai, cell["layer"], cell["head"], ci]
        # `active_manipulators` 가 비어 있으면 "아무도 아무것도 만질 수 없다" 이고, 그것이
        # 기본값이다. 지금까지 롤아웃이 이 값을 넘긴 적이 없어 **E1 의 접촉 권한이 통째로
        # 불활성**이었다 — 설계상 주입되는 입력이므로 코드 결함이 아니라 하네스의 공백이다.
        cs = ag.process(depth=head.depth, camera_intrinsics=head.camera_intrinsics,
                        T_base_cam=head.T_base_cam, attention_map=att,
                        robot_state=head.robot_state,
                        phase=phase_for(step.t_step, args.phase_boundaries),
                        active_manipulators=({Manipulator.LEFT} if ag.attached is not None
                                             else None))
        snap = scene_from_constraint_set(cs, lin.robot_radii, to_cfg)

        centres, radii = robot.sphere_centers_numeric(q)
        centres = np.asarray(centres, np.float64).reshape(1, -1, 3)

        resid = lin._esdf_clearance(centres, snap)[0, :, 0]      # 최적화기가 읽는 그 값
        d_esdf = np.asarray(snap.esdf.distance(centres.reshape(-1, 3)), np.float64)
        d_held = np.linalg.norm(centres[0] - held_c, axis=1) - APPLE_R

        # **필드에 그 물체가 있는가**를 먼저 본다. 물체 중심에서 질의했을 때 음수(= 안쪽)여야
        # 필드가 그것을 담고 있는 것이다. 들어올려진 뒤에는 손이 가려 사라지므로, 이 확인
        # 없이는 "사과 때문에 위반" 이라는 귀속 자체가 성립하지 않는다.
        d_at_held = float(snap.esdf.distance(held_c.reshape(1, 3))[0])
        in_field = d_at_held < 0.0

        # 귀속은 **양쪽**으로 본다. E1 원본의 한쪽 판정(`d >= d_target - tol`)은 target 이 필드에
        # 들어 있다는 전제에서만 옳다. 그 전제가 깨지면 "필드가 보는 장애물이 훨씬 멀다" 는
        # 경우까지 통과해 버린다 — 실제로 그래서 처음 측정이 틀렸다.
        is_held = in_field & (np.abs(d_esdf - d_held) <= args.voxel)

        bad = is_held & (resid < 0)
        worst = float(resid[bad].min()) if bad.any() else float("nan")
        wlink = links[int(np.argmin(np.where(bad, resid, np.inf)))] if bad.any() else "-"
        tn = "-" if cs.target is None else _nearest_body(scene, cs.target.centroid)
        print(f"{i:>2} {wl:>7.1f} {'O' if grasped else '.':>3} {tn:>7} "
              f"{'있음' if in_field else '없음':>6} {d_at_held*1000:>7.0f} "
              f"{int(is_held.sum()):>6} {int(bad.sum()):>8} "
              f"{worst*1000 if bad.any() else float('nan'):>9.1f} {wlink:<20}")
        rec.append(dict(i=i, wl=wl, grasped=grasped, target=tn, resid=resid.copy(),
                        is_held=is_held.copy(), d_held=d_held.copy(), in_field=in_field,
                        d_at_held=d_at_held, centres=centres[0].copy(), held_c=held_c.copy(),
                        radii=np.asarray(radii, float)))
    scene.close()
    if args.out_npz:
        out = {"links": np.asarray(links, object), "grasped": np.asarray([r["grasped"] for r in rec]),
               "in_field": np.asarray([r["in_field"] for r in rec]),
               "d_at_held": np.asarray([r["d_at_held"] for r in rec]),
               "target": np.asarray([r["target"] for r in rec], object)}
        for r in rec:
            out[f"resid_{r['i']}"] = r["resid"]
            out[f"is_held_{r['i']}"] = r["is_held"]
        np.savez(args.out_npz, n=len(rec), **out)
        print(f"wrote {args.out_npz}")
    _summary(rec, links, args)
    _figure(rec, links, args)



#: 쥔 물체를 매다는 링크와, 그 물체를 만져도 되는 링크들. 손목에 매다는 것은 물체가 손가락 하나가
#: 아니라 그리퍼 전체에 대해 고정이기 때문이다.
PARENT_LINK = "link_left_arm_6"
CONTACT_LINKS = ("ee_finger_l1", "ee_finger_l2")


def _sync_attachment(ag, grasped, held_centre, q, robot,
                     Primitive, PrimitiveType, TargetGeometry) -> None:
    """그리퍼 상태에 맞춰 `attach()` / `detach()` 를 부른다. 상태가 바뀔 때만.

    형상은 MuJoCo 참값에서 만든 반지름 `APPLE_R` 의 구다. 파지 순간의 grounding 을 쓸 수 없는
    이유가 F11 그 자체다 — 그때 attention 은 이미 목적지를 보고 있어서 `cs.target` 은 바구니다.
    실제 시스템에서 이 자리를 채우는 것은 무엇을 쥐라고 명령했는지 아는 과제 계층이고, 여기서
    그 역할은 하네스가 한다.
    """
    if grasped and ag.attached is None:
        sphere = Primitive(
            type=PrimitiveType.SPHERE,
            center=np.asarray(held_centre, np.float64),
            orientation=np.eye(3),
            dimensions=np.array([APPLE_R, 0.0, 0.0]),
        )
        target = TargetGeometry(
            id=0,
            points=np.asarray(held_centre, np.float64).reshape(1, 3),
            point_indices=np.zeros(1, np.int64),
            centroid=np.asarray(held_centre, np.float64),
            bounding_geometry=sphere,
            attention_score=1.0,
            confidence=1.0,
        )
        ag.attach(target, robot_state=q, parent_link=PARENT_LINK,
                  allowed_contact_links=CONTACT_LINKS, label="held_apple")
    elif not grasped and ag.attached is not None:
        ag.detach()


def _nearest_body(scene, centroid):
    import mujoco
    from benchmark.ag3s.experiments.mujoco_source import is_robot_body
    best, bd = "-", 1e9
    for b in range(scene.model.nbody):
        nm = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not nm or is_robot_body(nm) or any(
                k in nm for k in ("table", "shelf", "floor", "world", "ground",
                                  "com_target", "_ee_target", "office")):
            continue
        dd = float(np.linalg.norm(np.asarray(centroid) - scene.data.xpos[b]))
        if dd < bd:
            best, bd = nm, dd
    return best


def _summary(rec, links, args):
    g = [r for r in rec if r["grasped"]]
    obst = [r for r in g if r["in_field"]]      # 쥐었는데 아직 필드에 있다 -> 장애물로 남는다
    hole = [r for r in g if not r["in_field"]]  # 쥐었고 필드에서 사라졌다 -> 구멍
    print("\n" + "=" * 100)
    print(f"파지 구간: 스텝 {[r['i'] for r in g]}  ({len(g)} 스텝)")
    print(f"  그중 쥔 물체가 필드에 남아 있는 스텝 : {[r['i'] for r in obst]}")
    print(f"  그중 쥔 물체가 필드에서 사라진 스텝  : {[r['i'] for r in hole]}")

    print("\n[국면 1] 쥔 물체가 장애물로 남아 여유거리를 요구한 구간")
    per = {}
    for r in obst:
        bad = r["is_held"] & (r["resid"] < 0)
        for si in np.flatnonzero(bad):
            per.setdefault(links[si], []).append(float(r["resid"][si]))
    if per:
        print(f"   {'링크':<24} {'구-스텝':>7} {'최악':>10} {'평균':>10}")
        for k in sorted(per, key=lambda k: min(per[k])):
            v = per[k]
            print(f"   {k:<24} {len(v):>7} {min(v)*1000:>7.1f} mm {np.mean(v)*1000:>7.1f} mm")
        allv = [x for v in per.values() for x in v]
        print(f"   전체 최악 {min(allv)*1000:.1f} mm,  위반 구-스텝 {len(allv)} 개")
    else:
        print("   위반 없음")

    print("\n[국면 2] 쥔 물체가 필드에서 사라져 생긴 구멍")
    if hole:
        err = [(r["i"], r["d_at_held"]) for r in hole]
        print(f"   {'스텝':>5} {'d@물체중심':>11} {'참값':>8} {'오차':>9}")
        for i, dv in err:
            print(f"   {i:>5} {dv*1000:>10.0f} {-APPLE_R*1000:>7.0f} {(dv+APPLE_R)*1000:>8.0f} mm")
        w = max(dv for _, dv in err)
        print(f"   최대 오차 {(w+APPLE_R)*1000:.0f} mm — 실제로는 물체 안쪽인 지점을 "
              f"필드는 {w*1000:.0f} mm 빈 곳이라고 말한다")
    else:
        print("   해당 없음")

    tg = sorted({r["target"] for r in g})
    print(f"\n   파지 구간의 grounding target: {tg}  <- 사과가 아니므로 E1 완화가 걸리지 않는다")


def _figure(rec, links, args):
    import matplotlib.pyplot as plt
    g = [r for r in rec if r["grasped"]]
    obst = [r for r in g if r["in_field"]]
    hole = [r for r in g if not r["in_field"]]
    idx = [r["i"] for r in rec]
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.9))

    # (a) 실제 씬 — 쥔 물체가 아직 필드에 있는 마지막 스텝
    a = axs[0]
    r0 = obst[-1] if obst else g[0]
    c, hc = r0["centres"], r0["held_c"]
    bad = r0["is_held"] & (r0["resid"] < 0)
    a.scatter(c[~r0["is_held"], 0], c[~r0["is_held"], 2], s=16, c="0.78",
              label="사과가 최근접 아님")
    a.scatter(c[r0["is_held"] & ~bad, 0], c[r0["is_held"] & ~bad, 2], s=34, c="tab:green",
              label="사과 최근접 · 통과")
    a.scatter(c[bad, 0], c[bad, 2], s=80, c="tab:red", marker="X", label="사과 최근접 · 위반")
    th = np.linspace(0, 2 * np.pi, 90)
    a.plot(hc[0] + APPLE_R * np.cos(th), hc[2] + APPLE_R * np.sin(th), "-",
           color="darkred", lw=2.2)
    a.plot(hc[0] + (APPLE_R + args.esdf_margin) * np.cos(th),
           hc[2] + (APPLE_R + args.esdf_margin) * np.sin(th), "--", color="darkred", lw=1.4)
    a.text(hc[0], hc[2] + APPLE_R + args.esdf_margin + 0.012,
           f"사과 + 여유거리 {args.esdf_margin*1000:.0f} mm", ha="center", fontsize=9,
           color="darkred")
    a.set_title(f"(a) 실제 씬 — 스텝 {r0['i']} (쥐었고 아직 필드에 있다)\n"
                "빨간 X = 쥔 사과 때문에 위반인 구", fontsize=11)
    a.set_xlabel("x [m] — 앞쪽 →"); a.set_ylabel("z [m] — 위 ↑")
    a.set_aspect("equal"); a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.3)
    a.set_xlim(hc[0] - 0.26, hc[0] + 0.26); a.set_ylim(hc[2] - 0.26, hc[2] + 0.26)

    # (b) 두 국면
    a = axs[1]
    dd = [r["d_at_held"] * 1000 for r in rec]
    for r in g:
        a.axvspan(r["i"] - 0.5, r["i"] + 0.5,
                  color=("tab:red" if r["in_field"] else "tab:purple"), alpha=0.16)
    a.plot(idx, dd, "o-", color="k", lw=2, ms=5.5, label="필드가 말하는 사과 중심 거리")
    a.axhline(-APPLE_R * 1000, color="tab:green", ls="--", lw=2,
              label=f"참값 {-APPLE_R*1000:.0f} mm (물체 안쪽)")
    a.axhline(0, color="0.5", lw=1)
    a.set_title("(b) 쥔 사과는 들어올려지는 순간 필드에서 사라진다\n"
                "빨강 = 쥐었고 필드에 있음   보라 = 쥐었는데 필드에 없음", fontsize=11)
    a.set_xlabel("스텝"); a.set_ylabel("d @ 사과 중심 [mm]")
    a.legend(fontsize=9, loc="upper left"); a.grid(alpha=0.3)

    # (c) 표
    a = axs[2]; a.axis("off")
    per = {}
    for r in obst:
        b = r["is_held"] & (r["resid"] < 0)
        for si in np.flatnonzero(b):
            per.setdefault(links[si], []).append(float(r["resid"][si]))
    rows = [["국면", "스텝", "무엇이 문제인가", "규모"]]
    rows.append(["1 · 장애물로 남음", f"{[r['i'] for r in obst]}",
                 "쥔 사과가 여유거리를 요구", 
                 (f"최악 {min(x for v in per.values() for x in v)*1000:.0f} mm"
                  if per else "위반 없음")])
    if hole:
        w = max(r["d_at_held"] for r in hole)
        rows.append(["2 · 필드에 구멍", f"{[r['i'] for r in hole]}",
                     "사과가 있는 자리를 빈 곳이라 함", f"오차 최대 {(w+APPLE_R)*1000:.0f} mm"])
    t = a.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1.0, 2.6)
    for j in range(4):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    t[(1, 0)].set_facecolor("#f7d6d6")
    if hole:
        t[(2, 0)].set_facecolor("#e3d6f7")
    a.set_title("(c) 파지 구간에 두 국면이 연달아 온다\n"
                f"grounding target 은 {sorted({r['target'] for r in g})} — 사과가 아니라 "
                "E1 완화가 안 걸린다", fontsize=11, y=0.86)

    fig.suptitle("쥔 물체를 로봇으로 취급하지 않아서 생기는 피해 — run_0004 파지 구간 관측",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    o = OUT / f"grasp-damage-{args.records[-4:]}.png"
    fig.savefig(o, dpi=110, bbox_inches="tight")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
