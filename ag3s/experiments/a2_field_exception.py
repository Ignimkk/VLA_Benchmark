"""A2 — 쥔 물체를 거리장 밖에 두고 있는 것은 **무엇인가**, 그리고 얼마나 파내야 하는가.

E3(쥔 물체가 optimizer 에 도달하지 않는다)의 소비부가 들어가면서 쥔 물체는 **로봇 쪽
질의점**이 됐다 (F19 에 따라 primitive 가 아니라 점 기반). 그러면 장애물 쪽에서는 빠져야
한다. 안 빠지면 물체가 자기 자신에게 부딪히고, `linearize.py` 의 `_esdf_clearance` 가 쥔
물체의 점에는 완화를 주지 않으므로 (fail-closed 로 전체 마진) 그 행은 **어떤 관절 해로도 못
푼다** — 물체가 손에 강체로 붙어 있기 때문이다.

### 이 스크립트가 뒤집은 것

계획서에는 *"지금 빠지는 것은 자기 필터의 과잉 삭제라는 우연(F12)"* 이라고 적혀 있었다.
**대조군을 만들어 보니 아니었다** (2026-09-17):

    쥔 동안 보인 사과 픽셀      143,942   (3 카메라)
    그중 마스크가 지운 것        100.0 %   <- F12 의 관찰은 정확하다
    그런데 안 지웠을 때 점유     최대 4 복셀만 다르다

마스크를 걷어내도 사과는 필드로 안 돌아온다. TSDF 가 가중평균이라 **움직이는 표면**은 어느
복셀에서도 점유로 뒤집힐 만큼 가중치를 못 모으고, 새 관측이 광선을 따라 앞을 비워 옛 표면을
지우기까지 한다. **원인 서술이 틀렸던 것이고 F12 행을 고쳤다.**

진짜 문제는 **잔상**이다 — 사과가 테이블에 놓여 있던 동안 남긴 표면 위에 서서, 파지 직후
2 프레임 동안 자기 질의점이 **−72.7 mm** 를 읽는다.

### 무엇을 재는가

필드를 여럿 만든다. 관측과 적분 설정은 같고 다음만 다르다:

* `now` — AG3S 가 실제로 쓰는 마스크 (오늘의 동작)
* `unmasked` — 쥔 물체의 픽셀만 마스크에서 뺀 것 (자기 필터가 원인이라는 가설의 대조군)
* `carve<d>` — `now` 에 쥔 물체를 팽창 `d` 로 **파낸** 것 (수정안)

그리고 **E1(조작 대상을 파내면 손끝뿐 아니라 전신에게 사라진다) 검사를 함께 한다** —
권한 없는 링크의 여유거리가 파내기로 얼마나 **커졌는가**. 커지는 것은 낙관이고 위험한 쪽이다.
팽창 2 에서 +64.5 mm 로 절벽이 있어 **팽창 1 이 상한**이 됐다.

실행:
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -m \
        benchmark.ag3s.experiments.a2_field_exception --records run_0004 --frames 24 \
        --cameras all --carve-dilate 0 1 2 3
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures/a2-field-exception.png")
PARENT_LINK = "link_left_arm_6"
CONTACT_LINKS = frozenset({"ee_finger_l1", "ee_finger_l2"})
CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
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
    return figstyle


def backproject(depth, K, T_base_cam, mask):
    """`mask` 인 픽셀을 base 좌표계 점으로. 쥔 물체의 **관측된** 표면점을 얻는 데 쓴다."""
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
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--esdf-margin", type=float, default=0.05)
    ap.add_argument("--point-voxel", type=float, default=0.010)
    ap.add_argument("--carve-dilate", type=int, nargs="+", default=(0, 1, 2),
                    help="쥔 물체를 파낼 때의 팽창(복셀). 여럿 주면 전부 잰다 — 필드의 표면은 "
                         "TSDF 의 영교차가 정해서 관측 점이 떨어진 복셀과 꼭 같지 않고, "
                         "얼마나 벌려야 하는지는 씬이 정한다")
    ap.add_argument("--cameras", choices=("head", "all"), default="head",
                    help="head = 머리 하나. all = 손목 둘을 더한다 — F12 가 손목 카메라는 "
                         "사과를 15,910 px 로 본다고 적었고, 픽셀 수가 적어서 적분이 안 되는 "
                         "것이라면 여기서 갈린다")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--out-json", default="benchmark/ag3s/docs/figures/a2-field-exception.json")
    args = ap.parse_args()

    from benchmark.ag3s.attached import attach_from_target
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.esdf import CameraDepth, EsdfBuilder
    from benchmark.ag3s.experiments.grounding_report import (
        build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene
    from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS
    from benchmark.ag3s.types import Primitive, PrimitiveType, TargetGeometry

    run = load_run(args.records, limit=args.frames or None)
    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    robot = build_constraint_robot_model(scene)

    held_b = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, args.held)
    fl = [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, x)
          for x in ("ee_finger_l1", "ee_finger_l2")]
    gadr = {n: scene.model.jnt_qposadr[
                mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ("gripper_finger_l1", "gripper_finger_l2")}

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False},
    })
    from benchmark.ag3s.pipeline import AG3S
    ag = AG3S(cfg, robot_model=filter_robot, constraint_robot_model=robot)

    # 필드들. `now`/`unmasked` 는 **로봇 마스크만** 다르고, `carve<d>` 는 `now` 와 같은
    # 마스크에 쥔 물체를 팽창 d 로 **파낸** 것이다 (A2 의 수정안).
    import dataclasses as _dc
    builders = {"now": EsdfBuilder(cfg.esdf), "unmasked": EsdfBuilder(cfg.esdf)}
    carve_cfgs = {f"carve{d}": _dc.replace(cfg.esdf, attached_dilate_voxels=int(d))
                  for d in args.carve_dilate}
    builders.update({k: EsdfBuilder(c) for k, c in carve_cfgs.items()})
    carve_tags = list(carve_cfgs)

    attached = None
    rows = []
    print(f"{'i':>3} {'그립폭':>7} {'쥠':>3} {'사과px':>7} {'그중삭제':>8} "
          f"{'질의점':>6} {'점유now':>7} {'점유unm':>8} "
          f"{'d@중심n':>8} {'d@중심u':>8} {'now최악':>9} {'unm최악':>10}")

    cam_names = ("zed_left",) if args.cameras == "head" else CAMERAS
    for i, step in enumerate(run.steps):
        pose_scene(scene, step)
        q = np.asarray([scene.data.qpos[scene._qadr[j]] for j in DEFAULT_RBY1_JOINTS], float)
        caps = {c: scene.capture(c) for c in cam_names}
        head = caps["zed_left"]
        K = np.asarray(head.camera_intrinsics, np.float64)
        T = np.asarray(head.T_base_cam, np.float64)
        d_img = np.asarray(head.depth, np.float64)

        cams = {"now": [], "unmasked": []}
        n_held = n_deleted = 0
        for c, fr in caps.items():
            Kc = np.asarray(fr.camera_intrinsics, np.float64)
            Tc = np.asarray(fr.T_base_cam, np.float64)
            dc = np.asarray(fr.depth, np.float64)
            # AG3S 가 실제로 쓰는 마스크. F12 는 이것이 사과를 지운다고 적었다.
            m = ag._robot_mask_for(dc, Kc, Tc, fr.robot_state)
            m = np.zeros(dc.shape, bool) if m is None else np.asarray(m, bool)
            hp = np.asarray(fr.body_ids, np.int64) == held_b
            n_held += int(hp.sum())
            n_deleted += int((hp & m).sum())
            cams["now"].append(CameraDepth(c, dc, Kc, Tc, robot_mask=m))
            # 대조군: 쥔 물체의 픽셀만 마스크에서 빼낸다 -> 사과가 필드에 남을 기회를 준다.
            cams["unmasked"].append(CameraDepth(c, dc, Kc, Tc, robot_mask=m & ~hp))
            if c == "zed_left":
                held_px = hp

        # 파지 판정은 **하네스**가 준다 — AG3S 는 일부러 파지를 감지하지 않는다.
        wl = (scene.data.qpos[gadr["gripper_finger_l2"]]
              - scene.data.qpos[gadr["gripper_finger_l1"]]) * 1000
        hand = 0.5 * (scene.data.xpos[fl[0]] + scene.data.xpos[fl[1]])
        held_c = np.asarray(scene.data.xpos[held_b], np.float64)
        grasped = bool(wl < GRIP_CLOSED_MM
                       and np.linalg.norm(hand - held_c) * 1000 < HAND_NEAR_MM)

        if grasped and attached is None and int(held_px.sum()) > 0:
            # 쥔 물체의 **관측된** 표면점으로 붙인다. 중심 한 점으로 붙이면 F19 가 잰 점 기반의
            # 성질(꼭지·잎까지 덮는 구를 피한다)이 사라져 측정이 딴 것을 재게 된다.
            pts = backproject(d_img, K, T, held_px)
            if len(pts):
                # `bounding_geometry` 는 `attach_from_target` 이 요구하므로 채운다. 이 측정이
                # 쓰는 것은 **점**이고 primitive 는 아니다 (F19 — 단일 구는 여유를 먹는다).
                c = pts.mean(axis=0)
                sphere = Primitive(
                    type=PrimitiveType.SPHERE, center=c, orientation=np.eye(3),
                    dimensions=np.array([float(np.linalg.norm(pts - c, axis=1).max()), 0.0, 0.0]))
                tgt = TargetGeometry(id=0, points=pts,
                                     point_indices=np.arange(len(pts), dtype=np.int64),
                                     centroid=c, bounding_geometry=sphere,
                                     attention_score=1.0, confidence=1.0)
                attached = attach_from_target(
                    tgt, robot_state=q, robot_model=robot, parent_link=PARENT_LINK,
                    allowed_contact_links=CONTACT_LINKS, label="held_apple",
                    point_voxel=args.point_voxel)
        elif not grasped and attached is not None:
            attached = None

        attached_base = None
        if attached is not None:
            from benchmark.ag3s.attached import attached_points_in_base
            attached_base = attached_points_in_base(
                attached, robot_model=robot, robot_state=q)

        # `carve*` 는 `now` 와 같은 관측을 쓴다. 파내기는 적분 **뒤** 점유 단계에서 일어나므로
        # 쥔 물체의 지금 자리를 알아야 하는데, 그것은 attach 가 걸린 뒤에만 있다.
        fields = {}
        for tag in builders:
            src = "unmasked" if tag == "unmasked" else "now"
            fields[tag] = builders[tag].update(
                cams[src],
                attached_points=(attached_base if tag in carve_tags else None))

        # **대조군이 실제로 무언가를 바꿨는가**를 먼저 본다. 마스크에서 픽셀을 빼도 그 패치가
        # TSDF 의 가중치 문턱을 못 넘으면 점유가 안 생기고, 그러면 "차이가 없다" 는 결과가
        # 대조군이 성립했다는 뜻이 아니라 **대조군이 아무 일도 안 했다**는 뜻이 된다.
        occ = {tag: int(f.stats.get("n_occupied", -1)) for tag, f in fields.items()}
        d_centroid = {tag: float(f.distance(held_c.reshape(1, 3))[0])
                      for tag, f in fields.items()}

        # **E1 검사** — 파내기는 모두에게 지운다. 쥔 물체를 파내면서 팔·손가락이 볼 장애물까지
        # 지웠다면, 그 구들의 여유거리가 `now` 보다 **커진다**. 커지는 것은 낙관이고 위험한
        # 쪽이다. 여기서 재지 않으면 A2 의 수정이 E1 을 다시 저지른다.
        arm_c, arm_r = robot.sphere_centers_numeric(q)
        arm_clear = {tag: (np.asarray(f.distance(arm_c), np.float64) - arm_r
                           - args.esdf_margin) for tag, f in fields.items()}
        # **권한 있는 손가락과 나머지를 가른다.** 사과를 쥔 손끝에게 사과가 사라지는 것은
        # 의도다 (E1 의 접촉 권한). 팔꿈치·전완·반대팔에게까지 사라지는 것이 E1 이 금지한 것이다.
        auth = np.array([n in CONTACT_LINKS for n in robot.sphere_link_names])
        arm_opt = {t: float((arm_clear[t] - arm_clear["now"]).max()) for t in carve_tags}
        arm_opt_unauth = {t: float((arm_clear[t] - arm_clear["now"])[~auth].max())
                          for t in carve_tags}

        worst = {tag: float("nan") for tag in fields}
        d_min = {tag: float("nan") for tag in fields}
        n_q = 0
        if attached is not None and attached.points is not None and len(attached.points):
            # parent link 프레임 -> base. 쥔 물체는 손에 강체로 붙어 있으므로 FK 한 번이면 된다.
            Tp = np.asarray(robot.link_pose(q, attached.parent_link), np.float64)
            p_base = np.asarray(attached.points, np.float64) @ Tp[:3, :3].T + Tp[:3, 3]
            n_q = len(p_base)
            for tag, f in fields.items():
                # 반지름 0, 마진은 전체 50 mm (쥔 물체의 점은 완화를 안 받는다).
                dd = f.distance(p_base)
                d_min[tag] = float(dd.min())
                worst[tag] = float((dd - args.esdf_margin).min())

        rows.append(dict(i=i, grip_mm=float(wl), grasped=grasped, n_held_px=n_held,
                         n_deleted_px=n_deleted, n_query=n_q,
                         occ_now=occ["now"], occ_unmasked=occ["unmasked"],
                         d_centroid_now_mm=d_centroid["now"] * 1000.0,
                         d_centroid_unmasked_mm=d_centroid["unmasked"] * 1000.0,
                         d_min_now_mm=d_min["now"] * 1000.0,
                         d_min_unmasked_mm=d_min["unmasked"] * 1000.0,
                         worst_now_mm=worst["now"] * 1000.0,
                         worst_unmasked_mm=worst["unmasked"] * 1000.0,
                         carved={t: worst[t] * 1000.0 for t in carve_tags},
                         carved_voxels={
                             t: int(fields[t].stats.get("n_attached_voxels_carved", 0))
                             for t in carve_tags},
                         arm_worst_now_mm=float(arm_clear["now"].min()) * 1000.0,
                         arm_optimism_mm={t: arm_opt[t] * 1000.0 for t in carve_tags},
                         arm_optimism_unauth_mm={t: arm_opt_unauth[t] * 1000.0
                                                 for t in carve_tags}))
        print(f"{i:>3} {wl:>7.1f} {'O' if grasped else '.':>3} {n_held:>7} "
              f"{n_deleted:>8} {n_q:>6} "
              f"{occ['now']:>7} {occ['unmasked']:>8} "
              f"{d_centroid['now']*1000:>8.1f} {d_centroid['unmasked']*1000:>8.1f} "
              f"{worst['now']*1000:>9.1f} {worst['unmasked']*1000:>10.1f}  "
              + "  ".join(f"{t}:{worst[t]*1000:>7.1f}"
                          f"({fields[t].stats.get('n_attached_voxels_carved', 0)})"
                          for t in carve_tags))

    held_rows = [r for r in rows if r["n_query"] > 0]
    vis = [r for r in rows if r["n_held_px"] > 0]
    summary = {
        "records": args.records, "frames": len(rows), "held": args.held,
        "cameras": args.cameras,
        "esdf_margin_mm": args.esdf_margin * 1000.0,
        "grasp_frames": [r["i"] for r in held_rows],
        "held_pixels_seen_frames": len(vis),
        "held_pixels_total": sum(r["n_held_px"] for r in vis),
        "held_pixels_deleted": sum(r["n_deleted_px"] for r in vis),
        "deleted_fraction": (sum(r["n_deleted_px"] for r in vis)
                             / max(sum(r["n_held_px"] for r in vis), 1)),
        # F12 의 주장은 **쥔 동안** 100 % 삭제다. 전체 평균에는 파지 전(0 % 삭제) 프레임이
        # 섞여 있어 그 주장을 흐린다.
        "deleted_fraction_while_held": (
            sum(r["n_deleted_px"] for r in held_rows)
            / max(sum(r["n_held_px"] for r in held_rows), 1)),
        "occupancy_delta": [r["occ_unmasked"] - r["occ_now"] for r in rows],
        "centroid_delta_mm": [r["d_centroid_now_mm"] - r["d_centroid_unmasked_mm"]
                              for r in rows],
        "carve": {t: {"worst_mm": min((r["carved"][t] for r in held_rows),
                                      default=float("nan")),
                      "n_violating_frames": sum(1 for r in held_rows if r["carved"][t] < 0),
                      "voxels_total": sum(r["carved_voxels"][t] for r in held_rows),
                      # E1 검사: 팔 구의 여유거리가 파내기로 **얼마나 커졌는가**.
                      # 0 이어야 한다 — 커지면 실제 장애물을 지운 것이다.
                      "arm_optimism_max_mm": max((r["arm_optimism_mm"][t] for r in rows),
                                                 default=0.0),
                      # 권한 **없는** 링크의 낙관. 이것이 0 이 아니면 E1 을 다시 저지른 것이다.
                      "arm_optimism_unauthorized_max_mm": max(
                          (r["arm_optimism_unauth_mm"][t] for r in rows), default=0.0)}
                  for t in carve_tags},
        "now": {"worst_mm": min((r["worst_now_mm"] for r in held_rows), default=float("nan")),
                "n_violating_frames": sum(1 for r in held_rows if r["worst_now_mm"] < 0)},
        "unmasked": {"worst_mm": min((r["worst_unmasked_mm"] for r in held_rows),
                                     default=float("nan")),
                     "n_violating_frames": sum(1 for r in held_rows
                                               if r["worst_unmasked_mm"] < 0)},
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     indent=2, ensure_ascii=False))
    pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------ 그림
    fs = _style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.8, 8.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 0.98],
                          height_ratios=[1, 1], wspace=0.30, hspace=0.46)
    idx = [r["i"] for r in rows]
    held_span = [r["i"] for r in rows if r["n_query"] > 0]
    cols = {t: fs.CATEGORICAL[k % len(fs.CATEGORICAL)]
            for k, t in enumerate(carve_tags, start=3)}

    def _shade(a):
        for k in held_span:
            a.axvspan(k - 0.5, k + 0.5, color=fs.CATEGORICAL[0], alpha=0.10, lw=0)

    # ① 사과가 필드 안에 있는가 — 마스크는 이것을 못 바꾼다
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(idx, [r["d_centroid_now_mm"] for r in rows], "-o", ms=4,
            color=fs.CATEGORICAL[2], label="지금")
    ax.plot(idx, [r["d_centroid_unmasked_mm"] for r in rows], "--s", ms=3.5,
            color=fs.CATEGORICAL[7], label="사과를 안 지우면")
    ax.axhline(0, color=fs.INK, lw=1.2)
    _shade(ax)
    ax.set_xlabel("프레임  (음영 = 쥔 상태)"); ax.set_ylabel("사과 중심의 거리장 값 [mm]")
    ax.set_title("① 사과가 거리장 안에 있는가 (음수 = 있다)", fontsize=10.5, color=fs.INK)
    ax.legend(fontsize=8.4, frameon=False)

    # ② 쥔 물체 질의점의 최악 여유거리 — 파내기가 듣는가
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(idx, [r["worst_now_mm"] for r in rows], "-o", ms=4,
             color=fs.CATEGORICAL[2], label="파내기 없음 (지금)")
    for t in carve_tags:
        ax2.plot(idx, [r["carved"][t] for r in rows], "-^", ms=4, color=cols[t],
                 label=f"파냄 팽창 {t.replace('carve', '')}")
    ax2.axhline(0, color=fs.INK, lw=1.2)
    _shade(ax2)
    ax2.set_xlabel("프레임"); ax2.set_ylabel("쥔 물체 질의점의 최악 여유거리 [mm]")
    ax2.set_title("② 쥔 물체가 자기 자신에게 부딪히는가", fontsize=10.5, color=fs.INK)
    ax2.legend(fontsize=8.0, frameon=False, loc="lower right")

    # ③ 맞바꿈 — 얻는 것 대 E1 이 금지한 것
    ax3 = fig.add_subplot(gs[1, 0])
    xs = [int(t.replace("carve", "")) for t in carve_tags]
    ax3.plot(xs, [summary["carve"][t]["worst_mm"] for t in carve_tags], "-o", ms=6,
             color=fs.CATEGORICAL[0], label="쥔 물체 최악 여유거리 (클수록 좋다)")
    ax3.plot(xs, [summary["carve"][t]["arm_optimism_unauthorized_max_mm"] for t in carve_tags],
             "-s", ms=6, color=fs.CATEGORICAL[7],
             label="권한 없는 링크의 낙관 (작을수록 좋다)")
    ax3.axhline(summary["now"]["worst_mm"], color=fs.CATEGORICAL[2], ls=":", lw=1.4,
                label="파내기 없음")
    ax3.axhline(0, color=fs.INK, lw=1.0)
    ax3.set_xticks(xs); ax3.set_xlabel("팽창 (복셀)"); ax3.set_ylabel("[mm]")
    ax3.set_title("③ 맞바꿈 — 팽창 2 에서 E1 이 되살아난다", fontsize=10.5, color=fs.INK)
    ax3.legend(fontsize=8.0, frameon=False, loc="center left")

    # ④ 픽셀 — F12 재현
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar([r["i"] for r in vis], [r["n_held_px"] for r in vis],
            color=fs.CATEGORICAL[0], alpha=0.42, label="사과 픽셀 (보인다)")
    ax4.bar([r["i"] for r in vis], [r["n_deleted_px"] for r in vis],
            color=fs.CATEGORICAL[7], alpha=0.88, width=0.55, label="그중 마스크가 지운 것")
    _shade(ax4)
    ax4.set_xlabel("프레임"); ax4.set_ylabel("픽셀 수")
    ax4.set_title(f"④ F12 재현 — 쥔 동안 삭제율 "
                  f"{summary['deleted_fraction_while_held']*100:.1f} %, "
                  f"그런데 점유는 최대 "
                  f"{max(abs(r['occ_unmasked'] - r['occ_now']) for r in rows)} 복셀만 바뀐다",
                  fontsize=10.0, color=fs.INK)
    ax4.legend(fontsize=8.4, frameon=False)

    # ⑤ 표
    ax5 = fig.add_subplot(gs[:, 2])
    ax5.axis("off")
    ax5.set_title("⑤ 팽창을 얼마로 둘 것인가", fontsize=10.5, color=fs.INK)
    trows = [["팽창", "쥔 물체 최악", "권한없는 낙관", "파낸 복셀"],
             ["없음", f"{summary['now']['worst_mm']:+,.1f}", "—", "—"]]
    for t in carve_tags:
        c = summary["carve"][t]
        trows.append([t.replace("carve", ""), f"{c['worst_mm']:+,.1f}",
                      f"{c['arm_optimism_unauthorized_max_mm']:+,.1f}",
                      f"{c['voxels_total']}"])
    trows += [["", "", "", ""],
              ["카메라", args.cameras, "", ""],
              ["쥔 프레임", f"{len(held_span)} 개", "", ""],
              ["질의점", f"{rows[held_span[0]]['n_query'] if held_span else 0} 개", "", ""],
              ["쥔 동안 사과 픽셀",
               f"{sum(r['n_held_px'] for r in held_rows):,}", "", ""],
              ["그중 삭제",
               f"{summary['deleted_fraction_while_held']*100:.1f} %", "", ""]]
    t = ax5.table(cellText=trows, colWidths=[0.24, 0.28, 0.28, 0.20],
                  loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(8.4); t.scale(1, 1.58)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if r == 0:
            cell.set_facecolor("#ecebe7"); cell.set_text_props(weight="bold")

    fs.style_axes(fig, [ax, ax2, ax3, ax4])
    fig.suptitle("A2 — 쥔 물체를 필드 밖에 두는 것은 자기 필터가 아니다. 파내되 팽창 1 까지만"
                 f"   ({args.records}, {len(rows)} 프레임, {args.cameras} 카메라, "
                 f"마진 {args.esdf_margin*1000:.0f} mm)",
                 fontsize=12.5, color=fs.INK)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
