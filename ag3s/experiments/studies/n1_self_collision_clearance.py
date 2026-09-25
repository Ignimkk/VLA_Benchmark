"""N1 — self-collision 을 켠다면 무엇이 걸리는가. 쥔 물체와 로봇 자신의 여유거리.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.studies.n1_self_collision_clearance \\
        --records outputs/live_test/20260924_long16d/run_0000 \\
        --out benchmark/ag3s/docs/figures/r-16d

## 왜 이 스크립트가 새로 필요했나

**N1 은 "self-collision 제약이 `benchmark/trajopt/` 에 코드 0 줄로 아예 없다" 는 잠복
판정이다.** 그 판정을 받친 수치 — *움직일 수 있는 구의 최소 여유 **185.6 mm***,
*못 푸는 쌍 **27 개** 중 **24 개**가 50 mm 미달* — 는 2026-09-15 의 1 회성 측정이었고
**repo 에 그것을 내는 스크립트가 없었다** (2026-09-25 확인: `self_collision` 을 언급하는
파일은 `constraints/attached.py`·`constraints/constraint_builder.py`·`fields/esdf.py` 뿐이고
셋 다 라이브러리다). 모델을 16D 로 바꾼 뒤 다시 재려 했을 때 돌릴 것이 없어서 이 파일을 썼다.

## 무엇을 재는가

파지 구간의 프레임마다:

1. 쥔 물체의 점을 **MuJoCo 분할(segmentation)에서** 가져온다. grounding 이 아니라 참값이다 —
   이 기록에서 grounding 이 서지 않기 때문이고 (target 0/15), 그것은 N1 이 묻는 질문과
   무관하다. N1 은 *"물체가 팔 어디에 얼마나 가까운가"* 이지 *"지각이 물체를 찾는가"* 가 아니다.
2. `attach_from_target` 으로 그 점을 `parent_link` 에 태운다 — 파이프라인이 하는 것과 같은 계약.
3. `rigid_spheres()` 로 **최적화가 거리를 바꿀 수 없는 구**를 가른다. 손으로 쓴 허용목록이
   아니라 기구학에서 뽑는다: 자유 관절만 흔들어 보고 거리가 전혀 안 변하는 구를 뺀다.
4. 두 집합의 여유거리를 따로 낸다.

**두 집합을 갈라 보는 것이 이 측정의 전부다.**

| 집합 | 뜻 | self-collision 을 켜면 |
|---|---|---|
| **고정(rigid)** 구 | 손목처럼 물체와 **함께** 움직여 거리가 상수인 구 | **영원히 못 푸는 행**이 된다 — 최적화가 바꿀 수 없는 값에 제약을 건 것이라 매 프레임 violated 로 끝난다 |
| **움직일 수 있는** 구 | 최적화가 실제로 거리를 바꿀 수 있는 구 | 여기 여유가 마진보다 넉넉하면 켜도 아무것도 안 걸린다 = 켤 값어치가 없다(잠복) |

**전환 신호는 "움직일 수 있는 구가 마진 안으로 들어오는 것"** 이다. 이 스크립트가 내는
`movable_min_mm` 가 그 신호를 읽는 자리이고, 마진(기본 50 mm)보다 작아지면 N1 을 열 때다.

## 용어 (규칙 C)

*여유거리(clearance)* = 물체 점에서 로봇 구 표면까지의 거리. 음수면 물체가 구 **안에** 있다.
*쥔 물체(attached)* = 파지 뒤 로봇의 일부처럼 따라 움직이는 기하. *마진* = 제약이 요구하는
최소 여유. *잠복* = 지금 켜면 손해라서 안 켜지만 조건이 오면 켠다는 판정.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures/r-16d")
PARENT_LINK = "link_left_arm_6"
CONTACT_LINKS = frozenset({"ee_finger_l1", "ee_finger_l2"})
CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def backproject(depth, K, T_base_cam, mask):
    """마스크된 픽셀만 base 프레임 `(N,3)` 으로."""
    vs, us = np.nonzero(mask)
    z = np.asarray(depth, np.float64)[vs, us]
    good = np.isfinite(z) & (z > 1e-6)
    vs, us, z = vs[good], us[good], z[good]
    if z.size == 0:
        return np.zeros((0, 3))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam = np.column_stack([(us - cx) / fx * z, (vs - cy) / fy * z, z])
    T = np.asarray(T_base_cam, np.float64)
    return cam @ T[:3, :3].T + T[:3, 3]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments.common import figstyle
    return figstyle.use_korean()


def main() -> None:
    import mujoco

    from benchmark.ag3s.constraints.attached import (
        base_frame_points, rigid_spheres)
    from benchmark.ag3s.constraints.attached import attach_from_target
    from benchmark.ag3s.experiments.reports.attention_report import target_from_prompt
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from benchmark.ag3s.experiments.sources.policy_record import (
        grasp_onset, load_run, phase_boundaries_for_path, phase_for, pose_scene,
        replay_scene)
    from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS
    from benchmark.ag3s.types import Primitive, PrimitiveType, TargetGeometry
    from benchmark.trajopt.types import ChunkLayout

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--frames", type=int, default=0, help="0 이면 전부")
    ap.add_argument("--held", default=None,
                    help="쥔 물체의 MuJoCo body 이름. 기본은 prompt 에서 뽑는다")
    ap.add_argument("--parent-link", default=PARENT_LINK)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="제약이 요구할 최소 여유 [m]. 전환 신호를 읽는 기준선")
    ap.add_argument("--phase-boundaries", type=int, nargs=3, default=None,
                    metavar=("TRANSIT", "APPROACH", "PRE_GRASP"),
                    help="주지 않으면 기록의 그리퍼에서 뽑는다")
    ap.add_argument("--out", default=str(OUT), help="그림과 sidecar 를 쓸 디렉터리")
    args = ap.parse_args()

    run = load_run(args.records, limit=args.frames or None)
    held = args.held or target_from_prompt(run.prompt)

    if args.phase_boundaries is None:
        boundaries, phase_evidence = phase_boundaries_for_path(args.records)
    else:
        boundaries = tuple(int(x) for x in args.phase_boundaries)
        phase_evidence = {"source": "cli", "boundaries": list(boundaries)}
    onset = grasp_onset(run)

    scene = replay_scene(run)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    link_names = [str(x) for x in robot.sphere_link_names]
    # 최적화가 **실제로 바꾸는** 관절만 넘긴다. 고정 관절(토르소)을 함께 흔들면 움직일 수 없는
    # 쌍을 움직인다고 잘못 판정한다 — `rigid_spheres` 의 docstring 이 적은 실패다.
    layout = ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)
    free_joints = np.asarray(layout.q_indices, np.int64)

    held_body = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, held)
    if held_body < 0:
        raise SystemExit(f"body {held!r} 가 모델에 없다 — --held 로 이름을 직접 주라")

    print("=" * 92)
    print(f"N1 — 쥔 물체({held})와 로봇 자신의 여유거리   기록 {run.path}")
    print(f"단계 경계 {tuple(boundaries)} (출처 {phase_evidence['source']})"
          + (f"  파지 시작 {onset}" if onset else "  — 파지 없음"))
    print(f"제약 구 {robot.n_spheres}  자유 관절 {len(free_joints)}  "
          f"마진 {args.margin*1000:.0f} mm  parent_link {args.parent_link}")
    print("=" * 92)
    print(f"{'i':>3} {'t':>5} {'phase':<10} {'점':>6} {'고정구':>7} {'움직':>6} "
          f"{'고정 최소':>10} {'움직 최소':>10} {'움직<마진':>9}")

    rows = []
    for i, step in enumerate(run):
        phase = phase_for(step.t_step, boundaries)
        if phase != "grasp":
            continue
        pose_scene(scene, step)
        caps = {c: scene.capture(c) for c in CAMERAS}
        # 세 대를 다 쓴다. 머리 하나면 그리퍼가 물체를 가려 점이 거의 안 남는다.
        pts = np.vstack([
            backproject(np.asarray(f.depth, np.float64),
                        np.asarray(f.camera_intrinsics, np.float64),
                        np.asarray(f.T_base_cam, np.float64),
                        np.asarray(f.body_ids) == held_body)
            for f in caps.values()])
        if len(pts) < 8:
            print(f"{i:>3} {step.t_step:>5} {phase:<10} {len(pts):>6}   "
                  f"(점이 너무 적다 — 건너뜀)")
            continue

        q = np.asarray(caps["zed_left"].robot_state, np.float64)
        centre = pts.mean(axis=0)
        sphere = Primitive(
            type=PrimitiveType.SPHERE, center=centre, orientation=np.eye(3),
            dimensions=np.array([float(np.linalg.norm(pts - centre, axis=1).max()), 0.0, 0.0]))
        attached = attach_from_target(
            TargetGeometry(id=0, points=pts,
                           point_indices=np.arange(len(pts), dtype=np.int64),
                           centroid=centre, bounding_geometry=sphere,
                           attention_score=1.0, confidence=1.0),
            robot_state=q, robot_model=robot, parent_link=args.parent_link,
            allowed_contact_links=CONTACT_LINKS, label=f"held_{held}")

        # 여유거리: 쥔 점에서 각 로봇 구 표면까지. `rigid_spheres` 의 `gaps()` 와 같은 식이다.
        obj = base_frame_points(attached, robot_model=robot, robot_state=q)
        centres, radii = robot.sphere_centers_numeric(q)
        centres = np.asarray(centres, np.float64).reshape(-1, 3)
        radii = np.asarray(radii, np.float64).reshape(-1)
        gaps = np.linalg.norm(obj[:, None, :] - centres[None, :, :],
                              axis=2).min(axis=0) - radii

        rigid = rigid_spheres(attached, robot, q, free_joints)
        movable = ~rigid
        r_min = float(gaps[rigid].min()) if rigid.any() else float("nan")
        m_min = float(gaps[movable].min()) if movable.any() else float("nan")
        n_rigid_tight = int((gaps[rigid] < args.margin).sum()) if rigid.any() else 0
        n_move_tight = int((gaps[movable] < args.margin).sum()) if movable.any() else 0

        print(f"{i:>3} {step.t_step:>5} {phase:<10} {len(obj):>6} {int(rigid.sum()):>7} "
              f"{int(movable.sum()):>6} {r_min*1000:>10.1f} {m_min*1000:>10.1f} "
              f"{n_move_tight:>9}")
        rows.append(dict(
            i=i, t_step=int(step.t_step), phase=phase, n_points=int(len(obj)),
            n_rigid=int(rigid.sum()), n_movable=int(movable.sum()),
            rigid_min_mm=r_min * 1000.0, movable_min_mm=m_min * 1000.0,
            n_rigid_below_margin=n_rigid_tight, n_movable_below_margin=n_move_tight,
            rigid_links=sorted({link_names[j] for j in np.flatnonzero(rigid)}),
            movable_min_link=link_names[int(np.flatnonzero(movable)[
                int(np.argmin(gaps[movable]))])] if movable.any() else None))
    scene.close()

    if not rows:
        raise SystemExit(
            "파지 구간 프레임이 하나도 없다. 이 기록에 파지가 없거나 (그리퍼가 끝까지 열려 "
            "있다) --phase-boundaries 가 맞지 않는다. 위에 찍힌 단계 경계를 볼 것.")

    # ---------- 종합 ----------
    mv = np.array([r["movable_min_mm"] for r in rows])
    rg = np.array([r["rigid_min_mm"] for r in rows])
    verdict_movable = float(np.nanmin(mv))
    print()
    print("=" * 92)
    print(f"파지 프레임 {len(rows)} 개")
    print(f"  고정 구      최소 여유 중앙 {np.nanmedian(rg):+8.1f} mm   "
          f"최악 {np.nanmin(rg):+8.1f} mm   "
          f"— 켜면 못 푸는 행 (최적화가 이 거리를 못 바꾼다)")
    print(f"  움직일 수 있는 구  최소 여유 중앙 {np.nanmedian(mv):+8.1f} mm   "
          f"최악 {verdict_movable:+8.1f} mm")
    signal = verdict_movable < args.margin * 1000.0
    print()
    print(f">>> 전환 신호 (움직일 수 있는 구가 마진 {args.margin*1000:.0f} mm 안으로): "
          + ("**왔다** — N1 을 열 때다" if signal else
             f"아직 아니다 (마진의 {verdict_movable/(args.margin*1000):.1f} 배). 잠복 유지"))

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = pathlib.Path(str(args.records).rstrip("/")).name or "record"
    side = out_dir / f"n1-self-collision-{tag}.json"
    side.write_text(json.dumps({
        "records": str(args.records),
        "held": held,
        "parent_link": args.parent_link,
        "margin_mm": args.margin * 1000.0,
        "phase_boundaries": phase_evidence,
        "grasp_onset": onset,
        "n_grasp_frames": len(rows),
        "movable_min_mm": verdict_movable,
        "movable_min_mm_median": float(np.nanmedian(mv)),
        "rigid_min_mm": float(np.nanmin(rg)),
        "rigid_min_mm_median": float(np.nanmedian(rg)),
        "transition_signal": bool(signal),
        "per_frame": rows,
    }, ensure_ascii=False, indent=1))
    print(f"\nwrote {side}")
    _figure(rows, args, run, out_dir, tag)


def _figure(rows, args, run, out_dir, tag) -> None:
    """규칙 A — 그래프 + 표를 한 장에."""
    korean = _style()
    import matplotlib.pyplot as plt

    def L(ko, en):
        return ko if korean else en

    i = [r["i"] for r in rows]
    mv = [r["movable_min_mm"] for r in rows]
    rg = [r["rigid_min_mm"] for r in rows]
    margin_mm = args.margin * 1000.0

    fig, axs = plt.subplots(1, 3, figsize=(19.5, 5.4))

    a = axs[0]
    a.axhspan(min(min(rg), -80) - 20, margin_mm, color="crimson", alpha=0.08)
    a.axhline(margin_mm, color="crimson", ls="--", lw=2)
    a.text(i[0], margin_mm + 6, L(f"마진 {margin_mm:.0f} mm", f"margin {margin_mm:.0f} mm"),
           color="crimson", fontsize=10)
    a.axhline(0, color="0.4", lw=1)
    a.plot(i, mv, "o-", color="tab:blue", lw=2.4, ms=7,
           label=L("움직일 수 있는 구", "movable spheres"))
    a.plot(i, rg, "s--", color="tab:orange", lw=2.0, ms=6,
           label=L("고정 구 (못 푸는 행)", "rigid spheres (unsolvable)"))
    a.set_xlabel(L("정책 호출", "policy call"))
    a.set_ylabel(L("최소 여유거리 [mm]", "min clearance [mm]"))
    a.set_title(L("(a) 파지 구간의 최소 여유거리\n마진 아래로 내려오면 N1 전환 신호",
                  "(a) min clearance during grasp"), fontsize=11)
    a.legend(fontsize=9.5); a.grid(alpha=0.3)

    a = axs[1]
    a.bar([x - 0.2 for x in i], [r["n_rigid"] for r in rows], width=0.4,
          color="tab:orange", label=L("고정", "rigid"))
    a.bar([x + 0.2 for x in i], [r["n_movable"] for r in rows], width=0.4,
          color="tab:blue", label=L("움직일 수 있음", "movable"))
    a.set_xlabel(L("정책 호출", "policy call"))
    a.set_ylabel(L("구 개수", "spheres"))
    a.set_title(L("(b) 기구학이 가른 두 집합의 크기\n고정 구에 제약을 걸면 영원히 못 푼다",
                  "(b) split by kinematic reachability"), fontsize=11)
    a.legend(fontsize=9.5); a.grid(alpha=0.3, axis="y")

    a = axs[2]; a.axis("off")
    head = [L("호출", "call"), L("점", "pts"), L("고정", "rigid"), L("움직", "movable"),
            L("고정 최소", "rigid min"), L("움직 최소", "movable min")]
    body = [[str(r["i"]), f"{r['n_points']:,}", str(r["n_rigid"]), str(r["n_movable"]),
             f"{r['rigid_min_mm']:.1f}", f"{r['movable_min_mm']:.1f}"] for r in rows]
    t = a.table(cellText=body, colLabels=head, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1.0, 1.55)
    for j in range(len(head)):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    for ri, r in enumerate(rows, start=1):
        if r["movable_min_mm"] < margin_mm:
            for j in range(len(head)):
                t[(ri, j)].set_facecolor("#f7d6d6")
    a.set_title(L(f"(c) 분홍 = 움직일 수 있는 구가 마진({margin_mm:.0f} mm) 안",
                  f"(c) pink = movable sphere inside margin"), fontsize=11, y=0.9)

    fig.suptitle(L(
        f"N1 — 쥔 물체와 로봇 자신의 여유거리 ({run.meta.get('policy_model', '?')})",
        f"N1 — held object vs robot self clearance"), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = out_dir / f"n1-self-collision-{tag}.png"
    if out.exists():
        print(f"[figure] {out} 가 이미 있다 — 덮어쓴다. --out 을 바꿔라")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
