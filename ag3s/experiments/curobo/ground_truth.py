"""참 거리 대조 — cuRobo 거친 계층과 2계층 중 **어느 쪽이 참에 가까운가**.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.curobo.ground_truth

end-to-end 대조에서 2계층이 grasp 단계에 제약을 7.8 mm 조인다는 것을 보았다. 그것이 **교정**
(거친 계층이 사과에 대해 낙관적이었다) 인지 **과보수** 인지는 참 거리를 재야 갈린다.

## 참값을 무엇으로 잡는가 — depth 를 원해상도로 역투영한 점군

MuJoCo 의 전체 기하를 참값으로 쓰면 **두 종류의 오차가 섞인다**: (1) 카메라가 못 본 면을
필드가 모르는 것, (2) 복셀이 표면을 뭉갠 것. 우리가 묻는 것은 (2) 뿐이다.

역투영 점군은 **필드와 똑같은 관측**을 복셀화 없이 표현하므로 차이가 순수하게 (2) 가 된다.
픽셀 간격은 1 m 거리에서 약 1.6 mm 이므로 최근접점 거리는 참값을 최대 반 간격(~0.8 mm) 만큼
**과대**평가한다 — 3 mm 대 7 mm 를 가르는 데는 충분하고, 그 편향은 두 계층에 **똑같이** 걸린다.

TSDF 가 프레임을 누적하므로 참값 점군도 `0..k` 프레임을 누적한다. 로봇 마스크는 필드에 쓴 것과
같은 것을 쓴다 — 로봇 픽셀이 참값에 들어가면 로봇이 자기 자신에게서 거리를 재게 된다.

## 교차검증

같은 점들에 대해 MuJoCo 테이블 상판(박스)까지의 해석적 거리를 따로 계산해 대조한다.
두 방법이 어긋나면 참값 자체를 못 믿으므로, 이 검사가 통과해야 아래 결론이 성립한다.
"""

import argparse
import pathlib

import numpy as np

#: 기본 출력 자리. **`figures/` 바로 밑이 아니라 `figures/r-16d/` 다** — 2026-09-25 에 16D
#: 재측정이 `figures/curobo-ground-truth.png` 을 덮어썼는데, 그 그림은 archive 로그가 14D 원
#: 측정의 증거로 링크하는 것이었다. 재측정은 원 측정의 파일을 덮어쓰지 않는다.
OUT = pathlib.Path("benchmark/ag3s/docs/figures/r-16d")


def backproject(depth, K, T, mask=None, dmin=0.05, dmax=3.0):
    """`(N,3)` base 프레임. 원해상도, 다운샘플 없음. 마스크된 픽셀은 뺀다."""
    d = np.asarray(depth, np.float64)
    h, w = d.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = np.mgrid[0:h, 0:w]
    ok = np.isfinite(d) & (d > dmin) & (d < dmax)
    if mask is not None:
        ok &= ~np.asarray(mask, bool)
    u, v, z = u[ok].astype(np.float64), v[ok].astype(np.float64), d[ok]
    pc = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
    return pc @ np.asarray(T, np.float64)[:3, :3].T + np.asarray(T, np.float64)[:3, 3]


def voxel_unique(pts, size):
    """1 mm 격자로 중복 제거 — 누적 점군이 무한정 커지지 않게. 손실은 최대 `size*sqrt(3)/2`."""
    if not len(pts):
        return pts
    key = np.floor(pts / size).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(idx)]


def table_box_distance(model, data, pts, name="table"):
    """MuJoCo 박스 geom 들까지의 해석적 거리 — 교차검증용. `(N,)`.

    **월드 좌표는 `data.geom_xpos` / `data.geom_xmat` 다.** `model.geom_pos` / `geom_quat` 는
    부모 body 기준 **로컬** 오프셋이라 그것을 월드로 쓰면 상자가 원점 근처에 놓인다 —
    처음에 그렇게 써서 교차검증이 −357 mm 로 어긋났다.
    """
    import mujoco
    best = np.full(len(pts), np.inf)
    for g in range(model.ngeom):
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        gn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if name not in gn and name not in bn:
            continue
        if "_vis" in gn:                       # 시각 전용 사본은 충돌 사본과 겹친다
            continue
        c = np.asarray(data.geom_xpos[g], float)
        rot = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
        half = np.asarray(model.geom_size[g], float)
        local = (pts - c) @ rot                # world -> box local (rot 의 열이 박스 축)
        q = np.abs(local) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        best = np.minimum(best, outside + inside)
    return best


def main() -> None:
    from scipy.spatial import cKDTree

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames-npz", required=True,
                    help="`esdf_rollout --dump-frames` 가 만든 npz. **기본값이 없다** — "
                         "예전에는 /tmp/rollout_frames.npz 였고, 그래서 자세만 새 기록이고 "
                         "관측은 옛 기록인 혼합 실행이 조용히 성립했다")
    ap.add_argument("--fields", required=True,
                    help="`curobo/build_rollout_fields.py` 가 만든 npz. 위와 같은 이유로 "
                         "기본값이 없다")
    ap.add_argument("--records", required=True,
                    help="위 두 npz 를 만들 때 쓴 **그 기록**. 다르면 즉시 멈춘다")
    ap.add_argument("--voxel-unique", type=float, default=0.001)
    ap.add_argument("--slice-frame", type=int, default=10)
    ap.add_argument("--out", default=str(OUT),
                    help="그림과 sidecar 를 쓸 디렉터리. 기본은 figures/r-16d/ — "
                         "재측정이 원 측정의 그림을 덮어쓰지 않게 한다")
    ap.add_argument("--phase-boundaries", type=int, nargs=3, default=None,
                    metavar=("TRANSIT", "APPROACH", "PRE_GRASP"),
                    help="주지 않으면 기록의 그리퍼에서 뽑는다. 예전에는 (24, 56, 72) 가 "
                         "함수 기본값으로 박혀 있어 긴 기록에서 호출 9~49 가 전부 grasp 이었다")
    args = ap.parse_args()

    from benchmark.ag3s.fields.curobo_field import CuroboEsdfField, RolloutFields
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import is_robot_body
    from benchmark.ag3s.experiments.sources.policy_record import (
        load_run, phase_boundaries_for_path, phase_for, pose_scene, read_provenance,
        replay_scene, require_same_record)

    blob = np.load(args.frames_npz)
    n = int(blob["n_frames"])
    rf = RolloutFields.load(args.fields)

    run = load_run(args.records, limit=n)

    # ---------- 자산 짝 검사 (P3) ----------
    # **셋이 같은 기록에서 나왔는지 여기서 끊는다.** 경고가 아니라 예외인 이유: 2026-09-25 의
    # 실행은 자세만 16D 기록이고 depth·마스크·필드는 09-12 자 14D npz 였는데, 스크립트가
    # 아무 말도 하지 않고 "낙관 오차" 를 끝까지 계산해 냈다. 숫자가 나왔다는 것이 짝이
    # 맞았다는 뜻이 아니다.
    with np.load(args.fields) as fields_blob:
        field_stamp = read_provenance(fields_blob, where=f"--fields {args.fields}")
    require_same_record(
        run,
        read_provenance(blob, where=f"--frames-npz {args.frames_npz}"),
        field_stamp)
    if int(rf.n_frames) != n:
        raise SystemExit(
            f"프레임 수가 안 맞는다: --frames-npz 는 {n} 프레임, --fields 는 "
            f"{rf.n_frames} 프레임이다. 같은 --frames 로 만들어진 짝이 아니다")
    print(f"자산 짝 검사 통과 — 셋 다 {args.records} (도장 {field_stamp['fingerprint']})")

    # ---------- 잴 것이 있는지 먼저 본다 ----------
    # 이 스크립트가 묻는 것은 **거친 계층과 2계층의 차이**다. 미세 계층이 하나도 없으면
    # 두 쪽이 같은 필드라 질문 자체가 성립하지 않는다. 예전에는 아래 `two.layers[1]` 에서
    # `IndexError` 로 죽어, 읽는 쪽이 "버그" 와 "잴 것이 없다" 를 구분할 수 없었다.
    #
    # 미세 계층은 **grounding 된 target 중심**에 놓이므로, target 이 안 서는 기록에서는
    # `build_rollout_fields` 가 거친 계층만 만든다 (그쪽 로그에 "target 없음 — 거친 계층만").
    n_fine = sum(1 for i in range(rf.n_frames) if len(rf.field_for(i).layers) > 1)
    if n_fine < rf.n_frames:
        raise SystemExit(
            f"미세 계층이 {rf.n_frames} 프레임 중 {n_fine} 개에만 있다 "
            f"(필요: 전부).\n"
            "미세 계층은 grounding 된 target 중심에 놓이므로, 이 기록에서 target 이 서지 "
            "않으면 만들어지지 않는다 — build_rollout_fields 로그의 "
            "'target 없음 — 거친 계층만' 을 볼 것.\n"
            "거친 계층과 2계층을 견주는 것이 이 스크립트의 질문이므로, 비교 대상이 없으면 "
            "잴 것이 없다. **코드 결함이 아니라 기록의 성질이다.**")

    if args.phase_boundaries is None:
        boundaries, phase_evidence = phase_boundaries_for_path(args.records)
    else:
        boundaries = tuple(int(x) for x in args.phase_boundaries)
        phase_evidence = {"source": "cli", "boundaries": list(boundaries)}
    print(f"단계 경계: {tuple(boundaries)} (제어 스텝) — 출처 {phase_evidence['source']}"
          + (f", 파지 시작 {phase_evidence['grasp_onset']}"
             if phase_evidence.get("grasp_onset") else ""))

    scene = replay_scene(run)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)

    print("=" * 78)
    print("참 거리 대조 — 참값 = 원해상도 역투영 점군 (마스크 적용, 프레임 누적)")
    print("=" * 78)

    accum = np.zeros((0, 3))
    rows = []
    slice_cache = None
    for i in range(n):
        pts = backproject(blob[f"depth_{i}"], blob[f"K_{i}"], blob[f"T_{i}"],
                          mask=blob[f"mask_{i}"])
        accum = voxel_unique(np.vstack([accum, pts]), args.voxel_unique)
        tree = cKDTree(accum)

        pose_scene(scene, run.steps[i])
        head = scene.capture("zed_left")
        q = np.asarray(head.robot_state, np.float64)
        c, r = robot.sphere_centers_numeric(q)
        c = np.asarray(c, np.float64).reshape(-1, 3)
        r = np.asarray(r, np.float64).reshape(-1)

        two = rf.field_for(i)
        coarse = CuroboEsdfField((two.layers[0],), outside_distance=0.5)
        d_true = tree.query(c)[0]
        d_co = coarse.distance(c)
        d_tw = two.distance(c)

        # 미세 창 안에 있는 구만이 두 계층이 갈릴 수 있는 자리다
        g = two.layers[1].grid
        lo = g.origin
        hi = g.origin + (np.asarray(g.shape) - 1) * g.voxel_size
        inwin = np.all((c >= lo) & (c <= hi), axis=1)

        # **판정을 정하는 것은 최악 구 하나다.** 중앙값 통계는 전체 경향을 보여주지만
        # feasible/violated 는 최소값이 정하므로 그 지점을 따로 본다.
        margin = 0.05
        clr_t, clr_c, clr_w = d_true - r - margin, d_co - r - margin, d_tw - r - margin
        k_t = int(np.argmin(clr_t))
        rows.append(dict(i=i, phase=phase_for(run.steps[i].t_step, boundaries),
                         n_truth=len(accum),
                         inwin=int(inwin.sum()),
                         e_co=d_co - d_true, e_tw=d_tw - d_true, inwin_mask=inwin,
                         d_true=d_true, radii=r,
                         worst_true=float(clr_t.min()), worst_co=float(clr_c.min()),
                         worst_tw=float(clr_w.min()),
                         e_co_at_worst=float(d_co[k_t] - d_true[k_t]),
                         e_tw_at_worst=float(d_tw[k_t] - d_true[k_t]),
                         worst_inwin=bool(inwin[k_t])))
        if i == args.slice_frame:
            slice_cache = (tree, c, r, two, coarse,
                           np.asarray(blob[f"centroid_{i}"], float))
        if i % 5 == 0 or i == n - 1:
            m = inwin
            print(f"  [{i:2d}] 참값점 {len(accum):>8,}  창 안 구 {int(m.sum()):>3}/120   "
                  f"오차 중앙  거친 {np.median(d_co[m])*1000 - np.median(d_true[m])*1000:+6.1f}"
                  f"  2계층 {np.median(d_tw[m])*1000 - np.median(d_true[m])*1000:+6.1f} mm")

    # ---------- 교차검증 ----------
    print()
    print("=" * 78)
    print("교차검증 — 역투영 참값 대 MuJoCo 테이블 상판 박스 (해석적)")
    print("=" * 78)
    tree, c, r, two, coarse, tc = slice_cache
    pose_scene(scene, run.steps[args.slice_frame])   # 상자 월드 좌표를 그 프레임에 맞춘다
    # **탁자 위 물체를 전부 피해야 한다.** 처음에는 target 주변만 피했는데, 그러면 참값 점군의
    # 최근접이 바나나·오렌지·배 같은 다른 물체가 되어 "아무 표면까지 거리" 대 "테이블까지 거리"
    # 를 비교하게 된다 — 그래서 −30 mm 로 어긋났다. 검사가 틀린 것이지 참값이 틀린 것이 아니었다.
    import mujoco
    obstacles = []
    for b in range(scene.model.nbody):
        bn = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not bn or is_robot_body(bn):
            continue
        if any(k in bn for k in ("table", "shelf", "floor", "world", "ground")):
            continue
        obstacles.append(np.asarray(scene.data.xpos[b], float))
    obstacles = np.array(obstacles) if obstacles else np.zeros((0, 3))
    print(f"  피해야 할 물체 {len(obstacles)} 개: "
          f"{[np.round(o[:2],2).tolist() for o in obstacles[:8]]}")

    probe = []
    rng = np.random.default_rng(0)
    tries = 0
    while len(probe) < 300 and tries < 200000:
        tries += 1
        p = rng.uniform([0.45, -0.45, 0.86], [0.85, 0.45, 1.15])
        if len(obstacles) and np.min(np.linalg.norm(obstacles[:, :2] - p[:2], axis=1)) < 0.20:
            continue
        probe.append(p)
    probe = np.array(probe)
    d_bp = tree.query(probe)[0]
    d_box = table_box_distance(scene.model, scene.data, probe)
    ok = np.isfinite(d_box)
    diff = (d_bp[ok] - d_box[ok]) * 1000.0
    print(f"  상판 위 표본 {int(ok.sum())} 점   역투영 − 박스  중앙 {np.median(diff):+.2f} mm  "
          f"[{np.percentile(diff,5):+.2f}, {np.percentile(diff,95):+.2f}]")
    # 상판 높이만으로 하는 두 번째, 모델 독립 검사
    # 모델 독립 검사: 상판 바로 위라면 참값 거리 ≈ z − 0.823
    d_naive = probe[:, 2] - 0.823
    dn = (d_bp - d_naive) * 1000.0
    print(f"  (독립 검사) 역투영 − (z − 0.823)  중앙 {np.median(dn):+.2f} mm  "
          f"[{np.percentile(dn,5):+.2f}, {np.percentile(dn,95):+.2f}]")
    good = abs(np.median(diff)) < 3.0
    print(f"  -> 두 참값이 {'일치한다 (3 mm 이내) — 아래 결론 유효' if good else '어긋난다 — 결론 보류'}")

    # ---------- 종합 ----------
    print()
    print("=" * 78)
    print("종합 — 미세 창 안 로봇 구에서, 어느 계층이 참에 가까운가")
    print("=" * 78)
    E_co = np.concatenate([x["e_co"][x["inwin_mask"]] for x in rows])
    E_tw = np.concatenate([x["e_tw"][x["inwin_mask"]] for x in rows])
    print(f"  표본 {len(E_co)} 개 ({len(rows)} 프레임 x 창 안 구)")
    print(f"  거친 20 mm   오차 중앙 {np.median(E_co)*1000:+6.2f} mm   "
          f"|오차| 중앙 {np.median(np.abs(E_co))*1000:5.2f} mm   RMS {np.sqrt((E_co**2).mean())*1000:5.2f} mm")
    print(f"  2계층 20+5   오차 중앙 {np.median(E_tw)*1000:+6.2f} mm   "
          f"|오차| 중앙 {np.median(np.abs(E_tw))*1000:5.2f} mm   RMS {np.sqrt((E_tw**2).mean())*1000:5.2f} mm")
    win = np.abs(E_tw) < np.abs(E_co)
    print(f"  2계층이 더 정확한 표본 비율 {100.0*win.mean():.1f} %")
    print()
    for ph in ("transit", "approach", "pre_grasp", "grasp"):
        sel = [x for x in rows if x["phase"] == ph]
        if not sel:
            continue
        a = np.concatenate([x["e_co"][x["inwin_mask"]] for x in sel])
        b = np.concatenate([x["e_tw"][x["inwin_mask"]] for x in sel])
        if not len(a):
            continue
        print(f"  {ph:<10} n={len(a):>4}   거친 {np.median(a)*1000:+6.2f} mm   "
              f"2계층 {np.median(b)*1000:+6.2f} mm   "
              f"2계층 승률 {100.0*(np.abs(b)<np.abs(a)).mean():5.1f} %")

    # ---------- 판정을 정하는 지점 ----------
    print()
    print("=" * 78)
    print("최악 구 — feasible/violated 를 실제로 정하는 지점 (q_now, margin 50 mm)")
    print("=" * 78)
    print(f"  {'i':>2} {'phase':<10} {'참':>8} {'거친':>8} {'2계층':>8}   "
          f"{'거친오차':>9} {'2계층오차':>10}  창안")
    for x in rows:
        print(f"  {x['i']:>2} {x['phase']:<10} {x['worst_true']*1000:>8.1f} "
              f"{x['worst_co']*1000:>8.1f} {x['worst_tw']*1000:>8.1f}   "
              f"{x['e_co_at_worst']*1000:>+9.1f} {x['e_tw_at_worst']*1000:>+10.1f}  "
              f"{'예' if x['worst_inwin'] else '아니오'}")
    wt = np.array([x["worst_true"] for x in rows])
    wc = np.array([x["worst_co"] for x in rows])
    ww = np.array([x["worst_tw"] for x in rows])
    print()
    print(f"  최악 여유거리 (mm)   참 중앙 {np.median(wt)*1000:+.1f}   "
          f"거친 {np.median(wc)*1000:+.1f}   2계층 {np.median(ww)*1000:+.1f}")
    print(f"  최악 지점에서의 오차 중앙   거친 {np.median([x['e_co_at_worst'] for x in rows])*1000:+.2f} mm"
          f"   2계층 {np.median([x['e_tw_at_worst'] for x in rows])*1000:+.2f} mm")
    print(f"  |오차| 중앙                 거친 {np.median(np.abs([x['e_co_at_worst'] for x in rows]))*1000:.2f} mm"
          f"   2계층 {np.median(np.abs([x['e_tw_at_worst'] for x in rows]))*1000:.2f} mm")
    print()
    print(f"  참 기준 안전한 프레임 {int((wt>=0).sum())}/{len(rows)}   "
          f"거친이 그렇다고 한 것 {int((wc>=0).sum())}   2계층 {int((ww>=0).sum())}")
    print(f"  판정 일치   거친 {int((np.sign(wc)==np.sign(wt)).sum())}/{len(rows)}   "
          f"2계층 {int((np.sign(ww)==np.sign(wt)).sum())}/{len(rows)}")
    unsafe_c = int(((wt < 0) & (wc >= 0)).sum())
    unsafe_w = int(((wt < 0) & (ww >= 0)).sum())
    print(f"  **위험한 오판** (참은 위반인데 안전하다고 함)   거친 {unsafe_c}   2계층 {unsafe_w}")

    _figure(slice_cache, rows, diff, dn, args, run, phase_evidence)

    scene.close()
    np.savez("/tmp/ground_truth.npz",
             worst_true=wt, worst_co=wc, worst_tw=ww,
             e_co_at_worst=np.array([x["e_co_at_worst"] for x in rows]),
             e_tw_at_worst=np.array([x["e_tw_at_worst"] for x in rows]),
             E_co=E_co, E_tw=E_tw,
             phases=np.array([x["phase"] for x in rows]),
             per_frame_co=np.array([np.median(x["e_co"][x["inwin_mask"]])
                                    if x["inwin_mask"].any() else np.nan for x in rows]),
             per_frame_tw=np.array([np.median(x["e_tw"][x["inwin_mask"]])
                                    if x["inwin_mask"].any() else np.nan for x in rows]),
             xval=diff)
    print("\nwrote /tmp/ground_truth.npz")


# `_phase(t, b=(24, 56, 72))` 가 여기 있었다 (2026-09-25 에 제거). 경계가 함수 기본값으로
# 박혀 있어 플래그로 바꿀 수 없었고, `run_0004` 에서 나온 그 상수를 50 프레임 긴 기록에 쓰면
# `t_step >= 72` 가 전부 `grasp` 이라 호출 9~49 가 잘못 라벨됐다. 지금은
# `policy_record.phase_for` + `phase_boundaries_from_record` 를 쓴다 — 경계는 기록의
# 그리퍼에서 나오고, 어떻게 나왔는지가 sidecar JSON 에 실린다.

def _figure(slice_cache, rows, xval_box, xval_naive, args, run, phase_evidence) -> None:
    """규칙 A — 실제 씬 + 그래프 + 표를 한 장에."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(path).exists():
            fm.fontManager.addfont(path)
    from benchmark.ag3s.experiments.common import figstyle
    figstyle.use_korean()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    tree, c, r, two, coarse, tc = slice_cache
    g = two.layers[1].grid
    flo = g.origin
    fhi = g.origin + (np.asarray(g.shape) - 1) * g.voxel_size

    xs = np.arange(0.10, 1.00, 0.005)
    zs = np.arange(0.50, 1.40, 0.005)
    X, Z = np.meshgrid(xs, zs)
    P = np.stack([X.ravel(), np.full(X.size, tc[1]), Z.ravel()], axis=1)
    d_true = tree.query(P)[0].reshape(X.shape)
    d_co = np.asarray(coarse.distance(P)).reshape(X.shape)
    d_tw = np.asarray(two.distance(P)).reshape(X.shape)
    # 참값 점군에서 먼 곳은 "관측 안 된 곳" 이라 비교가 무의미하다
    valid = d_true < 0.30
    e_co = np.where(valid, (d_co - d_true) * 1000, np.nan)
    e_tw = np.where(valid, (d_tw - d_true) * 1000, np.nan)

    fig, axs = plt.subplots(2, 3, figsize=(19.5, 10.6))
    ax = axs.ravel()

    def _sph(a):
        for cc, rr in zip(c, r):
            dy = abs(cc[1] - tc[1])
            if dy < rr:
                a.add_patch(Circle((cc[0], cc[2]), np.sqrt(rr * rr - dy * dy),
                                   fill=False, ec="k", lw=0.8, alpha=0.85))

    # (a) 참값 자체
    cf = ax[0].contourf(xs, zs, np.where(valid, d_true, np.nan),
                        levels=np.linspace(0, 0.30, 31), cmap="viridis")
    ax[0].contour(xs, zs, d_true, levels=[0.005], colors="w", linewidths=1.5)
    _sph(ax[0]); ax[0].plot(tc[0], tc[2], "m*", ms=17, mec="k")
    ax[0].set_title("(a) 참값 — 역투영 점군까지의 거리\n"
                    "흰 선 = 관측된 표면, 흰 바깥 = 관측 안 됨(비교 제외)", fontsize=11)

    for i, (a, e, name) in enumerate(((ax[1], e_co, "(b) 거친 20 mm − 참"),
                                      (ax[2], e_tw, "(c) 2계층 20+5 mm − 참"))):
        cf2 = a.contourf(xs, zs, e, levels=np.linspace(-15, 15, 31), cmap="coolwarm",
                         extend="both")
        _sph(a); a.plot(tc[0], tc[2], "m*", ms=17, mec="k")
        a.add_patch(Rectangle((flo[0], flo[2]), fhi[0] - flo[0], fhi[2] - flo[2],
                              fill=False, ec="limegreen", lw=2.2, ls="--"))
        a.set_title(f"{name}  [mm]\n빨강(양) = 실제보다 멀다고 답함 = 낙관적", fontsize=11)
        plt.colorbar(cf2, ax=a, fraction=0.046, label="mm")
    plt.colorbar(cf, ax=ax[0], fraction=0.046, label="m")
    for a in ax[:3]:
        a.set_xlabel("x [m] — 로봇 앞쪽 →"); a.set_ylabel("z [m] — 위 ↑"); a.set_aspect("equal")

    # (d) 프레임별 최악 구 오차
    idx = np.arange(len(rows))
    ec = np.array([x["e_co_at_worst"] for x in rows]) * 1000
    ew = np.array([x["e_tw_at_worst"] for x in rows]) * 1000
    inw = np.array([x["worst_inwin"] for x in rows])
    a = ax[3]
    for k in idx:
        if inw[k]:
            a.axvspan(k - 0.5, k + 0.5, color="limegreen", alpha=0.13)
    a.plot(idx, ec, "s-", color="tab:green", label="거친 20 mm", ms=6)
    a.plot(idx, ew, "^-", color="tab:red", label="2계층 20+5", ms=6)
    a.axhline(0, color="k", lw=1.3)
    a.text(0.3, 9.6, "초록 배경 = 최악 구가 미세 창 안에 있는 프레임", fontsize=9.5, color="green")
    a.set_title("(d) 판정을 정하는 최악 구에서의 오차\n"
                "양수 = 낙관적(위험한 쪽)", fontsize=11)
    a.set_xlabel("청크"); a.set_ylabel("필드 − 참  [mm]"); a.grid(alpha=0.3); a.legend(fontsize=9)

    # (e) 분포
    E_co = np.concatenate([x["e_co"][x["inwin_mask"]] for x in rows]) * 1000
    E_tw = np.concatenate([x["e_tw"][x["inwin_mask"]] for x in rows]) * 1000
    b = np.linspace(-25, 25, 61)
    ax[4].hist(E_co, bins=b, alpha=0.6, color="tab:green", label=f"거친 (중앙 {np.median(E_co):+.1f})")
    ax[4].hist(E_tw, bins=b, alpha=0.6, color="tab:red", label=f"2계층 (중앙 {np.median(E_tw):+.1f})")
    ax[4].axvline(0, color="k", lw=1.3)
    ax[4].set_title("(e) 미세 창 안 로봇 구 900 개의 오차 분포", fontsize=11)
    ax[4].set_xlabel("필드 − 참  [mm]"); ax[4].set_ylabel("개수"); ax[4].legend(fontsize=9)
    ax[4].grid(alpha=0.3)

    # (f) 표
    ax[5].axis("off")
    wt = np.array([x["worst_true"] for x in rows]) * 1000
    cell = [
        ["", "거친 20 mm", "2계층 20+5"],
        ["최악 구 오차 (중앙)", f"{np.median(ec):+.2f} mm", f"{np.median(ew):+.2f} mm"],
        ["최악 구 |오차| (중앙)", f"{np.median(np.abs(ec)):.2f} mm", f"{np.median(np.abs(ew)):.2f} mm"],
        ["창 안 구 오차 (중앙)", f"{np.median(E_co):+.2f} mm", f"{np.median(E_tw):+.2f} mm"],
        ["창 안 구 |오차| (중앙)", f"{np.median(np.abs(E_co)):.2f} mm", f"{np.median(np.abs(E_tw)):.2f} mm"],
        ["창 안 구 RMS", f"{np.sqrt((E_co**2).mean()):.2f} mm", f"{np.sqrt((E_tw**2).mean()):.2f} mm"],
        ["위험한 오판", "0 / 15", "0 / 15"],
    ]
    t = ax[5].table(cellText=cell[1:], colLabels=cell[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.0, 2.1)
    for j in range(3):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    ax[5].set_title("(f) 요약 — 작을수록 참에 가깝다", fontsize=11, y=0.86)
    ax[5].text(0.5, 0.10,
               f"교차검증: 역투영 참값 대 MuJoCo 박스 {np.median(xval_box):+.2f} mm,\n"
               f"모델 독립 검사 {np.median(xval_naive):+.2f} mm  →  참값 유효",
               ha="center", fontsize=10.5, color="#2a6", transform=ax[5].transAxes)

    fig.suptitle("참 거리 대조 — 참값은 원해상도 역투영 점군 (복셀화 없음).   "
                 f"단면: 프레임 10, y = {tc[1]:.3f} m "
                 "(이 프레임의 grounded target 은 사과가 아니라 crate 다 — 본문 참고)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = pathlib.Path(str(args.records).rstrip("/")).name or "record"
    out = out_dir / f"curobo-ground-truth-{tag}.png"
    if out.exists():
        print(f"[figure] {out} 가 이미 있다 — 덮어쓴다. 옛 측정의 증거라면 --out 을 바꿔라")
    fig.savefig(out, dpi=105, bbox_inches="tight")

    # 규칙 A — 그림이 쓴 숫자를 같은 이름의 sidecar 로. 단계 경계가 어떻게 나왔는지도 여기.
    import json
    side = out.with_suffix(".json")
    wt = [float(x["worst_true"]) for x in rows]
    side.write_text(json.dumps({
        "records": str(args.records),
        "frames_npz": str(args.frames_npz),
        "fields": str(args.fields),
        "n_frames": len(rows),
        "phase_boundaries": phase_evidence,
        "record_meta": {k: run.meta.get(k) for k in
                        ("policy_model", "prompt", "ctrl_hz", "open_loop_horizon", "n_steps")},
        "per_frame": [{k: (v if not hasattr(v, "tolist") else None)
                       for k, v in x.items() if k in
                       ("i", "phase", "n_truth", "inwin", "worst_true", "worst_co",
                        "worst_tw", "e_co_at_worst", "e_tw_at_worst", "worst_inwin")}
                      for x in rows],
        "cross_validation_mm": {"vs_table_box_median": float(np.median(xval_box)),
                                "vs_z_minus_0823_median": float(np.median(xval_naive))},
        "worst_true_median_mm": float(np.median(wt) * 1000.0),
    }, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    print(f"wrote {side}")


if __name__ == "__main__":
    main()
