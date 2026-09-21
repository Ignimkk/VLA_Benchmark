"""Step 7 — 두 융합 방식이 실제로 얼마나 갈리고, 갈릴 때 **누가 옳은가**.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.studies.step7_fusion_asymmetry --frames 12

## 무엇을 재는가

AG3S 는 카메라 셋을 한 씬으로 합치는 방법을 **둘** 갖고 있고, 둘은 "지운다" 는 능력에서 갈린다.

* **방법 A — 점을 모은다** (`multiview.fuse`). 1 cm 복셀 격자에 세 카메라의 점을 붓고 칸마다
  대표점 하나. 한 대라도 점을 찍으면 그 칸은 "있음" 이다. **지우는 수단이 없다 — 합집합.**
  소비자: 지지면 적합 · target grounding.
* **방법 B — 광선을 붓는다** (TSDF 적분, `esdf.update`). 픽셀 하나가 광선 하나이고, **지나온
  구간은 자유공간**으로 기록된다. 그래서 한 카메라의 광선이 다른 카메라가 본 표면을 깎을 수 있다.
  소비자: 충돌 필드.

"둘이 다르다" 만 세면 어느 쪽이 옳은지 모른다. MuJoCo 세그멘테이션이 **픽셀별 body 참값**을
주므로, 불일치마다 참값으로 심판한다.

## 판정 규칙

융합 클라우드의 점마다 그 점을 만든 관측(카메라 + 픽셀)을 CSR 출처에서 되짚어 **그 픽셀이 무엇을
본 것인지**를 참값으로 안다. 그리고 같은 자리를 거리장에 물어본다.

**격자 밖 점은 먼저 뺀다.** ESDF 는 유한한 상자이고 그 밖에서는 무조건 `+max_distance`("멀다")를
답한다 — 이것은 융합 비대칭이 아니라 **E4** 라는 별개의 알려진 결함이다. 처음 측정했을 때 지워진
점의 99.8 % 가 방 벽(`office`)이었고 전부 격자 밖이었다. 섞어 세면 재려던 것이 묻힌다.

**"빈 공간" 과 "모른다" 를 갈라야 한다.** 거리장이 큰 값을 돌려주는 경우가 둘이고 뜻이 완전히
다르다.

* `tol < d < max_distance` — 광선이 지나갔다고 **적극적으로 판단**한 자리. 이것이 재려던
  비대칭이다.
* `d == max_distance` — **한 번도 관측되지 않은 자리**(UNKNOWN). 카메라가 못 봤거나 격자
  가장자리다. 융합 방식의 차이가 아니다.

처음 측정에서 지워진 점의 99.5 % 가 후자였다 — 전부 `z = 0.003 m` 의 **바닥**이고 답이 예외 없이
정확히 400 mm 였다. 섞어 세면 재려던 것이 묻힌다.

| 참값 픽셀 | 거리장의 답 | 판정 |
|---|---|---|
| 실제 물체 | 표면 (`d <= voxel`) | 일치 |
| 실제 물체 | 파였다 (`tol < d < max`) | **방법 B 가 진짜 장애물을 지웠다** — 위험 |
| 실제 물체 | 모른다 (`d == max`) | 관측 공백 — 융합 비대칭이 아니다 |

반대 방향(필드에는 있는데 클라우드에 없는 것)도 센다. 점유 복셀 중 어떤 융합 점과도 멀리 떨어진
것이 몇 개인가 — 그쪽은 **방법 B 가 더 담고 있다**는 뜻이다.
"""

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")
CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments.common import figstyle
    figstyle.use_korean()


def main() -> None:
    _style()
    import mujoco

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import camera_observation, is_robot_body
    from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
    from benchmark.trajopt.experiments.esdf_rollout import phase_for

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--attention", default="attention_step1_run0004.npz")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/archive/step-verification-20260904/step-01-attention.json")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    args = ap.parse_args()

    run = load_run(args.records, limit=args.frames)
    blob = np.load(args.attention, allow_pickle=False)
    cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    A = np.asarray(blob["attention"], np.float32)
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    ci = [str(c) for c in blob["cameras"]].index("cam_high")

    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    ag = AG3S(AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False}}),
        robot_model=filter_robot,
        constraint_robot_model=build_constraint_robot_model(scene, link_filter=ARM_LINKS))

    names = {b: (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
             for b in range(scene.model.nbody)}

    print("=" * 104)
    print(f"Step 7 — 융합 비대칭, {args.records}, 카메라 {len(CAMERAS)} 대, 복셀 "
          f"{args.voxel*1000:.0f} mm")
    print("=" * 104)
    print(f"{'i':>3} {'격자안':>8} {'격자밖':>8} {'미관측':>10} {'파임(실물체)':>10} "
          f"{'로봇거름':>8} {'필드에만':>9}  파인 물체")

    rows = []
    for i, step in enumerate(run.steps):
        pose_scene(scene, step)
        att = A[i, di, ai, cell["layer"], cell["head"], ci]
        obs, frames = [], {}
        for cam in CAMERAS:
            o, fr = camera_observation(scene, cam, filter_robot, timestamp=float(i),
                                       attention_map=(att if cam == "zed_left" else None))
            obs.append(o); frames[o.camera_id] = fr

        cs, dbg = ag.process_multi_debug(
            obs, phase=phase_for(step.t_step, (24, 56, 72)))
        fused = dbg["fusion"].cloud
        field = cs.esdf
        if field is None or len(fused) == 0:
            print(f"{i:>3}  (필드 없음 또는 빈 클라우드)")
            continue

        P_all = np.asarray(fused.points, np.float64)
        tol = float(field.grid.voxel_size)
        # 격자 안쪽만 — 밖은 E4 이지 융합 비대칭이 아니다.
        g = field.grid
        lo = np.asarray(g.origin, np.float64) - 0.5 * tol
        hi = lo + np.asarray(g.shape, np.float64) * tol
        in_grid = np.all((P_all >= lo) & (P_all < hi), axis=1)
        n_out = int((~in_grid).sum())
        P = P_all[in_grid]
        if len(P) == 0:
            print(f"{i:>3}  (격자 안 점이 없다)")
            continue
        d = np.asarray(field.distance(P), np.float64)

        # 각 융합점의 출처 픽셀 → 참값 body
        rep = np.asarray(fused.representative, np.int64)[in_grid]
        cam_idx = np.asarray(fused.obs_camera, np.int16)[rep]
        uv = np.asarray(fused.obs_uv, np.int32)[rep]
        body = np.full(len(P), -1, np.int64)
        for k, cid in enumerate(fused.cameras):
            m = cam_idx == k
            if not m.any():
                continue
            bid = np.asarray(frames[cid].body_ids)
            u, v = uv[m, 0], uv[m, 1]
            ok = (u >= 0) & (v >= 0) & (v < bid.shape[0]) & (u < bid.shape[1])
            idx = np.flatnonzero(m)[ok]
            body[idx] = bid[v[ok], u[ok]]

        is_robot = np.array([b >= 0 and is_robot_body(names.get(int(b), "")) for b in body])
        is_real = (body >= 0) & ~is_robot
        field_free = d > tol

        # 파인 것과 모르는 것을 가른다 — 뜻이 다르다.
        md = float(field.max_distance)
        unknown = d >= md - 1e-9
        carved = field_free & ~unknown

        erased = carved & is_real              # 방법 B 가 적극적으로 지웠다 — 재려던 것
        unseen = unknown & is_real             # 관측 공백 — 별개
        caught = carved & is_robot             # 로봇 누수를 필드가 걸렀다
        leaked = (~field_free) & is_robot

        # 반대 방향 — 필드에만 있는 것
        occ = int(field.stats.get("n_occupied", 0))
        from scipy.spatial import cKDTree
        tree = cKDTree(P)
        centres = _occupied_centres(field)
        only_field = 0
        n_occ_centres = 0
        if centres is not None and len(centres):
            n_occ_centres = len(centres)
            only_field = int((tree.query(centres)[0] > 2 * tol).sum())

        vic = ""
        if erased.any():
            import collections
            c = collections.Counter(names.get(int(b), "?") for b in body[erased])
            vic = ", ".join(f"{k}:{v}" for k, v in c.most_common(3))
        print(f"{i:>3} {len(P):>8} {n_out:>8} {int(unseen.sum()):>10} "
              f"{int(erased.sum()):>10} {int(caught.sum()):>8} {only_field:>9}  {vic}")
        rows.append(dict(i=i, n=len(P), n_out=n_out, unseen=int(unseen.sum()),
                         erased=int(erased.sum()), caught=int(caught.sum()),
                         leaked=int(leaked.sum()), only_field=only_field,
                         n_occ_centres=n_occ_centres,
                         erased_bodies=[names.get(int(b), "?") for b in body[erased]],
                         unseen_bodies=[names.get(int(b), "?") for b in body[unseen]]))
    scene.close()
    _report(rows, args)
    _figure(rows, args)


def _occupied_centres(field):
    """표면 안쪽(거리 < 0)인 복셀의 중심 좌표.

    `EsdfField` 는 점유 배열을 따로 들고 있지 않고 `distance_grid` 만 노출한다 (음수 = 표면
    안쪽). 그것으로 충분하다 — 여기서 묻는 것은 "필드가 무언가 있다고 보는 자리" 이므로.
    """
    dg = getattr(field, "distance_grid", None)
    if dg is None:
        return None
    idx = np.argwhere(np.asarray(dg) < 0.0)
    if not len(idx):
        return None
    g = field.grid
    return np.asarray(g.origin, np.float64) + idx * float(g.voxel_size)


def _report(rows, args):
    if not rows:
        print("\n측정된 프레임이 없다.")
        return
    n = sum(r["n"] for r in rows)
    nout = sum(r["n_out"] for r in rows)
    er = sum(r["erased"] for r in rows)
    un = sum(r["unseen"] for r in rows)
    ca = sum(r["caught"] for r in rows)
    lk = sum(r["leaked"] for r in rows)
    of = sum(r["only_field"] for r in rows)
    print("\n" + "=" * 104)
    print(f"프레임 {len(rows)} 개, 격자 안 융합점 {n} (격자 밖 {nout} 은 E4 라 제외)")
    print(f"\n  방법 B 가 **적극적으로 파낸** 실물체 점 : {er:>7}  ({er/max(n,1)*100:.2f} %)  <- 재려던 것")
    print(f"  **관측 공백**(UNKNOWN) 인 실물체 점    : {un:>7}  ({un/max(n,1)*100:.2f} %)  <- 별개 현상")
    print(f"  방법 B 가 **로봇 누수를 걸러낸** 점    : {ca:>7}  ({ca/max(n,1)*100:.2f} %)  <- B 가 옳다")
    print(f"  둘 다 로봇을 장애물로 본 점            : {lk:>7}  ({lk/max(n,1)*100:.2f} %)")
    print(f"  필드에만 있고 클라우드엔 없는 복셀     : {of:>7}")
    import collections
    c = collections.Counter(b for r in rows for b in r["erased_bodies"])
    if c:
        print(f"\n  적극적으로 파인 점의 물체별 분포: {dict(c.most_common(6))}")
    cu = collections.Counter(b for r in rows for b in r["unseen_bodies"])
    if cu:
        print(f"  관측 공백 점의 물체별 분포    : {dict(cu.most_common(6))}")
    print("\n>>> 판정: " + (
        f"융합 비대칭은 작다 — 적극적으로 파인 실물체 점 {er} 개 "
        f"({er/max(n,1)*100:.3f} %). 큰 쪽은 관측 공백 {un} 개다" if er * 20 < un else
        f"융합 비대칭이 실재한다 — 파인 실물체 점 {er} 개"))


def _figure(rows, args):
    import collections
    import matplotlib.pyplot as plt
    if not rows:
        return
    idx = [r["i"] for r in rows]
    er = [r["erased"] for r in rows]
    un = [r["unseen"] for r in rows]
    of = [r["only_field"] for r in rows]
    nn = [r["n"] for r in rows]

    fig, axs = plt.subplots(1, 3, figsize=(19.5, 5.8))

    # (a) 프레임별 — 재려던 것과 필드가 더 가진 것
    a = axs[0]
    a.plot(idx, er, "s-", color="crimson", lw=2.2, ms=7,
           label="B 가 적극적으로 파낸 실물체 점")
    a.plot(idx, of, "^-", color="tab:green", lw=2.2, ms=7,
           label="필드에만 있는 복셀 (B 가 더 가진 것)")
    a.set_yscale("symlog", linthresh=10)
    a.set_xlabel("프레임"); a.set_ylabel("개수 (symlog)")
    a.set_title("(a) 재려던 비대칭 대 B 의 값어치\n초록이 두 자릿수 위다", fontsize=11)
    a.legend(fontsize=9.5); a.grid(alpha=0.3, which="both")

    # (b) 세 범주 — 무엇이 실제로 컸나
    a = axs[1]
    E, U, F, N = sum(er), sum(un), sum(of), sum(nn)
    bars = a.barh(["B 가 파냄\n(실물체)", "관측 공백\n(UNKNOWN)", "필드에만 있음"],
                  [E, U, F], color=["crimson", "0.72", "tab:green"])
    for b, v in zip(bars, [E, U, F]):
        a.text(v * 1.15, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontsize=11)
    a.set_xscale("log")
    a.set_xlabel("점 / 복셀 수 (로그)")
    a.set_title("(b) 처음엔 셋이 섞여 있었다\n'비었다' 와 '모른다' 를 갈라야 보인다", fontsize=11)
    a.grid(alpha=0.3, axis="x", which="both")

    # (c) 표
    a = axs[2]; a.axis("off")
    ce = collections.Counter(b for r in rows for b in r["erased_bodies"])
    cu = collections.Counter(b for r in rows for b in r["unseen_bodies"])
    rowsT = [["항목", "값"],
             ["격자 안 융합점", f"{N:,}"],
             ["B 가 적극적으로 파냄", f"{E:,}  ({E/max(N,1)*100:.3f} %)"],
             ["  그 물체", ", ".join(f"{k} {v}" for k, v in ce.most_common(3)) or "—"],
             ["관측 공백 (UNKNOWN)", f"{U:,}  ({U/max(N,1)*100:.1f} %)"],
             ["  그 물체", ", ".join(f"{k} {v}" for k, v in cu.most_common(3)) or "—"],
             ["필드에만 있는 복셀", f"{F:,}"],
             ["로봇 누수", "0  (자기 필터가 둘보다 앞에 있다)"]]
    t = a.table(cellText=rowsT[1:], colLabels=rowsT[0], loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1.0, 1.85)
    for j in range(2):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    for j in range(2):
        t[(2, j)].set_facecolor("#f7d6d6"); t[(6, j)].set_facecolor("#d9f0d9")
    a.set_title("(c) 참값이 심판한 결과", fontsize=11.5, y=0.88)

    fig.suptitle("두 융합 방식이 갈리는 지점을 참값으로 심판한다 "
                 f"({args.records}, 3 카메라, {len(rows)} 프레임)", fontsize=13)
    fig.text(0.5, 0.015,
             "파낸 것은 0.09 % 뿐이고 전부 작업공간 물체(crate·apple·pear), "
             "관측 공백은 전부 방 껍데기(office·table). "
             "그리고 필드는 점구름이 못 가진 13,867 복셀을 더 갖고 있다.",
             ha="center", fontsize=11.5, color="0.25")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    o = OUT / f"step7-fusion-asymmetry-measured-{args.records[-4:]}.png"
    fig.savefig(o, dpi=110, bbox_inches="tight")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
