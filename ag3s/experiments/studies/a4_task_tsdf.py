"""작업 진행용 TSDF 를 따로 두자는 제안을 **잰다** (2026-09-17 사용자 제안).

제안은 다섯 단계였다:

1. 사과 attention 을 TSDF 위에 기록한다
2. **사과의 위치가 변동하면 pick 완료로 추정한다**   <- 이 스크립트가 재는 핵심
3. 다음 attention 은 바구니에 찍히고, 그것도 TSDF 에 기록한다
4. 사과가 바구니에 들어가면 바구니 attention 위에 사과가 안착한다
5. 이 과정으로 작업 완료를 판정한다

### 왜 2 번이 핵심인가

A2 에서 잰 것이 이것과 정면으로 부딪힌다: 쥔 사과는 프레임당 약 16,000 픽셀로 **보이는데**,
마스크를 걷어내도 점유 복셀이 **최대 4 개**밖에 안 바뀐다. TSDF 는 가중평균이라 **움직이는
표면은 어느 복셀에서도 점유로 뒤집힐 만큼 가중치를 못 모은다.** 그래서 "사과가 움직였다" 를
TSDF 로 보는 것이 되는지 안 되는지를 먼저 확정해야 한다.

다만 **뒤집어 읽으면 될 수도 있다**: 사과가 떠난 **옛 자리**는 TSDF 에 잔상으로 남아 있고,
그 잔상이 사라지면 "떠났다" 가 된다. 그것은 감쇠(F20 — 사라진 물체의 잔상이 8 프레임 뒤에도
남는다)가 하는 일이고, 지금 기본값은 **끔**이다. 그래서 감쇠 켬/끔을 나란히 잰다.

### 무엇을 재는가

부피 셋. 관측은 같고 **마스크와 감쇠만** 다르다:

* `obstacle`  — 지금 장애물용. 쥔 물체를 마스크로 지우고 감쇠 끔
* `task`      — 쥔 물체를 **안 지우고** 감쇠 끔 (제안의 소박한 형태)
* `task_decay`— 쥔 물체를 안 지우고 **감쇠 켬** (`a_t 0.99 · a_f 0.8`, F20 의 권고값)

그리고 세 영역의 점유를 센다:

* **옛 자리** — 사과가 테이블에 있던 프레임 0 의 자리 (2 번 신호가 여기서 나온다)
* **지금 자리** — 사과의 현재 자리 (2 번을 문자 그대로 하면 여기가 차야 한다)
* **바구니 안** — 목적지 내부 (4 번 신호)

영역의 정의에만 참값 자세를 쓴다 — **측정의 눈금자**이지 알고리즘의 입력이 아니다.

비용도 함께 쪼갠다: 작업용 TSDF 는 거리장(ESDF)이 필요 없고 점유와 라벨만 있으면 되므로,
**적분 대 거리변환**의 비율이 "따로 하나 더 두는 것이 싼가" 를 정한다.

실행:
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -m \
        benchmark.ag3s.experiments.studies.a4_task_tsdf --records run_0004 --frames 24
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/archive/14d-era-20260923/archive/14d-era-20260923/figures/a4-task-tsdf.png")
CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
PROBE_HALF = 0.06          # 사과 주변 탐침 상자의 반 변 (m)
#: 탐침 상자의 **아랫면**을 사과 중심보다 이만큼만 아래로 둔다. 전체 반 변을 쓰면 사과가
#: 놓여 있던 **테이블 상판**이 함께 세어져 (실측 182 복셀 중 대부분) 사과의 유무가 안 보인다.
PROBE_BELOW = 0.012


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


def _count_in_box(occ, grid, lo, hi):
    """`lo..hi` 상자 안의 점유 복셀 수."""
    from benchmark.ag3s.fields.esdf import OCCUPIED
    origin = np.asarray(grid.origin, float)
    size = float(grid.voxel_size)
    i0 = np.maximum(np.floor((np.asarray(lo) - origin) / size).astype(int), 0)
    i1 = np.minimum(np.ceil((np.asarray(hi) - origin) / size).astype(int) + 1,
                    np.asarray(grid.shape))
    if np.any(i1 <= i0):
        return 0
    sub = occ[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
    return int((sub == OCCUPIED).sum())


def main() -> None:
    import mujoco

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--held", default="apple")
    ap.add_argument("--destination", default="crate")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--time-decay", type=float, default=0.99)
    ap.add_argument("--frustum-decay", type=float, default=0.8)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--out-json", default="benchmark/ag3s/docs/archive/14d-era-20260923/archive/14d-era-20260923/figures/a4-task-tsdf.json")
    args = ap.parse_args()

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.fields.esdf import CameraDepth, EsdfBuilder, TsdfVolume
    from benchmark.ag3s.experiments.reports.grounding_report import (
        build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS

    run = load_run(args.records, limit=args.frames or None)
    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    robot = build_constraint_robot_model(scene)

    held_b = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, args.held)
    floor_g = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "crate_floor")
    rim_g = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "crate_wall_px")

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False},
    })
    ag = AG3S(cfg, robot_model=filter_robot, constraint_robot_model=robot)

    decay_cfg = dataclasses.replace(cfg.esdf, time_decay=args.time_decay,
                                    frustum_decay=args.frustum_decay)
    builders = {"obstacle": EsdfBuilder(cfg.esdf),
                "task": EsdfBuilder(cfg.esdf),
                "task_decay": EsdfBuilder(decay_cfg)}

    rows = []
    origin_box = None
    print(f"{'i':>3} {'사과px':>7} "
          + "  ".join(f"{k}:옛/지금/바구니" for k in builders)
          + f"   {'적분ms':>7} {'전체ms':>7}")

    for i, step in enumerate(run.steps):
        pose_scene(scene, step)
        caps = {c: scene.capture(c) for c in CAMERAS}

        cams = {"masked": [], "unmasked": []}
        n_held_px = 0
        for c, fr in caps.items():
            K = np.asarray(fr.camera_intrinsics, np.float64)
            T = np.asarray(fr.T_base_cam, np.float64)
            d = np.asarray(fr.depth, np.float64)
            m = ag._robot_mask_for(d, K, T, fr.robot_state)
            m = np.zeros(d.shape, bool) if m is None else np.asarray(m, bool)
            hp = np.asarray(fr.body_ids, np.int64) == held_b
            n_held_px += int(hp.sum())
            cams["masked"].append(CameraDepth(c, d, K, T, robot_mask=m))
            # 작업용은 쥔 물체를 **남긴다** — 그것이 판정 대상이기 때문이다.
            cams["unmasked"].append(CameraDepth(c, d, K, T, robot_mask=m & ~hp))

        # --- 탐침 영역 (참값은 **눈금자**로만 쓴다) -------------------------------------
        held_c = np.asarray(scene.data.xpos[held_b], np.float64)
        def _probe(centre):
            lo = np.asarray(centre, float) - PROBE_HALF
            lo[2] = float(centre[2]) - PROBE_BELOW      # 지지면을 빼낸다
            return lo, np.asarray(centre, float) + PROBE_HALF

        if origin_box is None:
            origin_box = _probe(held_c)
        now_box = _probe(held_c)
        f_pos = np.asarray(scene.data.geom_xpos[floor_g], np.float64)
        f_sz = np.asarray(scene.model.geom_size[floor_g], np.float64)
        rim_z = float(scene.data.geom_xpos[rim_g][2] + scene.model.geom_size[rim_g][2])
        crate_box = (np.array([f_pos[0] - f_sz[0], f_pos[1] - f_sz[1], f_pos[2] + 0.01]),
                     np.array([f_pos[0] + f_sz[0], f_pos[1] + f_sz[1], rim_z]))

        counts = {}
        t_int = t_all = 0.0
        for tag, b in builders.items():
            src = "masked" if tag == "obstacle" else "unmasked"
            t0 = time.perf_counter()
            field = b.update(cams[src])
            dt = (time.perf_counter() - t0) * 1000.0
            occ = b._occupancy
            counts[tag] = (_count_in_box(occ, b.grid, *origin_box),
                           _count_in_box(occ, b.grid, *now_box),
                           _count_in_box(occ, b.grid, *crate_box))
            if tag == "task":
                t_all = dt
                # 적분만 따로 — 작업용 TSDF 는 거리장이 필요 없으므로 이 부분만 내면 된다.
                scratch = TsdfVolume(b.grid, truncation=cfg.esdf.truncation)
                t0 = time.perf_counter()
                for cam in cams[src]:
                    scratch.integrate(cam.depth, cam.camera_intrinsics, cam.T_base_cam,
                                      depth_min=cfg.esdf.depth_min,
                                      depth_max=cfg.esdf.depth_max,
                                      max_weight=cfg.esdf.max_weight,
                                      robot_mask=cam.robot_mask)
                t_int = (time.perf_counter() - t0) * 1000.0

        rows.append(dict(i=i, n_held_px=n_held_px,
                         counts={k: list(v) for k, v in counts.items()},
                         integrate_ms=t_int, update_ms=t_all))
        print(f"{i:>3} {n_held_px:>7} "
              + "  ".join(f"{counts[k][0]:>3}/{counts[k][1]:>3}/{counts[k][2]:>3}"
                          for k in builders)
              + f"   {t_int:>7.1f} {t_all:>7.1f}")

    def _series(tag, k):
        return [r["counts"][tag][k] for r in rows]

    base = {t: (_series(t, 0)[0] or 1) for t in builders}
    summary = {
        "records": args.records, "frames": len(rows), "voxel_mm": args.voxel * 1000,
        "decay": {"time": args.time_decay, "frustum": args.frustum_decay},
        "cost": {"integrate_ms_median": float(np.median([r["integrate_ms"] for r in rows])),
                 "update_ms_median": float(np.median([r["update_ms"] for r in rows])),
                 "integrate_fraction": float(np.median([r["integrate_ms"] for r in rows])
                                             / max(np.median([r["update_ms"] for r in rows]), 1e-9))},
    }
    for t in builders:
        old, now, crate = _series(t, 0), _series(t, 1), _series(t, 2)
        summary[t] = {
            "old_site_first": old[0], "old_site_last": old[-1],
            "old_site_min": int(min(old)),
            # 2 번 신호: 옛 자리가 절반 아래로 비는 첫 프레임.
            "old_site_emptied_frame": next(
                (r["i"] for r in rows if r["counts"][t][0] < 0.5 * base[t]), None),
            # 2 번을 문자 그대로: 사과의 **지금** 자리가 차는가.
            "now_site_max_during_transport": int(max(now[10:19])) if len(now) > 18 else None,
            "now_site_max_overall": int(max(now)),
            "crate_first": crate[0], "crate_last": crate[-1],
            "crate_gain": crate[-1] - crate[0],
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_json).write_text(json.dumps(
        {**summary, "rows": rows}, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------ 그림
    fs = _style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.8, 8.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 0.95], wspace=0.30, hspace=0.44)
    idx = [r["i"] for r in rows]
    cols = {"obstacle": fs.CATEGORICAL[2], "task": fs.CATEGORICAL[0],
            "task_decay": fs.CATEGORICAL[7]}
    name = {"obstacle": "장애물용 (쥔 물체 지움, 감쇠 끔)",
            "task": "작업용 (쥔 물체 남김, 감쇠 끔)",
            "task_decay": f"작업용 + 감쇠 ({args.time_decay} · {args.frustum_decay})"}

    ax = fig.add_subplot(gs[0, 0])
    for t in builders:
        ax.plot(idx, _series(t, 0), "-o", ms=3.5, color=cols[t], label=name[t])
    ax.set_xlabel("프레임"); ax.set_ylabel("점유 복셀 수")
    ax.set_title("① 사과의 옛 자리 — 2 번 신호는 여기서 나온다", fontsize=10.2, color=fs.INK)
    ax.legend(fontsize=7.6, frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    for t in builders:
        ax2.plot(idx, _series(t, 1), "-s", ms=3.5, color=cols[t], label=name[t])
    ax2.set_xlabel("프레임"); ax2.set_ylabel("점유 복셀 수")
    ax2.set_title("② 사과의 지금 자리 — 움직이는 사과는 TSDF 에 안 들어온다",
                  fontsize=10.2, color=fs.INK)
    ax2.legend(fontsize=7.6, frameon=False)

    ax3 = fig.add_subplot(gs[1, 0])
    for t in builders:
        ax3.plot(idx, _series(t, 2), "-^", ms=3.5, color=cols[t], label=name[t])
    ax3.set_xlabel("프레임"); ax3.set_ylabel("점유 복셀 수")
    ax3.set_title("③ 바구니 안 — 4 번 신호 (안착)", fontsize=10.2, color=fs.INK)
    ax3.legend(fontsize=7.6, frameon=False)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(idx, [r["n_held_px"] for r in rows], "-o", ms=3.5, color=fs.CATEGORICAL[3])
    ax4.set_xlabel("프레임"); ax4.set_ylabel("사과 픽셀 수 (3 카메라)")
    ax4.set_title("④ 사과는 내내 잘 보인다 — 못 보는 것은 TSDF 다",
                  fontsize=10.2, color=fs.INK)

    ax5 = fig.add_subplot(gs[:, 2])
    ax5.axis("off")
    ax5.set_title("⑤ 제안의 단계별 판정", fontsize=10.5, color=fs.INK)
    trows = [["제안 단계", "결과"]]
    for t in builders:
        s = summary[t]
        trows.append([f"[{name[t].split(' (')[0]}]", ""])
        trows.append(["  옛 자리 첫/끝", f"{s['old_site_first']} → {s['old_site_last']}"])
        trows.append(["  옛 자리가 비는 프레임", f"{s['old_site_emptied_frame']}"])
        trows.append(["  지금 자리 최대 (옮기는 중)",
                      f"{s['now_site_max_during_transport']}"])
        trows.append(["  바구니 증가", f"{s['crate_gain']:+d}"])
    trows += [["", ""],
              ["적분 (중앙)", f"{summary['cost']['integrate_ms_median']:.0f} ms"],
              ["갱신 전체 (중앙)", f"{summary['cost']['update_ms_median']:.0f} ms"],
              ["적분이 차지하는 비율", f"{summary['cost']['integrate_fraction']*100:.0f} %"]]
    t = ax5.table(cellText=trows, colWidths=[0.58, 0.42], loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(7.8); t.scale(1, 1.42)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if r == 0 or (trows[r][1] == "" and trows[r][0]):
            cell.set_facecolor("#ecebe7"); cell.set_text_props(weight="bold")

    fs.style_axes(fig, [ax, ax2, ax3, ax4])
    fig.suptitle("작업 진행용 TSDF 를 따로 둔다는 제안 — 어느 단계가 되고 어느 단계가 안 되나"
                 f"   ({args.records}, {len(rows)} 프레임, 3 카메라, 복셀 "
                 f"{args.voxel*1000:.0f} mm)", fontsize=12.5, color=fs.INK)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
