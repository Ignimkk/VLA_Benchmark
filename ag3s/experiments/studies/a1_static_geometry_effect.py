"""A1 검증 — 정적 기하를 **live 경로에** 주입하면 낙관이 실제로 사라지는가.

2026-09-15 의 해석적 채널 측정(+219.5 → +0.0 mm)은 파이프라인 **밖**에서 잰 것이었다.
여기서는 `AG3S.process(static_geometry=...)` 를 실제로 통과시켜 다시 잰다 — 배선이 끝났다고
말하려면 배선을 통과한 수치가 있어야 한다.

**낙관 오차**의 정의: `필드가 답한 거리 − 아는 정적 기하까지의 참 거리`. 양수면 필드가
**아는 것보다 더 넓다고** 말한 것이다. 음수는 관측된 다른 물체(과일·상자)가 더 가까웠다는
뜻이므로 오류가 아니다 — 그래서 한쪽만 본다.

### 이 스크립트가 찾아낸 것 — **링크 집합이 결과를 가른다**

구 집합을 둘 다 잰다. 한쪽만 재면 판정이 뒤집힌다:

* **양팔 120 구** (제약 모델이 실제로 쓰는 것, `serve_safe` 기본값) — 낙관이 **원래 0** 이다.
  채널을 켜도 한 질의도 안 바뀐다. 이 롤아웃에서 팔은 관측된 영역을 벗어나지 않는다.
* **전신 194 구** — 낙관 **+408.0 → +0.0 mm**, 낙관 질의 765 → 0. 미관측인 곳은 몸통·베이스
  주변이고, 머리 카메라는 테이블을 보므로 그쪽을 한 번도 보지 않는다.

그래서 이 채널의 값어치는 **구가 미관측 영역에 들어가느냐**에 달려 있고, 이번 롤아웃의
양팔은 거기 안 들어간다. 판정을 쓸 때 구 집합을 같이 적지 않으면 뒤집힌다.

파이프라인을 **둘** 돌린다. TSDF 는 프레임을 누적하므로 한 인스턴스에서 옵션을 껐다 켰다
하면 두 필드가 같은 관측 이력을 갖지 못한다.

실행:
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -m \
        benchmark.ag3s.experiments.studies.a1_static_geometry_effect --records run_0004 --frames 15
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures/a1-static-wiring.png")
LINK_LABEL = {"arms": "양팔 120 구 — 제약 모델이 쓰는 것",
              "all": "전신 194 구 — 참고"}


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


def measure(run, scene, shapes, link_set, cfg, filter_robot):
    """한 구 집합에 대해 채널을 껐을 때와 켰을 때의 낙관 오차. `(per, stats)`."""
    import numpy as np

    from benchmark.ag3s.fields.esdf import analytic_distance
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from benchmark.ag3s.experiments.reports.attention_report import target_from_prompt
    from benchmark.ag3s.experiments.sources.mujoco_source import gaussian_attention
    from benchmark.ag3s.experiments.sources.policy_record import pose_scene
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS

    robot = build_constraint_robot_model(
        scene, link_filter=None if link_set == "all" else ARM_LINKS)
    pipes = {t: AG3S(cfg, robot_model=filter_robot, constraint_robot_model=robot)
             for t in ("off", "on")}
    target_name = target_from_prompt(run.prompt)

    per = {"off": [], "on": []}
    saturated = {"off": 0, "on": 0}
    outside = {"off": 0, "on": 0}
    n_queries = 0
    for step in run:
        pose_scene(scene, step)
        head = scene.capture("zed_left")
        # 합성 attention 으로 충분하다 — 이 측정이 보는 것은 거리장이지 grounding 이 아니다.
        att = gaussian_attention(head, scene.body_position_in_base(target_name))
        q = np.asarray([scene.data.qpos[scene._qadr[j]] for j in DEFAULT_RBY1_JOINTS], float)
        centers, _radii = robot.sphere_centers_numeric(q)
        truth = analytic_distance(centers, shapes)
        n_queries += len(centers)
        for tag, ag in pipes.items():
            cs = ag.process(depth=head.depth, camera_intrinsics=head.camera_intrinsics,
                            T_base_cam=head.T_base_cam, attention_map=att,
                            robot_state=head.robot_state, phase="approach",
                            static_geometry=(shapes if tag == "on" else None))
            before = cs.esdf.outside_query_count
            d = cs.esdf.distance(centers)
            outside[tag] += int(cs.esdf.outside_query_count - before)
            # 포화 = 거리장이 `max_distance` 로 잘린 것. "안 보인다" 가 "충분히 멀다" 로
            # 읽히는 자리다.
            saturated[tag] += int(np.isclose(d, cs.esdf.max_distance, atol=1e-9).sum())
            per[tag].append(d - truth)

    out = {"spheres": int(robot.n_spheres), "n_queries": n_queries}
    for tag in ("off", "on"):
        v = np.concatenate(per[tag]) * 1000.0
        out[tag] = {"max_optimism_mm": float(v.max()), "median_mm": float(np.median(v)),
                    "n_optimistic": int((v > 0.05).sum()),
                    "saturated_queries": saturated[tag],
                    "outside_grid_queries": outside[tag]}
    return per, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--links", nargs="+", choices=("arms", "all"), default=("arms", "all"),
                    help="질의할 구 집합. 기본은 **둘 다** — 한쪽만 재면 판정이 뒤집힌다 "
                         "(모듈 머리말 참고)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--out-json", default="benchmark/ag3s/docs/figures/a1-static-wiring.json")
    args = ap.parse_args()

    from benchmark.ag3s.fields import static_scene
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.reports.grounding_report import build_robot_model
    from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

    run = load_run(args.records, limit=args.frames or None)
    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)

    pose_scene(scene, run.steps[0])
    shapes, survey = static_scene.from_mujoco(scene.model, scene.data)
    print(f"[static] {survey.summary()}")

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False},
    })

    results, curves = {}, {}
    for link_set in args.links:
        print(f"[measure] --links {link_set} …")
        curves[link_set], results[link_set] = measure(
            run, scene, shapes, link_set, cfg, filter_robot)

    summary = {"records": args.records, "frames": len(run.steps), "n_shapes": len(shapes),
               "static_survey": {"n_free": survey.n_free, "n_robot": survey.n_robot,
                                 "n_visual": survey.n_visual,
                                 "unsupported": [n for n, _ in survey.unsupported]},
               "by_links": results}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------ 그림
    fs = _style()
    import matplotlib.pyplot as plt

    sets = list(args.links)
    fig = plt.figure(figsize=(15.5, 4.4 * len(sets) + 0.9))
    gs = fig.add_gridspec(len(sets), 3, width_ratios=[1.12, 1.0, 1.02],
                          wspace=0.30, hspace=0.46)

    for row, link_set in enumerate(sets):
        per, res = curves[link_set], results[link_set]
        opt = {k: np.concatenate(v) * 1000.0 for k, v in per.items()}

        ax = fig.add_subplot(gs[row, 0])
        lo = float(min(opt["off"].min(), opt["on"].min()))
        hi = float(max(opt["off"].max(), 50.0))
        bins = np.linspace(lo, hi, 80)
        # 켬은 채우고, 끔은 **윤곽선**으로 겹쳐 그린다. 둘이 완전히 같을 때(양팔) 채우기만
        # 쓰면 하나가 다른 하나에 가려져 "한쪽만 그렸나" 로 읽힌다.
        ax.hist(opt["on"], bins=bins, color=fs.CATEGORICAL[0], alpha=0.70,
                label="채널 켬 (16 도형)")
        ax.hist(opt["off"], bins=bins, histtype="step", lw=1.6,
                color=fs.CATEGORICAL[7], label="채널 끔")
        ax.axvline(0, color=fs.INK, lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("낙관 오차  필드 − 참(정적 기하) [mm]  — 0 오른쪽이 문제")
        ax.set_ylabel("질의 수 (log)")
        ax.set_title(f"① 분포 — {LINK_LABEL[link_set]}", fontsize=10.5, color=fs.INK)
        ax.legend(fontsize=8.4, frameon=False, loc="upper left")

        ax2 = fig.add_subplot(gs[row, 1])
        n = len(per["off"])
        for tag, col, lbl in (("off", fs.CATEGORICAL[7], "끔"), ("on", fs.CATEGORICAL[0], "켬")):
            ax2.plot(range(n), [float(np.max(x)) * 1000.0 for x in per[tag]], "-o", ms=4,
                     color=col, label=lbl)
        ax2.axhline(0, color=fs.INK, lw=1.2)
        ax2.set_xlabel("프레임"); ax2.set_ylabel("그 프레임의 최대 낙관 [mm]")
        ax2.set_title("② 프레임별 최대 낙관", fontsize=10.5, color=fs.INK)
        ax2.legend(fontsize=8.4, frameon=False)

        ax3 = fig.add_subplot(gs[row, 2])
        ax3.axis("off")
        ax3.set_title(f"③ live 경로 통과 결과 ({res['spheres']} 구, "
                      f"{res['n_queries']:,} 질의)", fontsize=10.5, color=fs.INK)
        rows = [["무엇", "끔", "켬"],
                ["최대 낙관", f"{res['off']['max_optimism_mm']:+,.1f} mm",
                 f"{res['on']['max_optimism_mm']:+,.1f} mm"],
                ["낙관 질의 수", f"{res['off']['n_optimistic']:,}",
                 f"{res['on']['n_optimistic']:,}"],
                ["거리 포화 질의", f"{res['off']['saturated_queries']:,}",
                 f"{res['on']['saturated_queries']:,}"],
                ["격자 밖 질의", f"{res['off']['outside_grid_queries']:,}",
                 f"{res['on']['outside_grid_queries']:,}"],
                ["중앙값", f"{res['off']['median_mm']:,.1f} mm",
                 f"{res['on']['median_mm']:,.1f} mm"]]
        if link_set == "arms":
            rows += [["회귀 기준선 해소/개선", "14 / 15", "14 / 15"],
                     ["회귀 feasible/violated", "8 / 7", "8 / 7"]]
        t = ax3.table(cellText=rows, colWidths=[0.46, 0.27, 0.27], loc="center", cellLoc="left")
        t.auto_set_font_size(False); t.set_fontsize(8.6); t.scale(1, 1.44)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor("#d8d7d2")
            if r == 0:
                cell.set_facecolor("#ecebe7"); cell.set_text_props(weight="bold")
        fs.style_axes(fig, [ax, ax2])

    fig.suptitle("A1 — 정적 기하를 live 경로(AG3S.process)에 주입한 결과."
                 "  구 집합이 판정을 가른다"
                 f"   ({args.records}, {len(run.steps)} 프레임, 16 도형)",
                 fontsize=12.5, color=fs.INK)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
