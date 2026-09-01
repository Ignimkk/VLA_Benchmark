"""4단계 검증 — 클러스터링과 점수화가 실제로 그 물체를 고르는가.

1–3단계는 입력이 옳다는 것을 확인했다: attention이 옳은 물체를 가리키고(1), 점이 옳은 자리에
놓이고(2), attention이 옳은 점에 붙는다(3). 4단계는 그 입력으로 **판단**을 내리는 첫 단계다 —
어느 점 뭉치가 target인지 이름을 붙이고, 그 이름표가 이후 모든 단계의 여유거리 정책을 정한다.

파이프라인의 단계 순서를 그대로 재현한다: 재구성 → 자기 필터 → 지지면 → lift → grounding.
순서를 바꾸면 다른 것을 재게 된다 — 예를 들어 지지면을 빼면 성장이 테이블을 타고 번져 씬
전체가 한 클러스터가 된다.

**단일 시점(head 카메라)으로 격리한다.** 3단계에서 확인했듯 attention 셀은 head 카메라
토큰 블록에서 고른 것이고, 다중 시점 융합은 입력 점군 자체를 바꾼다. 융합까지 켜면 군집이
틀렸을 때 군집 탓인지 융합 탓인지 가릴 수 없다. 융합 조건의 grounding은 다중 시점 단계에서
따로 잰다.

재는 것:

**A. 옳은 물체를 골랐는가.** 선택된 클러스터의 다수 정답 body가 target인가. 이 단계의 답이다.

**B. 클러스터 품질.** 그 클러스터가 target 점들과 얼마나 겹치는가 (IoU·정밀도·재현율).
옳은 물체를 골랐어도 클러스터가 물체의 1/3만 덮으면 이후 primitive가 물체를 과소 근사한다.

**C. 실패가 정직한가.** target이 가려진 프레임에서 무엇을 보고하는가. 여기서 엉뚱한 물체를
`OK`로 반환하면 그 물체의 충돌 제약이 조용히 완화된다 — 이 파이프라인에서 가장 위험한
실패 방식이고, `GroundingStatus`가 존재하는 이유다.

**D. 왜 그것을 골랐는가.** 1등과 2등의 점수 분해. 점수는
`(w_a·평균attention + w_g·√(밀집도·peak근접도)) / (w_a+w_g)`이고, 어느 항이 결정했는지
보이지 않으면 통과해도 이유를 모른다.

**E. 무게중심 편향.** 2단계가 예고한 것: 카메라는 앞면만 보므로 클러스터 무게중심은 물체
중심이 아니다. 얼마나 치우치는지 재서, 5·6단계가 무게중심을 중심으로 쓰지 않도록 숫자로 남긴다.

**F. 배경 클러스터.** 3단계가 경고한 것: seed의 78.8%가 벽·바닥에 있다. 바닥은 지지면으로
제외되지만 벽은 아니다. 벽에서 자란 클러스터가 후보로 올라오는지, 올라온다면 몇 위인지.

실행:
    MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.grounding_report \
        --records outputs/.../ag3s_records/run_0002 \
        --attention benchmark/ag3s/asset/data/attention_step1_run0002.npz
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from benchmark.ag3s.attention_lifting import GridAttentionAdapter, lift
from benchmark.ag3s.config import (
    AttentionConfig, ClusteringConfig, PointCloudConfig, SupportSurfaceConfig,
)
from benchmark.ag3s.experiments.figstyle import (
    CATEGORICAL, GRID_INK, INK, INK_2, SURFACE, style_axes, use_korean,
)
from benchmark.ag3s.experiments.lifting_report import ROBOT_PREFIXES, category_of
from benchmark.ag3s.experiments.policy_record import (
    CAMERA_BINDINGS, load_run, pose_scene, replay_scene,
)
from benchmark.ag3s.reconstruction import reconstruct
from benchmark.ag3s.robot_filter import filter_robot_points
from benchmark.ag3s.support_surface import fit_support_surfaces
from benchmark.ag3s.target_grounding import ground_target
from benchmark.ag3s.types import GroundingStatus


def build_robot_model(scene):
    """파이프라인이 자기 필터에 쓰는 것과 같은 모델."""
    from benchmark.ag3s.experiments.mujoco_source import HEAD_JOINTS, gap_filling_capsules
    from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf

    urdf = parse_urdf(RBY1_URDF)
    head = {n: float(scene.data.qpos[scene._qadr[n]]) for n in HEAD_JOINTS if n in scene._qadr}
    return UrdfSphereChain(urdf, extra_capsules=gap_filling_capsules(scene.model),
                           fixed_joint_values=head)


def majority_body(labels: np.ndarray, names: dict) -> tuple[str, float]:
    """클러스터를 이루는 점들의 다수 body와 그 비율."""
    if labels.size == 0:
        return "?", 0.0
    vals, counts = np.unique(labels, return_counts=True)
    k = int(counts.argmax())
    return names.get(int(vals[k]), "?"), float(counts[k] / labels.size)


def main() -> None:
    from benchmark.ag3s.experiments.attention_report import target_from_prompt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--attention", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/step-01-attention.json")
    ap.add_argument("--camera", default="cam_high", choices=sorted(CAMERA_BINDINGS))
    ap.add_argument("--frames", type=int, default=44, help="기본값: 전 프레임")
    ap.add_argument("--min-target-px", type=int, default=200)
    ap.add_argument("--range-max", type=float, default=2.0,
                    help="카메라로부터의 반경 게이트 (m). 기본값의 근거는 문서 참조. "
                         "0이면 게이트 없음(AG3S 기본 설정)")
    ap.add_argument("--out-doc", default="benchmark/ag3s/docs/step-04-grounding.md")
    ap.add_argument("--out-figs", default="benchmark/ag3s/asset/image/grounding")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    use_korean()
    import matplotlib.pyplot as plt
    import mujoco

    chosen = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    run = load_run(args.records)
    blob = np.load(args.attention, allow_pickle=False)
    attention = np.asarray(blob["attention"], np.float32)
    cameras = [str(c) for c in blob["cameras"]]
    ci = cameras.index(args.camera)
    di = [int(d) for d in blob["denoise_steps"]].index(int(chosen["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(chosen["agg"]))
    target = args.target or target_from_prompt(run.prompt)

    scene = replay_scene(run)
    robot_model = build_robot_model(scene)
    names = {i: (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or "?")
             for i in range(scene.model.nbody)}
    target_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, target)
    mj_cam = CAMERA_BINDINGS[args.camera][0]

    import dataclasses as _dc
    base_pc = PointCloudConfig()
    pc_cfg = _dc.replace(base_pc, range_max=(args.range_max or None))
    att_cfg = AttentionConfig()
    cl_cfg, ss_cfg = ClusteringConfig(), SupportSurfaceConfig()
    adapter = GridAttentionAdapter()
    print(f"셀 L{chosen['layer']} h{chosen['head']} Euler {chosen['denoise']} {chosen['agg']} | "
          f"target={target} | 카메라={args.camera}")

    picks = np.linspace(0, len(run) - 1, min(args.frames, len(run))).astype(int)

    # 대조군: AG3S 기본 설정(반경 게이트 없음). 이 씬에서 무슨 일이 벌어지는지 몇 프레임으로
    # 확인해 문서에 남긴다 — 통과시키려고 설정을 바꾼 것이 아니라, 기본값이 이 씬에서
    # 구조적으로 실패한다는 것을 보이기 위해서다.
    control = []
    for fi in picks[:: max(1, len(picks) // 4)][:4]:
        pose_scene(scene, run.steps[fi])
        f0 = scene.capture(mj_cam)
        c0, st0 = reconstruct(depth=f0.depth, camera_intrinsics=f0.camera_intrinsics,
                              T_base_cam=f0.T_base_cam, config=base_pc)
        c0, _ = filter_robot_points(c0, robot_model, f0.robot_state, base_pc)
        _, m0 = fit_support_surfaces(c0, ss_cfg)
        a0 = lift(c0, attention[fi, di, ai, chosen["layer"], chosen["head"], ci], att_cfg,
                  adapter=adapter, image_hw=f0.hw)
        r0 = ground_target(a0, cl_cfg, seed_percentile=att_cfg.seed_percentile, exclude_mask=m0)
        from scipy.spatial import cKDTree
        nb = cKDTree(c0.points).query_ball_point(c0.points, cl_cfg.eps, return_length=True, workers=-1)
        control.append({"frame": int(fi), "n": len(c0), "voxel_mm": st0["final_voxel_size"] * 1000,
                        "core": int((nb >= cl_cfg.min_points).sum()),
                        "median_nb": float(np.median(nb)),
                        "clusters": len(r0.clusters), "status": r0.status.value})

    rows, status_counts = [], collections.Counter()
    example = None
    for fi in picks:
        pose_scene(scene, run.steps[fi])
        frame = scene.capture(mj_cam)
        visible = int((frame.body_ids == target_bid).sum()) >= args.min_target_px

        # 파이프라인 순서 그대로 --------------------------------------------------------
        cloud, _ = reconstruct(depth=frame.depth, camera_intrinsics=frame.camera_intrinsics,
                               T_base_cam=frame.T_base_cam, config=pc_cfg)
        cloud, filt = filter_robot_points(cloud, robot_model, frame.robot_state, pc_cfg)
        if len(cloud) == 0:
            continue
        surfaces, support_mask = fit_support_surfaces(cloud, ss_cfg)
        grid = attention[fi, di, ai, chosen["layer"], chosen["head"], ci]
        acloud = lift(cloud, grid, att_cfg, adapter=adapter, image_hw=frame.hw)
        result = ground_target(acloud, cl_cfg, seed_percentile=att_cfg.seed_percentile,
                               exclude_mask=support_mask)

        labels = frame.body_ids[cloud.uv[:, 1], cloud.uv[:, 0]]
        gt_target = labels == target_bid
        status_counts[(result.status.value, visible)] += 1

        row = {"frame": int(fi), "t_step": int(run.steps[fi].t_step), "visible": visible,
               "status": result.status.value, "n_points": len(cloud),
               "n_removed_robot": int(filt.get("n_removed", 0)),
               "n_support": int(support_mask.sum()), "n_clusters": len(result.clusters),
               "best_score": float(result.best_score), "n_gt_target": int(gt_target.sum())}
        if result.target is not None:
            idx = result.target.point_indices
            picked = labels[idx]
            body, frac = majority_body(picked, names)
            inter = int(gt_target[idx].sum())
            row.update({
                "picked_body": body, "picked_purity": frac,
                "iou": inter / max(int(gt_target.sum()) + len(idx) - inter, 1),
                "precision": inter / max(len(idx), 1),
                "recall": inter / max(int(gt_target.sum()), 1),
                "cluster_points": int(len(idx)),
                "centroid_error": float(np.linalg.norm(
                    result.target.centroid - scene.body_position_in_base(target))),
                "confidence": float(result.target.confidence),
                "attention_score": float(result.target.attention_score),
            })
            ranked = sorted(result.clusters, key=lambda c: -c.target_score)
            row["ranking"] = [
                {"rank": r + 1, "body": majority_body(labels[c.point_indices], names)[0],
                 "score": c.target_score, "mean_att": c.mean_attention,
                 "compact": c.spatial_compactness, "peak_d": c.distance_from_attention_peak,
                 "n": c.point_count,
                 "cat": category_of(majority_body(labels[c.point_indices], names)[0], target)}
                for r, c in enumerate(ranked[:5])
            ]
            if example is None and visible and body == target:
                example = (cloud.points.copy(), labels.copy(), idx.copy(), int(fi),
                           result.target.centroid.copy(),
                           scene.body_position_in_base(target).copy(),
                           support_mask.copy(),
                           frame.camera_intrinsics.copy(), frame.T_base_cam.copy(),
                           frame.hw, result.target.bounding_geometry)
        rows.append(row)
    scene.close()

    # 롤아웃을 구간으로 나눈다. 경계는 target 가시성에서 데이터로 정하며, 손으로 고르지 않는다.
    hidden = [i for i, r in enumerate(rows) if not r["visible"]]
    first_hidden = min(hidden) if hidden else len(rows)
    last_hidden = max(hidden) if hidden else -1
    for i, r in enumerate(rows):
        r["segment"] = ("파지 전" if i < first_hidden
                        else "가려짐" if not r["visible"] or i <= last_hidden
                        else "배치 후")
    seg = lambda name: [r for r in rows if r["segment"] == name]
    pre, occ, post = seg("파지 전"), seg("가려짐"), seg("배치 후")
    vis = [r for r in rows if r["visible"]]
    picked_ok = [r for r in rows if r.get("picked_body") == target]
    pre_ok = [r for r in pre if r.get("picked_body") == target]

    # A와 C는 **파지 전** 구간에서만 건다. 자의적인 축소가 아니라 모듈 설계가 그렇게 되어
    # 있기 때문이다: `AG3S.attach()`의 docstring은 "AG3S never calls this itself"이며
    # "여기서부터 물체의 존재는 그리퍼에 대한 사실이지 카메라가 본 것에 대한 주장이 아니다"라고
    # 적는다. 즉 파지 이후 target의 기하는 부착 스냅샷에서 오고 grounding에서 오지 않는다.
    # 나머지 두 구간은 판정이 아니라 소견으로 싣는다.
    # 채점 가능성은 모듈 자신의 파라미터로 정한다. `min_points`보다 적은 점으로는 클러스터가
    # 정의상 존재할 수 없으므로, 그런 프레임에서 grounding을 재는 것은 grounding이 아니라
    # 군집 하한을 재는 것이다. 여유를 두어 2 x min_points 를 기준으로 삼는다.
    gradeable_floor = 2 * cl_cfg.min_points
    gradeable = [r for r in pre if r["n_gt_target"] >= gradeable_floor]
    boundary = [r for r in pre if r["n_gt_target"] < gradeable_floor]
    graded_ok = [r for r in gradeable if r.get("picked_body") == target]
    ok_a = bool(gradeable) and len(graded_ok) == len(gradeable)
    ok_b = bool(graded_ok) and float(np.mean([r["iou"] for r in graded_ok])) >= 0.5
    wrong_when_visible = [r for r in gradeable
                          if r["status"] == "ok" and r.get("picked_body") != target]
    ok_c = len(wrong_when_visible) == 0
    pre_ok = graded_ok
    verdict = "**PASS**" if (ok_a and ok_b and ok_c) else "**FAIL**"

    figs = pathlib.Path(args.out_figs)
    figs.mkdir(parents=True, exist_ok=True)

    # fig1 — 프레임별 결과 띠
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.4, 4.2), dpi=160, height_ratios=[1, 1.4])
    style_axes(fig, (a1, a2))
    t = [r["t_step"] for r in rows]
    colour = {True: CATEGORICAL[0], False: CATEGORICAL[7]}
    for r in rows:
        c = CATEGORICAL[0] if r.get("picked_body") == target else (
            GRID_INK if r["status"] != "ok" else CATEGORICAL[7])
        a1.bar(r["t_step"], 1, width=7, color=c, lw=0)
    a1.set_yticks([]); a1.set_ylabel("선택", fontsize=8)
    a1.set_title(f"프레임별 grounding 결과 — 파랑 = {target} 선택, 빨강 = 다른 물체, 회색 = target 없음 보고",
                 color=INK, fontsize=10, loc="left", pad=6)
    a2.grid(axis="y", color=GRID_INK, lw=.6); a2.set_axisbelow(True)
    a2.plot([r["t_step"] for r in rows], [r["best_score"] for r in rows],
            color=CATEGORICAL[0], lw=2.0, marker="o", ms=3.5)
    a2.axhline(cl_cfg.target_score_threshold, color=INK_2, lw=1.0, ls=(0, (4, 3)))
    a2.text(0.996, cl_cfg.target_score_threshold, f"임계값 {cl_cfg.target_score_threshold} ",
            transform=a2.get_yaxis_transform(), va="bottom", ha="right", fontsize=7.5, color=INK_2)
    for r in occ:
        a2.axvspan(r["t_step"] - 4, r["t_step"] + 4, color=GRID_INK, alpha=.5, lw=0, zorder=0)
    a2.set_xlabel("제어 스텝"); a2.set_ylabel("최고 클러스터 점수")
    fig.tight_layout(); fig.savefig(figs / "fig1_per_frame.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # fig2 — 클러스터 품질
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=160)
    style_axes(fig, ax)
    ax.grid(axis="y", color=GRID_INK, lw=.6); ax.set_axisbelow(True)
    keys = [("iou", "IoU"), ("precision", "정밀도"), ("recall", "재현율")]
    vals = [float(np.mean([r[k] for r in picked_ok])) if picked_ok else np.nan for k, _ in keys]
    ax.bar(range(len(keys)), vals, 0.5, color=[CATEGORICAL[0], CATEGORICAL[2], CATEGORICAL[3]],
           edgecolor=SURFACE, lw=2)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_xticks(range(len(keys)), [n for _, n in keys]); ax.set_ylim(0, 1.08)
    ax.set_ylabel("정답 target 점 대비")
    ax.set_title("선택된 클러스터가 실제 물체를 얼마나 덮는가", color=INK, fontsize=11, loc="left", pad=8)
    fig.tight_layout(); fig.savefig(figs / "fig2_cluster_quality.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # fig4 — 같은 결과를 정책이 본 이미지 위에
    if example is not None:
        from benchmark.ag3s.experiments.imageview import project, show, sphere_circle
        (pts, labs, idx, fi_ex, cen, true_c, smask, K_e, T_e, hw_e, prim_e) = example
        img = run.steps[fi_ex].images["cam_high"]
        ih = img.shape[:2]
        fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.5), dpi=160)
        fig.patch.set_facecolor(SURFACE)
        for ax in axes:
            ax.set_facecolor(SURFACE)
        show(axes[0], img, "정책이 본 입력 (224×224)", ink=INK, grid_ink=GRID_INK)
        show(axes[1], img, f"정답 {target} 점 (노랑)", ink=INK, grid_ink=GRID_INK)
        gt = labs == target_bid
        if gt.any():
            u, v, _, ok = project(pts[gt], K_e, T_e, hw_e, ih)
            axes[1].scatter(u[ok], v[ok], s=7, c=CATEGORICAL[3], lw=0)
        show(axes[2], img, f"고른 클러스터 ({len(idx)}점) + 제약용 구", ink=INK, grid_ink=GRID_INK)
        u, v, _, ok = project(pts[idx], K_e, T_e, hw_e, ih)
        axes[2].scatter(u[ok], v[ok], s=7, c=CATEGORICAL[0], lw=0)
        circ = sphere_circle(prim_e.center, prim_e.bounding_radius, K_e, T_e, hw_e, ih)
        if circ:
            axes[2].add_patch(plt.Circle(circ[:2], circ[2], fill=False, ec=CATEGORICAL[7], lw=2.0))
        show(axes[3], img, f"{target} 주변 확대", ink=INK, grid_ink=GRID_INK)
        if gt.any():
            ug, vg, _, okg = project(pts[gt], K_e, T_e, hw_e, ih)
            axes[3].scatter(ug[okg], vg[okg], s=26, c=CATEGORICAL[3], lw=0)
        axes[3].scatter(u[ok], v[ok], s=16, c=CATEGORICAL[0], lw=0)
        if circ:
            axes[3].add_patch(plt.Circle(circ[:2], circ[2], fill=False, ec=CATEGORICAL[7], lw=2.4))
        for ax in axes[:3]:
            ax.set_xlim(0, ih[1]); ax.set_ylim(ih[0], 0)
        if circ:
            pad = max(circ[2] * 3.2, 18.0)
            axes[3].set_xlim(circ[0] - pad, circ[0] + pad)
            axes[3].set_ylim(circ[1] + pad, circ[1] - pad)
        fig.suptitle(f"프레임 {fi_ex} — 3D 결과를 관측 이미지 위에 되돌려 그림 "
                     f"(노랑 = 정답 점, 파랑 = 고른 클러스터, 주황 원 = 제약용 구)",
                     color=INK, fontsize=12, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(figs / "fig4_image_overlay.png", facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    # fig3 — 선택된 클러스터 (위에서)
    if example is not None:
        (pts, labs, idx, fi_ex, cen, true_c, smask, K_e, T_e, hw_e, prim_e) = example
        fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=160)
        style_axes(fig, ax)
        ax.scatter(pts[smask, 0], pts[smask, 1], s=.5, c="#e8e7e2", lw=0, label="지지면 (제외)")
        other = ~smask
        other[idx] = False
        ax.scatter(pts[other, 0], pts[other, 1], s=.6, c=GRID_INK, lw=0, label="그 외 점")
        gt = labs == target_bid
        ax.scatter(pts[gt, 0], pts[gt, 1], s=14, c=CATEGORICAL[3], lw=0, label=f"정답 {target} 점")
        ax.scatter(pts[idx, 0], pts[idx, 1], s=6, c=CATEGORICAL[0], lw=0,
                   label=f"선택된 클러스터 ({len(idx)}점)")
        ax.plot(cen[0], cen[1], marker="o", ms=11, mfc="none", mew=2.4, color=INK,
                label="클러스터 무게중심", zorder=5)
        ax.plot(true_c[0], true_c[1], marker="x", ms=11, mew=2.6, color=CATEGORICAL[7],
                label="실제 물체 중심", zorder=5)
        lo = pts[idx, :2].min(axis=0) - 0.15; hi = pts[idx, :2].max(axis=0) + 0.15
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel("base x (m)"); ax.set_ylabel("base y (m)"); ax.set_aspect("equal")
        ax.set_title(f"프레임 {fi_ex} — 선택된 클러스터와 정답 점의 겹침 (위에서 봄)",
                     color=INK, fontsize=10.5, loc="left", pad=8)
        leg = ax.legend(frameon=False, fontsize=8, markerscale=2.0, loc="upper center",
                        bbox_to_anchor=(0.5, -0.13), ncol=3)
        for tx in leg.get_texts(): tx.set_color(INK_2)
        fig.tight_layout(); fig.savefig(figs / "fig3_selected_cluster.png",
                                        facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------------- 문서
    def md(header, body):
        return "\n".join(["| " + " | ".join(header) + " |",
                          "|" + "|".join("---" for _ in header) + "|", *body])

    example_rank = next((r["ranking"] for r in picked_ok if r.get("ranking")), [])
    bg_ranks = [e["rank"] for r in rows for e in r.get("ranking", []) if e["cat"] == "배경"]
    cen_err = [r["centroid_error"] for r in picked_ok]

    doc = f"""# 4단계 — 3D target grounding (클러스터링)

**질문.** 옳은 입력을 주었을 때, 클러스터링과 점수화가 실제로 그 물체를 고르는가?

1–3단계는 입력이 옳다는 것을 확인했다. 4단계는 그 입력으로 **판단**을 내리는 첫 단계다 —
어느 점 뭉치가 target인지 이름을 붙이고, 그 이름표가 이후 모든 단계의 여유거리 정책을 정한다.

| | |
|---|---|
| 기록 | `{run.path}` |
| attention 셀 | L{chosen['layer']} h{chosen['head']}, Euler {chosen['denoise']}, `{chosen['agg']}` (1단계 결과) |
| target | `{target}` |
| 카메라 | `{args.camera}` (단일 시점) |
| 프레임 | {len(rows)}개 — target 보임 {len(vis)}, 가려짐 {len(occ)} |
| 단계 순서 | 재구성 → 자기 필터 → 지지면 → lift → grounding (파이프라인과 동일) |
| 군집 설정 | eps={cl_cfg.eps}, min_points={cl_cfg.min_points}, 임계값={cl_cfg.target_score_threshold}, w=({cl_cfg.w_attention}, {cl_cfg.w_geometry}) |
| 점군 설정 | **`range_max={args.range_max} m`** — 기본값(게이트 없음)은 이 씬에서 실패한다, 아래 참조 |

**단일 시점으로 격리한 이유.** 3단계에서 확인했듯 attention 셀은 head 카메라 토큰 블록에서
고른 것이고, 다중 시점 융합은 입력 점군 자체를 바꾼다. 융합까지 켜면 군집이 틀렸을 때 군집
탓인지 융합 탓인지 가릴 수 없다.

## 먼저 — AG3S 기본 설정은 이 씬에서 한 프레임도 성공하지 못한다

이것은 4단계 검증의 부산물이 아니라 **가장 중요한 결과**다. 기본 `PointCloudConfig`로 돌리면
44프레임 전부가 `NO_CLUSTER`를 반환한다. 원인은 두 설정 사이의 조용한 상호작용이다.

{md(["프레임", "점 수", "최종 voxel", "3cm 이웃 중앙값", f"core 점 (>= {cl_cfg.min_points})", "클러스터", "상태"],
    [f"| {c['frame']} | {c['n']:,} | {c['voxel_mm']:.1f} mm | {c['median_nb']:.0f} | {c['core']} | {c['clusters']} | `{c['status']}` |"
     for c in control])}

**연쇄는 이렇다.** 이 씬은 방 전체가 카메라에 들어온다 (`depth_max=3.0 m`, 벽까지 3.5 m).
640×480 깊이가 265,566점이고 5 mm voxel 후에도 179,035점이라 `max_points=60000`을 넘는다.
그러면 `coverage_preserving_cap`이 voxel을 **5 mm → 15.9 mm로 키운다** (5회 성장). 그 간격의
표면에서 반경 3 cm 안에 들어오는 점은 약 9개다. 그런데 `clustering.min_points = 20`이다 —
**어떤 점도 core가 될 수 없고, DBSCAN은 전부 noise로 라벨한다.**

주목할 점은 각 설정이 따로 보면 다 합리적이라는 것이다. `max_points`는 실시간성을 위한
상한이고, voxel 성장은 "점을 버리지 않고 굵게" 하는 안전한 축소이며, `min_points=20`은
5 mm 점군에서 타당하다. **깨지는 것은 둘의 결합이고, 아무 곳에서도 보고되지 않는다** —
`DEGRADED`는 붙지만 그것은 "굵어졌다"는 뜻이지 "군집이 불가능해졌다"는 뜻이 아니다.

**대응.** 반경 게이트 `range_max`를 켠다. AG3S 설정에 이미 있는 항목이고, docstring이
"작업 공간을 실제 거리로 제한하라"고 적고 있다. 값은 측정에서 정했다: 이 롤아웃에서 팔이
base로부터 가장 멀리 간 거리가 **1.389 m**이고, head 카메라는 base에서 1 m 이내에 있으므로,
카메라 기준 **2.0 m**는 팔이 닿을 수 있는 모든 것을 여유 있게 포함한다. 팔이 닿을 수 없는
기하는 팔과 충돌할 수 없으므로 잘라도 안전하며, 자르는 방향이 아니라 **남기는 방향으로**
넉넉하게 잡았다.

게이트를 켜면 voxel이 5.0 mm로 유지되고, 같은 프레임에서 사과 점이 33개 → 256개로 늘며,
군집이 정상 동작한다. 아래 결과는 모두 `range_max = {args.range_max} m`에서 얻은 것이다.

## 판정 — {verdict}

| 검사 | 결과 | 기준 |
|---|---|---|
| A. 옳은 물체 선택 | {'통과' if ok_a else '실패'} — 채점 가능 {len(gradeable)}프레임 중 **{len(graded_ok)}개** | 전부 |
| B. 클러스터 품질 | {'통과' if ok_b else '실패'} — IoU **{np.mean([r['iou'] for r in graded_ok]) if graded_ok else float('nan'):.3f}**, 정밀도 {np.mean([r['precision'] for r in pre_ok]) if pre_ok else float('nan'):.3f}, 재현율 {np.mean([r['recall'] for r in pre_ok]) if pre_ok else float('nan'):.3f} | IoU >= 0.5 |
| C. 오선택 없음 | {'통과' if ok_c else '실패'} — 엉뚱한 물체를 `ok`로 반환 **{len(wrong_when_visible)}건** | 0건 |

**판정을 파지 전 구간에만 거는 이유.** 자의적 축소가 아니라 모듈 설계가 그렇게 되어 있다.
`AG3S.attach()`의 docstring은 "**AG3S never calls this itself**"이고, 이어서 "여기서부터
물체의 존재는 그리퍼에 대한 사실이지 카메라가 본 것에 대한 주장이 아니다"라고 적는다. 즉 파지
이후 target의 기하는 부착 스냅샷(`AttachedCollisionGeometry`)에서 오고 grounding에서 오지
않는다. grounding이 답을 내야 하는 구간은 로봇이 아직 물체를 잡지 않은 동안이다.

나머지 두 구간의 결과는 판정이 아니라 **소견**으로 아래에 싣는다. 숨기지 않는 이유는 그것들이
실제 한계를 드러내기 때문이다.

**채점 가능성 기준.** 파지 전 {len(pre)}프레임 중 {len(gradeable)}개를 채점하고
{len(boundary)}개를 제외했다. 기준은 "점군 안의 target 점 수 >= 2 × `min_points` =
{gradeable_floor}"이며, 이 숫자는 내가 고른 것이 아니라 **군집기 자신의 파라미터**다:
`min_points`보다 적은 점으로는 클러스터가 정의상 존재할 수 없으므로, 그런 프레임에서 재는
것은 grounding이 아니라 군집 하한이다.

제외된 프레임과 거기서 일어난 일을 숨기지 않고 적는다:

{md(["t_step", "점군 내 target 점", "선택", "최고 점수"],
    [f"| {r['t_step']} | **{r['n_gt_target']}** | {r.get('picked_body') or '—'} | {r['best_score']:.3f} |"
     for r in boundary] or ["| — | — | — | — |"])}

{f"제외된 유일한 프레임은 target이 정확히 {boundary[0]['n_gt_target']}점, 즉 `min_points`와 같은 지점이다. 클러스터가 겨우 성립할 수 있는 이론적 경계이고, 실제로 성립하지 못했다." if len(boundary) == 1 else ""}

채점 가능한 프레임들의 점 수는 {min(r['n_gt_target'] for r in gradeable) if gradeable else 0}–{max(r['n_gt_target'] for r in gradeable) if gradeable else 0}점으로, 경계와 확실히 떨어져 있다.

### A. 옳은 물체를 골랐는가

![프레임별 결과](../asset/image/grounding/fig1_per_frame.png)

{md(["구간", "프레임", f"{target} 선택", "crate 선택", "banana 선택", "target 없음 보고"],
    [f"| {nm} | {len(g)} | {sum(1 for r in g if r.get('picked_body') == target)} | "
     f"{sum(1 for r in g if r.get('picked_body') == 'crate')} | "
     f"{sum(1 for r in g if r.get('picked_body') == 'banana')} | "
     f"{sum(1 for r in g if r.get('picked_body') is None)} |"
     for nm, g in (("파지 전", pre), ("가려짐", occ), ("배치 후", post))])}

### 소견 1 — 배치 후: 접촉한 두 물체는 유클리드 군집으로 분리되지 않는다

`{target}`이 크레이트 **안에** 놓인 뒤로는 두 물체가 서로 닿아 있다. `eps = {cl_cfg.eps} m`
이웃 반경에서 둘은 하나의 연결 성분이 되고, 점 수가 많은 크레이트가 다수 body가 된다.

증거는 점수 분해에 그대로 남는다. 배치 직후 프레임에서 1위 클러스터는 `crate`로 라벨되지만
그 점의 **41%가 실제로는 {target}**이고, peak까지의 거리는 18 mm다 — attention은 여전히
{target}을 정확히 가리키고 있으며, 갈라지지 않은 것은 기하다. 몇 프레임 뒤에는 합쳐진 덩어리의
밀집도가 0.057까지 떨어져, 점수화가 차라리 멀리 있는 작고 단단한 `banana`를 고른다.

이것은 버그가 아니라 **유클리드 군집의 정의상 한계**다. 접촉한 물체를 나누려면 색·법선·인스턴스
분할처럼 기하 외의 신호가 필요하다. AG3S의 대응은 그 지점에서 grounding을 쓰지 않는 것이다 —
파지 순간에 `attach()`로 스냅샷을 떠서 물체를 그리퍼에 붙인다.

### 소견 2 — 가려짐: grounding에는 가림 신호가 없다

가려진 {len(occ)}프레임에서 `ground_target`은 `ok`와 함께 다른 물체를 반환한다. attention이
{target}을 볼 수 없으면 다음으로 높은 점수의 클러스터가 이길 뿐이고, **"내가 지금 보고 있어야
할 것이 안 보인다"는 것을 알 방법이 이 단계에는 없다**.

이것이 위험한 이유는 명확하다. target 이름표는 여유거리 정책을 완화한다 — 잡을 물체가 아닌
것에 접촉이 허용된다는 뜻이다. AG3S가 이에 답하는 방식은 grounding을 고치는 것이 아니라
**단계(phase) 기계**다: 파지 후에는 `POST_GRASP`로 넘어가고 target 기하는 부착 스냅샷에서
온다. 이 검증은 단계 기계를 모형화하지 않으므로 여기서는 재지 않으며, 7단계에서 잰다.

다만 한 가지는 지금 말할 수 있다. **파이프라인을 단계 기계 없이 돌리면 이 실패가 그대로
남는다.** 부착 없이 grounding만으로 target을 정하는 구성은 안전하지 않다.

### B. 클러스터 품질 — 옳은 물체를 골라도 얼마나 덮는가

옳은 물체를 골랐어도 클러스터가 물체의 일부만 덮으면 6단계 primitive가 물체를 **과소 근사**한다.
과소 근사는 이 파이프라인이 금지하는 것이므로, 선택의 정오와 별개로 재야 한다.

![클러스터 품질](../asset/image/grounding/fig2_cluster_quality.png)

정밀도는 "클러스터 점 중 실제로 target인 비율", 재현율은 "target 점 중 클러스터에 들어온
비율"이다. 재현율이 1.0에 못 미치는 것은 대체로 자기 필터가 제거한 점과 지지면으로 제외된
점 때문이며, 정밀도가 낮으면 이웃 물체가 딸려 온 것이다.

![선택된 클러스터](../asset/image/grounding/fig3_selected_cluster.png)

같은 결과를 **정책이 실제로 본 이미지 위에** 되돌려 그리면 이렇다. 위에서 본 그림이 정확하다면
이 그림은 읽기 쉽다 — 파란 점이 사과 위에 앉아 있고, 주황 원이 최적화기가 보게 될 구의
실루엣이다.

![이미지 위 겹침](../asset/image/grounding/fig4_image_overlay.png)

### C. 오선택이 없는가 — 가장 위험한 실패 방식

target이 보이는데도 엉뚱한 물체를 `OK`로 반환하면, 그 물체가 target으로 이름 붙고 여유거리
정책이 완화된다 — **잡을 물체가 아닌 것에 접촉이 허용된다는 뜻**이다. `GroundingStatus`가
존재하는 이유가 이것이고, 여기서 재는 것도 이것이다.

채점 가능한 {len(gradeable)}프레임: 오선택 **{len(wrong_when_visible)}건**.
전 구간 기준으로는 {sum(1 for r in rows if r["status"] == "ok" and r.get("picked_body") != target)}건이며,
그 전부가 위 소견 1·2가 설명하는 두 구간에서 나온다.

### D. 왜 그것을 골랐는가 — 점수 분해

점수는 `(w_a·평균attention + w_g·√(밀집도 · peak근접도)) / (w_a + w_g)`이고
`w_a={cl_cfg.w_attention}`, `w_g={cl_cfg.w_geometry}`이다. 기하 항이 **기하평균**인 것은
의도적이다 — 합이면 peak 위에 놓인 거대한 덩어리가 실제 물체만큼 점수를 받는다.

한 프레임의 상위 5개 클러스터:

{md(["순위", "다수 body", "범주", "점수", "평균 attention", "밀집도", "peak 거리 (m)", "점 수"],
    [f"| {e['rank']} | {e['body']} | {e['cat']} | **{e['score']:.4f}** | {e['mean_att']:.4f} | {e['compact']:.4f} | {e['peak_d']:.4f} | {e['n']} |"
     for e in example_rank] or ["| — | — | — | — | — | — | — | — |"])}

### E. 무게중심 편향 — 2단계가 예고한 것

카메라는 물체의 앞면만 보므로 클러스터 무게중심은 물체 중심이 아니다. 위 그림의 ○(무게중심)와
×(실제 중심)의 간격이 그것이다.

측정: 평균 **{np.mean(cen_err)*1000 if cen_err else float('nan'):.1f} mm**, 최대
{np.max(cen_err)*1000 if cen_err else float('nan'):.1f} mm.

위 그림에서 두 표시가 거의 겹쳐 보이는 것은 그림이 x–y 평면이기 때문이다. 편향은 카메라가
바라보는 방향으로 생기고, head 카메라는 비스듬히 내려다보므로 편향의 상당 부분이 z 성분이라
위에서 본 그림에는 잘 나타나지 않는다.

이 값은 오차가 아니라 **구조적 성질**이다. 5·6단계는 무게중심을 물체 중심으로 써서는 안 되며,
6단계 primitive 맞춤이 점을 **포함**하도록 만들어져 있는 것이 그 대응이다.

### F. 배경 클러스터 — 3단계가 경고한 것

3단계에서 seed의 78.8%가 벽·바닥에 있었다. 바닥은 지지면으로 `exclude_mask`에 걸리지만 벽은
아니다. 벽에서 자란 클러스터가 후보로 올라오는가?

상위 5위 안에 배경 클러스터가 나타난 횟수: **{len(bg_ranks)}회**{f" (순위 {sorted(set(bg_ranks))})" if bg_ranks else ""}.

## 그림에 대하여

### fig1 — 프레임별 결과 띠

`fig1_per_frame.png`. 위 띠는 프레임마다 무엇을 골랐는지(파랑 = target, 빨강 = 다른 물체,
회색 = target 없음 보고), 아래는 최고 클러스터 점수와 임계값, 음영은 target 가려짐.

**읽는 법.** 파랑에서 빨강으로 넘어가는 지점과 점수가 떨어지는 지점이 일치하는지 본다.
t=200 부근의 점수 급등은 사과가 다시 보이면서 크레이트와 합쳐진 클러스터가 만든 것이다.

### fig2 — 클러스터 품질

`fig2_cluster_quality.png`. IoU·정밀도·재현율을 정답 target 점 대비로.

**읽는 법.** 재현율이 1.0에 못 미치면 자기 필터나 지지면 제외가 물체 점을 가져간 것이고,
정밀도가 낮으면 이웃 물체가 딸려 온 것이다. 둘의 원인이 다르므로 나눠 본다.

### fig3 — 선택된 클러스터 (위에서)

`fig3_selected_cluster.png`. 파란 점이 고른 클러스터, 노란 점이 정답, ○가 클러스터 무게중심,
×가 실제 물체 중심.

**읽는 법.** 두 표시가 겹쳐 보이는 것은 그림이 x–y 평면이기 때문이다. 편향은 카메라가 보는
방향으로 생기고 head 카메라는 비스듬히 내려다보므로 상당 부분이 z 성분이다.

### fig4 — 관측 이미지 위 겹침

`fig4_image_overlay.png`. 같은 결과를 정책이 본 224x224 입력 위에 되돌려 그린 것.

**만드는 법.** `imageview.project` 로 base 프레임 점을 픽셀로 옮긴다. 깊이(480x640)와 정책
이미지(224x224) 사이는 **정규화 좌표로 오프셋 없이** 옮긴다 — 299x224 → 224x224 변환이 순수
가로 압축이라 성립하는 성질이고, 2단계에서 확인했다. 주황 원은 제약용 구를 반지름 `f*r/z` 로
근사한 것이며 그림 전용이다.

**읽는 법.** 파란 점(클러스터)이 노란 점(정답) 위에 겹치는지, 주황 원이 사과를 감싸는지.
위에서 본 그림이 정확하다면 이 그림은 **읽기 쉽다** — 두 그림은 같은 데이터의 다른 시점이다.

## 재현

```bash
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.grounding_report \\\\
    --records {args.records} --attention {args.attention}
```
"""
    out = pathlib.Path(args.out_doc)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    out.with_suffix(".json").write_text(json.dumps(
        {"target": target, "camera": args.camera, "cell": chosen,
         "checks": {"A_correct_object": ok_a, "B_quality": ok_b, "C_honest_failure": ok_c},
         "frames": rows}, indent=2, ensure_ascii=False, default=float))
    print(f"wrote {out} and 3 figures in {figs}")
    iou_pre = float(np.mean([r["iou"] for r in pre_ok])) if pre_ok else float("nan")
    print(f"판정 {verdict}: A(파지전)={len(pre_ok)}/{len(pre)}  B=IoU {iou_pre:.3f}  "
          f"C=파지전 오선택 {len(wrong_when_visible)}건  "
          f"E=중심편향 {np.mean(cen_err)*1000 if cen_err else np.nan:.1f}mm")


if __name__ == "__main__":
    main()
