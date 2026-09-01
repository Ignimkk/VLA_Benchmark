"""기록된 점군을 여러 각도에서 보여주는 시각화 모음.

검증 문서의 그림은 각각 하나의 질문에 답하도록 최소한으로 그린다. 이 모듈은 그것과 목적이
다르다 — 파이프라인이 무엇을 보고 무엇을 만들어 내는지를 **눈으로 이해하기 위한** 것이고,
발표와 디버깅에 쓴다.

기록에 depth가 있으면 그것을 쓰고, 없으면 `qpos`로 씬을 재생해 렌더한다. 어느 쪽을 썼는지는
그림 제목에 남긴다 — 기록된 depth는 uint16 밀리미터라 실제 센서와 같은 1 mm 양자화를 갖는
반면, 재생 depth는 렌더러의 float64라 그 양자화가 없다. 둘을 섞어 보고하면 어느 정확도를
말하는지 알 수 없게 된다.

만드는 그림:

* `g1_per_camera.png` — 카메라 3대가 각각 무엇을 보는가. 정답 물체별로 색.
* `g2_fusion.png` — 세 점군을 겹쳐 놓고 **출처 카메라로 색칠**. 외부 파라미터가 서로 맞는지
  숫자가 아니라 그림으로 보이는 곳이다 — 어긋나면 같은 표면이 색깔별로 갈라져 두꺼워진다.
* `g3_attention_2d.png` — RGB · depth · attention을 나란히. 2D attention이 어떤 깊이 위에
  얹히는지, 즉 3D로 올라가기 직전의 상태.
* `g4_attention_3d.png` — 점군을 attention으로 색칠. **낮은 attention 점도 그린다** — 아무것도
  버리지 않았다는 것이 이 파이프라인의 핵심 불변식이고, 그 증거가 이 그림이다.
* `g5_target.png` — 고른 클러스터, 맞춘 primitive, 정답 표면을 함께.
* `g6_stages.png` — 원본 → 자기 필터 → 지지면 제외 → seed → target 의 단계 띠.

실행:
    MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.cloud_gallery \
        --records outputs/.../ag3s_records/run_0002 \
        --attention benchmark/ag3s/asset/data/attention_step1_run0002.npz --frame 0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import numpy as np

from benchmark.ag3s.attention_lifting import GridAttentionAdapter, lift
from benchmark.ag3s.config import (
    AttentionConfig, ClusteringConfig, GeometryConfig, PointCloudConfig, SupportSurfaceConfig,
)
from benchmark.ag3s.experiments.figstyle import (
    CATEGORICAL, GRID_INK, INK, INK_2, SURFACE, sequential_cmap, use_korean,
)
from benchmark.ag3s.experiments.policy_record import (
    CAMERA_BINDINGS, load_run, pose_scene, replay_scene,
)
from benchmark.ag3s.reconstruction import backproject, reconstruct
from benchmark.ag3s.robot_filter import filter_robot_points
from benchmark.ag3s.support_surface import fit_support_surfaces
from benchmark.ag3s.target_grounding import extract_seeds, ground_target
from benchmark.ag3s.visualization import _equalise, _thin, _wire_sphere

OBJECT_BODIES = ("crate", "apple", "banana", "orange", "pear")
DEFAULT_CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def crop(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """상자 안의 점만. 3D 산점도는 축척이 넓어지면 전부 뭉개지므로, 솎아내기 전에 자른다."""
    if points.size == 0:
        return points
    m = np.all((points >= box[0]) & (points <= box[1]), axis=1)
    return points[m]


def workspace_box(object_points: np.ndarray, margin: float) -> np.ndarray:
    """물체가 차지하는 범위에 여백을 둔 상자. 방 전체를 그리면 아무것도 안 보인다."""
    if object_points.size == 0:
        return np.array([[-1e9] * 3, [1e9] * 3])
    return np.stack([object_points.min(axis=0) - margin, object_points.max(axis=0) + margin])


def fit_box(ax, box: np.ndarray) -> None:
    """상자에 축을 맞추되 **비율은 실제 그대로** 유지한다.

    `visualization._equalise`는 세 축을 같은 길이의 정육면체로 만든다. 4 cm 물체가 1.5 m 축
    위에서 팬케이크로 보이는 것을 막기 위한 것이고, 단일 물체를 볼 때는 옳다. 그런데 테이블
    장면은 x·y가 1 m대인데 z는 0.3 m라, 정육면체로 맞추면 세로의 2/3가 빈 공간이 되고 내용이
    작아진다. `set_box_aspect`에 실제 변 길이를 주면 왜곡 없이 프레임을 채운다.
    """
    lo, hi = box
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-6)))


def _ax3d(fig, spec, title):
    ax = fig.add_subplot(*spec, projection="3d")
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9.5, color=INK, pad=2)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(SURFACE)
        pane.pane.set_edgecolor(GRID_INK)
        pane._axinfo["grid"]["color"] = GRID_INK
    ax.tick_params(labelsize=6, colors=INK_2)
    ax.set_xlabel("x", fontsize=7, color=INK_2)
    ax.set_ylabel("y", fontsize=7, color=INK_2)
    ax.set_zlabel("z", fontsize=7, color=INK_2)
    return ax


def capture(scene, step, camera: str, prefer_recorded: bool):
    """(depth, K, T_base_cam, body_ids, 출처) — 기록된 depth를 우선한다."""
    frame = scene.capture(camera)          # 정답 세그멘테이션은 언제나 재생에서 온다
    if prefer_recorded and camera in step.depth:
        return (step.depth[camera], step.camera_intrinsics[camera], step.T_base_cam[camera],
                frame.body_ids, "기록된 depth (uint16 mm)")
    return frame.depth, frame.camera_intrinsics, frame.T_base_cam, frame.body_ids, "재생 depth (float64)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--attention", default=None)
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/step-01-attention.json")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--target", default=None)
    ap.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--replay-depth", action="store_true",
                    help="기록된 depth가 있어도 무시하고 재생 렌더를 쓴다 (대조용)")
    ap.add_argument("--out", default="benchmark/ag3s/asset/image/gallery")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    use_korean()
    import matplotlib.pyplot as plt
    import mujoco

    from benchmark.ag3s.experiments.attention_report import target_from_prompt
    from benchmark.ag3s.experiments.grounding_report import build_robot_model

    run = load_run(args.records)
    step = run.steps[args.frame]
    target = args.target or target_from_prompt(run.prompt)
    scene = replay_scene(run)
    robot_model = build_robot_model(scene)
    pose_scene(scene, step)
    names = {i: (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or "?")
             for i in range(scene.model.nbody)}
    bid = {n: mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in OBJECT_BODIES}

    pc_cfg = dataclasses.replace(PointCloudConfig(), range_max=(args.range_max or None))
    prefer = step.has_depth and not args.replay_depth
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    clouds, sources = {}, {}
    for camera in args.cameras:
        depth, K, T, seg, src = capture(scene, step, camera, prefer)
        cloud = backproject(depth, K, T, pc_cfg)
        clouds[camera] = (cloud, seg[cloud.uv[:, 1], cloud.uv[:, 0]] if len(cloud) else
                          np.zeros(0, np.int32))
        sources[camera] = src
    src_note = sources[args.cameras[0]]
    print(f"프레임 {args.frame} (t_step {step.t_step}) | depth 출처: {src_note}")

    # ---------------------------------------------------------------- g1 카메라별
    obj_all = np.concatenate([clouds[c][0].points[np.isin(clouds[c][1], list(bid.values()))]
                              for c in args.cameras] or [np.zeros((0, 3))])
    box = workspace_box(obj_all, 0.25)

    fig = plt.figure(figsize=(4.6 * len(args.cameras), 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for i, camera in enumerate(args.cameras):
        cloud, labs = clouds[camera]
        ax = _ax3d(fig, (1, len(args.cameras), i + 1), f"{camera}  (상자 안 표시)")
        rest = ~np.isin(labs, list(bid.values()))
        p = _thin(crop(cloud.points[rest], box), 14000)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=.7, c=GRID_INK, depthshade=False, linewidths=0)
        for slot, (name, b) in enumerate(bid.items()):
            m = labs == b
            if m.any():
                q = cloud.points[m]
                ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=6.0, c=CATEGORICAL[slot],
                           depthshade=False, linewidths=0, label=name if i == 0 else None)
        fit_box(ax, box)
        ax.view_init(elev=24, azim=-62)
    h, l = fig.axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, frameon=False, fontsize=8.5, ncol=len(bid), markerscale=3,
                     loc="lower center", bbox_to_anchor=(0.5, -0.01))
    for t in leg.get_texts(): t.set_color(INK_2)
    fig.suptitle(f"카메라 3대가 각각 무엇을 보는가 — 프레임 {args.frame}, {src_note}",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out / "g1_per_camera.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); written.append("g1_per_camera.png")

    # ---------------------------------------------------------------- g2 융합
    fig = plt.figure(figsize=(11.0, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for j, (title, zoom) in enumerate((("작업 영역 전체", False), (f"{target} 주변 확대", True))):
        ax = _ax3d(fig, (1, 2, j + 1), title)
        for slot, camera in enumerate(args.cameras):
            cloud, labs = clouds[camera]
            tgt_pts = np.concatenate([clouds[c][0].points[clouds[c][1] == bid[target]]
                                      for c in args.cameras] or [np.zeros((0, 3))])
            zbox = workspace_box(tgt_pts, 0.05) if zoom else box
            p = _thin(crop(cloud.points, zbox), 9000)
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=6.0 if zoom else 1.0,
                       c=CATEGORICAL[slot], depthshade=False, linewidths=0,
                       label=camera if j == 0 else None, alpha=.8)
        fit_box(ax, zbox)
        ax.view_init(elev=22, azim=-58)
    h, l = fig.axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, frameon=False, fontsize=9, ncol=3, markerscale=6,
                     loc="lower center", bbox_to_anchor=(0.5, 0.0))
    for t in leg.get_texts(): t.set_color(INK_2)
    fig.suptitle("세 카메라를 겹쳐 출처별로 색칠 — 색이 갈라지지 않으면 외부 파라미터가 맞는 것",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out / "g2_fusion.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig); written.append("g2_fusion.png")

    # ------------------------------------------------------ attention 이 있으면 g3~g6
    if args.attention:
        chosen = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
        blob = np.load(args.attention, allow_pickle=False)
        A = np.asarray(blob["attention"], np.float32)
        cams = [str(c) for c in blob["cameras"]]
        dq = [int(x) for x in blob["denoise_steps"]].index(chosen["denoise"])
        aq = [str(x) for x in blob["aggregations"]].index(chosen["agg"])
        pol = "cam_high"
        ciq = cams.index(pol)
        grid = A[args.frame, dq, aq, chosen["layer"], chosen["head"], ciq]
        mj_cam = CAMERA_BINDINGS[pol][0]
        depth, K, T, seg, _ = capture(scene, step, mj_cam, prefer)
        cmap = sequential_cmap()

        # g3 — RGB · depth · attention
        fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.3), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        for ax in axes:
            ax.set_facecolor(SURFACE); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_color(GRID_INK)
        rgb = step.images[pol]
        axes[0].imshow(rgb); axes[0].set_title("정책이 본 RGB (224×224)", fontsize=9.5, color=INK)
        dm = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
        im = axes[1].imshow(dm, cmap="magma_r")
        axes[1].set_title(f"depth (m) — {src_note}", fontsize=9.5, color=INK)
        cb = fig.colorbar(im, ax=axes[1], fraction=.046, pad=.02)
        cb.ax.tick_params(colors=INK_2, labelsize=7); cb.outline.set_edgecolor(GRID_INK)
        from benchmark.ag3s.attention_lifting import _resample
        # 파이프라인이 실제로 쓰는 이중선형 보간으로 확대한다. 정수 블록 복제로 그리면
        # 실제로 점에 붙는 값이 아니라 격자 무늬를 보여주게 된다.
        att_full = _resample(grid, depth.shape, "bilinear")
        axes[2].imshow(dm, cmap="Greys_r")
        axes[2].imshow(att_full / max(att_full.max(), 1e-9), cmap=cmap, alpha=.62)
        axes[2].set_title(f"attention (L{chosen['layer']}h{chosen['head']}) 를 depth 위에",
                          fontsize=9.5, color=INK)
        axes[3].imshow(rgb)
        axes[3].imshow(grid / max(grid.max(), 1e-9), cmap=cmap, alpha=.55,
                       extent=(0, rgb.shape[1], rgb.shape[0], 0), interpolation="bilinear")
        r, c = np.unravel_index(grid.argmax(), grid.shape)
        axes[3].plot((c + .5) / grid.shape[1] * rgb.shape[1],
                     (r + .5) / grid.shape[0] * rgb.shape[0], marker="o", ms=9,
                     mfc="none", mew=2.0, color=CATEGORICAL[1])
        axes[3].set_title("같은 attention 을 RGB 위에 (○ = argmax)", fontsize=9.5, color=INK)
        fig.suptitle("2D attention 이 3D 로 올라가기 직전 — 어떤 깊이 위에 얹히는가",
                     color=INK, fontsize=12, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out / "g3_attention_2d.png", facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig); written.append("g3_attention_2d.png")

        # 파이프라인 실행 (g4~g6 공통)
        raw, _ = reconstruct(depth=depth, camera_intrinsics=K, T_base_cam=T, config=pc_cfg)
        work, _ = filter_robot_points(raw, robot_model, scene.robot_state(), pc_cfg)
        surfaces, smask = fit_support_surfaces(work, SupportSurfaceConfig())
        acloud = lift(work, grid, AttentionConfig(), adapter=GridAttentionAdapter(),
                      image_hw=depth.shape)
        cl_cfg = ClusteringConfig()
        seeds = extract_seeds(acloud.attention, cl_cfg, AttentionConfig().seed_percentile)
        result = ground_target(acloud, cl_cfg, seed_percentile=AttentionConfig().seed_percentile,
                               exclude_mask=smask)
        labs_w = seg[work.uv[:, 1], work.uv[:, 0]]

        # g4 — attention 3D
        fig = plt.figure(figsize=(11.0, 5.2), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        for j, (title, zoom) in enumerate((("전체", False), (f"{target} 주변 확대", True))):
            ax = _ax3d(fig, (1, 2, j + 1), title)
            tp = work.points[labs_w == bid[target]]
            zbox = workspace_box(tp, 0.08) if (zoom and len(tp)) else box
            inside = np.all((work.points >= zbox[0]) & (work.points <= zbox[1]), axis=1)
            ii = np.nonzero(inside)[0]
            ii = ii[np.unique(np.linspace(0, len(ii) - 1, min(len(ii), 22000)).astype(np.int64))]
            sc3 = ax.scatter(work.points[ii, 0], work.points[ii, 1], work.points[ii, 2],
                             s=6.0 if zoom else 1.2, c=acloud.attention[ii], cmap=cmap,
                             vmin=0, vmax=1, depthshade=False, linewidths=0)
            fit_box(ax, zbox)
            ax.view_init(elev=22, azim=-58)
        cb = fig.colorbar(sc3, ax=fig.axes, fraction=.02, pad=.02)
        cb.set_label("정규화된 attention", color=INK_2, fontsize=8)
        cb.ax.tick_params(colors=INK_2, labelsize=7); cb.outline.set_edgecolor(GRID_INK)
        fig.suptitle("점군을 attention 으로 색칠 — 낮은 점도 전부 그린다 (아무것도 버리지 않았다는 증거)",
                     color=INK, fontsize=12, x=0.01, ha="left")
        fig.savefig(out / "g4_attention_3d.png", facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig); written.append("g4_attention_3d.png")

        # g5 — target 표현
        if result.target is not None:
            from benchmark.ag3s.geometry import to_spheres
            tgt = result.target
            fig = plt.figure(figsize=(11.0, 5.2), dpi=150)
            fig.patch.set_facecolor(SURFACE)
            gt = work.points[labs_w == bid[target]]
            for j, (title, show_prim) in enumerate(((f"고른 클러스터 vs 정답 {target} 점", False),
                                                    ("맞춘 primitive 와 제약용 구", True))):
                ax = _ax3d(fig, (1, 2, j + 1), title)
                near = np.linalg.norm(work.points - tgt.centroid, axis=1) < 0.18
                p = work.points[near & ~np.isin(np.arange(len(work)), tgt.point_indices)]
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=.8, c=GRID_INK, depthshade=False,
                           linewidths=0, label="주변 점" if j == 0 else None)
                if len(gt):
                    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=14, c=CATEGORICAL[3],
                               depthshade=False, linewidths=0,
                               label=f"정답 {target}" if j == 0 else None)
                ax.scatter(tgt.points[:, 0], tgt.points[:, 1], tgt.points[:, 2], s=5,
                           c=CATEGORICAL[0], depthshade=False, linewidths=0,
                           label=f"클러스터 ({len(tgt.points)}점)" if j == 0 else None)
                if show_prim:
                    for centre, radius in to_spheres(tgt.bounding_geometry):
                        _wire_sphere(ax, centre, radius, CATEGORICAL[7], alpha=.45)
                fit_box(ax, workspace_box(np.concatenate([tgt.points, gt])
                                          if len(gt) else tgt.points, 0.02))
                ax.view_init(elev=20, azim=-58)
            h, l = fig.axes[0].get_legend_handles_labels()
            leg = fig.legend(h, l, frameon=False, fontsize=9, ncol=3, markerscale=3,
                             loc="lower center", bbox_to_anchor=(0.5, 0.0))
            for t in leg.get_texts(): t.set_color(INK_2)
            prim = tgt.bounding_geometry
            fig.suptitle(f"target 표현 — {prim.type.value}, 신뢰도 {tgt.confidence:.3f}, "
                         f"제약용 구 {len(to_spheres(prim))}개",
                         color=INK, fontsize=12, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0.06, 1, 0.93))
            fig.savefig(out / "g5_target.png", facecolor=SURFACE, bbox_inches="tight")
            plt.close(fig); written.append("g5_target.png")

        # g6 — 단계 띠
        stages = [("① 원본 재구성", raw.points, GRID_INK, None),
                  ("② 자기 필터 후", work.points, GRID_INK, None),
                  ("③ 지지면 제외", work.points[~smask], CATEGORICAL[2], None),
                  ("④ seed (상위 5%)", work.points[seeds], CATEGORICAL[3], work.points),
                  ("⑤ 고른 target", (result.target.points if result.target is not None
                                    else np.zeros((0, 3))), CATEGORICAL[0], work.points)]
        fig = plt.figure(figsize=(4.0 * len(stages), 4.2), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        for i, (title, pts, colour, backdrop) in enumerate(stages):
            ax = _ax3d(fig, (1, len(stages), i + 1), f"{title}\n{len(pts):,}점")
            if backdrop is not None:
                b = _thin(crop(backdrop, box), 9000)
                ax.scatter(b[:, 0], b[:, 1], b[:, 2], s=.5, c="#eceae5",
                           depthshade=False, linewidths=0)
            q = _thin(crop(pts, box), 14000)
            if len(q):
                ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=2.2, c=colour, depthshade=False,
                           linewidths=0)
            fit_box(ax, box)
            ax.view_init(elev=22, azim=-58)
        fig.suptitle("파이프라인 단계별 점군 — 무엇이 언제 걸러지는가 (점 수는 전체, 그림은 작업 상자 안만)",
                     color=INK, fontsize=12, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out / "g6_stages.png", facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig); written.append("g6_stages.png")

    scene.close()
    print(f"wrote {len(written)} figures in {out}: {', '.join(written)}")


if __name__ == "__main__":
    main()
