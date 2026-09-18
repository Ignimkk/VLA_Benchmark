"""문서용 시각화 — attention lifting 과 target grounding 을 **3D 로** 보인다.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.doc_attention_stages --records run_0004 \\
        --attention attention_step1_run0004.npz --frame 10

기존 `step5-lifting.png` 은 올린 결과를 **위에서 내려다본 2D 격자**로 보였다. 여기서는 같은
프레임의 **실제 3D 점구름을 attention 으로 칠한 3인칭 뷰**를 만든다 — 사용자가 요청한
"실제 3D point cloud map 에 투영" 이 그것이다.

규칙 A: 실제 씬 · 3인칭 · 그래프 · 표를 한 장씩에 모두 넣는다. 새 판정은 하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
ATT_CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
LABEL = ("head (zed_left)", "왼손목", "오른손목")
ELEV, AZIM = 22, -118


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


def _box(ax, x=(0.10, 1.00), y=(-0.35, 0.50), z=(0.55, 1.15)):
    ax.set_xlim(*x); ax.set_ylim(*y); ax.set_zlim(*z)
    ax.set_xlabel("x [m] 앞", fontsize=8); ax.set_ylabel("y [m] 왼", fontsize=8)
    ax.set_zlabel("z [m] 위", fontsize=8)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.tick_params(labelsize=7)


def _inbox(p, x=(0.10, 1.00), y=(-0.35, 0.50), z=(0.55, 1.15)):
    return ((p[:, 0] > x[0]) & (p[:, 0] < x[1]) & (p[:, 1] > y[0]) & (p[:, 1] < y[1])
            & (p[:, 2] > z[0]) & (p[:, 2] < z[1]))


def _table(ax, rows, fs, title, widths=(0.34, 0.66), fontsize=8.4):
    ax.axis("off")
    t = ax.table(cellText=rows, colWidths=list(widths), loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(fontsize); t.scale(1, 1.34)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if rows[r][1] == "" and rows[r][0]:
            cell.set_facecolor("#ecebe7"); cell.set_text_props(weight="bold")
        elif not rows[r][0]:
            cell.set_edgecolor("none"); cell.set_facecolor(fs.SURFACE)
    ax.set_title(title, fontsize=11, y=1.01)


def main() -> None:
    fs = _style()
    import matplotlib.pyplot as plt

    from benchmark.ag3s.attention_lifting import GridAttentionAdapter, lift
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.multiview import fuse_observations
    from benchmark.ag3s.support_surface import fit_support_surfaces
    from benchmark.ag3s.target_grounding import ground_target
    from benchmark.ag3s.reconstruction import backproject
    from benchmark.ag3s.experiments.grounding_report import build_robot_model
    from benchmark.ag3s.experiments.mujoco_source import camera_observation
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--attention", default="attention_step1_run0004.npz")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/step-01-attention.json")
    ap.add_argument("--frame", type=int, default=10)
    args = ap.parse_args()

    blob = np.load(args.attention, allow_pickle=False)
    cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    grids = {c: np.asarray(blob["attention"][args.frame, di, ai, cell["layer"], cell["head"],
                                             [str(x) for x in blob["cameras"]].index(c)],
                           np.float32) for c in ATT_CAMS}

    run = load_run(args.records, limit=args.frame + 1)
    scene = replay_scene(run)
    robot = build_robot_model(scene)
    pose_scene(scene, run.steps[args.frame])
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "pointcloud": {"range_max": 2.0}})

    adapter = GridAttentionAdapter.from_config(cfg.attention)
    frames, clouds, lifted = {}, {}, {}
    for cam, ac in zip(CAMS, ATT_CAMS):
        f = scene.capture(cam)
        frames[cam] = f
        c = backproject(f.depth, f.camera_intrinsics, f.T_base_cam, cfg.pointcloud)
        clouds[cam] = c
        lifted[cam] = lift(c, grids[ac], cfg.attention, adapter=adapter, image_hw=f.hw)

    # 실제 서빙 경로와 같은 다중 카메라 전단 — 자기 필터까지 포함한다.
    obs = [camera_observation(scene, cam, robot, timestamp=float(args.frame),
                              attention_map=grids[ac])[0]
           for cam, ac in zip(CAMS, ATT_CAMS)]
    fusion = fuse_observations(obs, cfg, robot_model=robot)
    fused_cloud = fusion.pointcloud
    fused_att = np.asarray(fusion.attention, np.float32)

    surfaces, support_mask = fit_support_surfaces(fused_cloud, cfg.support_surface)
    from benchmark.ag3s.types import AttentionPointCloud
    apc = AttentionPointCloud(fused_cloud, fused_att,
                              np.asarray(fusion.raw_attention, np.float32))
    gr = ground_target(apc, cfg.clustering,
                       seed_percentile=cfg.attention.seed_percentile,
                       seed_threshold=cfg.attention.seed_threshold,
                       exclude_mask=support_mask)
    scene.close()

    rng = np.random.default_rng(0)
    fp = fused_cloud.points
    keep = _inbox(fp)

    # =============================================================== 그림 1 — lifting
    fig = plt.figure(figsize=(20.5, 11.8))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.26, top=0.87, bottom=0.05)

    # (a) VLA 원본 격자 3 대 — 한 축에 나란히
    ax = fig.add_subplot(gs[0, 0])
    gap = np.full((16, 2), np.nan)
    tiles, xt = [], []
    for i, ac in enumerate(ATT_CAMS):
        g = np.asarray(adapter.to_grid(grids[ac]), np.float64)
        g = g / max(g.max(), 1e-12)
        xt.append(sum(t.shape[1] for t in tiles) + 8)
        tiles += [g, gap]
    im = ax.imshow(np.hstack(tiles[:-1]), cmap="inferno", vmin=0, vmax=1)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{LABEL[i]}\n원시 최대 {grids[ac].max():.4f}"
                        for i, ac in enumerate(ATT_CAMS)], fontsize=8.5)
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.022, label="각 카메라 최대값으로 나눈 값")
    ax.set_title("(a) 입력 — VLA 가 준 2D attention map, 카메라마다 16x16 한 장\n"
                 f"layer {cell['layer']} head {cell['head']}, agg={cell['agg']}, "
                 f"denoise={cell['denoise']}\n"
                 "손목 카메라는 흩어져 있고 head 만 한 곳을 본다", fontsize=11)

    # (b) 실제 씬 — head depth 위에 확대된 attention
    ax = fig.add_subplot(gs[0, 1])
    head = frames["zed_left"]
    d = np.asarray(head.depth, float)
    ax.imshow(np.where(d > 2.5, np.nan, d), cmap="gray")
    pm = adapter.to_pixel_map(grids["cam_high"], head.hw)
    im = ax.imshow(pm, cmap="inferno", alpha=0.55)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.035, label="attention (원시)")
    ax.set_title("(b) 실제 씬에 투영 — head depth 위에 겹친 attention\n"
                 f"16x16 을 {head.hw[0]}x{head.hw[1]} 로 확대 ({cfg.attention.interpolation})\n"
                 "여기까지는 아직 2D 다 — 깊이가 없다", fontsize=11)

    # (c) head 점구름을 attention 으로 칠한 3인칭 뷰
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    L = lifted["zed_left"]
    p, v = L.points, np.asarray(L.attention, np.float32)
    k = _inbox(p)
    p, v = p[k], v[k]
    idx = rng.choice(len(p), size=min(26000, len(p)), replace=False)
    order = np.argsort(v[idx])          # 높은 값이 위에 오도록
    idx = idx[order]
    sc = ax.scatter(p[idx, 0], p[idx, 1], p[idx, 2], s=1.1, c=v[idx], cmap="inferno",
                    vmin=0, vmax=1, linewidths=0)
    _box(ax)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.10, label="정규화 attention")
    ax.set_title("(c) 실제 3D point cloud 에 투영 — head 카메라 한 대\n"
                 f"점 {len(L):,} 개, 하나도 버리지 않는다\n"
                 "uv 대응이 있어 2D 값이 3D 점으로 그대로 옮겨진다", fontsize=11)

    # (d) 3 대 융합 후 3D
    ax = fig.add_subplot(gs[1, 0], projection="3d")
    p, v = fp[keep], fused_att[keep]
    idx = rng.choice(len(p), size=min(26000, len(p)), replace=False)
    idx = idx[np.argsort(v[idx])]
    sc = ax.scatter(p[idx, 0], p[idx, 1], p[idx, 2], s=1.4, c=v[idx], cmap="inferno",
                    vmin=0, vmax=1, linewidths=0)
    _box(ax)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.10, label="정규화 attention")
    ax.set_title(f"(d) 카메라 3 대를 융합한 뒤 — 복셀 {cfg.timing.fusion_voxel_size*1000:.0f} mm 격자\n"
                 f"{fusion.metrics['n_points_before_fusion']:,} -> {len(fused_cloud):,} 점\n"
                 "같은 칸의 원시 attention 을 max 로 합친 뒤 정규화한다 (F2 수정)", fontsize=11)

    # (e) 그래프 — 카메라별 눈금이 다르다 + 융합본
    ax = fig.add_subplot(gs[1, 1])
    for i, cam in enumerate(CAMS):
        r = np.asarray(lifted[cam].raw_attention, np.float64)
        ax.hist(r, bins=180, histtype="step", lw=1.6, log=True,
                color=fs.CATEGORICAL[i],
                label=f"{LABEL[i]}  p99 {np.percentile(r, 99):.5f}")
    ax.set_xlabel("원시 attention 값 (정규화 전)"); ax.set_ylabel("점 개수 (log)")
    ax.legend(fontsize=8.5)
    ax.set_title("(e) 왜 융합 뒤에 정규화하는가 — 카메라마다 눈금이 다르다\n"
                 "카메라별로 먼저 0~1 로 펴면 max 융합이 '가장 후하게 스케일된 카메라' 를 고른다\n"
                 "F2(정규화가 카메라별이었다) 가 이것이고, 지금은 융합 뒤로 옮겼다", fontsize=11)

    # (f) 표
    rows = [
        ("무엇을 받는가", ""),
        ("입력 1", "PointCloud — uv 를 보존한 채로 온다"),
        ("입력 2", f"VLA attention — 16x16 격자 x 카메라 {len(CAMS)} 대"),
        ("입력 3", f"image_hw = {head.hw} — 어디까지 늘릴지 (F9)"),
        ("", ""),
        ("무엇을 내놓는가", ""),
        ("출력", "AttentionPointCloud — 점마다 값 2 벌"),
        ("값 2 벌인 이유", "정규화본 = 문턱용(동점 다수) · 원시본 = argmax 용"),
        ("버리는 점", "없음 — attention 이 낮다고 기하를 지우지 않는다"),
        ("", ""),
        ("이 프레임 실측", ""),
        ("카메라별 점", " · ".join(f"{LABEL[i]} {len(lifted[c]):,}"
                                 for i, c in enumerate(CAMS))),
        ("융합 후 점", f"{len(fused_cloud):,}  (압축 {fusion.metrics['fusion_compression']:.3f})"),
        ("자기 필터 제거", f"{fusion.metrics['n_self_filtered']:,} 점"),
        ("융합 attention 최대", f"{fused_att.max():.3f}   (0 인 점 {100.0*(fused_att<=0).mean():.1f} %)"),
    ]
    _table(fig.add_subplot(gs[1, 2]), rows, fs, "(f) 입력 · 출력 · 실측")

    fig.suptitle(f"attention lifting — 2D attention 을 3D 점으로 올린다.  "
                 f"{args.records} 프레임 {args.frame}, 실측 attention",
                 fontsize=14, y=0.955)
    fig.patch.set_facecolor(fs.SURFACE)
    out1 = OUT / "doc-stage-attention-lifting.png"
    fig.savefig(out1, dpi=118, bbox_inches="tight", facecolor=fs.SURFACE)
    plt.close(fig)
    print(f"wrote {out1}")

    # ============================================================= 그림 2 — grounding
    fig = plt.figure(figsize=(20.5, 11.8))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.26, top=0.87, bottom=0.05)

    seeds = np.asarray(gr.seed_indices, np.int64)
    seed_mask = np.zeros(len(fused_cloud), bool)
    seed_mask[seeds] = True

    # (a) 씨앗 — 어디서부터 물체를 찾기 시작하는가
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    rest = keep & ~seed_mask & ~support_mask
    r = fp[rest][rng.choice(int(rest.sum()), size=min(16000, int(rest.sum())), replace=False)]
    ax.scatter(r[:, 0], r[:, 1], r[:, 2], s=0.7, c="#c9c8c4", linewidths=0, alpha=0.5,
               label="그 밖의 점")
    sp = fp[support_mask & keep]
    sp = sp[rng.choice(len(sp), size=min(9000, len(sp)), replace=False)]
    ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=0.7, c="#9ec5f4", linewidths=0, alpha=0.5,
               label=f"지지면 {len(surfaces)} 장 (연결에서 제외)")
    s3 = fp[seed_mask & keep]
    ax.scatter(s3[:, 0], s3[:, 1], s3[:, 2], s=6, c="#eb6834", linewidths=0,
               label=f"씨앗 {len(seeds):,} 개")
    pk = fp[gr.attention_peak_index]
    ax.scatter(*pk, s=120, marker="*", c="#d62fd6", zorder=6, label="attention 최고점")
    _box(ax)
    ax.legend(loc="upper left", fontsize=7.5, markerscale=1.6, framealpha=0.85,
              borderpad=0.4, labelspacing=0.35)
    ax.set_title(f"(a) 씨앗 — attention 상위 {cfg.attention.seed_percentile:.0f} % 에서 출발한다\n"
                 "지지면(파랑)은 연결에서 빼야 영역 성장이 테이블을 타고 번지지 않는다\n"
                 "최고점은 정규화본이 아니라 원시본의 argmax 다", fontsize=11)

    # (b) 후보 덩어리
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    ax.scatter(r[:, 0], r[:, 1], r[:, 2], s=0.6, c="#dedcd7", linewidths=0, alpha=0.4)
    clusters = sorted(gr.clusters, key=lambda c: -c.target_score)[:6]
    for i, cl in enumerate(clusters):
        q = fp[np.asarray(cl.point_indices, np.int64)]
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=4.5, color=fs.CATEGORICAL[i % 8], linewidths=0,
                   label=f"#{cl.id}  점수 {cl.target_score:.3f}  ({cl.point_count} 점)")
    _box(ax)
    ax.legend(loc="upper left", fontsize=7.5, markerscale=1.8, framealpha=0.85,
              borderpad=0.4, labelspacing=0.35)
    ax.set_title(f"(b) 영역 성장이 만든 후보 {len(gr.clusters)} 덩어리 (상위 6 개)\n"
                 "씨앗에서 가까운 점으로 번져 나가며 물체 하나씩을 뭉친다\n"
                 "attention 은 어디서 시작할지만 정하고, 무엇이 한 덩어리인지는 기하가 정한다",
                 fontsize=11)

    # (c) 고른 target — 3D + 경계구
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    ax.scatter(r[:, 0], r[:, 1], r[:, 2], s=0.6, c="#dedcd7", linewidths=0, alpha=0.4)
    tgt = gr.target
    best = max(gr.clusters, key=lambda c: c.target_score)
    q = fp[np.asarray(best.point_indices, np.int64)]
    ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=9, color="#e34948", linewidths=0, label="고른 target")
    if tgt is not None:
        c0 = np.asarray(tgt.centroid, float)
        u, vv = np.mgrid[0:2*np.pi:24j, 0:np.pi:12j]
        rr = float(getattr(tgt, "radius", best.rms_radius) or best.rms_radius)
        ax.plot_wireframe(c0[0] + rr*np.cos(u)*np.sin(vv), c0[1] + rr*np.sin(u)*np.sin(vv),
                          c0[2] + rr*np.cos(vv), color="#0b0b0b", lw=0.4, alpha=0.5)
    _box(ax, x=(0.25, 0.85), y=(-0.25, 0.35), z=(0.70, 1.10))
    ax.legend(loc="upper left", fontsize=8, markerscale=1.6)
    ax.set_title("(c) 출력 — target 하나 (무게중심 + 점 집합)\n"
                 f"무게중심 {np.round(np.asarray(tgt.centroid, float), 3) if tgt else '—'} m  ·  "
                 f"경계구 r = {rr*1000:.0f} mm\n"
                 "구로 싸면 속이 빈 크레이트 내부까지 막힌다 — 그래서 점 기반을 쓴다 (F19)",
                 fontsize=11)

    # (d) 후보 점수 막대
    ax = fig.add_subplot(gs[1, 0])
    cl = sorted(gr.clusters, key=lambda c: -c.target_score)[:8]
    ys = np.arange(len(cl))[::-1]
    ax.barh(ys, [c.target_score for c in cl],
            color=["#e34948"] + ["#9ec5f4"] * (len(cl) - 1), height=0.62)
    for y, c in zip(ys, cl):
        ax.text(c.target_score + 0.008, y, f"{c.point_count} 점 · 최고 att {c.max_attention:.2f}",
                va="center", fontsize=8.5)
    ax.set_yticks(ys); ax.set_yticklabels([f"덩어리 #{c.id}" for c in cl], fontsize=9)
    ax.set_xlabel("target_score")
    ax.set_xlim(0, max(c.target_score for c in cl) * 1.45)
    gap = cl[0].target_score / max(cl[1].target_score, 1e-6) if len(cl) > 1 else float("inf")
    ax.set_title(f"(d) 왜 저것을 골랐나 — 1 위와 2 위의 격차 {gap:.1f} 배\n"
                 "점수는 attention 크기 · 조밀도 · 최고점까지의 거리로 만든다\n"
                 "점수가 정규화된 0~1 스케일을 전제하므로 (e) 의 순서가 중요하다", fontsize=11)

    # (e) 씨앗 컷 — F8 이 사는 자리
    ax = fig.add_subplot(gs[1, 1])
    att = fused_att
    cut = float(np.percentile(att, cfg.attention.seed_percentile))
    ax.hist(att, bins=200, log=True, color="#9ec5f4")
    ax.axvline(cut, color="#e34948", lw=1.8, ls="--")
    ax.annotate(f"p{cfg.attention.seed_percentile:.0f} 컷 = {cut:.4f}",
                xy=(cut, 0.82), xycoords=("data", "axes fraction"), fontsize=9.5,
                color="#e34948", ha="left")
    ax.set_xlabel("융합 후 정규화 attention"); ax.set_ylabel("점 개수 (log)")
    ax.set_title("(e) 씨앗 컷 — 이 컷이 0 이 되면 모든 점이 씨앗이 된다\n"
                 f"이 프레임: attention 이 0 인 점 {100.0*(att<=0).mean():.1f} %, 컷 {cut:.4f} > 0\n"
                 "F8(점의 95 % 이상이 0 이면 퍼센타일 컷이 0 이 된다) 은 여기서 발동하지 않는다",
                 fontsize=11)

    # (f) 표
    rows = [
        ("무엇을 받는가", ""),
        ("입력 1", "AttentionPointCloud — 점마다 값 2 벌"),
        ("입력 2", f"지지면 마스크 — 평면 {len(surfaces)} 장, {int(support_mask.sum()):,} 점"),
        ("", ""),
        ("무엇을 내놓는가", ""),
        ("출력", "GroundingResult — target + 후보 전부 + 실패 사유"),
        ("상태", f"{gr.status.name}"),
        ("", ""),
        ("이 프레임 실측", ""),
        ("씨앗", f"{len(seeds):,} 개 / 점 {len(fused_cloud):,}"),
        ("후보 덩어리", f"{len(gr.clusters)} 개"),
        ("1 위 점수 / 2 위", (f"{cl[0].target_score:.3f} / {cl[1].target_score:.3f}"
                           if len(cl) > 1 else f"{cl[0].target_score:.3f} / —")),
        ("target 무게중심",
         f"{np.round(np.asarray(gr.target.centroid, float), 3)} m" if gr.target else "—"),
        ("", ""),
        ("알려진 한계", ""),
        ("무상태였다 (F17)", "프레임마다 다시 골랐다 → grasp_latch.py 가 잠근다"),
        ("주목 ≠ 조작 (F11)", "파지 중 attention 은 목적지를 본다 — 아래 (c) 가 그 증거"),
    ]
    _table(fig.add_subplot(gs[1, 2]), rows, fs, "(f) 입력 · 출력 · 실측 · 한계")

    fig.suptitle(f"target grounding — 올린 점들에서 '정책이 말하는 물체' 하나를 고른다.  "
                 f"{args.records} 프레임 {args.frame}",
                 fontsize=14, y=0.955)
    fig.patch.set_facecolor(fs.SURFACE)
    out2 = OUT / "doc-stage-target-grounding.png"
    fig.savefig(out2, dpi=118, bbox_inches="tight", facecolor=fs.SURFACE)
    plt.close(fig)
    print(f"wrote {out2}")

    print(f"  융합 {fusion.metrics['n_points_before_fusion']:,} -> {len(fused_cloud):,}")
    print(f"  지지면 {len(surfaces)} 장  {int(support_mask.sum()):,} 점")
    print(f"  grounding {gr.status.name}  후보 {len(gr.clusters)}  씨앗 {len(seeds):,}")
    if gr.target is not None:
        print(f"  target centroid {np.round(np.asarray(gr.target.centroid, float), 3)}")


if __name__ == "__main__":
    main()
