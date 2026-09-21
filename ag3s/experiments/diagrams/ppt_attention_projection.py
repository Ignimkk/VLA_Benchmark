"""발표 슬라이드용 그림 — 모듈 2: 3D Attention Projection (attention lifting).

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.diagrams.ppt_attention_projection \\
        --records run_0004 --attention attention_step1_run0004.npz --frames 15 --frame 5

문서용 `doc_attention_stages.py` 와 **별도다.** 여기서는 슬라이드 두 장에 얹을 수 있게
16:9 · 그림당 패널 1~4 개 · 큰 글씨로 네 장을 낸다.

| 파일 | 슬라이드 | 무엇 |
|---|---|---|
| `ppt-ap-01-attention.png` | 1 | 같은 접근 프레임의 16×16 attention → 정책 RGB → 3D point cloud |
| `ppt-ap-02-io.png`        | 1 | 입력 3 / `lift()` / 출력(값 2 벌) 블록도 |
| `ppt-ap-03-lift3d.png`    | 2 | **3D 점구름에 칠한 attention** — 한 대, 그리고 세 대 융합 |
| `ppt-ap-04-beforeafter.png` | 2 | F9·F2 수정 전/후를 **같은 기록에서 다시 돌려** 비교 |

## 적용 전/후를 어떻게 만드나 — 되돌리는 스위치 둘

* **F9**(`lift()` 가 `image_hw` 없이 걸러진 uv 최댓값으로 해상도를 추정한다) —
  `CameraObservation.image_hw` 를 `None` 으로 바꾸면 수정 전 경로가 그대로 재현된다.
* **F2**(정규화가 카메라별이라 max 융합이 후하게 스케일된 카메라를 고른다) —
  `fuse(..., attention_config=None)` 이 **카메라별 정규화본을 max 로 합치는** 옛 동작이다
  (`multiview.py` 의 폴백 분기). 설정을 주면 융합된 원시본을 한 번에 정규화하는 지금 동작이다.

둘 다 코드를 고치지 않고 **호출만 바꿔** 재현한다. 세 갈래를 같은 프레임에 나란히 돌린다:

    fixed  : image_hw 줌   + 융합 후 정규화   (현재)
    f9_off : image_hw 없음 + 융합 후 정규화   (F9 이전)
    f2_off : image_hw 줌   + 카메라별 정규화  (F2 이전)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures/ppt")
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
ATT_CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
LABEL = ("머리 (zed_left)", "왼손목", "오른손목")

INK = "#0b0b0b"
MUTED = "#52514e"
RED = "#e34948"
BLUE = "#2a78d6"
GREEN = "#1baf7a"
AMBER = "#eda100"
PURPLE = "#4a3aa7"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments.common import figstyle
    if not figstyle.use_korean():
        raise SystemExit("CJK 폰트를 못 찾았다 — 한글이 두부로 나온다")
    return figstyle


def _box(ax, x, y, w, h, text, face, *, fontsize=14, text_colour=None, weight="bold"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.18",
                                facecolor=face, edgecolor=INK, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            weight=weight, color=text_colour or INK, linespacing=1.55)


def _arrow(ax, p0, p1, colour=INK, lw=2.0, style="-|>"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=22, color=colour, lw=lw))


def _box3d(ax, x=(0.10, 1.00), y=(-0.35, 0.50), z=(0.55, 1.15)):
    ax.set_xlim(*x); ax.set_ylim(*y); ax.set_zlim(*z)
    ax.set_xlabel("x [m] 앞", fontsize=11); ax.set_ylabel("y [m] 왼", fontsize=11)
    ax.set_zlabel("z [m] 위", fontsize=11)
    ax.view_init(elev=22, azim=-118)
    ax.tick_params(labelsize=9)


def _inbox(p, x=(0.10, 1.00), y=(-0.35, 0.50), z=(0.55, 1.15)):
    return ((p[:, 0] > x[0]) & (p[:, 0] < x[1]) & (p[:, 1] > y[0]) & (p[:, 1] < y[1])
            & (p[:, 2] > z[0]) & (p[:, 2] < z[1]))


# ------------------------------------------------------------------ 씬 물체 이름 붙이기
def _objects(scene):
    """씬 물체 이름 -> base 프레임 위치. target 이 무엇으로 잡혔는지 이름 붙이는 데 쓴다."""
    import mujoco
    from benchmark.ag3s.experiments.sources.mujoco_source import is_robot_body
    out = {}
    for b in range(scene.model.nbody):
        n = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not n or is_robot_body(n):
            continue
        if any(k in n for k in ("table", "shelf", "floor", "world", "ground", "com_target",
                                "_ee_target", "office")):
            continue
        # `scene.data.xpos[b]` is a MuJoCo-backed view and changes at every replay step.
        # Freeze the pose for the representative frame; otherwise figures label frame i using the
        # final frame's object locations.
        out[n] = np.asarray(scene.data.xpos[b], float).copy()
    return out


def _name_of(centroid, objects, tol=0.12):
    if centroid is None:
        return "없음"
    best, bd = "?", np.inf
    for n, p in objects.items():
        d = float(np.linalg.norm(np.asarray(centroid, float) - p))
        if d < bd:
            best, bd = n, d
    return best if bd < tol else f"{best}?"


# ------------------------------------------------------- 한 프레임의 세 갈래를 다 돌린다
def run_frame(scene, cfg, filter_robot, grids, ts, adapter):
    """`fixed` · `f9_off` · `f2_off` 세 갈래의 융합 + grounding 결과를 돌려준다."""
    from benchmark.ag3s.experiments.sources.mujoco_source import camera_observation
    from benchmark.ag3s.runtime.multiview import fuse, process_observation
    from benchmark.ag3s.stages.support_surface import fit_support_surfaces
    from benchmark.ag3s.stages.target_grounding import ground_target
    from benchmark.ag3s.types import AttentionPointCloud

    obs, frames = [], {}
    for cam, ac in zip(CAMS, ATT_CAMS):
        o, f = camera_observation(scene, cam, filter_robot, attention_map=grids[ac], timestamp=ts)
        obs.append(o)
        frames[cam] = f

    # F9 이전 = 관측이 image_hw 를 안 들고 오던 때. `lift` 가 걸러진 uv 로 스스로 추정한다.
    obs_legacy = [dataclasses.replace(o, image_hw=None) for o in obs]

    res_fixed = [process_observation(o, cfg, robot_model=filter_robot, attention_adapter=adapter)
                 for o in obs]
    res_legacy = [process_observation(o, cfg, robot_model=filter_robot, attention_adapter=adapter)
                  for o in obs_legacy]

    def _ground(results, *, attention_config):
        fused, att, raw = fuse(results, voxel_size=cfg.timing.fusion_voxel_size,
                               frame_id=cfg.frame_id, attention_config=attention_config)
        cloud = fused.as_pointcloud()
        _, support = fit_support_surfaces(cloud, cfg.support_surface, timestamp=ts)
        gr = ground_target(AttentionPointCloud(cloud, att, raw), cfg.clustering,
                           seed_percentile=cfg.attention.seed_percentile,
                           seed_threshold=cfg.attention.seed_threshold,
                           exclude_mask=support, min_radius=cfg.geometry.min_radius,
                           timestamp=ts)
        return dict(cloud=cloud, attention=np.asarray(att, np.float64),
                    raw=np.asarray(raw, np.float64), grounding=gr, support=support, fused=fused)

    return dict(
        fixed=_ground(res_fixed, attention_config=cfg.attention),
        f9_off=_ground(res_legacy, attention_config=cfg.attention),
        # attention_config 없이 부르면 `fuse` 가 **카메라별 정규화본을 max 로 합친다** = F2 이전
        f2_off=_ground(res_fixed, attention_config=None),
        frames=frames, res_fixed=res_fixed, res_legacy=res_legacy, obs=obs)


# --------------------------------------------------------------------- 01 입력 attention
def fig_attention(fs, stats, adapter, cfg, cell, out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    frames, grids = stats["frames"], stats["grids"]
    fig = plt.figure(figsize=(16.0, 8.4))
    fig.patch.set_facecolor(fs.SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.94, 1.02, 1.18], wspace=0.20,
                          left=0.035, right=0.965, top=0.82, bottom=0.11)

    # 1) VLA가 낸 원시 patch attention
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
    ax.set_xticklabels([f"{LABEL[i]}\nmax {grids[ac].max():.4f}"
                        for i, ac in enumerate(ATT_CAMS)], fontsize=10.5)
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("카메라별 표시 눈금", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    ax.set_title("① VLA output\n카메라마다 16 × 16 attention", fontsize=14, weight="bold", pad=10)

    # 2) 정책이 실제로 본 RGB에 같은 attention을 overlay
    ax = fig.add_subplot(gs[0, 1])
    rgb = np.asarray(stats["policy_image"])
    pm_rgb = adapter.to_pixel_map(grids["cam_high"], rgb.shape[:2])
    ax.imshow(rgb)
    ax.imshow(pm_rgb, cmap="inferno", alpha=0.52)
    ax.axis("off")

    # MuJoCo GT apple 중심은 설명용 표식일 뿐이며 lifting 계산에는 쓰지 않는다.
    apple = stats["objects"].get("apple")
    head = frames["zed_left"]
    if apple is not None:
        T = np.asarray(head.T_base_cam, float)
        K = np.asarray(head.camera_intrinsics, float)
        pc = (np.asarray(apple, float) - T[:3, 3]) @ T[:3, :3]
        if pc[2] > 0:
            u = (K[0, 0] * pc[0] / pc[2] + K[0, 2]) / head.hw[1] * rgb.shape[1]
            v = (K[1, 1] * pc[1] / pc[2] + K[1, 2]) / head.hw[0] * rgb.shape[0]
            ax.add_patch(Circle((u, v), 13, fill=False, lw=2.4, color="#40e0d0"))
            ax.annotate("apple", xy=(u, v), xytext=(u + 24, v - 24), color="#ffffff",
                        fontsize=13, weight="bold",
                        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#ffffff"))
    ax.set_title("② 2D 확인\n정책이 본 224 × 224 RGB에 overlay", fontsize=14, weight="bold", pad=10)

    # 3) 같은 frame의 head-camera point cloud에 attention을 부착
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    pts, val = stats["one"]
    keep = _inbox(pts)
    p3, v3 = pts[keep], val[keep]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(p3), size=min(24000, len(p3)), replace=False)
    idx = idx[np.argsort(v3[idx])]
    sc = ax.scatter(p3[idx, 0], p3[idx, 1], p3[idx, 2], s=1.8, c=v3[idx],
                    cmap="inferno", vmin=0, vmax=1, linewidths=0)
    if apple is not None:
        ax.scatter(apple[0], apple[1], apple[2], marker="*", s=180,
                   color="#40e0d0", edgecolor="black", linewidth=0.7, label="apple")
        ax.legend(loc="upper left", fontsize=10)
    _box3d(ax)
    cb = plt.colorbar(sc, ax=ax, fraction=0.028, pad=0.04)
    cb.set_label("attention", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    ax.set_title("③ 3D lifting\n각 point의 (u,v)에서 attention 조회", fontsize=14,
                 weight="bold", pad=6)

    fig.suptitle(f"사과에 접근하는 동안의 attention lifting — run_0004, frame {stats['frame']}",
                 fontsize=18, weight="bold", y=0.965)
    fig.text(0.5, 0.035,
             "같은 시점의 데이터: 16 × 16 attention → RGB에서 위치 확인 → 3D point마다 값 부착  "
             "(청록 표식은 MuJoCo GT이며 계산에는 사용하지 않음)",
             ha="center", fontsize=12.5, color=MUTED)
    fig.savefig(out, dpi=130, facecolor=fs.SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------ 02 I/O
def fig_io(fs, stats, out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16.0, 9.0))
    fig.patch.set_facecolor(fs.SURFACE)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(8.0, 8.55, "lift() — 2D attention 을 3D 점에 옮겨 붙인다",
            ha="center", fontsize=20, weight="bold", color=INK)

    ins = [
        ("PointCloud\nuv 를 보존한 채로 온다", "#9ec5f4"),
        ("VLA attention\n16 x 16 격자 x 카메라 3 대", "#cde2fb"),
        (f"image_hw = {stats['image_hw']}\n어디까지 늘릴지", "#cde2fb"),
    ]
    y = 6.25
    for text, face in ins:
        _box(ax, 0.45, y, 4.3, 1.28, text, face, fontsize=14, weight="normal")
        _arrow(ax, (4.85, y + 0.64), (5.75, 4.55), MUTED, lw=1.5)
        y -= 1.72

    _box(ax, 5.85, 3.45, 4.3, 2.2,
         "lift()\n\n① 격자를 픽셀맵으로 확대\n② 점의 uv 로 그 맵을 찍는다\n③ 정규화",
         GREEN, fontsize=14.5, text_colour="#ffffff")

    outs = [
        ("출력 — AttentionPointCloud\n점마다 값 두 벌\n"
         f"카메라 한 대 {stats['n_one']:,} 점 "
         f"(세 대 융합 후 {stats['n_points']:,})", "#f7d9a0"),
        ("정규화본 — 문턱 비교용\n퍼센타일로 0~1, 상위는 동점이 많다", "#fdf0d5"),
        ("원시본 — 최고점 argmax 용\n동점이 없어야 peak 를 고를 수 있다", "#fdf0d5"),
    ]
    y = 6.30
    for k, (text, face) in enumerate(outs):
        _arrow(ax, (10.25, 4.55), (10.85, y + 0.60), MUTED, lw=1.5)
        _box(ax, 10.95, y, 4.75, 1.24, text, face, fontsize=13, weight="normal")
        y -= 1.72

    ax.add_patch(FancyBboxPatch((0.45, 0.35), 15.25, 2.20,
                                boxstyle="round,pad=0.10,rounding_size=0.18",
                                facecolor="#ecebe7", edgecolor="#d8d7d2", lw=1.2))
    ax.text(8.05, 1.90, "버리는 점은 없다 — attention 이 낮다고 기하를 지우지 않는다",
            ha="center", fontsize=15.5, weight="bold", color=INK)
    ax.text(8.05, 1.00,
            "attention 은 target 이 무엇인지만 정하고, 무엇이 충돌할 수 있는지는 3D 기하가 정한다.\n"
            "attention 을 장애물 탐지 문턱으로 쓰면 정책이 안 보는 장애물이 사라진다.",
            ha="center", va="center", fontsize=13.5, color=MUTED, linespacing=1.7)

    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fs.SURFACE)
    plt.close(fig)


# -------------------------------------------------------------------------- 03 3D 투영
def fig_lift3d(fs, one, fused_cloud, fused_att, stats, out):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.0, 8.6))
    fig.patch.set_facecolor(fs.SURFACE)
    gs = fig.add_gridspec(1, 2, wspace=0.16, left=0.02, right=0.955, top=0.86, bottom=0.05)
    rng = np.random.default_rng(0)

    for col, (pts, val, title) in enumerate((
            (one[0], one[1],
             f"카메라 한 대 — 점 {len(one[0]):,} 개에 attention 하나씩"),
            (fused_cloud, fused_att,
             f"세 대 융합 — {stats['n_before']:,} → {stats['n_points']:,} 점"))):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        k = _inbox(pts)
        p, v = pts[k], val[k]
        idx = rng.choice(len(p), size=min(30000, len(p)), replace=False)
        idx = idx[np.argsort(v[idx])]          # 높은 값이 위에 오도록
        sc = ax.scatter(p[idx, 0], p[idx, 1], p[idx, 2], s=1.5, c=v[idx],
                        cmap="inferno", vmin=0, vmax=1, linewidths=0)
        _box3d(ax)
        cb = plt.colorbar(sc, ax=ax, fraction=0.026, pad=0.04)
        cb.set_label("정규화 attention", fontsize=11)
        cb.ax.tick_params(labelsize=10)
        ax.set_title(title, fontsize=15, weight="bold", pad=6)

    fig.suptitle(f"사과 접근 중 3D attention projection — run_0004, frame {stats['frame']}",
                 fontsize=18, weight="bold", y=0.965)
    fig.savefig(out, dpi=130, facecolor=fs.SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------- 04 적용 전/후
def fig_before_after(fs, rows, vis, out):
    """발표용 요약: F9와 F2를 각각 원인 → 수정 → 영향 한 줄로 읽게 한다."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    n = len(rows)
    gap = np.maximum(
        np.array([r["gap_h"] for r in rows], float),
        np.array([r["gap_w"] for r in rows], float),
    )
    rho = np.array([r["spearman"] for r in rows], float)

    def _moved(r, key, threshold=0.02):
        current, old = r["fixed_c"], r[f"{key}_c"]
        if current is None or old is None:
            return (current is None) != (old is None)
        return float(np.linalg.norm(current - old)) > threshold

    f9_obj = sum(1 for r in rows if r["f9_name"] != r["fixed_name"] or _moved(r, "f9"))
    f2_obj = sum(1 for r in rows if r["f2_name"] != r["fixed_name"] or _moved(r, "f2"))
    f9_lost = sum(1 for r in rows if r["f9_c"] is None and r["fixed_c"] is not None)
    f9_mm = max([float(np.linalg.norm(r["fixed_c"] - r["f9_c"])) * 1000
                 for r in rows if r["fixed_c"] is not None and r["f9_c"] is not None] or [0.0])
    f2_mm = max([float(np.linalg.norm(r["fixed_c"] - r["f2_c"])) * 1000
                 for r in rows if r["fixed_c"] is not None and r["f2_c"] is not None] or [0.0])

    fig, ax = plt.subplots(figsize=(16.0, 9.0))
    fig.patch.set_facecolor(fs.SURFACE)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(8.0, 8.55, "두 구현 오류가 target 선택을 바꿨다",
            ha="center", fontsize=21, weight="bold", color=INK)
    ax.text(8.0, 8.12, f"run_0004의 같은 {n} frame — 수정 전 경로와 현재 경로 비교",
            ha="center", fontsize=12.5, color=MUTED)

    def card(x, colour, code, title, old_text, now_text, result_title, result_text):
        ax.add_patch(FancyBboxPatch((x, 0.75), 7.15, 6.95,
                                    boxstyle="round,pad=0.12,rounding_size=0.24",
                                    facecolor="#f7f6f2", edgecolor="#d1d0cc", lw=1.4))
        ax.text(x + 0.38, 7.28, code, fontsize=15, weight="bold", color=colour,
                bbox=dict(boxstyle="round,pad=0.28", facecolor=colour, edgecolor="none", alpha=0.14))
        ax.text(x + 1.42, 7.32, title, fontsize=16.5, weight="bold", color=INK, va="center")

        ax.text(x + 1.72, 6.48, "수정 전", ha="center", fontsize=12, weight="bold", color=colour)
        ax.text(x + 5.42, 6.48, "현재", ha="center", fontsize=12, weight="bold", color=GREEN)
        _box(ax, x + 0.35, 4.47, 2.75, 1.65, old_text, "#f6d9d7",
             fontsize=12.2, weight="normal")
        _arrow(ax, (x + 3.22, 5.30), (x + 4.02, 5.30), MUTED, lw=2.0)
        _box(ax, x + 4.08, 4.47, 2.72, 1.65, now_text, "#d9eee6",
             fontsize=12.2, weight="normal")

        ax.text(x + 3.58, 3.95, result_title, ha="center", fontsize=12.3,
                color=MUTED, weight="bold")
        ax.add_patch(FancyBboxPatch((x + 0.35, 1.15), 6.45, 2.42,
                                    boxstyle="round,pad=0.10,rounding_size=0.18",
                                    facecolor="#ffffff", edgecolor=colour, lw=2.0))
        ax.text(x + 3.58, 2.36, result_text, ha="center", va="center",
                fontsize=15.0, weight="bold", color=INK, linespacing=1.65)

    true_h, true_w = rows[0]["true_h"], rows[0]["true_w"]
    min_est_w = true_w - int(max(r["gap_w"] for r in rows))
    max_est_w = true_w - int(min(r["gap_w"] for r in rows))
    card(
        0.45, RED, "F9", "image size를 잘못 추정",
        f"filtered uv 끝값을\nimage size로 사용\n폭 {min_est_w}–{max_est_w} px",
        f"camera metadata 사용\nimage_hw =\n({true_h}, {true_w})",
        "결과 — attention 좌표가 밀림",
        f"최대 {gap.max():.0f} px  >  한 cell {true_w/16:.0f} px\n"
        f"target 소실  {f9_lost} / {n} frame",
    )
    card(
        8.40, PURPLE, "F2", "camera 점수 눈금이 서로 다름",
        "camera마다 따로\n0–1 정규화\n그 뒤 max fusion",
        "raw attention을\n먼저 max fusion\n그 뒤 한 번 정규화",
        "결과 — 선택한 target이 달라짐",
        f"target 변경  {f2_obj} / {n} frame\n"
        f"무게중심 차이 최대  {f2_mm:.0f} mm",
    )

    ax.text(8.0, 0.28,
            "F9는 2D→3D 좌표 대응을, F2는 camera 간 점수 비교를 바로잡은 수정이다.",
            ha="center", fontsize=12.5, color=MUTED)
    fig.savefig(out, dpi=130, facecolor=fs.SURFACE)
    plt.close(fig)
    return dict(n=n, f9_obj=f9_obj, f2_obj=f2_obj, f9_mm=f9_mm, f2_mm=f2_mm,
                f9_lost=f9_lost, gap_med=float(np.median(gap)), gap_max=float(gap.max()),
                n_moved=0, rho_med=float(np.median(rho)), rho_min=float(rho.min()))


def main() -> None:
    fs = _style()
    from scipy.stats import spearmanr

    from benchmark.ag3s.stages.attention_lifting import GridAttentionAdapter
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.reports.grounding_report import build_robot_model
    from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--attention", default="attention_step1_run0004.npz")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/archive/step-verification-20260904/step-01-attention.json")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--frame", type=int, default=5, help="대표 그림에 쓸 접근 프레임")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    blob = np.load(args.attention, allow_pickle=False)
    cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    cam_idx = {c: [str(x) for x in blob["cameras"]].index(c) for c in ATT_CAMS}
    A = np.asarray(blob["attention"], np.float32)

    run = load_run(args.records, limit=args.frames)
    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "pointcloud": {"range_max": 2.0}})
    adapter = GridAttentionAdapter.from_config(cfg.attention)

    rows, vis, scatter = [], None, None
    print(f"{'i':>3}  {'현재':<14} {'F9 이전':<14} {'F2 이전':<14}   추정h  참h")
    for i in range(min(args.frames, len(run.steps))):
        pose_scene(scene, run.steps[i])
        grids = {c: A[i, di, ai, cell["layer"], cell["head"], cam_idx[c]] for c in ATT_CAMS}
        r = run_frame(scene, cfg, filter_robot, grids, float(i), adapter)
        objects = _objects(scene)

        def _c(v):
            g = r[v]["grounding"]
            return None if g.target is None else np.asarray(g.target.centroid, float)

        head = r["frames"]["zed_left"]
        true_h, true_w = (int(x) for x in np.asarray(head.depth).shape[:2])
        # `lift` 가 image_hw 없이 추정하는 값 — 걸러진 클라우드의 uv 최댓값 + 1. **세 카메라 다**
        # 본다: 머리 카메라만 보면 어긋남이 0 으로 보이는데 실제로는 손목 쪽에서 난다.
        est = {}
        for cam, res in zip(CAMS, r["res_legacy"]):
            c_ = res.cloud
            if c_.uv is not None and len(c_):
                est[cam] = (int(c_.uv[:, 1].max()) + 1, int(c_.uv[:, 0].max()) + 1)
            else:
                est[cam] = (true_h, true_w)
        gap_h = max(true_h - e[0] for e in est.values())
        gap_w = max(true_w - e[1] for e in est.values())

        fx, f2v = r["fixed"], r["f2_off"]
        sp_i = float(spearmanr(fx["attention"], f2v["attention"]).statistic)
        sa = set(np.asarray(fx["grounding"].seed_indices, int).tolist())
        sb = set(np.asarray(f2v["grounding"].seed_indices, int).tolist())
        seed_i = 100.0 * (1.0 - len(sa & sb) / max(len(sa | sb), 1))

        row = dict(frame=i, true_h=true_h, true_w=true_w, est=est,
                   gap_h=gap_h, gap_w=gap_w, spearman=sp_i, seed_diff=seed_i,
                   fixed_c=_c("fixed"), f9_c=_c("f9_off"), f2_c=_c("f2_off"))
        for k, v in (("fixed", "fixed"), ("f9", "f9_off"), ("f2", "f2_off")):
            row[f"{k}_name"] = _name_of(row[f"{k}_c" if k != "fixed" else "fixed_c"], objects)
        rows.append(row)
        print(f"{i:3d}  {row['fixed_name']:<14} {row['f9_name']:<14} {row['f2_name']:<14}"
              f"  추정 " + " ".join(f"{e[0]}x{e[1]}" for e in est.values())
              + f"  어긋남 {gap_h}x{gap_w} px  rho {sp_i:.3f}")

        # 산점도는 **순위상관이 가장 낮은 프레임** 것을 쓴다 — 차이가 가장 잘 보이는 자리.
        if scatter is None or sp_i < scatter["f2_spearman"]:
            scatter = dict(f2_scatter=(fx["attention"].copy(), f2v["attention"].copy()),
                           f2_spearman=sp_i, f2_seed_diff=seed_i, f2_frame=i)
        if i == args.frame:
            one = (r["res_fixed"][0].cloud.points,
                   np.asarray(r["res_fixed"][0].attention, np.float64))
            vis = dict(objects=objects, frames=r["frames"], grids=grids, one=one,
                       frame=i, policy_image=np.asarray(run.steps[i].images["cam_high"]),
                       cloud=fx["cloud"].points, att=fx["attention"],
                       n_points=len(fx["cloud"]), n_one=int(len(one[0])),
                       n_before=int(sum(len(x.cloud) for x in r["res_fixed"])),
                       image_hw=tuple(int(x) for x in head.hw))
    scene.close()

    if vis is None:
        raise SystemExit(f"--frame {args.frame} 이 --frames {args.frames} 범위 밖이다")
    vis.update(scatter)
    vis["objects"] = {k: v for k, v in vis["objects"].items()}

    fig_attention(fs, vis, adapter, cfg, cell, out / "ppt-ap-01-attention.png")
    fig_io(fs, vis, out / "ppt-ap-02-io.png")
    fig_lift3d(fs, vis["one"], vis["cloud"], vis["att"], vis,
               out / "ppt-ap-03-lift3d.png")
    summary = fig_before_after(fs, rows, vis, out / "ppt-ap-04-beforeafter.png")

    print(f"\nwrote 4 figures -> {out}")
    print(f"  프레임 {args.frame}: 점 {vis['n_before']:,} → {vis['n_points']:,}  "
          f"image_hw {vis['image_hw']}")
    print(f"  F9  해상도 어긋남 중앙 {summary['gap_med']:.0f} px, 최대 {summary['gap_max']:.0f} px"
          f"  ·  target 이 달라진 프레임 {summary['f9_obj']} / {summary['n']}"
          f"  (그중 소실 {summary['f9_lost']})")
    print(f"  F2  순위상관 최저 {vis['f2_spearman']:.3f} (프레임 {vis['f2_frame']}), "
          f"중앙 {summary['rho_med']:.3f}  씨앗 {vis['f2_seed_diff']:.1f} % 상이"
          f"  ·  target 이 달라진 프레임 {summary['f2_obj']} / {summary['n']}"
          f"  ·  무게중심 최대 {summary['f2_mm']:.1f} mm")


if __name__ == "__main__":
    main()
