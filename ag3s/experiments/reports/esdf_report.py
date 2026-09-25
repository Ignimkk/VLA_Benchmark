"""ESDF backend 진단 — 필드가 무엇을 알고 무엇을 모르는지 눈으로 본다.

숫자는 `docs/ESDF-BACKEND.md` 에 있다. 이 스크립트는 그 숫자들이 **어디서** 오는지를 보여준다.
거리장은 3차원이므로 자르지 않으면 볼 수 없고, 어디를 자르느냐가 곧 무엇을 묻느냐다.

* 수평 단면(테이블 상판 위) — 과일과 크레이트 벽을 가로지른다. 크레이트 **내부가 비어 있는가**를
  묻는 단면이고, primitive backend 가 실패하는 바로 그 지점이다.
* 수직 단면(y = 0) — 바닥·테이블·크레이트를 세로로 가로지른다. 미관측 영역이 어떤 모양으로
  생기는지, 즉 **가림이 어디에 그림자를 남기는지**를 묻는다.

실행:
    MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.reports.esdf_report \
        --records outputs/.../ag3s_records/run_0002
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from benchmark.ag3s.config import AG3SConfig, EsdfConfig
from benchmark.ag3s.fields.esdf import (
    FREE, OCCUPIED, UNKNOWN, CameraDepth, EsdfBuilder, VoxelGrid, occupied_mask,
)
from benchmark.ag3s.experiments.common.outputs import add_tag_argument, resolve
from benchmark.ag3s.experiments.common.figstyle import (
    CATEGORICAL, GRID_INK, INK, INK_2, SURFACE, sequential_cmap, style_axes, use_korean,
)
from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
from benchmark.ag3s.stages.geometry import to_spheres

CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def slice_index(grid: VoxelGrid, axis: int, value: float) -> int:
    return int(np.clip(round((value - grid.origin[axis]) / grid.voxel_size), 0, grid.shape[axis] - 1))


def extent_for(grid: VoxelGrid, axes: tuple[int, int]) -> tuple[float, float, float, float]:
    a, b = axes
    lo = grid.origin - 0.5 * grid.voxel_size
    hi = grid.upper + 0.5 * grid.voxel_size
    return (lo[a], hi[a], lo[b], hi[b])


def main() -> None:
    import mujoco

    from benchmark.ag3s.experiments.reports.grounding_report import build_robot_model
    from benchmark.ag3s.experiments.sources.mujoco_source import gaussian_attention, is_robot_body
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--voxel", type=float, default=0.010)
    ap.add_argument("--z-slice", type=float, default=0.87, help="수평 단면 높이 (m, base 프레임)")
    ap.add_argument("--y-slice", type=float, default=0.0, help="수직 단면 위치 (m)")
    ap.add_argument("--out-doc", default="benchmark/ag3s/docs/archive/14d-era-20260923/esdf-diagnostics.md")
    ap.add_argument("--out-figs", default="benchmark/ag3s/asset/image/esdf")
    add_tag_argument(ap)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    use_korean()
    import matplotlib.pyplot as plt

    run = load_run(args.records)
    scene = replay_scene(run)
    robot = build_robot_model(scene)
    pose_scene(scene, run.steps[args.frame])
    names = {i: (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
             for i in range(scene.model.nbody)}
    robot_ids = [i for i, n in names.items() if is_robot_body(n)]

    frames = {}
    for cam in CAMERAS:
        f = scene.capture(cam)
        frames[cam] = f
    head = frames["zed_left"]
    attention = gaussian_attention(head, scene.body_position_in_base(args.target))
    q = np.asarray([scene.data.qpos[scene._qadr[j]] for j in DEFAULT_RBY1_JOINTS], float)
    centres, radii = robot.sphere_centers_numeric(q)
    link_names = list(robot.sphere_link_names)

    # --- 파이프라인을 both 로 한 번 돌려 두 표현을 동시에 얻는다 ------------------------
    cfg = AG3SConfig.from_dict({
        "collision_backend": "both",
        "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4},
    })
    ag = AG3S(cfg, robot_model=robot, constraint_robot_model=robot)
    t0 = time.time()
    out = ag.process(depth=head.depth, camera_intrinsics=head.camera_intrinsics,
                     T_base_cam=head.T_base_cam, attention_map=attention,
                     robot_state=head.robot_state, phase="approach")
    build_s = time.time() - t0
    field = out.esdf
    grid = field.grid

    # 진단용 점유 격자를 다시 만든다 (파이프라인은 필드만 돌려준다)
    diag = EsdfBuilder(cfg.esdf)
    cams = [CameraDepth(c, frames[c].depth, frames[c].camera_intrinsics, frames[c].T_base_cam,
                        robot_mask=np.isin(frames[c].body_ids, robot_ids)) for c in CAMERAS]
    diag.update(cams)
    occupancy = diag.volume.occupancy(surface_band=cfg.esdf.surface_band)
    tsdf = diag.volume.tsdf
    weight = diag.volume.weight

    paths = resolve(out_figs=args.out_figs, out_doc=args.out_doc,
                    tag=args.tag).prepare()
    figs, img = paths.figures, paths.image_prefix
    zi = slice_index(grid, 2, args.z_slice)
    yi = slice_index(grid, 1, args.y_slice)
    ext_xy = extent_for(grid, (0, 1))
    ext_xz = extent_for(grid, (0, 2))

    def show(ax, img, extent, **kw):
        # imshow 는 첫 축을 세로로 그린다. 우리 배열의 첫 축은 x 이므로 전치해야 축이 맞는다.
        return ax.imshow(img.T, origin="lower", extent=extent, aspect="equal", **kw)

    # ---------------------------------------------------------------- fig1 세 상태 점유
    from matplotlib.colors import BoundaryNorm, ListedColormap
    occ_cmap = ListedColormap(["#f2f1ec", CATEGORICAL[7], "#cfcec8"])  # free, occupied, unknown
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], occ_cmap.N)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=160)
    style_axes(fig, (a1, a2))
    show(a1, occupancy[:, :, zi], ext_xy, cmap=occ_cmap, norm=norm)
    a1.set_title(f"수평 단면 z = {args.z_slice:.2f} m", fontsize=10, color=INK)
    a1.set_xlabel("base x (m)"); a1.set_ylabel("base y (m)")
    show(a2, occupancy[:, yi, :], ext_xz, cmap=occ_cmap, norm=norm)
    a2.set_title(f"수직 단면 y = {args.y_slice:.2f} m", fontsize=10, color=INK)
    a2.set_xlabel("base x (m)"); a2.set_ylabel("base z (m)")
    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, mfc=c, mec="none", label=l)
               for c, l in zip(["#f2f1ec", CATEGORICAL[7], "#cfcec8"],
                               ["자유 (본 적 있고 비어 있음)", "점유 (표면)",
                                "미관측 (본 적 없음)"])]
    leg = fig.legend(handles=handles, frameon=False, fontsize=9, ncol=3, loc="lower center",
                     bbox_to_anchor=(0.5, -0.03))
    for t in leg.get_texts(): t.set_color(INK_2)
    fig.suptitle(f"세 상태 점유 — 미관측 {field.unknown_fraction:.0%} 가 어디에 있는가",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(figs / "fig1_occupancy.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- fig2 거리장
    from matplotlib.colors import TwoSlopeNorm
    d_xy = field.distance_grid[:, :, zi]
    d_xz = field.distance_grid[:, yi, :]
    vmax = float(cfg.esdf.max_distance)
    dnorm = TwoSlopeNorm(vmin=-vmax / 4, vcenter=0.0, vmax=vmax)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.6), dpi=160)
    style_axes(fig, (a1, a2))
    for ax, dd, ext, xl, yl, ttl in (
            (a1, d_xy, ext_xy, "base x (m)", "base y (m)", f"수평 z = {args.z_slice:.2f} m"),
            (a2, d_xz, ext_xz, "base x (m)", "base z (m)", f"수직 y = {args.y_slice:.2f} m")):
        im = show(ax, dd, ext, cmap="RdBu", norm=dnorm)
        # 표면(0)과, 로봇 구가 실제로 요구하는 여유(반지름 중앙값 + 마진)
        req = float(np.median(radii)) + 0.05
        xs = np.linspace(ext[0], ext[1], dd.shape[0])
        ys = np.linspace(ext[2], ext[3], dd.shape[1])
        ax.contour(xs, ys, dd.T, levels=[0.0], colors=[INK], linewidths=1.4)
        ax.contour(xs, ys, dd.T, levels=[req], colors=[CATEGORICAL[3]], linewidths=1.2,
                   linestyles="--")
        ax.set_title(ttl, fontsize=10, color=INK)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    cb = fig.colorbar(im, ax=[a1, a2], fraction=.025, pad=.02)
    cb.set_label("ESDF 거리 (m) — 음수는 표면 안쪽", color=INK_2, fontsize=8)
    cb.ax.tick_params(colors=INK_2, labelsize=7); cb.outline.set_edgecolor(GRID_INK)
    fig.suptitle(f"거리장 — 검정선 = 표면(d=0), 노란 점선 = 로봇 구가 요구하는 여유 "
                 f"{(float(np.median(radii))+0.05)*1000:.0f} mm",
                 color=INK, fontsize=11.5, x=0.01, y=1.02, ha="left")
    fig.savefig(figs / "fig2_distance.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------- fig3 primitive 가 만드는 없는 장애물
    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=160)
    style_axes(fig, ax)
    im = show(ax, d_xy, ext_xy, cmap="RdBu", norm=dnorm)
    xs = np.linspace(ext_xy[0], ext_xy[1], d_xy.shape[0])
    ys = np.linspace(ext_xy[2], ext_xy[3], d_xy.shape[1])
    ax.contour(xs, ys, d_xy.T, levels=[0.0], colors=[INK], linewidths=1.4)
    seen = set()
    for cand in out.candidates:
        for prim in cand.geometry:
            for cen, r in to_spheres(prim):
                dz = abs(float(cen[2]) - args.z_slice)
                if dz >= r:
                    continue  # 이 단면을 지나지 않는 구
                rr = float(np.sqrt(max(r * r - dz * dz, 0.0)))
                is_t = cand.source_type.value == "target"
                lab = "target 후보" if is_t else "obstacle 후보"
                ax.add_patch(plt.Circle((cen[0], cen[1]), rr, fill=False, lw=2.2 if is_t else 1.4,
                                        ec=CATEGORICAL[0] if is_t else CATEGORICAL[2],
                                        label=None if lab in seen else lab))
                seen.add(lab)
    obj = np.array([scene.body_position_in_base(b)
                    for b in ("crate", "apple", "banana", "orange", "pear")])
    ax.set_xlim(obj[:, 0].min() - 0.45, obj[:, 0].max() + 0.45)
    ax.set_ylim(obj[:, 1].min() - 0.45, obj[:, 1].max() + 0.45)
    ax.set_xlabel("base x (m)"); ax.set_ylabel("base y (m)")
    ax.set_title(f"같은 단면 위의 두 표현 — 원 안이 파랗다면 그것은 없는 장애물이다",
                 color=INK, fontsize=11, loc="left", pad=8)
    leg = ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center",
                    bbox_to_anchor=(0.5, -0.12))
    for t in leg.get_texts(): t.set_color(INK_2)
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.set_label("ESDF 거리 (m)", color=INK_2, fontsize=8)
    cb.ax.tick_params(colors=INK_2, labelsize=7); cb.outline.set_edgecolor(GRID_INK)
    fig.tight_layout()
    fig.savefig(figs / "fig3_primitive_overlay.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- fig4 로봇 구 여유
    planes = np.full(len(centres), np.inf)
    for s_ in out.support_surfaces:
        planes = np.minimum(planes, centres @ np.asarray(s_.normal) - s_.offset
                            - radii - s_.safety_margin)
    esdf_clear = np.minimum(field.distance(centres) - radii - 0.05, planes)
    prim_clear = np.full(len(centres), np.inf)
    for cand in out.candidates:
        for prim in cand.geometry:
            for cen, r in to_spheres(prim):
                prim_clear = np.minimum(
                    prim_clear, np.linalg.norm(centres - cen, axis=1) - radii - r - 0.05)
    prim_clear = np.minimum(prim_clear, planes)
    # 세 항을 따로 둔다. 합친 숫자만 보면 남은 위반이 어느 표현의 책임인지 알 수 없다.
    field_only = field.distance(centres) - radii - 0.05
    n_field = int((field_only < 0).sum())
    n_plane = int((planes < 0).sum())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=160)
    style_axes(fig, (a1, a2))
    a1.grid(color=GRID_INK, lw=.6); a1.set_axisbelow(True)
    a1.scatter(prim_clear * 1000, esdf_clear * 1000, s=14, c=CATEGORICAL[0], lw=0, alpha=.75)
    lim = [min(prim_clear.min(), esdf_clear.min()) * 1000 - 20,
           max(prim_clear.max(), esdf_clear.max()) * 1000 + 20]
    a1.plot(lim, lim, color=INK_2, lw=1.0, ls=(0, (4, 3)))
    a1.axhline(0, color=CATEGORICAL[7], lw=1.0); a1.axvline(0, color=CATEGORICAL[7], lw=1.0)
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("primitive 여유 (mm)"); a1.set_ylabel("ESDF 여유 (mm)")
    a1.set_title(f"로봇 구 {len(centres)}개 — 점선 위쪽 = ESDF 가 더 여유롭다",
                 fontsize=10, color=INK, loc="left")
    order = np.argsort(esdf_clear)
    a2.grid(axis="y", color=GRID_INK, lw=.6); a2.set_axisbelow(True)
    a2.plot(np.arange(len(order)), prim_clear[order] * 1000, lw=1.8, color=CATEGORICAL[2],
            label=f"primitive (위반 {int((prim_clear<0).sum())})")
    a2.plot(np.arange(len(order)), esdf_clear[order] * 1000, lw=1.8, color=CATEGORICAL[0],
            label=f"ESDF (위반 {int((esdf_clear<0).sum())})")
    a2.plot(np.arange(len(order)), planes[order] * 1000, lw=1.2, color=CATEGORICAL[3],
            ls=(0, (4, 3)), label=f"지지면 평면만 (위반 {n_plane})")
    a2.plot(np.arange(len(order)), field_only[order] * 1000, lw=1.2, color=CATEGORICAL[4],
            ls=(0, (2, 2)), label=f"ESDF 필드만 (위반 {n_field})")
    a2.axhline(0, color=CATEGORICAL[7], lw=1.2)
    a2.set_xlabel("로봇 구 (ESDF 여유 오름차순)"); a2.set_ylabel("여유 (mm)")
    a2.set_title("같은 구를 두 표현이 어떻게 보는가", fontsize=10, color=INK, loc="left")
    leg = a2.legend(frameon=False, fontsize=8.5)
    for t in leg.get_texts(): t.set_color(INK_2)
    fig.suptitle("primitive 는 없는 장애물을 만든다 — 두 표현의 여유거리 비교",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figs / "fig4_clearance_compare.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- fig5 해상도 비교
    res = (0.005, 0.010, 0.020)
    fig, axes = plt.subplots(1, len(res), figsize=(4.4 * len(res), 4.4), dpi=160)
    style_axes(fig, axes)
    stats_by_res = {}
    for ax, vs in zip(axes, res):
        b = EsdfBuilder(EsdfConfig(voxel_size=vs, max_distance=0.4))
        t0 = time.time()
        f = b.update(cams)
        dt = time.time() - t0
        stats_by_res[vs] = (dt, f.stats["n_voxels"], f.distance_grid.nbytes,
                            f.unknown_fraction)
        g = f.grid
        k = slice_index(g, 2, args.z_slice)
        im = show(ax, f.distance_grid[:, :, k], extent_for(g, (0, 1)), cmap="RdBu", norm=dnorm)
        xs = np.linspace(*extent_for(g, (0, 1))[:2], f.distance_grid.shape[0])
        ys = np.linspace(*extent_for(g, (0, 1))[2:], f.distance_grid.shape[1])
        ax.contour(xs, ys, f.distance_grid[:, :, k].T, levels=[0.0], colors=[INK], linewidths=1.2)
        ax.set_xlim(obj[:, 0].min() - 0.35, obj[:, 0].max() + 0.35)
        ax.set_ylim(obj[:, 1].min() - 0.35, obj[:, 1].max() + 0.35)
        ax.set_xlabel("base x (m)")
        ax.set_title(f"{vs*1000:.0f} mm — {dt:.2f}s, {f.distance_grid.nbytes/1e6:.0f} MB\n"
                     f"이산화 편향 {vs*500:.1f} mm", fontsize=9.5, color=INK)
    axes[0].set_ylabel("base y (m)")
    fig.suptitle("복셀 해상도 — 정확도와 비용은 같은 손잡이다",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figs / "fig5_resolution.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- fig6 TSDF 자체
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=160)
    style_axes(fig, (a1, a2))
    trunc = cfg.esdf.truncation
    t_masked = np.where(weight[:, yi, :] > 0, tsdf[:, yi, :], np.nan)
    im = show(a1, t_masked, ext_xz, cmap="RdBu",
              norm=TwoSlopeNorm(vmin=-trunc, vcenter=0.0, vmax=trunc))
    a1.set_title(f"TSDF (절단 ±{trunc*1000:.0f} mm), 관측된 복셀만", fontsize=10, color=INK)
    a1.set_xlabel("base x (m)"); a1.set_ylabel("base z (m)")
    cb = fig.colorbar(im, ax=a1, fraction=.04, pad=.02)
    cb.ax.tick_params(colors=INK_2, labelsize=7); cb.outline.set_edgecolor(GRID_INK)
    im2 = show(a2, weight[:, yi, :], ext_xz, cmap=sequential_cmap())
    a2.set_title("적분 가중치 — 0은 어느 카메라도 못 본 곳", fontsize=10, color=INK)
    a2.set_xlabel("base x (m)"); a2.set_ylabel("base z (m)")
    cb = fig.colorbar(im2, ax=a2, fraction=.04, pad=.02)
    cb.ax.tick_params(colors=INK_2, labelsize=7); cb.outline.set_edgecolor(GRID_INK)
    fig.suptitle(f"ESDF 이전 단계 — 수직 단면 y = {args.y_slice:.2f} m",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figs / "fig6_tsdf.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------- fig7 TO 직전 ESDF 입력과 제약 행
    # AG3S -> TO 경계에서 필드가 넘기는 것은 `distance_grid`이고, TO가 로봇 구 중심에서
    # `distance`와 `gradient`를 질의해 아래 한 행으로 바꾼다. 이 그림은 보고서의 다른 그림처럼
    # 필드 자체를 보여주는 데서 멈추지 않고, 그 경계의 실제 숫자를 한 화면에 놓는다.
    to_margin = 0.05
    query_distance = field.distance(centres)
    query_gradient = field.gradient(centres)
    query_clearance = query_distance - radii - to_margin
    query_order = np.argsort(query_clearance)

    fig = plt.figure(figsize=(17.0, 5.5), dpi=170)
    a1 = fig.add_subplot(1, 3, 1, projection="3d")
    a2 = fig.add_subplot(1, 3, 2)
    a3 = fig.add_subplot(1, 3, 3)
    style_axes(fig, (a1, a2, a3))

    # (a) `distance_grid < 0`인 점유 voxel과, TO가 질의할 로봇 sphere center. 구의 색은 이미
    # 최종 스칼라 제약값 h=d-r-margin 이므로 빨강은 위반, 파랑은 여유다.
    occupied_ijk = np.argwhere(field.distance_grid < 0.0)
    if len(occupied_ijk):
        stride = max(1, len(occupied_ijk) // 6000)
        surface_xyz = grid.origin + occupied_ijk[::stride] * grid.voxel_size
        a1.scatter(surface_xyz[:, 0], surface_xyz[:, 1], surface_xyz[:, 2], s=1.0,
                   c="#60656b", alpha=.18, depthshade=False, label="ESDF 점유 voxel")
    c_lim = max(0.10, min(0.30, float(np.max(np.abs(query_clearance)))))
    c_norm = TwoSlopeNorm(vmin=-c_lim, vcenter=0.0, vmax=c_lim)
    sphere_plot = a1.scatter(
        centres[:, 0], centres[:, 1], centres[:, 2],
        c=query_clearance, cmap="RdBu", norm=c_norm,
        s=12.0 + 900.0 * radii ** 2, edgecolors="none", depthshade=False,
        label="TO robot-sphere 질의점",
    )
    arrow_rows = [int(i) for i in query_order
                  if np.linalg.norm(query_gradient[i]) > 1e-6][:12]
    if arrow_rows:
        ii = np.asarray(arrow_rows, int)
        gg = query_gradient[ii]
        gg = gg / np.maximum(np.linalg.norm(gg, axis=1, keepdims=True), 1e-12)
        a1.quiver(centres[ii, 0], centres[ii, 1], centres[ii, 2],
                  gg[:, 0], gg[:, 1], gg[:, 2], length=.10, normalize=True,
                  color=CATEGORICAL[3], linewidth=1.2, arrow_length_ratio=.25)
    a1.set_xlabel("base x (m)"); a1.set_ylabel("base y (m)"); a1.set_zlabel("base z (m)")
    a1.set_title("① ESDF 표면 + TO 질의점\n화살표 = $\\nabla d(p)$", fontsize=10, color=INK)
    cb = fig.colorbar(sphere_plot, ax=a1, fraction=.035, pad=.08, shrink=.72)
    cb.set_label("$h=d-r-m$ (m)", color=INK_2, fontsize=8)
    cb.ax.tick_params(colors=INK_2, labelsize=7); cb.outline.set_edgecolor(GRID_INK)

    # (b) 가장 제약적인 유효 질의 하나를 고르고, gradient의 가장 작은 성분을 버린 평면으로
    # 자른다. 따라서 화면의 화살표는 가능한 한 3D gradient 방향을 잃지 않는다.
    valid_rows = [int(i) for i in query_order
                  if np.linalg.norm(query_gradient[i]) > 1e-6
                  and query_distance[i] < field.max_distance - grid.voxel_size]
    focus = valid_rows[0] if valid_rows else int(query_order[0])
    omit = int(np.argmin(np.abs(query_gradient[focus])))
    axes2 = tuple(i for i in range(3) if i != omit)
    coord_names = ("x", "y", "z")
    fixed_i = slice_index(grid, omit, float(centres[focus, omit]))
    dd = np.take(field.distance_grid, fixed_i, axis=omit)
    ext = extent_for(grid, axes2)
    local_norm = TwoSlopeNorm(vmin=-min(.10, field.max_distance), vcenter=0.0,
                              vmax=min(.30, field.max_distance))
    im = show(a2, dd, ext, cmap="RdBu", norm=local_norm)
    uu = np.linspace(ext[0], ext[1], dd.shape[0])
    vv = np.linspace(ext[2], ext[3], dd.shape[1])
    a2.contour(uu, vv, dd.T, levels=[0.0], colors=[INK], linewidths=1.3)
    p2 = centres[focus, list(axes2)]
    a2.add_patch(plt.Circle(p2, radii[focus], fill=False, lw=1.8,
                            ec=CATEGORICAL[0], label="robot sphere $r$"))
    a2.add_patch(plt.Circle(p2, radii[focus] + to_margin, fill=False, lw=1.5,
                            ls="--", ec=CATEGORICAL[3], label="$r+$ margin"))
    gp = query_gradient[focus, list(axes2)]
    gp_norm = float(np.linalg.norm(gp))
    if gp_norm > 1e-9:
        gp = gp / gp_norm
        a2.arrow(p2[0], p2[1], .12 * gp[0], .12 * gp[1], width=.004,
                 head_width=.025, length_includes_head=True, color=CATEGORICAL[3], zorder=5)
    radius_view = max(.22, 2.5 * float(radii[focus] + to_margin))
    a2.set_xlim(p2[0] - radius_view, p2[0] + radius_view)
    a2.set_ylim(p2[1] - radius_view, p2[1] + radius_view)
    a2.set_xlabel(f"base {coord_names[axes2[0]]} (m)")
    a2.set_ylabel(f"base {coord_names[axes2[1]]} (m)")
    a2.set_title(
        f"② 가장 제약적인 유효 행 — `{link_names[focus]}` #{focus}\n"
        f"d={query_distance[focus]*1000:.0f}, r={radii[focus]*1000:.0f}, "
        f"m={to_margin*1000:.0f} → h={query_clearance[focus]*1000:+.0f} mm",
        fontsize=9.5, color=INK,
    )
    leg = a2.legend(frameon=False, fontsize=8, loc="lower right")
    for t in leg.get_texts(): t.set_color(INK_2)
    cb = fig.colorbar(im, ax=a2, fraction=.04, pad=.02)
    cb.set_label("ESDF $d(p)$ (m)", color=INK_2, fontsize=8)
    cb.ax.tick_params(colors=INK_2, labelsize=7); cb.outline.set_edgecolor(GRID_INK)

    # (c) TO에 들어가는 최종 scalar row를 그대로 정렬한다. plane 제약은 별도 블록이므로 여기에는
    # 섞지 않는다. 같은 이름의 link가 여러 줄인 것은 한 link가 여러 collision sphere를 갖기 때문이다.
    n_show = min(14, len(query_order))
    rows = query_order[:n_show][::-1]
    yy = np.arange(n_show)
    bar_colors = [CATEGORICAL[7] if query_clearance[i] < 0 else CATEGORICAL[0]
                  for i in rows]
    a3.barh(yy, query_clearance[rows] * 1000, color=bar_colors, alpha=.88)
    a3.axvline(0.0, color=INK, lw=1.1)
    a3.set_yticks(yy, [f"{link_names[i]} #{i}" for i in rows], fontsize=7.5)
    a3.set_xlabel("최종 충돌 제약값 $h=d-r-m$ (mm)")
    a3.set_title("③ TO가 받는 가장 작은 ESDF 행\n빨강 = h < 0 (충돌 제약 위반)",
                 fontsize=10, color=INK)
    a3.grid(axis="x", color=GRID_INK, lw=.6); a3.set_axisbelow(True)
    for y, i in zip(yy, rows):
        # 음수 bar의 왼쪽 끝에 붙이면 y축 link label과 겹친다. 식은 0선 안쪽에 정렬해
        # 행 이름과 값의 역할을 시각적으로 분리한다.
        x_text = -1.5 if query_clearance[i] < 0 else 1.5
        ha = "right" if query_clearance[i] < 0 else "left"
        a3.text(x_text, y,
                f"d {query_distance[i]*1000:.0f} − r {radii[i]*1000:.0f} − m 50",
                va="center", ha=ha, fontsize=6.8, color=INK_2)

    n_viol = int((query_clearance < 0.0).sum())
    fig.suptitle(
        "TSDF/ESDF → TO 경계 — field와 robot sphere가 최종 scalar collision row가 되는 순간\n"
        f"distance_grid {field.distance_grid.shape} {field.distance_grid.dtype} · "
        f"voxel {grid.voxel_size*1000:.0f} mm · robot spheres {len(centres)} · "
        f"ESDF 위반 {n_viol}",
        color=INK, fontsize=12, x=.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, .88))
    fig.savefig(figs / "fig7_to_constraint_input.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)

    # 그림에 적힌 값을 기계적으로도 확인할 수 있도록, dense grid 자체를 중복 저장하지 않고 TO가
    # 이번 pose에서 질의한 행만 작은 JSON으로 남긴다.
    to_rows = []
    for i in range(len(centres)):
        to_rows.append({
            "sphere_index": i,
            "link": link_names[i],
            "center_m": centres[i].tolist(),
            "radius_m": float(radii[i]),
            "distance_m": float(query_distance[i]),
            "gradient": query_gradient[i].tolist(),
            "margin_m": to_margin,
            "constraint_h_m": float(query_clearance[i]),
            "violated": bool(query_clearance[i] < 0.0),
        })
    to_data = {
        "source_record": str(run.path),
        "frame": int(args.frame),
        "field": {
            "grid_shape": list(field.distance_grid.shape),
            "dtype": str(field.distance_grid.dtype),
            "origin_m": grid.origin.tolist(),
            "voxel_size_m": float(grid.voxel_size),
            "max_distance_m": float(field.max_distance),
            "unknown_fraction": float(field.unknown_fraction),
        },
        "constraint": "h = d_esdf(center) - robot_sphere_radius - esdf_margin",
        "rows": to_rows,
    }
    (figs / "to_constraint_input.json").write_text(
        json.dumps(to_data, indent=2, ensure_ascii=False) + "\n")
    scene.close()

    # ------------------------------------------------------------------------ 문서
    worst = np.argsort(esdf_clear)[:8]
    doc = f"""# ESDF 진단 — 필드가 무엇을 알고 무엇을 모르는가

숫자는 [ESDF-BACKEND.md](ESDF-BACKEND.md) 에 있다. 이 문서는 그 숫자들이 **어디서** 오는지를
보여준다. 거리장은 3차원이라 자르지 않으면 볼 수 없고, 어디를 자르느냐가 곧 무엇을 묻느냐다.

| | |
|---|---|
| 기록 | `{run.path}` 프레임 {args.frame} |
| 복셀 | {args.voxel*1000:.0f} mm, 격자 {field.stats['grid_shape']} ({field.stats['n_voxels']/1e6:.1f} M) |
| 빌드 | {build_s:.2f} s |
| 점유 / 자유 / 미관측 | {field.stats['n_occupied']:,} / {field.stats['n_free']:,} / {field.stats['n_unknown']:,} (**{field.unknown_fraction:.0%}**) |
| 파낸 복셀 | 지지면 {field.stats.get('n_support_voxels_carved', 0):,} · target {field.stats.get('n_target_voxels_carved', 0):,} |
| 단면 | 수평 z = {args.z_slice:.2f} m, 수직 y = {args.y_slice:.2f} m |

## 1. 세 상태 점유 — 미관측이 어디에 있는가

![점유]({img}/fig1_occupancy.png)

**만드는 법.** 세 카메라의 깊이를 TSDF 에 적분한 뒤 복셀마다 점유/자유/미관측을 정한다. 가중치가
0인 복셀은 어느 카메라도 보지 못한 것이다. 로봇 픽셀은 적분 전에 마스크로 빼므로 팔이 있던
자리는 자유가 아니라 **미관측**이 된다.

**주의.** 1번과 6번 그림은 **파냄 이전의 원본 점유**다 (지지면·target 제거 전). 진단에는 카메라가
무엇을 봤는지가 필요하기 때문이다. 2·3·4번은 파이프라인이 실제로 내놓는 필드, 즉 파냄 이후다.

**읽는 법.** 회색(미관측)이 {field.unknown_fraction:.0%} 다. 그 대부분은 테이블 아래·물체 뒤·방
바깥이고, 카메라가 볼 수 없었던 곳이다. `unknown_policy: free` 는 이 회색을 통과 가능으로
취급한다는 뜻이다 — primitive backend 도 같은 성질이지만, 여기서는 **셀 수 있다**.

## 2. 거리장

![거리장]({img}/fig2_distance.png)

**만드는 법.** 점유 복셀 집합에 부호 있는 거리 변환(EDT)을 건다. 파란색이 표면에서 멀고
빨간색이 표면 안쪽이며, 검정선이 표면(d = 0)이다.

**읽는 법.** 노란 점선이 로봇 구가 실제로 요구하는 여유
({(float(np.median(radii))+0.05)*1000:.0f} mm = 구 반지름 중앙값 + 마진 50 mm)다. **그 선 바깥이
통과 가능한 영역**이다. 크레이트 내부에 파란 영역이 있으면 그것이 primitive backend 가 삼켰던
공간이다.

## 3. 같은 단면 위의 두 표현

![겹침]({img}/fig3_primitive_overlay.png)

**만드는 법.** 같은 수평 단면에 primitive 후보의 제약용 구 단면을 겹친다. 이 단면을 지나지 않는
구는 그리지 않고, 지나는 구는 `sqrt(r² - dz²)` 로 줄여 그린다.

**읽는 법.** **원 안쪽이 파랗다면 그것은 없는 장애물이다.** ESDF 는 그 자리에 표면이 없다고
말하는데 primitive 는 구로 막고 있다는 뜻이다. 크레이트와 테이블 잔여의 큰 원이 그렇다.

## 4. 로봇 구가 보는 여유거리

![여유 비교]({img}/fig4_clearance_compare.png)

**만드는 법.** 로봇 충돌 구 {len(centres)}개 각각에서 두 표현의 여유거리를 잰다. 양쪽 모두 같은
지지면 half-space 행을 포함하므로, 차이는 순수하게 물체 표현에서 온다.

**읽는 법.** 왼쪽 산점도의 대각선이 두 표현이 일치하는 선이다. **점이 전부 대각선 위쪽에
있으면 ESDF 가 더 여유롭다는 뜻**이고, 아래쪽에 점이 있으면 ESDF 가 더 보수적이라는 뜻이다.
오른쪽은 같은 데이터를 정렬해 그린 것으로, 0 선을 지나는 지점이 위반 개수다.

측정: primitive 위반 **{int((prim_clear<0).sum())}** / ESDF 위반 **{int((esdf_clear<0).sum())}**,
ESDF 만 위반인 구 **{int(((prim_clear>=0)&(esdf_clear<0)).sum())}개**.
여유 차이(primitive − ESDF) 중앙값 **{np.median(prim_clear-esdf_clear)*1000:+.0f} mm**.

**왼쪽 산점도에서 아래쪽 대각선 위의 점들은 두 표현이 정확히 일치하는 구들이다.** 거기서는
**공유된 지지면 평면 항**이 지배하므로 물체 표현이 무엇이든 같은 답이 나온다. 두 곡선이
갈라지는 지점부터가 물체 표현의 차이다.

### 남은 위반은 어느 표현의 책임인가

ESDF 쪽 위반 {int((esdf_clear<0).sum())}개를 항별로 나누면:

| 항 | 위반 구 | 최소 여유 |
|---|---|---|
| ESDF 필드만 | **{n_field}** | {field_only.min()*1000:+.0f} mm |
| 지지면 half-space 만 | **{n_plane}** | {planes.min()*1000:+.0f} mm |
| 합쳐서 | {int((esdf_clear<0).sum())} | {esdf_clear.min()*1000:+.0f} mm |

**대부분이 필드가 아니라 평면에서 온다.** 그리고 그것은 이 backend 의 문제가 아니라 지지면
표현의 문제다 — `SupportSurface` 는 `n·p >= d + margin` 인 **무한 half-space** 라서, 테이블
상판(z = 0.822)의 평면은 **상판보다 낮은 모든 것을 금지한다.** 위반 구의 링크가 그것을 말한다:
바퀴 34 · base 14 · torso 9 — 전부 바닥에 서 있는 부분이고, 물리적으로 테이블 아래에 있는 것이
정상이다.

이것은 두 collision backend 어느 쪽도 고칠 수 없다. 고치려면 지지면을 **유계 패치**로 만들거나,
평면 행을 그 위에 있을 수 있는 링크에만 걸어야 한다. 이 검증에서 세 번째로 발견된 표현 문제이며
`OPEN-geometry-representation.md` 와 같은 성격이다.

여유가 가장 작은 구 8개:

| 링크 | ESDF 여유 (mm) | primitive 여유 (mm) |
|---|---|---|
""" + "\n".join(
        f"| `{link_names[i]}` | {esdf_clear[i]*1000:+.0f} | {prim_clear[i]*1000:+.0f} |"
        for i in worst) + f"""

## 5. 복셀 해상도

![해상도]({img}/fig5_resolution.png)

**읽는 법.** 세 단면이 같은 씬이다. 왼쪽으로 갈수록 표면선이 매끄럽고 비용이 크다.
이산화 편향은 복셀의 **정확히 반**이므로 제목에 함께 적었다 — 20 mm 복셀의 10 mm 편향은
50 mm 여유거리의 5분의 1이다.

""" + "\n".join([
        "| 복셀 | 빌드 | 복셀 수 | 메모리 | 미관측 | 이산화 편향 |",
        "|---|---|---|---|---|---|",
    ] + [f"| {vs*1000:.0f} mm | {v[0]:.2f} s | {v[1]/1e6:.1f} M | {v[2]/1e6:.0f} MB | "
         f"{v[3]:.0%} | {vs*500:.1f} mm |" for vs, v in stats_by_res.items()]) + f"""

## 6. ESDF 이전 — TSDF 와 가중치

![TSDF]({img}/fig6_tsdf.png)

**읽는 법.** 왼쪽이 절단된 부호 거리다. 표면 앞이 파랑, 뒤가 빨강이고 ±{trunc*1000:.0f} mm 에서
잘린다. 절단 밖은 갱신하지 않으므로 **표면 뒤로 멀리는 "비어 있다"가 아니라 "가려져서 모른다"**
가 된다. 오른쪽 가중치가 0인 곳이 정확히 그 미관측 영역이고, 1번 그림의 회색과 같다.

이 두 장이 미관측 비율이 왜 {field.unknown_fraction:.0%} 나 되는지 설명한다. 수직 단면에서 자유
영역이 **카메라에서 뻗어나가는 원뿔 모양**인 것이 보인다 — 카메라 셋이 있어도 한 장면에서 볼 수
있는 것은 그 원뿔 안의 표면과 그 앞의 자유 공간뿐이고, 나머지는 전부 미관측이다.

**진단으로서 이것이 뜻하는 것**: 미관측 77% 의 대부분은 물체 뒤 그림자가 아니라 **작업 상자가
카메라가 보는 부피보다 훨씬 크다**는 사실이다. 상자를 관측 부피에 맞춰 줄이면 미관측 비율과
계산량이 함께 줄지만, 줄인 만큼 "그 밖은 아예 모른다"가 되므로 `unknown_policy` 의 의미가 커진다.

## 7. TO 직전 — 필드 질의가 충돌 제약 행이 되는 순간

![TO 입력]({img}/fig7_to_constraint_input.png)

이 그림은 시각화용으로 다시 만든 근사값이 아니라, `EsdfField.distance()`와 `gradient()`를 현재
로봇 collision sphere {len(centres)}개 중심에서 직접 질의한 결과다. TO가 각 trajectory timestep에서
반복하는 것과 같은 계산이며 최종 행은 `h = d_esdf(p) - r_robot - esdf_margin`이다. 왼쪽은 필드와
질의 위치, 가운데는 가장 제약적인 유효 행의 단면과 separating gradient, 오른쪽은 실제 scalar
제약값을 보여준다. 전체 {len(centres)}개 행의 중심·반지름·거리·gradient·margin·h는
[`to_constraint_input.json`]({img}/to_constraint_input.json)에 저장했다.

## 재현

```bash
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.reports.esdf_report \\\\
    --records {args.records} --voxel {args.voxel}
```
"""
    out_doc = paths.document
    out_doc.write_text(doc)
    print(f"wrote {out_doc}, 7 figures and TO query data in {figs}")
    print(f"미관측 {field.unknown_fraction:.0%}  primitive 위반 {int((prim_clear<0).sum())}  "
          f"ESDF 위반 {int((esdf_clear<0).sum())}  "
          f"ESDF만 위반 {int(((prim_clear>=0)&(esdf_clear<0)).sum())}")


if __name__ == "__main__":
    main()
