"""Observation 기반 TSDF/ESDF를 AG3S step 1--6 및 표준 방식으로 시각화한다.

두 그림을 만든다.

* ``fig1_steps_1_to_6.png``: 같은 multi-camera depth에서 얻은 SDF를 기하 배경으로 두고,
  attention -> backprojection -> lifting -> grounding -> separation -> TO query가 누적되는 과정.
* ``fig2_standard_sdf_views.png``: depth image, signed TSDF slice, TSDF zero-level set,
  ESDF heatmap/iso-contour, ESDF gradient quiver, robot-sphere distance query. 로보틱스에서
  TSDF/ESDF를 점검할 때 대표적으로 쓰는 표현들이다.

TSDF/ESDF는 attention이나 target label에서 만들어지는 것이 아니다. step 1--6과 같은 depth
observation을 공유하는 병렬 기하 branch이며, semantic stage가 진행돼도 field 값은 바뀌지 않는다.
target/support carving만 최종 collision ESDF를 만들 때 적용된다.

실행::

    MPLCONFIGDIR=/tmp/ag3s-matplotlib MUJOCO_GL=osmesa \
      src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.observation_sdf_report \
      --records outputs/rby1_atomic_infer/ag3s_step1/ag3s_records/run_0002
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.experiments.esdf_report import extent_for, slice_index
from benchmark.ag3s.experiments.outputs import add_tag_argument, resolve
from benchmark.ag3s.experiments.figstyle import (
    CATEGORICAL, GRID_INK, INK, INK_2, SURFACE, style_axes, use_korean,
)
from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene
from benchmark.ag3s.types import CameraID, CameraObservation, SourceType


CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
CAMERA_IDS = {
    "zed_left": CameraID.HEAD,
    "wrist_cam_l": CameraID.LEFT_WRIST,
    "wrist_cam_r": CameraID.RIGHT_WRIST,
}
POLICY_IMAGES = {
    "zed_left": "cam_high",
    "wrist_cam_l": "cam_left_wrist",
    "wrist_cam_r": "cam_right_wrist",
}


def _thin(points: np.ndarray, limit: int = 6000) -> np.ndarray:
    points = np.asarray(points)
    if len(points) <= limit:
        return points
    return points[np.linspace(0, len(points) - 1, limit).astype(np.int64)]


def _thin_with_values(points: np.ndarray, values: np.ndarray, limit: int = 9000):
    if len(points) <= limit:
        return np.asarray(points), np.asarray(values)
    idx = np.linspace(0, len(points) - 1, limit).astype(np.int64)
    return np.asarray(points)[idx], np.asarray(values)[idx]


def _show(ax, image: np.ndarray, extent, **kwargs):
    return ax.imshow(np.asarray(image).T, origin="lower", extent=extent, aspect="equal", **kwargs)


def _scene_axes(fig, position: int, title: str, view_bounds):
    ax = fig.add_subplot(2, 3, position, projection="3d")
    ax.set_title(title, fontsize=9.5, color=INK, pad=8)
    ax.set_xlabel("base x", fontsize=7)
    ax.set_ylabel("base y", fontsize=7)
    ax.set_zlabel("base z", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlim(*view_bounds[0]); ax.set_ylim(*view_bounds[1]); ax.set_zlim(*view_bounds[2])
    ax.view_init(elev=25, azim=-63)
    return ax


def _scatter_scene(ax, points, *, color="#6b7280", size=.35, alpha=.35, limit=6000):
    p = _thin(np.asarray(points), limit)
    if len(p):
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=size, c=color, alpha=alpha,
                   depthshade=False, linewidths=0)
    return p


def _grid_points(grid, mask: np.ndarray, limit: int = 8000) -> np.ndarray:
    ijk = np.argwhere(mask)
    ijk = _thin(ijk, limit)
    return grid.origin + ijk * grid.voxel_size


def _scene_bounds(scene):
    names = ("crate", "apple", "banana", "orange", "pear")
    p = np.asarray([scene.body_position_in_base(name) for name in names])
    return (
        (float(p[:, 0].min() - .45), float(p[:, 0].max() + .45)),
        (float(p[:, 1].min() - .48), float(p[:, 1].max() + .48)),
        (0.66, 1.18),
    )


def _build(scene, run, frame_index: int, voxel: float, target_name: str):
    from benchmark.ag3s.experiments.grounding_report import build_robot_model
    from benchmark.ag3s.experiments.mujoco_source import gaussian_attention
    from benchmark.ag3s.pipeline import AG3S

    step = run.steps[frame_index]
    pose_scene(scene, step)
    frames = {name: scene.capture(name) for name in CAMERAS}
    attention = gaussian_attention(frames["zed_left"], scene.body_position_in_base(target_name))

    observations = []
    for k, name in enumerate(CAMERAS):
        f = frames[name]
        observations.append(CameraObservation(
            camera_id=CAMERA_IDS[name],
            depth=f.depth,
            camera_intrinsics=f.camera_intrinsics,
            T_base_cam=f.T_base_cam,
            robot_state=f.robot_state,
            timestamp=.015 * k,
            attention_map=attention if name == "zed_left" else None,
            image_hw=f.hw,
        ))

    cfg = AG3SConfig.from_dict({
        "collision_backend": "both",
        "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": voxel, "max_distance": .4},
    })
    robot = build_robot_model(scene)
    ag = AG3S(cfg, robot_model=robot, constraint_robot_model=robot)
    result, debug = ag.process_multi_debug(observations, phase="approach")
    if result.esdf is None or ag._esdf_builder is None:
        raise RuntimeError("multi-camera observation에서 ESDF가 만들어지지 않았다")
    return step, frames, attention, robot, ag, result, debug


def _stage_figure(path, *, step, attention, field, volume, result, debug, robot,
                  view_bounds, margin: float):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    grid = field.grid
    tsdf_sites = (volume.weight > 0.0) & (np.abs(volume.tsdf) <= grid.voxel_size)
    tsdf_surface = _grid_points(grid, tsdf_sites, 9000)
    esdf_surface = _grid_points(grid, field.distance_grid < 0.0, 9000)
    raw = debug["raw_cloud"]
    cloud = debug["filtered_cloud"]
    attention_cloud = debug["attention_cloud"]
    target = debug["grounding"].target

    fig = plt.figure(figsize=(17.2, 10.0), dpi=170)
    fig.patch.set_facecolor(SURFACE)

    # Step 1 -- policy image + attention. Attention names the target; it never deletes geometry.
    ax = fig.add_subplot(2, 3, 1)
    style_axes(fig, ax)
    image = step.images["cam_high"]
    ax.imshow(image)
    att = ax.imshow(attention, cmap="inferno", alpha=.58,
                   extent=(0, image.shape[1], image.shape[0], 0), interpolation="bilinear")
    ax.set_title("Step 1 · 2D attention\n색은 target 이름표, depth geometry는 그대로", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(att, ax=ax, fraction=.045, pad=.02)
    cb.set_label("attention", fontsize=8); cb.ax.tick_params(labelsize=7)

    # Step 2 -- observation backprojection and TSDF zero-crossing samples.
    ax = _scene_axes(fig, 2, "Step 2 · backprojection → TSDF\n회색=관측점, 청록=TSDF zero-crossing", view_bounds)
    _scatter_scene(ax, raw.points, color="#9ca3af", alpha=.18, limit=8500)
    _scatter_scene(ax, tsdf_surface, color="#00a6a6", size=1.0, alpha=.72, limit=7000)

    # Step 3 -- attention is lifted onto all fused 3D points; the TSDF surface remains unchanged.
    ax = _scene_axes(fig, 3, "Step 3 · attention lifting\n모든 3D 점 유지 + attention 색", view_bounds)
    p, a = _thin_with_values(attention_cloud.points, attention_cloud.attention, 9000)
    sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=.6, c=a, cmap="inferno",
                    vmin=0.0, vmax=1.0, depthshade=False, linewidths=0)
    fig.colorbar(sc, ax=ax, fraction=.025, pad=.01, shrink=.62)

    # Step 4 -- selected target on the final collision field zero band.
    ax = _scene_axes(fig, 4, "Step 4 · 3D grounding\n회색=ESDF obstacle surface, 빨강=target", view_bounds)
    _scatter_scene(ax, esdf_surface, color="#72777d", alpha=.18, limit=8000)
    if target is not None:
        _scatter_scene(ax, target.points, color=CATEGORICAL[7], size=3.0, alpha=.95, limit=3000)
        ax.scatter(*target.centroid, c=INK, s=45, marker="+")

    # Step 5 -- semantic candidate partition over the same ESDF geometry.
    ax = _scene_axes(fig, 5, "Step 5 · target / obstacle separation\n색=semantic candidate, 회색=ESDF surface", view_bounds)
    _scatter_scene(ax, esdf_surface, color="#a5a8ac", alpha=.12, limit=6000)
    for candidate in result.candidates:
        if candidate.source_type is SourceType.SUPPORT_SURFACE or not len(candidate.point_indices):
            continue
        pts = cloud.points[candidate.point_indices]
        color = (CATEGORICAL[7] if candidate.source_type is SourceType.TARGET
                 else CATEGORICAL[0] if candidate.source_type is SourceType.OBJECT
                 else CATEGORICAL[3])
        _scatter_scene(ax, pts, color=color, size=1.3, alpha=.72, limit=1800)

    # Step 6 -- what the ESDF branch sends to collision linearization at q_now.
    q = result.robot_state
    centres, radii = robot.sphere_centers_numeric(q)
    distance = field.distance(centres)
    gradient = field.gradient(centres)
    h = distance - radii - margin
    ax = _scene_axes(fig, 6, "Step 6 · TO-ready geometry\nrobot sphere 색=h, 화살표=$\\nabla d$", view_bounds)
    _scatter_scene(ax, esdf_surface, color="#6b7075", alpha=.14, limit=7000)
    norm = TwoSlopeNorm(vmin=-.10, vcenter=0.0, vmax=.20)
    sc = ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2], c=h, cmap="RdBu", norm=norm,
                    s=10 + 700 * radii ** 2, depthshade=False, linewidths=0)
    order = [int(i) for i in np.argsort(h) if np.linalg.norm(gradient[i]) > 1e-6][:14]
    if order:
        ii = np.asarray(order)
        gg = gradient[ii] / np.maximum(np.linalg.norm(gradient[ii], axis=1, keepdims=True), 1e-12)
        ax.quiver(centres[ii, 0], centres[ii, 1], centres[ii, 2],
                  gg[:, 0], gg[:, 1], gg[:, 2], length=.08, normalize=True,
                  color=CATEGORICAL[3], linewidth=1.0)
    cb = fig.colorbar(sc, ax=ax, fraction=.025, pad=.01, shrink=.62)
    cb.set_label("$h=d-r-m$ (m)", fontsize=8); cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Observation 기반 TSDF/ESDF 위에서 본 AG3S Step 1–6\n"
        "SDF는 같은 depth에서 만들어지는 공통 기하 branch이고, 단계가 진행되며 의미 정보가 겹쳐진다",
        fontsize=13, color=INK, x=.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, .925))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return centres, radii, distance, gradient, h, tsdf_surface, esdf_surface


def _standard_figure(path, *, step, frames, field, volume, centres, radii, h,
                     tsdf_surface, esdf_surface, z_slice: float, y_slice: float):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    grid = field.grid
    yi = slice_index(grid, 1, y_slice)
    zi = slice_index(grid, 2, z_slice)
    ext_xz = extent_for(grid, (0, 2))
    ext_xy = extent_for(grid, (0, 1))
    trunc = volume.truncation

    fig = plt.figure(figsize=(17.2, 10.2), dpi=170)
    fig.patch.set_facecolor(SURFACE)

    # 1. Sensor input: depth maps are the actual measurements fused into TSDF.
    ax = fig.add_subplot(2, 3, 1)
    style_axes(fig, ax)
    depth = frames["zed_left"].depth
    valid = np.isfinite(depth) & (depth > 0)
    dshow = np.where(valid, depth, np.nan)
    im = ax.imshow(dshow, cmap="turbo", vmin=np.nanpercentile(dshow, 2),
                   vmax=np.nanpercentile(dshow, 98))
    ax.set_title("① Depth observation\nTSDF가 적분하는 metric depth", fontsize=9.5)
    ax.set_xlabel("u (pixel)"); ax.set_ylabel("v (pixel)")
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.02)
    cb.set_label("depth (m)", fontsize=8); cb.ax.tick_params(labelsize=7)

    # 2. Standard TSDF signed slice: diverging colours and the zero crossing.
    ax = fig.add_subplot(2, 3, 2)
    style_axes(fig, ax)
    t = np.where(volume.weight[:, yi, :] > 0.0, volume.tsdf[:, yi, :], np.nan)
    im = _show(ax, t, ext_xz, cmap="RdBu",
               norm=TwoSlopeNorm(vmin=-trunc, vcenter=0.0, vmax=trunc))
    xx = np.linspace(ext_xz[0], ext_xz[1], t.shape[0])
    zz = np.linspace(ext_xz[2], ext_xz[3], t.shape[1])
    ax.contour(xx, zz, t.T, levels=[0.0], colors=[INK], linewidths=1.2)
    ax.set_title(f"② TSDF signed slice · y={y_slice:.2f} m\n검정=zero crossing, 흰색=unobserved",
                 fontsize=9.5)
    ax.set_xlabel("base x (m)"); ax.set_ylabel("base z (m)")
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.set_label("projective TSDF (m)", fontsize=8); cb.ax.tick_params(labelsize=7)

    # 3. Standard TSDF surface view. Without Open3D/skimage we display zero-band samples, the
    # discrete equivalent of the zero-level mesh sites consumed by marching cubes.
    ax = fig.add_subplot(2, 3, 3, projection="3d")
    style_axes(fig, ax)
    p = _thin(tsdf_surface, 12000)
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=.5, c=p[:, 2], cmap="viridis",
               depthshade=False, linewidths=0)
    ax.set_xlim(-.3, 1.2); ax.set_ylim(-.9, .9); ax.set_zlim(0, 1.6)
    ax.view_init(elev=24, azim=-62)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("③ TSDF zero-level surface samples\n|TSDF| ≤ 1 voxel (mesh site 표현)", fontsize=9.5)

    # 4. Standard ESDF planning slice: distance heatmap and metric iso-distance contours.
    ax = fig.add_subplot(2, 3, 4)
    style_axes(fig, ax)
    d = field.distance_grid[:, :, zi]
    im = _show(ax, d, ext_xy, cmap="RdBu",
               norm=TwoSlopeNorm(vmin=-.10, vcenter=0.0, vmax=.4))
    xx = np.linspace(ext_xy[0], ext_xy[1], d.shape[0])
    yy = np.linspace(ext_xy[2], ext_xy[3], d.shape[1])
    levels = [0.0, .05, .10, .20]
    cs = ax.contour(xx, yy, d.T, levels=levels,
                    colors=[INK, CATEGORICAL[7], CATEGORICAL[3], CATEGORICAL[0]],
                    linewidths=[1.4, 1.1, 1.0, .9])
    ax.clabel(cs, inline=True, fontsize=7, fmt=lambda x: f"{x*100:.0f} cm")
    ax.set_title(f"④ ESDF slice + iso-distance · z={z_slice:.2f} m\n경로계획에서 가장 흔한 표현",
                 fontsize=9.5)
    ax.set_xlabel("base x (m)"); ax.set_ylabel("base y (m)")
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.set_label("Euclidean distance (m)", fontsize=8); cb.ax.tick_params(labelsize=7)

    # 5. Standard ESDF gradient field on the same slice.
    ax = fig.add_subplot(2, 3, 5)
    style_axes(fig, ax)
    _show(ax, d, ext_xy, cmap="Greys", vmin=-.05, vmax=.30, alpha=.72)
    stride = 8
    gx, gy = np.gradient(d.astype(np.float64), grid.voxel_size, edge_order=1)
    X, Y = np.meshgrid(xx[::stride], yy[::stride], indexing="ij")
    U, V = gx[::stride, ::stride], gy[::stride, ::stride]
    mag = np.sqrt(U * U + V * V)
    valid_q = (mag > .05) & (d[::stride, ::stride] < .28)
    ax.quiver(X[valid_q], Y[valid_q], U[valid_q], V[valid_q], color=CATEGORICAL[3],
              angles="xy", scale_units="xy", scale=12, width=.003)
    ax.contour(xx, yy, d.T, levels=[0.0], colors=[INK], linewidths=1.2)
    ax.set_title("⑤ ESDF gradient vector field\n화살표=$\\nabla d$: obstacle에서 멀어지는 방향", fontsize=9.5)
    ax.set_xlabel("base x (m)"); ax.set_ylabel("base y (m)")

    # 6. Robot queries: standard downstream use of ESDF for sphere collision checks.
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    style_axes(fig, ax)
    surf = _thin(esdf_surface, 9000)
    ax.scatter(surf[:, 0], surf[:, 1], surf[:, 2], s=.35, c="#73777c", alpha=.16,
               depthshade=False, linewidths=0)
    norm = TwoSlopeNorm(vmin=-.10, vcenter=0.0, vmax=.20)
    sc = ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2], c=h, cmap="RdBu", norm=norm,
                    s=10 + 700 * radii ** 2, depthshade=False, linewidths=0)
    ax.set_xlim(-.3, 1.2); ax.set_ylim(-.9, .9); ax.set_zlim(0, 1.6)
    ax.view_init(elev=24, azim=-62)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("⑥ Robot-sphere ESDF query\n색=$d(p)-r-margin$", fontsize=9.5)
    cb = fig.colorbar(sc, ax=ax, fraction=.025, pad=.01, shrink=.68)
    cb.set_label("clearance h (m)", fontsize=8); cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        "대표적인 TSDF / ESDF 시각화 — 실제 RB-Y1 multi-camera observations",
        fontsize=13, color=INK, x=.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    use_korean()

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--voxel", type=float, default=.010)
    ap.add_argument("--z-slice", type=float, default=.87)
    ap.add_argument("--y-slice", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=.05)
    ap.add_argument("--out", default="benchmark/ag3s/asset/image/observation_sdf")
    ap.add_argument("--out-doc", default="benchmark/ag3s/docs/observation-sdf-visualization.md")
    add_tag_argument(ap)
    args = ap.parse_args()

    run = load_run(args.records)
    scene = replay_scene(run)
    try:
        step, frames, attention, robot, ag, result, debug = _build(
            scene, run, args.frame, args.voxel, args.target)
        field = result.esdf
        volume = ag._esdf_builder.volume
        paths = resolve(out_figs=args.out, out_doc=args.out_doc, tag=args.tag).prepare()
        out, img = paths.figures, paths.image_prefix
        bounds = _scene_bounds(scene)

        values = _stage_figure(
            out / "fig1_steps_1_to_6.png", step=step, attention=attention,
            field=field, volume=volume, result=result, debug=debug, robot=robot,
            view_bounds=bounds, margin=args.margin,
        )
        centres, radii, distance, gradient, h, tsdf_surface, esdf_surface = values
        _standard_figure(
            out / "fig2_standard_sdf_views.png", step=step, frames=frames,
            field=field, volume=volume, centres=centres, radii=radii, h=h,
            tsdf_surface=tsdf_surface, esdf_surface=esdf_surface,
            z_slice=args.z_slice, y_slice=args.y_slice,
        )

        target = debug["grounding"].target
        payload = {
            "record": str(run.path),
            "frame": args.frame,
            "cameras": list(CAMERAS),
            "target": None if target is None else {
                "id": target.id,
                "centroid_m": target.centroid.tolist(),
                "confidence": target.confidence,
                "n_points": len(target.points),
            },
            "field": {
                "grid_shape": list(field.grid.shape),
                "voxel_size_m": field.grid.voxel_size,
                "truncation_m": volume.truncation,
                "max_distance_m": field.max_distance,
                "unknown_fraction": field.unknown_fraction,
                "n_tsdf_zero_band": len(tsdf_surface),
                "n_esdf_occupied_samples": len(esdf_surface),
            },
            "to_query": {
                "n_robot_spheres": len(centres),
                "margin_m": args.margin,
                "n_violated": int((h < 0).sum()),
                "minimum_h_m": float(h.min()),
            },
        }
        (out / "summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n")

        doc = f"""# Observation 기반 TSDF/ESDF 시각화

| 항목 | 값 |
|---|---|
| 기록 | `{run.path}`, frame {args.frame} |
| 카메라 | `{', '.join(CAMERAS)}` |
| field | `{field.grid.shape}`, {field.grid.voxel_size*1000:.0f} mm voxel |
| TSDF truncation | ±{volume.truncation*1000:.0f} mm |
| ESDF max distance | {field.max_distance:.2f} m |
| 미관측 | {field.unknown_fraction:.1%} |
| robot sphere query | {len(centres)}개, h<0: {int((h<0).sum())}개 |

## AG3S Step 1–6와 SDF

![step 1-6]({img}/fig1_steps_1_to_6.png)

TSDF/ESDF는 attention에서 생성되는 것이 아니라 세 카메라의 depth observation에서 만들어지는
공통 기하 branch다. 따라서 step마다 field를 새로 계산한 것처럼 그리지 않고, 동일 field 위에 각
단계가 만드는 의미 정보가 누적되는 모습을 표시했다. Step 6 패널의 색이 최종
`h=d_esdf(p)-r_robot-margin`이다.

## 대표적인 TSDF/ESDF 표현

![standard views]({img}/fig2_standard_sdf_views.png)

1. metric depth image
2. signed TSDF slice와 zero crossing
3. TSDF zero-level surface samples (`|TSDF| <= 1 voxel`)
4. ESDF distance heatmap과 metric iso-distance contours
5. ESDF gradient vector field
6. robot collision-sphere ESDF query

현재 환경에는 Open3D/scikit-image가 없으므로 3번은 marching-cubes triangle mesh 대신 같은
zero-level set을 이루는 voxel samples로 표시했다. 계산에 쓰인 TSDF/ESDF 값은 동일하다.

수치 요약은 [`summary.json`]({img}/summary.json)에 있다.
"""
        doc_path = paths.document
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(doc)
        print(f"wrote {out / 'fig1_steps_1_to_6.png'}")
        print(f"wrote {out / 'fig2_standard_sdf_views.png'}")
        print(f"wrote {out / 'summary.json'} and {doc_path}")
        print(f"target={None if target is None else target.id}  unknown={field.unknown_fraction:.1%}  "
              f"ESDF violations={int((h < 0).sum())}/{len(h)}")
    finally:
        scene.close()


if __name__ == "__main__":
    main()

