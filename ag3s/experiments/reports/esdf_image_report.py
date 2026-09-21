"""ESDF 를 관측 이미지 공간에서 본다 — 필드가 만든 기하와 거리장을 카메라 시점으로.

단면 그림은 정확하지만 사람이 보는 공간이 아니다. 여기서는 세 가지를 정책이 실제로 본 프레임
위에 놓는다.

* **분리** — 필드가 들고 있는 기하를 카메라로 투영해, obstacle / target(파냄) / 지지면(파냄) /
  미관측으로 칠한다. target 과 obstacle 이 분리되어 있다는 것이 픽셀 단위로 보인다.
* **레이마칭** — 각 픽셀의 광선을 따라 거리장이 0을 지나는 지점을 찾는다. **필드가 스스로
  재구성한 깊이 이미지**이고, 실제 깊이와의 차이가 곧 필드의 오차다.
* **자유 공간** — 각 픽셀의 표면에서 카메라 쪽으로 물러난 지점의 거리장. "보이는 것 앞에 얼마나
  여유가 있는가" 를 이미지로 답한다.

실행:
    MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.reports.esdf_image_report \
        --records outputs/.../ag3s_records/run_0002
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.fields.esdf import CameraDepth, EsdfBuilder, OCCUPIED, UNKNOWN
from benchmark.ag3s.experiments.common.outputs import add_tag_argument, resolve
from benchmark.ag3s.experiments.common.figstyle import (
    CATEGORICAL, GRID_INK, INK, INK_2, SURFACE, sequential_cmap, use_korean,
)
from benchmark.ag3s.experiments.sources.policy_record import CAMERA_BINDINGS, load_run, pose_scene, replay_scene

CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def rays(K, T_base_cam, hw):
    """각 픽셀의 base 프레임 광선. `(단위 방향, z=1 로 정규화된 방향, 원점)`.

    두 가지를 모두 돌려주는 이유가 있다. 구면 추적에는 **단위** 방향이 필요하고(보폭이 곧
    거리여야 하므로), 깊이에서 표면점을 만들 때는 **z 성분이 1인** 방향이 필요하다. 깊이 이미지가
    담는 것은 광선 길이가 아니라 광축 방향 z 이기 때문이다.

    단위 방향에 z-depth 를 곱하면 표면점이 카메라 쪽으로 `cos(광축과의 각)` 만큼 당겨진다.
    화면 중앙에서는 티가 안 나고 가장자리에서 커지므로, 이 실수는 "가장자리만 이상한" 그림으로
    나타난다. (첫 판이 그래서 표면 픽셀의 96%를 자유 공간으로 분류했다.)
    """
    h, w = hw
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    unit = d / np.linalg.norm(d, axis=-1, keepdims=True)
    R = T_base_cam[:3, :3]
    return unit @ R.T, d @ R.T, T_base_cam[:3, 3]


def raymarch(field, origin, direction, *, t_min=0.05, t_max=3.0, step_scale=0.9, iters=96):
    """구면 추적(sphere tracing). 거리장이 곧 안전 보폭이므로 고정 간격보다 훨씬 적게 돈다.

    표면에 닿았는지는 `d < 표면 판정 두께` 로 본다. 격자가 이산이라 정확한 0 교차는 존재하지
    않으므로, 복셀 반 칸을 두께로 쓴다 — 그것이 필드가 표현할 수 있는 가장 얇은 껍질이다.
    """
    shape = direction.shape[:2]
    t = np.full(shape, float(t_min))
    hit = np.zeros(shape, bool)
    eps = 0.5 * field.grid.voxel_size
    for _ in range(iters):
        live = (~hit) & (t < t_max)
        if not live.any():
            break
        p = origin + direction[live] * t[live][:, None]
        d = field.distance(p)
        newly = d < eps
        idx = np.nonzero(live)
        hit[idx[0][newly], idx[1][newly]] = True
        t[live] = t[live] + np.maximum(d * step_scale, 0.5 * eps)
    return np.where(hit, t, np.nan)


def main() -> None:
    import mujoco

    from benchmark.ag3s.experiments.reports.grounding_report import build_robot_model
    from benchmark.ag3s.experiments.sources.mujoco_source import gaussian_attention, is_robot_body
    from benchmark.ag3s.runtime.pipeline import AG3S

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--voxel", type=float, default=0.010)
    ap.add_argument("--camera", default="zed_left")
    ap.add_argument("--out-doc", default="benchmark/ag3s/docs/esdf-image-view.md")
    ap.add_argument("--out-figs", default="benchmark/ag3s/asset/image/esdf")
    add_tag_argument(ap)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    use_korean()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, TwoSlopeNorm

    run = load_run(args.records)
    scene = replay_scene(run)
    robot = build_robot_model(scene)
    pose_scene(scene, run.steps[args.frame])
    names = {i: (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
             for i in range(scene.model.nbody)}
    robot_ids = [i for i, n in names.items() if is_robot_body(n)]
    frames = {c: scene.capture(c) for c in CAMERAS}
    cam = frames[args.camera]
    attention = gaussian_attention(frames["zed_left"],
                                   scene.body_position_in_base(args.target))

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf", "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4},
    })
    ag = AG3S(cfg, robot_model=robot, constraint_robot_model=robot)
    head = frames["zed_left"]
    t0 = time.time()
    out = ag.process(depth=head.depth, camera_intrinsics=head.camera_intrinsics,
                     T_base_cam=head.T_base_cam, attention_map=attention,
                     robot_state=head.robot_state, phase="grasp")
    build_s = time.time() - t0
    field = out.esdf
    grid = field.grid

    # 분리를 픽셀로 보이려면 파냄 **이전** 점유가 필요하다. 파낸 뒤에는 target 이 그냥 사라져
    # 있어서 "어디가 파였는지" 를 말할 수 없다.
    raw = EsdfBuilder(cfg.esdf)
    raw.update([CameraDepth(c, frames[c].depth, frames[c].camera_intrinsics,
                            frames[c].T_base_cam,
                            robot_mask=np.isin(frames[c].body_ids, robot_ids)) for c in CAMERAS])
    occ_raw = raw.volume.occupancy(surface_band=cfg.esdf.surface_band)

    target_geo = out.target
    support_pts = None
    # 파이프라인이 무엇을 파냈는지 같은 규칙으로 다시 만든다.
    from benchmark.ag3s.stages.reconstruction import reconstruct
    from benchmark.ag3s.stages.robot_filter import filter_robot_points
    from benchmark.ag3s.stages.support_surface import fit_support_surfaces
    cloud, _ = reconstruct(depth=head.depth, camera_intrinsics=head.camera_intrinsics,
                           T_base_cam=head.T_base_cam, config=cfg.pointcloud)
    cloud, _ = filter_robot_points(cloud, robot, head.robot_state, cfg.pointcloud)
    _, smask = fit_support_surfaces(cloud, cfg.support_surface)
    support_pts = cloud.points[smask]

    def voxel_set(points, dilate: int = 1):
        """복셀 집합. **한 칸 부풀린다** — 부풀리지 않으면 라벨이 거의 붙지 않는다.

        target 과 지지면의 점들은 다운샘플·자기필터를 거친 클라우드의 *대표점*이라 원본 깊이의
        표면점과 최대 복셀 반 칸 어긋나 있다. 10 mm 복셀에서 그 어긋남이 이웃 복셀로 넘어가면
        같은 표면인데 라벨이 안 붙는다. (첫 판이 target 을 1 픽셀만 찾았다.) 부풀리는 것은
        **그림용 대응**일 뿐이고, 실제 파냄은 `carve` 가 별도 설정으로 한다.
        """
        m = np.zeros(grid.shape, bool)
        if points is None or len(points) == 0:
            return m
        idx, inside = grid.to_index(points)
        idx = idx[inside]
        m[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        if dilate > 0:
            from scipy import ndimage
            m = ndimage.binary_dilation(m, iterations=int(dilate))
        return m

    is_target = voxel_set(None if target_geo is None else target_geo.points)
    is_support = voxel_set(support_pts)

    # 0 없음 · 1 obstacle · 2 target · 3 지지면 · 4 미관측
    label = np.zeros(grid.shape, np.int8)
    label[occ_raw == OCCUPIED] = 1
    label[(occ_raw == OCCUPIED) & is_support] = 3
    label[(occ_raw == OCCUPIED) & is_target] = 2
    label[occ_raw == UNKNOWN] = 4

    K, T = cam.camera_intrinsics, cam.T_base_cam
    hw = cam.hw
    dirs, dirs_z, origin = rays(K, T, hw)

    paths = resolve(out_figs=args.out_figs, out_doc=args.out_doc,
                    tag=args.tag).prepare()
    figs, img = paths.figures, paths.image_prefix
    rgb = run.steps[args.frame].images["cam_high"]

    # ------------------------------------------------------------------ fig8 분리
    # 관측 깊이의 표면점을 그대로 라벨링한다. 레이마칭보다 정직하다 — 실제로 본 표면이 어떤
    # 라벨을 받았는지만 말하고, 필드가 만들어낸 표면을 섞지 않는다.
    depth = cam.depth
    valid = np.isfinite(depth) & (depth > cfg.pointcloud.depth_min) & (depth < cfg.pointcloud.depth_max)
    pts = origin + dirs_z * depth[..., None]   # z-depth 에는 z=1 방향을 곱한다
    lab_img = np.zeros(hw, np.int8)
    idx, inside = grid.to_index(pts.reshape(-1, 3))
    ok = inside & valid.reshape(-1)
    lab_flat = np.zeros(hw[0] * hw[1], np.int8)
    lab_flat[ok] = label[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    # 격자 밖은 "자유"가 아니라 **필드가 다루지 않는 곳**이다. 같은 색으로 칠하면 작업 상자가
    # 카메라가 보는 것보다 훨씬 작다는 사실이 그림에서 사라진다.
    lab_flat[valid.reshape(-1) & ~inside] = 5
    lab_img = lab_flat.reshape(hw)

    cmap = ListedColormap(["#e9e8e3", CATEGORICAL[2], CATEGORICAL[0], CATEGORICAL[3],
                           "#b9b8b2", "#6f6e69"])
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_color(GRID_INK)
    axes[0].imshow(rgb); axes[0].set_title("정책이 본 입력", fontsize=10, color=INK)
    axes[1].imshow(depth, cmap="magma_r")
    axes[1].set_title(f"관측 깊이 ({args.camera})", fontsize=10, color=INK)
    axes[2].imshow(np.zeros(hw + (3,)) + 0.98)
    axes[2].imshow(lab_img, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
    n = {k: int((lab_img == v).sum()) for k, v in
         (("obstacle", 1), ("target", 2), ("지지면", 3), ("미관측", 4), ("격자 밖", 5))}
    axes[2].set_title(f"필드가 만든 기하 — obstacle {n['obstacle']:,} · target {n['target']:,} · "
                      f"지지면 {n['지지면']:,} · 상자 밖 {n['격자 밖']:,}", fontsize=9.5, color=INK)
    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, mec="none", mfc=c, label=l)
               for c, l in ((CATEGORICAL[2], "obstacle (충돌 필드에 남음)"),
                            (CATEGORICAL[0], f"target `{args.target}` (필드에서 파냄)"),
                            (CATEGORICAL[3], "지지면 (평면 행으로 따로 나감)"),
                            ("#b9b8b2", "미관측"),
                            ("#6f6e69", "작업 상자 밖 (필드가 다루지 않음)"))]
    leg = fig.legend(handles=handles, frameon=False, fontsize=9, ncol=4, loc="lower center",
                     bbox_to_anchor=(0.5, -0.04))
    for t_ in leg.get_texts(): t_.set_color(INK_2)
    fig.suptitle(f"관측 이미지 위의 target / obstacle 분리 — phase `grasp`",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(figs / "fig8_image_separation.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ fig9 레이마칭
    t_hit_len = raymarch(field, origin, dirs)
    # 구면 추적이 돌려주는 것은 **광선 길이**다. 관측 깊이는 z-depth 이므로 같은 단위로 바꾼다.
    cos = 1.0 / np.linalg.norm(dirs_z, axis=-1)
    t_hit = t_hit_len * cos
    # 격자 안에 표면이 있는 픽셀만 비교한다. 상자 밖 표면은 필드에 아예 없으므로 광선이
    # 끝까지 가고, 그 차이를 오차로 세면 격자 크기를 재는 셈이 된다.
    in_grid = np.zeros(hw, bool)
    in_grid.reshape(-1)[inside & valid.reshape(-1)] = True
    diff = np.where(np.isfinite(t_hit) & in_grid, t_hit - depth, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_color(GRID_INK)
    vmin, vmax = np.nanpercentile(depth[in_grid], [1, 99])
    axes[0].imshow(np.where(in_grid, depth, np.nan), cmap="magma_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("관측 깊이 (작업 상자 안만)", fontsize=10, color=INK)
    im = axes[1].imshow(np.where(in_grid, t_hit, np.nan), cmap="magma_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("거리장을 구면 추적해 얻은 깊이", fontsize=10, color=INK)
    cb = fig.colorbar(im, ax=axes[1], fraction=.046, pad=.02)
    cb.set_label("m", color=INK_2, fontsize=8); cb.ax.tick_params(colors=INK_2, labelsize=7)
    lim = float(np.nanpercentile(np.abs(diff), 98)) if np.isfinite(diff).any() else 0.05
    im2 = axes[2].imshow(diff * 1000, cmap="RdBu",
                         norm=TwoSlopeNorm(vmin=-lim * 1000, vcenter=0.0, vmax=lim * 1000))
    finite = np.isfinite(diff)
    axes[2].set_title(f"차이 (필드 − 관측): 중앙 {np.nanmedian(diff)*1000:+.1f} mm, "
                      f"|차이| 95% {np.nanpercentile(np.abs(diff),95)*1000:.1f} mm",
                      fontsize=9.5, color=INK)
    cb = fig.colorbar(im2, ax=axes[2], fraction=.046, pad=.02)
    cb.set_label("mm", color=INK_2, fontsize=8); cb.ax.tick_params(colors=INK_2, labelsize=7)
    fig.suptitle("거리장이 스스로 재구성한 장면 — 흰 곳은 광선이 표면을 못 만난 곳(미관측)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figs / "fig9_raymarch.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ fig10 자유 공간
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_color(GRID_INK)
    axes[0].imshow(rgb); axes[0].set_title("정책이 본 입력", fontsize=10, color=INK)
    for ax, back in zip(axes[1:], (0.05, 0.15)):
        q = origin + dirs_z * np.maximum(depth - back, 0.0)[..., None]
        d = field.distance(q.reshape(-1, 3)).reshape(hw)
        d = np.where(in_grid, d, np.nan)
        im = ax.imshow(d * 1000, cmap=sequential_cmap(), vmin=0, vmax=200)
        ax.set_title(f"표면에서 카메라 쪽으로 {back*100:.0f} cm 물러난 지점의 거리장",
                     fontsize=9.5, color=INK)
        cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
        cb.set_label("mm", color=INK_2, fontsize=8); cb.ax.tick_params(colors=INK_2, labelsize=7)
    fig.suptitle("보이는 것 앞에 얼마나 여유가 있는가 — **밝을수록 표면에 가깝다** "
                 "(어두운 곳이 빈 공간)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figs / "fig10_free_space.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    scene.close()

    doc = f"""# ESDF 를 관측 이미지에서 보기

단면 그림({{'esdf-diagnostics.md'}})은 정확하지만 사람이 보는 공간이 아니다. 여기서는 필드가 만든
기하와 거리장을 **정책이 실제로 본 프레임 위에** 놓는다.

| | |
|---|---|
| 기록 | `{run.path}` 프레임 {args.frame} |
| 카메라 | `{args.camera}` |
| 복셀 | {args.voxel*1000:.0f} mm, 격자 {field.stats['grid_shape']} |
| phase | `grasp` — 접촉이 허가되므로 target 이 필드에서 파인다 |
| 빌드 | {build_s:.2f} s |

## 1. target / obstacle 분리

![분리]({img}/fig8_image_separation.png)

**만드는 법.** 관측 깊이의 표면점을 그대로 복셀 라벨에 대응시킨다. 레이마칭보다 정직하다 —
실제로 본 표면이 어떤 라벨을 받았는지만 말하고 필드가 만들어낸 표면을 섞지 않는다. 라벨은
파냄 **이전** 점유에서 만든다. 파낸 뒤에는 target 이 그냥 사라져 있어 "어디가 파였는지" 를
말할 수 없기 때문이다.

**읽는 법.** 초록이 충돌 필드에 남는 obstacle, 파랑이 `{args.target}` — **필드에서 파여 장애물로
세지 않는 부분**이다. 노랑은 지지면이고, 필드가 아니라 half-space 평면 행으로 따로 나간다.

픽셀 수: obstacle {n['obstacle']:,} · target {n['target']:,} · 지지면 {n['지지면']:,} ·
미관측 {n['미관측']:,}.

파냄이 **제거가 아니라 이관**이라는 점은 그대로다 — `{args.target}` 은
`CollisionConstraintSet.candidates` 에 여전히 후보로 있고, 필드에서만 빠진다.

## 2. 거리장이 스스로 재구성한 장면

![레이마칭]({img}/fig9_raymarch.png)

**만드는 법.** 픽셀마다 광선을 따라 구면 추적(sphere tracing)을 한다. 거리장이 곧 안전 보폭이라
고정 간격보다 훨씬 적게 돌고, 거리가 복셀 반 칸보다 작아지면 표면으로 본다 — 이산 격자가 표현할
수 있는 가장 얇은 껍질이 그것이다.

**읽는 법.** 가운데가 **필드만 가지고 만든 깊이 이미지**다. 오른쪽이 관측 깊이와의 차이이고,
그것이 곧 이 표현의 오차다. 중앙값 {np.nanmedian(diff)*1000:+.1f} mm, |차이| 95 백분위
{np.nanpercentile(np.abs(diff),95)*1000:.1f} mm.

흰 곳은 광선이 끝까지 표면을 만나지 못한 곳이다 — 그 방향은 **미관측**이라 필드에 아무것도 없다.
`unknown_policy: free` 가 뜻하는 바가 이 흰색이다.

**로봇 팔 영역이 크게 어긋나는 것은 정상이다.** 팔은 필드에서 마스크로 빠지므로 광선이 그
자리를 통과해 뒤의 표면까지 간다. 관측 깊이는 팔을, 필드는 팔 뒤를 말하니 차이가 클 수밖에 없다.

중앙값 {np.nanmedian(diff)*1000:+.1f} mm 가 **복셀 반 칸(-{args.voxel*500:.1f} mm)에 가까운 것**이
이 표현의 이산화 편향이 예측대로라는 뜻이다.

## 3. 보이는 것 앞의 여유

![자유 공간]({img}/fig10_free_space.png)

**만드는 법.** 픽셀마다 표면점에서 카메라 쪽으로 5 cm / 15 cm 물러난 지점의 거리장을 읽는다.

**읽는 법.** "이 픽셀이 보이는 표면 앞에 얼마나 빈 공간이 있는가" 다. **밝을수록 표면에
가깝고 어두울수록 여유가 크다.** 테이블 상판 바로 위가 밝고(붙어 있다), 크레이트 안쪽이
15 cm 물러난 그림에서 밝게 남는 것이 그 안이 좁다는 뜻이다.

**로봇 팔이 어둡게 나오는 것은 여유가 커서가 아니다.** 팔은 필드에서 마스크로 빠져 있어 그
자리에 아무 표면도 없고, 그래서 거리장이 크게 나온다. 물리적으로는 팔이 거기 있다 — 자기 충돌은
로봇 모델이 따로 다루고 이 필드의 일이 아니라는 뜻이며, 그림을 읽을 때 그 구분을 해야 한다.

## 재현

```bash
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.reports.esdf_image_report \\\\
    --records {args.records} --voxel {args.voxel}
```
"""
    out_doc = paths.document
    out_doc.write_text(doc)
    print(f"wrote {out_doc} and 3 figures in {figs}")
    print(f"픽셀 라벨: obstacle {n['obstacle']:,} target {n['target']:,} "
          f"지지면 {n['지지면']:,} 미관측 {n['미관측']:,}")
    print(f"레이마칭 오차: 중앙 {np.nanmedian(diff)*1000:+.1f}mm  "
          f"95% {np.nanpercentile(np.abs(diff),95)*1000:.1f}mm")


if __name__ == "__main__":
    main()
