"""2단계 검증 — 깊이에서 나온 3D 점이 실제로 물체가 있는 곳에 놓이는가.

1단계는 attention이 옳은 물체를 가리키는지 확인했다. 그 이름표가 붙는 대상은 3D 점들이고,
그 점들이 틀린 곳에 있으면 이후 모든 단계 — 클러스터링, primitive 근사, 여유거리 — 가 틀린
기하 위에서 정확하게 계산될 뿐이다. 그래서 여기서는 점 자체를 잰다.

네 가지를 서로 다른 실패 방식에 대응시켜 잰다. 한 가지만 재면 어느 것이 깨졌는지 알 수 없다.

**A. 재투영 왕복** — 역투영한 점을 같은 `K`, `T`로 다시 투영해 원래 픽셀로 돌아오는가.
산술만 검사하며 규약(convention)은 검사하지 않는다. A가 통과하고 B가 실패하면 규약이 틀린
것이고, A가 실패하면 식 자체가 틀린 것이다. 이 둘을 구분하지 못하면 디버깅이 추측이 된다.

**B. 지지면 평면** — 테이블 픽셀만 역투영해 평면을 맞춘다. 법선이 base 프레임의 +z와
이루는 각, 높이와 실제 테이블 상면 높이의 차, 그리고 잔차의 퍼짐. 카메라 회전이 조금이라도
틀어져 있으면 평면이 기울고, 이것이 가장 민감하게 드러나는 곳이다.

**C. 물체 표면 오차** — 각 물체의 점에서 그 물체의 *실제* 표면까지의 거리. 정답 표면은
MuJoCo 메시 정점을 실제 자세로 변환한 볼록 껍질이다. 중심 좌표와 비교하지 않는 이유는
카메라가 앞면만 보기 때문이다 — 보이는 점들의 무게중심은 카메라 쪽으로 반지름만큼 치우쳐
있으므로, 중심과의 거리는 오차가 없어도 0이 되지 않는다.

**D. 카메라 간 일치** — 같은 물체를 head와 손목 카메라에서 각각 역투영했을 때 같은 자리에
오는가. 정답이 필요 없는 검사이면서, 손목 카메라의 외부 파라미터가 FK를 타고 오기 때문에
`T_base_cam(t) = FK(q, mount_link) · T_link_cam` 사슬 전체를 검사한다. 마운트 링크가 틀렸거나
`T_link_cam`이 틀리면 B와 C는 head 카메라만으로 통과하면서 여기서만 벌어진다.

실행:
    MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.backprojection_report \
        --records outputs/.../ag3s_records/run_0002
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Optional, Sequence

import numpy as np

from benchmark.ag3s.config import PointCloudConfig
from benchmark.ag3s.experiments.outputs import add_tag_argument, resolve
from benchmark.ag3s.experiments.figstyle import (
    CATEGORICAL, GRID_INK, INK, INK_2, SURFACE, sequential_cmap, style_axes, use_korean,
)
from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene
from benchmark.ag3s.reconstruction import backproject

#: 표면 오차를 잴 물체. 테이블과 선반은 평면 검사(B)가 따로 담당한다.
OBJECT_BODIES = ("crate", "apple", "banana", "orange", "pear")

#: 각 물체가 의미 있는 통계를 내려면 최소한 이만큼의 점이 필요하다.
MIN_POINTS = 30


def hull_distance(points: np.ndarray, vertices: np.ndarray) -> Optional[np.ndarray]:
    """점에서 `vertices`의 볼록 껍질 표면까지의 부호 없는 거리. 볼록한 물체(과일)에만 쓴다."""
    from scipy.spatial import ConvexHull, QhullError

    if len(vertices) < 4:
        return None
    try:
        hull = ConvexHull(np.asarray(vertices, np.float64))
    except QhullError:
        return None
    eq = hull.equations  # normal.p + offset <= 0 이 안쪽
    signed = points @ eq[:, :3].T + eq[:, 3]
    outside = signed.max(axis=1)
    return np.where(outside > 0.0, outside, -outside)


def _box_surface_distance(points: np.ndarray, T: np.ndarray, half: np.ndarray) -> np.ndarray:
    """점에서 하나의 상자 표면까지의 정확한 거리. `T`는 상자 → base."""
    local = (points - T[:3, 3]) @ T[:3, :3]
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = -q.max(axis=1)          # 안쪽일 때 가장 가까운 면까지
    return np.where(q.max(axis=1) > 0.0, outside, inside)


def surface_distance(points, scene, body_name: str, T_base_world: np.ndarray,
                     mesh_vertices: np.ndarray) -> Optional[np.ndarray]:
    """점에서 물체의 **실제 표면**까지의 거리. geom 종류에 따라 방식을 나눈다.

    볼록 껍질을 모든 물체에 쓸 수는 없다. 크레이트는 상자 36개로 만든 **속이 빈** 통이라,
    껍질은 열린 입구를 덮어 내부를 통째로 안쪽으로 만들어 버린다. 그러면 안쪽 벽에 찍힌 점이
    껍질 표면에서 크레이트 반너비만큼 떨어진 것으로 계산되어, 역투영이 완벽해도 수십 mm의
    가짜 오차가 보고된다.

    그래서 상자 geom에는 점-상자 표면 거리를 정확히 계산해 geom들의 최솟값을 취하고
    (속이 빈 형상이 자연스럽게 처리된다), 메시 geom에는 볼록 껍질을 쓴다 (과일은 볼록하다).
    한 물체가 둘 다 가지면 두 결과의 최솟값을 쓴다.
    """
    mujoco = scene.mujoco
    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    best = None
    R_bw, t_bw = T_base_world[:3, :3], T_base_world[:3, 3]
    for gid in range(scene.model.ngeom):
        if scene.model.geom_bodyid[gid] != bid:
            continue
        if int(scene.model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        T = np.eye(4)
        T[:3, :3] = R_bw @ scene.data.geom_xmat[gid].reshape(3, 3)
        T[:3, 3] = R_bw @ scene.data.geom_xpos[gid] + t_bw
        d = np.abs(_box_surface_distance(points, T, np.asarray(scene.model.geom_size[gid], float)))
        best = d if best is None else np.minimum(best, d)
    if len(mesh_vertices):
        d = hull_distance(points, hull_vertices(scene, body_name, mesh_vertices, T_base_world))
        if d is not None:
            best = d if best is None else np.minimum(best, d)
    return best


def hull_vertices(scene, name, verts_local, T_base_world) -> np.ndarray:
    """물체의 메시 정점을 그 프레임의 실제 자세로 옮겨 base 프레임에 놓는다."""
    T_bw = scene.body_pose(name)
    v_world = verts_local @ T_bw[:3, :3].T + T_bw[:3, 3]
    return v_world @ T_base_world[:3, :3].T + T_base_world[:3, 3]


def fit_plane(points: np.ndarray) -> Optional[tuple[np.ndarray, float, np.ndarray, int]]:
    """상판 윗면 하나를 뽑아낸다. `(단위법선, 오프셋, 잔차, 내점 수)`.

    전체 점에 최소제곱을 그냥 걸면 안 된다. 상판은 두께 40 mm의 box이므로 머리 카메라가
    내려다볼 때 **옆면**도 함께 보이고, 그 점들은 윗면보다 최대 40 mm 아래에 있으면서
    수직이다. 이들을 포함해 맞추면 평면이 0.8° 기울고 9 mm 내려앉는다 — 역투영은 완벽한데
    검사가 실패한다. (이 보고서의 두 번째 판이 정확히 그 숫자를 보고했다.)

    그래서 AG3S 자신의 RANSAC 평면 추정기를 쓴다. 지배적인 평면(점 수가 압도적인 윗면)을
    고르고 옆면을 이상점으로 버리며, 무엇보다 **7단계 support surface 추출이 실제로 쓰는 바로
    그 코드**다. 여기서 통과한다는 것은 그 단계가 같은 씬에서 옳은 평면을 잡는다는 뜻이기도 하다.
    """
    from benchmark.ag3s.support_surface import fit_plane_ransac

    normal, offset, inliers = fit_plane_ransac(points, seed=0)
    if normal is None or len(inliers) < 100:
        return None
    if normal[2] < 0:
        normal, offset = -normal, -offset
    residual = points[inliers] @ normal - offset
    return normal, float(offset), residual, int(len(inliers))


def tabletop_geom(scene) -> tuple[Optional[int], Optional[float]]:
    """상판 geom의 id와 그 윗면의 실제 높이 (base 프레임).

    `table`은 **한 개의 body이지만 다섯 개의 geom**이다 — 상판 하나와 다리 넷. body 라벨로
    점을 고르면 수직인 다리까지 들어와 평면 맞춤이 기울어지고, 그러면 역투영이 완벽해도
    검사가 실패한다. 그래서 geom 단위로 고른다. 상판은 윗면이 가장 높은 geom으로 식별하며,
    높이는 geom 자체에서 읽지 추정하지 않는다.
    """
    mujoco = scene.mujoco
    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "table")
    if bid < 0:
        return None, None
    T_base_world = np.linalg.inv(scene.body_pose("base"))
    best = (None, -np.inf)
    for gid in range(scene.model.ngeom):
        if scene.model.geom_bodyid[gid] != bid:
            continue
        p_base = (T_base_world[:3, :3] @ scene.data.geom_xpos[gid]) + T_base_world[:3, 3]
        top = float(p_base[2] + scene.model.geom_size[gid][2])
        if top > best[1]:
            best = (int(gid), top)
    return best[0], (None if best[0] is None else best[1])


def analyse_frame(scene, camera: str, config: PointCloudConfig):
    """한 프레임, 한 카메라: 역투영 결과와 정답 라벨."""
    frame = scene.capture(camera)
    cloud = backproject(frame.depth, frame.camera_intrinsics, frame.T_base_cam, config)
    if cloud.uv is None or len(cloud) == 0:
        return frame, cloud, np.zeros(0, np.int32)
    labels = frame.body_ids[cloud.uv[:, 1], cloud.uv[:, 0]]
    return frame, cloud, labels


def reprojection_error(cloud, frame) -> np.ndarray:
    """A: 점을 다시 픽셀로 투영했을 때의 픽셀 오차."""
    T = frame.T_base_cam
    K = frame.camera_intrinsics
    p_cam = (cloud.points - T[:3, 3]) @ T[:3, :3]
    uv = np.stack([K[0, 0] * p_cam[:, 0] / p_cam[:, 2] + K[0, 2],
                   K[1, 1] * p_cam[:, 1] / p_cam[:, 2] + K[1, 2]], axis=1)
    # backproject 는 픽셀 중심을 정수 인덱스로 쓰므로 비교 대상도 정수 좌표다.
    return np.abs(uv - cloud.uv.astype(np.float64)).max(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--cameras", nargs="+", default=["zed_left", "wrist_cam_l", "wrist_cam_r"])
    ap.add_argument("--frames", type=int, default=8, help="롤아웃에서 균등 간격으로 뽑을 프레임 수")
    ap.add_argument("--out-doc", default="benchmark/ag3s/docs/step-02-backprojection.md")
    ap.add_argument("--out-figs", default="benchmark/ag3s/asset/image/backprojection")
    add_tag_argument(ap)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    use_korean()
    import matplotlib.pyplot as plt
    import mujoco

    from benchmark.ag3s.experiments.mujoco_source import _body_vertices

    run = load_run(args.records)
    scene = replay_scene(run)
    config = PointCloudConfig()
    picks = np.linspace(0, len(run) - 1, min(args.frames, len(run))).astype(int)
    body_id = {n: mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, n)
               for n in OBJECT_BODIES}
    body_id = {n: i for n, i in body_id.items() if i >= 0}
    # 메시 정점은 body 프레임에서 한 번만 읽는다. 자세는 프레임마다 바뀌지만 형상은 아니다.
    verts_local = {n: _body_vertices(scene.model, i) for n, i in body_id.items()}

    top_gid, top_z = tabletop_geom(scene)
    reproj, plane_rows = [], []
    # C는 (물체, 카메라)별로 나눠 담는다. 한 카메라의 외부 파라미터만 틀린 경우가
    # 합쳐 놓으면 다른 카메라들 뒤에 숨는다.
    surf = {(n, c): [] for n in body_id for c in args.cameras}
    fused = {n: [] for n in body_id}
    scatter = None

    for fi in picks:
        pose_scene(scene, run.steps[fi])
        T_base_world = np.linalg.inv(scene.body_pose("base"))
        per_camera_points = {n: {} for n in body_id}
        for camera in args.cameras:
            frame, cloud, labels = analyse_frame(scene, camera, config)
            if len(cloud) == 0:
                continue
            reproj.append(float(reprojection_error(cloud, frame).max()))

            if camera == "zed_left":
                sel = (frame.geom_ids[cloud.uv[:, 1], cloud.uv[:, 0]] == top_gid)
                fit = fit_plane(cloud.points[sel]) if sel.sum() >= 200 else None
                if fit is not None:
                    normal, offset, residual, n_in = fit
                    tilt = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1, 1))))
                    plane_rows.append((int(sel.sum()), tilt, offset,
                                       float(np.sqrt((residual ** 2).mean())),
                                       float(np.percentile(np.abs(residual), 95)), n_in))
                if scatter is None:
                    scatter = (cloud.points.copy(), labels.copy())
                    T_base_world0 = T_base_world.copy()

            for name, bid in body_id.items():
                pts = cloud.points[labels == bid]
                if len(pts) < MIN_POINTS:
                    continue
                per_camera_points[name][camera] = pts
                d = surface_distance(pts, scene, name, T_base_world, verts_local[name])
                if d is not None:
                    surf[(name, camera)].append(np.abs(d))

        # D: 세 카메라를 합쳤을 때 표면이 두꺼워지는가. 외부 파라미터가 서로 맞으면 융합
        # 오차는 개별 카메라 오차 중 최댓값 근처에 머물고, 하나라도 틀어져 있으면 그 위로
        # 뛴다. 무게중심 비교와 달리 시점 차이에 반응하지 않는다 — 각 카메라가 물체의 다른
        # 면을 보는 것은 정상이고, 그 면들이 같은 표면 위에 있는지만 묻기 때문이다.
        for name, by_cam in per_camera_points.items():
            if len(by_cam) < 2:
                continue
            allp = np.concatenate(list(by_cam.values()))
            d = surface_distance(allp, scene, name, T_base_world, verts_local[name])
            if d is not None:
                fused[name].append((np.abs(d), sorted(by_cam)))
    scene.close()

    # ---------------------------------------------------------------- 요약
    scene2 = replay_scene(run)
    pose_scene(scene2, run.steps[picks[0]])
    true_pos = {n: scene2.body_position_in_base(n) for n in body_id}
    scene2.close()

    reproj_max = max(reproj) if reproj else float("nan")
    plane = np.asarray(plane_rows, float) if plane_rows else np.zeros((0, 6))

    def _stats(chunks):
        d = np.concatenate(chunks)
        return len(d), float(np.median(d)), float(np.percentile(d, 95)), float(d.max())

    surf_stats = {k: _stats(v) for k, v in surf.items() if v}
    per_object = {n: _stats([np.concatenate(v) for (m, _), v in surf.items() if m == n and v])
                  for n in body_id
                  if any(v for (m, _), v in surf.items() if m == n)}
    fused_stats = {}
    for name, entries in fused.items():
        if not entries:
            continue
        d = np.concatenate([e[0] for e in entries])
        cams = sorted({c for e in entries for c in e[1]})
        single = max((surf_stats[(name, c)][2] for c in cams if (name, c) in surf_stats),
                     default=float("nan"))
        fused_stats[name] = (len(d), float(np.percentile(d, 95)), single, cams)

    paths = resolve(out_figs=args.out_figs, out_doc=args.out_doc,
                    tag=args.tag).prepare()
    figs, img = paths.figures, paths.image_prefix

    # fig1 — 테이블 평면
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.2), dpi=160)
    style_axes(fig, (a1, a2))
    a1.grid(axis="y", color=GRID_INK, lw=.6); a1.set_axisbelow(True)
    a1.bar(range(len(plane)), plane[:, 1], color=CATEGORICAL[0], edgecolor=SURFACE, lw=2)
    a1.set_xlabel("프레임"); a1.set_ylabel("평면 기울기 (도)")
    a1.set_title("테이블 법선이 base +z와 이루는 각", color=INK, fontsize=10, loc="left")
    a2.grid(axis="y", color=GRID_INK, lw=.6); a2.set_axisbelow(True)
    a2.bar(range(len(plane)), plane[:, 3] * 1000, color=CATEGORICAL[2], edgecolor=SURFACE, lw=2,
           label="RMS 잔차")
    a2.plot(range(len(plane)), plane[:, 4] * 1000, color=CATEGORICAL[1], lw=1.6, marker="o",
            ms=4, label="95 백분위 잔차")
    a2.set_xlabel("프레임"); a2.set_ylabel("평면 잔차 (mm)")
    a2.set_title("평면 두께", color=INK, fontsize=10, loc="left")
    leg = a2.legend(frameon=False, fontsize=8)
    for t in leg.get_texts(): t.set_color(INK_2)
    fig.tight_layout(); fig.savefig(figs / "fig1_table_plane.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # fig2 — 물체 표면 오차, 카메라별
    names = [n for n in OBJECT_BODIES if n in per_object]
    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=160)
    style_axes(fig, ax)
    ax.grid(axis="y", color=GRID_INK, lw=.6); ax.set_axisbelow(True)
    x = np.arange(len(names))
    width = 0.8 / max(len(args.cameras), 1)
    for ci, camera in enumerate(args.cameras):
        vals = [surf_stats[(n, camera)][2] * 1000 if (n, camera) in surf_stats else np.nan
                for n in names]
        pos = x + ci * width - 0.4 + width / 2
        ax.bar(pos, vals, width * 0.88, color=CATEGORICAL[ci], edgecolor=SURFACE, lw=2,
               label=camera)
        for xi, v in zip(pos, vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(x, names)
    ax.set_ylabel("실제 표면까지 거리, 95 백분위 (mm)")
    ax.set_title("역투영한 점에서 물체의 실제 표면까지 — 카메라별",
                 color=INK, fontsize=11, loc="left", pad=8)
    leg = ax.legend(frameon=False, fontsize=8, ncol=len(args.cameras), loc="upper center",
                    bbox_to_anchor=(0.5, -0.14))
    for t in leg.get_texts(): t.set_color(INK_2)
    fig.tight_layout(); fig.savefig(figs / "fig2_surface_error.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # fig3 — 위에서 본 점군과 실제 물체 위치
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=160)
    style_axes(fig, ax)
    pts, labs = scatter
    other = ~np.isin(labs, list(body_id.values()))
    ax.scatter(pts[other, 0], pts[other, 1], s=.6, c=GRID_INK, lw=0, label="그 외 (테이블·바닥·로봇)")
    for slot, (name, bid) in enumerate(body_id.items()):
        sel = labs == bid
        if sel.sum() == 0:
            continue
        ax.scatter(pts[sel, 0], pts[sel, 1], s=3.0, c=CATEGORICAL[slot], lw=0, label=name)
        t = true_pos[name]
        ax.plot(t[0], t[1], marker="x", ms=9, mew=2.0, color=CATEGORICAL[slot])
    # 방 전체를 그리면 물체가 몇 픽셀로 뭉개진다. 물체가 차지하는 범위에 여백을 두고 자른다.
    obj = np.isin(labs, list(body_id.values()))
    if obj.any():
        lo = pts[obj, :2].min(axis=0) - 0.30
        hi = pts[obj, :2].max(axis=0) + 0.30
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("base x (m)"); ax.set_ylabel("base y (m)"); ax.set_aspect("equal")
    ax.set_title("역투영한 점군 (위에서 봄) — × 는 실제 물체 중심\n작업 영역으로 확대; 방 나머지는 잘림",
                 color=INK, fontsize=10.5, loc="left", pad=8)
    leg = ax.legend(frameon=False, fontsize=8, markerscale=3, loc="upper left")
    for t in leg.get_texts(): t.set_color(INK_2)
    fig.tight_layout(); fig.savefig(figs / "fig3_topdown.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- 문서
    def md(header, rows):
        return "\n".join(["| " + " | ".join(header) + " |",
                          "|" + "|".join("---" for _ in header) + "|", *rows])

    height_err = (plane[:, 2] - top_z) if (len(plane) and top_z is not None) else np.zeros(0)
    ok_a = reproj_max < 1e-6
    ok_b = bool(len(plane) and plane[:, 1].max() < 0.5 and np.abs(height_err).max() < 5e-3)
    ok_c = bool(surf_stats) and all(v[2] < 0.01 for v in surf_stats.values())
    # 융합이 개별 카메라 최댓값보다 유의미하게 두꺼워지지 않아야 한다.
    ok_d = bool(fused_stats) and all(
        f[1] <= max(f[2], 0.002) + 0.005 for f in fused_stats.values())
    verdict = "**PASS**" if (ok_a and ok_b and ok_c and ok_d) else "**FAIL**"

    worst_c = max((v[2] for v in surf_stats.values()), default=float("nan"))
    worst_gap = max((f[1] - f[2] for f in fused_stats.values()), default=float("nan"))

    doc = f"""# 2단계 — 3D back-projection

**질문.** 깊이에서 나온 3D 점이 실제로 물체가 있는 곳에 놓이는가?

1단계는 attention이 옳은 물체를 가리키는지 확인했다. 그 이름표가 붙는 대상은 이 점들이고,
점이 틀린 곳에 있으면 이후 모든 단계 — 클러스터링, primitive 근사, 여유거리 — 가 틀린 기하
위에서 정확하게 계산될 뿐이다.

| | |
|---|---|
| 기록 | `{run.path}` |
| 검사 프레임 | {len(picks)}개 ({len(run)}개 중 균등 간격) |
| 카메라 | {", ".join(f"`{c}`" for c in args.cameras)} |
| 역투영 | `benchmark.ag3s.reconstruction.backproject` (AG3S 본체 코드 그대로) |
| 정답 | MuJoCo 세그멘테이션(geom 단위) + geom 자세 + 메시 정점 |

## 판정 — {verdict}

| 검사 | 무엇이 깨지면 걸리는가 | 결과 | 기준 |
|---|---|---|---|
| A. 재투영 왕복 | 역투영 식 자체 | {'통과' if ok_a else '실패'} — 최대 {reproj_max:.2e} px | < 1e-6 px |
| B. 지지면 평면 | 카메라 회전 규약 | {'통과' if ok_b else '실패'} — 기울기 최대 {plane[:, 1].max() if len(plane) else float('nan'):.4f}°, 높이 오차 최대 {np.abs(height_err).max()*1000 if len(height_err) else float('nan'):.2f} mm | < 0.5°, < 5 mm |
| C. 물체 표면 오차 | 깊이 스케일, 내부 파라미터 | {'통과' if ok_c else '실패'} — 95 백분위 최대 {worst_c*1000:.2f} mm | < 10 mm |
| D. 카메라 간 정합 | 손목 카메라 외부 파라미터 (FK 사슬) | {'통과' if ok_d else '실패'} — 융합 시 두께 증가 최대 {worst_gap*1000:+.2f} mm | < +5 mm |

네 검사는 서로 다른 실패 방식에 대응한다. 한 가지만 재면 어느 것이 깨졌는지 알 수 없다.

### A. 재투영 왕복 — 산술과 규약을 분리한다

역투영한 점을 같은 `K`, `T_base_cam`으로 다시 투영해 원래 픽셀로 돌아오는지 본다. 이 검사는
**규약이 옳은지는 전혀 묻지 않는다** — 규약이 통째로 틀려 있어도 왕복은 완벽하게 닫힌다.
그래서 유용하다: A가 통과하고 B가 실패하면 문제는 카메라 규약이고, A가 실패하면 식 자체다.
이 둘을 구분하지 못하면 이후 디버깅이 추측이 된다.

전체 {len(reproj)}회 (프레임 × 카메라) 최대 오차 **{reproj_max:.2e} px** — 배정밀도 반올림
수준이다.

### B. 지지면 평면 — 회전 오차가 가장 먼저 드러나는 곳

테이블 픽셀만 역투영해 전최소제곱 평면을 맞춘다. 카메라 회전이 조금이라도 틀어져 있으면
평면이 기울고, 넓고 평평한 면이라 그 기울기가 크게 증폭된다.

**두 번 좁혀야 한다.** 첫째, `table`은 한 개의 body이지만 **다섯 개의 geom** — 상판 하나와
다리 넷 — 이므로 body 라벨로 고르면 수직인 다리가 들어온다 (첫 판: 기울기 37.55°). 둘째,
상판 geom 자체가 두께 40 mm의 box라 **옆면**도 보이고, 그 점들은 윗면보다 최대 40 mm 아래에
있다 (두 번째 판: 기울기 0.84°, 높이 −9 mm). 둘 다 역투영은 옳은데 검사가 틀린 경우다.

그래서 상판 geom의 점에 **AG3S 자신의 RANSAC 평면 추정기**(`support_surface.fit_plane_ransac`)를
건다. 점 수가 압도적인 윗면을 지배 평면으로 잡고 옆면을 이상점으로 버린다. 이것은 7단계
support surface 추출이 실제로 쓰는 바로 그 코드이므로, 여기서 통과한다는 것은 그 단계가 같은
씬에서 옳은 평면을 잡는다는 뜻이기도 하다.

MuJoCo geom에서 읽은 상판 윗면의 실제 높이: **{f'{top_z:.6f} m' if top_z is not None else '읽지 못함'}** (base 프레임)

{md(["프레임", "상판 geom 점", "평면 내점", "기울기 (°)", "측정 높이 (m)", "높이 오차 (mm)", "RMS 잔차 (mm)", "95 백분위 잔차 (mm)"],
    [f"| {i} | {int(r[0])} | {int(r[5])} | {r[1]:.4f} | {r[2]:.6f} | {(r[2]-top_z)*1000 if top_z is not None else float('nan'):+.2f} | {r[3]*1000:.2f} | {r[4]*1000:.2f} |"
     for i, r in enumerate(plane)])}

![테이블 평면]({img}/fig1_table_plane.png)

### C. 물체 표면 오차 — 중심이 아니라 표면과 비교한다

각 점에서 그 물체의 *실제* 표면까지의 거리를 잰다. 정답 표면은 MuJoCo 메시 정점을 그 프레임의
실제 자세로 변환한 볼록 껍질이다.

**중심 좌표와 비교하지 않는 이유**는 카메라가 앞면만 보기 때문이다. 보이는 점들의 무게중심은
카메라 쪽으로 대략 반지름만큼 치우쳐 있으므로, 역투영이 완벽해도 중심과의 거리는 0이 되지
않는다. 그 편차를 오차로 보고하면 없는 문제를 만들어내게 된다.

정답 표면은 geom 종류에 따라 다르게 만든다. 과일은 메시이고 볼록하므로 정점의 볼록 껍질을
쓴다. **크레이트는 상자 36개로 만든 속이 빈 통**이라 볼록 껍질을 쓰면 열린 입구가 덮여 내부가
통째로 안쪽이 되고, 안쪽 벽의 점이 크레이트 반너비만큼 떨어진 것으로 계산된다 — 역투영이
완벽해도 수십 mm의 가짜 오차가 나온다. 그래서 상자에는 점-상자 표면 거리를 정확히 계산해
geom별 최솟값을 취한다. 속이 빈 형상이 자연히 처리되고 근사가 아니다.

카메라별로 나눠 싣는다 — 한 카메라의 외부 파라미터만 틀린 경우가, 합쳐 놓으면 나머지 뒤에
숨기 때문이다.

{md(["body", "카메라", "점 수", "중앙값 (mm)", "95 백분위 (mm)", "최대 (mm)"],
    [f"| {n} | {c} | {surf_stats[(n, c)][0]} | {surf_stats[(n, c)][1]*1000:.2f} | {surf_stats[(n, c)][2]*1000:.2f} | {surf_stats[(n, c)][3]*1000:.2f} |"
     for n in OBJECT_BODIES for c in args.cameras if (n, c) in surf_stats])}

![표면 오차]({img}/fig2_surface_error.png)

### D. 카메라 간 정합 — 외부 파라미터 사슬 전체를 검사한다

손목 카메라의 외부 파라미터는 `T_base_cam(t) = FK(q, mount_link) · T_link_cam`을 타고 온다.
마운트 링크나 `T_link_cam`이 틀리면 head 카메라만으로는 A·B·C가 전부 통과하면서 손목만
어긋난다.

**무게중심을 비교하면 안 된다.** 카메라마다 물체의 다른 면을 보므로 무게중심은 정상적으로
다르다 — 이 씬에서 크레이트의 카메라별 무게중심은 최대 14 cm 벌어지는데, 그것은 보정 오차가
아니라 시점 차이다. (이 보고서의 첫 판이 그 값을 오차로 보고했다.)

대신 **세 카메라를 합쳤을 때 표면이 두꺼워지는가**를 본다. 외부 파라미터가 서로 맞으면 융합
오차는 개별 카메라 오차 중 최댓값 근처에 머물고, 하나라도 틀어져 있으면 그 위로 뛴다. 각
카메라가 다른 면을 보는 것은 이 지표를 흔들지 않는다 — 그 면들이 **같은 표면 위에 있는지**만
묻기 때문이다.

{md(["body", "합친 카메라", "융합 95 백분위 (mm)", "개별 최댓값 (mm)", "증가 (mm)"],
    [f"| {n} | {', '.join(f[3])} | {f[1]*1000:.2f} | {f[2]*1000:.2f} | {(f[1]-f[2])*1000:+.2f} |"
     for n, f in fused_stats.items()] or ["| — | — | — | — | — |"])}

### 점군과 실제 위치

![위에서 본 점군]({img}/fig3_topdown.png)

× 는 MuJoCo가 보고한 실제 물체 중심이다. 점들이 × 를 **둘러싸지 않고 한쪽에 몰려 있는 것이
정상**이다 — 카메라는 앞면만 보므로 뒷면 점은 존재하지 않는다. 이것이 C에서 중심 대신 표면과
비교하는 이유이자, 4단계에서 클러스터 무게중심을 물체 중심으로 쓰면 안 되는 이유다.

## 이 숫자들이 재지 *않은* 것

여기서 잰 것은 **역투영 수학과 좌표 규약의 정확도**이지, 실제 센서에서의 정확도가 아니다.
두 가지를 명시해 둔다.

**depth에 잡음이 없다.** MuJoCo 렌더러가 내는 깊이에는 실제 ZED가 갖는 잡음, 반사면에서의
결측, 물체 경계의 flying pixel, 스테레오 정합 실패가 없다. 위의 "표면 오차 0.4–7 mm"는
잡음 없는 깊이에 대한 값이므로, 실기에서 같은 숫자가 나올 것으로 읽으면 안 된다.
기록기에 `--record-depth`를 켜면 실제 센서와 같은 **uint16 밀리미터**로 저장되어 1 mm 양자화는
반영되지만, 그것은 잡음 모형이 아니라 표현 형식일 뿐이다.

**정답이 시뮬레이터에서 온다.** 물체 자세·메시 정점·세그멘테이션이 전부 MuJoCo가 알려준
값이다. 실기에는 이런 정답이 없으므로, 실기로 넘어가면 정답을 다른 방식으로 마련하거나
(마커, 수동 라벨) 정답 없이 자기일관성 검사만으로 만족해야 한다. 이 검증의 최대 강점이
시뮬레이터라는 점이고, 최대 한계도 같은 점이다.

## 그림에 대하여

공통 규격은 `benchmark/ag3s/experiments/figstyle.py`에 있다.

### fig1 — 테이블 평면 (2패널)

`fig1_table_plane.png`. 왼쪽은 프레임별 평면 기울기(도), 오른쪽은 잔차(RMS와 95 백분위, mm).

**만드는 법.** 프레임마다 상판 geom 픽셀만 역투영하고 `support_surface.fit_plane_ransac`으로
지배 평면을 뽑아, 법선과 base +z 사이 각과 내점 잔차를 계산한다.

**읽는 법.** 최대 잔차 대신 95 백분위를 그린 이유는 최댓값이 RANSAC의 내점 임계값(8 mm)에
눌려 매 프레임 같은 값이 나오기 때문이다 — 측정값이 아니라 설정값이 되어 아무것도 알려주지
않는다.

### fig2 — 물체 표면 오차, 카메라별

`fig2_surface_error.png`. 물체마다 카메라 3대의 95 백분위 오차를 나란히 놓은 막대.

**만드는 법.** 각 점에서 그 물체의 실제 표면까지 거리를 재고 95 백분위를 취한다. 정답 표면은
메시 물체(과일)면 정점의 볼록 껍질, 상자 물체(크레이트)면 geom별 점-상자 표면 거리의
최솟값이다.

**왜 카메라별로 나누는가.** 한 카메라의 외부 파라미터만 틀린 경우가, 세 카메라를 합쳐 놓으면
나머지 뒤에 숨는다. 나눠 놓으면 어느 카메라가 문제인지 막대 하나로 보인다.

### fig3 — 위에서 본 점군

`fig3_topdown.png`. 한 프레임의 점군을 x–y 평면에 투영한 산점도. 색은 정답 물체, ×는 MuJoCo가
보고한 실제 물체 중심, 회색은 나머지(테이블·바닥·로봇)다.

**만드는 법.** 역투영한 점에 세그멘테이션 라벨을 붙이고 물체별로 색을 준다. 범위는 물체가
차지하는 구간에 30 cm 여백을 두고 자른다 — 방 전체를 그리면 물체가 몇 픽셀로 뭉개진다.

**읽는 법.** 점이 ×를 **둘러싸지 않고 한쪽에 몰리는 것이 정상**이다. 카메라는 앞면만 보므로
뒷면 점은 존재하지 않는다. 크레이트가 속이 빈 사각 링으로 나오는 것도 같은 이유이며, 이것이
클러스터 무게중심을 물체 중심으로 쓰면 안 되는 이유를 그대로 보여준다.

## 재현

```bash
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.backprojection_report \\
    --records {args.records}
```
"""
    out = paths.document
    out.write_text(doc)
    out.with_suffix(".json").write_text(json.dumps({
        "reprojection_max_px": reproj_max,
        "tabletop_z": top_z,
        "plane": [dict(zip(("n_points", "tilt_deg", "offset_m", "rms_m", "p95_m", "n_inliers"), r))
                  for r in plane.tolist()],
        "surface_by_camera": {f"{n}|{c}": dict(zip(("n", "median_m", "p95_m", "max_m"), v))
                              for (n, c), v in surf_stats.items()},
        "fused": {n: {"n": f[0], "p95_m": f[1], "best_single_p95_m": f[2], "cameras": f[3]}
                  for n, f in fused_stats.items()},
        "checks": {"A_reprojection": ok_a, "B_plane": ok_b, "C_surface": ok_c, "D_cross_camera": ok_d},
    }, indent=2))
    print(f"wrote {out} and 3 figures in {figs}")
    print(f"판정 {verdict}: A={reproj_max:.1e}px  "
          f"B=기울기 {plane[:,1].max() if len(plane) else float('nan'):.4f}°/높이 "
          f"{np.abs(height_err).max()*1000 if len(height_err) else float('nan'):.2f}mm  "
          f"C={worst_c*1000:.2f}mm  D=+{worst_gap*1000:.2f}mm")


if __name__ == "__main__":
    main()
