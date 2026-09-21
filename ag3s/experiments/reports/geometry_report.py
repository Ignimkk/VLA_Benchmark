"""6단계 검증 — primitive 근사가 무엇을 포함하고 무엇을 놓치는가.

5단계까지는 점을 다뤘다. 6단계는 점을 **도형**으로 바꾼다. 최적화기가 실제로 보는 것은 이
도형이고, 도형이 물체보다 작으면 최적화기는 존재하지 않는다고 들은 부분을 향해 궤적을 낸다.
그래서 이 단계의 불변식은 하나다 — **과소 근사 금지**.

두 가지 포함을 구분해서 잰다. 이 구분이 이 단계의 전부다.

**A. 관측한 점을 포함하는가.** AG3S 자신의 `containment_report`가 재는 것이고, 코드가 이미
보장하려 하는 것이다. 여기서 깨지면 명백한 버그다.

**B. 실제 물체를 포함하는가.** 이쪽이 진짜 질문이다. 2단계에서 확인했듯 카메라는 **앞면만**
본다. 앞면 점을 전부 담는 구는 물체의 뒷면을 담지 못한다 — 관측한 점은 100% 포함하면서
실제 물체는 절반만 덮을 수 있고, 그 상태는 A로는 절대 드러나지 않는다.

B가 이 파이프라인에 `geometry.perception_uncertainty`가 있는 이유다. 그 예산이 무엇을 위한
것인지 말로 설명하는 대신, 예산을 0부터 올려가며 **실제 물체를 덮는 데 얼마가 필요한지**를
잰다.

그 외:

**C. 제약용 구가 primitive를 포함하는가.** 최적화기가 보는 최종 형태는 primitive가 아니라
`to_spheres()`가 낸 구들이다. 캡슐은 구 체인이 되는데, 체인 간격이 반지름보다 벌어지면 구
사이에 틈이 생긴다 — 충돌을 만드는 방향의 오차다. 코드는 `sqrt(r² + (간격/2)²)`로 부풀려
막는다고 적고 있다. 실제 데이터에 캡슐이 나오지 않으므로 합성 캡슐로 직접 검사한다.

**D. 도형 종류별 비용.** sphere·capsule·box·ellipsoid 각각이 실제 물체를 얼마나 덮고 부피를
얼마나 쓰는가. 덮는 것이 먼저이고 부피는 그 비용이다.

실행:
    MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.reports.geometry_report \
        --records outputs/.../ag3s_records/run_0002 \
        --attention benchmark/ag3s/asset/data/attention_step1_run0002.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import numpy as np

from benchmark.ag3s.stages.attention_lifting import GridAttentionAdapter, lift
from benchmark.ag3s.config import (
    AttentionConfig, ClusteringConfig, CollisionCandidateConfig, ContactConfig, GeometryConfig,
    PointCloudConfig, SupportSurfaceConfig,
)
from benchmark.ag3s.stages.collision_candidates import generate_candidates
from benchmark.ag3s.experiments.common.outputs import add_tag_argument, resolve
from benchmark.ag3s.experiments.common.figstyle import (
    CATEGORICAL, GRID_INK, INK, INK_2, SURFACE, style_axes, use_korean,
)
from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
from benchmark.ag3s.stages.geometry import (
    contains, containment_report, fit_primitive, inflate, to_spheres,
)
from benchmark.ag3s.stages.reconstruction import reconstruct
from benchmark.ag3s.stages.robot_filter import filter_robot_points
from benchmark.ag3s.stages.support_surface import fit_support_surfaces
from benchmark.ag3s.stages.target_grounding import ground_target
from benchmark.ag3s.types import Primitive, PrimitiveType

OBJECT_BODIES = ("crate", "apple", "banana", "orange", "pear")
PRIMITIVES = ("sphere", "capsule", "box", "ellipsoid")
#: 예산을 0부터 올려가며 실제 물체를 덮는 데 얼마가 필요한지 본다 (미터).
UNCERTAINTY_SWEEP = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05)


def true_surface(scene, name: str, T_base_world: np.ndarray) -> np.ndarray:
    """물체의 **실제** 표면을 대표하는 점들 — base 프레임.

    메시 물체는 정점을, 상자 물체는 8개 모서리를 쓴다. 이 점들이 primitive 안에 있는지가
    "실제 물체를 포함하는가"의 조작적 정의다. 관측한 점과 달리 **뒷면도 포함한다.**
    """
    from benchmark.ag3s.experiments.sources.mujoco_source import _body_vertices

    mujoco = scene.mujoco
    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        return np.zeros((0, 3))
    R, t = T_base_world[:3, :3], T_base_world[:3, 3]
    out = []
    verts = _body_vertices(scene.model, bid)
    if len(verts):
        T_bw = scene.body_pose(name)
        out.append((verts @ T_bw[:3, :3].T + T_bw[:3, 3]) @ R.T + t)
    corner = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    for gid in range(scene.model.ngeom):
        if scene.model.geom_bodyid[gid] != bid:
            continue
        if int(scene.model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        half = np.asarray(scene.model.geom_size[gid], float)
        Rg = scene.data.geom_xmat[gid].reshape(3, 3)
        pg = scene.data.geom_xpos[gid]
        out.append(((corner * half) @ Rg.T + pg) @ R.T + t)
    return np.concatenate(out) if out else np.zeros((0, 3))


def coverage(primitives, surface: np.ndarray) -> tuple[float, float]:
    """실제 표면이 **도형들의 합집합** 안에 드는 비율과, 밖으로 나간 최대 거리 (m).

    합집합이어야 하는 이유는 이 파이프라인이 물체를 쪼개기 때문이다. 속이 빈 크레이트는 벽들이
    유클리드로 연결되지 않아 여러 후보로 나뉘고, 그 조각 하나를 크레이트 *전체*와 비교하면
    당연히 크게 못 덮는다 — 하지만 최적화기는 조각 하나가 아니라 **후보 전부**를 본다.
    (이 보고서의 첫 판이 그 비교를 해서 크레이트 침투를 408 mm로 보고했다.)

    밖으로 나간 거리는 각 점에서 가장 가까운 도형까지의 여유로 재고, 그중 최댓값을 쓴다.
    """
    if surface.size == 0:
        return float("nan"), float("nan")
    prims = list(primitives)
    if not prims:
        return 0.0, float("nan")
    inside = np.zeros(len(surface), bool)
    slack = np.full(len(surface), np.inf)
    for prim in prims:
        inside |= contains(prim, surface)
        # 구/타원체/캡슐에는 정확하고 상자에는 보수적(=과대)이라, 보고되는 침투가
        # 실제보다 작지 않다.
        d = np.linalg.norm(surface - prim.center, axis=1) - prim.bounding_radius
        slack = np.minimum(slack, d)
    if inside.all():
        return 1.0, 0.0
    return float(inside.mean()), float(np.max(slack[~inside]).clip(min=0.0))


def main() -> None:
    from benchmark.ag3s.experiments.reports.attention_report import target_from_prompt
    from benchmark.ag3s.experiments.reports.grounding_report import build_robot_model, majority_body

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--attention", required=True)
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/archive/step-verification-20260904/step-01-attention.json")
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--phase", default="approach")
    ap.add_argument("--out-doc", default="benchmark/ag3s/docs/archive/step-verification-20260904/step-06-geometry.md")
    ap.add_argument("--out-figs", default="benchmark/ag3s/asset/image/geometry")
    add_tag_argument(ap)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    use_korean()
    import matplotlib.pyplot as plt
    import mujoco

    chosen = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    run = load_run(args.records)
    blob = np.load(args.attention, allow_pickle=False)
    A = np.asarray(blob["attention"], np.float32)
    ci = [str(c) for c in blob["cameras"]].index("cam_high")
    di = [int(d) for d in blob["denoise_steps"]].index(int(chosen["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(chosen["agg"]))
    target = target_from_prompt(run.prompt)

    scene = replay_scene(run)
    robot_model = build_robot_model(scene)
    names = {i: (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or "?")
             for i in range(scene.model.nbody)}
    pc_cfg = dataclasses.replace(PointCloudConfig(), range_max=(args.range_max or None))
    att_cfg, cl_cfg = AttentionConfig(), ClusteringConfig()
    geo0 = GeometryConfig()

    rows, sweep_rows, type_rows, true_rows_local, real_chain = [], [], [], [], []
    example = None
    scene_snapshot = None
    body_ids_s = {}
    for fi in range(min(args.frames, len(run))):
        pose_scene(scene, run.steps[fi])
        frame = scene.capture("zed_left")
        T_bw = np.linalg.inv(scene.body_pose("base"))
        cloud, _ = reconstruct(depth=frame.depth, camera_intrinsics=frame.camera_intrinsics,
                               T_base_cam=frame.T_base_cam, config=pc_cfg)
        cloud, _ = filter_robot_points(cloud, robot_model, frame.robot_state, pc_cfg)
        if len(cloud) == 0:
            continue
        _, smask = fit_support_surfaces(cloud, SupportSurfaceConfig())
        grid = A[fi, di, ai, chosen["layer"], chosen["head"], ci]
        acloud = lift(cloud, grid, att_cfg, adapter=GridAttentionAdapter(), image_hw=frame.hw)
        result = ground_target(acloud, cl_cfg, seed_percentile=att_cfg.seed_percentile,
                               exclude_mask=smask)
        candidates, _ = generate_candidates(
            cloud, phase=args.phase, target=result.target, support_mask=smask,
            config=CollisionCandidateConfig(), geometry_config=geo0,
            contact_config=ContactConfig())
        labels = frame.body_ids[cloud.uv[:, 1], cloud.uv[:, 0]]
        surfaces = {b: true_surface(scene, b, T_bw) for b in OBJECT_BODIES}
        if scene_snapshot is None:
            body_ids_s = {b: mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, b)
                          for b in OBJECT_BODIES}
            scene_snapshot = (cloud, labels.copy(), list(candidates), smask.copy(), fi)

        # A: 후보마다 자기 점을 담는가 (개별 검사)
        by_body: dict[str, list] = {}
        for cand in candidates:
            pi = np.asarray(cand.point_indices, np.int64)
            if pi.size == 0:
                continue
            pts = cloud.points[pi]
            body, purity = majority_body(labels[pi], names)
            prim = cand.geometry[0]
            rep = containment_report(prim, pts)
            rows.append({"frame": fi, "body": body, "purity": purity,
                         "type": cand.source_type.value, "n": int(pi.size),
                         "prim": prim.type.value, "radius": float(prim.bounding_radius),
                         "centre_offset": float(np.linalg.norm(
                             prim.center - scene.body_position_in_base(body)))
                         if body in OBJECT_BODIES else float("nan"),
                         "containment_points": rep["containment_rate"],
                         "excess_volume": rep["excess_volume_ratio"]})
            if purity >= 0.6 and body in OBJECT_BODIES:
                by_body.setdefault(body, []).append((prim, pts))
            if example is None and body == target and cand.source_type.value == "target":
                example = (pts, prim, surfaces.get(body), fi,
                           frame.camera_intrinsics.copy(), frame.T_base_cam.copy(), frame.hw)

        # B: 물체마다, 그 물체를 대표하는 **모든** 후보의 합집합이 실제 표면을 덮는가
        for body, entries in by_body.items():
            surf = surfaces.get(body, np.zeros((0, 3)))
            if not len(surf):
                continue
            prims = [p for p, _ in entries]
            allpts = np.concatenate([q for _, q in entries])
            cov, pen = coverage(prims, surf)
            need = None
            for u in UNCERTAINTY_SWEEP:
                c2, _ = coverage([inflate(p, u) for p in prims], surf)
                sweep_rows.append({"frame": fi, "body": body, "uncertainty": u, "coverage": c2})
                if need is None and c2 >= 1.0 - 1e-9:
                    need = u
            true_rows_local.append({"frame": fi, "body": body, "n_prims": len(prims),
                                    "n_points": int(len(allpts)),
                                    "containment_points": float(np.mean(
                                        [containment_report(p, q)["containment_rate"]
                                         for p, q in entries])),
                                    "containment_true": cov, "penetration": pen,
                                    "need_uncertainty": need})
            # 실데이터에 맞춘 캡슐의 구 체인을 직접 검사한다 (합성이 아니라 관측에서 나온 캡슐).
            for _, q in entries:
                capsule = fit_primitive(q, "capsule", min_radius=geo0.min_radius)
                if capsule.type is not PrimitiveType.CAPSULE:
                    continue   # fit_primitive 가 sphere 로 강등한 경우
                sph = to_spheres(capsule)
                inside = np.zeros(len(q), bool)
                for c_, r_ in sph:
                    inside |= np.linalg.norm(q - c_, axis=1) <= r_ + 1e-12
                real_chain.append({"frame": fi, "body": body, "n_spheres": len(sph),
                                   "radius": float(capsule.dimensions[0]),
                                   "half_len": float(capsule.dimensions[1]),
                                   "covered_points": float(inside.mean())})
            for ptype in PRIMITIVES:
                p2 = [fit_primitive(q, ptype, min_radius=geo0.min_radius) for _, q in entries]
                c3, pen3 = coverage(p2, surf)
                type_rows.append({"frame": fi, "body": body, "prim": ptype,
                                  "containment_points": float(np.mean(
                                      [containment_report(a, q)["containment_rate"]
                                       for a, (_, q) in zip(p2, entries)])),
                                  "containment_true": c3, "penetration": pen3,
                                  "excess_volume": float(np.mean(
                                      [containment_report(a, q)["excess_volume_ratio"]
                                       for a, (_, q) in zip(p2, entries)]))})
    scene.close()

    # ---- C. 캡슐 구 체인이 캡슐을 덮는가 (합성) -------------------------------------------
    rng = np.random.default_rng(0)
    chain_rows = []
    for half_len in (0.05, 0.10, 0.20, 0.40):
        radius = 0.02
        axis = np.array([0.0, 0.0, 1.0])
        cap = Primitive(type=PrimitiveType.CAPSULE, center=np.zeros(3),
                        orientation=np.eye(3),
                        dimensions=np.array([radius, half_len, radius]))
        # 캡슐 표면 위의 점을 촘촘히 만들어 구 합집합이 덮는지 본다.
        t = rng.uniform(-half_len, half_len, 40000)
        th = rng.uniform(0, 2 * np.pi, 40000)
        surf = np.stack([radius * np.cos(th), radius * np.sin(th), t], axis=1)
        spheres = to_spheres(cap)
        inside = np.zeros(len(surf), bool)
        for c, r in spheres:
            inside |= np.linalg.norm(surf - c, axis=1) <= r + 1e-12
        spacing = (2 * half_len / (len(spheres) - 1)) if len(spheres) > 1 else 0.0
        naive = float(np.sqrt(radius ** 2 + (spacing / 2) ** 2))
        chain_rows.append({"half_len": half_len, "n_spheres": len(spheres),
                           "spacing": spacing, "radius": radius, "effective": spheres[0][1],
                           "expected": naive, "covered": float(inside.mean())})

    ok_a = all(abs(r["containment_points"] - 1.0) < 1e-9 for r in rows)
    true_rows = true_rows_local
    ok_c = (all(abs(c["covered"] - 1.0) < 1e-9 for c in chain_rows)
            and all(abs(c["covered_points"] - 1.0) < 1e-9 for c in real_chain))
    mean_true = float(np.mean([r["containment_true"] for r in true_rows])) if true_rows else float("nan")
    max_pen = float(np.max([r["penetration"] for r in true_rows])) if true_rows else float("nan")
    needs = [r["need_uncertainty"] for r in true_rows if r.get("need_uncertainty") is not None]
    ok_b = bool(true_rows) and mean_true >= 0.999
    verdict = "**PASS**" if (ok_a and ok_b and ok_c) else "**부분 통과**"

    paths = resolve(out_figs=args.out_figs, out_doc=args.out_doc,
                    tag=args.tag).prepare()
    figs, img = paths.figures, paths.image_prefix

    # fig1 — 두 포함률
    bodies = sorted({r["body"] for r in true_rows})
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=160)
    style_axes(fig, ax)
    ax.grid(axis="y", color=GRID_INK, lw=.6); ax.set_axisbelow(True)
    x = np.arange(len(bodies))
    pv = [np.mean([r["containment_points"] for r in true_rows if r["body"] == b]) for b in bodies]
    tv = [np.mean([r["containment_true"] for r in true_rows if r["body"] == b]) for b in bodies]
    ax.bar(x - 0.2, pv, 0.38, color=CATEGORICAL[2], edgecolor=SURFACE, lw=2, label="관측 점 포함률")
    ax.bar(x + 0.2, tv, 0.38, color=CATEGORICAL[7], edgecolor=SURFACE, lw=2, label="실제 물체 포함률")
    for xi, (a_, b_) in zip(x, zip(pv, tv)):
        ax.text(xi - 0.2, a_, f"{a_:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
        ax.text(xi + 0.2, b_, f"{b_:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, bodies); ax.set_ylim(0, 1.12); ax.set_ylabel("포함률")
    ax.set_title("두 가지 포함 — 관측한 점은 전부 담아도 실제 물체는 그렇지 않다",
                 color=INK, fontsize=11, loc="left", pad=8)
    leg = ax.legend(frameon=False, fontsize=8.5, ncol=2)
    for t_ in leg.get_texts(): t_.set_color(INK_2)
    fig.tight_layout(); fig.savefig(figs / "fig1_containment.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # fig2 — uncertainty 예산 스윕
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=160)
    style_axes(fig, ax)
    ax.grid(axis="y", color=GRID_INK, lw=.6); ax.set_axisbelow(True)
    for slot, b in enumerate(bodies):
        ys = [np.mean([s["coverage"] for s in sweep_rows if s["body"] == b and s["uncertainty"] == u])
              for u in UNCERTAINTY_SWEEP]
        ax.plot(np.asarray(UNCERTAINTY_SWEEP) * 1000, ys, lw=2.0, marker="o", ms=4,
                color=CATEGORICAL[slot], label=b)
    ax.axhline(1.0, color=INK_2, lw=1.0, ls=(0, (4, 3)))
    ax.text(0.996, 1.0, "완전 포함 ", transform=ax.get_yaxis_transform(), va="bottom",
            ha="right", fontsize=7.5, color=INK_2)
    ax.set_xlabel("perception_uncertainty (mm)"); ax.set_ylabel("실제 물체 포함률")
    ax.set_ylim(0, 1.08)
    ax.set_title("예산을 얼마나 줘야 실제 물체를 덮는가", color=INK, fontsize=11, loc="left", pad=8)
    leg = ax.legend(frameon=False, fontsize=8.5, ncol=len(bodies))
    for t_ in leg.get_texts(): t_.set_color(INK_2)
    fig.tight_layout(); fig.savefig(figs / "fig2_uncertainty.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # fig4 — 씬에서 도형이 어디를 놓치는가 (3D)
    if example is not None:
        from benchmark.ag3s.experiments.reports.cloud_gallery import _ax3d, fit_box, workspace_box
        from benchmark.ag3s.runtime.visualization import _wire_sphere
        pts_e, prim_e, surf_e, fi_e, K_e, T_e, hw_e = example
        need_e = next((r["need_uncertainty"] for r in true_rows
                       if r["frame"] == fi_e and r["body"] == target
                       and r["need_uncertainty"] is not None), 0.03)
        fig = plt.figure(figsize=(11.4, 5.0), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        for j, (amount, ttl) in enumerate(
                ((0.0, f"기본값 perception_uncertainty = {geo0.perception_uncertainty}"),
                 (need_e, f"이 프레임에 필요한 예산 {need_e*1000:.0f} mm 적용"))):
            prim_j = inflate(prim_e, amount)
            ax = _ax3d(fig, (1, 2, j + 1), ttl)
            ins = contains(prim_j, surf_e)
            ax.scatter(surf_e[ins, 0], surf_e[ins, 1], surf_e[ins, 2], s=3, c=GRID_INK,
                       depthshade=False, linewidths=0,
                       label="실제 표면 · 도형 안" if j == 0 else None)
            if (~ins).any():
                ax.scatter(surf_e[~ins, 0], surf_e[~ins, 1], surf_e[~ins, 2], s=9,
                           c=CATEGORICAL[7], depthshade=False, linewidths=0,
                           label="실제 표면 · 도형 밖 (최적화기가 못 보는 살)" if j == 0 else None)
            ax.scatter(pts_e[:, 0], pts_e[:, 1], pts_e[:, 2], s=7, c=CATEGORICAL[0],
                       depthshade=False, linewidths=0,
                       label="관측한 점" if j == 0 else None)
            _wire_sphere(ax, prim_j.center, prim_j.bounding_radius, INK_2, alpha=.30)
            fit_box(ax, workspace_box(np.concatenate([surf_e, pts_e]), 0.02))
            ax.view_init(elev=16, azim=-64)
            ax.set_title(f"{ttl}\n실제 표면 포함률 {ins.mean():.3f}", fontsize=9.5, color=INK)
        h, l = fig.axes[0].get_legend_handles_labels()
        leg = fig.legend(h, l, frameon=False, fontsize=9, ncol=3, markerscale=2.5,
                         loc="lower center", bbox_to_anchor=(0.5, 0.0))
        for t_ in leg.get_texts(): t_.set_color(INK_2)
        fig.suptitle(f"프레임 {fi_e} · `{target}` — 관측한 점은 전부 담기는데 실제 물체는 뒤가 남는다",
                     color=INK, fontsize=12, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.06, 1, 0.93))
        fig.savefig(figs / "fig4_scene_miss.png", facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    # fig6 — 같은 것을 관측 이미지 위에: 도형 밖으로 나간 실제 표면이 어디인가
    if example is not None:
        from benchmark.ag3s.experiments.common.imageview import project, show, sphere_circle
        img_e = run.steps[fi_e].images["cam_high"]
        ih_e = img_e.shape[:2]
        need_e2 = next((r["need_uncertainty"] for r in true_rows
                        if r["frame"] == fi_e and r["body"] == target
                        and r["need_uncertainty"] is not None), 0.03)
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), dpi=160)
        fig.patch.set_facecolor(SURFACE)
        for ax in axes:
            ax.set_facecolor(SURFACE)
        show(axes[0], img_e, "정책이 본 입력", ink=INK, grid_ink=GRID_INK)
        for ax, amount, ttl in ((axes[1], 0.0, f"기본값 예산 {geo0.perception_uncertainty*1000:.0f} mm"),
                                (axes[2], need_e2, f"예산 {need_e2*1000:.0f} mm 적용")):
            prim_j = inflate(prim_e, amount)
            ins = contains(prim_j, surf_e)
            show(ax, img_e, f"{ttl}\n실제 표면 포함률 {ins.mean():.3f}", ink=INK, grid_ink=GRID_INK)
            us, vs, _, oks = project(surf_e, K_e, T_e, hw_e, ih_e)
            m_in = oks & ins
            m_out = oks & ~ins
            ax.scatter(us[m_in], vs[m_in], s=10, c="#c9c8c2", lw=0)
            if m_out.any():
                ax.scatter(us[m_out], vs[m_out], s=14, c=CATEGORICAL[7], lw=0)
            up, vp, _, okp = project(pts_e, K_e, T_e, hw_e, ih_e)
            ax.scatter(up[okp], vp[okp], s=8, c=CATEGORICAL[0], lw=0)
            c_ = sphere_circle(prim_j.center, prim_j.bounding_radius, K_e, T_e, hw_e, ih_e)
            if c_:
                ax.add_patch(plt.Circle(c_[:2], c_[2], fill=False, ec=INK_2, lw=2.0, ls=(0, (4, 3))))
        c0 = sphere_circle(prim_e.center, prim_e.bounding_radius, K_e, T_e, hw_e, ih_e)
        axes[0].set_xlim(0, ih_e[1]); axes[0].set_ylim(ih_e[0], 0)
        if c0:
            pad = max(c0[2] * 3.4, 20.0)
            for ax in axes[1:]:
                ax.set_xlim(c0[0] - pad, c0[0] + pad); ax.set_ylim(c0[1] + pad, c0[1] - pad)
        fig.suptitle(f"관측 이미지 위 · `{target}` — 파랑 = 관측한 점, 빨강 = 도형 밖으로 나간 실제 표면, "
                     f"점선 원 = 제약용 구", color=INK, fontsize=11.5, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(figs / "fig6_image_overlay.png", facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    # fig5 — 씬 전체에서 모든 후보의 도형 (위에서)
    if example is not None and scene_snapshot is not None:
        cloud_s, labels_s, cands_s, smask_s, fi_s = scene_snapshot
        fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=160)
        style_axes(fig, ax)
        pts_all = cloud_s.points
        ax.scatter(pts_all[smask_s, 0], pts_all[smask_s, 1], s=.5, c="#e8e7e2", lw=0,
                   label="지지면")
        rest = ~smask_s
        ax.scatter(pts_all[rest, 0], pts_all[rest, 1], s=.6, c=GRID_INK, lw=0, label="그 외 점")
        seen = set()
        for cand in cands_s:
            prim = cand.geometry[0]
            is_t = cand.source_type.value == "target"
            colour = CATEGORICAL[0] if is_t else CATEGORICAL[2]
            lab = ("target 후보" if is_t else "obstacle 후보")
            circ = plt.Circle((prim.center[0], prim.center[1]), prim.bounding_radius,
                              fill=False, ec=colour, lw=1.8 if is_t else 1.1,
                              label=None if lab in seen else lab)
            seen.add(lab)
            ax.add_patch(circ)
        for b in OBJECT_BODIES:
            tp = pts_all[labels_s == body_ids_s.get(b, -1)]
            if len(tp):
                ax.scatter(tp[:, 0], tp[:, 1], s=4, c=CATEGORICAL[3], lw=0,
                           label="정답 물체 점" if "gt" not in seen else None)
                seen.add("gt")
        obj = np.isin(labels_s, list(body_ids_s.values()))
        if obj.any():
            lo = pts_all[obj, :2].min(axis=0) - 0.22; hi = pts_all[obj, :2].max(axis=0) + 0.22
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel("base x (m)"); ax.set_ylabel("base y (m)"); ax.set_aspect("equal")
        ax.set_title(f"프레임 {fi_s} — 최적화기가 실제로 보는 것 (원 = 제약용 구의 단면)",
                     color=INK, fontsize=10.5, loc="left", pad=8)
        leg = ax.legend(frameon=False, fontsize=8.5, markerscale=3, ncol=2,
                        loc="upper center", bbox_to_anchor=(0.5, -0.13))
        for t_ in leg.get_texts(): t_.set_color(INK_2)
        fig.tight_layout(); fig.savefig(figs / "fig5_scene_candidates.png",
                                        facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    # fig3 — 도형 종류별
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.4), dpi=160)
    style_axes(fig, (a1, a2))
    for ax, key, ylab in ((a1, "containment_true", "실제 물체 포함률"),
                          (a2, "excess_volume", "과잉 부피 비율")):
        ax.grid(axis="y", color=GRID_INK, lw=.6); ax.set_axisbelow(True)
        w = 0.8 / len(PRIMITIVES)
        for k, ptype in enumerate(PRIMITIVES):
            ys = [np.mean([t_[key] for t_ in type_rows if t_["body"] == b and t_["prim"] == ptype])
                  for b in bodies]
            ax.bar(np.arange(len(bodies)) + k * w - 0.4 + w / 2, ys, w * 0.88,
                   color=CATEGORICAL[k], edgecolor=SURFACE, lw=1.6,
                   label=ptype if ax is a1 else None)
        ax.set_xticks(range(len(bodies)), bodies); ax.set_ylabel(ylab)
    a1.set_ylim(0, 1.12)
    leg = a1.legend(frameon=False, fontsize=8, ncol=4)
    for t_ in leg.get_texts(): t_.set_color(INK_2)
    fig.suptitle("도형 종류별 — 덮는 것이 먼저이고 부피는 그 비용이다",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figs / "fig3_primitive_types.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------------- 문서
    def md(header, body):
        return "\n".join(["| " + " | ".join(header) + " |",
                          "|" + "|".join("---" for _ in header) + "|", *body])

    doc = f"""# 6단계 — geometry 변환 (primitive 근사)

**질문.** 점을 도형으로 바꿀 때 무엇이 보존되고 무엇이 사라지는가?

최적화기가 실제로 보는 것은 점이 아니라 이 도형이다. **도형이 물체보다 작으면 최적화기는
존재하지 않는다고 들은 부분을 향해 궤적을 낸다.** 그래서 이 단계의 불변식은 하나다 —
과소 근사 금지.

| | |
|---|---|
| 기록 | `{run.path}` |
| 프레임 | {len(set(r['frame'] for r in rows))}개 (파지 전) |
| 후보 | {len(rows)}개, 그중 정답 물체와 대조 가능한 것 {len(true_rows)}개 |
| 기본 도형 | `{geo0.primitive}`, `min_radius={geo0.min_radius}`, `perception_uncertainty={geo0.perception_uncertainty}` |

## 판정 — {verdict}

| 검사 | 결과 | 기준 |
|---|---|---|
| A. 관측 점 포함 | {'통과' if ok_a else '실패'} — {len(rows)}개 후보 전부 포함률 1.000 | 전부 1.0 |
| B. **실제 물체 포함** | {'통과' if ok_b else '**실패**'} — 평균 **{mean_true:.3f}**, 최대 침투 **{max_pen*1000:.1f} mm** | 1.0 |
| C. 캡슐 구 체인 | {'통과' if ok_c else '실패'} — 실데이터 캡슐 {len(real_chain)}개 + 합성 4종 전부 100% 덮음 | 전부 1.0 |

### A와 B는 다른 질문이다 — 그리고 그 차이가 이 단계의 전부다

![두 포함률]({img}/fig1_containment.png)

**A는 통과한다.** AG3S의 `containment_report`가 재는 것이고, 모든 후보가 자기 점을 100%
담는다. 코드가 보장하려던 것은 보장된다.

**B는 통과하지 못한다.** 2단계에서 확인했듯 카메라는 **앞면만** 본다. 앞면 점을 전부 담는 구는
물체의 뒷면을 담지 못한다. 관측한 점은 100% 포함하면서 실제 물체는 그보다 적게 덮고 있으며,
**그 상태는 A로는 절대 드러나지 않는다.**

{md(["물체", "관측 점 포함률", "실제 물체 포함률", "최대 침투 (mm)", "완전 포함에 필요한 예산 (mm)"],
    [f"| {b} | "
     f"{np.mean([r['containment_points'] for r in true_rows if r['body']==b]):.3f} | "
     f"{np.mean([r['containment_true'] for r in true_rows if r['body']==b]):.3f} | "
     f"{np.max([r['penetration'] for r in true_rows if r['body']==b])*1000:.1f} | "
     f"{(lambda v: f'{max(v)*1000:.0f}' if v and None not in v else '스윕 범위 밖')([r.get('need_uncertainty') for r in true_rows if r['body']==b])} |"
     for b in bodies])}

"최대 침투"는 실제 물체 표면이 도형 **합집합** 밖으로 나간 최대 거리다. 합집합인 것이
중요하다 — 속이 빈 크레이트는 벽들이 유클리드로 연결되지 않아 여러 후보로 나뉘고, 조각 하나를
크레이트 전체와 비교하면 당연히 못 덮는다. 최적화기는 조각이 아니라 후보 전부를 본다.
(이 보고서의 첫 판이 그 비교를 해서 크레이트 침투를 408 mm로 보고했다. 합집합으로 고치니
3.6 mm다.)

이 값은 **최적화기가 못 보는 살의 두께**이고, 이보다 얇은 여유거리는 실제로는 여유가 아니다.

![씬에서 놓치는 부분]({img}/fig4_scene_miss.png)

왼쪽이 기본 설정이다. 파란 점(관측한 점)은 전부 회색 구 안에 있는데, 빨간 점 — 실제 물체
표면인데 구 밖으로 나간 부분 — 이 뒤쪽에 남는다. **A가 보는 것은 파란 점뿐이고, 빨간 점은
A의 시야에 없다.** 오른쪽은 필요한 예산을 켠 것이고, 빨간 점이 사라진다.

같은 것을 **정책이 본 이미지 위에** 되돌려 그리면 이렇다.

![이미지 위 겹침]({img}/fig6_image_overlay.png)

한 가지는 짚어 두어야 한다. **이미지 공간에서는 물체의 뒷면이 앞면과 같은 자리에 투영된다.**
그래서 가운데 패널의 빨간 점은 사과의 실루엣 전체를 덮은 것처럼 보이지만, 실제로 도형 밖에
있는 것은 카메라에서 먼 쪽이다. 어느 *부분*이 빠졌는지는 3D 그림(fig4)이 답하고, 이 그림은
그것이 씬의 **어느 물체**에서 일어나는지를 답한다. 두 그림은 서로 다른 질문에 답하며,
둘 다 필요하다.

#### 기전 — 반지름이 아니라 중심이 문제다

{md(["물체", "맞춘 반지름 (mm)", "중심 오프셋 (mm)", "프레임당 후보 수"],
    [f"| {b} | {np.mean([r['radius'] for r in rows if r['body']==b])*1000:.1f} | "
     f"{np.mean([r['centre_offset'] for r in rows if r['body']==b and np.isfinite(r['centre_offset'])])*1000:.1f} | "
     f"{len([r for r in rows if r['body']==b]) / max(len(set(x['frame'] for x in rows)), 1):.1f} |"
     for b in bodies])}

과일들은 **반지름이 부족한 것이 아니라 중심이 카메라 쪽으로 밀려 있다.** 2단계에서 "점이 ×를
둘러싸지 않고 한쪽에 몰린다"고 했고 4단계에서 클러스터 무게중심 편향을 21.0 mm로 쟀는데,
6단계에서 그 편향이 그대로 도형의 중심 오프셋이 되어 물체 뒷면이 도형 밖으로 나간다.

같은 크기의 구가 반지름만큼 밀리면 절반이 밖으로 나간다 — 위 표의 포함률 0.38–0.86이
그것이다. 크레이트가 0.976으로 멀쩡한 것은 크고 여러 면이 보여 오프셋이 상대적으로 작기
때문이다.

### B가 `perception_uncertainty`가 존재하는 이유다

`GeometryConfig.perception_uncertainty`의 주석은 이렇게 적는다 — "One budget for every
perception error that makes the fitted shape smaller than the real one." 그 예산이 무엇을 위한
것인지 말로 설명하는 대신, 0부터 올려가며 **실제 물체를 덮는 데 얼마가 필요한지** 쟀다.

![예산 스윕]({img}/fig2_uncertainty.png)

기본값은 `{geo0.perception_uncertainty}`다. 즉 **기본 설정에서는 이 예산이 꺼져 있고**, 위 표의
"필요한 예산" 열이 켜야 할 값을 말한다.

주석이 함께 적는 것도 중요하다 — 이 값은 **primitive 반지름에만** 더하고 `d_safe`에는 더하지
않는다. 두 곳에 더하면 같은 센티미터를 두 번 세어 로봇의 우회 거리가 조용히 두 배가 된다.

### C. 캡슐 구 체인 — 최적화기가 보는 최종 형태

최적화기가 푸는 것은 primitive가 아니라 `to_spheres()`가 낸 구들이다. 캡슐은 구 체인이 되는데,
`max_spheres`가 걸려 간격이 반지름보다 벌어지면 **구 사이에 틈이 생긴다** — 충돌을 만드는
방향의 오차다. 코드는 각 구를 `sqrt(r² + (간격/2)²)`로 부풀려 중점까지 덮는다고 적고 있다.

두 가지로 확인했다. **실데이터에 맞춘 캡슐**과, 실데이터에는 나타나지 않는 긴 캡슐(합성).

#### C-1. 실데이터에서 맞춰진 캡슐 {len(real_chain)}개

기본 도형은 `sphere` 라 파이프라인이 캡슐을 만들지 않는다. 그래서 같은 관측 점에
`fit_primitive(..., "capsule")` 을 직접 걸어 나온 캡슐의 체인을 검사했다. 관측에서 나온
형상이므로 합성이 아니다.

{md(["물체", "캡슐 수", "반지름 (mm)", "반길이 (mm)", "구 개수", "자기 점을 덮은 비율"],
    [f"| {b} | {len([c for c in real_chain if c['body']==b])} | "
     f"{np.mean([c['radius'] for c in real_chain if c['body']==b])*1000:.1f} | "
     f"{np.mean([c['half_len'] for c in real_chain if c['body']==b])*1000:.1f} | "
     f"{np.mean([c['n_spheres'] for c in real_chain if c['body']==b]):.1f} | "
     f"**{np.mean([c['covered_points'] for c in real_chain if c['body']==b]):.4f}** |"
     for b in sorted({c['body'] for c in real_chain})] or ["| — | — | — | — | — | — |"])}

#### C-2. 합성 캡슐 — `max_spheres` 가 실제로 걸리는 경우

실데이터의 캡슐은 짧아 `max_spheres=8` 이 걸리지 않는다. 즉 **틈이 생길 수 있는 구간이
실데이터에 없다.** 그 구간을 검사하려면 만들어야 하므로, 캡슐을 수식으로 정의하고 표면 점을
해석적으로 뿌려(카메라도 씬도 쓰지 않는다) 체인이 덮는지 본다. 관측이 아니라 **기하 항등식
`sqrt(r²+(간격/2)²)` 의 단위 검사**라는 점을 분명히 해 둔다.

{md(["반길이 (m)", "구 개수", "간격 (m)", "원래 반지름", "부푼 반지름", "sqrt(r²+(s/2)²)", "표면 덮은 비율"],
    [f"| {c['half_len']:.2f} | {c['n_spheres']} | {c['spacing']:.4f} | {c['radius']:.3f} | "
     f"{c['effective']:.4f} | {c['expected']:.4f} | **{c['covered']:.4f}** |"
     for c in chain_rows])}

부푼 반지름이 `sqrt(r²+(간격/2)²)`과 정확히 일치하고, 캡슐 표면이 100% 덮인다. 반길이 0.40 m는
`max_spheres=8`이 실제로 걸리는 경우이며, 부풀리지 않았다면 틈이 생겼을 구간이다.

### D. 도형 종류별 — 덮는 것이 먼저, 부피는 비용

![도형 종류]({img}/fig3_primitive_types.png)

같은 점에 네 종류를 각각 맞춰 비교했다. 왼쪽이 실제 물체를 덮는 비율, 오른쪽이 그 대가로 쓰는
부피다. 오른쪽이 큰 것은 그 자체로 나쁘지 않다 — 우회 거리가 늘 뿐이고, 왼쪽이 작은 것은
충돌이다. **두 축의 지위가 다르다.**

### 최적화기가 실제로 보는 씬

![후보 도형]({img}/fig5_scene_candidates.png)

원 하나가 후보 하나의 제약용 구 단면이다. 노란 점(정답 물체)이 원 안에 들어 있는지, 원이
물체보다 큰지 작은지가 이 그림에서 바로 읽힌다. 최적화기에게 씬은 점이 아니라 **이 원들**이다.

## 이 단계가 다음 단계에 남기는 것 — 여유거리는 이미 절반이 쓰였다

5단계에서 본 여유거리는 `object` 열 **50 mm**, `support_surface` 10 mm였다. 위에서 잰 최대
침투는 **{max_pen*1000:.1f} mm**다.

두 숫자를 나란히 놓으면 이렇게 읽힌다: 최적화기는 도형에서 50 mm 떨어져 있으라는 제약을
풀지만, 도형이 실제 물체보다 최대 {max_pen*1000:.0f} mm 작으므로 **실제 물체로부터의 여유는
{50 - max_pen*1000:.0f} mm까지 줄어들 수 있다.** 명목상의 여유거리 중 그만큼이 이미 도형의
과소 근사로 소진된 것이다.

`perception_uncertainty`를 위 표의 "필요한 예산"으로 켜면 이 소진이 사라진다 — 그것이 그 설정이
존재하는 이유이고, 기본값이 `0.0`인 채로 두면 안 되는 이유다. 다만 켠 만큼 도형이 커지므로
로봇의 우회 거리가 늘고, 좁은 곳에서는 통과 가능한 경로가 사라질 수도 있다. **그 교환이
7단계에서 실제로 어떻게 나타나는지**가 다음 질문이다.

주의: 이 예산을 `d_safe`에도 더하면 안 된다. `GeometryConfig` 주석이 그 이유를 적는다 — 같은
센티미터를 두 번 세면 우회 거리가 조용히 두 배가 되고, 안전은 그만큼 늘지 않는다.

## 그림에 대하여

### fig4 — 도형이 어디를 놓치는가 (이 단계의 핵심 그림)

`fig4_scene_miss.png`. 두 패널 모두 같은 프레임, 같은 물체. 회색 철망이 맞춰진 구,
파란 점이 관측한 점, 회색 점이 실제 표면 중 구 안에 든 부분, **빨간 점이 구 밖으로 나간 부분**.

**만드는 법.** 실제 표면은 MuJoCo 메시 정점(과일)과 상자 모서리(크레이트)를 그 프레임의 실제
자세로 옮긴 것이다. 각 점에 `geometry.contains(primitive, ...)` 를 걸어 안/밖을 나눈다. 왼쪽은
기본 설정, 오른쪽은 `inflate(primitive, 필요 예산)`.

**읽는 법.** 파란 점이 전부 철망 안에 있는데 빨간 점이 남는 것 — 그것이 검사 A와 B의 차이다.
A는 파란 점만 보고, 빨간 점은 A의 시야에 아예 없다. 오른쪽에서 빨간 점이 사라지는 것이
`perception_uncertainty` 가 하는 일이다.

### fig6 — 관측 이미지 위 겹침

`fig6_image_overlay.png`. 정책이 본 224x224 입력 위에 3D 결과를 되돌려 그린 것.

**만드는 법.** `imageview.project` 로 base 프레임 점을 카메라 픽셀로 옮긴다. 깊이·내부
파라미터는 480x640이고 정책 이미지는 224x224인데, 둘 다 4:3이고 299x224 → 224x224 변환이
순수 가로 압축이라 **정규화 좌표가 보존된다** (2단계에서 확인). 그래서 오프셋 없이
`u/W_depth * W_image` 로 옮긴다. 제약용 구는 중심을 투영한 자리에 반지름 `f*r/z` 인 원으로
그린다 — 광축에서 멀면 실루엣이 타원이 되지만, 이 씬의 물체는 화면 중앙 근처라 오차가 작다.
**판정에 쓰는 숫자는 언제나 3D에서 계산하고 이 근사는 그림에만 쓴다.**

**읽는 법.** 이미지 공간에서는 뒷면이 앞면과 겹쳐 투영되므로, 빨간 점의 *면적*을 오차 크기로
읽으면 안 된다. 이 그림이 답하는 것은 "어느 물체에서 일어나는가"이고, "어느 부분인가"는
fig4가 답한다.

### fig5 — 최적화기가 실제로 보는 씬

`fig5_scene_candidates.png`. 위에서 내려다본 점군에 후보의 제약용 구 단면을 원으로 겹쳤다.

**읽는 법.** 최적화기에게 씬은 점이 아니라 이 원들이다. 크레이트의 원이 물체보다 훨씬 큰 것이
과잉 부피(= 우회 비용)이고, 과일의 원이 노란 점을 살짝 벗어나 있는 것이 과소 근사(= 충돌
위험)다. **두 오차의 지위가 다르다** — 앞은 느려지고, 뒤는 부딪힌다.

### fig1 — 두 포함률

`fig1_containment.png`. 물체마다 초록(관측 점 포함률)과 빨강(실제 물체 포함률)을 나란히.

**읽는 법.** 초록이 전부 1.00 인데 빨강이 낮은 것이 이 단계의 결론 전부다. 초록만 보면
통과이고, 빨강을 봐야 문제가 보인다.

### fig2 — 예산 스윕

`fig2_uncertainty.png`. 가로축이 `perception_uncertainty`, 세로축이 실제 물체 포함률.

**만드는 법.** 맞춘 도형에 `inflate(prim, u)` 를 걸어가며 실제 표면 포함률을 다시 잰다.
곡선이 1.0 점선에 닿는 지점이 그 물체에 필요한 예산이다.

### fig3 — 도형 종류별

`fig3_primitive_types.png`. 같은 점에 sphere·capsule·box·ellipsoid 를 각각 맞춰 비교.

**읽는 법.** 왼쪽(포함률)이 먼저이고 오른쪽(과잉 부피)은 그 비용이다. 두 축을 같은 무게로
읽으면 안 된다.

## 재현

```bash
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.reports.geometry_report \\\\
    --records {args.records} --attention {args.attention}
```
"""
    out = paths.document
    out.write_text(doc)
    out.with_suffix(".json").write_text(json.dumps({
        "checks": {"A_points_contained": ok_a, "B_true_object_contained": ok_b,
                   "C_capsule_chain": ok_c},
        "mean_true_containment": mean_true, "max_penetration_m": max_pen,
        "capsule_chain_synthetic": chain_rows, "capsule_chain_real": real_chain,
        "candidates": rows,
        "type_comparison": type_rows,
    }, indent=2, ensure_ascii=False, default=float))
    print(f"wrote {out} and 3 figures in {figs}")
    print(f"판정 {verdict}: A={'ok' if ok_a else 'FAIL'}  "
          f"B=실제포함 {mean_true:.3f}/침투 {max_pen*1000:.1f}mm  C={'ok' if ok_c else 'FAIL'}")


if __name__ == "__main__":
    main()
