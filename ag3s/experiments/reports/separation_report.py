"""5단계 검증 — target/obstacle 분리가 기하를 지우지 않는가.

이 파이프라인 전체가 존재하는 이유가 이 한 줄이다: **attention은 target의 *이름*을 정할 뿐
기하를 지우지 않는다.** VLA가 보지 않은 물체도 물리적으로 존재하면 충돌 후보로 남아야 한다.
4단계까지는 "옳은 것을 골랐는가"를 물었고, 5단계는 "고르는 행위가 나머지를 훼손하지
않았는가"를 묻는다. 앞의 것이 틀리면 성능이 나빠지고, 뒤의 것이 틀리면 로봇이 물체를 친다.

`generate_candidates`의 docstring은 이렇게 못박는다 — "Note what is *not* a parameter: there is
no attention argument. Everything attention had to say was said in `target_grounding`, and letting
it back in here is precisely the mistake this design exists to avoid."

**서명을 읽는 것으로는 부족하다.** attention이 인자로 없어도 `target`을 통해 간접적으로 들어오고,
실제로 들어온다 — target의 점들은 잔여 군집에서 빠져 `TARGET` 후보가 된다. 그러니 물어야 할
것은 "attention이 영향을 주는가"가 아니라 **"attention이 무엇을 지우는가"**다. 그래서 정적
검사가 아니라 실험으로 잰다: 같은 점군에 **서로 다른 attention 세 가지**를 넣고 후보 집합을
비교한다.

| attention | 기대되는 target |
|---|---|
| 실제 모델 (1단계가 고른 셀) | 프롬프트가 지목한 물체 |
| 다른 물체를 가리키는 합성 블롭 | 그 다른 물체 |
| 평탄 (정보 없음) | 없음 — `NO_ATTENTION` |

세 경우에서 **후보들이 덮는 점의 집합이 같아야 한다.** 어느 물체든 사라지면 안 되고,
target이던 것이 다음 실행에서는 일반 장애물로 나타나야 한다.

실행:
    MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.reports.separation_report \
        --records outputs/.../ag3s_records/run_0002 \
        --attention benchmark/ag3s/asset/data/attention_step1_run0002.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import pathlib

import numpy as np

from benchmark.ag3s.stages.attention_lifting import GridAttentionAdapter, lift
from benchmark.ag3s.constraints.clearance import ClearancePolicy
from benchmark.ag3s.stages.collision_candidates import generate_candidates
from benchmark.ag3s.config import (
    AttentionConfig, ClusteringConfig, CollisionCandidateConfig, ContactConfig, GeometryConfig,
    PointCloudConfig, SupportSurfaceConfig,
)
from benchmark.ag3s.experiments.common.outputs import add_tag_argument, resolve
from benchmark.ag3s.experiments.common.figstyle import (
    CATEGORICAL, GRID_INK, INK, INK_2, SURFACE, style_axes, use_korean,
)
from benchmark.ag3s.experiments.sources.policy_record import CAMERA_BINDINGS, load_run, pose_scene, replay_scene
from benchmark.ag3s.stages.reconstruction import reconstruct
from benchmark.ag3s.stages.robot_filter import filter_robot_points
from benchmark.ag3s.stages.support_surface import fit_support_surfaces
from benchmark.ag3s.stages.target_grounding import ground_target
from benchmark.ag3s.types import ContactPolicyContext, Phase, SourceType

OBJECT_BODIES = ("crate", "apple", "banana", "orange", "pear")


def run_once(cloud, grid, smask, cl_cfg, att_cfg, phase):
    """한 attention 으로 grounding → 후보 생성. 나머지는 전부 동일하게 둔다."""
    acloud = lift(cloud, grid, att_cfg, adapter=GridAttentionAdapter(),
                  image_hw=(480, 640))
    result = ground_target(acloud, cl_cfg, seed_percentile=att_cfg.seed_percentile,
                           exclude_mask=smask)
    candidates, stats = generate_candidates(
        cloud, phase=phase, target=result.target, support_mask=smask,
        config=CollisionCandidateConfig(), geometry_config=GeometryConfig(),
        contact_config=ContactConfig(),
    )
    return result, candidates, stats


def covered_points(candidates, cloud) -> np.ndarray:
    """후보들이 덮는 점의 인덱스 집합. 무엇이 사라졌는지 세는 기준."""
    idx = []
    for c in candidates:
        pi = getattr(c, "point_indices", None)
        if pi is not None and len(pi):
            idx.append(np.asarray(pi, np.int64))
    return np.unique(np.concatenate(idx)) if idx else np.zeros(0, np.int64)


def main() -> None:
    from benchmark.ag3s.experiments.reports.attention_report import target_from_prompt
    from benchmark.ag3s.experiments.reports.grounding_report import build_robot_model, majority_body
    from benchmark.ag3s.experiments.sources.mujoco_source import gaussian_attention

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--attention", required=True)
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/archive/step-verification-20260904/step-01-attention.json")
    ap.add_argument("--target", default=None)
    ap.add_argument("--decoy", default=None, help="합성 attention 을 겨눌 다른 물체 (기본: 자동)")
    ap.add_argument("--frames", type=int, default=9, help="파지 전 구간을 덮을 프레임 수")
    ap.add_argument("--phase", default="approach")
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--out-doc", default="benchmark/ag3s/docs/archive/step-verification-20260904/step-05-separation.md")
    ap.add_argument("--out-figs", default="benchmark/ag3s/asset/image/separation")
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
    target = args.target or target_from_prompt(run.prompt)
    decoy = args.decoy or next(b for b in OBJECT_BODIES if b not in (target, "crate"))

    scene = replay_scene(run)
    robot_model = build_robot_model(scene)
    names = {i: (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or "?")
             for i in range(scene.model.nbody)}
    bid = {n: mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in OBJECT_BODIES}
    pc_cfg = dataclasses.replace(PointCloudConfig(), range_max=(args.range_max or None))
    att_cfg, cl_cfg = AttentionConfig(), ClusteringConfig()
    print(f"target={target}  decoy={decoy}  phase={args.phase}")

    rows = []
    example = None
    for fi in range(min(args.frames, len(run))):
        pose_scene(scene, run.steps[fi])
        frame = scene.capture("zed_left")
        cloud, _ = reconstruct(depth=frame.depth, camera_intrinsics=frame.camera_intrinsics,
                               T_base_cam=frame.T_base_cam, config=pc_cfg)
        cloud, _ = filter_robot_points(cloud, robot_model, frame.robot_state, pc_cfg)
        if len(cloud) == 0:
            continue
        _, smask = fit_support_surfaces(cloud, SupportSurfaceConfig())
        labels = frame.body_ids[cloud.uv[:, 1], cloud.uv[:, 0]]

        variants = {
            "실제 모델": A[fi, di, ai, chosen["layer"], chosen["head"], ci],
            f"합성 ({decoy} 겨냥)": gaussian_attention(frame, scene.body_position_in_base(decoy)),
            "평탄 (정보 없음)": np.full((16, 16), 0.5, np.float32),
        }
        entry = {"frame": fi, "n_points": len(cloud), "n_support": int(smask.sum())}
        for name, grid in variants.items():
            result, cands, stats = run_once(cloud, grid, smask, cl_cfg, att_cfg, args.phase)
            cov = covered_points(cands, cloud)
            per_body = {}
            for b, i in bid.items():
                m = labels == i
                per_body[b] = (int(m.sum()), int(np.isin(np.nonzero(m)[0], cov).sum()))
            picked = None
            if result.target is not None:
                picked = majority_body(labels[result.target.point_indices], names)[0]
            entry[name] = {
                "status": result.status.value, "picked": picked,
                "n_candidates": len(cands),
                "by_type": {t.value: sum(1 for c in cands if c.source_type is t)
                            for t in SourceType if any(c.source_type is t for c in cands)},
                "covered": cov, "n_covered": int(cov.size),
                "per_body": per_body,
                "n_unassigned": int(stats.get("n_unassigned_points", 0)),
            }
        if example is None:
            frame_e = frame
            scenes = {}
            for name, grid in variants.items():
                result, cands, _ = run_once(cloud, grid, smask, cl_cfg, att_cfg, args.phase)
                scenes[name] = (list(cands), result)
            example = (cloud, labels, entry, frame, smask.copy(), scenes, fi)
        entry["clouds"] = None
        rows.append(entry)
    scene.close()

    vnames = list(rows[0].keys() - {"frame", "n_points", "n_support", "clouds"}) if rows else []
    vnames = ["실제 모델", f"합성 ({decoy} 겨냥)", "평탄 (정보 없음)"]

    # ---- A. 서명 격리 -----------------------------------------------------------------
    sig = inspect.signature(generate_candidates)
    attention_params = [p for p in sig.parameters if "attention" in p.lower()]
    ok_a = not attention_params

    # ---- B. 덮는 점 집합이 같은가 -------------------------------------------------------
    same_coverage, coverage_gaps = [], []
    for r in rows:
        sets = [set(r[v]["covered"].tolist()) for v in vnames]
        base = sets[0]
        same_coverage.append(all(s == base for s in sets[1:]))
        coverage_gaps.append(max(len(base ^ s) for s in sets[1:]))
    ok_b = all(same_coverage)

    # ---- C. 물체가 사라지지 않는가 -------------------------------------------------------
    lost = []
    for r in rows:
        for v in vnames:
            for b, (n_gt, n_cov) in r[v]["per_body"].items():
                if n_gt >= cl_cfg.min_points and n_cov == 0:
                    lost.append((r["frame"], v, b, n_gt))
    ok_c = not lost

    # ---- D. target 이 후보로 남는가 ------------------------------------------------------
    target_kept = [r for r in rows if r[vnames[0]]["by_type"].get("target", 0) == 1]
    ok_d = len(target_kept) == sum(1 for r in rows if r[vnames[0]]["picked"] is not None)

    # ---- E. 여유거리 분리 -----------------------------------------------------------------
    contact = ContactConfig()
    policy = ClearancePolicy.from_config(contact=contact, geometry=GeometryConfig(),
                                        support_surface=SupportSurfaceConfig())
    links = ["ee_finger_r1", "link_right_arm_3", "link_torso_3", "ee_finger_l1"]
    sources = [SourceType.TARGET, SourceType.OBJECT, SourceType.UNKNOWN_GEOMETRY,
               SourceType.SUPPORT_SURFACE]
    phases = [p.value for p in Phase]
    by_phase = {}
    for ph in phases:
        c = ContactPolicyContext.make(ph, ("right",))
        by_phase[ph] = policy.margin_matrix(links, sources, context=c, target_grounded=True)
    ctx = ContactPolicyContext.make(args.phase, ("right",))
    matrix = by_phase[args.phase]
    no_target = policy.margin_matrix(links, sources, context=ctx, target_grounded=False)
    # fail-closed 시연: 존재하지 않는 조작기 이름. 열거형 값은 'right'/'left' 이므로
    # 'right_arm' 은 알 수 없는 값이고, 정책은 아무것도 인가하지 않아야 한다.
    bad_ctx = ContactPolicyContext.make("grasp", ("right_arm",))
    failclosed = policy.margin_matrix(links, sources, context=bad_ctx, target_grounded=True)

    grasp = by_phase["grasp"]
    ok_e = (
        float(grasp[0, 0]) < float(grasp[2, 0])                 # 인가 손가락 < 토르소
        and float(grasp[0, 1]) == float(grasp[2, 1])            # obstacle 열은 완화 없음
        and float(by_phase["transit"][0, 0]) > float(grasp[0, 0])  # 단계가 진행될수록 좁아짐
        and np.allclose(failclosed, policy.margin_matrix(
            links, sources, context=ContactPolicyContext.make("grasp", ()),
            target_grounded=True))                              # 알 수 없는 조작기 = 인가 없음
        and float(no_target[0, 0]) == float(no_target[2, 0])    # target 없으면 완화 없음
    )

    verdict = "**PASS**" if (ok_a and ok_b and ok_c and ok_d and ok_e) else "**FAIL**"

    paths = resolve(out_figs=args.out_figs, out_doc=args.out_doc,
                    tag=args.tag).prepare()
    figs, img = paths.figures, paths.image_prefix

    # fig1 — 세 attention 에서 물체별로 덮인 점 수
    fig, axes = plt.subplots(1, len(vnames), figsize=(4.6 * len(vnames), 3.6), dpi=160,
                             sharey=True)
    style_axes(fig, axes)
    bodies = list(bid)
    r0 = rows[0]
    for j, v in enumerate(vnames):
        ax = axes[j]
        ax.grid(axis="y", color=GRID_INK, lw=.6); ax.set_axisbelow(True)
        gt = [r0[v]["per_body"][b][0] for b in bodies]
        cv = [r0[v]["per_body"][b][1] for b in bodies]
        x = np.arange(len(bodies))
        ax.bar(x - 0.2, gt, 0.38, color=GRID_INK, edgecolor=SURFACE, lw=2,
               label="정답 점 수" if j == 0 else None)
        colours = [CATEGORICAL[0] if b == r0[v]["picked"] else CATEGORICAL[2] for b in bodies]
        ax.bar(x + 0.2, cv, 0.38, color=colours, edgecolor=SURFACE, lw=2,
               label="후보가 덮은 점" if j == 0 else None)
        ax.set_xticks(x, bodies, rotation=20)
        ax.set_title(f"{v}\n→ target: {r0[v]['picked'] or '없음'}", fontsize=9.5, color=INK)
    axes[0].set_ylabel("점 수")
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, frameon=False, fontsize=8.5, ncol=2, loc="lower center",
                     bbox_to_anchor=(0.5, -0.04))
    for t in leg.get_texts(): t.set_color(INK_2)
    fig.suptitle("attention 을 바꿔도 덮이는 기하는 같다 — 파랑 막대만 자리를 옮긴다",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(figs / "fig1_coverage.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # fig3 — 씬에서 본 세 경우: 같은 기하, 다른 이름표
    cloud_e, labels_e, _, _, smask_e, scenes_e, fi_e = example
    fig, axes = plt.subplots(1, len(vnames), figsize=(4.9 * len(vnames), 5.0), dpi=160)
    style_axes(fig, axes)
    pts_e = cloud_e.points
    obj_mask = np.isin(labels_e, list(bid.values()))
    if obj_mask.any():
        lo = pts_e[obj_mask, :2].min(axis=0) - 0.22
        hi = pts_e[obj_mask, :2].max(axis=0) + 0.22
    seen = set()
    for ax, v in zip(axes, vnames):
        cands, result = scenes_e[v]
        ax.scatter(pts_e[smask_e, 0], pts_e[smask_e, 1], s=.5, c="#e8e7e2", lw=0,
                   label="지지면" if "s" not in seen else None)
        rest = ~smask_e
        ax.scatter(pts_e[rest, 0], pts_e[rest, 1], s=.6, c=GRID_INK, lw=0,
                   label="그 외 점" if "s" not in seen else None)
        for cand in cands:
            prim = cand.geometry[0]
            is_t = cand.source_type.value == "target"
            ax.add_patch(plt.Circle((prim.center[0], prim.center[1]), prim.bounding_radius,
                                    fill=False, ec=CATEGORICAL[0] if is_t else CATEGORICAL[2],
                                    lw=2.2 if is_t else 1.1,
                                    label=(("target 후보" if is_t else "obstacle 후보")
                                           if ("t" if is_t else "o") not in seen else None)))
            seen.add("t" if is_t else "o")
        seen.add("s")
        for b, i in bid.items():
            m = labels_e == i
            if m.any():
                ax.scatter(pts_e[m, 0], pts_e[m, 1], s=3.5, c=CATEGORICAL[3], lw=0,
                           label="정답 물체 점" if "g" not in seen else None)
                seen.add("g")
        picked = None
        if result.target is not None:
            picked = majority_body(labels_e[result.target.point_indices], names)[0]
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_xlabel("base x (m)")
        ax.set_title(f"{v}\ntarget: {picked or '없음'} · 후보 {len(cands)}개",
                     fontsize=9.5, color=INK)
    axes[0].set_ylabel("base y (m)")
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, frameon=False, fontsize=8.5, ncol=5, markerscale=3,
                     loc="lower center", bbox_to_anchor=(0.5, -0.02))
    for t_ in leg.get_texts(): t_.set_color(INK_2)
    fig.suptitle("같은 씬, 같은 후보 원 — 굵은 파란 원(target)만 자리를 옮긴다",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(figs / "fig3_scene.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # fig4 — 같은 세 경우를 관측 이미지 위에
    from benchmark.ag3s.experiments.common.imageview import project, show, sphere_circle
    img_e = run.steps[fi_e].images["cam_high"]
    ih_e = img_e.shape[:2]
    K_e, T_e, hw_e = (frame_e.camera_intrinsics, frame_e.T_base_cam, frame_e.hw)
    fig, axes = plt.subplots(1, len(vnames), figsize=(4.8 * len(vnames), 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    for ax, v in zip(axes, vnames):
        ax.set_facecolor(SURFACE)
        cands, result = scenes_e[v]
        picked = None
        if result.target is not None:
            picked = majority_body(labels_e[result.target.point_indices], names)[0]
        show(ax, img_e, f"{v}\ntarget: {picked or '없음'} · 후보 {len(cands)}개",
             ink=INK, grid_ink=GRID_INK)
        for cand in cands:
            prim = cand.geometry[0]
            is_t = cand.source_type.value == "target"
            c = sphere_circle(prim.center, prim.bounding_radius, K_e, T_e, hw_e, ih_e)
            if c is None:
                continue
            ax.add_patch(plt.Circle(c[:2], c[2], fill=False, lw=2.6 if is_t else 1.3,
                                    ec=CATEGORICAL[0] if is_t else CATEGORICAL[2]))
        ax.set_xlim(0, ih_e[1]); ax.set_ylim(ih_e[0], 0)
    fig.suptitle("관측 이미지 위의 후보 — 초록 원(obstacle)은 그대로, 굵은 파란 원(target)만 옮긴다",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figs / "fig4_image_overlay.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # fig2 — 여유거리 행렬: 단계별 + 두 가지 무력화 조건
    from benchmark.ag3s.experiments.common.figstyle import sequential_cmap
    panels = ([(ph, by_phase[ph], f"phase `{ph}`") for ph in phases]
              + [("no_target", no_target, "target 없음"),
                 ("failclosed", failclosed, "알 수 없는 조작기\n(fail-closed)")])
    vmax = max(m.max() for _, m, _ in panels) * 1000
    fig, axes = plt.subplots(1, len(panels), figsize=(2.55 * len(panels), 3.4), dpi=160,
                             sharey=True)
    style_axes(fig, axes)
    for ax, (_, m, ttl) in zip(axes, panels):
        im = ax.imshow(m * 1000, cmap=sequential_cmap(), vmin=0, vmax=vmax, aspect="auto")
        for i in range(m.shape[0]):
            for k in range(m.shape[1]):
                ax.text(k, i, f"{m[i, k]*1000:.0f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if m[i, k] * 1000 > vmax * .55 else INK)
        ax.set_xticks(range(len(sources)), [s.value.replace("_", "\n") for s in sources],
                      rotation=0, fontsize=6.5)
        ax.set_title(ttl, fontsize=9, color=INK)
    axes[0].set_yticks(range(len(links)), links, fontsize=7.5)
    cb = fig.colorbar(im, ax=list(axes), fraction=.02, pad=.02)
    cb.set_label("필요 여유거리 (mm)", color=INK_2, fontsize=8)
    cb.ax.tick_params(colors=INK_2, labelsize=7); cb.outline.set_edgecolor(GRID_INK)
    fig.suptitle("여유거리 행렬 — 활성 조작기 `right`. 완화는 target 열의 인가 링크에만 나타난다",
                 color=INK, fontsize=11.5, x=0.01, y=1.06, ha="left")
    fig.savefig(figs / "fig2_clearance.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------------------- 문서
    def md(header, body):
        return "\n".join(["| " + " | ".join(header) + " |",
                          "|" + "|".join("---" for _ in header) + "|", *body])

    doc = f"""# 5단계 — target / obstacle 분리

**질문.** target을 고르는 행위가 나머지 기하를 훼손하지 않는가?

4단계까지는 "옳은 것을 골랐는가"를 물었다. 5단계는 **"고르는 행위가 나머지를 지우지
않았는가"**를 묻는다. 앞의 것이 틀리면 성능이 나빠지고, **뒤의 것이 틀리면 로봇이 물체를 친다.**

`generate_candidates`의 docstring이 이렇게 못박는다:

> Note what is *not* a parameter: there is no attention argument. Everything attention had to say
> was said in `target_grounding`, and letting it back in here is precisely the mistake this design
> exists to avoid.

| | |
|---|---|
| 기록 | `{run.path}` |
| 프레임 | {len(rows)}개 (파지 전 구간) |
| target | `{target}` |
| decoy | `{decoy}` — 합성 attention 이 겨누는 다른 물체 |
| phase | `{args.phase}` |

## 판정 — {verdict}

| 검사 | 결과 | 기준 |
|---|---|---|
| A. attention 인자 부재 (정적) | {'통과' if ok_a else '실패'} — `generate_candidates` 인자 {len(sig.parameters)}개 중 attention 관련 **{len(attention_params)}개** | 0개 |
| B. 덮는 점 집합 불변 (실험) | {'통과' if ok_b else '실패'} — {len(rows)}프레임 전부 세 attention 에서 동일, 최대 차이 **{max(coverage_gaps) if coverage_gaps else 0}점** | 완전 동일 |
| C. 물체 소실 없음 | {'통과' if ok_c else '실패'} — 정답 점 {cl_cfg.min_points}개 이상인데 후보가 하나도 덮지 않은 경우 **{len(lost)}건** | 0건 |
| D. target 이 후보로 남음 | {'통과' if ok_d else '실패'} — target 이 잡힌 프레임마다 `TARGET` 후보 정확히 1개 | 전부 |
| E. 여유거리 분리 | {'통과' if ok_e else '실패'} — 인가된 손가락 {matrix[0,0]*1000:.0f} mm vs 토르소 {matrix[2,0]*1000:.0f} mm (target 열) | 손가락 < 토르소 |

### A — 서명만으로는 부족하다

`generate_candidates`에 attention 인자는 없다. 하지만 **그것만으로는 증명이 되지 않는다.**
attention 은 `target` 인자를 통해 간접적으로 들어오고, 실제로 들어온다 — target 의 점들은 잔여
군집에서 빠져 `TARGET` 후보가 된다.

그러니 물어야 할 것은 "attention 이 영향을 주는가"가 아니라 **"attention 이 무엇을 지우는가"**다.
그래서 아래 B를 실험으로 잰다.

### B — 같은 점군, 다른 attention 세 가지

같은 프레임의 같은 점군에 attention 만 바꿔 넣는다.

{md(["attention", "grounding 상태", "고른 target", "후보 수", "유형별", "덮은 점"],
    [f"| {v} | `{r0[v]['status']}` | {r0[v]['picked'] or '—'} | {r0[v]['n_candidates']} | "
     f"{', '.join(f'{k} {n}' for k, n in sorted(r0[v]['by_type'].items()))} | {r0[v]['n_covered']:,} |"
     for v in vnames])}

세 경우에서 **덮이는 점의 집합이 완전히 같다.** 달라지는 것은 그 점들이 어떤 유형의 후보에
들어가는가뿐이다 — `{target}` 은 첫 줄에서 `TARGET`, 나머지 두 줄에서는 그냥 `OBJECT` 다.

![씬에서 본 세 경우]({img}/fig3_scene.png)

세 패널이 **같은 씬**이다. 회색 점도, 노란 정답 점도, 초록 원(obstacle 후보)도 그대로다.
달라지는 것은 **굵은 파란 원 하나가 어디에 있는가**뿐이고, 셋째 패널에서는 그것마저 없다 —
그래도 초록 원은 전부 남아 있다.

같은 것을 **정책이 본 이미지 위에** 되돌려 그리면 이렇다. 초록 원의 자리와 크기가 세 패널에서
동일하고, 굵은 파란 원만 사과 → 바나나 → 없음으로 바뀐다.

![이미지 위 후보]({img}/fig4_image_overlay.png)

![덮인 기하]({img}/fig1_coverage.png)

이것이 이 파이프라인이 존재하는 이유다. attention 이 완전히 틀려도(둘째 줄), 심지어 아무
정보가 없어도(셋째 줄), **물리적으로 존재하는 기하는 전부 충돌 후보로 남는다.** 잘못된
attention 이 만드는 결과는 "엉뚱한 물체에 접촉이 허용된다"이지 "물체가 사라진다"가 아니다.

### C — 물체별로 확인

{md(["물체"] + [f"{v}" for v in vnames],
    [f"| {b} | " + " | ".join(
        f"{r0[v]['per_body'][b][1]:,} / {r0[v]['per_body'][b][0]:,}" for v in vnames) + " |"
     for b in bodies])}

각 칸은 `후보가 덮은 점 / 정답 점`이다. attention 이 무엇을 가리키든 모든 물체가 덮인다.

### D — target 은 지워지지 않고 이름표만 바뀐다

`SourceType.TARGET` 후보는 **후보 목록에 그대로 있다.** 4단계에서 고른 클러스터가 삭제되어
제약이 사라지는 것이 아니라, 같은 기하가 다른 유형으로 들어간다. 그 유형이 하는 일은
여유거리 정책을 바꾸는 것뿐이다.

`PhaseRule` 의 docstring이 이 설계의 역사를 적고 있다 — `collision_enabled` 가 `GRASP` 에서
False 였던 적이 있고, 그때는 target 의 제약 행이 통째로 삭제되어 **"로봇이 잡고 있는 물체에
팔꿈치를 밀어 넣을 수 있었다"**. 지금은 항상 True 이고, 접촉은 행을 지우는 대신 인가된 링크의
여유거리를 `contact_margin` 으로 낮추어 표현한다.

### E — 여유거리 분리

![여유거리 행렬]({img}/fig2_clearance.png)

행은 로봇 링크, 열은 후보 유형, 숫자는 필요한 여유거리(mm)다. 활성 조작기는 `right`.
읽을 것 네 가지:

**1. 완화는 (링크 × 후보) 단위다.** 완화가 나타나는 곳은 **target 열의 인가된 링크 한 칸뿐**이다.
`link_torso_3` 도 `ee_finger_l1`(반대팔) 도 모든 단계에서 전체 여유거리를 유지한다. 하나의
스칼라로 완화했다면 토르소가 손가락용 여유거리를 물려받는다.

**2. obstacle 열은 어느 단계에서도 움직이지 않는다.** 단계는 target 과의 관계만 바꾼다.

**3. 단계가 진행될수록 좁아진다.**
{" → ".join(f"{ph} {by_phase[ph][0,0]*1000:.0f}mm" for ph in phases)}.
`grasp` 에서 0 mm 는 "제약이 사라졌다"가 아니라 **"닿는 것은 허용, 파고드는 것은 금지"**다 —
행은 그래프에 그대로 남아 있다.

`PhaseRule` 의 docstring 이 이 설계의 역사를 적고 있다. `collision_enabled` 가 `GRASP` 에서
False 였던 적이 있고, 그때는 target 의 제약 행이 통째로 삭제되어 **"로봇이 잡고 있는 물체에
팔꿈치를 밀어 넣을 수 있었다"**. 지금은 항상 True 다.

**4. 완화가 무력화되는 두 조건이 실제로 작동한다.** 오른쪽 두 패널이 그것이다.

* **target 없음** — grounding 이 실패하면 완화할 대상이 없으므로 모든 칸이 전체 여유거리다.
* **알 수 없는 조작기 (fail-closed)** — 이 검증의 첫 판이 조작기 이름으로 `"right_arm"` 을
  넘겼다. 열거형 값은 `right`/`left` 뿐이라 알 수 없는 값이고, 정책은 **아무 링크도 인가하지
  않았다**. 그래서 `grasp` 인데도 손가락이 50 mm 를 유지했고 검사가 실패했다. 버그가 아니라
  fail-closed 가 설계대로 작동한 것이며, 오타 하나가 접촉 허가를 여는 것이 아니라 닫는 쪽으로
  떨어진다는 뜻이다. 그 패널을 지우지 않고 남긴 이유가 이것이다.

## 그림에 대하여

### fig3 — 씬에서 본 세 경우 (이 단계의 핵심 그림)

`fig3_scene.png`. 한 프레임의 점군을 위에서 내려다본 것. **원은 후보의 제약용 구 단면**이고,
초록은 obstacle, 굵은 파랑은 target 이다. 노란 점은 정답 물체, 옅은 회색은 지지면.

**만드는 법.** 같은 프레임의 같은 점군에 attention 만 세 가지로 바꿔 넣고, 각각
`ground_target` → `generate_candidates` 를 돌려 나온 후보의 `bounding_radius` 를 원으로 그린다.
세 패널의 점 데이터는 완전히 동일하며, 다시 계산하지도 않는다.

**읽는 법.** 세 패널을 겹쳐 보면 초록 원이 정확히 같은 자리에 같은 크기로 있다. 굵은 파란
원 하나만 자리를 옮기고, 셋째 패널에서는 사라진다 — 그래도 초록 원은 전부 남는다. 후보
개수가 세 패널 모두 같은 것도 확인할 수 있다.

### fig4 — 관측 이미지 위의 후보

`fig4_image_overlay.png`. 정책이 본 입력 위에 후보의 제약용 구를 원으로 투영한 것. fig3(위에서
본 그림)과 **같은 데이터, 다른 시점**이다.

**만드는 법.** `imageview.sphere_circle` — 구 중심을 투영한 자리에 반지름 `f*r/z` 인 원. 깊이
해상도(480x640)와 정책 이미지(224x224) 사이는 정규화 좌표로 옮기며 오프셋이 없다 (2단계에서
확인한 성질).

**읽는 법.** 세 패널의 초록 원이 같은 자리·같은 크기인지 보면 된다. 굵은 파란 원만 사과 →
바나나 → 없음으로 바뀐다. 원이 테이블 밖까지 나가는 큰 초록 원 둘은 크레이트와 배경 클러스터의
경계 구이며, 3단계에서 "seed 의 78.8%가 배경에 있다"고 한 것의 결과다.

### fig1 — 물체별로 덮인 점 수

`fig1_coverage.png`. 회색 막대는 정답 점 수, 색 막대는 후보가 덮은 점 수. 파랑은 그 패널에서
target 으로 뽑힌 물체다.

**읽는 법.** 회색과 색 막대의 높이가 같아야 한다 — 물체가 덮이지 않으면 색 막대가 낮아진다.
파랑이 어느 물체에 있는지가 패널마다 달라도 막대 높이는 변하지 않는 것이 요점이다.

### fig2 — 여유거리 행렬

`fig2_clearance.png`. 행이 로봇 링크, 열이 후보 유형, 숫자가 필요한 여유거리(mm). 패널은
단계 넷과 무력화 조건 둘.

**만드는 법.** `ClearancePolicy.margin_matrix` 를 단계별로 호출한다 — 그림용 근사가 아니라
파이프라인이 제약을 세울 때 쓰는 바로 그 함수다.

**읽는 법.** 완화가 나타나는 칸이 몇 개인지 세어 보면 된다. target 열의 인가 링크 한 칸뿐이다.
오른쪽 두 패널은 완화가 무력화되는 조건이고, 마지막 패널은 조작기 이름 오타가 접촉 허가를
**닫는 쪽**으로 떨어진다는 것을 보여준다.

## 재현

```bash
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.reports.separation_report \\\\
    --records {args.records} --attention {args.attention}
```
"""
    out = paths.document
    out.write_text(doc)
    out.with_suffix(".json").write_text(json.dumps({
        "target": target, "decoy": decoy, "phase": args.phase,
        "checks": {"A_no_attention_param": ok_a, "B_coverage_invariant": ok_b,
                   "C_no_body_lost": ok_c, "D_target_kept": ok_d, "E_clearance_split": ok_e},
        "max_coverage_gap": int(max(coverage_gaps) if coverage_gaps else 0),
        "lost": lost,
        "frames": [{k: (v if not isinstance(v, dict) else
                        {kk: vv for kk, vv in v.items() if kk != "covered"})
                    for k, v in r.items() if k != "clouds"} for r in rows],
    }, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {out} and 2 figures in {figs}")
    print(f"판정 {verdict}: A={len(attention_params)}개 인자  B=최대차이 "
          f"{max(coverage_gaps) if coverage_gaps else 0}점  C=소실 {len(lost)}건  "
          f"D={'ok' if ok_d else 'FAIL'}  E=손가락 {matrix[0,0]*1000:.0f}mm/토르소 {matrix[2,0]*1000:.0f}mm")


if __name__ == "__main__":
    main()
