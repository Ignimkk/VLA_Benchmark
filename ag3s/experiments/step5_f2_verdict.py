"""F2 판정 — 카메라별 정규화가 실제로 target 을 바꾸는가.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.step5_f2_verdict

`multiview.py:123-125` 의 주석은 "one normalization for all of them" 이라고 적혀 있는데
`lift()` 는 카메라별로 정규화한다. 상한이 카메라 간 2.7 배 차이나는 것까지는 쟀다.
**남은 질문은 그것이 판정을 바꾸는가** 다.

세 가지를 나란히 돌린다.

  A. 현재 코드          — 카메라별 정규화 (`lift` 가 각 카메라 점들에만 퍼센타일)
  B. 전역 정규화        — 3 대 점을 모아 한 번에 정규화 (주석이 말한 것)
  C. 문턱도 원시본      — 정규화본을 씨앗 선택에서 빼고 원시본으로만 고른다

비교하는 것: 융합된 attention, 씨앗 선택, **최종 grounded target**.
셋이 같은 target 을 내면 F2 는 "실질 영향 없음", 다르면 "확정" 이다.
"""

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
ATT_CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
LABEL = ("head", "왼손목", "오른손목")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments import figstyle
    figstyle.use_korean()


def _objects(scene):
    """씬 물체 이름 -> base 프레임 위치. target 이 무엇으로 잡혔는지 이름 붙이는 데 쓴다."""
    import mujoco
    from benchmark.ag3s.experiments.mujoco_source import is_robot_body
    out = {}
    for b in range(scene.model.nbody):
        n = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not n or is_robot_body(n):
            continue
        if any(k in n for k in ("table", "shelf", "floor", "world", "ground", "com_target")):
            continue
        out[n] = np.asarray(scene.data.xpos[b], float)
    return out


def _name_of(centroid, objects, tol=0.12):
    if centroid is None:
        return "없음", np.inf
    best, bd = "?", np.inf
    for n, p in objects.items():
        d = float(np.linalg.norm(np.asarray(centroid, float) - p))
        if d < bd:
            best, bd = n, d
    return (best if bd < tol else f"{best}?"), bd


def run_variant(variant, *, scene, robot, filter_robot, grids, cfg, step, ts):
    """variant: 'A' 현재 / 'B' 전역 정규화 / 'C' 문턱도 원시본."""
    import dataclasses

    from benchmark.ag3s.attention_lifting import GridAttentionAdapter, normalize_attention
    from benchmark.ag3s.experiments.mujoco_source import camera_observation
    from benchmark.ag3s.multiview import fuse_observations
    from benchmark.ag3s.support_surface import fit_support_surfaces
    from benchmark.ag3s.target_grounding import ground_target
    from benchmark.ag3s.types import AttentionPointCloud

    adapter = GridAttentionAdapter.from_config(cfg.attention)
    obs = []
    for cam, ac in zip(CAMS, ATT_CAMS):
        o, _ = camera_observation(scene, cam, filter_robot, attention_map=grids[ac], timestamp=ts)
        obs.append(o)

    fused = fuse_observations(obs, cfg, robot_model=filter_robot, attention_adapter=adapter)
    cloud = fused.cloud.as_pointcloud()
    attention = np.asarray(fused.attention, np.float32)     # 정규화본을 max 융합한 것
    raw = np.asarray(fused.raw_attention, np.float32)       # 원시본을 max 융합한 것

    if variant == "B":
        # 전역 정규화 — 융합된 **원시** 값 전체에 퍼센타일을 한 번만 건다.
        attention = normalize_attention(raw, cfg.attention)
    elif variant == "C":
        # 문턱도 원시본 — 정규화를 아예 안 쓴다 (순위는 어차피 보존되므로 퍼센타일 씨앗은 동작한다).
        attention = raw.astype(np.float32)

    # **지지면 마스크를 반드시 넘긴다.** 이것 없이 부르면 영역 성장이 테이블을 타고 번져
    # 씬 전체가 한 클러스터가 되고 target 을 아예 못 찾는다 (`no_target`) — 파이프라인이
    # `fit_support_surfaces` 를 어느 모드에서도 켜 두는 이유이고, 처음에 빠뜨려 4/4 프레임이
    # "없음" 으로 나왔다.
    _, support_mask = fit_support_surfaces(cloud, cfg.support_surface, timestamp=ts)
    acloud = AttentionPointCloud(cloud, attention, raw)
    grounding = ground_target(
        acloud, cfg.clustering,
        seed_percentile=cfg.attention.seed_percentile,
        seed_threshold=cfg.attention.seed_threshold,
        exclude_mask=support_mask,
        min_radius=cfg.geometry.min_radius,
        timestamp=ts)
    return dict(cloud=cloud, attention=attention, raw=raw, grounding=grounding, fused=fused)


def main() -> None:
    _style()
    import matplotlib.pyplot as plt

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--attention", default="attention_step1_run0004.npz")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/step-01-attention.json")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--slice-frame", type=int, default=10)
    args = ap.parse_args()

    blob = np.load(args.attention, allow_pickle=False)
    cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    cam_idx = {c: [str(x) for x in blob["cameras"]].index(c) for c in ATT_CAMS}

    run = load_run(args.records, limit=args.frames)
    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "pointcloud": {"range_max": 2.0}})

    print("=" * 86)
    print("F2 판정 — 카메라 3 대 융합. A 현재(카메라별) / B 전역 정규화 / C 문턱도 원시본")
    print("=" * 86)
    print(f"  {'i':>2}  {'A 현재':<22} {'B 전역':<22} {'C 원시':<22}  일치")

    rows, cache = [], None
    for i in range(args.frames):
        pose_scene(scene, run.steps[i])
        objects = _objects(scene)
        grids = {c: np.asarray(blob["attention"][i, di, ai, cell["layer"], cell["head"],
                                                 cam_idx[c]], np.float32) for c in ATT_CAMS}
        res = {}
        for v in ("A", "B", "C"):
            res[v] = run_variant(v, scene=scene, robot=robot, filter_robot=filter_robot,
                                 grids=grids, cfg=cfg, step=run.steps[i], ts=float(i))
        names, cents = {}, {}
        for v in "ABC":
            t = res[v]["grounding"].target
            c = None if t is None else np.asarray(t.centroid, float)
            cents[v] = c
            nm, d = _name_of(c, objects)
            names[v] = f"{nm} ({d*1000:.0f}mm)" if c is not None else "없음"
        same = (names["A"] == names["B"] == names["C"])
        # A 와 B 의 centroid 거리
        dAB = (np.linalg.norm(cents["A"] - cents["B"]) * 1000
               if cents["A"] is not None and cents["B"] is not None else np.nan)
        dAC = (np.linalg.norm(cents["A"] - cents["C"]) * 1000
               if cents["A"] is not None and cents["C"] is not None else np.nan)
        rows.append(dict(i=i, names=names, same=same, dAB=dAB, dAC=dAC,
                         status={v: res[v]["grounding"].status.value for v in "ABC"}))
        st = res['A']['grounding'].status.value
        print(f"  {i:>2}  {names['A']:<22} {names['B']:<22} {names['C']:<22}  "
              f"{'같음' if same else '<<< 다름'}   [{st}]")
        if i == args.slice_frame:
            cache = (res, objects, grids)

    n_same = sum(r["same"] for r in rows)
    print()
    print(f"  target 이름이 셋 다 같은 프레임: {n_same}/{len(rows)}")
    dAB = np.array([r["dAB"] for r in rows], float)
    dAC = np.array([r["dAC"] for r in rows], float)
    print(f"  centroid 이동  A→B  중앙 {np.nanmedian(dAB):.1f} mm  최대 {np.nanmax(dAB):.1f} mm")
    print(f"  centroid 이동  A→C  중앙 {np.nanmedian(dAC):.1f} mm  최대 {np.nanmax(dAC):.1f} mm")

    if cache is not None:
        _figure(cache, rows, scene)
    else:
        print('  (그림 생략 — --slice-frame 이 --frames 밖)')
    scene.close()


def _figure(cache, rows, scene):
    import matplotlib.pyplot as plt
    res, objects, grids = cache

    fig, axs = plt.subplots(2, 3, figsize=(19.5, 10.4))
    ax = axs.ravel()

    # (a) 실제 씬 — 융합 클라우드를 위에서, A 의 attention 으로 색칠
    cloud = res["A"]["cloud"]
    for j, (v, title) in enumerate((("A", "(a) A 현재 — 카메라별 정규화"),
                                    ("B", "(b) B 전역 정규화"),
                                    ("C", "(c) C 문턱도 원시본"))):
        a = ax[j]
        val = np.asarray(res[v]["attention"], float)
        val = val / max(val.max(), 1e-12)
        o = np.argsort(val)
        p = res[v]["cloud"].points
        s = a.scatter(p[o, 0], p[o, 1], c=val[o], s=1.0, cmap="inferno", vmin=0, vmax=1)
        t = res[v]["grounding"].target
        for oi, (nm, q) in enumerate(sorted(objects.items())):
            a.plot(q[0], q[1], "wo", ms=5, mec="k")
            a.text(q[0], q[1] + (0.035 if oi % 2 == 0 else -0.055), nm, fontsize=8,
                   ha="center", color="w",
                   bbox=dict(fc="k", alpha=0.45, pad=0.8, lw=0))
        if t is not None:
            c = np.asarray(t.centroid, float)
            a.plot(c[0], c[1], "c*", ms=22, mec="k", label="grounded target")
            a.legend(loc="lower left", fontsize=8, framealpha=0.9)
        a.set_title(title, fontsize=11)
        a.set_xlabel("x [m] — 앞쪽 →"); a.set_ylabel("y [m] — 왼쪽 ↑")
        a.set_xlim(0.15, 1.0); a.set_ylim(-0.65, 0.65); a.set_aspect("equal")
        plt.colorbar(s, ax=a, fraction=0.046, label="attention (최대=1)")

    # (d) 카메라별 눈금 — 어느 카메라가 융합 max 를 가져갔나
    a = ax[3]
    fc = res["A"]["fused"].cloud
    obs_att = np.asarray(fc.obs_attention, float)          # 카메라별로 정규화된 값
    cams = np.asarray(fc.obs_camera)                       # (K,) index into fc.cameras
    names = [c.value if hasattr(c, "value") else str(c) for c in fc.cameras]
    for u, col in zip(range(len(names)), ("tab:blue", "tab:orange", "tab:green")):
        m = cams == u
        if not m.any():
            continue
        a.hist(obs_att[m], bins=50, histtype="step", lw=2, color=col,
               label=f"{names[u]}  (1.0 인 점 {int((obs_att[m] >= 1.0 - 1e-6).sum()):,})")
    a.set_yscale("log"); a.legend(fontsize=8.5)
    a.set_title("(d) 융합 전 값 분포 — 카메라별 정규화의 결과\n"
                "세 카메라가 각자 1.0 까지 펴진다 = F2 의 원인", fontsize=11)
    a.set_xlabel("정규화된 attention"); a.set_ylabel("점 개수 (log)"); a.grid(alpha=0.3)

    # (e) A vs B vs C 의 값 자체
    a = ax[4]
    for v, col in (("A", "tab:red"), ("B", "tab:blue"), ("C", "tab:green")):
        val = np.asarray(res[v]["attention"], float)
        val = val / max(val.max(), 1e-12)
        a.hist(val, bins=60, histtype="step", lw=2, color=col, label=f"{v}")
    a.set_yscale("log")
    a.set_title("(e) 융합 후 값 분포 — 세 변형", fontsize=11)
    a.set_xlabel("attention (최대=1 로 재조정)"); a.set_ylabel("점 개수 (log)")
    a.legend(fontsize=9); a.grid(alpha=0.3)

    # (f) 표
    a = ax[5]; a.axis("off")
    n_same = sum(r["same"] for r in rows)
    import numpy as _np
    dAB = _np.array([r["dAB"] for r in rows], float)
    dAC = _np.array([r["dAC"] for r in rows], float)
    cell = [
        ["", "값"],
        ["프레임 수", f"{len(rows)}"],
        ["target 이 셋 다 같은 프레임", f"{n_same} / {len(rows)}"],
        ["centroid 이동 A→B (중앙 / 최대)",
         f"{_np.nanmedian(dAB):.1f} / {_np.nanmax(dAB):.1f} mm"],
        ["centroid 이동 A→C (중앙 / 최대)",
         f"{_np.nanmedian(dAC):.1f} / {_np.nanmax(dAC):.1f} mm"],
        ["A = 현재 코드", "카메라별 정규화"],
        ["B = 주석이 말한 것", "전역 정규화"],
        ["C = 대안", "문턱도 원시본"],
    ]
    t = a.table(cellText=cell[1:], colLabels=cell[0], loc="center", cellLoc="left",
                colWidths=[0.58, 0.42])
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.0, 2.0)
    for j in range(2):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    a.set_title("(f) 요약 — F2 가 판정을 바꾸는가", fontsize=11, y=0.9)

    fig.suptitle("F2 판정 — 카메라 3 대 융합에서 정규화 방식이 grounded target 을 바꾸는가 "
                 "(run_0004, 실측 attention)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "step5-f2-verdict.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
