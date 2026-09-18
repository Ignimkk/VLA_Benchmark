"""의미 융합 규칙 비교 — `max` 대 `mean` 대 head 우선.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.step5_fusion_rules

`fuse()` 는 복셀 하나에 들어온 여러 카메라의 attention 을 `max` 로 합친다. 그 규칙이 옳은지는
판정한 적이 없다. overlap 측정(`step5_overlap.py`)에서 조건이 충족됐다.

  씨앗 중 2 대 이상이 본 비율   평균 49.8 %   <- 규칙이 실제로 작동하는 자리가 있다
  겹친 점의 카메라 간 값 폭     중앙 0.0004, 폭 > 0.1 인 비율 7.9 % (원시 스케일)

즉 **대부분은 카메라들이 동의하고, 8 % 에서 크게 갈린다.** 그 8 % 가 판정을 바꾸는지 본다.

규칙 셋. 전부 **원시 스케일**에서 합친 뒤 한 번 정규화한다 (F2 수정과 같은 순서).

  max   — 현재. 가장 확신하는 카메라가 이긴다.
  mean  — 평균. 한 대가 튀는 것을 눌러 준다.
  head  — head 우선. head 가 봤으면 head 값, 아니면 나머지의 max.
          `CameraID` docstring 이 "head = global backbone, wrists = local refinement" 라고
          적어 둔 역할 분담을 그대로 규칙으로 만든 것.

융합은 프레임당 **한 번만** 하고 CSR 블록에 규칙을 각각 적용한다 — F2 실험처럼 세 번 융합하지
않으므로 훨씬 빠르고, 세 규칙이 **완전히 같은 기하**를 본다는 것도 보장된다.
"""

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
ATT_CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
RULES = ("max", "mean", "head")


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


def _name_of(c, objects, tol=0.12):
    if c is None:
        return "없음", np.inf
    best, bd = "?", np.inf
    for n, p in objects.items():
        d = float(np.linalg.norm(np.asarray(c, float) - p))
        if d < bd:
            best, bd = n, d
    return (best if bd < tol else f"{best}?"), bd


def aggregate(rule, obs_raw, starts, offsets, obs_cam, head_idx):
    """CSR 블록마다 규칙을 적용해 점당 값 하나. 전부 원시 스케일."""
    if rule == "max":
        return np.maximum.reduceat(obs_raw, starts)
    if rule == "mean":
        total = np.add.reduceat(obs_raw, starts)
        return total / np.maximum(np.diff(offsets), 1)
    if rule == "head":
        # head 가 본 관측만 골라 max, 없으면 전체 max 로 떨어진다.
        is_head = (obs_cam == head_idx).astype(np.float64)
        head_max = np.maximum.reduceat(np.where(is_head > 0, obs_raw, -np.inf), starts)
        allmax = np.maximum.reduceat(obs_raw, starts)
        return np.where(np.isfinite(head_max), head_max, allmax)
    raise ValueError(rule)


def main() -> None:
    _style()

    from benchmark.ag3s.attention_lifting import GridAttentionAdapter, normalize_attention
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.multiview import fuse, process_observation
    from benchmark.ag3s.support_surface import fit_support_surfaces
    from benchmark.ag3s.target_grounding import extract_seeds, ground_target
    from benchmark.ag3s.types import AttentionPointCloud, CameraID
    from benchmark.ag3s.experiments.grounding_report import build_robot_model
    from benchmark.ag3s.experiments.mujoco_source import camera_observation
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--attention", default="attention_step1_run0004.npz")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/step-01-attention.json")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--slice-frame", type=int, default=7)
    args = ap.parse_args()

    blob = np.load(args.attention, allow_pickle=False)
    cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    cix = {c: [str(x) for x in blob["cameras"]].index(c) for c in ATT_CAMS}

    run = load_run(args.records, limit=args.frames)
    scene = replay_scene(run)
    filt = build_robot_model(scene)
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "pointcloud": {"range_max": 2.0}})
    adapter = GridAttentionAdapter.from_config(cfg.attention)

    print("=" * 92)
    print("의미 융합 규칙 비교 — 융합은 한 번, 규칙만 바꿔 적용")
    print("=" * 92)
    print(f"  {'i':>2}  {'max (현재)':<22} {'mean':<22} {'head 우선':<22}  일치")

    rows, cache = [], None
    for i in range(args.frames):
        pose_scene(scene, run.steps[i])
        objects = _objects(scene)
        grids = {c: np.asarray(blob["attention"][i, di, ai, cell["layer"], cell["head"], cix[c]],
                               np.float32) for c in ATT_CAMS}
        obs = [camera_observation(scene, cam, filt, attention_map=grids[a], timestamp=float(i))[0]
               for cam, a in zip(CAMS, ATT_CAMS)]
        results = [process_observation(o, cfg, robot_model=filt, attention_adapter=adapter)
                   for o in obs]
        fc, _, _ = fuse(results, voxel_size=cfg.timing.fusion_voxel_size,
                        attention_config=cfg.attention)
        cloud = fc.as_pointcloud()
        offsets = np.asarray(fc.obs_offset, np.int64)
        starts = offsets[:-1]
        obs_cam = np.asarray(fc.obs_camera, np.int64)
        n = len(starts)

        # 관측별 원시값을 fuse 와 같은 순서로 복제 (CSR 블록은 정규화본만 들고 있다)
        _cl = [r for r in results if not r.cloud.is_empty]
        _pts = np.vstack([r.cloud.points for r in _cl])
        _raw = np.concatenate([r.raw_attention for r in _cl]).astype(np.float64)
        _k = np.floor(_pts / float(cfg.timing.fusion_voxel_size)).astype(np.int64)
        _k -= _k.min(axis=0)
        _e = _k.max(axis=0) + 1
        _p = (_k[:, 0] * _e[1] + _k[:, 1]) * _e[2] + _k[:, 2]
        _o = np.argsort(_p, kind="stable")
        obs_raw = _raw[_o]
        assert len(np.flatnonzero(np.r_[True, _p[_o][1:] != _p[_o][:-1]])) == n

        head_idx = [j for j, c in enumerate(fc.cameras) if c == CameraID.HEAD]
        head_idx = head_idx[0] if head_idx else -1
        _, mask = fit_support_surfaces(cloud, cfg.support_surface, timestamp=float(i))

        ncam = np.array([len(np.unique(obs_cam[offsets[j]:offsets[j + 1]])) for j in range(n)])
        out = {}
        for rule in RULES:
            agg = aggregate(rule, obs_raw, starts, offsets, obs_cam, head_idx)
            att = normalize_attention(agg.astype(np.float32), cfg.attention)
            g = ground_target(
                AttentionPointCloud(cloud, att, agg.astype(np.float32)), cfg.clustering,
                seed_percentile=cfg.attention.seed_percentile,
                seed_threshold=cfg.attention.seed_threshold,
                exclude_mask=mask, min_radius=cfg.geometry.min_radius, timestamp=float(i))
            out[rule] = dict(agg=agg, att=np.asarray(att, float), g=g)

        # 씨앗에서의 불일치 — overlap 측정에서 빠졌던 조각
        seeds = np.asarray(extract_seeds(out["max"]["att"].astype(np.float32), cfg.clustering,
                                         cfg.attention.seed_percentile,
                                         cfg.attention.seed_threshold)).ravel().astype(np.int64)
        sm = seeds[ncam[seeds] >= 2]
        denom = max(float(obs_raw.max()), 1e-12)
        sspread = np.array([(obs_raw[offsets[j]:offsets[j + 1]].max()
                             - obs_raw[offsets[j]:offsets[j + 1]].min()) / denom for j in sm])

        names, cents = {}, {}
        for rule in RULES:
            t = out[rule]["g"].target
            c = None if t is None else np.asarray(t.centroid, float)
            cents[rule] = c
            nm, d = _name_of(c, objects)
            names[rule] = f"{nm} ({d*1000:.0f}mm)" if c is not None else "없음"
        same = names["max"] == names["mean"] == names["head"]
        shift = {r: (np.linalg.norm(cents["max"] - cents[r]) * 1000
                     if cents["max"] is not None and cents[r] is not None else np.nan)
                 for r in ("mean", "head")}
        rows.append(dict(i=i, names=names, same=same, shift=shift, ncam=ncam,
                         seed_spread=sspread, n_seed_multi=len(sm), n_seed=len(seeds)))
        print(f"  {i:>2}  {names['max']:<22} {names['mean']:<22} {names['head']:<22}  "
              f"{'같음' if same else '<<< 다름'}")
        if i == args.slice_frame:
            cache = (cloud, out, objects, ncam, seeds)

    print()
    ns = sum(r["same"] for r in rows)
    print(f"  세 규칙이 같은 target 을 낸 프레임: {ns}/{len(rows)}")
    for r in ("mean", "head"):
        v = np.array([x["shift"][r] for x in rows], float)
        print(f"  centroid 이동 max→{r:<5} 중앙 {np.nanmedian(v):6.1f} mm   최대 {np.nanmax(v):7.1f} mm")
    allss = np.concatenate([x["seed_spread"] for x in rows if len(x["seed_spread"])])
    print(f"\n  겹친 씨앗에서의 카메라 간 값 폭 (원시 스케일)  n={len(allss):,}")
    print(f"     중앙 {np.median(allss):.4f}   90분위 {np.percentile(allss,90):.4f}   "
          f"폭 > 0.1 비율 {100.0*(allss>0.1).mean():.1f} %")

    if cache is not None:
        _figure(cache, rows, args.slice_frame, cfg)
    scene.close()


def _figure(cache, rows, k, cfg):
    import matplotlib.pyplot as plt
    cloud, out, objects, ncam, seeds = cache
    pts = np.asarray(cloud.points, float)

    fig, axs = plt.subplots(2, 3, figsize=(19.5, 10.4))
    ax = axs.ravel()

    titles = {"max": "(a) max — 현재 규칙", "mean": "(b) mean", "head": "(c) head 우선"}
    for j, rule in enumerate(RULES):
        a = ax[j]
        v = out[rule]["att"]
        v = v / max(v.max(), 1e-12)
        o = np.argsort(v)
        s = a.scatter(pts[o, 0], pts[o, 1], c=v[o], s=1.2, cmap="inferno", vmin=0, vmax=1)
        for oi, (nm, q) in enumerate(sorted(objects.items())):
            a.plot(q[0], q[1], "wo", ms=5, mec="k")
            a.text(q[0], q[1] + (0.035 if oi % 2 == 0 else -0.055), nm, fontsize=8,
                   ha="center", color="w", bbox=dict(fc="k", alpha=0.45, pad=0.8, lw=0))
        t = out[rule]["g"].target
        if t is not None:
            c = np.asarray(t.centroid, float)
            a.plot(c[0], c[1], "c*", ms=20, mec="k")
        a.set_title(f"{titles[rule]}   (프레임 {k})", fontsize=11)
        a.set_xlabel("x [m] — 앞쪽 →"); a.set_ylabel("y [m] — 왼쪽 ↑")
        a.set_aspect("equal"); a.set_xlim(0.15, 1.0); a.set_ylim(-0.65, 0.65)
        plt.colorbar(s, ax=a, fraction=0.046, label="attention (최대=1)")

    # (d) 규칙 간 값 차이 — 겹친 점에서만
    a = ax[3]
    multi = np.flatnonzero(ncam >= 2)
    for rule, col in (("mean", "tab:blue"), ("head", "tab:green")):
        d = (out[rule]["att"] - out["max"]["att"])[multi]
        a.hist(d, bins=60, histtype="step", lw=2, color=col, label=f"{rule} − max")
    a.set_yscale("log"); a.axvline(0, color="k", lw=1)
    a.set_title("(d) 겹친 점에서 규칙이 만드는 값 차이\n1 대만 본 점은 세 규칙이 정의상 같다",
                fontsize=11)
    a.set_xlabel("정규화된 attention 차이"); a.set_ylabel("점 개수 (log)")
    a.legend(fontsize=9); a.grid(alpha=0.3)

    # (e) 씨앗 겹침에서의 불일치
    a = ax[4]
    allss = np.concatenate([x["seed_spread"] for x in rows if len(x["seed_spread"])])
    a.hist(allss, bins=60, color="tab:purple", alpha=0.8)
    a.axvline(float(np.median(allss)), color="k", lw=1.5, label=f"중앙 {np.median(allss):.3f}")
    a.set_yscale("log")
    a.set_title("(e) 겹친 씨앗에서 카메라 간 값 폭 (원시 스케일)\n"
                "규칙이 실제로 판정을 좌우하는 자리", fontsize=11)
    a.set_xlabel("max − min"); a.set_ylabel("개수 (log)"); a.legend(fontsize=9); a.grid(alpha=0.3)

    # (f) 표
    a = ax[5]; a.axis("off")
    ns = sum(r["same"] for r in rows)
    mn = np.array([x["shift"]["mean"] for x in rows], float)
    hd = np.array([x["shift"]["head"] for x in rows], float)
    cellt = [
        ["", "값"],
        ["프레임 수", f"{len(rows)}"],
        ["세 규칙이 같은 target", f"{ns} / {len(rows)}"],
        ["centroid 이동 max→mean (중앙/최대)",
         f"{np.nanmedian(mn):.1f} / {np.nanmax(mn):.1f} mm"],
        ["centroid 이동 max→head (중앙/최대)",
         f"{np.nanmedian(hd):.1f} / {np.nanmax(hd):.1f} mm"],
        ["겹친 씨앗 값 폭 (중앙)", f"{np.median(allss):.4f}"],
        ["겹친 씨앗 값 폭 > 0.1", f"{100.0*(allss>0.1).mean():.1f} %"],
        ["겹친 씨앗 개수 (전 프레임)", f"{len(allss):,}"],
    ]
    t = a.table(cellText=cellt[1:], colLabels=cellt[0], loc="center", cellLoc="left",
                colWidths=[0.62, 0.38])
    t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1.0, 2.0)
    for j in range(2):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    a.set_title("(f) 요약 — 규칙을 바꾸면 판정이 바뀌는가", fontsize=11, y=0.92)

    fig.suptitle("의미 융합 규칙 비교 — max / mean / head 우선 (같은 기하, 규칙만 다름)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUT.mkdir(parents=True, exist_ok=True)
    o = OUT / "step5-fusion-rules.png"
    fig.savefig(o, dpi=105, bbox_inches="tight")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
