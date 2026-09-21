"""multi-view overlap 측정 — 융합 점 하나를 몇 대의 카메라가 실제로 보는가.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.studies.step5_overlap

## 왜 재는가

`fuse()` 는 서로 다른 두 일을 한 번에 한다.

  ① **기하 융합** — 3 대의 점을 base 프레임 복셀 격자에 얹고 복셀당 대표점 하나를 남긴다.
  ② **의미 융합** — 그 복셀에 들어온 관측들의 attention 을 `max` 로 합친다.

F2 는 ② 의 **입력**(눈금)을 고친 것이고, ② 의 **규칙**(`max`)이 옳은지는 아직 판정한 적이 없다.
그런데 그 질문은 **겹침이 있어야만 의미가 있다** — 복셀 대부분을 한 대만 본다면 `max` 든
평균이든 같은 값이 나온다.

그래서 두 가지를 잰다.

  (1) **겹침 비율** — 융합 점 중 2 대 이상이 본 것이 몇 %인가. 씨앗(높은 attention)에서는?
  (2) **불일치 크기** — 겹친 점에서 카메라들의 attention 이 얼마나 다른가.
      겹치는데 값이 같다면 규칙 선택은 여전히 무의미하다.

`max` 대 `mean` 대 가중 비교는 (1)(2) 가 유의미할 때만 한다.
"""

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
ATT_CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments.common import figstyle
    figstyle.use_korean()


def measure(fused, attention, raw, cfg):
    """`fused` 의 CSR 블록에서 겹침과 불일치를 뽑는다."""
    off = np.asarray(fused.obs_offset, np.int64)
    cam = np.asarray(fused.obs_camera, np.int64)
    n = len(off) - 1
    counts = np.diff(off)                       # 점당 관측 수
    # 점당 **서로 다른 카메라** 수 (같은 카메라가 두 픽셀로 들어올 수 있다)
    ncam = np.zeros(n, np.int64)
    for i in range(n):
        ncam[i] = len(np.unique(cam[off[i]:off[i + 1]]))

    obs_raw_per_point = [np.asarray(fused.obs_attention[off[i]:off[i + 1]], float)
                         for i in range(n)]
    return counts, ncam, obs_raw_per_point


def main() -> None:
    _style()
    import matplotlib.pyplot as plt

    from benchmark.ag3s.stages.attention_lifting import GridAttentionAdapter
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.runtime.multiview import fuse, process_observation
    from benchmark.ag3s.stages.support_surface import fit_support_surfaces
    from benchmark.ag3s.stages.target_grounding import extract_seeds
    from benchmark.ag3s.experiments.reports.grounding_report import build_robot_model
    from benchmark.ag3s.experiments.sources.mujoco_source import camera_observation
    from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--attention", default="attention_step1_run0004.npz")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/archive/step-verification-20260904/step-01-attention.json")
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

    print("=" * 88)
    print(f"multi-view overlap — 융합 복셀 {cfg.timing.fusion_voxel_size*1000:.0f} mm")
    print("=" * 88)
    print(f"  {'i':>2}  {'융합점':>8} {'관측':>9}  "
          f"{'1대':>7} {'2대':>7} {'3대':>7}   {'씨앗 2대+':>10}  {'불일치 중앙':>11}")

    rows, cache = [], None
    for i in range(args.frames):
        pose_scene(scene, run.steps[i])
        grids = {c: np.asarray(blob["attention"][i, di, ai, cell["layer"], cell["head"], cix[c]],
                               np.float32) for c in ATT_CAMS}
        obs = [camera_observation(scene, cam, filt, attention_map=grids[a], timestamp=float(i))[0]
               for cam, a in zip(CAMS, ATT_CAMS)]
        results = [process_observation(o, cfg, robot_model=filt, attention_adapter=adapter)
                   for o in obs]
        fc, att, raw = fuse(results, voxel_size=cfg.timing.fusion_voxel_size,
                            attention_config=cfg.attention)
        att = np.asarray(att, float); raw = np.asarray(raw, float)

        off = np.asarray(fc.obs_offset, np.int64)
        cam = np.asarray(fc.obs_camera, np.int64)
        obs_norm = np.asarray(fc.obs_attention, float)  # 카메라별 정규화본 (출처 보존용)
        n = len(off) - 1
        ncam = np.array([len(np.unique(cam[off[j]:off[j + 1]])) for j in range(n)])

        # **관측별 원시값을 fuse 와 같은 순서로 복제한다.** CSR 블록은 정규화본만 들고 있는데,
        # 그 값은 카메라별 눈금이라 "불일치" 를 재면 진짜 이견과 눈금 차이가 섞인다 (그게 F2 다).
        # `fuse` 의 키·정렬을 그대로 재현해 원시 스케일에서 다시 잰다.
        _clouds = [r for r in results if not r.cloud.is_empty]
        _pts = np.vstack([r.cloud.points for r in _clouds])
        _raw = np.concatenate([r.raw_attention for r in _clouds]).astype(np.float64)
        _k = np.floor(_pts / float(cfg.timing.fusion_voxel_size)).astype(np.int64)
        _k -= _k.min(axis=0)
        _e = _k.max(axis=0) + 1
        _packed = (_k[:, 0] * _e[1] + _k[:, 1]) * _e[2] + _k[:, 2]
        _order = np.argsort(_packed, kind="stable")
        obs_raw = _raw[_order]
        _sk = _packed[_order]
        _starts = np.flatnonzero(np.r_[True, _sk[1:] != _sk[:-1]])
        assert len(_starts) == n, f"복제 실패: {len(_starts)} != {n}"   # fuse 와 같은 그룹인가

        # 씨앗에서의 겹침 — 여기가 의미 융합이 실제로 판정을 좌우하는 자리다
        seeds = extract_seeds(att.astype(np.float32), cfg.clustering,
                              cfg.attention.seed_percentile, cfg.attention.seed_threshold)
        seeds = np.asarray(seeds).ravel().astype(np.int64)

        # 불일치 — 겹친 점에서 카메라 간 값의 폭 (max − min).
        # **원시 스케일**이 융합이 실제로 비교하는 것이다 (F2 수정 후 `max` 는 원시본에 걸린다).
        # 정규화 스케일 폭도 같이 재서, 겉보기 불일치 중 얼마가 눈금 차이였는지 보인다.
        multi = np.flatnonzero(ncam >= 2)
        if len(multi):
            rmax = np.array([obs_raw[off[j]:off[j + 1]].max() for j in multi])
            rmin = np.array([obs_raw[off[j]:off[j + 1]].min() for j in multi])
            denom = max(float(obs_raw.max()), 1e-12)
            spread = (rmax - rmin) / denom          # 씬 최댓값으로 나눠 0~1 로
            nmax = np.array([obs_norm[off[j]:off[j + 1]].max() for j in multi])
            nmin = np.array([obs_norm[off[j]:off[j + 1]].min() for j in multi])
            spread_norm = nmax - nmin
        else:
            spread = np.zeros(0); spread_norm = np.zeros(0)

        frac = [float((ncam == k).mean()) for k in (1, 2, 3)]
        seed_multi = float((ncam[seeds] >= 2).mean()) if len(seeds) else float("nan")
        rows.append(dict(i=i, n=n, k=len(cam), frac=frac, seed_multi=seed_multi,
                         spread=spread, spread_norm=spread_norm, ncam=ncam, seeds=seeds))
        print(f"  {i:>2}  {n:>8,} {len(cam):>9,}  "
              f"{frac[0]*100:>6.1f}% {frac[1]*100:>6.1f}% {frac[2]*100:>6.1f}%   "
              f"{seed_multi*100:>9.1f}%  "
              f"{(np.median(spread) if len(spread) else float('nan')):>11.4f}")
        if i == args.slice_frame:
            cache = (fc, att, raw, ncam, seeds, spread)

    print()
    f = np.array([r["frac"] for r in rows])
    sm = np.array([r["seed_multi"] for r in rows])
    allspread = np.concatenate([r["spread"] for r in rows if len(r["spread"])])
    print(f"  전체 평균   1대 {f[:,0].mean()*100:.1f}%   2대 {f[:,1].mean()*100:.1f}%   "
          f"3대 {f[:,2].mean()*100:.1f}%")
    print(f"  씨앗 중 2대 이상이 본 비율   평균 {np.nanmean(sm)*100:.1f}%  "
          f"[{np.nanmin(sm)*100:.1f}, {np.nanmax(sm)*100:.1f}]")
    allnorm = np.concatenate([r["spread_norm"] for r in rows if len(r["spread_norm"])])
    print(f"  겹친 점의 카메라 간 값 폭 — 원시 스케일 (융합이 실제로 비교하는 것)")
    print(f"     중앙 {np.median(allspread):.4f}  90분위 {np.percentile(allspread,90):.4f}  "
          f"최대 {allspread.max():.4f}   폭 > 0.1 비율 {100.0*(allspread > 0.1).mean():.1f} %")
    print(f"  (참고) 카메라별 정규화 스케일에서 재면")
    print(f"     중앙 {np.median(allnorm):.4f}  90분위 {np.percentile(allnorm,90):.4f}   "
          f"폭 > 0.1 비율 {100.0*(allnorm > 0.1).mean():.1f} %   <- 이 중 상당수가 눈금 차이다")

    if cache is not None:
        _figure(cache, rows, args.slice_frame, cfg)
    scene.close()


def _figure(cache, rows, k, cfg):
    import matplotlib.pyplot as plt
    fc, att, raw, ncam, seeds, spread = cache
    pts = np.asarray(fc.points, float)

    fig, axs = plt.subplots(2, 3, figsize=(19.5, 10.4))
    ax = axs.ravel()

    # (a) 실제 씬 — 점마다 몇 대가 봤는가
    a = ax[0]
    order = np.argsort(ncam)
    s = a.scatter(pts[order, 0], pts[order, 1], c=ncam[order], s=1.4, cmap="viridis",
                  vmin=1, vmax=3)
    a.set_title(f"(a) 실제 씬 — 점마다 몇 대가 봤는가 (프레임 {k})\n"
                "위에서 내려다본 융합 클라우드", fontsize=11)
    a.set_xlabel("x [m] — 앞쪽 →"); a.set_ylabel("y [m] — 왼쪽 ↑")
    a.set_aspect("equal"); a.set_xlim(0.15, 1.0); a.set_ylim(-0.65, 0.65)
    cb = plt.colorbar(s, ax=a, fraction=0.046, ticks=[1, 2, 3]); cb.set_label("카메라 수")

    # (b) 씨앗만
    a = ax[1]
    a.scatter(pts[:, 0], pts[:, 1], c="0.85", s=0.8)
    sm = ncam[seeds]
    s2 = a.scatter(pts[seeds, 0], pts[seeds, 1], c=sm, s=8, cmap="viridis", vmin=1, vmax=3)
    a.set_title(f"(b) 씨앗만 ({len(seeds):,} 개) — 여기가 판정을 좌우한다\n"
                f"2 대 이상이 본 씨앗 {100.0*(sm>=2).mean():.1f} %", fontsize=11)
    a.set_xlabel("x [m] — 앞쪽 →"); a.set_ylabel("y [m] — 왼쪽 ↑")
    a.set_aspect("equal"); a.set_xlim(0.15, 1.0); a.set_ylim(-0.65, 0.65)
    cb = plt.colorbar(s2, ax=a, fraction=0.046, ticks=[1, 2, 3]); cb.set_label("카메라 수")

    # (c) 겹친 점의 불일치 공간 분포
    a = ax[2]
    multi = np.flatnonzero(ncam >= 2)
    a.scatter(pts[:, 0], pts[:, 1], c="0.9", s=0.8)
    if len(multi):
        o = np.argsort(spread)
        s3 = a.scatter(pts[multi[o], 0], pts[multi[o], 1], c=spread[o], s=4,
                       cmap="magma", vmin=0, vmax=max(0.2, float(np.percentile(spread, 99))))
        plt.colorbar(s3, ax=a, fraction=0.046, label="카메라 간 값 폭")
    a.set_title("(c) 겹친 점에서 카메라들이 얼마나 다르게 말하는가\n"
                "밝을수록 불일치가 크다", fontsize=11)
    a.set_xlabel("x [m] — 앞쪽 →"); a.set_ylabel("y [m] — 왼쪽 ↑")
    a.set_aspect("equal"); a.set_xlim(0.15, 1.0); a.set_ylim(-0.65, 0.65)

    # (d) 프레임별 겹침 비율
    a = ax[3]
    idx = np.arange(len(rows))
    f = np.array([r["frac"] for r in rows])
    a.stackplot(idx, f[:, 0] * 100, f[:, 1] * 100, f[:, 2] * 100,
                labels=["1 대", "2 대", "3 대"], colors=["#440154", "#21918c", "#fde725"])
    a.plot(idx, np.array([r["seed_multi"] for r in rows]) * 100, "w-", lw=2.5)
    a.plot(idx, np.array([r["seed_multi"] for r in rows]) * 100, "k--", lw=1.6,
           label="씨앗 중 2 대 이상")
    a.set_title("(d) 프레임별 겹침 비율", fontsize=11)
    a.set_xlabel("청크"); a.set_ylabel("융합 점의 %"); a.set_ylim(0, 100)
    a.legend(fontsize=9, loc="center right")

    # (e) 불일치 분포
    a = ax[4]
    allspread = np.concatenate([r["spread"] for r in rows if len(r["spread"])])
    a.hist(allspread, bins=60, color="tab:purple", alpha=0.8)
    a.axvline(float(np.median(allspread)), color="k", lw=1.5,
              label=f"중앙 {np.median(allspread):.3f}")
    a.set_yscale("log")
    a.set_title("(e) 겹친 점의 카메라 간 값 폭 (전 프레임)", fontsize=11)
    a.set_xlabel("max − min (원시 스케일, 씬 최댓값으로 정규화)"); a.set_ylabel("점 개수 (log)")
    a.legend(fontsize=9); a.grid(alpha=0.3)

    # (f) 표
    a = ax[5]; a.axis("off")
    sm_all = np.array([r["seed_multi"] for r in rows])
    cellt = [
        ["", "값"],
        ["융합 복셀 크기", f"{cfg.timing.fusion_voxel_size*1000:.0f} mm"],
        ["융합 점 (프레임 평균)", f"{np.mean([r['n'] for r in rows]):,.0f}"],
        ["1 대만 본 점", f"{np.mean([r['frac'][0] for r in rows])*100:.1f} %"],
        ["2 대가 본 점", f"{np.mean([r['frac'][1] for r in rows])*100:.1f} %"],
        ["3 대가 본 점", f"{np.mean([r['frac'][2] for r in rows])*100:.1f} %"],
        ["씨앗 중 2 대 이상", f"{np.nanmean(sm_all)*100:.1f} %"],
        ["겹친 점 값 폭 (중앙)", f"{np.median(allspread):.4f}"],
        ["값 폭 > 0.1 인 비율", f"{100.0*(allspread>0.1).mean():.1f} %"],
    ]
    t = a.table(cellText=cellt[1:], colLabels=cellt[0], loc="center", cellLoc="left",
                colWidths=[0.62, 0.38])
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.0, 2.0)
    for j in range(2):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    a.set_title("(f) 요약 — 의미 융합 규칙이 중요한가", fontsize=11, y=0.92)

    fig.suptitle("multi-view overlap — 융합 점 하나를 몇 대가 보는가, 그리고 서로 다르게 말하는가",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "step5-overlap.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
