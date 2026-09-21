"""Step 5 설명용 시각화 — `attention_lifting.py` 가 무엇을 하는가.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.studies.step5_lifting_figures

규칙 A: 실제 씬 + 그래프 + 표. 실측 attention(`attention_step1_run0004.npz`, 카메라 3대)을 쓴다.
판정은 하지 않는다 — 이 그림은 **설명**이고, F2/F9 는 사용자와 함께 판정한다.
"""

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
ATT_CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
LABEL = ("head (zed_left)", "왼손목", "오른손목")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments.common import figstyle
    figstyle.use_korean()


def main() -> None:
    _style()
    import matplotlib.pyplot as plt

    from benchmark.ag3s.stages.attention_lifting import (
        GridAttentionAdapter, lift, normalize_attention)
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.stages.reconstruction import backproject
    from benchmark.ag3s.experiments.reports.grounding_report import build_robot_model
    from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--attention", default="attention_step1_run0004.npz")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/archive/step-verification-20260904/step-01-attention.json")
    ap.add_argument("--frame", type=int, default=10)
    args = ap.parse_args()

    blob = np.load(args.attention, allow_pickle=False)
    cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    grids = {c: np.asarray(blob["attention"][args.frame, di, ai, cell["layer"], cell["head"],
                                              [str(x) for x in blob["cameras"]].index(c)],
                           np.float32) for c in ATT_CAMS}

    run = load_run(args.records, limit=args.frame + 1)
    scene = replay_scene(run)
    build_robot_model(scene)
    pose_scene(scene, run.steps[args.frame])
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "pointcloud": {"range_max": 2.0}})

    frames, clouds, lifted = {}, {}, {}
    adapter = GridAttentionAdapter.from_config(cfg.attention)
    for cam, ac in zip(CAMS, ATT_CAMS):
        f = scene.capture(cam)
        frames[cam] = f
        c = backproject(f.depth, f.camera_intrinsics, f.T_base_cam, cfg.pointcloud)
        clouds[cam] = c
        lifted[cam] = lift(c, grids[ac], cfg.attention, adapter=adapter, image_hw=f.hw)

    fig, axs = plt.subplots(2, 3, figsize=(19.5, 10.4))
    ax = axs.ravel()
    head = frames["zed_left"]

    # (a) 실제 씬 — depth 위에 확대된 attention
    pm = adapter.to_pixel_map(grids["cam_high"], head.hw)
    ax[0].imshow(np.where(head.depth > 2.5, np.nan, head.depth), cmap="gray")
    im = ax[0].imshow(pm, cmap="inferno", alpha=0.55)
    ax[0].set_title(f"(a) 실제 씬 — head depth 위에 attention\n"
                    f"16x16 격자를 {head.hw[0]}x{head.hw[1]} 로 확대 "
                    f"({cfg.attention.interpolation})", fontsize=11)
    ax[0].axis("off")
    plt.colorbar(im, ax=ax[0], fraction=0.04, label="attention (raw)")

    # (b) 원시 16x16 격자 3대
    ax[1].axis("off")
    ax[1].set_title("(b) VLA 가 실제로 준 것 — 카메라별 16x16 격자\n"
                    "`GridAttentionAdapter` 가 여기서 경계를 긋는다", fontsize=11)
    for j, (ac, lb) in enumerate(zip(ATT_CAMS, LABEL)):
        sub = ax[1].inset_axes([0.02 + j * 0.33, 0.18, 0.30, 0.62])
        g = grids[ac]
        sub.imshow(g, cmap="inferno")
        sub.set_title(f"{lb}\nmax {g.max():.3f}  min {g.min():.3f}", fontsize=8.5)
        sub.set_xticks([]); sub.set_yticks([])

    # (c) 3D 로 올린 결과 — 위에서 본 산점도
    a = ax[2]
    c = clouds["zed_left"]
    v = lifted["zed_left"].attention
    o = np.argsort(v)
    s = a.scatter(c.points[o, 0], c.points[o, 1], c=v[o], s=1.2, cmap="inferno", vmin=0, vmax=1)
    a.set_title("(c) 올린 결과 — 점마다 attention 하나\n"
                f"점 {len(c):,} 개 — 하나도 버리지 않는다", fontsize=11)
    a.set_xlabel("x [m] — 앞쪽 →"); a.set_ylabel("y [m] — 왼쪽 ↑")
    a.set_aspect("equal"); a.set_xlim(0.1, 1.1); a.set_ylim(-0.7, 0.7)
    plt.colorbar(s, ax=a, fraction=0.046, label="정규화된 attention")

    # (d) 카메라별 원시 값 분포 — F2 가 사는 자리
    a = ax[3]
    for (cam, ac), lb, col in zip(zip(CAMS, ATT_CAMS), LABEL,
                                  ("tab:blue", "tab:orange", "tab:green")):
        raw = np.asarray(lifted[cam].raw_attention, float)
        a.hist(raw, bins=60, histtype="step", lw=2, color=col,
               label=f"{lb}  중앙 {np.median(raw):.4f}  p99 {np.percentile(raw,99):.4f}")
        lo, hi = np.percentile(raw, cfg.attention.percentile_range)
        a.axvspan(lo, hi, color=col, alpha=0.07)
    a.set_yscale("log")
    a.set_title("(d) 카메라마다 원시 attention 의 눈금이 다르다\n"
                "옅은 띠 = 그 카메라의 percentile 구간 (정규화가 여기를 0~1 로 편다)",
                fontsize=11)
    a.set_xlabel("원시 attention"); a.set_ylabel("점 개수 (log)")
    a.legend(fontsize=8.5); a.grid(alpha=0.3)

    # (e) 정규화 전/후
    a = ax[4]
    raw = np.asarray(lifted["zed_left"].raw_attention, float)
    nrm = np.asarray(lifted["zed_left"].attention, float)
    o = np.argsort(raw)
    a.plot(np.linspace(0, 100, len(raw)), raw[o] / max(raw.max(), 1e-9), lw=2,
           label="원시 (최대값으로 나눔)")
    a.plot(np.linspace(0, 100, len(raw)), nrm[o], lw=2, label="정규화 후")
    lo, hi = cfg.attention.percentile_range
    a.axvline(lo, color="k", ls=":", lw=1.2); a.axvline(hi, color="k", ls=":", lw=1.2)
    a.text(hi, 0.5, f"  percentile_range\n  = ({lo}, {hi})", fontsize=9)
    a.set_title(f"(e) `normalize_attention` — mode = {cfg.attention.normalization}\n"
                "순위는 보존하고 값만 편다 (head 카메라)", fontsize=11)
    a.set_xlabel("점의 분위 [%]"); a.set_ylabel("값"); a.legend(fontsize=9); a.grid(alpha=0.3)

    # (f) 표
    a = ax[5]; a.axis("off")
    rows = [
        ["입력 1", "PointCloud — uv 를 보존한 채로 온다"],
        ["입력 2", "VLA attention (여기서는 16x16 격자 x 카메라 3대)"],
        ["출력", f"AttentionPointCloud — 점 {len(clouds['zed_left']):,} 개, 값 2 벌"],
        ["값 2 벌인 이유", "정규화본은 문턱용(동점 다수), 원시본은 최고점용(argmax)"],
        ["경계 1", "`AttentionAdapter` — VLA 종류를 여기서 격리"],
        ["경계 2", "정규화는 픽셀맵이 아니라 올린 점 값에 건다"],
        ["버리는 점", "없음 — attention 이 낮다고 기하를 지우지 않는다"],
        ["이 스텝의 미판정", "F2 (카메라별 정규화), F9 (image_hw 폴백)"],
    ]
    t = a.table(cellText=rows, loc="center", cellLoc="left", colWidths=[0.32, 0.68])
    t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1.0, 1.9)
    for i in range(len(rows)):
        t[(i, 0)].set_facecolor("#eeeeee")
        t[(i, 0)].set_text_props(fontweight="bold")
    a.set_title("(f) 모듈 요약 — 무엇을 받아 무엇을 내놓는가", fontsize=11, y=0.93)

    fig.suptitle(f"Step 5 — attention_lifting.py.  run_0004 프레임 {args.frame}, "
                 f"실측 attention (layer {cell['layer']} head {cell['head']}, "
                 f"agg={cell['agg']}, denoise={cell['denoise']})", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "step5-lifting.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print(f"wrote {out}")

    # 숫자도 같이 찍는다 (표에 넣을 것)
    print()
    print("카메라별 원시 attention 통계 (같은 프레임, 같은 layer/head)")
    print(f"  {'카메라':<16} {'중앙':>10} {'p95':>10} {'p99':>10} {'최대':>10}  "
          f"{'정규화 lo':>10} {'정규화 hi':>10}")
    for cam, ac, lb in zip(CAMS, ATT_CAMS, LABEL):
        r = np.asarray(lifted[cam].raw_attention, float)
        lo, hi = np.percentile(r, cfg.attention.percentile_range)
        print(f"  {lb:<16} {np.median(r):>10.5f} {np.percentile(r,95):>10.5f} "
              f"{np.percentile(r,99):>10.5f} {r.max():>10.5f}  {lo:>10.5f} {hi:>10.5f}")
    scene.close()


if __name__ == "__main__":
    main()
