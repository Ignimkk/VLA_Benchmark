"""F8 판정 — 시드 추출의 퍼센타일 경로가 실측 attention 에서 실제로 퇴화하는가.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.step6_f8_seeds --records run_0004 \\
        --attention attention_step1_run0004.npz

## 무엇이 문제라고 되어 있나

계획서의 F8: `extract_seeds` (`target_grounding.py:183`) 는 절대 문턱(`seed_threshold`)이 없으면
퍼센타일로 자른다.

    cut = float(np.percentile(att, seed_percentile))   # seed_percentile = 95
    keep = np.nonzero(att >= cut)[0]

`cut` 이 0.0 이 되려면 **점의 95 % 이상이 0** 이어야 한다 (p95 는 "95 % 가 이 값 이하" 인 값이다 —
0 이 하위에 몇 % 있느냐는 상관없다). 그렇게 되면 `att >= 0.0` 이 모든 점을 고르고, attention 이
0 인 점도 시드가 된다.

**`NO_ATTENTION` 검사는 완전히 평탄한 맵만 거른다.** 그래서 위험 구간은 정확히
**0 셀 95 % 이상 100 % 미만** — 아무도 잡지 않는 띠다.

함수의 docstring 도 이 성질을 숨기지 않는다 — *"a percentile always yields seeds — even from a
flat map, where the top 5% is an arbitrary 5%."* 설계자가 알고 둔 것이다. 그러므로 판정할 것은
**"코드가 그런가" 가 아니라 "실측에서 실제로 일어나는가, 일어나면 결과가 달라지는가"** 다.

## 어떻게 재는가

파이프라인을 실제로 돌리면서 `extract_seeds` 를 감싸 **그 함수가 받은 바로 그 배열**을 기록한다.
합성 attention 이 아니라 실측 attention 이 lifting 을 거쳐 점마다 실린 값이다. 프레임마다:

* 점 개수와 그중 attention 이 0 인 점의 비율
* p95 컷 값 — **0.0 이면 F8 이 발동한 것**
* 컷을 통과한 점 수, `max_seed_points`(4000) 로 자른 뒤 남은 시드 수
* **시드로 뽑힌 점들의 최소 attention** — 0 이면 "0 값 점이 시드가 됐다" 는 뜻
"""

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments import figstyle
    figstyle.use_korean()


def main() -> None:
    _style()
    from benchmark.ag3s import target_grounding as tg
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.pipeline import AG3S
    from benchmark.ag3s.experiments.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene
    from benchmark.trajopt.experiments.esdf_rollout import phase_for

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--attention", default="attention_step1_run0004.npz")
    ap.add_argument("--step1-json", default="benchmark/ag3s/docs/step-01-attention.json")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    args = ap.parse_args()

    run = load_run(args.records, limit=args.frames)
    blob = np.load(args.attention, allow_pickle=False)
    cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    A = np.asarray(blob["attention"], np.float32)
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    ci = [str(c) for c in blob["cameras"]].index("cam_high")

    scene = replay_scene(run)
    ag = AG3S(AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False}}),
        robot_model=build_robot_model(scene),
        constraint_robot_model=build_constraint_robot_model(scene, link_filter=ARM_LINKS))

    calls = []
    original = tg.extract_seeds

    def spy(attention, config, seed_percentile, seed_threshold=None):
        att = np.asarray(attention, np.float32).reshape(-1)
        out = original(attention, config, seed_percentile, seed_threshold)
        cut = (float(np.percentile(att, seed_percentile)) if seed_threshold is None
               else float(seed_threshold))
        keep_before = int((att >= cut).sum()) if att.size else 0
        calls.append(dict(
            n=int(att.size),
            n_zero=int((att <= 0.0).sum()),
            cut=cut,
            keep_before=keep_before,
            n_seeds=int(np.asarray(out).size),
            seed_min=(float(att[np.asarray(out, np.int64)].min())
                      if np.asarray(out).size else float("nan")),
            att_max=float(att.max()) if att.size else float("nan"),
            threshold_used=seed_threshold is not None,
        ))
        return out

    tg.extract_seeds = spy
    try:
        for i, step in enumerate(run.steps):
            pose_scene(scene, step)
            head = scene.capture("zed_left")
            ag.process(depth=head.depth, camera_intrinsics=head.camera_intrinsics,
                       T_base_cam=head.T_base_cam,
                       attention_map=A[i, di, ai, cell["layer"], cell["head"], ci],
                       robot_state=head.robot_state,
                       phase=phase_for(step.t_step, (24, 56, 72)))
    finally:
        tg.extract_seeds = original
        scene.close()

    sweep = _sweep(args)
    _report(calls, args)
    _sweep_report(sweep)
    _figure(calls, sweep, args)


def _sweep(args):
    """이 모델에서 F8 이 **도달 가능한가** — 모든 layer/head/denoise/agg/frame 조합의 희소도.

    선택된 cell 하나가 안전한 것과 결함이 없는 것은 다르다. Step 1 이 다른 head 를 골랐다면
    어땠을지를 같은 기록에서 바로 볼 수 있다.
    """
    blob = np.load(args.attention, allow_pickle=False)
    A = np.asarray(blob["attention"], np.float32)
    z = (A == 0).mean(axis=(-1, -2)) * 100.0
    band = (z >= 95.0) & (z < 100.0)
    flat = z >= 100.0
    import collections
    idx = np.argwhere(band)
    top = collections.Counter((int(i[3]), int(i[4])) for i in idx).most_common(5)
    return dict(z=z, n=int(z.size), n_flat=int(flat.sum()), n_band=int(band.sum()), top=top)


def _sweep_report(sw):
    print("\n" + "=" * 100)
    print("F8 이 이 모델에서 도달 가능한가 — 모든 layer/head/denoise/agg/frame 조합")
    print(f"   전체 조합                                    : {sw['n']}")
    print(f"   0 셀 100 %   (NO_ATTENTION 이 잡는다)        : {sw['n_flat']}")
    print(f"   0 셀 95~99.99 % (아무도 잡지 않는다 = 위험)  : {sw['n_band']}"
          f"  ({sw['n_band']/max(sw['n'],1)*100:.2f} %)")
    print(f"   위험이 몰린 (layer, head)                    : {sw['top']}")


def _report(calls, args):
    print("=" * 100)
    print(f"F8 판정 — {args.records}, seed_percentile=95, max_seed_points=4000, "
          f"seed_threshold=None (퍼센타일 경로 활성)")
    print("=" * 100)
    print(f"{'호출':>4} {'점 수':>8} {'0인 점':>8} {'0 비율':>7} {'p95 컷':>10} "
          f"{'컷 통과':>8} {'시드':>7} {'시드 최소 att':>13}  F8?")
    for k, c in enumerate(calls):
        z = c["n_zero"] / max(c["n"], 1) * 100
        fired = c["cut"] <= 0.0
        degen = fired and c["seed_min"] <= 0.0
        tag = "발동+0값시드" if degen else ("컷=0" if fired else "")
        print(f"{k:>4} {c['n']:>8} {c['n_zero']:>8} {z:>6.1f}% {c['cut']:>10.6f} "
              f"{c['keep_before']:>8} {c['n_seeds']:>7} {c['seed_min']:>13.6f}  {tag}")

    n_fire = sum(1 for c in calls if c["cut"] <= 0.0)
    n_degen = sum(1 for c in calls if c["cut"] <= 0.0 and c["seed_min"] <= 0.0)
    n_capped = sum(1 for c in calls if c["keep_before"] > c["n_seeds"])
    print("\n" + "=" * 100)
    print(f"호출 {len(calls)} 회 중")
    print(f"  p95 컷이 0.0 이 된 횟수          : {n_fire}")
    print(f"  그래서 0 값 점이 시드가 된 횟수  : {n_degen}")
    print(f"  max_seed_points(4000) 로 잘린 횟수: {n_capped}")
    if calls:
        zs = [c["n_zero"] / max(c["n"], 1) * 100 for c in calls]
        print(f"  attention 이 0 인 점의 비율      : {min(zs):.1f}% ~ {max(zs):.1f}% "
              f"(평균 {np.mean(zs):.1f}%)")
    print("\n>>> 판정: " + (
        "F8 발동 — 0 값 점이 시드가 된다" if n_degen else
        ("컷이 0 이 되지만 캡이 강한 시드만 남겨 0 값 시드는 없다" if n_fire else
         "선택된 cell 에서는 발동하지 않는다 — p95 컷이 0 이 되려면 점의 95 % 이상이 "
         "0 이어야 하는데 실측은 5~6 % 다")))


def _figure(calls, sweep, args):
    import matplotlib.pyplot as plt
    if not calls:
        return
    k = np.arange(len(calls))
    zpct = [c["n_zero"] / max(c["n"], 1) * 100 for c in calls]
    cut = [c["cut"] for c in calls]
    smin = [c["seed_min"] for c in calls]
    amax = [c["att_max"] for c in calls]
    nseed = [c["n_seeds"] for c in calls]

    fig, axs = plt.subplots(1, 3, figsize=(19.5, 5.4))

    a = axs[0]
    a.plot(k, zpct, "o-", color="tab:purple", lw=2, ms=5, label="실측 (선택된 cell)")
    a.axhline(95.0, color="crimson", ls="--", lw=2.2)
    a.text(0.2, 88, "95 % — 이 위로 가야 p95 컷이 0 이 되고 F8 이 발동한다",
           color="crimson", fontsize=9.5)
    a.set_ylim(0, 103)
    a.annotate("", xy=(len(k) * 0.5, 95), xytext=(len(k) * 0.5, np.mean(zpct)),
               arrowprops=dict(arrowstyle="<->", color="tab:green", lw=2))
    a.text(len(k) * 0.52, 50, f"여유 {95-np.mean(zpct):.0f} %p", color="tab:green",
           fontsize=11, fontweight="bold")
    a.set_title("(a) 선택된 cell 은 발동 조건에서 멀다", fontsize=11)
    a.set_xlabel("extract_seeds 호출"); a.set_ylabel("attention 이 0 인 점의 비율 [%]")
    a.legend(fontsize=9, loc="center left"); a.grid(alpha=0.3)

    a = axs[1]
    z = sweep["z"].reshape(-1)
    a.hist(z, bins=60, color="0.7", edgecolor="0.4")
    a.axvspan(95, 100, color="crimson", alpha=0.22)
    a.axvline(100, color="tab:blue", lw=2)
    a.text(93, a.get_ylim()[1] * 0.55, f"위험 띠 95~99.99 %\n{sweep['n_band']} 조합\n"
           "아무도 잡지 않는다", color="crimson", fontsize=9.5, ha="right", fontweight="bold")
    a.text(100, a.get_ylim()[1] * 0.2, f" 100 % = {sweep['n_flat']} 조합\n NO_ATTENTION 이 잡는다",
           color="tab:blue", fontsize=9.5)
    a.axvline(float(np.mean(zpct)), color="tab:green", lw=2.5)
    a.text(float(np.mean(zpct)) + 2, a.get_ylim()[1] * 0.8, "선택된 cell", color="tab:green",
           fontsize=10, fontweight="bold")
    a.set_yscale("log")
    a.set_title("(b) 이 모델에서 F8 은 도달 가능한가\n"
                "모든 layer/head/denoise/agg/frame 조합의 희소도", fontsize=11)
    a.set_xlabel("16x16 맵에서 0 인 셀의 비율 [%]"); a.set_ylabel("조합 수 (로그)")
    a.grid(alpha=0.3)

    a = axs[2]; a.axis("off")
    n_fire = sum(1 for c in calls if c["cut"] <= 0.0)
    n_degen = sum(1 for c in calls if c["cut"] <= 0.0 and c["seed_min"] <= 0.0)
    n_capped = sum(1 for c in calls if c["keep_before"] > c["n_seeds"])
    rows = [["항목", "값"],
            ["extract_seeds 호출 수", f"{len(calls)}"],
            ["p95 컷이 0.0 이 된 횟수", f"{n_fire}"],
            ["0 값 점이 시드가 된 횟수", f"{n_degen}"],
            ["0 인 점 비율 (선택된 cell)", f"{min(zpct):.1f}% ~ {max(zpct):.1f}%"],
            ["발동에 필요한 비율", "95 % 이상"],
            ["시드 수 (최소~최대)", f"{min(nseed)} ~ {max(nseed)}"],
            ["", ""],
            ["전체 조합 (모든 layer/head)", f"{sweep['n']}"],
            ["0셀 100 % — NO_ATTENTION 이 잡음", f"{sweep['n_flat']}"],
            ["0셀 95~99.99 % — 아무도 안 잡음", f"{sweep['n_band']}"],
            ["위험이 몰린 (layer, head)",
             ", ".join(f"L{l}H{h}" for (l, h), _ in sweep["top"][:3])]]
    t = a.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.0, 1.5)
    for j in range(2):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    for j in range(2):
        t[(10, j)].set_facecolor("#f7d6d6")
    a.set_title("(c) 요약 — 선택된 cell 은 안전, 그러나 결함은 도달 가능",
                fontsize=11, y=0.92)

    fig.suptitle(f"F8 — 시드 추출의 퍼센타일 경로가 실측 attention 에서 퇴화하는가 "
                 f"({args.records})", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    o = OUT / f"step6-f8-seeds-{args.records[-4:]}.png"
    fig.savefig(o, dpi=110, bbox_inches="tight")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
