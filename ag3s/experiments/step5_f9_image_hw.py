"""F9 판정 — `lift()` 의 `image_hw` 기본값이 실측에서 실제로 어긋나는가.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.step5_f9_image_hw --records run_0004 \\
        --attention attention_step1_run0004.npz

## 무엇이 문제라고 되어 있나

계획서의 F9: `lift()` (`attention_lifting.py:265`) 는 `image_hw` 를 받지 못하면 **클라우드에 남아
있는 uv 의 최댓값**으로 해상도를 추정한다.

    hw = image_hw or (int(cloud.uv[:, 1].max()) + 1, int(cloud.uv[:, 0].max()) + 1)

클라우드는 이미 여러 번 걸러진 뒤다 — 깊이 유효성, `range_max`, 그리고 **로봇 자기 필터**.
그래서 uv 가 이미지 경계까지 못 닿으면 추정 해상도가 실제보다 작고, `to_pixel_map` 이 16x16
attention 을 **틀린 크기로 펼친다.** 펼친 맵을 원래 uv 로 샘플링하므로 attention 이 이미지 위에서
밀린다. 함수의 docstring 은 이 성질을 숨기지 않는다 — *"pass it explicitly if the cloud has been
heavily downsampled and its `uv` no longer reaches the image border."*

`multiview.py` 는 `image_hw=observation.image_hw` 를 넘긴다. 그러나 **`AG3S.process` 의
`image_hw` 는 기본값이 `None`** 이고, 이 검토의 모든 롤아웃이 그 인자를 넘기지 않았다. 즉 실측
경로가 정확히 F9 조건 위에 있다.

## 어떻게 재는가

파이프라인을 돌리면서 `lift` 를 감싸 **그 함수가 실제로 쓴 hw** 와 **깊이 이미지의 참 크기**를
나란히 기록한다. 그리고 어긋남이 attention 을 얼마나 밀어내는지를 픽셀로 환산한다 — 16x16 맵을
`H_est` 로 펼치면 셀 하나가 `H_est/16` 픽셀이므로, 이미지 아래쪽 끝에서의 어긋남은
`H_true - H_est` 픽셀이다.

그리고 **어긋남이 답을 바꾸는지**를 본다. 같은 프레임을 파이프라인 둘에 나란히 넣는다 — 하나는
지금처럼 `image_hw` 를 넘기지 않고, 하나는 깊이 이미지의 참 크기를 넘긴다. 둘의 grounding 결과
(target 이 어느 물체인지, 무게중심이 어디인지)를 비교한다. 어긋나도 답이 같다면 F9 는 잠복이고,
답이 달라진다면 실효 결함이다.
"""

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")

#: spy 가 F9 수정 이전 동작을 재현할지. 루프가 호출 직전에 켜고 끈다.
LEGACY = {"on": False}


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
    from benchmark.ag3s import attention_lifting as al
    from benchmark.ag3s import pipeline as pl
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

    def _make():
        return AG3S(AG3SConfig.from_dict({
            "collision_backend": "esdf",
            "pointcloud": {"range_max": args.range_max},
            "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                     "exclude_support_surfaces": False}}),
            robot_model=build_robot_model(scene),
            constraint_robot_model=build_constraint_robot_model(scene, link_filter=ARM_LINKS))

    # 둘은 **독립된 TSDF 를 쌓는다.** 거리장이 프레임에 걸쳐 누적되므로 한 인스턴스를 두 번
    # 부르면 두 번째가 첫 번째의 적분 위에 얹혀 비교가 성립하지 않는다.
    ag = _make()
    ag_fixed = _make()

    calls: list[dict] = []
    original = al.lift

    def spy(cloud, attention, config=None, *, adapter=None, image_hw=None, **kw):
        if LEGACY["on"]:
            # F9 수정 이전의 동작 — `lift` 가 걸러진 uv 로 스스로 추정하게 둔다.
            image_hw = None
        used = image_hw
        if used is None and getattr(cloud, "uv", None) is not None and len(cloud):
            used = (int(cloud.uv[:, 1].max()) + 1, int(cloud.uv[:, 0].max()) + 1)
        out = original(cloud, attention, config, adapter=adapter, image_hw=image_hw, **kw)
        calls.append(dict(passed=image_hw is not None, used=used,
                          n_points=int(len(cloud)),
                          uv_min=(None if getattr(cloud, "uv", None) is None or not len(cloud)
                                  else (int(cloud.uv[:, 1].min()), int(cloud.uv[:, 0].min()))),
                          # 입력이 실제로 달라졌는지 확인하려면 lift 가 내놓은 값을 봐야 한다.
                          # 이것이 같다면 hw 는 아무 영향이 없다는 뜻이고, 그러면 F9 는 성립하지
                          # 않는다 — "답이 같다" 와 "입력이 애초에 같았다" 는 다른 판정이다.
                          att=np.asarray(out.raw_attention if out.raw_attention is not None
                                         else out.attention, np.float64).copy()))
        return out

    al.lift = spy
    pl.lift = spy          # pipeline 이 from-import 로 이름을 잡아 두었을 수 있다
    compare: list[dict] = []
    try:
        for i, step in enumerate(run.steps):
            pose_scene(scene, step)
            head = scene.capture("zed_left")
            att = A[i, di, ai, cell["layer"], cell["head"], ci]
            true_hw = tuple(int(x) for x in np.asarray(head.depth).shape[:2])
            common = dict(depth=head.depth, camera_intrinsics=head.camera_intrinsics,
                          T_base_cam=head.T_base_cam, attention_map=att,
                          robot_state=head.robot_state,
                          phase=phase_for(step.t_step, (24, 56, 72)))
            calls_before = len(calls)
            # `pipeline` 이 이제 depth 로 기본값을 채우므로 (F9 수정), **수정 전 동작을
            # 재현하려면** 그 기본값을 spy 에서 도로 지워야 한다. 비교하는 것은 "고치기 전"과
            # "고친 뒤"이지 두 인자 스타일이 아니다.
            LEGACY["on"] = True
            cs_now = ag.process(**common)
            LEGACY["on"] = False
            cs_fixed = ag_fixed.process(**common)
            for c in calls[calls_before:]:
                c["true_hw"] = true_hw
                c["frame"] = i
            compare.append(dict(
                frame=i,
                now=_describe(scene, cs_now), fixed=_describe(scene, cs_fixed)))
    finally:
        al.lift = original
        pl.lift = original
        _report(calls, args)
        _compare_report(compare)
        scene.close()

    _figure(calls, args, compare)


def _describe(scene, cs):
    """grounding 결과를 비교 가능한 형태로 — 어느 물체이고 무게중심이 어디인가."""
    import mujoco
    from benchmark.ag3s.experiments.mujoco_source import is_robot_body
    if cs.target is None:
        return dict(name="-", centroid=None)
    c = np.asarray(cs.target.centroid, float)
    best, bd = "-", 1e9
    for b in range(scene.model.nbody):
        nm = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not nm or is_robot_body(nm) or any(
                k in nm for k in ("table", "shelf", "floor", "world", "ground",
                                  "com_target", "_ee_target", "office")):
            continue
        dd = float(np.linalg.norm(c - scene.data.xpos[b]))
        if dd < bd:
            best, bd = nm, dd
    return dict(name=best, centroid=c)


def _compare_report(compare):
    print("\n" + "=" * 92)
    print("어긋남이 답을 바꾸는가 — image_hw 를 넘기지 않은 쪽 대 넘긴 쪽")
    print("=" * 92)
    print(f"{'프레임':>6} {'지금 target':>13} {'고친 target':>13} {'무게중심 차이':>13}  다른가")
    n_diff_name = 0
    shifts = []
    for r in compare:
        a, b = r["now"], r["fixed"]
        if a["centroid"] is None or b["centroid"] is None:
            d = float("nan")
        else:
            d = float(np.linalg.norm(a["centroid"] - b["centroid"])) * 1000
            shifts.append(d)
        diff = a["name"] != b["name"]
        n_diff_name += diff
        print(f"{r['frame']:>6} {a['name']:>13} {b['name']:>13} {d:>12.1f} mm  "
              f"{'<<< 바뀐다' if diff else ''}")
    print(f"\n  target 물체가 달라진 프레임 : {n_diff_name} / {len(compare)}")
    if shifts:
        print(f"  무게중심 이동             : {min(shifts):.1f} ~ {max(shifts):.1f} mm "
              f"(평균 {np.mean(shifts):.1f})")
    print("\n>>> 판정: " + (
        f"실효 결함 — {n_diff_name} 프레임에서 target 물체 자체가 바뀐다" if n_diff_name else
        (f"target 물체는 같지만 무게중심이 최대 {max(shifts):.0f} mm 움직인다"
         if shifts and max(shifts) > 5 else
         "답이 바뀌지 않는다 — 잠복")))


def _report(calls, args):
    print("=" * 92)
    print(f"F9 판정 — {args.records}, AG3S.process 에 image_hw 를 넘기지 않는 경로")
    print("=" * 92)
    if not calls:
        print("lift 가 한 번도 불리지 않았다 — 계측 지점을 다시 봐야 한다.")
        return
    print(f"{'프레임':>6} {'넘겼나':>7} {'쓴 hw':>14} {'참 hw':>14} {'모자란 행':>9} "
          f"{'모자란 열':>9} {'uv 최소':>12} {'점 수':>8}")
    for c in calls:
        used = c["used"] or (0, 0)
        true = c.get("true_hw", (0, 0))
        print(f"{c['frame']:>6} {'예' if c['passed'] else '아니오':>7} "
              f"{str(used):>14} {str(true):>14} {true[0]-used[0]:>9} {true[1]-used[1]:>9} "
              f"{str(c['uv_min']):>12} {c['n_points']:>8}")

    pairs = [(calls[i], calls[i + 1]) for i in range(0, len(calls) - 1, 2)
             if not calls[i]["passed"] and calls[i + 1]["passed"]]
    if pairs:
        deltas = [float(np.abs(a["att"] - b["att"]).max()) for a, b in pairs
                  if a["att"].shape == b["att"].shape]
        changed = [float((a["att"] != b["att"]).mean()) * 100 for a, b in pairs
                   if a["att"].shape == b["att"].shape]
        print(f"\n  lift 가 내놓은 attention 이 실제로 달라졌는가")
        print(f"    최대 절대차          : {max(deltas):.6g}")
        print(f"    값이 달라진 점의 비율 : {min(changed):.1f}% ~ {max(changed):.1f}%")

    dh = [c["true_hw"][0] - (c["used"] or (0, 0))[0] for c in calls]
    dw = [c["true_hw"][1] - (c["used"] or (0, 0))[1] for c in calls]
    print("\n" + "=" * 92)
    print(f"lift 호출 {len(calls)} 회, image_hw 를 넘긴 횟수 {sum(c['passed'] for c in calls)}")
    print(f"  모자란 행 (참 H - 쓴 H) : {min(dh)} ~ {max(dh)} 픽셀 (평균 {np.mean(dh):.1f})")
    print(f"  모자란 열 (참 W - 쓴 W) : {min(dw)} ~ {max(dw)} 픽셀 (평균 {np.mean(dw):.1f})")
    if calls:
        H = calls[0]["true_hw"][0]
        W = calls[0]["true_hw"][1]
        print(f"  참 해상도 {H} x {W},  16x16 맵의 셀 하나 = {H/16:.1f} x {W/16:.1f} 픽셀")
    worst = max(max(dh), max(dw))
    print("\n>>> 판정: " + (
        f"F9 발동 — 최대 {worst} 픽셀 어긋난다" if worst > 0 else
        "이 데이터에서는 어긋나지 않는다 — uv 가 이미지 경계까지 닿는다"))


def _figure(calls, args, compare=None):
    import matplotlib.pyplot as plt
    if not calls:
        return
    k = np.arange(len(calls))
    dh = [c["true_hw"][0] - (c["used"] or (0, 0))[0] for c in calls]
    dw = [c["true_hw"][1] - (c["used"] or (0, 0))[1] for c in calls]
    H, W = calls[0]["true_hw"]

    fig, axs = plt.subplots(1, 3, figsize=(19.5, 5.3))

    a = axs[0]
    a.plot(k, dh, "o-", color="crimson", lw=2, ms=5, label="모자란 행 (H)")
    a.plot(k, dw, "s-", color="tab:blue", lw=2, ms=5, label="모자란 열 (W)")
    a.axhline(0, color="k", lw=1.2)
    a.axhline(H / 16, color="0.5", ls="--", lw=1.4)
    a.text(0.2, H / 16 + 1, f"attention 셀 한 칸 = {H/16:.0f} 픽셀", fontsize=9, color="0.35")
    a.set_title("(a) 추정 해상도가 참값에서 얼마나 모자란가\n0 이면 어긋나지 않는다", fontsize=11)
    a.set_xlabel("lift 호출"); a.set_ylabel("픽셀"); a.legend(fontsize=9); a.grid(alpha=0.3)

    a = axs[1]
    used_h = [(c["used"] or (0, 0))[0] for c in calls]
    used_w = [(c["used"] or (0, 0))[1] for c in calls]
    a.plot(k, used_h, "o-", color="crimson", lw=2, ms=5, label="쓴 H")
    a.plot(k, used_w, "s-", color="tab:blue", lw=2, ms=5, label="쓴 W")
    a.axhline(H, color="crimson", ls="--", lw=1.6)
    a.axhline(W, color="tab:blue", ls="--", lw=1.6)
    a.text(0.2, H + 4, f"참 H = {H}", color="crimson", fontsize=9)
    a.text(0.2, W + 4, f"참 W = {W}", color="tab:blue", fontsize=9)
    a.set_title("(b) 실제로 쓴 해상도 대 참 해상도", fontsize=11)
    a.set_xlabel("lift 호출"); a.set_ylabel("픽셀"); a.legend(fontsize=9); a.grid(alpha=0.3)

    a = axs[2]; a.axis("off")
    rows = [["프레임", "지금", "고치면", "무게중심"]]
    n_diff = 0
    for r in (compare or []):
        an, bn = r["now"]["name"], r["fixed"]["name"]
        if r["now"]["centroid"] is None or r["fixed"]["centroid"] is None:
            d = float("nan")
        else:
            d = float(np.linalg.norm(r["now"]["centroid"] - r["fixed"]["centroid"])) * 1000
        if an != bn or d > 5:
            n_diff += an != bn
            rows.append([str(r["frame"]), an, bn, f"{d:.0f} mm"])
    if len(rows) == 1:
        rows.append(["(차이 없음)", "-", "-", "-"])
    t = a.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.0, 1.8)
    for j in range(4):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    for ri, row in enumerate(rows[1:], start=1):
        if len(row) == 4 and row[1] != row[2] and row[1] != "-":
            for j in range(4):
                t[(ri, j)].set_facecolor("#f7d6d6")
    a.set_title(f"(c) 어긋남이 답을 바꾸는가 — 달라진 프레임만\n"
                f"target 물체 자체가 바뀐 프레임 {n_diff} 개 (분홍)",
                fontsize=11, y=0.9)

    fig.suptitle(f"F9 — lift() 의 image_hw 기본값이 실측에서 어긋나는가 ({args.records})",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    o = OUT / f"step5-f9-image-hw-{args.records[-4:]}.png"
    fig.savefig(o, dpi=110, bbox_inches="tight")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
