"""Step 7 — 상태 지연이 얼마부터 해로운가. 설정 한계는 그 지점에 맞게 잡혀 있는가.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.step7_state_lag --frames 8

## 왜 이 질문인가

`TimingConfig` 의 docstring 이 이 자리를 **직접 열어 두고 있다**:

> *"The defaults are deliberately loose (100 ms), because the numbers that matter come from the
> real robot's transport and control rates and **nobody has measured them here yet**."*

그런데 `multiview.py` 머리말은 같은 문서에서 이렇게 쓴다:

> *"A wrist camera moving at 1 rad/s and a 50 ms state lag put the cloud 2.5 cm from where it
> belongs — enough to smear a crate edge and, worse, to **misalign the self-filter so that the
> arm's own points survive and cluster into a phantom obstacle attached to the gripper**."*

즉 **설정 한계(100 ms)가 모듈 자신이 해롭다고 든 예(50 ms)보다 2 배 느슨하다.** 어느 쪽이 맞는지는
재야 안다. 그리고 이 기록의 관절 속도는 중앙 0.077 · 95 % 7.6 · 최대 17.4 rad/s 로, 예시의
1 rad/s 보다 훨씬 빠르다.

## 왜 지연을 만들어 넣는가

우리 기록은 카메라 셋이 **같은 MuJoCo 스텝**에서 나오므로 지연이 0 이다. 그래서 "촬영 시각
기구학이 좋다" 를 그냥 재면 아무것도 안 나온다. 대신 **지연을 넣어** 해가 시작되는 지점을 찾는다.

기록 한 스텝이 16 ms 이므로 `k` 스텝 전의 자세를 쓰면 정확히 `16k` ms 지연이다. 속도를 적분해
만드는 것보다 이쪽이 실제에 가깝다 — 진짜로 그 자세였던 순간이 있기 때문이다.

## 무엇을 재는가

영상은 프레임 `i` 의 것을 쓰고, **로봇 자세만** 프레임 `i-k` 의 것으로 바꾼다. 그러면 자기 필터
마스크도, 손목 카메라의 외부 파라미터도 함께 밀린다 — 실제 지연이 하는 일 그대로다.

* **로봇 누수** — 자기 필터를 통과해 **실제로 클라우드에 들어간** 점 중 참값이 로봇인 것.
  이것이 "그리퍼에 붙은 유령 장애물" 의 씨앗이다.
  마스크를 빠져나간 *픽셀* 을 세면 안 된다 — `range_max` 와 깊이 유효성에서 어차피 떨어지는
  픽셀이 섞여 지연 0 에서도 수만 개가 나온다. 실제로 통과한 점을 세야 뜻이 있고, 그 값은
  지연 0 에서 **0** 이어야 한다 (Step 7 융합 측정과 일치).
* **잃은 실물체 점** — 지연 때문에 마스크가 밀려 삭제된 실물체 점.
* **카메라 이동** — 지연된 자세로 푼 `T_base_cam` 이 참값에서 얼마나 떨어지는가.
"""

import argparse
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")
CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
STEP_MS = 16.0


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
    import dataclasses
    import mujoco

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.pipeline import AG3S
    from benchmark.ag3s.experiments.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.mujoco_source import camera_observation, is_robot_body
    from benchmark.ag3s.multiview import process_observation
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--frames", type=int, default=8, help="지연을 걸어 볼 프레임 수")
    ap.add_argument("--first", type=int, default=8, help="첫 프레임 (앞쪽에 지연분 여유가 필요)")
    ap.add_argument("--lags", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6, 8],
                    help="기록 스텝 단위 지연. 1 스텝 = 16 ms")
    ap.add_argument("--range-max", type=float, default=2.0)
    args = ap.parse_args()

    run = load_run(args.records, limit=args.first + args.frames)
    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    ag = AG3S(AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max}}),
        robot_model=filter_robot,
        constraint_robot_model=build_constraint_robot_model(scene, link_filter=ARM_LINKS))
    names = {b: (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
             for b in range(scene.model.nbody)}

    # 프레임별로 (관측, 참값 프레임) 을 미리 만들어 둔다 — 자세만 나중에 갈아 끼운다.
    built = {}
    for i in range(args.first + args.frames):
        pose_scene(scene, run.steps[i])
        built[i] = {}
        for cam in CAMERAS:
            o, fr = camera_observation(scene, cam, filter_robot, timestamp=float(i))
            built[i][cam] = (o, fr)

    print("=" * 96)
    print(f"Step 7 — 상태 지연의 해 ({args.records}, 프레임 {args.first}~"
          f"{args.first + args.frames - 1}, 기록 스텝 = {STEP_MS:.0f} ms)")
    print(f"설정: timing.max_state_age_sec = "
          f"{ag.config.timing.max_state_age_sec * 1000:.0f} ms")
    print("=" * 96)
    print(f"{'지연':>7} {'로봇 누수 점':>13} {'실물체 점':>11} {'카메라 이동 mm':>14}  "
          f"{'설정 안?':>8}")

    rows = []
    for k in args.lags:
        leak = over = 0
        shifts = []
        for i in range(args.first, args.first + args.frames):
            q_lag = built[i - k][CAMERAS[0]][0].robot_state
            for cam in CAMERAS:
                o, fr = built[i][cam]
                bid = np.asarray(fr.body_ids)
                is_rb = np.vectorize(
                    lambda b: bool(b >= 0 and is_robot_body(names.get(int(b), ""))))(bid)
                is_obj = (bid >= 0) & ~is_rb

                o_lag = dataclasses.replace(o, robot_state=np.asarray(q_lag, np.float64))
                T_true = np.asarray(o.resolve_T_base_cam(filter_robot), np.float64)
                T_lag = np.asarray(o_lag.resolve_T_base_cam(filter_robot), np.float64)
                shifts.append(float(np.linalg.norm(T_true[:3, 3] - T_lag[:3, 3])) * 1000.0)

                # **실제로 클라우드에 남는 점**을 센다. 파이프라인이 쓰는 그 경로 그대로.
                res = process_observation(o_lag, ag.config, robot_model=filter_robot)
                cl = res.cloud
                if cl.is_empty or cl.uv is None:
                    continue
                uvp = np.asarray(cl.uv, np.int32)
                ok = ((uvp[:, 0] >= 0) & (uvp[:, 1] >= 0)
                      & (uvp[:, 1] < bid.shape[0]) & (uvp[:, 0] < bid.shape[1]))
                b = bid[uvp[ok, 1], uvp[ok, 0]]
                rb = np.array([bool(x >= 0 and is_robot_body(names.get(int(x), ""))) for x in b])
                leak += int(rb.sum())
                over += int(((b >= 0) & ~rb).sum())
        ms = k * STEP_MS
        inside = ms <= ag.config.timing.max_state_age_sec * 1000
        print(f"{ms:>5.0f}ms {leak:>14} {over:>10} {np.mean(shifts):>13.1f}  "
              f"{'예' if inside else '아니오':>8}")
        rows.append(dict(ms=ms, leak=leak, over=over, shift=float(np.mean(shifts)),
                         shift_max=float(np.max(shifts)), inside=inside))
    scene.close()
    _report(rows, ag.config.timing.max_state_age_sec * 1000)
    _figure(rows, ag.config.timing.max_state_age_sec * 1000, args)


def _report(rows, limit_ms):
    print("\n" + "=" * 96)
    base = rows[0]["leak"] if rows else 0
    first_bad = next((r for r in rows if r["leak"] > max(base, 0) and r["ms"] > 0), None)
    print(f"지연 0 에서의 로봇 누수 (기준선) : {base}")
    if first_bad:
        print(f"누수가 시작되는 지연            : {first_bad['ms']:.0f} ms "
              f"(누수 {first_bad['leak']} 픽셀, 카메라 {first_bad['shift']:.1f} mm 이동)")
    inside = [r for r in rows if r["inside"] and r["ms"] > 0]
    if inside:
        w = max(inside, key=lambda r: r["leak"])
        print(f"\n설정 한계({limit_ms:.0f} ms) **안**에서 일어날 수 있는 최악:")
        print(f"  로봇 누수 {w['leak']} 픽셀,  카메라 이동 평균 {w['shift']:.1f} mm "
              f"/ 최대 {w['shift_max']:.1f} mm  (지연 {w['ms']:.0f} ms)")
    print("\n>>> 판정: " + (
        f"설정 한계 안에서 이미 로봇 누수가 {max(r['leak'] for r in rows if r['inside'])} 픽셀 "
        f"생긴다 — 한계가 느슨하다"
        if any(r["leak"] > base for r in rows if r["inside"] and r["ms"] > 0) else
        "설정 한계 안에서는 누수가 늘지 않는다"))


def _figure(rows, limit_ms, args):
    import matplotlib.pyplot as plt
    ms = [r["ms"] for r in rows]
    leak = [r["leak"] for r in rows]
    over = [r["over"] for r in rows]
    shift = [r["shift"] for r in rows]
    smax = [r["shift_max"] for r in rows]

    fig, axs = plt.subplots(1, 3, figsize=(19.5, 5.5))

    a = axs[0]
    a.axvspan(0, limit_ms, color="tab:green", alpha=0.12)
    a.text(limit_ms / 2, max(max(leak), 1) * 0.92, "설정이 허용하는 범위",
           ha="center", fontsize=10.5, color="#2e7d32", fontweight="bold")
    a.plot(ms, leak, "o-", color="crimson", lw=2.4, ms=7, label="클라우드에 들어간 로봇 점")
    a.axvline(limit_ms, color="crimson", ls="--", lw=2)
    a.text(limit_ms + 2, max(max(leak), 1) * 0.5, f"한계 {limit_ms:.0f} ms",
           color="crimson", fontsize=10)
    a.axvline(50, color="0.4", ls=":", lw=1.8)
    a.text(52, max(max(leak), 1) * 0.25, "모듈이 든 예 50 ms", color="0.35", fontsize=9.5)
    a.set_xlabel("상태 지연 [ms]"); a.set_ylabel("로봇 점 수")
    a.set_title("(a) 팔 자신의 점이 장애물로 새는 양\n'그리퍼에 붙은 유령 장애물' 의 씨앗",
                fontsize=11)
    a.legend(fontsize=9.5); a.grid(alpha=0.3)

    a = axs[1]
    a.axvspan(0, limit_ms, color="tab:green", alpha=0.12)
    a.plot(ms, shift, "o-", color="tab:blue", lw=2.2, ms=6, label="평균")
    a.plot(ms, smax, "s--", color="tab:blue", lw=1.6, ms=5, alpha=0.65, label="최대")
    a.axhline(25, color="0.4", ls=":", lw=1.8)
    a.text(2, 27, "모듈이 든 2.5 cm", fontsize=9.5, color="0.35")
    a.axvline(limit_ms, color="crimson", ls="--", lw=2)
    a.set_xlabel("상태 지연 [ms]"); a.set_ylabel("카메라 위치 오차 [mm]")
    a.set_title("(b) 지연된 자세로 푼 카메라가 얼마나 어긋나는가", fontsize=11)
    a.legend(fontsize=9.5); a.grid(alpha=0.3)

    a = axs[2]; a.axis("off")
    rowsT = [["지연", "로봇 누수 점", "실물체 점", "카메라 이동", "설정 안?"]]
    for r in rows:
        rowsT.append([f"{r['ms']:.0f} ms", f"{r['leak']:,}", f"{r['over']:,}",
                      f"{r['shift']:.1f} mm", "예" if r["inside"] else "아니오"])
    t = a.table(cellText=rowsT[1:], colLabels=rowsT[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1.0, 1.75)
    for j in range(5):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    for ri, r in enumerate(rows, start=1):
        if r["inside"] and r["leak"] > rows[0]["leak"]:
            for j in range(5):
                t[(ri, j)].set_facecolor("#f7d6d6")
    a.set_title("(c) 분홍 = 설정이 허용하는데 이미 새는 구간", fontsize=11.5, y=0.9)

    fig.suptitle("상태 지연은 얼마부터 해로운가 — 설정 한계는 그 지점에 맞는가", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    o = OUT / f"step7-state-lag-{args.records[-4:]}.png"
    fig.savefig(o, dpi=110, bbox_inches="tight")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
