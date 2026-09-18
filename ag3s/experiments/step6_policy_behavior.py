"""정책이 실제로 무엇을 집었는가 — 기록된 롤아웃에서 물체와 손의 궤적을 잰다.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.step6_policy_behavior

## 왜 재는가

F11 을 "grounding 이 사과 대신 바구니를 target 으로 낸다" 는 **grounding 의 결함 후보**로
적어 두었다. 그런데 사용자 관찰(2026-09-14): **정책 자체가 사과를 집는 도중 바구니를 집고
있고, 작업이 끝난 뒤에도 그것을 인지하지 못한 채 다음 동작으로 넘어간다.**

그렇다면 grounding 은 **정책이 보는 것을 정직하게 보고한 것**이고 결함이 아니다. 결함은
우리가 검토 중인 파이프라인보다 **상류**(VLA 체크포인트)에 있다.

이 구분은 추측으로 두면 안 된다. 기록이 답을 갖고 있다 — **정책이 바구니를 집었다면 기록에서
바구니가 움직였을 것**이다. 물체들은 free joint (`crate_free`, `apple_free`, …) 라 위치가
매 스텝 기록되어 있다.

재는 것:

  1. **물체별 이동 거리** — 무엇이 움직였나. 사과인가 바구니인가.
  2. **손끝과 물체의 거리** — 어느 손이 무엇에 붙어 다녔나.
  3. **grounding 이 낸 target 과 대조** — 정직한 보고인가, 어긋난 보고인가.

프롬프트는 `put the apple in the basket` 이므로 **바구니는 목적지**다. 목적지를 보는 것 자체는
자연스러울 수 있다 — 갈리는 것은 **집은 것이 무엇인가** 다.
"""

import argparse
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")
#: 손끝 body. 좌우 두 손가락의 중점을 그 손의 위치로 본다.
FINGERS = {"left": ("ee_finger_l1", "ee_finger_l2"),
           "right": ("ee_finger_r1", "ee_finger_r2")}


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
    import matplotlib.pyplot as plt
    import mujoco

    from benchmark.ag3s.experiments.mujoco_source import is_robot_body
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--frames", type=int, default=0, help="0 이면 전부")
    args = ap.parse_args()

    run = load_run(args.records, limit=args.frames or None)
    scene = replay_scene(run)
    n = len(run.steps)

    # 추적할 물체 — 로봇도 테이블도 아닌 것
    bodies = {}
    for b in range(scene.model.nbody):
        nm = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not nm or is_robot_body(nm):
            continue
        if any(k in nm for k in ("table", "shelf", "floor", "world", "ground", "com_target")):
            continue
        bodies[nm] = b
    fingers = {}
    for hand, (a, c) in FINGERS.items():
        ia = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, a)
        ic = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, c)
        if ia >= 0 and ic >= 0:
            fingers[hand] = (ia, ic)

    print("=" * 84)
    print(f"정책 거동 — {args.records}, 프롬프트: {run.prompt!r}, {n} 스텝")
    print("=" * 84)

    obj_xyz = {k: np.zeros((n, 3)) for k in bodies}
    hand_xyz = {k: np.zeros((n, 3)) for k in fingers}
    t_step = np.zeros(n, int)
    for i, st in enumerate(run.steps):
        pose_scene(scene, st)
        t_step[i] = int(st.t_step)
        for nm, b in bodies.items():
            obj_xyz[nm][i] = scene.data.xpos[b]
        for hand, (ia, ic) in fingers.items():
            hand_xyz[hand][i] = 0.5 * (scene.data.xpos[ia] + scene.data.xpos[ic])

    # 1. 무엇이 움직였나
    print("\n1. 물체별 이동 — 정책이 무엇을 집었는지가 여기 나온다")
    print(f"   {'물체':<10} {'총 경로길이':>12} {'시작→끝 직선':>14} {'최대 높이변화':>14}")
    moved = {}
    for nm, p in sorted(obj_xyz.items()):
        path = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        net = float(np.linalg.norm(p[-1] - p[0]))
        dz = float(p[:, 2].max() - p[:, 2].min())
        moved[nm] = dict(path=path, net=net, dz=dz)
        flag = "  <<< 움직였다" if net > 0.02 or dz > 0.02 else ""
        print(f"   {nm:<10} {path*1000:>10.1f} mm {net*1000:>12.1f} mm {dz*1000:>12.1f} mm{flag}")

    # 2. 손과 물체
    print("\n2. 각 손이 어느 물체에 가장 가까웠나 (스텝별 최근접)")
    for hand, hp in hand_xyz.items():
        best = []
        for i in range(n):
            d = {nm: float(np.linalg.norm(hp[i] - obj_xyz[nm][i])) for nm in bodies}
            k = min(d, key=d.get)
            best.append((k, d[k]))
        import collections
        cnt = collections.Counter(k for k, _ in best)
        near = sum(1 for k, dv in best if dv < 0.10)
        print(f"   {hand:<6} 최근접 물체 분포 {dict(cnt.most_common())}   "
              f"10 cm 이내였던 스텝 {near}/{n}")

    scene.close()
    _figure(t_step, obj_xyz, hand_xyz, moved, run.prompt, args.records)


def _figure(t_step, obj_xyz, hand_xyz, moved, prompt, records):
    import matplotlib.pyplot as plt
    n = len(t_step)
    idx = np.arange(n)

    fig, axs = plt.subplots(2, 3, figsize=(19.5, 10.4))
    ax = axs.ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # (a) 실제 씬 — 위에서 본 궤적
    a = ax[0]
    for j, (nm, p) in enumerate(sorted(obj_xyz.items())):
        a.plot(p[:, 0], p[:, 1], "-", color=colors[j], lw=1.6, label=nm)
        a.plot(p[0, 0], p[0, 1], "o", color=colors[j], ms=7, mec="k")
        a.plot(p[-1, 0], p[-1, 1], "s", color=colors[j], ms=8, mec="k")
    for hand, hp, ls in (("left", hand_xyz.get("left"), "--"),
                         ("right", hand_xyz.get("right"), ":")):
        if hp is None:
            continue
        a.plot(hp[:, 0], hp[:, 1], ls, color="k", lw=1.4, label=f"{hand} 손끝")
    a.set_title("(a) 실제 씬 — 위에서 본 궤적\n○ 시작  □ 끝", fontsize=11)
    a.set_xlabel("x [m] — 앞쪽 →"); a.set_ylabel("y [m] — 왼쪽 ↑")
    a.set_aspect("equal"); a.legend(fontsize=8, ncol=2); a.grid(alpha=0.3)

    # (b) 높이 — 집어 올렸는지가 여기 보인다
    a = ax[1]
    for j, (nm, p) in enumerate(sorted(obj_xyz.items())):
        a.plot(idx, p[:, 2] * 1000, "-", color=colors[j], lw=1.8, label=nm)
    a.axhline(823, color="k", ls=":", lw=1.2)
    a.text(0.5, 826, "테이블 상판 823 mm", fontsize=9)
    a.set_title("(b) 물체 높이 — 들어 올려진 것이 있는가", fontsize=11)
    a.set_xlabel("스텝"); a.set_ylabel("z [mm]"); a.legend(fontsize=8, ncol=2); a.grid(alpha=0.3)

    # (c) 손끝-물체 거리
    a = ax[2]
    hp = hand_xyz.get("left")
    if hp is not None:
        for j, (nm, p) in enumerate(sorted(obj_xyz.items())):
            a.plot(idx, np.linalg.norm(hp - p, axis=1) * 1000, "-", color=colors[j],
                   lw=1.6, label=nm)
    a.axhline(50, color="k", ls=":", lw=1.2)
    a.text(0.5, 54, "50 mm", fontsize=9)
    a.set_title("(c) 왼손끝에서 각 물체까지 거리", fontsize=11)
    a.set_xlabel("스텝"); a.set_ylabel("거리 [mm]"); a.set_ylim(0, 700)
    a.legend(fontsize=8, ncol=2); a.grid(alpha=0.3)

    a = ax[3]
    hp = hand_xyz.get("right")
    if hp is not None:
        for j, (nm, p) in enumerate(sorted(obj_xyz.items())):
            a.plot(idx, np.linalg.norm(hp - p, axis=1) * 1000, "-", color=colors[j],
                   lw=1.6, label=nm)
    a.axhline(50, color="k", ls=":", lw=1.2)
    a.set_title("(d) 오른손끝에서 각 물체까지 거리", fontsize=11)
    a.set_xlabel("스텝"); a.set_ylabel("거리 [mm]"); a.set_ylim(0, 700)
    a.legend(fontsize=8, ncol=2); a.grid(alpha=0.3)

    # (e) 이동량 막대
    a = ax[4]
    names = sorted(moved)
    net = [moved[k]["net"] * 1000 for k in names]
    dz = [moved[k]["dz"] * 1000 for k in names]
    w = 0.38
    xs = np.arange(len(names))
    a.bar(xs - w / 2, net, w, label="시작→끝 직선", color="tab:blue")
    a.bar(xs + w / 2, dz, w, label="최대 높이변화", color="tab:orange")
    a.set_xticks(xs); a.set_xticklabels(names, rotation=20, ha="right")
    a.axhline(20, color="k", ls=":", lw=1.2)
    a.text(-0.4, 22, "20 mm — 움직였다고 볼 선", fontsize=9)
    a.set_title("(e) 물체별 이동량 — 정책이 무엇을 옮겼나", fontsize=11)
    a.set_ylabel("mm"); a.legend(fontsize=9); a.grid(alpha=0.3, axis="y")

    # (f) 표
    a = ax[5]; a.axis("off")
    rows = [["물체", "경로길이", "시작→끝", "높이변화"]]
    for k in names:
        rows.append([k, f"{moved[k]['path']*1000:.0f} mm",
                     f"{moved[k]['net']*1000:.0f} mm", f"{moved[k]['dz']*1000:.0f} mm"])
    t = a.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1.0, 1.9)
    for j in range(4):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    a.set_title(f"(f) 요약 — 프롬프트: {prompt!r}", fontsize=11, y=0.93)

    fig.suptitle(f"정책이 실제로 무엇을 집었는가 — {records}, 기록된 {n} 스텝 전체",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUT.mkdir(parents=True, exist_ok=True)
    o = OUT / f"step6-policy-behavior-{records[-4:]}.png"
    fig.savefig(o, dpi=105, bbox_inches="tight")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
