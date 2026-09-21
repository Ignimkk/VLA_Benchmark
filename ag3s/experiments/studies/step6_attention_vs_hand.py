"""attention 이 가리키는 것과 손이 잡고 있는 것이 어긋나는 지점.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.studies.step6_attention_vs_hand

## 무엇을 보는가

`step6_policy_behavior.py` 로 `run_0004` 가 **사과를 성공적으로 집어 바구니에 넣은 롤아웃**임을
확인했다 (사과만 302 mm 이동, 186 mm 상승; 바구니는 4 mm). 정책 실패가 아니다.

그런데 grounding 은 프레임 7 부터 target 을 **바구니**라고 낸다. 손은 그때 사과를 잡고 있다.
이 그림이 그 어긋남을 시간축에 놓는다.

## 왜 중요한가

AG3S 는 **"target = 지금 조작하는 물체"** 를 전제한다.

* `ClearancePolicy` 의 접촉 권한이 target 을 기준으로 부여된다 (E1). target 이 바구니면 손끝은
  *바구니* 를 만져도 된다고 허가받는데, 실제로 만지는 것은 *사과* 다. 그러면 **쥐고 있는 사과가
  완전 여유거리를 요구하는 장애물**이 된다 — E3(attached object 가 optimizer 에 도달하지 않음)와
  같은 구멍이 다른 방향에서 열린다.
* 미세 ESDF 창의 중심도 target 을 따라간다. 참 거리 대조에서 "프레임 7~14 에 최악 구가 창 안에
  들어온 것은 부분적으로 우연" 이라고 적었는데, 그 우연의 정체가 이것이다.

정책의 attention 이 목적지를 보는 것 자체는 자연스러울 수 있다 — **틀린 것은 그 출력을
"조작 대상" 으로 읽는 우리 쪽 전제**일 수 있다. 판정은 사용자와 함께 한다.
"""

import argparse
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
    from benchmark.ag3s.experiments.common import figstyle
    figstyle.use_korean()


def main() -> None:
    _style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import mujoco

    from benchmark.ag3s.experiments.sources.mujoco_source import is_robot_body
    from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--dump", default="/tmp/rollout_frames.npz")
    ap.add_argument("--steps", type=int, default=22)
    args = ap.parse_args()

    run = load_run(args.records)
    scene = replay_scene(run)
    dump = np.load(args.dump)
    nd = int(dump["n_frames"])
    n = min(args.steps, len(run.steps))

    bod = {}
    for b in range(scene.model.nbody):
        nm = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if nm and not is_robot_body(nm) and not any(
                k in nm for k in ("table", "shelf", "floor", "world", "ground",
                                  "com_target", "_ee_target", "office")):
            bod[nm] = b
    fl = [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, x)
          for x in ("ee_finger_l1", "ee_finger_l2")]

    z_apple = np.zeros(n); d_ap = np.zeros(n); d_cr = np.zeros(n)
    tgt = []
    hand_p = np.zeros((n, 3)); ap_p = np.zeros((n, 3))
    for i in range(n):
        pose_scene(scene, run.steps[i])
        hand = 0.5 * (scene.data.xpos[fl[0]] + scene.data.xpos[fl[1]])
        a, c = scene.data.xpos[bod["apple"]], scene.data.xpos[bod["crate"]]
        hand_p[i] = hand; ap_p[i] = a
        z_apple[i] = a[2] * 1000
        d_ap[i] = np.linalg.norm(hand - a) * 1000
        d_cr[i] = np.linalg.norm(hand - c) * 1000
        if i < nd:
            cen = np.asarray(dump[f"centroid_{i}"], float)
            best = min(bod, key=lambda k: np.linalg.norm(cen - scene.data.xpos[bod[k]]))
            tgt.append(best)
        else:
            tgt.append(None)
    scene.close()

    fig, axs = plt.subplots(1, 3, figsize=(19.5, 5.6))

    # (a) 실제 씬 — 사과와 손의 궤적, grounding target 으로 색칠
    a = axs[0]
    col = {"apple": "tab:red", "crate": "tab:blue", None: "0.7"}
    a.plot(ap_p[:, 0], ap_p[:, 2] * 1, "-", color="0.6", lw=1.2)
    for i in range(n):
        a.plot(ap_p[i, 0], ap_p[i, 2], "o", color=col.get(tgt[i], "0.5"), ms=7, mec="k", mew=0.4)
    a.plot(hand_p[:, 0], hand_p[:, 2], "k--", lw=1.3, label="왼손끝")
    a.axhline(0.823, color="k", ls=":", lw=1.2)
    a.text(0.42, 0.828, "테이블 상판", fontsize=9)
    a.set_title("(a) 실제 씬 — 옆에서 본 사과 궤적\n점 색 = 그때 grounding 이 낸 target",
                fontsize=11)
    a.set_xlabel("x [m] — 앞쪽 →"); a.set_ylabel("z [m] — 위 ↑")
    a.legend(handles=[Patch(color="tab:red", label="target = apple"),
                      Patch(color="tab:blue", label="target = crate"),
                      Patch(color="0.7", label="측정 안 함")], fontsize=9, loc="upper left")
    a.grid(alpha=0.3)

    # (b) 시간축
    a = axs[1]
    idx = np.arange(n)
    for i in range(n):
        if tgt[i] is not None:
            a.axvspan(i - 0.5, i + 0.5, color=col[tgt[i]], alpha=0.13)
    a.plot(idx, d_ap, "o-", color="tab:red", lw=2, ms=5, label="왼손끝 — 사과")
    a.plot(idx, d_cr, "s-", color="tab:blue", lw=2, ms=5, label="왼손끝 — 바구니")
    a.axhline(50, color="k", ls=":", lw=1.2)
    a.text(0.2, 58, "50 mm — 잡고 있다고 볼 선", fontsize=9)
    a2 = a.twinx()
    a2.plot(idx, z_apple, "-", color="tab:green", lw=2.2, label="사과 높이")
    a2.set_ylabel("사과 높이 [mm]", color="tab:green")
    a2.tick_params(axis="y", colors="tab:green")
    a.axvline(6.5, color="crimson", lw=2)
    a.text(6.7, 620, "여기서 target 이\n사과 → 바구니", color="crimson", fontsize=10,
           fontweight="bold")
    a.set_title("(b) 손은 사과를 잡는데 target 은 바구니로 넘어간다\n"
                "배경색 = 그때의 grounding target", fontsize=11)
    a.set_xlabel("스텝"); a.set_ylabel("거리 [mm]"); a.set_ylim(0, 700)
    a.legend(fontsize=9, loc="upper right"); a.grid(alpha=0.3)

    # (c) 표
    a = axs[2]; a.axis("off")
    rows = [["스텝", "사과높이", "손-사과", "손-바구니", "target"]]
    for i in range(0, n, 2):
        rows.append([f"{i}", f"{z_apple[i]:.0f}", f"{d_ap[i]:.0f}", f"{d_cr[i]:.0f}",
                     tgt[i] or "-"])
    t = a.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1.0, 1.32)
    for j in range(5):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    for r, i in enumerate(range(0, n, 2), start=1):
        if tgt[i] == "crate":
            for j in range(5):
                t[(r, j)].set_facecolor("#dbe7f5")
    a.set_title("(c) 파란 행 = target 이 바구니인데 손은 사과를 쥐고 있는 구간",
                fontsize=11, y=0.97)

    fig.suptitle("attention 이 가리키는 것 대 손이 잡고 있는 것 — run_0004 "
                 "(사과를 성공적으로 옮긴 롤아웃)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    o = OUT / f"step6-attention-vs-hand-{args.records[-4:]}.png"
    fig.savefig(o, dpi=110, bbox_inches="tight")
    print(f"wrote {o}")

    grasp = [i for i in range(n) if d_ap[i] < 55]
    crate_t = [i for i in range(n) if tgt[i] == "crate"]
    both = sorted(set(grasp) & set(crate_t))
    print(f"\n손이 사과를 쥔 스텝 (50 mm 대) : {grasp}")
    print(f"target 이 바구니인 스텝          : {crate_t}")
    print(f"**둘이 겹치는 스텝**             : {both}   ({len(both)} 개)")


if __name__ == "__main__":
    main()
