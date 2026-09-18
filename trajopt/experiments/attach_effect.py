"""접촉 권한이 조작 대상에 붙었을 때 무엇이 달라지는가 — `--attach` 껐다 켠 비교.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.trajopt.experiments.attach_effect

`grasp_damage.py --out-npz` 가 남긴 스텝별 잔차를 두 모드에서 읽어 겹쳐 그린다. 재는 것은 하나다:
**권한이 허용된 손끝에만 붙고 나머지에는 안 붙는가.**
"""

import argparse
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")
AUTHORIZED = ("ee_finger_l1", "ee_finger_l2")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments import figstyle
    figstyle.use_korean()


def _load(path):
    b = np.load(path, allow_pickle=True)
    n = int(b["n"])
    return dict(
        n=n, links=[str(x) for x in b["links"]],
        grasped=np.asarray(b["grasped"], bool), in_field=np.asarray(b["in_field"], bool),
        resid=[np.asarray(b[f"resid_{i}"], float) for i in range(n)],
        is_held=[np.asarray(b[f"is_held_{i}"], bool) for i in range(n)],
        target=[str(x) for x in b["target"]],
    )


def _worst_per_link(d, link_pick, grasped_only=False):
    """스텝별로 (해당 링크들 중, 쥔 물체가 최근접이면서 위반인 구의) 최악 잔차.

    `grasped_only` 는 요약용이다. 파지 전 스텝은 **양쪽 모드에서 attach 가 없으므로 정의상 같고**,
    전 구간의 최악을 잡으면 그 값이 차이를 덮어 버린다 — 처음 그림을 그렸을 때 실제로 +0.0 이
    나왔다. 효과는 무언가 쥐고 있는 구간에만 존재한다. (덮였다는 사실 자체는 "접근 단계는
    바뀌지 않는다" 는 설계 성질의 확인이기도 하다.)
    """
    out = np.full(d["n"], np.nan)
    mask = np.asarray([ln in link_pick for ln in d["links"]], bool)
    for i in range(d["n"]):
        if grasped_only and not d["grasped"][i]:
            continue
        sel = mask & d["is_held"][i] & (d["resid"][i] < 0)
        if sel.any():
            out[i] = d["resid"][i][sel].min() * 1000
    return out


def main() -> None:
    _style()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", nargs="+", default=["0004", "0005"])
    ap.add_argument("--prefix", default="/tmp/gd")
    args = ap.parse_args()

    fig, axs = plt.subplots(2, len(args.records) + 1,
                            figsize=(6.6 * (len(args.records) + 1), 10.2))
    if len(axs.shape) == 1:
        axs = axs.reshape(2, -1)

    summary = []
    for col, rec in enumerate(args.records):
        off = _load(f"{args.prefix}_{rec}_off.npz")
        on = _load(f"{args.prefix}_{rec}_on.npz")
        idx = np.arange(off["n"])
        others = [ln for ln in set(off["links"]) if ln not in AUTHORIZED]

        # 위: 허용된 손끝
        a = axs[0, col]
        for i in np.flatnonzero(off["grasped"]):
            a.axvspan(i - 0.5, i + 0.5, color="tab:orange", alpha=0.15)
        a.plot(idx, _worst_per_link(off, AUTHORIZED, True), "o-", color="crimson", lw=2.2, ms=7,
               label="attach 끔 — 권한 없음")
        a.plot(idx, _worst_per_link(on, AUTHORIZED, True), "s-", color="tab:green", lw=2.2, ms=7,
               label="attach 켬 — 권한 있음")
        a.axhline(0, color="k", lw=1.2)
        a.set_title(f"(위) run_{rec} — 쥔 손가락 {', '.join(AUTHORIZED)}\n"
                    "주황 = 쥐고 있는 구간", fontsize=11)
        a.set_xlabel("스텝"); a.set_ylabel("최악 잔차 [mm]")
        a.legend(fontsize=9); a.grid(alpha=0.3)

        # 아래: 권한 없는 링크
        a = axs[1, col]
        for i in np.flatnonzero(off["grasped"]):
            a.axvspan(i - 0.5, i + 0.5, color="tab:orange", alpha=0.15)
        wo = _worst_per_link(off, others, grasped_only=True)
        wn = _worst_per_link(on, others, grasped_only=True)
        a.plot(idx, wo, "o-", color="crimson", lw=2.2, ms=6, label="attach 끔")
        a.plot(idx, wn, "s--", color="tab:green", lw=2.2, ms=6, label="attach 켬")
        a.axhline(0, color="k", lw=1.2)
        same = np.allclose(np.nan_to_num(wo, nan=0), np.nan_to_num(wn, nan=0), atol=1e-6)
        a.set_title(f"(아래) run_{rec} — 권한 없는 링크 (손목·전완)\n"
                    + ("두 선이 완전히 겹친다 — 완화가 새지 않았다" if same
                       else "두 선이 다르다 — 완화가 샜다"), fontsize=11)
        a.set_xlabel("스텝"); a.set_ylabel("최악 잔차 [mm]")
        a.legend(fontsize=9); a.grid(alpha=0.3)

        for links, label in ((AUTHORIZED, "쥔 손가락"), (tuple(others), "권한 없는 링크")):
            o = _worst_per_link(off, links, grasped_only=True)
            n_ = _worst_per_link(on, links, grasped_only=True)
            if np.isfinite(o).any():
                summary.append([f"run_{rec}", label, f"{np.nanmin(o):.1f}", f"{np.nanmin(n_):.1f}",
                                f"{np.nanmin(n_) - np.nanmin(o):+.1f}"])

    for r in range(2):
        a = axs[r, -1]; a.axis("off")
    a = axs[0, -1]
    rows = [["기록", "링크", "끔", "켬", "차이"]] + summary  # 파지 구간 한정
    t = a.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.0, 2.1)
    for j in range(5):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    for r, row in enumerate(summary, start=1):
        t[(r, 0)].set_facecolor("#eaf6ea" if row[1] == "쥔 손가락" else "#f2f2f2")
    a.set_title("최악 잔차 [mm] — 권한은 손끝에만 붙어야 한다", fontsize=11.5, y=0.78)

    a = axs[1, -1]
    a.text(0.02, 0.92, "무엇을 보는 그림인가", fontsize=13, fontweight="bold",
           transform=a.transAxes)
    a.text(0.02, 0.06,
           "접촉 권한(여유거리 완화)이 **주목 대상**이 아니라\n"
           "**조작 대상**에 붙도록 바꾼 뒤의 차이다 (F11, 갈래 3).\n\n"
           "· 위 줄 — 사과를 쥔 손끝. 권한을 받아 요구가\n"
           "  여유거리 50 mm 만큼 줄어야 한다.\n\n"
           "· 아래 줄 — 손목·전완. allowlist 밖이므로\n"
           "  한 톨도 줄면 안 된다. 두 선이 겹쳐야 옳다.\n\n"
           "남는 음수는 관통이고 결함이 아니다. 권한 0 은\n"
           "'만져도 된다'이지 '뚫어도 된다'가 아니다.\n"
           "그 부분은 쥔 물체를 로봇 쪽 기하로 옮겨야\n"
           "사라진다 (E3, Step 10).".replace("**", ""),
           fontsize=11, transform=a.transAxes, va="bottom", linespacing=1.6)

    fig.suptitle("접촉 권한을 조작 대상에 붙였을 때 — attach 껐다 켠 비교", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    o = OUT / "step6-attach-effect.png"
    fig.savefig(o, dpi=105, bbox_inches="tight")
    print(f"wrote {o}")
    for row in summary:
        print("  ", row)


if __name__ == "__main__":
    main()
