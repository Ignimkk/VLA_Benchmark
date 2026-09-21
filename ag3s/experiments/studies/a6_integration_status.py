"""통합 경로가 실제로 무엇을 돌리는가 — 고치기 전과 후 (2026-09-18).

`safe_replay.py` 는 `serve_safe.py` 와 **같은 `SafePolicy` 경로**다. 여기서 안 도는 모듈은
실기에서도 안 돈다. 이 스크립트는 `safe_replay --out-json` 이 남긴 청크별 모듈 상태를 읽어
그림으로 만든다 — **코드를 다시 돌리지 않는다.**

전(`--attention` 없음, attached 슬롯 미예약)과 후(넷 다 배선 + 슬롯 예약)의 JSON 을 둘 다
주면 나란히 그린다.

실행:
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -m \
        benchmark.ag3s.experiments.studies.a6_integration_status \
        --before before.json --after after.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures/a6-integration-status.png")
#: 그림에 그릴 모듈과, 청크 한 줄에서 그 모듈이 "돌았다" 를 어떻게 읽는가.
MODULES = [
    ("AG3S 가 돌았다", lambda r: r["ag3s_status"] != "no_geometry" and r.get("scene_ok", True)),
    ("target 있음", lambda r: bool(r["modules"]["target"])),
    ("잠금 걸림", lambda r: r["modules"]["latch"] != "SEARCHING"),
    ("쥔 물체 질의점", lambda r: r["modules"]["held_points"] > 0),
    ("쥔 물체 파내기 (A2)", lambda r: r["modules"]["carved"] > 0),
    ("목적지 라벨 (F18·A3)", lambda r: bool(r["modules"]["destination"])),
    ("기하 인증", lambda r: bool(r["geometry_certified"])),
    ("safe", lambda r: bool(r["safe"])),
]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments.common import figstyle
    figstyle.use_korean()
    return figstyle


def _grid(rows):
    """`(모듈 수, 청크 수)` bool. 모듈 상태가 없는 옛 기록은 전부 거짓으로 읽는다."""
    out = np.zeros((len(MODULES), len(rows)), bool)
    for c, r in enumerate(rows):
        if "modules" not in r:
            r = dict(r, modules={"target": False, "latch": "SEARCHING", "held_points": 0,
                                 "carved": 0, "destination": False})
        for m, (_, fn) in enumerate(MODULES):
            try:
                out[m, c] = bool(fn(r))
            except Exception:       # noqa: BLE001 — 옛 기록에 없는 키는 "안 돌았다" 다
                out[m, c] = False
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--before", required=True, help="고치기 전 safe_replay JSON")
    ap.add_argument("--after", required=True, help="고친 뒤 safe_replay JSON")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    before = json.loads(pathlib.Path(args.before).read_text())
    after = json.loads(pathlib.Path(args.after).read_text())
    gb, ga = _grid(before["frames"]), _grid(after["frames"])

    fs = _style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#fdeceb", "#cfe8da"])
    fig = plt.figure(figsize=(15.6, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1, 1],
                          wspace=0.24, hspace=0.42)

    for row, (g, title) in enumerate(((gb, "① 고치기 전 — 배선 없음"),
                                      (ga, "② 고친 뒤 — 넷 다 배선 + attached 슬롯 예약"))):
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(g, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(MODULES)))
        ax.set_yticklabels([m for m, _ in MODULES], fontsize=8.4)
        ax.set_xticks(range(0, g.shape[1], 2))
        ax.set_xticklabels([str(k + 1) for k in range(0, g.shape[1], 2)], fontsize=8)
        ax.set_xlabel("청크")
        ax.set_title(f"{title}  (초록 = 돌았다)", fontsize=10.6, color=fs.INK)
        for m in range(len(MODULES) + 1):
            ax.axhline(m - 0.5, color="#ffffff", lw=1.4)

    # --- 표 ------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.axis("off")
    ax3.set_title("③ 청크 중 몇 개에서 돌았나", fontsize=10.6, color=fs.INK, pad=16)
    nb, na = gb.shape[1], ga.shape[1]
    trows = [["모듈", f"전 ({nb})", f"후 ({na})"]]
    for m, (name, _) in enumerate(MODULES):
        trows.append([name, f"{int(gb[m].sum())}", f"{int(ga[m].sum())}"])
    t = ax3.table(cellText=trows, colWidths=[0.52, 0.24, 0.24], loc="upper center",
                  cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(8.6); t.scale(1, 1.7)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if r == 0:
            cell.set_facecolor("#ecebe7"); cell.set_text_props(weight="bold")

    fig.suptitle("통합 경로(SafePolicy)가 실제로 돌린 모듈 — 고치기 전과 후",
                 fontsize=13.0, color=fs.INK)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {out}")
    for m, (name, _) in enumerate(MODULES):
        print(f"  {name:>22}  {int(gb[m].sum()):>3} -> {int(ga[m].sum()):>3}")


if __name__ == "__main__":
    main()
