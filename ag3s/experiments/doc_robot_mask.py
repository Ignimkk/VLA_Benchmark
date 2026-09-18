"""문서용 시각화 — 로봇 마스크(self-filter) 단계.

    PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.doc_robot_mask \\
        --frame /tmp/frame10.npz --field /tmp/field10.npz --field-raw /tmp/field10_raw.npz

`--field-raw` 는 같은 프레임을 **마스크 없이** 적분한 대조군이다
(`curobo.build_field --raw`). 두 필드를 같은 로봇 구로 질의하면 "마스크를 빼면 무슨 일이
일어나는가" 가 숫자로 나온다 — 발견 C1 이 그것이다.

규칙 A: 실제 씬 · 3인칭 · 그래프 · 표.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")
IDS = ("head", "left_wrist", "right_wrist")
LABEL = ("head (zed_left)", "왼손목", "오른손목")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments import figstyle
    figstyle.use_korean()
    return figstyle


def _table(ax, rows, fs, title, widths=(0.36, 0.64), fontsize=8.4):
    ax.axis("off")
    t = ax.table(cellText=rows, colWidths=list(widths), loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(fontsize); t.scale(1, 1.34)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if rows[r][1] == "" and rows[r][0]:
            cell.set_facecolor("#ecebe7"); cell.set_text_props(weight="bold")
        elif not rows[r][0]:
            cell.set_edgecolor("none"); cell.set_facecolor(fs.SURFACE)
    ax.set_title(title, fontsize=11, y=1.01)


def main() -> None:
    fs = _style()
    import matplotlib.pyplot as plt

    from benchmark.ag3s.curobo_field import CuroboEsdfField, layer_from_arrays
    from benchmark.ag3s.experiments.doc_curobo_role import backproject

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frame", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--field-raw", required=True)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--out", default=str(OUT / "doc-stage-robot-mask.png"))
    args = ap.parse_args()

    fr = np.load(args.frame)
    centres = np.asarray(np.load(args.field)["sphere_centers"], np.float64)
    radii = np.asarray(np.load(args.field)["sphere_radii"], np.float64)

    def field(path):
        d = np.load(path)
        return CuroboEsdfField(layers=(
            layer_from_arrays(d["coarse_values"], np.asarray(d["coarse_origin"], float),
                              float(d["coarse_voxel_size"])),
            layer_from_arrays(d["fine_values"], np.asarray(d["fine_origin"], float),
                              float(d["fine_voxel_size"]))))

    ok, bad = field(args.field), field(args.field_raw)
    c_ok = ok.distance(centres) - radii - args.margin
    c_bad = bad.distance(centres) - radii - args.margin

    fig = plt.figure(figsize=(20.5, 11.8))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.26, top=0.87, bottom=0.05)

    # (a) 세 카메라의 마스크
    ax = fig.add_subplot(gs[0, 0])
    tiles = []
    for cid in IDS:
        d = np.asarray(fr[f"depth_{cid}"], float)
        m = np.asarray(fr[f"robot_mask_{cid}"], bool)
        rgb = plt.get_cmap("viridis")(np.clip(d / 2.5, 0, 1))[..., :3]
        rgb[m] = (0.90, 0.20, 0.18)
        tiles.append(rgb[::2, ::2])
    ax.imshow(np.hstack(tiles))
    n = tiles[0].shape[1]
    ax.set_xticks([n * 0.5, n * 1.5, n * 2.5])
    ax.set_xticklabels([f"{LABEL[i]}\n{100.0*np.asarray(fr[f'robot_mask_{c}'], bool).mean():.1f} % 삭제"
                        for i, c in enumerate(IDS)], fontsize=9)
    ax.set_yticks([])
    ax.set_title("(a) 실제 씬 — 세 카메라에서 지운 픽셀 (빨강)\n"
                 "FK 로 로봇 구를 각 카메라에 투영해 만든다 — 그 카메라의 촬영 시각 자세로\n"
                 "카메라마다 지우는 양이 다르다 (아래 d)", fontsize=11)

    # (b) 3인칭 — 마스크 전/후 점구름
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    rng = np.random.default_rng(0)
    raw = backproject(fr["depth_head"], fr["K_head"], fr["T_head"])
    kept = backproject(fr["depth_masked_head"], fr["K_head"], fr["T_head"])
    kv = {tuple(np.round(p, 4)) for p in kept[rng.choice(len(kept), 40000, replace=False)]}
    box = lambda p: ((p[:, 0] > 0.0) & (p[:, 0] < 1.1) & (np.abs(p[:, 1]) < 0.6)
                     & (p[:, 2] > 0.4) & (p[:, 2] < 1.4))
    rb, kb = raw[box(raw)], kept[box(kept)]
    s = rb[rng.choice(len(rb), min(15000, len(rb)), replace=False)]
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=0.7, c="#e34948", linewidths=0, alpha=0.30,
               label=f"마스크 전 {len(raw):,}")
    s = kb[rng.choice(len(kb), min(15000, len(kb)), replace=False)]
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=0.7, c="#2a78d6", linewidths=0, alpha=0.45,
               label=f"마스크 후 {len(kept):,}")
    ax.set_xlim(0.0, 1.1); ax.set_ylim(-0.6, 0.6); ax.set_zlim(0.4, 1.4)
    ax.set_xlabel("x [m] 앞", fontsize=8); ax.set_ylabel("y [m] 왼", fontsize=8)
    ax.set_zlabel("z [m] 위", fontsize=8)
    ax.view_init(elev=20, azim=-120); ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=8.5, markerscale=7, framealpha=0.85)
    ax.set_title("(b) 3인칭 — 지운 것은 팔이다 (head 카메라)\n"
                 "빨강만 남은 곳 = 삭제된 로봇 픽셀\n"
                 "이것을 안 지우면 로봇이 자기 팔을 장애물로 적분한다", fontsize=11)

    # (c) 마스크 유무로 여유거리가 어떻게 달라지나
    ax = fig.add_subplot(gs[0, 2])
    o = np.argsort(c_ok)
    k = np.arange(len(o))
    ax.plot(k, c_ok[o] * 1000, lw=2.0, color="#2a78d6", label="마스크 O (실제 경로)")
    ax.plot(k, c_bad[o] * 1000, lw=2.0, color="#e34948", label="마스크 X (대조군)")
    ax.axhline(0, color="#52514e", lw=1.2, ls="--")
    ax.fill_between(k, c_bad[o] * 1000, c_ok[o] * 1000, color="#e34948", alpha=0.15)
    ax.set_xlabel("로봇 구 (마스크 O 기준 오름차순)"); ax.set_ylabel("여유거리 [mm]")
    ax.legend(fontsize=9)
    ax.set_title("(c) 마스크를 빼면 — 같은 프레임, 같은 구, 같은 cuRobo 설정\n"
                 f"violated 구 {int((c_ok<0).sum())} → {int((c_bad<0).sum())} / {len(centres)}  ·  "
                 f"최악 {c_ok.min()*1000:+.1f} → {c_bad.min()*1000:+.1f} mm\n"
                 "이것이 발견 C1(검증 경로가 마스킹 안 된 depth 를 썼다) 이다", fontsize=11)

    # (d) 삭제량 — 프레임 전체가 아니라 카메라별
    ax = fig.add_subplot(gs[1, 0])
    frac = [100.0 * np.asarray(fr[f"robot_mask_{c}"], bool).mean() for c in IDS]
    bars = ax.bar(range(3), frac, color=[fs.CATEGORICAL[i] for i in range(3)], width=0.55)
    for b, f in zip(bars, frac):
        ax.text(b.get_x() + b.get_width() / 2, f + 0.4, f"{f:.2f} %", ha="center", fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(LABEL, fontsize=9.5)
    ax.set_ylabel("삭제된 픽셀 [%]")
    ax.set_ylim(0, max(frac) * 1.30)
    ax.set_title("(d) 카메라마다 얼마나 지우나 — 실측\n"
                 "세 대가 같은 로봇을 보는데 삭제 비율은 2.7 배 차이가 난다\n"
                 "마스크는 카메라마다 자기 촬영 시각의 자세로 따로 만들어진다", fontsize=11)

    # (e) 지운 픽셀이 '자유' 가 아니라 '미관측' 이 되는가
    ax = fig.add_subplot(gs[1, 1])
    hist_kw = dict(bins=120, range=(0.0, 1.6), histtype="step", lw=1.8, log=True)
    for i, cid in enumerate(IDS):
        d = np.asarray(fr[f"depth_{cid}"], float)
        m = np.asarray(fr[f"robot_mask_{cid}"], bool)
        ax.hist(d[m], color=fs.CATEGORICAL[i], label=f"{LABEL[i]} — 지운 픽셀의 원래 depth",
                **hist_kw)
    ax.set_xlabel("depth [m]"); ax.set_ylabel("픽셀 수 (log)")
    ax.legend(fontsize=8.5)
    ax.set_title("(e) 지운 픽셀은 어디에 있었나 — 전부 가까운 거리다\n"
                 "마스크된 픽셀은 0.0 으로 둔다. TSDF 도 cuRobo 도 그 광선을 통째로 건너뛰므로\n"
                 "'자유' 가 아니라 '미관측' 이 된다 — 이 구분이 E4 와 이어진다", fontsize=11)

    # (f) 표
    rows = [
        ("무엇을 받는가", ""),
        ("입력 1", "depth 한 장 + K + T_base_cam"),
        ("입력 2", "그 카메라 촬영 시각의 robot_state"),
        ("입력 3", "URDF 구 사슬 — 로봇 몸을 구로 근사한 것"),
        ("", ""),
        ("무엇을 내놓는가", ""),
        ("출력 1", "ESDF 용 이미지 마스크 — 픽셀을 0 으로 (미관측)"),
        ("출력 2", "클라우드용 self-filter — 점 단위 삭제"),
        ("둘이 다른 이유", "E7 — 해상도가 달라 재사용하면 89.3 % 를 놓친다"),
        ("", ""),
        ("겪은 문제", ""),
        ("C1", "검증 경로가 마스크 안 된 depth 를 썼다 — (c) 가 대조"),
        ("F12", "사과 픽셀 100 % 가 로봇으로 오인돼 삭제됐다"),
        ("F14", "상태 지연 한계 100 ms 가 느슨 — 96 ms 에 6,275 점 샘"),
        ("", ""),
        ("이 프레임 실측", ""),
        ("삭제 비율", " · ".join(f"{LABEL[i].split(chr(32))[0]} {frac[i]:.2f}%" for i in range(3))),
        ("여유거리 최악", f"마스크 O {c_ok.min()*1000:+.1f} mm  /  X {c_bad.min()*1000:+.1f} mm"),
        ("violated 구", f"마스크 O {int((c_ok<0).sum())} / X {int((c_bad<0).sum())} (전체 {len(centres)})"),
    ]
    _table(fig.add_subplot(gs[1, 2]), rows, fs, "(f) 입력 · 출력 · 겪은 문제 · 실측")

    fig.suptitle("로봇 마스크 (self-filter) — 로봇이 자기 몸을 장애물로 보지 않게 한다.  "
                 "run_0004 프레임 10", fontsize=14, y=0.955)
    fig.patch.set_facecolor(fs.SURFACE)
    for a in fig.axes:
        if a.name != "3d":
            a.set_facecolor(fs.SURFACE)
    fig.savefig(args.out, dpi=118, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {args.out}")
    print(f"  삭제 비율 {[f'{f:.2f}%' for f in frac]}")
    print(f"  여유거리 최악  마스크 O {c_ok.min()*1000:+.2f} mm  /  X {c_bad.min()*1000:+.2f} mm")
    print(f"  violated 구   마스크 O {int((c_ok<0).sum())}  /  X {int((c_bad<0).sum())} / {len(centres)}")


if __name__ == "__main__":
    main()
