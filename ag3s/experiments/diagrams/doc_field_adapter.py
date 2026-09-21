"""문서용 시각화 — 거리장 어댑터가 optimizer 에게 **무엇을** 답하는가 (distance / gradient).

    PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.diagrams.doc_field_adapter \\
        --frame /tmp/frame10.npz --field /tmp/field10.npz

`ag3s/fields/curobo_field.py` 의 `CuroboEsdfField` 는 cuRobo 가 만든 거리 격자 여러 층을 받아
`min()` 으로 합치고, 임의의 점에 대해 **거리와 기울기**를 답한다. 순수 numpy 이므로 cuRobo 가
없는 `.venv-ag3s` 에서 그대로 돈다 — 이 그림이 그 사실의 증명이기도 하다.

규칙 A: 실제 씬(단면) · 3인칭 · 그래프 · 표를 한 장에. 새 판정은 하지 않는다.
"""

from __future__ import annotations

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

    from benchmark.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frame", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--margin", type=float, default=0.05, help="esdf_margin [m]")
    ap.add_argument("--out", default=str(OUT / "doc-stage-field-adapter.png"))
    args = ap.parse_args()

    fr, fd = np.load(args.frame), np.load(args.field)
    centres = np.asarray(fd["sphere_centers"], np.float64)
    radii = np.asarray(fd["sphere_radii"], np.float64)
    tc = np.asarray(fd["target_centroid"], np.float64)

    coarse = layer_from_arrays(fd["coarse_values"], np.asarray(fd["coarse_origin"], float),
                               float(fd["coarse_voxel_size"]))
    fine = layer_from_arrays(fd["fine_values"], np.asarray(fd["fine_origin"], float),
                             float(fd["fine_voxel_size"]))
    two = CuroboEsdfField(layers=(coarse, fine), outside_distance=None)
    one = CuroboEsdfField(layers=(coarse,), outside_distance=None)

    d2 = two.distance(centres)
    d1 = one.distance(centres)
    g2 = two.gradient(centres)
    clear2 = d2 - radii - args.margin
    clear1 = d1 - radii - args.margin

    # 기울기가 거리와 어긋나지 않는가 — 중심차분과 대조한다. **차분 간격은 그 점의 거리를
    # 낸 계층의 복셀에 맞춘다.** 20 mm 계층이 답한 점을 2.5 mm 간격으로 차분하면 같은 복셀
    # 안을 두 번 찍게 되어 보간의 기울기가 아니라 격자의 계단을 재게 된다.
    winner = two._evaluate(centres, want_winner=True)[1]
    hstep = np.array([two.layers[int(w)].grid.voxel_size for w in winner]) * 0.5
    fdg = np.zeros_like(g2)
    for k in range(3):
        e = np.zeros((len(centres), 3)); e[:, k] = hstep
        fdg[:, k] = (two.distance(centres + e) - two.distance(centres - e)) / (2 * hstep)
    gerr = np.linalg.norm(g2 - fdg, axis=1)
    gnorm = np.linalg.norm(g2, axis=1)

    fig = plt.figure(figsize=(20.5, 11.8))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.26, top=0.87, bottom=0.05)

    # ------------------------------------ (a) 실제 씬 — 수직 단면 + 구 + 기울기 화살표
    ax = fig.add_subplot(gs[0, 0])
    cv = np.asarray(fd["coarse_values"], np.float64)
    co, cvs = np.asarray(fd["coarse_origin"], float), float(fd["coarse_voxel_size"])
    xs = co[0] + np.arange(cv.shape[0]) * cvs
    zs = co[2] + np.arange(cv.shape[2]) * cvs
    j = int(np.argmin(np.abs((co[1] + np.arange(cv.shape[1]) * cvs) - tc[1])))
    sl = cv[:, j, :]
    im = ax.pcolormesh(xs, zs, np.clip(sl, -0.05, 0.30).T, cmap="RdYlBu", shading="auto",
                       vmin=-0.05, vmax=0.30)
    ax.contour(xs, zs, sl.T, levels=[0.0], colors="k", linewidths=1.4)
    near = np.abs(centres[:, 1] - (co[1] + j * cvs)) < 0.08
    for c, r in zip(centres[near], radii[near]):
        ax.add_patch(plt.Circle((c[0], c[2]), r, fill=False, lw=0.9, color="k"))
    q = centres[near]
    gq = g2[near]
    ax.quiver(q[:, 0], q[:, 2], gq[:, 0], gq[:, 2], color="#1baf7a", width=0.005,
              scale=18, zorder=6)
    ax.plot(tc[0], tc[2], marker="*", ms=16, color="#d62fd6", ls="none")
    ax.set_xlim(0.0, 1.05); ax.set_ylim(0.5, 1.35)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m] — 앞 →"); ax.set_ylabel("z [m] — 위 ↑")
    plt.colorbar(im, ax=ax, fraction=0.04, label="d [m]")
    ax.set_title(f"(a) 실제 씬 — y = {co[1] + j*cvs:.3f} m 수직 단면\n"
                 "검은 원 = 로봇 구, 초록 화살표 = 그 구가 받은 ∇d\n"
                 "화살표는 '표면에서 멀어지는 방향' 이고 QP 가 이걸 따라 민다", fontsize=11)

    # ---------------------------------------- (b) 3인칭 — 구를 여유거리로 칠한 것
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    sc = ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2],
                    s=np.clip(radii * 4200, 6, 90), c=clear2 * 1000.0,
                    cmap="RdYlGn", vmin=-60, vmax=120, linewidths=0.3, edgecolors="#333")
    ax.scatter(*tc, s=180, marker="*", color="#d62fd6", zorder=6)
    ax.set_xlim(0.0, 0.9); ax.set_ylim(-0.45, 0.45); ax.set_zlim(0.55, 1.35)
    ax.set_xlabel("x [m] 앞", fontsize=8); ax.set_ylabel("y [m] 왼", fontsize=8)
    ax.set_zlabel("z [m] 위", fontsize=8)
    ax.view_init(elev=20, azim=-120); ax.tick_params(labelsize=7)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.10, label="여유거리 d − r − margin [mm]")
    ax.set_title(f"(b) 로봇 구 {len(centres)} 개를 여유거리로 칠한 것 (3인칭)\n"
                 f"margin = {args.margin*1000:.0f} mm.  0 미만(빨강)이 violated\n"
                 f"이 프레임 최악 {clear2.min()*1000:+.1f} mm, "
                 f"음수 {int((clear2 < 0).sum())} 개", fontsize=11)

    # ------------------------------------------------ (c) 구별 d / r / 여유거리
    ax = fig.add_subplot(gs[0, 2])
    o = np.argsort(clear2)
    k = np.arange(len(o))
    ax.plot(k, d2[o] * 1000, lw=1.4, color="#2a78d6", label="필드가 답한 거리 d")
    ax.plot(k, radii[o] * 1000, lw=1.2, color="#52514e", ls=":", label="구 반지름 r")
    ax.plot(k, clear2[o] * 1000, lw=2.0, color="#eb6834", label="여유거리 d − r − margin")
    ax.axhline(0, color="#e34948", lw=1.2, ls="--")
    ax.fill_between(k, clear2[o] * 1000, 0, where=clear2[o] < 0, color="#e34948", alpha=0.22)
    ax.set_xlabel("로봇 구 (여유거리 오름차순)"); ax.set_ylabel("[mm]")
    ax.legend(fontsize=8.5)
    ax.set_title("(c) 어댑터 한 번 호출이 내놓는 제약 행 전부\n"
                 "제약은 구 하나당 한 행: d(p(q)) − r − margin ≥ 0\n"
                 f"여기서는 {len(centres)} 행. primitive 는 (구 x 슬롯) 이 행 수였다",
                 fontsize=11)

    # ----------------------------- (d) 2계층이 거친 계층 단독과 얼마나 다른가 (C5/C3)
    ax = fig.add_subplot(gs[1, 0])
    diff = (d2 - d1) * 1000.0
    inwin = np.all((centres >= fine.grid.origin) &
                   (centres <= fine.grid.origin + fine.grid.voxel_size *
                    (np.asarray(fd["fine_values"].shape) - 1)), axis=1)
    ax.hist(diff[inwin], bins=40, color="#2a78d6", alpha=0.85,
            label=f"미세 창 안 {int(inwin.sum())} 구")
    ax.axvline(0, color="#52514e", lw=1.2)
    ax.axvline(float(np.median(diff[inwin])), color="#e34948", lw=1.8, ls="--")
    ax.annotate(f"중앙 {np.median(diff[inwin]):+.2f} mm",
                xy=(float(np.median(diff[inwin])), 0.85), xycoords=("data", "axes fraction"),
                color="#e34948", fontsize=9.5)
    ax.set_xlabel("2계층 − 거친 단독  [mm]   (음수 = 2계층이 더 조인다)")
    ax.set_ylabel("구 개수")
    ax.legend(fontsize=8.5)
    ax.set_title("(d) 미세 계층을 얹으면 답이 어떻게 바뀌나\n"
                 "min() 합성이므로 결코 더 후해지지 않는다 — 조이는 쪽으로만 움직인다\n"
                 "C5(거친 20 mm 는 판정 지점에서 +7.56 mm 낙관적) 가 이 조임의 근거다",
                 fontsize=11)

    # ------------------------------ (e) 기울기가 거리와 어긋나지 않는가 (SQP 의 전제)
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(gnorm, gerr * 1000.0, s=26, color="#2a78d6", alpha=0.75, linewidths=0)
    ax.axvline(1.0, color="#e34948", ls="--", lw=1.4)
    ax.set_xlabel("|∇d|  (이상적으로 1)"); ax.set_ylabel("해석 기울기 − 중심차분  [mm/m]")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title("(e) 기울기 검증 — 어댑터가 준 ∇d 대 같은 필드의 중심차분\n"
                 f"|∇d| 중앙 {np.median(gnorm):.3f}  ·  어긋남 중앙 {np.median(gerr)*1000:.2e} mm/m\n"
                 "값과 기울기가 어긋나면 SQP 가 수렴하지 않는다 — 그래서 같은 EsdfField 를 재사용한다",
                 fontsize=11)

    # ------------------------------------------------------------------- (f) 표
    rows = [
        ("무엇을 받는가", ""),
        ("입력 1", f"cuRobo 거리 격자 {len(two.layers)} 층 (거친 것부터)"),
        ("입력 2", f"질의점 — 로봇 구 {len(centres)} 개 + 쥔 물체의 점"),
        ("", ""),
        ("무엇을 내놓는가", ""),
        ("distance(p)", "(N,) m — 삼선형 보간, 음수 = 표면 안쪽"),
        ("gradient(p)", "(N,3) — 거리를 낸 바로 그 층의 중심차분"),
        ("합성 규칙", "층별로 구해 min() — 범위는 거친, 정밀도는 미세"),
        ("", ""),
        ("겪은 문제", ""),
        ("C4", "compute_esdf 결과를 덮어씀 → |∇d| = 0.25"),
        ("C3", "미세 창 경계 불연속 (중앙 +4.5, 최대 +41.6 mm)"),
        ("E4 / F16", "격자 밖을 조용히 '자유' 로 답하던 것"),
        ("", ""),
        ("이 프레임 실측", ""),
        ("여유거리 최악", f"{clear2.min()*1000:+.1f} mm   (음수 {int((clear2<0).sum())} / {len(centres)})"),
        ("2계층 − 거친", f"중앙 {np.median(diff[inwin]):+.2f} mm (미세 창 안 {int(inwin.sum())} 구)"),
        ("|∇d| 중앙", f"{np.median(gnorm):.4f}"),
    ]
    _table(fig.add_subplot(gs[1, 2]), rows, fs, "(f) 입력 · 출력 · 겪은 문제 · 실측")

    fig.suptitle("거리장 어댑터 — cuRobo 격자를 optimizer 가 묻는 두 함수로 바꾼다: "
                 "distance(p) 와 gradient(p).  run_0004 프레임 10",
                 fontsize=14, y=0.955)
    fig.patch.set_facecolor(fs.SURFACE)
    for a in fig.axes:
        if a.name != "3d":
            a.set_facecolor(fs.SURFACE)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=118, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {args.out}")
    print(f"  여유거리 최악 {clear2.min()*1000:+.2f} mm  음수 {int((clear2<0).sum())}/{len(centres)}")
    print(f"  거친 단독 최악 {clear1.min()*1000:+.2f} mm")
    print(f"  2계층 − 거친: 미세 창 안 {int(inwin.sum())} 구, 중앙 {np.median(diff[inwin]):+.3f} mm, "
          f"최소 {diff[inwin].min():+.3f} mm")
    print(f"  |∇d| 중앙 {np.median(gnorm):.4f}   기울기 어긋남 중앙 {np.median(gerr):.3e}")


if __name__ == "__main__":
    main()
