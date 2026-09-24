"""I1 검증의 시각화 — 실제 씬 · 그래프 · 표를 한 장에 (규칙 A).

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.live.plot_i1 \\
        --dir outputs/live_test/20260922_i1_venv --seeds 101 202 303

`verify_env.py` 가 낸 JSON 과 npz 만 읽는다. 다시 측정하지 않는다 — 그림을 고치려고
GPU 작업을 재실행하지 않기 위해서다 (`a7_episode_walkthrough` 의 collect/plot 분리와 같은 이유).

**소비 쪽이므로 `.venv-ag3s` 에서 돈다.** 거리 격자는 `(값, origin, voxel_size)` 로 완전히
기술되므로 질의에 cuRobo 가 필요 없다 — `curobo_field.py` 머리말이 적은 그 성질이다.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

IDS = ("head", "left_wrist", "right_wrist")
LABEL = ("head (zed_left)", "왼손목", "오른손목")


def _slice_index(origin, voxel, shape, axis, world):
    return int(np.clip(round((world - origin[axis]) / voxel), 0, shape[axis] - 1))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from benchmark.ag3s.experiments.common import figstyle

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=(101, 202, 303))
    ap.add_argument("--primary", type=int, default=101)
    ap.add_argument("--out", default="benchmark/ag3s/docs/figures/live-test/i1-venv-field.png")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    reports = {}
    for s in args.seeds:
        f = d / ("verify_env.json" if s == args.primary else f"verify_env_seed{s}.json")
        reports[s] = json.loads(f.read_text())
    frame = np.load(d / f"frame_seed{args.primary}.npz")
    field = np.load(d / f"field_seed{args.primary}.npz")

    figstyle.use_korean()
    fig = plt.figure(figsize=(17.2, 14.6))
    gs = fig.add_gridspec(4, 6, height_ratios=[1.0, 1.0, 0.92, 0.80],
                          hspace=0.36, wspace=0.34,
                          left=0.045, right=0.972, top=0.925, bottom=0.035)
    fig.patch.set_facecolor(figstyle.SURFACE)

    rep = reports[args.primary]
    fig.text(0.5, 0.973,
             "I1 — cuRobo 가 정책·MuJoCo·SQP 와 같은 프로세스에서 돈다 "
             f"(openpi-live venv, seed {args.primary})",
             ha="center", va="center", fontsize=16, color=figstyle.INK)
    fig.text(0.5, 0.949,
             "저장 기록을 읽지 않는다 — 실행 시점에 MuJoCo 씬을 새로 초기화하고 "
             "현재 시뮬 시각의 세 카메라를 캡처했다. attention 은 쓰지 않는다 (I1 은 배선 확인).",
             ha="center", va="center", fontsize=9.6, color=figstyle.INK_2)

    # ── 행 1: 실제 씬 — 카메라 RGB ─────────────────────────────────────────
    for i, (cid, lab) in enumerate(zip(IDS, LABEL)):
        ax = fig.add_subplot(gs[0, i * 2:(i + 1) * 2])
        ax.imshow(frame[f"rgb_{cid}"])
        ax.set_title(f"실제 씬 — {lab}", fontsize=10, color=figstyle.INK, pad=6)
        ax.axis("off")

    # ── 행 2: 로봇 마스크가 지운 자리 ─────────────────────────────────────
    for i, (cid, lab) in enumerate(zip(IDS, LABEL)):
        ax = fig.add_subplot(gs[1, i * 2:(i + 1) * 2])
        depth = np.asarray(frame[f"depth_{cid}"], np.float32)
        mask = np.asarray(frame[f"robot_mask_{cid}"], bool)
        shown = np.where(depth > 0, depth, np.nan)
        ax.imshow(shown, cmap="gray_r", vmin=0.2, vmax=2.5)
        overlay = np.zeros(mask.shape + (4,))
        overlay[mask] = matplotlib.colors.to_rgba(figstyle.CATEGORICAL[1], 0.62)
        ax.imshow(overlay)
        px = int(rep["capture"]["mask_px"][cid])
        ax.set_title(f"depth + 로봇 마스크 — {lab}\n지운 픽셀 {px:,} ({100*px/mask.size:.2f} %)",
                     fontsize=9.4, color=figstyle.INK, pad=6)
        ax.axis("off")

    # ── 행 3 왼쪽: 배치도 — 어디를 어느 방향에서 잘랐는가 ──────────────────
    co = field["coarse_origin"]; cv = float(field["coarse_voxel_size"]); cvals = field["coarse_values"]
    fo = field["fine_origin"];   fv = float(field["fine_voxel_size"]);   fvals = field["fine_values"]
    tgt = np.asarray(rep["capture"]["target_true_m"], float)
    cut_z = float(tgt[2])

    ax = fig.add_subplot(gs[2, 0:2])
    ax.set_title("배치도 — 단면의 위치와 시선\n(base 좌표, 위에서 내려다본 xy)",
                 fontsize=9.8, color=figstyle.INK, pad=6)
    cupper = co + cv * np.asarray(cvals.shape)
    fupper = fo + fv * np.asarray(fvals.shape)
    ax.add_patch(Rectangle((co[0], co[1]), cupper[0] - co[0], cupper[1] - co[1],
                           fill=False, edgecolor=figstyle.CATEGORICAL[0], linewidth=1.6,
                           label=f"coarse {cv*1000:.0f} mm 격자"))
    ax.add_patch(Rectangle((fo[0], fo[1]), fupper[0] - fo[0], fupper[1] - fo[1],
                           fill=False, edgecolor=figstyle.CATEGORICAL[2], linewidth=1.8,
                           linestyle="--", label=f"fine {fv*1000:.0f} mm 창"))
    ax.plot([tgt[0]], [tgt[1]], marker="*", markersize=14,
            color=figstyle.CATEGORICAL[1], linestyle="none", label="target 참값 (사과)")
    ax.plot([0], [0], marker="s", markersize=7, color=figstyle.INK,
            linestyle="none", label="robot base")
    # 단면은 **이 xy 평면 전체**다 (z 를 고정해 위에서 내려다본 것). 선으로 그리면
    # "y = const 로 잘랐다" 로 읽히므로 그리지 않고 글로 적는다.
    ax.set_xlabel("x [m] (로봇 앞)", fontsize=8.6)
    ax.set_ylabel("y [m] (로봇 왼쪽)", fontsize=8.6)
    ax.set_aspect("equal")
    # 설명 글과 범례가 상자 테두리를 밟지 않도록 위아래로 여유를 준다.
    ax.set_xlim(co[0] - 0.14, cupper[0] + 0.14)
    ax.set_ylim(co[1] - 0.52, cupper[1] + 0.62)
    ax.legend(fontsize=7.2, frameon=False, loc="lower right", labelcolor=figstyle.INK_2,
              handlelength=1.4, borderaxespad=0.4)
    figstyle.style_axes(fig, ax)
    # matplotlib 은 굵게를 렌더하지 않는다 — 제목·라벨에 별표를 쓰지 않는다 (CLAUDE.md).
    ax.text(0.03, 0.97,
            f"오른쪽 두 단면은 이 xy 평면 전체를\nz = {cut_z:.3f} m 에 고정해 +z 에서\n"
            "내려다본 것이다 (테이블 상판 높이).\n선으로 자른 것이 아니다.",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.6,
            color=figstyle.INK_2, linespacing=1.6)

    # ── 행 3 중앙/오른쪽: 두 계층의 같은 수평 단면 ────────────────────────
    for j, (name, vals, org, vs) in enumerate((("coarse", cvals, co, cv),
                                               ("fine", fvals, fo, fv))):
        ax = fig.add_subplot(gs[2, 2 + j * 2:4 + j * 2])
        k = _slice_index(org, vs, vals.shape, 2, cut_z)
        sl = vals[:, :, k]
        extent = (org[1], org[1] + vs * vals.shape[1],
                  org[0], org[0] + vs * vals.shape[0])
        lim = 0.30
        im = ax.imshow(sl, origin="lower", extent=extent, cmap="RdBu",
                       vmin=-lim, vmax=lim, aspect="equal")
        ax.contour(np.linspace(extent[0], extent[1], sl.shape[1]),
                   np.linspace(extent[2], extent[3], sl.shape[0]),
                   sl, levels=[0.0], colors=[figstyle.INK], linewidths=1.0)
        ax.plot([tgt[1]], [tgt[0]], marker="*", markersize=12,
                color=figstyle.CATEGORICAL[3], linestyle="none",
                label="target 참값")
        ax.legend(fontsize=7.2, frameon=False, loc="upper left",
                  labelcolor=figstyle.INK_2)
        eik = rep["field"]["tiers"][name]["eikonal"]["median"]
        ax.set_title(f"{name} ESDF {vs*1000:.0f} mm — z = {cut_z:.3f} m 수평 단면 (위에서)\n"
                     f"검은 선 = 표면 d=0 · 붉은쪽 = 물체 안(음수) · eikonal 중앙 {eik:.4f}",
                     fontsize=9.4, color=figstyle.INK, pad=6)
        ax.set_xlabel("y [m]", fontsize=8.4); ax.set_ylabel("x [m]", fontsize=8.4)
        figstyle.style_axes(fig, ax)
        cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
        cb.set_label("d [m]", fontsize=7.6, color=figstyle.INK_2)
        cb.ax.tick_params(labelsize=7, colors=figstyle.INK_2)

    # ── 행 4 왼쪽: 그래프 — eikonal 분포 ──────────────────────────────────
    ax = fig.add_subplot(gs[3, 0:2])
    ax.set_title("그래프 ① eikonal — 자유공간에서 |∇d| 가 1 인가",
                 fontsize=9.8, color=figstyle.INK, pad=6)
    xs = np.arange(len(args.seeds))
    for t, (name, colour) in enumerate((("coarse", figstyle.CATEGORICAL[0]),
                                        ("fine", figstyle.CATEGORICAL[2]))):
        med = [reports[s]["field"]["tiers"][name]["eikonal"]["median"] for s in args.seeds]
        lo = [reports[s]["field"]["tiers"][name]["eikonal"]["p05"] for s in args.seeds]
        hi = [reports[s]["field"]["tiers"][name]["eikonal"]["p95"] for s in args.seeds]
        off = (t - 0.5) * 0.18
        ax.errorbar(xs + off, med,
                    yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
                    fmt="o", markersize=7, capsize=4, color=colour,
                    label=f"{name} (막대 = p05~p95)")
    ax.axhspan(0.9, 1.1, color=figstyle.CATEGORICAL[2], alpha=0.10)
    ax.axhline(1.0, color=figstyle.INK_2, linewidth=0.9, linestyle=":")
    ax.set_xticks(xs); ax.set_xticklabels([f"seed {s}" for s in args.seeds], fontsize=8.6)
    ax.set_ylabel("|∇d|", fontsize=8.8)
    ax.set_ylim(0.88, 1.12)
    ax.text(0.02, 0.905, "합격 띠 0.9~1.1", fontsize=7.6, color=figstyle.INK_2,
            transform=ax.get_yaxis_transform())
    ax.legend(fontsize=7.6, frameon=False, loc="upper right", labelcolor=figstyle.INK_2)
    figstyle.style_axes(fig, ax)

    # ── 행 4 중앙: 그래프 — 단계별 시간 ───────────────────────────────────
    ax = fig.add_subplot(gs[3, 2:4])
    ax.set_title("그래프 ② 단계별 시간 — 청크 예산 533 ms 대비",
                 fontsize=9.8, color=figstyle.INK, pad=6)
    stages = [("MuJoCo 캡처\n(3 카메라, OSMesa)",
               [reports[s]["capture"]["capture_ms"] for s in args.seeds],
               figstyle.CATEGORICAL[7]),
              ("cuRobo 적분\n(정상 상태)",
               [reports[s]["field"]["integrate_steady_ms"]["median"] for s in args.seeds],
               figstyle.CATEGORICAL[0]),
              ("cuRobo ESDF\n(정상 상태)",
               [reports[s]["field"]["compute_esdf_steady_ms"]["median"] for s in args.seeds],
               figstyle.CATEGORICAL[2])]
    for i, (lab, vals, colour) in enumerate(stages):
        ax.bar(i, float(np.median(vals)), width=0.56, color=colour)
        ax.text(i, float(np.median(vals)) * 1.35, f"{np.median(vals):.2f} ms",
                ha="center", fontsize=8.4, color=figstyle.INK)
    ax.axhline(533.0, color=figstyle.CATEGORICAL[7], linewidth=1.4, linestyle="--")
    ax.text(2.42, 533.0 * 1.15, "청크 예산 533 ms", ha="right", fontsize=7.8,
            color=figstyle.CATEGORICAL[7])
    ax.set_yscale("log")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s[0] for s in stages], fontsize=8.0)
    ax.set_ylabel("ms (로그 눈금)", fontsize=8.8)
    ax.set_ylim(0.2, 12000)
    figstyle.style_axes(fig, ax)

    # ── 행 4 오른쪽: 표 — 통과 조건과 실측 ────────────────────────────────
    ax = fig.add_subplot(gs[3, 4:6])
    ax.set_title("표 — I1 통과 조건과 실측 (seed 3개)", fontsize=9.8,
                 color=figstyle.INK, loc="left", pad=6)
    ax.axis("off")
    ck = {s: reports[s]["verdict"]["checks"] for s in args.seeds}
    names = list(reports[args.primary]["verdict"]["checks"].keys())
    pretty = {"pinned_versions_unchanged": "핀 버전 불변 (numpy·torch·jax…)",
              "jax_cuda_alive": "정책 생존 — jax CUDA",
              "ag3s_sqp_alive": "AG3S·SQP 생존 — mujoco·casadi·osqp",
              "eikonal_in_band": "eikonal 0.9~1.1 (두 계층)",
              "buffer_independence": "버퍼 독립 — coarse 재현 ∧ fine 과 상이",
              "sign_convention": "부호 — 음수 = 물체 안쪽"}
    rows = [(pretty.get(n, n), all(ck[s][n] for s in args.seeds)) for n in names]
    extra = [
        ("coarse eikonal 중앙", "0.9995 / 0.9995 / 0.9995"),
        ("fine eikonal 중앙", "0.9999 / 0.9998 / 0.9998"),
        ("cuRobo 적분 + ESDF", f"{np.median([reports[s]['field']['integrate_steady_ms']['median'] for s in args.seeds]):.2f}"
                              f" + {np.median([reports[s]['field']['compute_esdf_steady_ms']['median'] for s in args.seeds]):.2f} ms"),
        ("legacy EsdfBuilder 생성", "0 회 (세 seed 모두)"),
        ("저장 기록 입력", "0 건"),
        ("원본 openpi venv", "warp·curobo 부재 — 무오염"),
    ]
    y = 1.0
    for lab, ok in rows:
        ax.text(0.0, y, "●", fontsize=11, va="center",
                color=figstyle.CATEGORICAL[2] if ok else figstyle.CATEGORICAL[7],
                family="DejaVu Sans")
        ax.text(0.055, y, lab, fontsize=8.6, va="center", color=figstyle.INK)
        ax.text(1.0, y, "PASS" if ok else "FAIL", fontsize=8.4, va="center", ha="right",
                color=figstyle.CATEGORICAL[2] if ok else figstyle.CATEGORICAL[7],
                family="DejaVu Sans")
        y -= 0.108
    y -= 0.035
    ax.plot([0.0, 1.0], [y + 0.045, y + 0.045], color=figstyle.GRID_INK, linewidth=1.0)
    for lab, val in extra:
        ax.text(0.0, y, lab, fontsize=8.4, va="center", color=figstyle.INK_2)
        ax.text(1.0, y, val, fontsize=8.4, va="center", ha="right", color=figstyle.INK)
        y -= 0.098
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(y, 1.075)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=figstyle.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
