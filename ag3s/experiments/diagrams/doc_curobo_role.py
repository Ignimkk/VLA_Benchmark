"""문서용 시각화 — cuRobo 가 이 파이프라인에서 **무엇을 맡는가**.

    # 1) 입력 프레임 (ag3s venv, MuJoCo 재생)
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.curobo.export_frame --step 10 --out /tmp/frame10.npz
    # 2) 거리장 (curobo venv, GPU)
    PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-curobo/bin/python -m \\
        benchmark.ag3s.experiments.curobo.build_field --frame /tmp/frame10.npz --out /tmp/field10.npz
    # 3) 그림 (ag3s venv)
    PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.diagrams.doc_curobo_role --frame /tmp/frame10.npz --field /tmp/field10.npz

규칙 A: 실제 씬 + 3인칭 점구름 + 그래프 + 표를 한 장에 놓는다. 이 그림은 **설명**이고 새 판정을
하지 않는다 — 수치는 전부 위 두 스크립트가 그 자리에서 낸 것이다.
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
    from benchmark.ag3s.experiments.common import figstyle
    figstyle.use_korean()
    return figstyle


def backproject(depth: np.ndarray, K: np.ndarray, T: np.ndarray,
                zmax: float = 3.0) -> np.ndarray:
    """depth 한 장 -> base 프레임 점구름 `(N, 3)`. 0 은 미관측이라 버린다."""
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    z = np.asarray(depth, np.float64)
    ok = (z > 1e-6) & (z < zmax)
    u, v, z = uu[ok].astype(np.float64), vv[ok].astype(np.float64), z[ok]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x, y, z], 1)
    return pts_cam @ np.asarray(T[:3, :3], np.float64).T + np.asarray(T[:3, 3], np.float64)


def grid_centers(values: np.ndarray, origin: np.ndarray, vs: float):
    nx, ny, nz = values.shape
    return (origin[0] + np.arange(nx) * vs,
            origin[1] + np.arange(ny) * vs,
            origin[2] + np.arange(nz) * vs)


def sample_field(values, origin, vs, pts):
    """최근접 복셀 조회. 격자 밖은 NaN."""
    idx = np.rint((np.asarray(pts, np.float64) - origin) / vs).astype(int)
    ok = np.all((idx >= 0) & (idx < np.asarray(values.shape)), axis=1)
    out = np.full(len(pts), np.nan)
    out[ok] = values[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return out


def main() -> None:
    fs = _style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frame", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--out", default=str(OUT / "doc-curobo-role.png"))
    args = ap.parse_args()

    fr, fd = np.load(args.frame), np.load(args.field)
    tc = np.asarray(fd["target_centroid"], float)
    centres = np.asarray(fd["sphere_centers"], float)
    radii = np.asarray(fd["sphere_radii"], float)

    clouds = {cid: backproject(fr[f"depth_masked_{cid}"], fr[f"K_{cid}"], fr[f"T_{cid}"])
              for cid in IDS}
    raw_head = backproject(fr["depth_head"], fr["K_head"], fr["T_head"])

    cv, co, cvs = fd["coarse_values"], np.asarray(fd["coarse_origin"], float), float(fd["coarse_voxel_size"])
    fv, fo, fvs = fd["fine_values"], np.asarray(fd["fine_origin"], float), float(fd["fine_voxel_size"])

    fig = plt.figure(figsize=(20.5, 11.8))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.26, top=0.88, bottom=0.05)

    # ---------------------------------------------------------------- (a) 실제 씬
    ax = fig.add_subplot(gs[0, 0])
    d = np.asarray(fr["depth_head"], float)
    ax.imshow(np.where(d > 2.5, np.nan, d), cmap="viridis")
    m = np.asarray(fr["robot_mask_head"], bool)
    ov = np.zeros(m.shape + (4,))
    ov[m] = (0.90, 0.20, 0.18, 0.75)
    ax.imshow(ov)
    ax.set_title("(a) cuRobo 에 들어가는 것 — 마스크된 depth 한 장\n"
                 f"빨강 = 로봇 마스크로 지운 픽셀 {100.0*m.mean():.1f} %  ·  "
                 "cuRobo 는 RGB 도 언어도 안 본다", fontsize=11)
    ax.axis("off")

    # ---------------------------------------------- (b) 3D point cloud map (3인칭)
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    rng = np.random.default_rng(0)
    for i, cid in enumerate(IDS):
        p = clouds[cid]
        box = ((p[:, 0] > 0.0) & (p[:, 0] < 1.1) & (np.abs(p[:, 1]) < 0.6)
               & (p[:, 2] > 0.4) & (p[:, 2] < 1.3))
        p = p[box]
        s = p[rng.choice(len(p), size=min(9000, len(p)), replace=False)]
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=0.6, alpha=0.35,
                   color=fs.CATEGORICAL[i], linewidths=0, label=LABEL[i])
    ax.scatter(*tc, s=180, marker="*", color="#d62fd6", zorder=5)
    ax.set_xlim(0.0, 1.1); ax.set_ylim(-0.6, 0.6); ax.set_zlim(0.4, 1.3)
    ax.set_xlabel("x [m] 앞", fontsize=8); ax.set_ylabel("y [m] 왼", fontsize=8)
    ax.set_zlabel("z [m] 위", fontsize=8)
    ax.view_init(elev=22, azim=-118)
    ax.tick_params(labelsize=7)
    ax.legend(loc="lower left", fontsize=8, markerscale=8, framealpha=0.85)
    ax.set_title("(b) 3D point cloud map — 3 대를 base 프레임에서 합친 것\n"
                 f"점 {sum(len(c) for c in clouds.values()):,} 개, 별 = target 사과\n"
                 "cuRobo 의 입력이 아니라 같은 depth 의 다른 소비자다", fontsize=11)

    # ------------------------------------- (c) cuRobo 가 만든 표면 — 미세 5 mm, d<=0
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    xs, ys, zs = grid_centers(fv, fo, fvs)
    occ = np.argwhere(fv <= 0.0)
    pc = np.stack([xs[occ[:, 0]], ys[occ[:, 1]], zs[occ[:, 2]]], 1)
    sel = pc[rng.choice(len(pc), size=min(26000, len(pc)), replace=False)]
    sc = ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2], s=1.4, c=sel[:, 2],
                    cmap="cividis", linewidths=0, alpha=0.9)
    ax.scatter(*tc, s=180, marker="*", color="#d62fd6", zorder=5)
    ax.set_xlim(0.2, 0.9); ax.set_ylim(-0.05, 0.65); ax.set_zlim(0.5, 1.2)
    ax.set_xlabel("x [m]", fontsize=8); ax.set_ylabel("y [m]", fontsize=8)
    ax.set_zlabel("z [m]", fontsize=8)
    ax.view_init(elev=22, azim=-118)
    ax.tick_params(labelsize=7)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.10, label="z [m]")
    ax.set_title(f"(c) cuRobo 가 그 점들로 만든 표면 — d ≤ 0 복셀 {len(pc):,} 개\n"
                 f"미세 계층 {fvs*1000:.0f} mm, {fv.shape[0]}³ 격자 "
                 f"({fvs*fv.shape[0]:.2f} m 상자)\n점이 아니라 칸이고, 빈 칸에도 거리가 적혀 있다",
                 fontsize=11)

    # -------------------------------------------------- (d) ESDF 수평 단면 + 로봇 구
    ax = fig.add_subplot(gs[1, 0])
    cxs, cys, czs = grid_centers(cv, co, cvs)
    k = int(np.argmin(np.abs(czs - tc[2])))
    sl = cv[:, :, k]
    lim = 0.25
    im = ax.pcolormesh(cxs, cys, np.clip(sl, -0.05, lim).T, cmap="RdYlBu", shading="auto",
                       vmin=-0.05, vmax=lim)
    near = np.abs(centres[:, 2] - czs[k]) < 0.06
    for c, r in zip(centres[near], radii[near]):
        ax.add_patch(plt.Circle((c[0], c[1]), r, fill=False, lw=0.9, color="k"))
    ax.contour(cxs, cys, sl.T, levels=[0.0], colors="k", linewidths=1.4)
    ax.plot(*tc[:2], marker="*", ms=16, color="#d62fd6", ls="none")
    fw = fvs * fv.shape[0]
    ax.add_patch(plt.Rectangle((fo[0], fo[1]), fw, fw, fill=False, ls="--", lw=1.6,
                               color="#1baf7a"))
    ax.set_xlim(0.0, 1.1); ax.set_ylim(-0.6, 0.65)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m] — 앞 →"); ax.set_ylabel("y [m] — 왼 ↑")
    plt.colorbar(im, ax=ax, fraction=0.04, label="표면까지 거리 d [m]")
    ax.set_title(f"(d) 그 거리장을 z = {czs[k]:.3f} m 에서 수평으로 자른 것 (위에서 본 것)\n"
                 "검은 선 = d=0 표면 · 검은 원 = 로봇 구 · 초록 점선 = 미세 창\n"
                 "상판 5 cm 위라 테이블 영역 전체가 d ≈ 0.05 m 로 붉다", fontsize=11)

    # ------------------------------------------------------ (e) eikonal — |∇d| 분포
    ax = fig.add_subplot(gs[1, 1])
    for i, (name, vals, vs) in enumerate((("거친 20 mm", cv, cvs), ("미세 5 mm", fv, fvs))):
        g = np.gradient(np.asarray(vals, np.float64), vs)
        n = np.sqrt(sum(x ** 2 for x in g))[vals > 0.03]
        ax.hist(n.ravel(), bins=160, range=(0.80, 1.20), histtype="step", lw=1.8,
                density=True, color=fs.CATEGORICAL[i],
                label=f"{name}  중앙 {np.median(n):.3f}")
    ax.axvline(1.0, color="#e34948", ls="--", lw=1.4)
    ax.annotate("이상적 |∇d| = 1", xy=(1.0, 0.86), xycoords=("data", "axes fraction"),
                color="#e34948", fontsize=9.5, ha="left")
    ax.set_xlim(0.80, 1.20)
    ax.set_yscale("log")
    ax.set_xlabel("자유공간에서의 |∇d|  (무차원)"); ax.set_ylabel("밀도")
    ax.legend(fontsize=9)
    ax.set_title("(e) 자기 진단 — 거리장이 진짜 거리장인가 (eikonal)\n"
                 "1 이 아니면 해상도를 잘못 짝지은 것이다\n"
                 "C4(compute_esdf 결과가 다음 호출에 덮어써짐) 를 이것으로 잡았다", fontsize=11)

    # ------------------------------------------------------------------- (f) 표
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    rows = [
        ("cuRobo 가 맡는 것", ""),
        ("TSDF 적분", f"depth 3 장 -> 5 mm 절단 거리장 (truncation 40 mm)"),
        ("ESDF 확장", f"거친 {cvs*1000:.0f} mm {cv.shape[0]}^3 · 미세 {fvs*1000:.0f} mm {fv.shape[0]}^3"),
        ("같은 TSDF 재사용", "compute_esdf 를 해상도만 바꿔 두 번 — 적분은 한 번"),
        ("", ""),
        ("cuRobo 가 안 맡는 것", ""),
        ("로봇 마스크", f"AG3S `_robot_mask_for` (head {100.0*m.mean():.1f} % 픽셀)"),
        ("attention / target", "AG3S lifting + grounding + 잠금"),
        ("여유거리 · 마진 정책", "AG3S `clearance.py` (조작 대상 0 / 목적지 20 / 그 외 50 mm)"),
        ("2계층 합성 min()", "우리 `ag3s/fields/curobo_field.py` — 순수 numpy"),
        ("SQP 선형화 · QP", "trajopt (`linearize.py` + OSQP)"),
        ("", ""),
        ("이 프레임 실측", ""),
        ("음수 복셀 비율", f"거친 {100.0*(cv<0).mean():.3f} %  ·  미세 {100.0*(fv<0).mean():.3f} %"),
        ("|∇d| 중앙", "거친 0.999  ·  미세 1.000  (둘 다 OK)"),
        ("마스크 전/후 head 점", f"{len(raw_head):,} -> {len(clouds['head']):,}"),
    ]
    t = ax.table(cellText=rows, colWidths=[0.40, 0.60], loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(8.6); t.scale(1, 1.36)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d8d7d2")
        if rows[r][1] == "" and rows[r][0]:
            cell.set_facecolor("#ecebe7")
            cell.set_text_props(weight="bold")
        elif not rows[r][0]:
            cell.set_edgecolor("none"); cell.set_facecolor(fs.SURFACE)
    ax.set_title("(f) 역할 경계 — 누가 무엇을 하는가", fontsize=11)

    fig.suptitle("cuRobo 의 역할 — run_0004 프레임 10 (grasp).  "
                 "cuRobo 는 depth 를 거리장으로 바꾸는 한 상자다: 의미도 정책도 만들지 않는다",
                 fontsize=14, y=0.965)
    for a in fig.axes:
        if a.name != "3d":
            a.set_facecolor(fs.SURFACE)
    fig.patch.set_facecolor(fs.SURFACE)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=118, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"wrote {args.out}")
    print(f"  점구름 head {len(clouds['head']):,} (raw {len(raw_head):,})  "
          f"좌손목 {len(clouds['left_wrist']):,}  우손목 {len(clouds['right_wrist']):,}")
    print(f"  미세 표면 복셀 {len(pc):,}   단면 z = {czs[k]:.3f} m")


if __name__ == "__main__":
    main()
