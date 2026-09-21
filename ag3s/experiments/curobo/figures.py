"""어댑터 검증의 시각화. **ag3s venv 에서** 돌린다.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.curobo.figures

`AG3S_REVIEW_PLAN.md` 의 **규칙 A**(2026-09-12): 모든 스텝은 시각화 자료를 남긴다. 우선순위는
실제 씬 > 그래프 > 표. 이 스크립트가 그 첫 번째를 만든다 — 실제 MuJoCo 프레임에서 세 필드의
단면을 같은 좌표에 겹쳐 그린다.

만드는 것:
  fig1  실제 씬 — depth · 로봇 마스크 · 세 필드의 수직 단면 · 미세 계층이 조이는 곳
  fig2  15 청크 추이 — 세 백엔드의 before/after 와 단계별 조임
"""

import argparse
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(path).exists():
            fm.fontManager.addfont(path)
    from benchmark.ag3s.experiments.common import figstyle
    if not figstyle.use_korean():
        print("경고: 한글 폰트를 못 찾았다 — 라벨이 깨진다")


def _slice(field, y, xs, zs):
    """`(len(zs), len(xs))` — y 평면에서의 거리장 단면."""
    X, Z = np.meshgrid(xs, zs)
    pts = np.stack([X.ravel(), np.full(X.size, y), Z.ravel()], axis=1)
    return np.asarray(field.distance(pts), float).reshape(X.shape)


def fig1(args) -> None:
    """실제 씬 — **어디를 자른 것인지**를 먼저 보여주고, 직교하는 두 단면을 나란히 놓는다.

    (b)~(f) 는 카메라 사진이 아니라 **공간을 자른 단면**이다. 그 사실이 안 보여서 읽기 어렵다는
    지적을 받아 (a) 배치도와 시점 표시를 넣었다 (2026-09-12).

    좌표계는 로봇 base 프레임: **x = 로봇 앞쪽, y = 로봇 왼쪽, z = 위쪽.**
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.fields.curobo_field import CuroboEsdfField, RolloutFields
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import gaussian_attention
    from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

    k = args.frame
    run = load_run(args.records, limit=k + 1)
    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    pose_scene(scene, run.steps[k])
    head = scene.capture("zed_left")

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf", "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": 0.020, "max_distance": 0.4,
                 "exclude_support_surfaces": False}})
    ag = AG3S(cfg, robot_model=filter_robot, constraint_robot_model=robot)
    cs = ag.process(depth=head.depth, camera_intrinsics=head.camera_intrinsics,
                    T_base_cam=head.T_base_cam,
                    attention_map=gaussian_attention(
                        head, scene.body_position_in_base(args.target)),
                    robot_state=head.robot_state, phase="grasp")
    mask = ag._robot_mask_for(np.asarray(head.depth, float),
                              np.asarray(head.camera_intrinsics, float),
                              np.asarray(head.T_base_cam, float), head.robot_state)

    two = RolloutFields.load(args.fields).field_for(k)
    coarse = CuroboEsdfField((two.layers[0],), outside_distance=0.5)
    tc = np.asarray(cs.target.centroid, float)
    q = np.asarray(head.robot_state, float)
    centres, radii = robot.sphere_centers_numeric(q)
    centres = np.asarray(centres, float).reshape(-1, 3)
    radii = np.asarray(radii, float).reshape(-1)

    fg = two.layers[1].grid
    flo = fg.origin
    fhi = fg.origin + (np.asarray(fg.shape) - 1) * fg.voxel_size

    xs = np.arange(0.05, 1.05, 0.004)
    zs = np.arange(0.45, 1.45, 0.004)
    ys = np.arange(-0.60, 0.62, 0.004)
    v_co = _slice(coarse, tc[1], xs, zs)
    v_tw = _slice(two, tc[1], xs, zs)

    def _hslice(field, z):
        X, Y = np.meshgrid(xs, ys)
        pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)
        return np.asarray(field.distance(pts), float).reshape(X.shape)
    h_co, h_tw = _hslice(coarse, tc[2]), _hslice(two, tc[2])

    fig, axs = plt.subplots(2, 3, figsize=(19.5, 11.0))
    ax = axs.ravel()
    lvl = np.linspace(-0.05, 0.35, 41)

    def _spheres(a, axis):
        """단면에 걸리는 로봇 구를, 잘린 반지름으로 그린다."""
        for c, r in zip(centres, radii):
            if axis == "xz":
                d = abs(c[1] - tc[1]); u, w = c[0], c[2]
            else:
                d = abs(c[2] - tc[2]); u, w = c[0], c[1]
            if d < r:
                a.add_patch(Circle((u, w), np.sqrt(r * r - d * d), fill=False,
                                   ec="k", lw=0.8, alpha=0.85))

    # ---- (a) 배치도: 어디를 어느 방향에서 자른 것인가 -------------------------
    a = ax[0]
    a.add_patch(Rectangle((0.40, -0.49), 0.48, 0.98, fc="#d9c9a3", ec="#8a7a55",
                          lw=1.5, alpha=0.85))
    a.text(0.64, -0.42, "테이블 상판", ha="center", fontsize=10, color="#5c4f33")
    a.plot(0, 0, "ks", ms=13); a.text(0.015, -0.075, "로봇 base\n(원점)", fontsize=9.5)
    a.plot(tc[0], tc[1], "m*", ms=20, mec="k")
    a.annotate("사과 (target)", xy=(tc[0], tc[1]), xytext=(tc[0] + 0.10, tc[1] + 0.20),
               fontsize=10, color="m",
               arrowprops=dict(arrowstyle="-|>", color="m", lw=1.1))
    a.scatter(centres[:, 0], centres[:, 1], s=9, c="tab:blue", alpha=0.55, label="로봇 구 120개")
    a.add_patch(Rectangle((flo[0], flo[1]), fhi[0] - flo[0], fhi[1] - flo[1],
                          fill=False, ec="limegreen", lw=2.0, ls="--"))
    a.text(flo[0] + 0.02, flo[1] + 0.03, "미세 5 mm 창", color="green", fontsize=9.5)
    a.axhline(tc[1], color="crimson", lw=2.4)
    a.text(1.03, tc[1] - 0.10, f"자르는 면  y = {tc[1]:.3f} m", color="crimson",
           fontsize=10, ha="right", fontweight="bold")
    for xx in (0.15, 0.45, 0.75):
        a.annotate("", xy=(xx, tc[1] - 0.05), xytext=(xx, -0.60),
                   arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.4, alpha=0.8))
    a.text(0.45, -0.71, "화살표 = 보는 방향 (로봇 오른쪽에서 왼쪽으로)", color="crimson",
           fontsize=10.5, ha="center", fontweight="bold")
    a.set_xlim(-0.15, 1.05); a.set_ylim(-0.80, 0.70); a.set_aspect("equal")
    a.set_xlabel("x [m]  — 로봇 앞쪽 →"); a.set_ylabel("y [m]  — 로봇 왼쪽 ↑")
    a.set_title("(a) 어디를 자른 것인가 — 위에서 내려다본 배치도", fontsize=11)
    a.legend(loc="upper left", fontsize=8.5); a.grid(alpha=0.25)

    # ---- (b) 실제 카메라 입력 -------------------------------------------------
    d = np.asarray(head.depth, float)
    ax[1].imshow(np.where(d > 2.5, np.nan, d), cmap="viridis")
    if mask is not None:
        ov = np.zeros((*mask.shape, 4)); ov[mask] = (1, 0, 0, 0.55)
        ax[1].imshow(ov)
    ax[1].set_title(f"(b) 이것만 카메라 사진이다 — head depth\n"
                    f"빨강 = 로봇 마스크 ({100.0*mask.mean():.1f}%, 발견 C1)", fontsize=11)
    ax[1].axis("off")

    # ---- (c)(d) 수직 단면 -----------------------------------------------------
    for i, (a, v, name) in enumerate(((ax[2], v_co, "(c) cuRobo 거친 20 mm"),
                                      (ax[3], v_tw, "(d) cuRobo 2계층 20+5 mm"))):
        cf = a.contourf(xs, zs, v, levels=lvl, cmap="RdYlBu", extend="both")
        a.contour(xs, zs, v, levels=[0.0], colors="k", linewidths=1.8)
        _spheres(a, "xz")
        a.plot(tc[0], tc[2], "m*", ms=18, mec="k")
        if i == 0:
            a.annotate("테이블 상판\n(검은 선 = 표면)", xy=(0.78, 0.823), xytext=(0.62, 0.60),
                       fontsize=10, arrowprops=dict(arrowstyle="-|>", lw=1.2))
            a.annotate("로봇 팔 (검은 원 = 제약 구)", xy=(0.33, 1.06), xytext=(0.10, 1.34),
                       fontsize=10, arrowprops=dict(arrowstyle="-|>", lw=1.2))
            a.annotate("사과", xy=(tc[0], tc[2]), xytext=(tc[0] + 0.13, 1.02),
                       fontsize=10, color="m", arrowprops=dict(arrowstyle="-|>", lw=1.2,
                                                               color="m"))
            a.annotate("카메라가 못 본 곳\n= 자유로 답함 (E4 계열)", xy=(0.12, 0.70),
                       xytext=(0.07, 0.50), fontsize=9.5, color="navy")
        else:
            a.add_patch(Rectangle((flo[0], flo[2]), fhi[0] - flo[0], fhi[2] - flo[2],
                                  fill=False, ec="limegreen", lw=2.2, ls="--"))
            a.text(flo[0] + 0.01, fhi[2] - 0.05, "미세 5 mm 창", color="green", fontsize=10)
        a.set_title(f"{name}   —   수직 단면 (y = {tc[1]:.3f} m 로 자름)", fontsize=11)
        a.set_xlabel("x [m]  — 로봇 앞쪽 →"); a.set_ylabel("z [m]  — 위 ↑")
        a.set_aspect("equal")
        plt.colorbar(cf, ax=a, fraction=0.046, label="표면까지 거리 d [m]")

    # ---- (e) 수직 단면에서의 차이 ---------------------------------------------
    diff = (v_tw - v_co) * 1000.0
    m = max(1.0, float(np.nanpercentile(np.abs(diff), 99)))
    cf = ax[4].contourf(xs, zs, diff, levels=np.linspace(-m, m, 41), cmap="PuOr_r", extend="both")
    ax[4].contour(xs, zs, v_co, levels=[0.0], colors="k", linewidths=1.0, alpha=0.5)
    _spheres(ax[4], "xz"); ax[4].plot(tc[0], tc[2], "m*", ms=18, mec="k")
    ax[4].add_patch(Rectangle((flo[0], flo[2]), fhi[0] - flo[0], fhi[2] - flo[2],
                              fill=False, ec="limegreen", lw=2.2, ls="--"))
    ax[4].set_title("(e) 2계층 − 거친  [mm]   —   같은 수직 단면\n"
                    "보라(음) = 미세 계층이 더 가깝다고 답함", fontsize=11)
    ax[4].set_xlabel("x [m]  — 로봇 앞쪽 →"); ax[4].set_ylabel("z [m]  — 위 ↑")
    ax[4].set_aspect("equal")
    plt.colorbar(cf, ax=ax[4], fraction=0.046, label="mm")

    # ---- (f) 수평 단면에서의 차이 (직교하는 두 번째 칼) ------------------------
    dh = (h_tw - h_co) * 1000.0
    m2 = max(1.0, float(np.nanpercentile(np.abs(dh), 99)))
    cf = ax[5].contourf(xs, ys, dh, levels=np.linspace(-m2, m2, 41), cmap="PuOr_r", extend="both")
    ax[5].contour(xs, ys, h_co, levels=[0.0], colors="k", linewidths=1.0, alpha=0.5)
    _spheres(ax[5], "xy"); ax[5].plot(tc[0], tc[1], "m*", ms=18, mec="k")
    ax[5].add_patch(Rectangle((flo[0], flo[1]), fhi[0] - flo[0], fhi[1] - flo[1],
                              fill=False, ec="limegreen", lw=2.2, ls="--"))
    ax[5].axhline(tc[1], color="crimson", lw=1.6, ls=":")
    ax[5].text(0.07, tc[1] + 0.015, "(c)(d)(e) 가 자른 면", color="crimson", fontsize=9)
    ax[5].set_title(f"(f) 2계층 − 거친  [mm]   —   수평 단면 (z = {tc[2]:.3f} m 로 자름)\n"
                    "(a) 와 같은 방향: 위에서 내려다본 것", fontsize=11)
    ax[5].set_xlabel("x [m]  — 로봇 앞쪽 →"); ax[5].set_ylabel("y [m]  — 로봇 왼쪽 ↑")
    ax[5].set_aspect("equal")
    plt.colorbar(cf, ax=ax[5], fraction=0.046, label="mm")

    fig.suptitle(f"실제 씬 — run_0004 프레임 {k} (grasp 단계).  "
                 f"(b) 만 카메라 사진이고 나머지는 공간을 자른 단면이다.  "
                 f"좌표계: x = 앞쪽, y = 왼쪽, z = 위쪽 (로봇 base 프레임)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "curobo-adapter-scene.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print(f"wrote {out}")
    scene.close()


def fig2(args) -> None:
    import json
    import matplotlib.pyplot as plt

    runs = [("AG3S numpy 20 mm", args.base, "tab:blue", "o"),
            ("cuRobo 거친 20 mm", args.coarse_json, "tab:green", "s"),
            ("cuRobo 2계층 20+5", args.two_json, "tab:red", "^")]
    data = {n: json.load(open(p))["frames"] for n, p, _, _ in runs}
    n = len(next(iter(data.values())))
    idx = np.arange(n)
    phases = [r["phase"] for r in next(iter(data.values()))]

    fig, ax = plt.subplots(1, 3, figsize=(19, 4.6))
    for name, _, col, mk in runs:
        r = data[name]
        ax[0].plot(idx, [x["clearance_before_mm"] for x in r], mk + "-", color=col,
                   label=name, ms=5)
        ax[1].plot(idx, [x["clearance_after_mm"] for x in r], mk + "-", color=col,
                   label=name, ms=5)
    ax[0].set_title("(a) 최적화 **전** 최악 여유거리", fontsize=11)
    ax[1].set_title("(b) 최적화 **후** 최악 여유거리\n0 선을 사이에 두고 상태가 갈린다",
                    fontsize=11)
    ax[1].axhline(0, color="k", lw=1.2)
    ax[1].axhspan(-8, 0, color="red", alpha=0.07)
    ax[1].axhspan(0, 8, color="green", alpha=0.07)
    ax[1].set_ylim(-8, 8)
    for a in ax[:2]:
        a.set_xlabel("청크"); a.set_ylabel("mm"); a.grid(alpha=0.3); a.legend(fontsize=8)

    # (c) 단계별 조임
    two = data["cuRobo 2계층 20+5"]; co = data["cuRobo 거친 20 mm"]
    tighten = np.array([t["clearance_after_mm"] - c["clearance_after_mm"]
                        for t, c in zip(two, co)])
    order = ["transit", "approach", "pre_grasp", "grasp"]
    med = [np.median(tighten[[i for i, p in enumerate(phases) if p == ph]])
           if any(p == ph for p in phases) else np.nan for ph in order]
    bars = ax[2].bar(order, med, color=["#bbb", "#8ecae6", "#219ebc", "#023047"])
    for b, v in zip(bars, med):
        ax[2].text(b.get_x() + b.get_width() / 2, v / 2, f"{v:+.1f} mm",
                   ha="center", va="center", fontsize=11,
                   color="w" if abs(v) > 4 else "k", fontweight="bold")
    ax[2].axhline(0, color="k", lw=1)
    ax[2].set_title("(c) 미세 계층이 조이는 양 (2계층 − 거친)\n"
                    "설계가 노린 자리(grasp)에서 가장 크다", fontsize=11)
    ax[2].set_ylabel("mm"); ax[2].grid(alpha=0.3, axis="y")

    fig.suptitle("15 청크 end-to-end — 필드만 바꾸고 나머지 배선은 동일", fontsize=12)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "curobo-adapter-rollout.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


def fig3(args) -> None:
    """전체 틀에서 지금 어디인가 (규칙 B). 데이터가 아니라 지도다."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    DONE, NOW, TODO = "#2a9d8f", "#e9c46a", "#e5e5e5"
    stages = [
        ("카메라 depth\n롤아웃 1대 / 검증 3대", "AG3S", DONE, "G1 단위\nG3 카메라"),
        ("로봇 마스크\n(self-filter)", "AG3S", DONE, "C1 확정·수정"),
        ("attention\nlifting", "AG3S", TODO, "Step 5  F2·F9"),
        ("target\ngrounding", "AG3S", TODO, "Step 6  F8"),
        ("TSDF → ESDF\n(2계층)", "cuRobo", DONE, "E4·E6 소멸\nC2·C4"),
        ("거리장 어댑터\ndistance/gradient", "우리", DONE, "C3 미판정"),
        ("SQP 선형화\nd−r−margin≥0", "trajopt", DONE, "E1 반영\nE3 미배선"),
        ("QP 해\n(OSQP)", "trajopt", DONE, "반복 1회"),
    ]
    fig, ax = plt.subplots(figsize=(19, 5.4))
    w, h, gap = 2.0, 1.15, 0.42
    for i, (label, owner, col, note) in enumerate(stages):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.6), w, h, boxstyle="round,pad=0.06",
                                    fc=col, ec="k", lw=1.3))
        ax.text(x + w / 2, 2.18, label, ha="center", va="center", fontsize=10.5,
                fontweight="bold")
        ax.text(x + w / 2, 1.76, owner, ha="center", va="center", fontsize=8.5,
                style="italic", color="#333")
        ax.text(x + w / 2, 1.32, note, ha="center", va="top", fontsize=8.5, color="#444")
        if i:
            ax.add_patch(FancyArrowPatch((x - gap, 2.17), (x - 0.04, 2.17),
                                         arrowstyle="-|>", mutation_scale=14, lw=1.3))
    total = len(stages) * (w + gap) - gap
    ax.annotate("", xy=(4 * (w + gap) - gap / 2, 3.15), xytext=(0, 3.15),
                arrowprops=dict(arrowstyle="<->", color="#264653", lw=1.6))
    ax.text(2 * (w + gap), 3.28, "지각 — 무엇이 어디에 있는가", ha="center",
            fontsize=11, color="#264653", fontweight="bold")
    ax.annotate("", xy=(total, 3.15), xytext=(4 * (w + gap) - gap / 2, 3.15),
                arrowprops=dict(arrowstyle="<->", color="#7f5539", lw=1.6))
    ax.text((4 * (w + gap) + total) / 2, 3.28, "기하 → 제약 → 해", ha="center",
            fontsize=11, color="#7f5539", fontweight="bold")
    for j, (c, t) in enumerate(((DONE, "닫힘 (수치로 검증)"), (TODO, "미검토"))):
        ax.add_patch(FancyBboxPatch((j * 5.0, 0.15), 0.5, 0.32,
                                    boxstyle="round,pad=0.03", fc=c, ec="k"))
        ax.text(j * 5.0 + 0.65, 0.31, t, va="center", fontsize=10)
    ax.text(11.0, 0.31, "다음 1순위: MuJoCo 참 거리로 미세 계층 −7.8 mm 가 교정인지 판정",
            va="center", fontsize=10.5, color="#9b2226", fontweight="bold")
    ax.set_xlim(-0.4, total + 0.4); ax.set_ylim(0, 3.7); ax.axis("off")
    ax.set_title("전체 틀에서 지금 어디인가 — 2026-09-12", fontsize=13, pad=2)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "curobo-adapter-map.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--frame", type=int, default=10)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--fields", default="/tmp/rollout_fields.npz")
    ap.add_argument("--base", default="/tmp/base.json")
    ap.add_argument("--coarse-json", default="/tmp/curobo1.json")
    ap.add_argument("--two-json", default="/tmp/curobo.json")
    ap.add_argument("--only", choices=("1", "2", "3"), default=None)
    args = ap.parse_args()
    _style()
    if args.only in (None, "2"):
        fig2(args)
    if args.only in (None, "3"):
        fig3(args)
    if args.only in (None, "1"):
        fig1(args)


if __name__ == "__main__":
    main()
