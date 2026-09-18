"""발표 슬라이드용 그림 — 모듈 1: Robot Mask (self-filter).

    PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.ppt_robot_mask \\
        --frame /tmp/frame10.npz --field /tmp/field10.npz --field-raw /tmp/field10_raw.npz

문서용 `doc_robot_mask.py` 와 **별도다.** 그쪽은 6 패널 밀집본이고 그대로 유효하다. 여기서는
슬라이드 두 장에 나눠 얹을 수 있게 **그림 하나당 패널 1~3 개, 16:9, 큰 글씨**로 네 장을 낸다.

| 파일 | 슬라이드 | 무엇 |
|---|---|---|
| `ppt-rm-01-scene.png` | 1 | 세 카메라에서 지운 픽셀 — 문제 사진 |
| `ppt-rm-02-io.png`    | 1 | 입력 4 / 처리 / 출력 2 블록도 |
| `ppt-rm-03-method.png`| 2 | 방법론 3 단계 + 20 mm 부풀림의 실제 효과 |
| `ppt-rm-04-beforeafter.png` | 2 | **마스크 X / O 거리장 단면** + 여유거리 곡선 + 큰 수치 |

`--field-raw` 는 같은 프레임을 **마스크 없이** 적분한 대조군이다
(`curobo.build_field --raw`). 04 번 그림이 그 둘을 나란히 놓은 것이고, **이 발표에서 "적용
전/후" 를 한 눈에 보이는 유일한 그림**이다.

수치는 전부 이 스크립트가 그 자리에서 계산해 표준출력으로도 찍는다 — 로그의 C1
(cuRobo 검증 경로가 마스킹 안 된 depth 를 써서 로봇이 자기 몸을 장애물로 봤다) 과 맞춰 볼 것.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures/ppt")
IDS = ("head", "left_wrist", "right_wrist")
LABEL = ("머리 (zed_left)", "왼손목", "오른손목")

INK = "#0b0b0b"
MUTED = "#52514e"
RED = "#e34948"
BLUE = "#2a78d6"
GREEN = "#1baf7a"
AMBER = "#eda100"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if pathlib.Path(p).exists():
            fm.fontManager.addfont(p)
    from benchmark.ag3s.experiments import figstyle
    if not figstyle.use_korean():
        raise SystemExit("CJK 폰트를 못 찾았다 — 한글이 두부로 나온다. Noto CJK 를 설치할 것")
    return figstyle


def _box(ax, x, y, w, h, text, face, *, fontsize=14, edge=INK, text_colour=None, weight="bold"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.18",
                                facecolor=face, edgecolor=edge, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            weight=weight, color=text_colour or INK, linespacing=1.55)


def _arrow(ax, p0, p1, colour=INK, lw=2.0, style="-|>"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=22, color=colour, lw=lw))


def backproject_with(depth, K, T, extra, *, zmax: float = 3.0):
    """depth -> base 프레임 점 `(N,3)` 과, 같은 픽셀에서 뽑은 `extra` 값 `(N,)`.

    `robot_filter.robot_sphere_mask` 로 판정을 **다시 계산하지 않기** 위해 있다. 저장된
    `robot_mask_*` 는 파이프라인이 **전신 194 구 필터 모델**로 만든 것이고, `frame10.npz` 에
    실린 구 120 개는 **제약용 팔 모델**이다. 둘은 다른 집합이라, 구 120 개로 다시 칠하면
    슬라이드가 실제 파이프라인과 다른 그림을 보이게 된다.
    """
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    z = np.asarray(depth, np.float64)
    ok = (z > 1e-6) & (z < zmax)
    u, v, z = uu[ok].astype(np.float64), vv[ok].astype(np.float64), z[ok]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], 1) @ np.asarray(T[:3, :3], np.float64).T + np.asarray(T[:3, 3], np.float64)
    return pts, np.asarray(extra)[ok]


# ----------------------------------------------------------------------------- 01 장면
def fig_scene(fs, fr, out):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16.0, 6.0))
    fig.patch.set_facecolor(fs.SURFACE)
    frac = []
    for ax, cid, lab in zip(axs, IDS, LABEL):
        d = np.asarray(fr[f"depth_{cid}"], float)
        m = np.asarray(fr[f"robot_mask_{cid}"], bool)
        rgb = plt.get_cmap("viridis")(np.clip(d / 2.5, 0, 1))[..., :3]
        rgb[m] = (0.89, 0.19, 0.17)
        ax.imshow(rgb)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d8d7d2")
        f = 100.0 * m.mean()
        frac.append(f)
        ax.set_title(lab, fontsize=17, weight="bold", pad=9)
        ax.set_xlabel(f"지운 픽셀  {f:.1f} %", fontsize=17, weight="bold", color=RED, labelpad=9)

    fig.text(0.5, 0.035,
             "빨강 = 로봇 자기 몸으로 판정해 지운 픽셀.  지우지 않으면 이 팔이 그대로 "
             "장애물 거리장에 적분된다",
             ha="center", fontsize=15, color=MUTED)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.16, wspace=0.06)
    fig.savefig(out, dpi=130, facecolor=fs.SURFACE)
    plt.close(fig)
    return frac


# -------------------------------------------------------------------------------- 02 I/O
def fig_io(fs, frac, n_self_filtered, out):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16.0, 9.0))
    fig.patch.set_facecolor(fs.SURFACE)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

    ax.text(8.0, 8.55, "Robot self-filter — 같은 로봇 판정을 두 소비 경로에 적용",
            ha="center", fontsize=20, weight="bold", color=INK)

    ins = [
        ("depth  (480 x 640) float\n미터 단위 · 카메라 3 대", "#cde2fb"),
        ("K  (pinhole)\nT_base_cam", "#cde2fb"),
        ("robot_state  20 DoF\n그 카메라 촬영 시각의 q", "#9ec5f4"),
        ("URDF collision spheres\nself-filter model: 194 spheres", "#cde2fb"),
    ]
    y = 6.85
    for text, face in ins:
        _box(ax, 0.45, y, 4.3, 1.24, text, face, fontsize=14, weight="normal")
        _arrow(ax, (4.85, y + 0.62), (5.75, 4.55), MUTED, lw=1.5)
        y -= 1.55

    _box(ax, 5.85, 3.55, 4.3, 2.0,
         "Robot self-filter\n\nrobot_sphere_mask()\n점이 로봇 구 안인지 판정",
         "#1baf7a", fontsize=15.5, text_colour="#ffffff")
    ax.text(8.0, 3.25, "부풀림 self_filter_inflation = 20 mm",
            ha="center", fontsize=14, weight="bold", color=GREEN)

    outs = [
        ("출력 1 — cuRobo depth 경로\n(H, W) robot_mask\n→ TSDF 적분에서 해당 ray 제외\n"
         f"머리 {frac[0]:.1f} %  ·  손목 {frac[1]:.1f} / {frac[2]:.1f} %", "#f7d9a0"),
        ("출력 2 — AG3S point-cloud 경로\nrobot-free PointCloud + stats\n→ plane · attention · grounding · obstacles\n"
         f"이 프레임 {n_self_filtered:,} 점 제거", "#f7d9a0"),
    ]
    y = 5.60
    for text, face in outs:
        _arrow(ax, (10.25, 4.55), (10.85, y + 0.78), MUTED, lw=1.5)
        _box(ax, 10.95, y, 4.75, 1.56, text, face, fontsize=13.5, weight="normal")
        y -= 2.10

    ax.text(13.32, 3.10,
            "출력 둘은 중복이 아니다 (E7)\n"
            "ESDF 는 raw depth 를, 클라우드는 다운샘플된 점을 먹는다\n"
            "해상도가 달라 재사용하면 로봇 픽셀 89.3 % 를 놓친다",
            ha="center", va="top", fontsize=12.5, color=BLUE, linespacing=1.65)

    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((0.45, 0.35), 15.15, 1.35,
                                boxstyle="round,pad=0.10,rounding_size=0.18",
                                facecolor="#ecebe7", edgecolor="#d8d7d2", lw=1.2))
    ax.text(8.0, 1.02,
            "지운 픽셀은 '자유' 가 아니라 '미관측' 이다 — TSDF 도 cuRobo 도 그 광선을 통째로 건너뛴다.\n"
            "What is behind the arm this frame is not something the camera saw.",
            ha="center", va="center", fontsize=14.5, color=INK, linespacing=1.7)

    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fs.SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------- 03 방법론
def fig_method(fs, fr, out):
    import matplotlib.pyplot as plt
    from benchmark.ag3s.experiments.doc_curobo_role import backproject

    fig = plt.figure(figsize=(16.0, 9.0))
    fig.patch.set_facecolor(fs.SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.10,
                          left=0.03, right=0.97, top=0.88, bottom=0.26)

    # --- 왼쪽: 3 단계 도식
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(1.55, 9)
    ax.text(5.0, 8.45, "학습된 세그멘테이션이 아니다 — 로봇이 이미 아는 자기 자세로 지운다",
            ha="center", fontsize=16.5, weight="bold", color=INK)

    _box(ax, 0.3, 6.35, 3.5, 1.30, "q  (촬영 시각)\n+ 미리 만든 sphere model", "#cde2fb", fontsize=14, weight="normal")
    _arrow(ax, (3.9, 7.0), (5.5, 7.0))
    ax.text(4.7, 7.24, "FK", ha="center", fontsize=13.5, weight="bold", color=MUTED)
    _box(ax, 5.6, 6.35, 4.1, 1.30, "194개 구의 현재 위치\nc_i(q) + (고정 반지름 + 20 mm)", "#9ec5f4",
         fontsize=14.0, weight="normal")
    ax.text(5.0, 5.88,
            "194 = 47 capsules를 한 번 이산화: URDF 17 + 누락 형상 보완 30",
            ha="center", fontsize=11.8, color=MUTED)

    _box(ax, 0.3, 4.15, 3.5, 1.30, "depth  (H, W)", "#cde2fb", fontsize=15, weight="normal")
    _arrow(ax, (3.9, 4.80), (5.5, 4.80))
    ax.text(4.7, 5.04, "back-project", ha="center", fontsize=13.5, weight="bold", color=MUTED)
    _box(ax, 5.6, 4.15, 2.95, 1.30, "점구름\nuv 를 보존한 채로", "#cde2fb",
         fontsize=14.5, weight="normal")

    # 두 갈래가 초록 상자에서 만난다. 점구름 상자를 뚫지 않도록 오른쪽으로 비켜 세운다.
    _arrow(ax, (8.95, 6.30), (8.95, 3.35), GREEN, lw=2.2)
    _arrow(ax, (7.05, 4.10), (7.05, 3.35), GREEN, lw=2.2)
    _box(ax, 3.1, 1.85, 6.6, 1.45,
         "구 안에 든 점 = 로봇  →  그 점의 uv 로 산포  →  (H, W) 마스크",
         GREEN, fontsize=15.5, text_colour="#ffffff")

    ax.text(5.0, 1.05,
            "점당이 아니라 구당 질의 —  같은 출력에 257 ms → 8 ms  (약 30 배)",
            ha="center", fontsize=14.5, weight="bold", color=BLUE)
    ax.text(5.0, 0.40,
            "카메라마다 자기 촬영 시각의 q 를 쓴다.  현재 q 하나를 셋에 쓰면\n"
            "손목 카메라에서 자유공간을 지우면서 팔은 남긴다 — 두 방향으로 동시에 틀린다",
            ha="center", va="center", fontsize=13.5, color=MUTED, linespacing=1.6)

    # --- 오른쪽: 20 mm 부풀림의 실제 효과 (실측 점으로)
    #
    # 점 색은 **전체 마스크의 최종 판정**이다 — 구 하나만 보면 "이 구 밖" 을 "씬" 이라고
    # 잘못 읽게 된다. 실제로 처음 고른 큰 어깨 구에서는 그 밖의 점이 100 % 다른 구에
    # 지워지고 있었다. 그래서 색은 전체 판정으로 칠하고, 원 둘만 이 구의 것으로 그린다.
    ax = fig.add_subplot(gs[0, 1])
    centres = np.asarray(fr["sphere_centers"], float)
    radii = np.asarray(fr["sphere_radii"], float)
    infl = 0.02
    # **판정은 저장된 진짜 마스크에서 읽는다** — 구 120 개로 다시 계산하지 않는다 (위 helper 참고).
    cloud, deleted = backproject_with(fr["depth_head"], fr["K_head"], fr["T_head"],
                                      np.asarray(fr["robot_mask_head"], bool))

    from scipy.spatial import cKDTree
    ts, tr = cKDTree(cloud[~deleted]), cKDTree(cloud[deleted])

    # 로봇과 씬이 **둘 다 몰려 있는** 구 = 경계가 실제로 갈리는 자리. 손끝 구는 카메라가
    # 스치듯 보아 관측점이 성기므로, 그림은 조밀한 구로 그리고 손끝 비율은 숫자로 적는다.
    score = [min(len(ts.query_ball_point(c_, r_ + 0.06)),
                 len(tr.query_ball_point(c_, r_ + 0.06)))
             for c_, r_ in zip(centres, radii)]
    i = int(np.argmax(score))
    c0, r0 = centres[i], radii[i]
    r_tip = float(radii.min())          # 가장 작은 구 = 손끝

    span = r0 + infl + 0.032
    slab = max(0.008, 0.16 * r0)
    idx = np.asarray(cKDTree(cloud).query_ball_point(c0, span), int)
    near, keep = cloud[idx], ~deleted[idx]
    sel = np.abs(near[:, 1] - c0[1]) < slab
    for mask_, colour, lab in ((sel & ~keep, RED, "마스크가 지운 점"),
                               (sel & keep, BLUE, "살아남아 적분되는 점")):
        q = near[mask_]
        ax.scatter((q[:, 0] - c0[0]) * 1000, (q[:, 2] - c0[2]) * 1000,
                   s=9, color=colour, linewidths=0, label=lab)

    ax.add_patch(plt.Circle((0, 0), r0 * 1000, fill=False, lw=2.2, color=INK))
    ax.add_patch(plt.Circle((0, 0), (r0 + infl) * 1000, fill=False, lw=2.2, ls="--", color=AMBER))
    ax.annotate(f"r = {r0*1000:.0f} mm", xy=(0, r0 * 1000), xytext=(-span*1000*0.94, r0*1000 + 16),
                fontsize=13, weight="bold", color=INK,
                arrowprops=dict(arrowstyle="-", lw=1.4, color=INK))
    ax.annotate(f"r + 20 = {(r0+infl)*1000:.0f} mm", xy=(0, (r0 + infl) * 1000),
                xytext=(-span*1000*0.94, (r0+infl)*1000 + 15),
                fontsize=13, weight="bold", color=AMBER,
                arrowprops=dict(arrowstyle="-", lw=1.4, color=AMBER))

    lim = span * 1000
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_facecolor(fs.SURFACE)
    ax.tick_params(labelsize=12, colors=MUTED)
    ax.set_xlabel("구 중심에서 [mm]", fontsize=13.5)
    ax.legend(fontsize=12.5, loc="lower right", framealpha=0.92)
    ax.set_title("20 mm 부풀림이 실제로 어디를 지우는가\n"
                 "점 색 = 파이프라인이 쓴 진짜 마스크 · 원 둘 = 구 하나의 r 과 r+20",
                 fontsize=15, weight="bold", pad=10)
    fig.text(0.79, 0.075,
             f"덜 지우면 그리퍼에 유령 장애물이 용접되고, 더 지우면 진짜 장애물을 놓친다.\n"
             f"이 구는 r {r0*1000:.0f} mm 라 삭제 반경이 {(r0+infl)/r0:.2f} 배지만, "
             f"손끝 구(r {r_tip*1000:.0f} mm)에서는 {(r_tip+infl)/r_tip:.2f} 배다\n"
             f"— 손끝에서 이 비율이 커지는 것이 F12(쥔 사과가 100 % 지워진다)의 기하학적 원인이다",
             ha="center", fontsize=12.5, color=MUTED, linespacing=1.65)

    fig.savefig(out, dpi=130, facecolor=fs.SURFACE)
    plt.close(fig)
    return i, r0, float((r_tip + infl) / r_tip)


# ------------------------------------------------------------------------ 04 적용 전/후
def fig_before_after(fs, fr, fd_ok, fd_bad, margin, out):
    import matplotlib.pyplot as plt
    from benchmark.ag3s.curobo_field import CuroboEsdfField, layer_from_arrays

    def field(d):
        return CuroboEsdfField(layers=(
            layer_from_arrays(d["coarse_values"], np.asarray(d["coarse_origin"], float),
                              float(d["coarse_voxel_size"])),
            layer_from_arrays(d["fine_values"], np.asarray(d["fine_origin"], float),
                              float(d["fine_voxel_size"]))))

    centres = np.asarray(fd_ok["sphere_centers"], float)
    radii = np.asarray(fd_ok["sphere_radii"], float)
    tc = np.asarray(fd_ok["target_centroid"], float)
    c_ok = field(fd_ok).distance(centres) - radii - margin
    c_bad = field(fd_bad).distance(centres) - radii - margin

    fig = plt.figure(figsize=(16.0, 9.0))
    fig.patch.set_facecolor(fs.SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.28, 1.0], width_ratios=[1.0, 1.0],
                          hspace=0.34, wspace=0.16, left=0.055, right=0.965,
                          top=0.84, bottom=0.075)

    co = np.asarray(fd_ok["coarse_origin"], float)
    cvs = float(fd_ok["coarse_voxel_size"])
    shape = fd_ok["coarse_values"].shape
    xs = co[0] + np.arange(shape[0]) * cvs
    zs = co[2] + np.arange(shape[2]) * cvs
    ys = co[1] + np.arange(shape[1]) * cvs
    j = int(np.argmin(np.abs(ys - tc[1])))
    near = np.abs(centres[:, 1] - ys[j]) < 0.08

    slice_y = float(ys[j])
    panels = [(fd_bad, "Self-filter OFF — 로봇 팔까지 표면으로 적분", RED),
              (fd_ok, "Self-filter ON — 로봇 점을 제거한 뒤 적분", BLUE)]
    for col, (d, title, colour) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        sl = np.asarray(d["coarse_values"], float)[:, j, :]
        im = ax.pcolormesh(xs, zs, np.clip(sl, -0.05, 0.30).T, cmap="RdYlBu",
                           shading="auto", vmin=-0.05, vmax=0.30)
        ax.contour(xs, zs, sl.T, levels=[0.0], colors="k", linewidths=1.8)
        for c, r in zip(centres[near], radii[near]):
            ax.add_patch(plt.Circle((c[0], c[2]), r, fill=False, lw=1.2, color=INK))
        ax.set_xlim(0.05, 1.02); ax.set_ylim(0.55, 1.32)
        ax.set_aspect("equal")
        ax.set_facecolor(fs.SURFACE)
        ax.tick_params(labelsize=12, colors=MUTED)
        ax.set_xlabel("x [m] — 앞 →", fontsize=13.5)
        if col == 0:
            ax.set_ylabel("z [m] — 위 ↑", fontsize=13.5)
        ax.set_title(title, fontsize=16, weight="bold", color=colour, pad=10)
        ax.plot(tc[0], tc[2], marker="*", ms=15, color="#d62fd6", mec="white",
                mew=0.8, zorder=8)
        if col == 0:
            ax.annotate("팔이 여기 타 있다", xy=(0.36, 1.03), xytext=(0.60, 1.22),
                        fontsize=14.5, weight="bold", color=INK,
                        arrowprops=dict(arrowstyle="-|>", lw=2.0, color=INK))
        if col == 1:
            cb = plt.colorbar(im, ax=ax, fraction=0.037, pad=0.02)
            cb.set_label("표면까지 거리 d [m]", fontsize=12.5)
            cb.ax.tick_params(labelsize=11)

    # 아래 왼쪽 — 여유거리 곡선
    ax = fig.add_subplot(gs[1, 0])
    o = np.argsort(c_ok)
    k = np.arange(len(o))
    ax.plot(k, c_bad[o] * 1000, lw=2.6, color=RED, label="마스크 없음")
    ax.plot(k, c_ok[o] * 1000, lw=2.6, color=BLUE, label="마스크 있음")
    ax.fill_between(k, c_bad[o] * 1000, c_ok[o] * 1000, color=RED, alpha=0.13)
    ax.axhline(0, color=INK, lw=1.4, ls="--")
    ax.set_xlabel(f"로봇 구 {len(centres)} 개 (여유거리 오름차순)", fontsize=13.5)
    ax.set_ylabel("여유거리 [mm]", fontsize=13.5)
    ax.set_facecolor(fs.SURFACE)
    ax.tick_params(labelsize=12, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=13)
    ax.set_title("여유거리 = d − r − margin   (0 미만이 violated)",
                 fontsize=14.5, weight="bold", pad=8)

    # 아래 오른쪽 — 큰 숫자
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((0.15, 0.4), 9.7, 9.2,
                                boxstyle="round,pad=0.12,rounding_size=0.25",
                                facecolor="#ecebe7", edgecolor="#d8d7d2", lw=1.3))
    ax.text(5.0, 9.0, "같은 프레임 · 같은 구 · 같은 cuRobo 설정, 마스크만 뺀 대조군",
            ha="center", fontsize=13.5, color=MUTED)
    ax.text(2.1, 7.75, "마스크 없음", ha="center", fontsize=15, weight="bold", color=RED)
    ax.text(7.7, 7.75, "마스크 있음", ha="center", fontsize=15, weight="bold", color=BLUE)

    ax.text(5.0, 6.15, "violated 로봇 구", ha="center", fontsize=13.5, color=MUTED)
    ax.text(2.1, 4.95, f"{int((c_bad<0).sum())} / {len(centres)}", ha="center",
            fontsize=34, weight="bold", color=RED)
    ax.text(7.7, 4.95, f"{int((c_ok<0).sum())} / {len(centres)}", ha="center",
            fontsize=34, weight="bold", color=BLUE)
    _arrow(ax, (3.9, 5.25), (5.9, 5.25), INK, lw=2.4)

    ax.text(5.0, 3.35, "최악 여유거리", ha="center", fontsize=13.5, color=MUTED)
    ax.text(2.1, 2.25, f"{c_bad.min()*1000:+.1f} mm", ha="center",
            fontsize=25, weight="bold", color=RED)
    ax.text(7.7, 2.25, f"{c_ok.min()*1000:+.1f} mm", ha="center",
            fontsize=25, weight="bold", color=BLUE)
    _arrow(ax, (3.9, 2.55), (5.9, 2.55), INK, lw=2.4)

    ax.text(5.0, 1.15,
            "발견 C1 — cuRobo 검증 경로가 마스킹 안 된 depth 를 썼다",
            ha="center", fontsize=13, weight="bold", color=INK)

    fig.suptitle(f"같은 장면 A/B 비교 — run_0004 프레임 10, margin {margin*1000:.0f} mm",
                 fontsize=19, weight="bold", y=0.975)
    fig.text(0.5, 0.925,
             f"위 두 그림은 카메라 영상이 아니라 robot base frame의 같은 x-z 수직 단면"
             f" (y = {slice_y:.3f} m)  ·  x: 로봇 앞  ·  z: 위  ·  ★: 절단 기준 물체 중심",
             ha="center", va="center", fontsize=13.5, color=MUTED)
    fig.savefig(out, dpi=130, facecolor=fs.SURFACE)
    plt.close(fig)
    return c_ok, c_bad


def main() -> None:
    fs = _style()

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frame", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--field-raw", required=True)
    ap.add_argument("--margin", type=float, default=0.05, help="esdf_margin [m]")
    ap.add_argument("--n-self-filtered", type=int, default=6445,
                    help="융합 경로에서 self-filter 가 지운 점 수 (doc_attention_stages 실측)")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fr = np.load(args.frame)
    fd_ok, fd_bad = np.load(args.field), np.load(args.field_raw)

    frac = fig_scene(fs, fr, out / "ppt-rm-01-scene.png")
    fig_io(fs, frac, args.n_self_filtered, out / "ppt-rm-02-io.png")
    i, r0, ratio = fig_method(fs, {**{k: fr[k] for k in fr.files},
                                   "sphere_centers": fd_ok["sphere_centers"],
                                   "sphere_radii": fd_ok["sphere_radii"]},
                              out / "ppt-rm-03-method.png")
    c_ok, c_bad = fig_before_after(fs, fr, fd_ok, fd_bad, args.margin,
                                   out / "ppt-rm-04-beforeafter.png")

    print(f"wrote 4 figures -> {out}")
    print(f"  삭제 비율     머리 {frac[0]:.2f} %  왼손목 {frac[1]:.2f} %  오른손목 {frac[2]:.2f} %")
    print(f"  violated 구   마스크 없음 {int((c_bad<0).sum())}  /  있음 {int((c_ok<0).sum())}"
          f"  (전체 {len(c_ok)})")
    print(f"  최악 여유거리  마스크 없음 {c_bad.min()*1000:+.2f} mm  /  "
          f"있음 {c_ok.min()*1000:+.2f} mm")
    print(f"  부풀림 예시 구  #{i}, r = {r0*1000:.1f} mm  -> 삭제 반경 "
          f"{(r0+0.02)*1000:.1f} mm ({(r0+0.02)/r0:.2f} 배)   손끝 구는 {ratio:.2f} 배")


if __name__ == "__main__":
    main()
