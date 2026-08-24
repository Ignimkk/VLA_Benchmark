"""Draw the ellipsoid representation over the geometry it is standing in for.

Every number in docs/14 comes from ellipsoids fitted to meshes, and an ellipsoid that looks
reasonable in a table can still be obviously wrong in space -- the crate's 36 cm semi-axis read as
a plausible height for two revisions before a plot would have shown it pointing sideways. So the
figures here always draw the source point cloud underneath the ellipsoid: the gap between them is
the approximation error the safety filter actually pays for.

    src/openpi/.venv/bin/python -m benchmark.knows_vla.visualize_ellipsoids --out data/knows_figs
"""

from __future__ import annotations

import argparse
import glob
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark.knows_vla.cbf.ellipsoid import Ellipsoid, barrier, optimal_normal  # noqa: E402
from benchmark.knows_vla.check_admissibility import (  # noqa: E402
    CRATE_SHELL,
    _obj_vertices,
    _parse_mjcf,
    axis_extent,
    block_gripper,
    collect_bodies,
    fit_ellipsoid,
    rby1_gripper,
)

MJCF_ROOT = pathlib.Path("src/rby1_description/models/rby1a/mujoco")
FRUIT = ("apple", "banana", "orange", "pear")

# Set from --dpi. The 3D panels scatter thousands of mesh points, so a figure committed to docs/
# is rendered lower than one being read on screen.
DPI = 150

# Fruit warm, crate cool, robot red -- the target/obstacle/robot split the filter itself makes.
COLOURS = {
    "apple": "#c0392b", "banana": "#d4ac0d", "orange": "#e67e22", "pear": "#7d9f35",
    "crate": "#8b6f47", "hand": "#2471a3", "finger_p": "#5dade2", "finger_n": "#5dade2",
    "gripper": "#2471a3",
}


def _colour(name: str) -> str:
    return COLOURS.get(name, COLOURS.get(name.split("_")[0], "#7f8c8d"))


def ellipsoid_surface(E: Ellipsoid, n: int = 40):
    """Points on the ellipsoid boundary: c + Q^(1/2) u over the unit sphere."""
    w, V = np.linalg.eigh(E.Q)
    A = V @ np.diag(np.sqrt(np.maximum(w, 0.0))) @ V.T
    u, v = np.mgrid[0 : 2 * np.pi : n * 2j, 0 : np.pi : n * 1j]
    S = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], -1)
    P = S.reshape(-1, 3) @ A.T + E.c
    return [P[:, i].reshape(S.shape[:2]) for i in range(3)]


def draw(ax, E: Ellipsoid, colour: str, label=None, alpha=0.18, wire=True):
    X, Y, Z = ellipsoid_surface(E)
    ax.plot_surface(X, Y, Z, color=colour, alpha=alpha, linewidth=0, shade=True)
    if wire:
        ax.plot_wireframe(X, Y, Z, color=colour, alpha=0.35, rstride=8, cstride=6, linewidth=0.5)
    if label:
        ax.text(*(E.c + [0, 0, np.sqrt(E.Q[2, 2]) * 1.15]), label, color=colour,
                fontsize=8, ha="center", weight="bold")


def scatter(ax, pts: np.ndarray, colour: str, stride: int = 1, size=1.4):
    P = pts[::stride]
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=size, c=colour, alpha=0.55, depthshade=False)


def equalise(ax, pts: np.ndarray, pad: float = 1.05):
    """3D axes with a true 1:1:1 aspect -- otherwise a stretched axis fakes a shape the fit never had."""
    lo, hi = pts.min(0), pts.max(0)
    c, r = 0.5 * (lo + hi), 0.5 * (hi - lo).max() * pad
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:  # matplotlib < 3.3
        pass
    for a, lbl in ((ax.set_xlabel, "x [m]"), (ax.set_ylabel, "y [m]"), (ax.set_zlabel, "z [m]")):
        a(lbl, fontsize=8)
    ax.tick_params(labelsize=7)


def load_scene(split=None):
    mf = {m.get("name"): m.get("file")
          for m in _parse_mjcf(MJCF_ROOT / "scenes" / "transport_prop_assets.xml").iter("mesh")
          if m.get("name") and m.get("file")}
    bodies = collect_bodies(MJCF_ROOT / "scenes" / "scene_transport.xml", MJCF_ROOT, mf, split_geoms=split)
    return {n: p for n, p in bodies.items() if n in FRUIT or n.startswith("crate")}


def fig_scene(out: pathlib.Path) -> None:
    """The crate whole vs decomposed, over the same mesh. The place phase lives or dies here."""
    fig = plt.figure(figsize=(13, 6.2))
    for k, (split, title) in enumerate([
        (None, "논문 그대로 — 객체 하나 = 타원체 하나\n크레이트 내부가 메워진다"),
        (CRATE_SHELL, "껍질 분해 — 벽 4 + 바닥 + 손잡이 2\n내부가 실제로 비어 담을 수 있다"),
    ]):
        bodies = load_scene(split)
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for name, pts in bodies.items():
            E = fit_ellipsoid(pts, "mvee")
            scatter(ax, pts, "#2c3e50", stride=max(1, len(pts) // 1200))
            draw(ax, E, _colour(name), label=name if name in FRUIT or split is None else None,
                 alpha=0.13 if name.startswith("crate") else 0.20)
        ax.set_title(title, fontsize=10, pad=2)
        ax.view_init(elev=22, azim=-58)
        equalise(ax, np.vstack(list(bodies.values())))
    fig.suptitle("transport 씬의 타원체 표현 (점 = 실제 메시, 면 = 필터가 보는 형상)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "01-scene-ellipsoids.png", dpi=DPI)
    plt.close(fig)


def fig_gripper(out: pathlib.Path) -> None:
    """Measured RB-Y1 hand against the 4x4x7 block we had assumed."""
    assets = MJCF_ROOT / "assets"
    hand = np.vstack([_obj_vertices(pathlib.Path(f))
                      for f in sorted(glob.glob(str(assets / "EE_BODY/EE_BODY_collision_*.obj")))]) + [0, 0, 0.100]
    fing = np.vstack([_obj_vertices(pathlib.Path(f))
                      for f in sorted(glob.glob(str(assets / "EE_FINGER/EE_FINGER_collision_*.obj")))]) + [0, 0, 0.027]
    parts = rby1_gripper(assets)
    opening = float(parts["finger_p"].c[1] - parts["finger_n"].c[1]) / 2.0
    meshes = np.vstack([hand, fing + [0, opening, 0], fing - [0, opening, 0]])

    fig = plt.figure(figsize=(13, 6.2))
    for k, (robot, title) in enumerate([
        (block_gripper((0.04, 0.04, 0.07)), "이전 가정 — 4×4×7 cm 블록 하나\n손 본체를 과소, 손가락 사이를 과대 평가"),
        (parts, "실측 — 손 본체 1 + 손가락 2\n막는 것은 언제나 손 본체다"),
    ]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        scatter(ax, meshes, "#2c3e50", stride=3)
        for name, E in robot.items():
            draw(ax, E, _colour(name), label=f"{name}  {np.round(axis_extent(E) * 100, 1)}", alpha=0.20)
        ax.scatter([0], [0], [0], s=45, c="#e74c3c", marker="x")
        ax.text(0, 0, -0.012, "EEF site", color="#e74c3c", fontsize=8, ha="center")
        ax.set_title(title, fontsize=10, pad=2)
        ax.view_init(elev=14, azim=-72)
        equalise(ax, np.vstack([meshes, np.array([[0, 0, -0.02], [0, 0, 0.13]])]))
    fig.suptitle("RB-Y1 그리퍼 (점 = 충돌 메시, 면 = 필터가 보는 형상)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "02-gripper-ellipsoids.png", dpi=DPI)
    plt.close(fig)


def fig_grasp(out: pathlib.Path) -> None:
    """Why the orange grasp needs 3 cm of lift: the hand, not the fingers, is what overlaps."""
    bodies = load_scene()
    E = {n: fit_ellipsoid(p, "mvee") for n, p in bodies.items() if n in FRUIT}
    parts = rby1_gripper(MJCF_ROOT / "assets")
    tgt, obs = "orange", "apple"

    fig = plt.figure(figsize=(13, 6.2))
    for k, dz in enumerate([0.0, 0.03]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        origin = E[tgt].c + [0, 0, dz]
        worst, wname = np.inf, ""
        for name, part in parts.items():
            moved = Ellipsoid(origin + part.c, part.Q)
            h = barrier(optimal_normal(moved, E[obs]), moved, E[obs])
            if h < worst:
                worst, wname = h, name
            draw(ax, moved, "#c0392b" if h <= 0 else _colour(name), alpha=0.22)
        for name in (tgt, obs):
            scatter(ax, bodies[name], "#2c3e50", stride=2)
            draw(ax, E[name], _colour(name), label=f"{name}{' (타깃)' if name == tgt else ''}", alpha=0.25)
        ax.set_title(f"EEF 높이 = 과일 중심 {dz * 100:+.0f} cm\n"
                     f"min h = {worst * 100:+.1f} cm  ({wname})  "
                     f"{'BLOCKED' if worst <= 0 else 'ok'}",
                     fontsize=10, color="#c0392b" if worst <= 0 else "#1e8449", pad=2)
        ax.view_init(elev=10, azim=-88)
        equalise(ax, np.vstack([bodies[tgt], bodies[obs], np.array([origin + [0, 0, 0.13]])]))
    fig.suptitle("오렌지 파지 — 사과가 막는가 (붉은 타원체 = h ≤ 0)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "03-grasp-clearance.png", dpi=DPI)
    plt.close(fig)


def fig_place(out: pathlib.Path) -> None:
    """Descending into the crate, whole vs decomposed."""
    fig = plt.figure(figsize=(13, 6.2))
    parts = rby1_gripper(MJCF_ROOT / "assets")
    for k, (split, title) in enumerate([(None, "통짜 크레이트"), (CRATE_SHELL, "껍질 분해")]):
        bodies = {n: p for n, p in load_scene(split).items() if n.startswith("crate")}
        crate = {n: fit_ellipsoid(p, "mvee") for n, p in bodies.items()}
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for name, pts in bodies.items():
            scatter(ax, pts, "#2c3e50", stride=max(1, len(pts) // 900))
            draw(ax, crate[name], _colour(name), alpha=0.12)
        origin = np.array([0.52, 0.0, 0.95])
        worst = np.inf
        for name, part in parts.items():
            moved = Ellipsoid(origin + part.c, part.Q)
            worst = min(worst, min(barrier(optimal_normal(moved, C), moved, C) for C in crate.values()))
            draw(ax, moved, "#c0392b" if worst <= 0 else "#2471a3", alpha=0.30)
        ax.set_title(f"{title}\n림 높이에서 min h = {worst * 100:+.1f} cm  "
                     f"{'BLOCKED' if worst <= 0 else 'ok'}",
                     fontsize=10, color="#c0392b" if worst <= 0 else "#1e8449", pad=2)
        ax.view_init(elev=12, azim=-70)
        equalise(ax, np.vstack(list(bodies.values()) + [np.array([origin + [0, 0, 0.14]])]))
    fig.suptitle("담기 국면 — 그리퍼를 크레이트 안으로 내릴 때", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "04-place-clearance.png", dpi=DPI)
    plt.close(fig)


def slice_ellipse(E: Ellipsoid, axis: int, value: float):
    """Cut the ellipsoid with the plane `axis = value`; returns (centre2d, 2x2 shape) or None.

    A wireframe cannot show whether a container's interior is free -- the far wall draws right
    through the gap. A cross-section can, so this is what the crate figures use.

    Writing the quadratic form (x-c)'Q^-1(x-c) <= 1 in block form with the cut coordinate held
    fixed and completing the square gives an ellipse in the remaining two coordinates.
    """
    keep = [i for i in range(3) if i != axis]
    M = np.linalg.inv(E.Q)
    Mpp, Mpq, Mqq = M[np.ix_(keep, keep)], M[np.ix_(keep, [axis])].ravel(), M[axis, axis]
    d = value - E.c[axis]
    b = Mpq * d
    shift = np.linalg.solve(Mpp, b)
    r = 1.0 - Mqq * d**2 + b @ shift
    if r <= 0.0:
        return None  # the plane misses the ellipsoid
    return E.c[keep] - shift, Mpp / r


def draw_section(ax, E: Ellipsoid, axis: int, value: float, colour: str, lw=1.4, fill=0.13):
    got = slice_ellipse(E, axis, value)
    if got is None:
        return
    c2, S = got
    w, V = np.linalg.eigh(np.linalg.inv(S))  # S is the inverse-shape; invert back to radii^2
    t = np.linspace(0, 2 * np.pi, 200)
    P = (V @ (np.sqrt(np.maximum(w, 0))[:, None] * np.stack([np.cos(t), np.sin(t)]))).T + c2
    ax.fill(P[:, 0], P[:, 1], color=colour, alpha=fill, linewidth=0)
    ax.plot(P[:, 0], P[:, 1], color=colour, linewidth=lw)


def box_rects(body_name: str, axis: int, value: float):
    """True cross-section of a body's box geoms: (u0, v0, du, dv) rectangles.

    A box contributes only its 8 corners to the point cloud, so a thin slab around the cutting
    plane catches nothing and the "actual geometry" layer would come out empty. Reading the boxes
    back from the MJCF draws what is really there.
    """
    from benchmark.knows_vla.check_admissibility import _floats  # local: visualisation-only helper

    keep = [i for i in range(3) if i != axis]
    tree = _parse_mjcf(MJCF_ROOT / "scenes" / "scene_transport.xml")
    out = []
    for body in tree.iter("body"):
        if body.get("name") != body_name:
            continue
        bpos = _floats(body, "pos", np.zeros(3))
        for geom in body.findall("geom"):
            if geom.get("type") != "box":
                continue
            size = _floats(geom, "size")
            centre = bpos + _floats(geom, "pos", np.zeros(3))
            if size is None or abs(value - centre[axis]) > size[axis]:
                continue  # the plane misses this box
            lo = centre[keep] - size[keep]
            out.append((lo[0], lo[1], 2 * size[keep[0]], 2 * size[keep[1]]))
    return out


def fig_section(out: pathlib.Path) -> None:
    """x-z cross-section through the crate centre: the one view that shows the interior."""
    parts = rby1_gripper(MJCF_ROOT / "assets")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
    for ax, (split, title) in zip(axes, [
        (None, "논문 그대로 — 타원체 하나\n내부가 메워져 담을 수 없다"),
        (CRATE_SHELL, "껍질 분해 — 벽·바닥·손잡이\n내부가 비어 그리퍼가 들어간다"),
    ]):
        bodies = {n: p for n, p in load_scene(split).items() if n.startswith("crate")}
        crate = {n: fit_ellipsoid(p, "mvee") for n, p in bodies.items()}
        for x0, z0, dx, dz in box_rects("crate", 1, 0.0):
            ax.add_patch(plt.Rectangle((x0, z0), dx, dz, facecolor="#2c3e50",
                                       edgecolor="#2c3e50", alpha=0.65, zorder=4))
        for name in bodies:
            draw_section(ax, crate[name], 1, 0.0, _colour(name))
        origin = np.array([0.52, 0.0, 0.95])
        worst = min(barrier(optimal_normal(Ellipsoid(origin + p.c, p.Q), C),
                            Ellipsoid(origin + p.c, p.Q), C)
                    for p in parts.values() for C in crate.values())
        for p in parts.values():
            draw_section(ax, Ellipsoid(origin + p.c, p.Q), 1, 0.0,
                         "#c0392b" if worst <= 0 else "#2471a3", lw=1.8, fill=0.22)
        ax.plot(*origin[[0, 2]], "x", color="#e74c3c", ms=9, mew=2, zorder=5)
        ax.set_title(f"{title}\n림 높이에서 min h = {worst * 100:+.1f} cm  "
                     f"{'BLOCKED' if worst <= 0 else 'ok'}",
                     fontsize=10, color="#c0392b" if worst <= 0 else "#1e8449")
        ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.set_xlim(0.30, 0.74); ax.set_ylim(0.78, 1.12)
    fig.suptitle("크레이트 단면 (y = 0) — 회색 = 실제 형상, 선 = 필터가 보는 형상", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "05-crate-section.png", dpi=DPI)
    plt.close(fig)


def fig_grasp_section(out: pathlib.Path) -> None:
    """x-z section through the fruit pair: why the orange grasp needs 3 cm of lift."""
    bodies = {n: p for n, p in load_scene().items() if n in FRUIT}
    E = {n: fit_ellipsoid(p, "mvee") for n, p in bodies.items()}
    parts = rby1_gripper(MJCF_ROOT / "assets")
    tgt, obs, y0 = "orange", "apple", float(E["orange"].c[1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, dz in zip(axes, [0.0, 0.03]):
        origin = E[tgt].c + [0, 0, dz]
        worst, wname = np.inf, ""
        for name, p in parts.items():
            moved = Ellipsoid(origin + p.c, p.Q)
            h = barrier(optimal_normal(moved, E[obs]), moved, E[obs])
            if h < worst:
                worst, wname = h, name
            draw_section(ax, moved, 1, y0, "#c0392b" if h <= 0 else "#2471a3", lw=1.8, fill=0.20)
        for name in (tgt, obs):
            near = bodies[name][np.abs(bodies[name][:, 1] - y0) < 0.012]
            ax.scatter(near[:, 0], near[:, 2], s=3.0, c="#2c3e50", alpha=0.55, zorder=3)
            draw_section(ax, E[name], 1, y0, _colour(name), lw=1.6)
            ax.annotate(name, (E[name].c[0], E[name].c[2] + 0.055), color=_colour(name),
                        fontsize=9, ha="center", weight="bold")
        ax.plot(origin[0], origin[2], "x", color="#e74c3c", ms=9, mew=2, zorder=5)
        ax.set_title(f"EEF 높이 = 과일 중심 {dz * 100:+.0f} cm\n"
                     f"min h = {worst * 100:+.1f} cm ({wname})  {'BLOCKED' if worst <= 0 else 'ok'}",
                     fontsize=10, color="#c0392b" if worst <= 0 else "#1e8449")
        ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.set_xlim(0.38, 0.68); ax.set_ylim(0.80, 1.02)
    fig.suptitle(f"오렌지 파지 단면 (y = {y0:.2f}) — 사과가 막는가", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "06-grasp-section.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/knows_figs")
    p.add_argument("--only", nargs="*",
                   choices=["scene", "gripper", "grasp", "place", "section", "grasp-section"])
    p.add_argument("--dpi", type=int, default=DPI)
    a = p.parse_args()
    globals()["DPI"] = a.dpi
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Korean labels render as empty boxes without a CJK face, and matplotlib only warns. Pick the
    # first family actually installed rather than assuming one.
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for family in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic"):
        if family in installed:
            matplotlib.rcParams["font.family"] = family
            break
    else:
        print("경고: 한글 폰트를 찾지 못해 라벨이 깨집니다 (apt install fonts-noto-cjk)")
    matplotlib.rcParams["axes.unicode_minus"] = False

    for name, fn in (("scene", fig_scene), ("gripper", fig_gripper),
                     ("grasp", fig_grasp), ("place", fig_place),
                     ("section", fig_section), ("grasp-section", fig_grasp_section)):
        if a.only and name not in a.only:
            continue
        fn(out)
        print(f"wrote {out}/{name}")


if __name__ == "__main__":
    main()
