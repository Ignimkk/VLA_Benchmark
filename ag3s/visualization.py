"""Static per-stage dumps.

Every stage gets its own figure so a failure can be localised by looking rather than by bisecting:
if the target cluster is wrong, the seeds panel says whether attention or connectivity caused it; if
a constraint is missing, the primitives panel says whether the geometry was ever fitted.

**matplotlib is imported inside the functions, never at module scope.** The core pipeline must import
on a machine with no plotting stack, and `benchmark/ag3s/__init__.py` deliberately does not pull this
module in. Nothing here is on the hot path.

The repo has no 3-D visualisation library in any venv (no open3d, no pyvista), so these are
matplotlib scatter dumps — which the spec explicitly allows as the fallback. They are written to PNG
rather than shown, because the GPU server this project uses has no display.
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional, Sequence

import numpy as np

from benchmark.ag3s.geometry import to_spheres
from benchmark.ag3s.types import (
    AttentionPointCloud,
    CollisionCandidate,
    PointCloud,
    SourceType,
    SupportSurface,
    TargetGeometry,
)

#: Where visual artefacts live. Images and prose are kept in separate subdirectories so the PNGs can
#: be regenerated wholesale (`--out` defaults here) without touching the written analysis beside them.
ASSET_DIR = pathlib.Path(__file__).resolve().parent / "asset"
ASSET_IMAGE_DIR = ASSET_DIR / "image"
ASSET_DOC_DIR = ASSET_DIR / "doc"

#: Colour per candidate class. Chosen so the two that must never be confused — the target and
#: everything else — are maximally distinct in both hue and lightness.
SOURCE_COLORS = {
    SourceType.OBJECT: "#3b82f6",
    SourceType.TARGET: "#ef4444",
    SourceType.SUPPORT_SURFACE: "#a3a3a3",
    SourceType.UNKNOWN_GEOMETRY: "#f59e0b",
    SourceType.RESTRICTED_REGION: "#8b5cf6",
}


def _axes(fig, index, title, total=1):
    ax = fig.add_subplot(1, total, index, projection="3d")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.set_zlabel("z", fontsize=7)
    ax.tick_params(labelsize=6)
    return ax


def _thin(points: np.ndarray, limit: int) -> np.ndarray:
    """Deterministic stride, matching `reconstruction.cap_points` — a plot must not re-randomise."""
    if points.shape[0] <= limit:
        return points
    return points[np.unique(np.linspace(0, points.shape[0] - 1, limit).astype(np.int64))]


def _equalise(ax, points: np.ndarray) -> None:
    """Equal aspect, which matplotlib's 3-D axes do not do by default.

    Without it a 4 cm sphere drawn on axes spanning 1.5 m in x and 0.1 m in z looks like a pancake,
    and every judgement made from the picture is wrong.
    """
    if points.size == 0:
        return
    centre = (points.max(axis=0) + points.min(axis=0)) / 2.0
    radius = float((points.max(axis=0) - points.min(axis=0)).max()) / 2.0 or 0.1
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)


def _wire_sphere(ax, centre, radius, color, alpha=0.25):
    u = np.linspace(0, 2 * np.pi, 14)
    v = np.linspace(0, np.pi, 8)
    x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = centre[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.4)


# --------------------------------------------------------------------------- single panels


def plot_point_cloud(ax, cloud: PointCloud, *, color="#64748b", size=0.4, limit=20000, label=None):
    points = _thin(cloud.points if isinstance(cloud, PointCloud) else np.asarray(cloud), limit)
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, c=color, label=label,
                   depthshade=False, linewidths=0)
    return points


def plot_attention(ax, attention_cloud: AttentionPointCloud, *, limit=20000):
    """Attention-weighted cloud. Low-attention points are drawn, not hidden.

    Drawing them matters: this panel is the visual proof that nothing was discarded for being
    uninteresting, which is the invariant the whole design rests on.
    """
    index = np.unique(np.linspace(0, len(attention_cloud) - 1, min(len(attention_cloud), limit)).astype(np.int64))
    points = attention_cloud.points[index]
    values = attention_cloud.attention[index]
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.6, c=values,
                         cmap="inferno", vmin=0.0, vmax=1.0, depthshade=False, linewidths=0)
    return scatter, points


# ------------------------------------------------------------------------------ the bundle


def dump_stages(
    out_dir: str | pathlib.Path,
    *,
    raw_cloud: PointCloud,
    working_cloud: Optional[PointCloud] = None,
    attention_cloud: Optional[AttentionPointCloud] = None,
    seed_indices: Optional[np.ndarray] = None,
    target: Optional[TargetGeometry] = None,
    candidates: Sequence[CollisionCandidate] = (),
    support_surfaces: Sequence[SupportSurface] = (),
    constraint_spec: Any = None,
    prefix: str = "ag3s",
    dpi: int = 130,
) -> list[pathlib.Path]:
    """Write the eight stage dumps as PNGs. Returns the paths actually written.

    `raw_cloud` is the cloud before self-filtering; `working_cloud` is the one after it, and it is
    the **only** cloud that `point_indices` are valid against. Getting that wrong is not a cosmetic
    error: the self-filter removes points from the middle of the array, so every later index shifts
    and the candidate panels come out as stripes scattered across the whole scene — a picture that
    looks like a clustering failure when clustering was fine. Every indexed lookup below therefore
    goes through `indexed`, never through `raw_cloud`.

    Stages whose inputs were not supplied are skipped rather than emitted blank, so the file list is
    itself a record of how far the pipeline got.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display on the GPU server; also makes this importable under pytest
    import matplotlib.pyplot as plt

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    indexed = working_cloud if working_cloud is not None else raw_cloud

    def _save(fig, name):
        path = out / f"{prefix}_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)

    # 1. raw point cloud ------------------------------------------------------------------
    fig = plt.figure(figsize=(5, 4.5))
    ax = _axes(fig, 1, f"1. raw point cloud ({len(raw_cloud)} pts)")
    _equalise(ax, plot_point_cloud(ax, raw_cloud))
    _save(fig, "01_raw_cloud")

    # 2. self-filtered --------------------------------------------------------------------
    if working_cloud is not None:
        fig = plt.figure(figsize=(9, 4.5))
        ax = _axes(fig, 1, f"2a. before self-filter ({len(raw_cloud)})", total=2)
        _equalise(ax, plot_point_cloud(ax, raw_cloud))
        ax = _axes(fig, 2, f"2b. after self-filter ({len(working_cloud)})", total=2)
        _equalise(ax, plot_point_cloud(ax, working_cloud, color="#0f766e"))
        _save(fig, "02_self_filtered")

    # 3. attention-weighted ---------------------------------------------------------------
    if attention_cloud is not None:
        fig = plt.figure(figsize=(5.5, 4.5))
        ax = _axes(fig, 1, "3. attention-weighted cloud (all points kept)")
        scatter, points = plot_attention(ax, attention_cloud)
        fig.colorbar(scatter, ax=ax, shrink=0.6, label="attention")
        _equalise(ax, points)
        _save(fig, "03_attention")

        # 4. seeds ------------------------------------------------------------------------
        if seed_indices is not None and seed_indices.size:
            fig = plt.figure(figsize=(5, 4.5))
            ax = _axes(fig, 1, f"4. attention seeds ({seed_indices.size})")
            plot_point_cloud(ax, attention_cloud.cloud, color="#d4d4d8", size=0.3)
            seeds = attention_cloud.points[seed_indices]
            ax.scatter(seeds[:, 0], seeds[:, 1], seeds[:, 2], s=3.0, c="#dc2626",
                       depthshade=False, linewidths=0)
            _equalise(ax, attention_cloud.points)
            _save(fig, "04_seeds")

    # 5. target cluster -------------------------------------------------------------------
    if target is not None:
        fig = plt.figure(figsize=(5, 4.5))
        ax = _axes(fig, 1, f"5. target (id {target.id}, conf {target.confidence:.3f})")
        plot_point_cloud(ax, indexed, color="#e4e4e7", size=0.3)
        ax.scatter(target.points[:, 0], target.points[:, 1], target.points[:, 2],
                   s=3.0, c="#ef4444", depthshade=False, linewidths=0)
        ax.scatter(*target.centroid, s=60, c="#000000", marker="+")
        _equalise(ax, target.points)
        _save(fig, "05_target")

    # 6. candidate clusters ---------------------------------------------------------------
    if candidates:
        fig = plt.figure(figsize=(5.5, 4.5))
        ax = _axes(fig, 1, f"6. collision candidates ({len(candidates)})")
        drawn = []
        for candidate in candidates:
            if candidate.source_type is SourceType.SUPPORT_SURFACE:
                continue
            points = _thin(indexed.points[candidate.point_indices], 4000)
            drawn.append(points)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.5,
                       c=SOURCE_COLORS.get(candidate.source_type, "#000000"),
                       label=f"{candidate.source_type.value} #{candidate.id}",
                       depthshade=False, linewidths=0)
        for surface in support_surfaces:
            points = _thin(indexed.points[surface.point_indices], 3000)
            drawn.append(points)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.3, c="#a3a3a3",
                       depthshade=False, linewidths=0)
        if drawn:
            _equalise(ax, np.vstack(drawn))
        ax.legend(fontsize=5, loc="upper left", markerscale=3)
        _save(fig, "06_candidates")

        # 7. primitive approximations -----------------------------------------------------
        fig = plt.figure(figsize=(5.5, 4.5))
        ax = _axes(fig, 1, "7. primitive approximation")
        drawn = []
        for candidate in candidates:
            if not candidate.geometry:
                continue
            points = _thin(indexed.points[candidate.point_indices], 2000)
            drawn.append(points)
            color = SOURCE_COLORS.get(candidate.source_type, "#000000")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.0, c=color,
                       alpha=0.4, depthshade=False, linewidths=0)
            for primitive in candidate.geometry:
                for centre, radius in to_spheres(primitive):
                    _wire_sphere(ax, centre, radius, color)
        if drawn:
            _equalise(ax, np.vstack(drawn))
        _save(fig, "07_primitives")

    # 8. final constraints ----------------------------------------------------------------
    if constraint_spec is not None:
        fig = plt.figure(figsize=(5.5, 4.5))
        active = int(constraint_spec.active_mask.sum())
        ax = _axes(
            fig, 1,
            f"8. constraint slots ({active}/{constraint_spec.max_candidates} active, "
            f"{constraint_spec.n_constraints} rows)",
        )
        plot_point_cloud(ax, indexed, color="#e4e4e7", size=0.3)
        layout = constraint_spec.layout
        lo_pos = layout["pos"][0]
        lo_r = layout["radius"][0]
        lo_d = layout["d_safe"][0]
        centres = []
        for slot in range(constraint_spec.max_candidates):
            if constraint_spec.active_mask[slot] <= 0.0:
                continue
            centre = constraint_spec.parameter_values[lo_pos + 3 * slot : lo_pos + 3 * slot + 3]
            radius = float(constraint_spec.parameter_values[lo_r + slot])
            margin = float(constraint_spec.parameter_values[lo_d + slot])
            centres.append(centre)
            _wire_sphere(ax, centre, radius, "#2563eb", alpha=0.5)
            # The margin is drawn separately: it is what the phase rules move, and seeing it shrink
            # from TRANSIT to GRASP is the fastest check that phase handling is wired up.
            _wire_sphere(ax, centre, radius + margin, "#f97316", alpha=0.2)
        if centres:
            _equalise(ax, np.vstack([indexed.points, np.stack(centres)]))
        _save(fig, "08_constraints")

    return written


def dump_from_pipeline(
    out_dir: str | pathlib.Path,
    ag3s: Any,
    constraint_set: Any,
    debug: dict[str, Any],
    **kwargs: Any,
) -> list[pathlib.Path]:
    """Convenience wrapper over the artifacts `AG3S.process_debug` hands back."""
    return dump_stages(
        out_dir,
        raw_cloud=debug["raw_cloud"],
        working_cloud=debug.get("filtered_cloud"),
        attention_cloud=debug.get("attention_cloud"),
        seed_indices=debug.get("seed_indices"),
        target=constraint_set.target,
        candidates=constraint_set.candidates,
        support_surfaces=constraint_set.support_surfaces,
        constraint_spec=constraint_set.constraints,
        **kwargs,
    )


def main(argv: Optional[list[str]] = None) -> int:  # pragma: no cover - demo entry point
    """Render all eight dumps for the synthetic fixture."""
    import argparse

    parser = argparse.ArgumentParser(description="AG3S stage visualisation")
    parser.add_argument("--config", default="benchmark/ag3s/configs/default.yaml")
    parser.add_argument(
        "--out",
        default=str(ASSET_IMAGE_DIR),
        help="output directory for the PNGs (default: benchmark/ag3s/asset/image)",
    )
    parser.add_argument("--phase", default="approach")
    args = parser.parse_args(argv)

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.pipeline import AG3S

    try:
        from tests.ag3s.fixtures import DEFAULT_ARM_Q, PlanarTwoLinkArm, make_scene
    except ImportError as exc:
        print(f"the demo scene lives in tests/ag3s/fixtures.py; run from the workspace root ({exc})")
        return 1

    arm = PlanarTwoLinkArm()
    ag3s = AG3S(AG3SConfig.from_yaml(args.config), robot_model=arm)
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    constraint_set, debug = ag3s.process_debug(
        attention_map=scene.attention_grid,
        depth=scene.depth,
        camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam,
        robot_state=DEFAULT_ARM_Q,
        phase=args.phase,
        image_hw=scene.hw,
    )
    for path in dump_from_pipeline(args.out, ag3s, constraint_set, debug):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ASSET_DIR",
    "ASSET_DOC_DIR",
    "ASSET_IMAGE_DIR",
    "SOURCE_COLORS",
    "dump_from_pipeline",
    "dump_stages",
    "main",
    "plot_attention",
    "plot_point_cloud",
]
