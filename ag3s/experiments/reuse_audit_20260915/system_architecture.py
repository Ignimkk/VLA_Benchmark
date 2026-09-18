"""System architecture grouped by AG3S, cuRobo, and trajectory optimizer ownership."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


OUT = Path("benchmark/ag3s/docs/figures/reuse-audit-20260915")
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 10})

COLORS = {
    "external": "#eef1f5",
    "curobo": "#dcebd9",
    "ag3s": "#e7e0f3",
    "to": "#dce9f7",
    "edge": "#526780",
    "text": "#17212b",
}


fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")


def container(x, y, w, h, title, subtitle, color):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.14",
        facecolor=color, edgecolor=COLORS["edge"], linewidth=2.1, alpha=0.56,
    ))
    ax.text(x + 0.28, y + h - 0.38, title, ha="left", va="center",
            fontsize=16, fontweight="bold", color=COLORS["text"])
    ax.text(x + w - 0.25, y + h - 0.39, subtitle, ha="right", va="center",
            fontsize=9.5, color="#53606d")


def block(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.07,rounding_size=0.10",
        facecolor=color, edgecolor=COLORS["edge"], linewidth=1.15,
    ))
    ax.text(x + w / 2, y + h * 0.69, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=COLORS["text"])
    ax.text(x + w / 2, y + h * 0.31, body, ha="center", va="center",
            fontsize=9.1, linespacing=1.25, color=COLORS["text"])


def extension_block(x, y, w, h, title, body, color):
    """A required adapter that is not supplied by the public cuRobo Mapper API."""
    block(x, y, w, h, title, body, color)
    ax.add_patch(FancyBboxPatch(
        (x + 0.035, y + 0.035), w - 0.07, h - 0.07,
        boxstyle="round,pad=0.07,rounding_size=0.10", fill=False,
        edgecolor="#9a5d19", linewidth=1.55, linestyle="--",
    ))


def external(x, y, w, h, title, body):
    block(x, y, w, h, title, body, COLORS["external"])


def arrow(start, end, label=None, *, dashed=False, bend=0.0, label_xy=None):
    style = "--" if dashed else "-"
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="->", color=COLORS["edge"], linewidth=1.55,
                        linestyle=style, connectionstyle=f"arc3,rad={bend}"),
    )
    if label:
        lx, ly = label_xy if label_xy is not None else (
            (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.18)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=8.8,
                color="#415365", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def route(points, label=None, *, dashed=False, label_xy=None):
    """Orthogonal route for cross-module contracts that should stay outside containers."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], "--" if dashed else "-", color=COLORS["edge"], linewidth=1.55)
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="->", color=COLORS["edge"], linewidth=1.55,
                                linestyle="--" if dashed else "-"))
    if label:
        lx, ly = label_xy
        ax.text(lx, ly, label, ha="center", va="center", fontsize=8.8,
                color="#415365", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


# External contracts: these are not owned by the three modules.
external(1.8, 9.25, 4.1, 1.05, "Sensors & Robot State",
         "RGB-D · intrinsics · camera pose · joint state")
external(7.25, 9.25, 3.8, 1.05, "VLA Attention",
         "현재 프레임의 attention map")
external(13.0, 9.25, 4.0, 1.05, "VLA Action Chunk",
         "기준 action · timing · gripper channel")

# Three ownership boundaries.
container(0.45, 1.75, 5.1, 6.7, "cuRobo",
          "Geometry service · Mapper + object-aware adapter", COLORS["curobo"])
container(6.05, 1.75, 5.9, 6.7, "AG3S", "의미 · 상태 · 접촉 정책", COLORS["ag3s"])
container(12.45, 1.75, 5.1, 6.7, "TO", "Trajectory optimization · final validation", COLORS["to"])

# cuRobo sub-blocks.
block(0.82, 6.62, 4.35, 1.15, "TSDF Mapping",
      "masked depth를 한 번 적분\nblock-sparse environment map", "#edf6eb")
block(0.82, 5.05, 4.35, 1.15, "Surface Extraction",
      "표면점 · visibility 정보\nattention 투영용 3D geometry", "#edf6eb")
extension_block(0.82, 3.48, 4.35, 1.15, "Object-aware Field Adapter",
                "AG3S의 object ID · role · surface ownership 반영\npublic Mapper 위에 추가할 우리 adapter", "#fff4df")
block(0.82, 2.08, 4.35, 0.95, "Policy-aware ESDF Set",
      "D_all · D_destination · D_other\ndistance · gradient · valid · bounds · map version", "#edf6eb")

# AG3S sub-blocks.
block(6.38, 6.62, 2.48, 1.15, "Robot Mask",
      "robot depth 제거\n주변 물체 보존", "#f3eff9")
block(6.38, 5.05, 2.48, 1.15, "3D Attention Projection",
      "현재 attention을\n가시 표면에 투영", "#f3eff9")
block(9.13, 5.05, 2.48, 1.15, "Target Grounding",
      "경계 · 후보 ID · confidence\n3D connectivity", "#f3eff9")
block(7.35, 3.42, 3.30, 1.30, "Object State Registry",
      "object ID · role · confidence · last seen\nobserved → latched → attached → released\nobject-local geometry · parent link", "#f3eff9")
block(6.38, 2.08, 2.48, 0.95, "Scene Geometry Contract",
      "object ID · role · static/dynamic · field group\nsurface ownership · attach/exclude", "#f3eff9")
block(9.13, 2.08, 2.48, 0.95, "Contact Policy",
      "link × object × phase\nmargin · permission", "#f3eff9")

# TO sub-blocks.
block(12.82, 6.62, 4.35, 1.15, "Chunk Adapter",
      "VLA action을 joint trajectory로 변환\ntiming · gripper channel 보존", "#eef5fb")
block(12.82, 5.05, 4.35, 1.15, "Trajectory Refinement",
      "기준 궤적 추종\nSQP · joint/velocity/collision constraints", "#eef5fb")
block(12.82, 3.48, 4.35, 1.15, "Final Safety Check",
      "후보 궤적 전체 · inter-step 검사\n환경 · 파지 물체 · 관측 범위", "#eef5fb")
block(12.82, 2.08, 4.35, 0.95, "Result",
      "corrected chunk · status · failure reason", "#eef5fb")

# Internal flow.
arrow((3.0, 6.62), (3.0, 6.20))
arrow((3.0, 5.05), (3.0, 4.63))
arrow((3.0, 3.48), (3.0, 3.03))
arrow((8.86, 5.62), (9.13, 5.62))
arrow((10.37, 5.05), (9.0, 4.72), bend=0.06)
arrow((8.45, 3.42), (7.62, 3.03), bend=-0.05)
arrow((9.55, 3.42), (10.37, 3.03), bend=0.05)
arrow((15.0, 6.62), (15.0, 6.20))
arrow((15.0, 5.05), (15.0, 4.63))
arrow((15.0, 3.48), (15.0, 3.03))

# Cross-module contracts.
arrow((4.7, 9.25), (7.62, 7.77), "depth · pose · joint state",
      bend=0.03, label_xy=(6.22, 8.86))
arrow((6.38, 7.18), (5.17, 7.18), "masked depth", label_xy=(5.78, 7.45))
arrow((5.17, 5.62), (6.38, 5.62), "surface points", label_xy=(5.78, 5.91))
arrow((9.15, 9.25), (7.62, 6.20), "current attention", bend=0.04,
      label_xy=(8.82, 8.27))
arrow((15.0, 9.25), (15.0, 7.77), "action chunk", label_xy=(15.62, 8.58))
route([(11.61, 2.55), (12.16, 2.55), (12.16, 5.62), (12.82, 5.62)],
      "Margin Policy", label_xy=(11.84, 2.82))
route([(5.17, 2.55), (5.17, 1.42), (12.08, 1.42), (12.08, 5.30), (12.82, 5.30)],
      "Geometry Snapshot", label_xy=(8.62, 1.18))
route([(6.38, 2.55), (5.72, 2.55), (5.72, 4.05), (5.17, 4.05)])
ax.text(5.72, 3.31, "object geometry + field group", ha="center", va="center",
        rotation=90, fontsize=8.3, color="#415365",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

# Actual execution decision belongs to the caller, not TO.
external(13.0, 0.35, 4.0, 0.85, "Execution Supervisor",
         "status를 확인해 execute / hold / replan 결정")
arrow((15.0, 2.08), (15.0, 1.20))

ax.text(9.0, 10.72, "AG3S · cuRobo · TO 시스템 아키텍처",
        ha="center", va="center", fontsize=21, fontweight="bold", color=COLORS["text"])
ax.text(9.0, 0.06,
        "갈색 점선 테두리: public cuRobo Mapper에는 없는 object-aware adapter",
        ha="center", va="bottom", fontsize=9.5, color="#53606d")

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"architecture-system.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)
