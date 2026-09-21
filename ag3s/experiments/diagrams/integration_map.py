"""cuRobo · attention · trajopt 가 어디서 만나고 무엇이 남았는가 — 통합 지도."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if pathlib.Path(p).exists():
        fm.fontManager.addfont(p)
import sys; sys.path.insert(0, "/mnt/dev/work")
from benchmark.ag3s.experiments.common import figstyle
figstyle.use_korean()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, axs = plt.subplots(1, 2, figsize=(19, 8.2))

def box(a, x, y, w, h, text, fc, ec="0.25", fs=10, lw=1.6, ls="-"):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                               fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=2))
    a.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)

def arrow(a, x1, y1, x2, y2, color="0.3", lw=2.0, ls="-"):
    a.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=17,
                                color=color, lw=lw, linestyle=ls, zorder=1))

GREEN, BLUE, GREY, RED = "#d6ecd8", "#d8e6f5", "#ececec", "#f7d9d9"

# ---------------- 왼쪽: 지금 ----------------
a = axs[0]; a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")
a.set_title("지금 — 오프라인으로 만난다 (검증은 끝)", fontsize=13, fontweight="bold")

a.add_patch(FancyBboxPatch((0.2, 6.55), 9.6, 3.1, boxstyle="round,pad=0.02",
                           fc="#fbf6ec", ec="#c9a227", lw=2))
a.text(0.45, 9.35, "프로세스 A  ·  .venv-curobo  ·  numpy 1.26  ·  GPU",
       fontsize=10.5, fontweight="bold", color="#8a6d1a")
box(a, 0.6, 7.6, 2.4, 1.0, "depth\n(기록/카메라)", GREY)
box(a, 3.4, 7.6, 2.6, 1.0, "cuRobo\nTSDF → ESDF", GREEN)
box(a, 6.5, 7.6, 3.0, 1.0, "build_rollout_fields.py\n→ npz 로 굽는다", GREEN)
arrow(a, 3.0, 8.1, 3.4, 8.1); arrow(a, 6.0, 8.1, 6.5, 8.1)
a.text(5.0, 6.95, "적분 2.1~2.7 ms + ESDF 1.2~1.3 ms = 3.5~4.0 ms",
       ha="center", fontsize=10, color="#8a6d1a", fontweight="bold")

a.annotate("", xy=(5.0, 5.75), xytext=(5.0, 6.5),
           arrowprops=dict(arrowstyle="-|>", lw=3, color="#c9a227"))
a.text(5.15, 6.1, "npz  (오프라인 경계)", fontsize=10.5, color="#8a6d1a", fontweight="bold")

a.add_patch(FancyBboxPatch((0.2, 0.35), 9.6, 5.3, boxstyle="round,pad=0.02",
                           fc="#f2f7fb", ec="#3a6ea5", lw=2))
a.text(0.45, 5.3, "프로세스 B  ·  .venv-ag3s  ·  numpy 2.4  ·  CPU",
       fontsize=10.5, fontweight="bold", color="#26547c")
box(a, 0.6, 4.05, 2.6, 0.95, "attention\nlifting ✓", BLUE)
box(a, 3.6, 4.05, 2.6, 0.95, "target\ngrounding ✓", BLUE)
box(a, 6.6, 4.05, 2.8, 0.95, "접촉 권한\n(조작 대상) ✓", BLUE)
arrow(a, 3.2, 4.52, 3.6, 4.52); arrow(a, 6.2, 4.52, 6.6, 4.52)

box(a, 0.6, 2.55, 4.0, 1.0, "pipeline._build_esdf\n아직 numpy — 2.5~3.1 s", RED, ec="#b03030")
box(a, 5.2, 2.55, 4.2, 1.0, "curobo_field.py (어댑터)\ndistance · gradient · voxel_size", GREEN)
a.text(4.9, 3.05, "대체", ha="center", fontsize=10, color="#b03030", fontweight="bold")
a.annotate("", xy=(5.2, 3.05), xytext=(4.6, 3.05),
           arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#b03030", linestyle="--"))

box(a, 2.4, 1.0, 5.2, 1.0, "trajopt — SQP 선형화 · QP(osqp)\ncasadi 는 로봇 FK 에만", BLUE)
arrow(a, 5.0, 4.05, 5.0, 3.55); arrow(a, 5.0, 2.55, 5.0, 2.0)

# ---------------- 오른쪽: 합쳐진 뒤 ----------------
a = axs[1]; a.set_xlim(0, 10); a.set_ylim(-0.6, 10); a.axis("off")
a.set_title("합쳐진 뒤 — 한 프로세스, 실시간", fontsize=13, fontweight="bold")

a.add_patch(FancyBboxPatch((0.2, 1.5), 9.6, 8.1, boxstyle="round,pad=0.02",
                           fc="#f2faf3", ec="#2e7d32", lw=2.4))
a.text(0.45, 9.25, "한 프로세스  ·  numpy 1.26  ·  curobo + casadi + osqp  (MuJoCo 불필요)",
       fontsize=10.5, fontweight="bold", color="#2e7d32")

box(a, 0.7, 7.75, 2.5, 0.95, "depth\n(실 카메라)", GREY)
box(a, 3.6, 7.75, 2.6, 0.95, "cuRobo\nTSDF → ESDF", GREEN)
box(a, 6.6, 7.75, 2.7, 0.95, "curobo_field\n어댑터", GREEN)
arrow(a, 3.2, 8.22, 3.6, 8.22); arrow(a, 6.2, 8.22, 6.6, 8.22)

box(a, 0.7, 6.1, 2.5, 0.95, "attention\nlifting", BLUE)
box(a, 3.6, 6.1, 2.6, 0.95, "target\ngrounding", BLUE)
box(a, 6.6, 6.1, 2.7, 0.95, "조작 대상\n(쥔 물체)", BLUE)
arrow(a, 3.2, 6.57, 3.6, 6.57); arrow(a, 6.2, 6.57, 6.6, 6.57)
arrow(a, 1.95, 7.75, 1.95, 7.05)

box(a, 1.6, 4.35, 6.8, 1.1,
    "Step 10 이 정하는 것 — 쥔 물체가 어디로 가는가\n"
    "필드에서 빼고 · 로봇 구로 넣고 · 권한은 그것에", "#fff3d6", ec="#c9a227", lw=2)
arrow(a, 5.0, 6.1, 5.0, 5.45, color="#c9a227", lw=2.6)

box(a, 2.2, 2.75, 5.6, 1.05, "trajopt — SQP 선형화 · QP(osqp)", BLUE)
arrow(a, 5.0, 4.35, 5.0, 3.8)

a.text(5.0, 2.2, "청크당 지각 3.5~4.0 ms  /  예산 66.7 ms", ha="center",
       fontsize=12, fontweight="bold", color="#2e7d32")

a.text(0.3, 1.18, "합치기 전에 확인할 것", fontsize=11.5, fontweight="bold")
a.text(0.3, 0.05,
       "1. .venv-curobo 에 casadi·osqp 가 설치되는가 (numpy 1.26 위에서)\n"
       "   → 안 되면 프로세스 분리를 유지하고 IPC 로 간다. 설계가 달라진다\n"
       "2. Step 10 이 먼저다 — 필드에 무엇이 들어가는지가 그때 정해진다",
       fontsize=10.5, va="bottom", linespacing=1.75)

fig.suptitle("cuRoboV2 · attention · trajopt 는 언제 합쳐지는가", fontsize=14.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
o = pathlib.Path("benchmark/ag3s/docs/figures/integration-map.png")
fig.savefig(o, dpi=110, bbox_inches="tight")
print("wrote", o)
