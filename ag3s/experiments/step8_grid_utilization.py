"""격자의 몇 %가 실제로 질의되는가 — 블록-스파스 할당의 근거.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.step8_grid_utilization

## 질문 (사용자, 2026-09-14)

*"cuRoboV2 논문에서는 타겟 근처만 데이터를 할당하는 블록-스파스 방식을 사용합니다. 우리도
attention 근처만 처리하면 안 되나요? 바닥을 처리할 이유는 없어 보입니다."*

낭비 규모를 재고, **"attention 근처" 와 "로봇이 갈 곳 근처" 가 다르다**는 것을 같이 남긴다.
F11 에서 정책의 attention 은 파지 착수 순간 목적지로 옮겨갔다 — attention 중심으로 할당하면
그 순간 창이 바구니 위로 가고 정작 로봇이 있는 곳이 빈다. cuRobo 의 블록-스파스는 **질의점**
주변에 할당하는 것이고, 우리 계획서도 같은 이유로 **swept volume** 을 택했다.

## 무엇을 재는가

롤아웃 전체에서 제약 모델 구의 중심을 모으고, 격자 복셀 각각이 그 중 가장 가까운 구에서 얼마나
떨어져 있는지 본다. `구 반지름 + 여유거리 + 보간 1 복셀` 안에 드는 복셀만이 **실제로 제약
평가에 쓰인다.** 나머지는 만들고 버린다.
"""

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
    from benchmark.ag3s.experiments import figstyle
    figstyle.use_korean()


def main() -> None:
    _style()
    from scipy.spatial import cKDTree

    from benchmark.ag3s.experiments.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene
    from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()

    run = load_run(args.records)
    scene = replay_scene(run)
    rb = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    C, R = [], None
    for st in run.steps:
        pose_scene(scene, st)
        q = np.asarray([scene.data.qpos[scene._qadr[j]] for j in DEFAULT_RBY1_JOINTS], float)
        c, r = rb.sphere_centers_numeric(q)
        C.append(np.asarray(c, float)); R = np.asarray(r, float)
    C = np.concatenate(C)
    per_step = len(R)
    scene.close()

    v = args.voxel
    lo = np.array([-0.3, -0.9, 0.0]); shape = (76, 90, 80)
    hi = lo + np.array(shape) * v
    n_tot = int(np.prod(shape))
    g = [lo[k] + (np.arange(shape[k]) + 0.5) * v for k in range(3)]
    G = np.stack(np.meshgrid(*g, indexing="ij"), axis=-1).reshape(-1, 3)
    dmin = cKDTree(C).query(G, workers=-1)[0]
    reach = float(R.max() + args.margin + v)

    radii = np.linspace(0.05, 0.8, 40)
    frac = [float((dmin <= rr).mean()) * 100 for rr in radii]
    inq = G[dmin <= reach]

    print("=" * 88)
    print(f"격자 이용률 — {args.records}, 롤아웃 {len(run.steps)} 스텝 × {per_step} 구")
    print("=" * 88)
    print(f"구 반지름 {R.min()*1000:.0f} ~ {R.max()*1000:.0f} mm,  여유거리 {args.margin*1000:.0f} mm")
    print(f"질의 도달 반경 = 반지름 + 여유거리 + 보간 1 복셀 = {reach*1000:.0f} mm")
    print(f"\n격자 {shape} = {n_tot:,} 복셀  ({v*1000:.0f} mm)")
    for rr, lab in ((reach, "실제 질의권"), (0.30, "30 cm"), (0.50, "50 cm")):
        n = int((dmin <= rr).sum())
        print(f"  {rr*1000:>5.0f} mm 이내: {n:>9,}  ({n/n_tot*100:5.2f} %)   [{lab}]")
    print(f"\n질의권 bbox  x [{inq[:,0].min():.3f}, {inq[:,0].max():.3f}]"
          f"  y [{inq[:,1].min():.3f}, {inq[:,1].max():.3f}]"
          f"  z [{inq[:,2].min():.3f}, {inq[:,2].max():.3f}]")
    tight = np.ceil((inq.max(axis=0) - inq.min(axis=0)) / v).astype(int) + 1
    print(f"질의권을 딱 감싸는 상자: {tuple(tight)} = {int(np.prod(tight)):,} 복셀 "
          f"({np.prod(tight)/n_tot*100:.1f} % — 블록-스파스 없이 상자만 줄여도)")
    print(f"\n바닥 z=0.003 에서 가장 낮은 질의권까지: {(inq[:,2].min()-0.003)*1000:.0f} mm")

    _figure(dmin, radii, frac, C, R, inq, lo, hi, n_tot, reach, v, args)


def _figure(dmin, radii, frac, C, R, inq, lo, hi, n_tot, reach, v, args):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axs = plt.subplots(1, 3, figsize=(19.5, 6.2))

    # (a) 실제 씬 — 옆에서 본 격자 · 구 · 질의권
    a = axs[0]
    a.add_patch(Rectangle((lo[0], lo[2]), hi[0]-lo[0], hi[2]-lo[2],
                          fc="0.93", ec="0.5", lw=1.6))
    a.text(lo[0]+0.03, hi[2]-0.08, f"ESDF 격자 {n_tot:,} 복셀", fontsize=10, color="0.35")
    a.scatter(inq[::37, 0], inq[::37, 2], s=2, c="tab:green", alpha=0.35,
              label=f"질의권 ({len(inq)/n_tot*100:.1f} %)")
    a.scatter(C[::7, 0], C[::7, 2], s=6, c="crimson", alpha=0.7, label="로봇 구 중심 (전 롤아웃)")
    a.axhline(0.823, color="tab:orange", ls="--", lw=2)
    a.text(hi[0]-0.02, 0.833, "테이블 0.823", ha="right", fontsize=9.5, color="#b26a00")
    a.axhline(0.003, color="tab:brown", ls="--", lw=2)
    a.text(hi[0]-0.02, 0.03, "바닥 0.003", ha="right", fontsize=9.5, color="#7a4a2a")
    a.annotate("", xy=(0.95, 0.003), xytext=(0.95, inq[:,2].min()),
               arrowprops=dict(arrowstyle="<->", color="tab:brown", lw=2))
    a.text(0.97, (inq[:,2].min()+0.003)/2, f"{(inq[:,2].min()-0.003)*1000:.0f} mm\n비어 있다",
           fontsize=10, color="#7a4a2a", fontweight="bold", va="center")
    a.set_xlabel("x [m]"); a.set_ylabel("z [m]"); a.set_aspect("equal")
    a.set_title("(a) 실제 씬 — 격자 대 실제로 쓰이는 부분\n옆에서 본 단면", fontsize=11)
    a.legend(fontsize=9, loc="upper left"); a.grid(alpha=0.25)

    # (b) 반경별 이용률
    a = axs[1]
    a.plot(np.asarray(radii)*1000, frac, "-", color="tab:blue", lw=2.6)
    a.axvline(reach*1000, color="crimson", ls="--", lw=2.2)
    f_at = float((dmin <= reach).mean())*100
    a.plot([reach*1000], [f_at], "o", color="crimson", ms=10, zorder=5)
    a.annotate(f"실제 질의권\n{reach*1000:.0f} mm → {f_at:.1f} %",
               xy=(reach*1000, f_at), xytext=(reach*1000+90, f_at+22),
               arrowprops=dict(arrowstyle="->", color="crimson", lw=1.8),
               fontsize=11, color="crimson", fontweight="bold")
    a.set_xlabel("로봇 구에서의 거리 [mm]"); a.set_ylabel("격자 중 그 안에 드는 비율 [%]")
    a.set_title("(b) 얼마나 멀리까지 쓰나\n실제로 쓰는 것은 격자의 일부다", fontsize=11)
    a.grid(alpha=0.3)

    # (c) 표
    a = axs[2]; a.axis("off")
    tight = np.ceil((inq.max(axis=0)-inq.min(axis=0))/v).astype(int)+1
    rows = [["항목", "값"],
            ["격자", f"{n_tot:,} 복셀 ({v*1000:.0f} mm)"],
            ["구 반지름", f"{R.min()*1000:.0f} ~ {R.max()*1000:.0f} mm"],
            ["질의 도달 반경", f"{reach*1000:.0f} mm"],
            ["", ""],
            ["실제 질의권", f"{int((dmin<=reach).sum()):,}  ({f_at:.2f} %)"],
            ["쓰이지 않는 부분", f"{100-f_at:.2f} %"],
            ["", ""],
            ["질의권 z 범위", f"{inq[:,2].min():.3f} ~ {inq[:,2].max():.3f} m"],
            ["바닥까지 빈 거리", f"{(inq[:,2].min()-0.003)*1000:.0f} mm"],
            ["딱 감싸는 상자", f"{int(np.prod(tight)):,} 복셀 ({np.prod(tight)/n_tot*100:.1f} %)"]]
    t = a.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.0, 1.7)
    for j in range(2):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
        t[(5, j)].set_facecolor("#d9f0d9"); t[(6, j)].set_facecolor("#f7d6d6")
    a.set_title("(c) 블록-스파스 할당의 근거", fontsize=11.5, y=0.9)

    fig.suptitle("격자의 몇 %가 실제로 질의되는가 — 나머지는 만들고 버린다", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    o = OUT / f"step8-grid-utilization-{args.records[-4:]}.png"
    fig.savefig(o, dpi=110, bbox_inches="tight")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
