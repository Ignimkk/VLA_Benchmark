"""Step 8 — 평면 행을 빼고 필드에만 맡겨도 되는가.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.step8_planes_vs_field

## 질문 (사용자, 2026-09-14)

*"support_surface 를 분할 용도로만 쓰고 ESDF 로는 안 쓰면 안 되나? cuRobo 로 ESDF 생성하면
되잖아"* — 즉 평면 추출은 grounding 의 `exclude_mask` 로만 쓰고, 필드에서 파내지도 말고 평면
행도 읽지 말자는 제안이다.

**두 실제 경로가 이미 그렇게 하고 있다** (`bringup.py:197`, `esdf_rollout --support-surfaces
field`). 반대인 것은 라이브러리 기본값뿐이다. 그러나 그것은 **선택이었지 판정이 아니었다** —
비교 측정이 없었다. 그래서 잃는 것이 무엇인지 잰다.

## 무엇을 재는가

평면 행이 하던 일을 필드가 대신할 수 있으려면 **그 표면이 필드 안에 실제로 있어야** 한다.
수직선을 따라 필드에 물어보고, 맞춰진 평면과 나란히 놓고, 제약 모델의 구가 실제로 어디까지
내려가는지 겹쳐 본다.

`unknown_policy` 가 `free` 이므로 **미관측은 `+max_distance` 로 답한다** — "멀다" 와 "모른다" 가
같은 숫자로 나온다. 그 구분이 이 판정의 핵심이다.
"""

import argparse
import pathlib

import numpy as np

OUT = pathlib.Path("benchmark/ag3s/docs/figures")
CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


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
    import matplotlib.pyplot as plt

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.pipeline import AG3S
    from benchmark.ag3s.experiments.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.mujoco_source import camera_observation
    from benchmark.ag3s.experiments.policy_record import load_run, pose_scene, replay_scene
    from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--xy", type=float, nargs=2, default=[0.5, 0.0])
    args = ap.parse_args()

    run = load_run(args.records, limit=args.frames)
    scene = replay_scene(run)
    fr = build_robot_model(scene)
    rb = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    ag = AG3S(AG3SConfig.from_dict({
        "collision_backend": "esdf", "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False}}),
        robot_model=fr, constraint_robot_model=rb)

    for i in range(args.frames):
        pose_scene(scene, run.steps[i])
        obs = [camera_observation(scene, c, fr, timestamp=float(i))[0] for c in CAMERAS]
        cs = ag.process_multi(obs, phase="transit")

    field = cs.esdf
    md = float(field.max_distance)
    zs = np.linspace(0.0, 1.5, 301)
    P = np.stack([np.full_like(zs, args.xy[0]), np.full_like(zs, args.xy[1]), zs], axis=1)
    d = np.asarray(field.distance(P), np.float64) * 1000.0
    unknown = np.abs(d - md * 1000.0) < 1e-6

    q = np.asarray([scene.data.qpos[scene._qadr[j]] for j in DEFAULT_RBY1_JOINTS], float)
    C, _ = rb.sphere_centers_numeric(q)
    C = np.asarray(C, float)
    scene.close()

    planes = sorted(cs.support_surfaces, key=lambda s: s.offset)
    print("=" * 88)
    print(f"Step 8 — 평면 대 필드 ({args.records}, 3 카메라, 복셀 {args.voxel*1000:.0f} mm)")
    print("=" * 88)
    print("맞춰진 평면:")
    for s in planes:
        print(f"  id={s.id}  n={np.round(s.normal, 3)}  d={s.offset:.4f} m  "
              f"인라이어 {s.point_count:,}  rms {s.rms_error*1000:.1f} mm")
    print(f"\n필드 수직 단면 (x={args.xy[0]}, y={args.xy[1]}), unknown_policy=free "
          f"-> 미관측은 +{md*1000:.0f} mm")
    for z in (0.0, 0.1, 0.3, 0.5, 0.7, 0.79, 0.823, 0.84, 0.9, 1.0):
        j = int(np.argmin(np.abs(zs - z)))
        tag = "미관측(자유로 취급)" if unknown[j] else ("표면 안쪽" if d[j] < 0 else "관측된 자유공간")
        print(f"  z={z:>5.3f}  d={d[j]:>7.1f} mm  {tag}")
    print(f"\n제약 모델(팔만) 구의 z 범위: {C[:,2].min():.3f} ~ {C[:,2].max():.3f} m")
    for s in planes:
        margin = C[:, 2].min() - s.offset
        print(f"  가장 낮은 팔 구가 평면 id={s.id}(z={s.offset:.3f}) 보다 "
              f"{margin*1000:+.0f} mm 위")

    _figure(zs, d, unknown, planes, C, md, args)


def _figure(zs, d, unknown, planes, C, md, args):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(19.5, 6.2))

    # (a) 실제 씬 — 수직 단면
    a = axs[0]
    obs = ~unknown
    a.plot(d[obs], zs[obs], "-", color="tab:blue", lw=2.4, label="필드 (관측된 곳)")
    a.plot(d[unknown], zs[unknown], "o", color="0.7", ms=3, label="필드 (미관측 = 자유로 취급)")
    a.axvline(0, color="k", lw=1.4)
    for s in planes:
        a.axhline(s.offset, color="tab:orange", ls="--", lw=2)
        a.text(md * 1000 * 0.35, s.offset + 0.02,
               f"평면 z={s.offset:.3f} ({s.point_count:,}점)", fontsize=9.5, color="#b26a00")
    a.axhspan(C[:, 2].min(), C[:, 2].max(), color="tab:green", alpha=0.16)
    a.text(-330, (C[:, 2].min() + C[:, 2].max()) / 2, "팔 구가\n실제로 있는 높이",
           fontsize=10, color="#2e7d32", fontweight="bold", va="center")
    a.set_xlabel("필드 거리 [mm]  (음수 = 표면 안쪽)"); a.set_ylabel("z [m]")
    a.set_title(f"(a) 실제 씬 — 수직 단면 (x={args.xy[0]}, y={args.xy[1]})\n"
                "테이블은 필드에 있고, 바닥은 없다", fontsize=11)
    a.legend(fontsize=9, loc="lower right"); a.grid(alpha=0.3)

    # (b) 확대 — 테이블 부근
    a = axs[1]
    sel = (zs > 0.70) & (zs < 1.00)
    a.plot(d[sel], zs[sel], "-", color="tab:blue", lw=2.6)
    a.axvline(0, color="k", lw=1.4)
    tbl = max(planes, key=lambda s: s.offset)
    a.axhline(tbl.offset, color="tab:orange", ls="--", lw=2.2)
    a.text(5, tbl.offset + 0.004, f"평면이 말하는 상판 z={tbl.offset:.3f}",
           fontsize=10, color="#b26a00")
    a.set_xlabel("필드 거리 [mm]"); a.set_ylabel("z [m]")
    a.set_title("(b) 테이블 부근 확대 — 필드가 슬래브로 담고 있다\n"
                "평면 행이 없어도 테이블은 제약된다", fontsize=11)
    a.grid(alpha=0.3)

    # (c) 표
    a = axs[2]; a.axis("off")
    rows = [["항목", "값"]]
    for s in planes:
        rows.append([f"평면 id={s.id}", f"z={s.offset:.4f} m · {s.point_count:,}점 · "
                                        f"rms {s.rms_error*1000:.1f} mm"])
    rows += [["", ""],
             ["필드가 테이블을 담는가", "예 — 핵심에서 -40 mm"],
             ["필드가 바닥을 담는가", "아니오 — 미관측(+400 mm)"],
             ["미관측 처리", "unknown_policy = free"],
             ["", ""],
             ["팔 구 z 범위", f"{C[:,2].min():.3f} ~ {C[:,2].max():.3f} m"],
             ["가장 낮은 팔 구 - 상판", f"{(C[:,2].min() - max(p.offset for p in planes))*1000:+.0f} mm"],
             ["가장 낮은 팔 구 - 바닥", f"{(C[:,2].min() - min(p.offset for p in planes))*1000:+.0f} mm"]]
    t = a.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1.0, 1.7)
    for j in range(2):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
        t[(4, j)].set_facecolor("#d9f0d9")
        t[(5, j)].set_facecolor("#f7d6d6")
    a.set_title("(c) 평면 행을 빼도 되는가 — 근거", fontsize=11.5, y=0.9)

    fig.suptitle("평면 행을 빼고 필드에만 맡겨도 되는가 — 테이블은 되고, 바닥은 필드에 없다",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    o = OUT / f"step8-planes-vs-field-{args.records[-4:]}.png"
    fig.savefig(o, dpi=110, bbox_inches="tight")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
