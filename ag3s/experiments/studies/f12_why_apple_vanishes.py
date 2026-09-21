"""F12 원인 규명 — 쥔 사과가 거리장에서 사라지는 것은 마스크 탓인가, 가림 탓인가.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \\
        benchmark.ag3s.experiments.studies.f12_why_apple_vanishes --frames 20

## 무엇을 가리는가

`grasp_damage.py` 로 **사과가 들어올려지는 순간(스텝 12) 거리장에서 사라진다**는 것을 확인했다
(사과 중심에서 필드가 +39~+121 mm 를 반환, 참값 -33 mm, 오차 최대 154 mm). 원인 후보가 둘이고
**고칠 층이 완전히 다르다**:

* **(a) 자기 필터 마스크** — `_robot_mask_for` 는 backproject 한 점이 로봇 구(20 mm 부풀림,
  `self_filter_inflation`) 안에 들어가면 그 픽셀을 지운다. 손가락 사이에 물린 사과는 손가락 구
  안에 들어가므로 **로봇으로 오인되어 지워질** 수 있다. 고칠 곳은 자기 필터.
* **(b) 가림** — 그리퍼가 카메라 시선을 막아 사과가 아예 안 보인다. 고칠 곳은 카메라 융합.

## 어떻게 가리는가

MuJoCo 세그멘테이션이 **픽셀별 body 참값**을 준다 (`CameraFrame.body_ids`). 그래서 추정할 것이
없다:

* **가림이면** — 사과 픽셀 수 자체가 0 에 가깝다. 앞을 그리퍼가 막고 있으므로 그 픽셀들은
  그리퍼로 라벨된다.
* **마스크면** — 사과 픽셀은 멀쩡히 있는데 그중 대부분이 로봇 마스크에 덮인다.

카메라 셋을 모두 본다. 파이프라인은 지금 `zed_left` **하나만** 쓰지만, 손목 카메라가 봤을지를
알아야 "카메라 융합" 이 답이 되는지 판단할 수 있다.
"""

import argparse
import json
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
    from benchmark.ag3s.experiments.common import figstyle
    figstyle.use_korean()


def main() -> None:
    _style()
    import mujoco

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default="run_0004")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--held", default="apple")
    args = ap.parse_args()

    run = load_run(args.records, limit=args.frames)
    scene = replay_scene(run)
    ag = AG3S(AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False}}),
        robot_model=build_robot_model(scene),
        constraint_robot_model=build_constraint_robot_model(scene, link_filter=ARM_LINKS))

    held_b = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, args.held)
    gl = {n: scene.model.jnt_qposadr[
              mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
          for n in ("gripper_finger_l1", "gripper_finger_l2")}

    print("=" * 104)
    print(f"F12 원인 규명 — {args.records}, 쥔 물체 {args.held!r}, 자기필터 부풀림 "
          f"{ag.config.pointcloud.self_filter_inflation*1000:.0f} mm")
    print("가림이면 사과 픽셀 자체가 없다.  마스크면 픽셀은 있는데 덮인다.")
    print("=" * 104)
    hdr = f"{'i':>2} {'쥠':>3}"
    for c in CAMERAS:
        hdr += f" | {c[:11]:>11} 픽셀 덮임 남음"
    print(hdr)

    rec = []
    for i, step in enumerate(run.steps):
        pose_scene(scene, step)
        wl = (scene.data.qpos[gl["gripper_finger_l2"]]
              - scene.data.qpos[gl["gripper_finger_l1"]]) * 1000
        row = dict(i=i, grasped=bool(wl < 80), cams={})
        line = f"{i:>2} {'O' if row['grasped'] else '.':>3}"
        for cam in CAMERAS:
            fr = scene.capture(cam)
            is_ap = np.asarray(fr.body_ids) == held_b
            n_ap = int(is_ap.sum())
            mask = ag._robot_mask_for(fr.depth, fr.camera_intrinsics, fr.T_base_cam,
                                      fr.robot_state)
            if mask is None:
                n_mask, n_left = 0, n_ap
            else:
                m = np.asarray(mask, bool)
                n_mask = int((is_ap & m).sum())
                n_left = n_ap - n_mask
            row["cams"][cam] = dict(n=n_ap, masked=n_mask, left=n_left)
            line += f" | {n_ap:>11} {n_mask:>4} {n_left:>4}"
            if cam == "zed_left":
                row["img"] = dict(is_ap=is_ap, mask=(None if mask is None
                                                     else np.asarray(mask, bool)),
                                  depth=np.asarray(fr.depth))
        print(line)
        rec.append(row)
    scene.close()
    _verdict(rec, args)
    _figure(rec, args)


def _verdict(rec, args):
    g = [r for r in rec if r["grasped"]]
    print("\n" + "=" * 104)
    print("판정 — 머리 카메라 zed_left (파이프라인이 실제로 쓰는 유일한 카메라)")
    pre = [r for r in rec if not r["grasped"] and r["i"] < 10]
    print(f"\n{'구간':<22} {'사과 픽셀':>10} {'그중 덮임':>10} {'덮인 비율':>10} {'남는 픽셀':>10}")
    for label, rows in (("파지 전 (0~9)", pre), ("파지 중 (10~18)", g)):
        if not rows:
            continue
        n = np.mean([r["cams"]["zed_left"]["n"] for r in rows])
        m = np.mean([r["cams"]["zed_left"]["masked"] for r in rows])
        l = np.mean([r["cams"]["zed_left"]["left"] for r in rows])
        pct = (m / n * 100) if n else float("nan")
        print(f"{label:<22} {n:>10.0f} {m:>10.0f} {pct:>9.0f}% {l:>10.0f}")

    if g:
        n = np.mean([r["cams"]["zed_left"]["n"] for r in g])
        m = np.mean([r["cams"]["zed_left"]["masked"] for r in g])
        l = np.mean([r["cams"]["zed_left"]["left"] for r in g])
        if n < 20:
            v = "(b) 가림 — 사과 픽셀 자체가 거의 없다"
        elif m / max(n, 1) > 0.6:
            v = "(a) 자기 필터 마스크 — 픽셀은 보이는데 로봇으로 오인되어 지워진다"
        elif l > 20:
            v = "둘 다 아님 — 픽셀이 살아남는데도 필드에 없다. 적분 쪽을 봐야 한다"
        else:
            v = "혼합 — 가림과 마스크가 함께 작용"
        print(f"\n>>> 판정: {v}")

    print("\n손목 카메라가 봤는가 (지금 파이프라인은 안 쓴다)")
    for cam in CAMERAS[1:]:
        if g:
            n = np.mean([r["cams"][cam]["n"] for r in g])
            l = np.mean([r["cams"][cam]["left"] for r in g])
            print(f"   {cam:<14} 파지 중 사과 픽셀 평균 {n:>7.0f},  마스크 후 남는 픽셀 {l:>7.0f}")


def _figure(rec, args):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    idx = [r["i"] for r in rec]
    g = [r for r in rec if r["grasped"]]
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.9))

    # (a) 실제 씬 — 파지 중 한 프레임의 머리 카메라
    a = axs[0]
    r0 = g[len(g) // 2] if g else rec[0]
    im = r0["img"]
    rgb = np.zeros(im["is_ap"].shape + (3,), float)
    d = im["depth"]
    dn = np.clip((d - np.nanmin(d)) / max(np.nanmax(d) - np.nanmin(d), 1e-9), 0, 1)
    rgb[..., 0] = rgb[..., 1] = rgb[..., 2] = 1 - dn * 0.75
    if im["mask"] is not None:
        rgb[im["mask"]] = [0.35, 0.55, 0.95]          # 로봇 마스크 = 파랑
    both = im["is_ap"] & (im["mask"] if im["mask"] is not None else False)
    only = im["is_ap"] & ~(im["mask"] if im["mask"] is not None else False)
    rgb[only] = [0.15, 0.75, 0.25]                     # 살아남은 사과 = 초록
    rgb[both] = [0.95, 0.15, 0.15]                     # 덮인 사과 = 빨강
    a.imshow(rgb, interpolation="nearest")
    a.set_title(f"(a) 실제 씬 — zed_left, 스텝 {r0['i']} (쥐고 있다)\n"
                "회색=깊이  파랑=로봇 마스크", fontsize=11)
    a.legend(handles=[Patch(color=(0.95, 0.15, 0.15), label="사과인데 마스크에 덮임"),
                      Patch(color=(0.15, 0.75, 0.25), label="사과이고 살아남음"),
                      Patch(color=(0.35, 0.55, 0.95), label="로봇 마스크")],
             fontsize=8.5, loc="upper right")
    a.set_xticks([]); a.set_yticks([])

    # (b) 스텝별
    a = axs[1]
    n = [r["cams"]["zed_left"]["n"] for r in rec]
    m = [r["cams"]["zed_left"]["masked"] for r in rec]
    l = [r["cams"]["zed_left"]["left"] for r in rec]
    for r in g:
        a.axvspan(r["i"] - 0.5, r["i"] + 0.5, color="tab:orange", alpha=0.15)
    a.plot(idx, n, "o-", color="k", lw=2, ms=5, label="사과 픽셀 (참값)")
    a.plot(idx, m, "s-", color="crimson", lw=2, ms=5, label="그중 마스크에 덮임")
    a.plot(idx, l, "^-", color="tab:green", lw=2, ms=5, label="살아남아 적분되는 픽셀")
    a.axvline(11.5, color="purple", lw=2, ls="--")
    a.text(11.7, max(n) * 0.82, "여기서부터 필드에\n사과가 없다", color="purple", fontsize=9.5,
           fontweight="bold")
    a.set_title("(b) zed_left — 사과 픽셀은 남는가, 지워지는가\n주황 = 쥐고 있는 구간",
                fontsize=11)
    a.set_xlabel("스텝"); a.set_ylabel("픽셀 수"); a.legend(fontsize=9); a.grid(alpha=0.3)

    # (c) 표
    a = axs[2]; a.axis("off")
    pre = [r for r in rec if not r["grasped"] and r["i"] < 10]
    rows = [["구간", "사과 픽셀", "덮임", "덮인 비율", "남음"]]
    for label, rr in (("파지 전 0~9", pre), ("파지 중 10~18", g)):
        if not rr:
            continue
        nn = np.mean([r["cams"]["zed_left"]["n"] for r in rr])
        mm = np.mean([r["cams"]["zed_left"]["masked"] for r in rr])
        ll = np.mean([r["cams"]["zed_left"]["left"] for r in rr])
        rows.append([label, f"{nn:.0f}", f"{mm:.0f}",
                     f"{mm/nn*100:.0f}%" if nn else "-", f"{ll:.0f}"])
    rows.append(["", "", "", "", ""])
    for cam in CAMERAS[1:]:
        if g:
            nn = np.mean([r["cams"][cam]["n"] for r in g])
            ll = np.mean([r["cams"][cam]["left"] for r in g])
            rows.append([f"{cam} (미사용)", f"{nn:.0f}", "-", "-", f"{ll:.0f}"])
    t = a.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.0, 1.9)
    for j in range(5):
        t[(0, j)].set_facecolor("#dddddd"); t[(0, j)].set_text_props(fontweight="bold")
    a.set_title("(c) 파지 전후 비교 — 그리고 손목 카메라는 봤는가\n"
                "파이프라인은 지금 zed_left 하나만 쓴다", fontsize=11, y=0.9)

    fig.suptitle("F12 — 쥔 사과가 거리장에서 사라지는 이유: 마스크인가 가림인가",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    o = OUT / f"f12-why-apple-vanishes-{args.records[-4:]}.png"
    fig.savefig(o, dpi=110, bbox_inches="tight")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
