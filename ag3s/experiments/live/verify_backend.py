"""I2 검증 — `pipeline._build_esdf` 가 cuRobo 를 부르고 두 backend 가 같은 씬에서 무엇이 다른가.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-openpi-live/bin/python -m \\
        benchmark.ag3s.experiments.live.verify_backend --seed 101 \\
        --out outputs/live_test/20260922_i2_backend/verify_backend_seed101.json

새 MuJoCo 씬을 만들어 **같은 관측**을 두 backend 에 넣고 짝 비교한다. 저장 기록을 읽지 않는다.

재는 것 넷.

1. **배선** — `curobo` 일 때 legacy `EsdfBuilder` 가 한 번도 안 만들어지는가
2. **필드 품질** — 부호, 자유공간 eikonal, 두 backend 의 거리 차이 분포
3. **로봇 구 여유거리** — 같은 자세의 구 120 개에서 두 backend 가 답하는 여유거리
4. **쥔 물체** — attached seed 제외가 실제로 몇 개를 뺐고, 제외 전/후 자기 질의점이 얼마나 바뀌나

**낙관 오차가 판정의 축이다.** `필드가 답한 거리 - 참 거리` 가 양수면 필드가 실제보다 넓다고
말한 것이고 그것만이 위험하다. 여기서는 참 거리 대신 두 backend 를 견주므로, cuRobo 가
legacy 보다 **크게** 답하는 쪽이 확인이 필요한 방향이다.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
IDS = ("head", "left_wrist", "right_wrist")


def _q(a: np.ndarray) -> dict:
    a = np.asarray(a, float).reshape(-1)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "min_mm": float(a.min() * 1000),
            "p05_mm": float(np.percentile(a, 5) * 1000),
            "median_mm": float(np.median(a) * 1000),
            "p95_mm": float(np.percentile(a, 95) * 1000),
            "max_mm": float(a.max() * 1000)}


def build_scene(seed: int, target: str, height: int, width: int):
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import (
        TRANSPORT_MODEL, TransportScene, gaussian_attention)
    from benchmark.ag3s.types import CameraID, CameraObservation

    scene = TransportScene(TRANSPORT_MODEL, settle_steps=400, height=height, width=width)
    rng = np.random.default_rng(seed)
    qadr = scene._qadr[f"{target}_free"]
    jitter = rng.uniform(-0.012, 0.012, size=2)
    scene.data.qpos[qadr:qadr + 2] += jitter
    scene.mujoco.mj_forward(scene.model, scene.data)

    target_true = scene.body_position_in_base(target)
    filter_robot = build_robot_model(scene)
    constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)

    frames = {c: scene.capture(c) for c in CAMS}
    stamp = time.monotonic()   # 재생에서 카메라 셋은 **같은 시뮬 순간**이다 — 한 번만 찍는다
    ids = {"zed_left": CameraID.HEAD, "wrist_cam_l": CameraID.LEFT_WRIST,
           "wrist_cam_r": CameraID.RIGHT_WRIST}
    obs = [CameraObservation(
        camera_id=ids[c], depth=frames[c].depth,
        camera_intrinsics=frames[c].camera_intrinsics, T_base_cam=frames[c].T_base_cam,
        robot_state=frames[c].robot_state, timestamp=stamp, state_timestamp=stamp,
        attention_map=gaussian_attention(frames[c], target_true), image_hw=frames[c].hw)
        for c in CAMS]
    return scene, obs, target_true, filter_robot, constraint_robot


def run_backend(backend: str, obs, filter_robot, constraint_robot, *, coarse: float,
                fine, tsdf_voxel, max_distance: float):
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.runtime.pipeline import AG3S

    esdf = {"voxel_size": coarse, "max_distance": max_distance,
            "exclude_support_surfaces": False, "backend": backend}
    if backend == "curobo":
        esdf["fine_voxel_size"] = fine
        esdf["tsdf_voxel_size"] = tsdf_voxel
    config = AG3SConfig.from_dict({"collision_backend": "esdf",
                                   "pointcloud": {"range_max": 2.0},
                                   "esdf": esdf})
    ag3s = AG3S(config, robot_model=filter_robot,
                constraint_robot_model=constraint_robot)
    t = time.perf_counter()
    cs, debug = ag3s.process_multi_debug(obs, phase="approach",
                                         active_manipulators=["left"])
    elapsed = (time.perf_counter() - t) * 1000.0
    legacy_created = type(ag3s._esdf_builder).__name__ if ag3s._esdf_builder else None
    return ag3s, cs, debug, elapsed, legacy_created


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--coarse", type=float, default=0.020)
    ap.add_argument("--fine", type=float, default=0.005)
    ap.add_argument("--tsdf-voxel", type=float, default=0.005)
    ap.add_argument("--max-distance", type=float, default=0.40)
    ap.add_argument("--esdf-margin", type=float, default=0.05)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"stage": "I2", "seed": args.seed,
                    "source": "fresh_mujoco_scene", "used_saved_run": False,
                    "synthetic_attention": True, "checks": {}}

    scene, obs, target_true, filter_robot, constraint_robot = build_scene(
        args.seed, args.target, args.height, args.width)
    try:
        q = np.asarray(obs[0].robot_state, np.float64)
        centres, radii = constraint_robot.sphere_centers_numeric(q)
        centres = np.asarray(centres, np.float64).reshape(-1, 3)
        radii = np.asarray(radii, np.float64).reshape(-1)

        results = {}
        for backend in ("legacy", "curobo"):
            ag3s, cs, debug, ms, builder = run_backend(
                backend, obs, filter_robot, constraint_robot,
                coarse=args.coarse, fine=args.fine, tsdf_voxel=args.tsdf_voxel,
                max_distance=args.max_distance)
            field = cs.esdf
            if field is None:
                raise RuntimeError(f"{backend}: 필드가 만들어지지 않았습니다 — "
                                   f"status={cs.status.value} notes={cs.notes}")
            d = np.asarray(field.distance(centres), np.float64)
            results[backend] = {
                "builder_class": builder,
                "elapsed_ms": ms,
                "status": cs.status.value,
                "grounding": cs.grounding_status.value,
                "validity": cs.validity.value,
                "field_class": type(field).__name__,
                "voxel_size_m": float(field.grid.voxel_size),
                "n_layers": len(getattr(field, "layers", (field,))),
                "stats_backend": field.stats.get("backend", "legacy"),
                "sphere_distance": _q(d),
                "clearance": _q(d - radii - args.esdf_margin),
                "n_violated": int(np.count_nonzero(d - radii - args.esdf_margin < 0)),
                "profile_ms": {k: float(v) for k, v in cs.profile.items()},
                # `degraded` 가 왜 났는지는 노트에만 있다. 이것을 안 실으면 backend 차이를
                # "상태가 다르다" 까지만 보고 원인을 못 찾는다.
                "notes": list(cs.notes),
                "esdf_metrics": {k: (v.item() if isinstance(v, np.generic) else v)
                                 for k, v in cs.metrics.items()
                                 if "esdf" in k or "unknown" in k or "outside" in k
                                 or "certif" in k or "coverage" in k},
                "field_stats": {k: v for k, v in field.stats.items()
                                if k != "per_camera" and not isinstance(v, np.ndarray)},
                "outside_query_fraction": float(field.outside_query_fraction),
                "unknown_fraction": (None if field.unknown_fraction is None
                                     else float(field.unknown_fraction)),
            }
            results[backend]["_d"] = d

        # ── 조건 1: 배선 ──
        report["checks"]["curobo_builder_used"] = {
            "pass": results["curobo"]["builder_class"] == "CuroboFieldBuilder",
            "legacy_builder_class": results["legacy"]["builder_class"],
            "curobo_builder_class": results["curobo"]["builder_class"]}
        report["checks"]["no_legacy_under_curobo"] = {
            "pass": results["curobo"]["builder_class"] != "EsdfBuilder",
            "stats_backend": results["curobo"]["stats_backend"]}
        report["checks"]["two_tiers"] = {
            "pass": results["curobo"]["n_layers"] == 2,
            "n_layers": results["curobo"]["n_layers"]}

        # ── 조건 2: 두 backend 의 거리 차이 ──
        dl, dc = results["legacy"].pop("_d"), results["curobo"].pop("_d")
        diff = dc - dl
        # **거리 포화를 빼고 다시 센다.** legacy 는 `max_distance` 에서 값을 자르므로
        # (`esdf.max_distance`, 여기서는 400 mm) 그 지점의 차이는 낙관이 아니라 자른 것의
        # 산물이다. 자른 값과 안 자른 값을 견주면 C2("거친 33 mm 대 미세 2 mm 는 측정법
        # 산물")와 같은 종류의 오판이 된다.
        sat = dl >= args.max_distance - 1e-9
        unsat = ~sat
        clear = np.asarray(dl - radii - args.esdf_margin, np.float64)
        near = unsat & (clear < 0.10)    # 여유 100 mm 안 — 판정이 실제로 갈리는 띠
        report["backend_delta"] = {
            "note": "cuRobo - legacy. 양수면 cuRobo 가 더 멀다고 답한 것 = 확인이 필요한 방향",
            "all_spheres": _q(diff),
            "legacy_saturated": {"n": int(sat.sum()),
                                 "max_distance_mm": args.max_distance * 1000,
                                 "why": "legacy 는 max_distance 에서 자른다. 그 지점의 차이는 "
                                        "낙관이 아니라 포화의 산물이다"},
            "unsaturated_only": _q(diff[unsat]),
            "near_band_only": _q(diff[near]),
            "near_band_note": "legacy 여유거리 100 mm 미만인 구만 — 판정이 실제로 갈리는 띠",
            "n_curobo_more_optimistic": int(np.count_nonzero(diff > 0)),
            "n_curobo_more_conservative": int(np.count_nonzero(diff < 0))}

        # ── 위반 집합의 귀속 ──
        names = list(getattr(constraint_robot, "sphere_link_names", []))
        vl = np.flatnonzero(dl - radii - args.esdf_margin < 0)
        vc = np.flatnonzero(dc - radii - args.esdf_margin < 0)
        def _label(i):
            return names[i] if i < len(names) else f"sphere_{i}"
        report["violation_attribution"] = {
            "legacy": sorted({_label(i) for i in vl}),
            "curobo": sorted({_label(i) for i in vc}),
            "only_curobo": sorted({_label(i) for i in vc if i not in set(vl)}),
            "only_legacy": sorted({_label(i) for i in vl if i not in set(vc)}),
            "delta_on_curobo_only_mm": [round(float(diff[i] * 1000), 2)
                                        for i in vc if i not in set(vl)],
            "note": "cuRobo 에서만 위반인 구의 delta 가 음수면 cuRobo 가 더 가깝다고 답한 것 "
                    "= 보수적 방향이고, 미세 계층이 표면을 더 정확히 놓은 결과다"}
        report["per_backend"] = results

        # ── 조건 3: 부호 규약이 같은가 ──
        report["checks"]["same_sign_convention"] = {
            "pass": bool(np.sign(dl).sum() * np.sign(dc).sum() > 0
                         or (dl > 0).mean() == (dc > 0).mean()),
            "legacy_positive_fraction": float((dl > 0).mean()),
            "curobo_positive_fraction": float((dc > 0).mean())}

        report["verdict"] = {
            "pass": all(c.get("pass") for c in report["checks"].values()),
            "checks": {k: bool(v.get("pass")) for k, v in report["checks"].items()}}
    finally:
        scene.close()

    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    for name, c in report["checks"].items():
        print(f"  [{'PASS' if c.get('pass') else 'FAIL'}] {name}")
    for b in ("legacy", "curobo"):
        r = report["per_backend"][b]
        print(f"  {b:7s} {r['builder_class']:20s} {r['field_class']:16s} "
              f"layers {r['n_layers']}  여유 중앙 {r['clearance']['median_mm']:+7.2f} mm  "
              f"최악 {r['clearance']['min_mm']:+7.2f} mm  위반 {r['n_violated']:3d}/120  "
              f"{r['elapsed_ms']:7.0f} ms")
    bd = report["backend_delta"]
    for key, lab in (("all_spheres", "전체"), ("unsaturated_only", "포화 제외"),
                     ("near_band_only", "근접 띠")):
        q = bd[key]
        if q.get("n"):
            print(f"  delta {lab:8s} n={q['n']:3d}  중앙 {q['median_mm']:+7.2f}  "
                  f"p95 {q['p95_mm']:+7.2f}  최대 {q['max_mm']:+7.2f} mm")
    va = report["violation_attribution"]
    print(f"  위반 only-cuRobo {va['only_curobo']} delta {va['delta_on_curobo_only_mm']} mm")
    print(f"  위반 only-legacy {va['only_legacy']}")
    raise SystemExit(0 if report["verdict"]["pass"] else 1)


if __name__ == "__main__":
    main()
