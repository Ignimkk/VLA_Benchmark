"""I2 검증 — legacy 필드와 cuRobo 필드가 **같은 씬에서** 얼마나 다른가.

## 왜 두 단계인가 (2026-09-25, 사용자 판정)

예전에는 한 프로세스에서 backend 를 바꿔 가며 두 번 돌렸다. 그것이 **T0 불변식에 걸린다**:

    esdf.backend='curobo' 인데 legacy EsdfBuilder 가 이 프로세스에서 1 번 만들어졌다.
    T0 의 즉시 실패 조건이다                                  (`runtime/pipeline.py:906`)

불변식이 옳다. 두 backend 가 한 프로세스에 공존하면 **어느 필드가 판정에 쓰였는지 기록만
보고는 되짚을 수 없고**, 그것이 바로 이 검토가 반복해서 만난 실패다. 그래서 불변식을
건드리지 않고 **프로세스를 나눈다**:

    1) dump  — backend 하나로 한 프로세스. 필드가 답한 구 거리를 npz 로 낸다. 두 번 돌린다.
    2) compare — npz 둘을 읽어 대조한다. 파이프라인을 아예 임포트하지 않는다.

`dump` 는 **씬을 seed 로 결정론적으로 만든다**. 두 실행이 같은 seed 면 같은 씬이고, 같은
자세의 같은 구에 두 필드가 답한 값을 짝지어 비교할 수 있다. npz 에 씬 지문을 같이 실어
`compare` 가 그것을 확인한다 — 다른 씬의 두 덤프를 견주면 backend 차이가 아니라 씬 차이를
재게 된다.

## 쓰는 법

    V=/mnt/dev/work/.venv-openpi-live/bin/python
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work $V -m \\
        benchmark.ag3s.experiments.live.verify_backend dump --backend legacy \\
        --seed 101 --out outputs/verify/R/R1_legacy.npz
    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work $V -m \\
        benchmark.ag3s.experiments.live.verify_backend dump --backend curobo \\
        --seed 101 --out outputs/verify/R/R1_curobo.npz
    PYTHONPATH=/mnt/dev/work $V -m \\
        benchmark.ag3s.experiments.live.verify_backend compare \\
        --legacy outputs/verify/R/R1_legacy.npz --curobo outputs/verify/R/R1_curobo.npz \\
        --out outputs/verify/R/R1_verify_backend.json

## 재는 것

1. **배선** — `curobo` 일 때 legacy `EsdfBuilder` 가 한 번도 안 만들어지는가
   (이제 `dump` 가 그 프로세스 안에서 직접 확인해 npz 에 적는다)
2. **필드 품질** — 부호, 두 backend 의 거리 차이 분포
3. **로봇 구 여유거리** — 같은 자세의 구에서 두 backend 가 답하는 여유거리

**낙관 오차가 판정의 축이다.** `필드가 답한 거리 − 참 거리` 가 양수면 필드가 실제보다 넓다고
말한 것이고 그것만이 위험하다. 여기서는 참 거리 대신 두 backend 를 견주므로, cuRobo 가
legacy 보다 **크게** 답하는 쪽이 확인이 필요한 방향이다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


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


def scene_fingerprint(obs, centres, radii) -> str:
    """이 덤프가 어느 씬에서 나왔는가. `compare` 가 두 덤프의 씬이 같은지 확인하는 근거.

    seed 만 적어서는 부족하다 — MuJoCo 버전이나 모델 XML 이 달라지면 같은 seed 가 다른
    씬을 낸다. 실제로 필드에 들어간 **관측과 구 위치**를 해시한다.
    """
    h = hashlib.sha256()
    for o in obs:
        h.update(np.asarray(o.depth, np.float32).tobytes())
        h.update(np.asarray(o.T_base_cam, np.float64).tobytes())
    h.update(np.asarray(centres, np.float64).tobytes())
    h.update(np.asarray(radii, np.float64).tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------- dump


def cmd_dump(args) -> int:
    """backend **하나**로 한 프로세스. 다른 backend 는 이 프로세스에 들어오지 않는다."""
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.runtime.pipeline import AG3S

    scene, obs, target_true, filter_robot, constraint_robot = build_scene(
        args.seed, args.target, args.height, args.width)
    try:
        q = np.asarray(obs[0].robot_state, np.float64)
        centres, radii = constraint_robot.sphere_centers_numeric(q)
        centres = np.asarray(centres, np.float64).reshape(-1, 3)
        radii = np.asarray(radii, np.float64).reshape(-1)

        esdf = {"voxel_size": args.coarse, "max_distance": args.max_distance,
                "exclude_support_surfaces": False, "backend": args.backend}
        if args.backend == "curobo":
            esdf["fine_voxel_size"] = args.fine
            esdf["tsdf_voxel_size"] = args.tsdf_voxel
        config = AG3SConfig.from_dict({"collision_backend": "esdf",
                                       "pointcloud": {"range_max": 2.0},
                                       "esdf": esdf})
        ag3s = AG3S(config, robot_model=filter_robot,
                    constraint_robot_model=constraint_robot)
        t = time.perf_counter()
        cs, _debug = ag3s.process_multi_debug(obs, phase="approach",
                                              active_manipulators=["left"])
        elapsed = (time.perf_counter() - t) * 1000.0

        field = cs.esdf
        if field is None:
            raise SystemExit(f"{args.backend}: 필드가 만들어지지 않았다 — "
                             f"status={cs.status.value} notes={cs.notes}")
        d = np.asarray(field.distance(centres), np.float64)

        # **배선 확인은 이 프로세스 안에서만 뜻이 있다.** 두 backend 가 한 프로세스에 없으므로,
        # "legacy 가 만들어졌는가" 는 여기서 물어야 답이 의미를 갖는다.
        builder = type(ag3s._esdf_builder).__name__ if ag3s._esdf_builder else None
        from benchmark.ag3s.fields.esdf import EsdfBuilder
        legacy_instances = int(EsdfBuilder.instances_created)

        meta = {
            "backend": args.backend,
            "seed": int(args.seed),
            "target": args.target,
            "builder_class": builder,
            "legacy_instances_created": legacy_instances,
            "elapsed_ms": elapsed,
            "status": cs.status.value,
            "grounding": cs.grounding_status.value,
            "validity": cs.validity.value,
            "has_target": bool(cs.has_target),
            "field_class": type(field).__name__,
            "voxel_size_m": float(field.grid.voxel_size),
            "n_layers": len(getattr(field, "layers", (field,))),
            "stats_backend": field.stats.get("backend", "legacy"),
            "outside_query_fraction": float(field.outside_query_fraction),
            "unknown_fraction": (None if field.unknown_fraction is None
                                 else float(field.unknown_fraction)),
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
            "max_distance_m": float(args.max_distance),
            "esdf_margin_m": float(args.esdf_margin),
            "scene_fingerprint": scene_fingerprint(obs, centres, radii),
        }
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            sphere_distance=d, sphere_centres=centres, sphere_radii=radii,
            link_names=np.asarray(list(getattr(constraint_robot,
                                               "sphere_link_names", [])), dtype=object),
            target_true=np.asarray(target_true, np.float64),
            meta=np.asarray(json.dumps(meta, ensure_ascii=False)))
    finally:
        scene.close()

    print(f"wrote {out}")
    print(f"  backend {args.backend}  builder {builder}  "
          f"legacy 생성 {legacy_instances} 회  계층 {meta['n_layers']}  {elapsed:.0f} ms")
    print(f"  씬 지문 {meta['scene_fingerprint']}  구 {len(d)}  "
          f"여유 중앙 {np.median(d - radii - args.esdf_margin)*1000:+.2f} mm")
    return 0


# ---------------------------------------------------------------------------- compare


def _load_dump(path: str) -> tuple[dict, dict]:
    blob = np.load(path, allow_pickle=True)
    meta = json.loads(str(blob["meta"].item()))
    arrays = {k: blob[k] for k in blob.files if k != "meta"}
    return meta, arrays


def cmd_compare(args) -> int:
    """npz 둘을 대조한다. **파이프라인을 임포트하지 않는다** — 그럴 이유가 없고, 임포트하면
    이 단계가 다시 한 프로세스에 두 backend 를 부르는 길이 열린다."""
    ml, al = _load_dump(args.legacy)
    mc, ac = _load_dump(args.curobo)

    if ml["backend"] != "legacy" or mc["backend"] != "curobo":
        raise SystemExit(f"덤프의 backend 가 기대와 다르다: --legacy 는 {ml['backend']!r}, "
                         f"--curobo 는 {mc['backend']!r}")
    if ml["scene_fingerprint"] != mc["scene_fingerprint"]:
        raise SystemExit(
            f"두 덤프가 **다른 씬**에서 나왔다 (지문 {ml['scene_fingerprint']} ≠ "
            f"{mc['scene_fingerprint']}). 그대로 견주면 backend 차이가 아니라 씬 차이를 "
            f"재게 된다. 같은 --seed 로 두 dump 를 다시 돌려라 "
            f"(legacy seed={ml['seed']}, curobo seed={mc['seed']})")

    margin = float(ml["esdf_margin_m"])
    max_distance = float(ml["max_distance_m"])
    dl = np.asarray(al["sphere_distance"], np.float64)
    dc = np.asarray(ac["sphere_distance"], np.float64)
    radii = np.asarray(al["sphere_radii"], np.float64)
    if dl.shape != dc.shape:
        raise SystemExit(f"구 개수가 다르다: legacy {dl.shape} vs curobo {dc.shape}")

    report: dict = {"stage": "I2", "seed": ml["seed"],
                    "source": "fresh_mujoco_scene", "used_saved_run": False,
                    "synthetic_attention": True,
                    "process_split": True,
                    "process_split_why":
                        "T0 불변식(pipeline.py:906)이 한 프로세스에 두 backend 를 금지한다. "
                        "불변식을 건드리지 않고 프로세스를 나눴다 (2026-09-25 사용자 판정)",
                    "scene_fingerprint": ml["scene_fingerprint"],
                    "checks": {}}

    # ── 조건 1: 배선 ──
    report["checks"]["curobo_builder_used"] = {
        "pass": mc["builder_class"] == "CuroboFieldBuilder",
        "legacy_builder_class": ml["builder_class"],
        "curobo_builder_class": mc["builder_class"]}
    report["checks"]["no_legacy_under_curobo"] = {
        # 이제 **같은 프로세스 안에서 센 값**이다. 예전에는 두 backend 가 섞여 있어
        # 이 숫자가 무엇을 세는지 자체가 불분명했다.
        "pass": mc["legacy_instances_created"] == 0,
        "legacy_instances_created_in_curobo_process": mc["legacy_instances_created"],
        "stats_backend": mc["stats_backend"]}
    report["checks"]["two_tiers"] = {
        "pass": mc["n_layers"] == 2,
        "n_layers": mc["n_layers"],
        "note": "미세 계층은 grounding 된 target 중심에 놓인다 — target 이 안 서면 1 이 된다",
        "curobo_has_target": mc.get("has_target")}

    # ── 조건 2: 두 backend 의 거리 차이 ──
    diff = dc - dl
    # **거리 포화를 빼고 다시 센다.** legacy 는 `max_distance` 에서 값을 자르므로 그 지점의
    # 차이는 낙관이 아니라 자른 것의 산물이다. 자른 값과 안 자른 값을 견주면 C2("거친 33 mm
    # 대 미세 2 mm 는 측정법 산물")와 같은 종류의 오판이 된다.
    sat = dl >= max_distance - 1e-9
    unsat = ~sat
    clear = dl - radii - margin
    near = unsat & (clear < 0.10)    # 여유 100 mm 안 — 판정이 실제로 갈리는 띠
    report["backend_delta"] = {
        "note": "cuRobo - legacy. 양수면 cuRobo 가 더 멀다고 답한 것 = 확인이 필요한 방향",
        "all_spheres": _q(diff),
        "legacy_saturated": {"n": int(sat.sum()),
                             "max_distance_mm": max_distance * 1000,
                             "why": "legacy 는 max_distance 에서 자른다. 그 지점의 차이는 "
                                    "낙관이 아니라 포화의 산물이다"},
        "unsaturated_only": _q(diff[unsat]),
        "near_band_only": _q(diff[near]),
        "near_band_note": "legacy 여유거리 100 mm 미만인 구만 — 판정이 실제로 갈리는 띠",
        "n_curobo_more_optimistic": int(np.count_nonzero(diff > 0)),
        "n_curobo_more_conservative": int(np.count_nonzero(diff < 0))}

    # ── 위반 집합의 귀속 ──
    names = [str(x) for x in np.asarray(al["link_names"]).reshape(-1)]
    vl = np.flatnonzero(dl - radii - margin < 0)
    vc = np.flatnonzero(dc - radii - margin < 0)

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

    per_backend = {}
    for tag, m, d in (("legacy", ml, dl), ("curobo", mc, dc)):
        entry = dict(m)
        entry["sphere_distance"] = _q(d)
        entry["clearance"] = _q(d - radii - margin)
        entry["n_violated"] = int(np.count_nonzero(d - radii - margin < 0))
        per_backend[tag] = entry
    report["per_backend"] = per_backend

    # ── 조건 3: 부호 규약이 같은가 ──
    report["checks"]["same_sign_convention"] = {
        "pass": bool(np.sign(dl).sum() * np.sign(dc).sum() > 0
                     or (dl > 0).mean() == (dc > 0).mean()),
        "legacy_positive_fraction": float((dl > 0).mean()),
        "curobo_positive_fraction": float((dc > 0).mean())}

    report["verdict"] = {
        "pass": all(c.get("pass") for c in report["checks"].values()),
        "checks": {k: bool(v.get("pass")) for k, v in report["checks"].items()}}

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    print(f"  씬 지문 {ml['scene_fingerprint']} (두 덤프 일치)")
    for name, c in report["checks"].items():
        print(f"  [{'PASS' if c.get('pass') else 'FAIL'}] {name}")
    n = len(dl)
    for b in ("legacy", "curobo"):
        r = per_backend[b]
        print(f"  {b:7s} {str(r['builder_class']):20s} {r['field_class']:16s} "
              f"layers {r['n_layers']}  여유 중앙 {r['clearance']['median_mm']:+7.2f} mm  "
              f"최악 {r['clearance']['min_mm']:+7.2f} mm  위반 {r['n_violated']:3d}/{n}  "
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
    return 0 if report["verdict"]["pass"] else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="backend 하나로 한 프로세스. 구 거리를 npz 로 낸다")
    d.add_argument("--backend", choices=("legacy", "curobo"), required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--seed", type=int, default=101,
                   help="씬 seed. **두 dump 가 같아야 한다** — compare 가 확인한다")
    d.add_argument("--target", default="apple")
    d.add_argument("--height", type=int, default=480)
    d.add_argument("--width", type=int, default=640)
    d.add_argument("--coarse", type=float, default=0.020)
    d.add_argument("--fine", type=float, default=0.005)
    d.add_argument("--tsdf-voxel", type=float, default=0.005)
    d.add_argument("--max-distance", type=float, default=0.40)
    d.add_argument("--esdf-margin", type=float, default=0.05)
    d.set_defaults(func=cmd_dump)

    c = sub.add_parser("compare", help="npz 둘을 대조한다. 파이프라인을 임포트하지 않는다")
    c.add_argument("--legacy", required=True)
    c.add_argument("--curobo", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
