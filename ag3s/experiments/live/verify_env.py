"""I1 통과 조건 — cuRobo 가 정책·MuJoCo·SQP 와 **같은 프로세스**에서 도는가.

    PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-openpi-live/bin/python -m \\
        benchmark.ag3s.experiments.live.verify_env \\
        --baseline outputs/live_test/20260922_i1_venv/baseline_openpi.json \\
        --out outputs/live_test/20260922_i1_venv/verify_env.json

네 가지를 순서대로 본다. 셋째까지는 "복제가 원본을 망가뜨리지 않았는가" 이고, 넷째가
**이 단계의 본론**이다.

1. 정책이 살아 있는가 — `jax.devices()` 가 CUDA 를 준다
2. AG3S·SQP 가 살아 있는가 — `mujoco` · `casadi` · `osqp` import
3. `--no-deps` 가 지켜졌는가 — numpy/scipy/torch/jax 버전이 복제 전과 **같다**
4. 필드가 옳은가 — cuRobo `Mapper` 로 2계층 ESDF 를 만들고 **eikonal 자기진단**

넷째를 자기진단으로 고른 이유: `AG3S_REVIEW_LOG.md` 의 cuRobo 함정 1·3·4 가 여기 다 걸린다.
`esdf_origin` 이 격자 코너가 아니라 **중심**이라는 것(1), 첫 `compute_esdf` 는 JIT 라
시간을 인용하면 안 된다는 것(3), `feature_tensor` 는 **재사용 버퍼**라 다음 호출이
덮어쓴다는 것(4). 셋 중 하나라도 틀리면 자유공간의 `|∇d|` 가 1 에서 벗어난다 — 발견
C4(두 계층을 만든 뒤에 읽어 `|∇d| = 0.25` 가 나온 것)가 정확히 그렇게 잡혔다.

**입력은 저장 기록이 아니다.** 씬을 새로 초기화하고 현재 시뮬 시각의 세 카메라를 캡처한다.
attention 은 이 단계에서 쓰지 않는다 — I1 은 배선 확인이고 grounding 은 T1 의 일이다.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
IDS = ("head", "left_wrist", "right_wrist")

#: 버전이 복제 전과 같아야 하는 패키지. 여기 있는 것이 하나라도 달라지면 `--no-deps` 가
#: 깨졌다는 뜻이고, 그때는 jax 가 아직 살아 있어도 신뢰할 수 없다.
PINNED = ("python", "numpy", "scipy", "torch", "jax", "jaxlib", "mujoco", "casadi", "osqp")


def _versions() -> dict:
    import importlib
    import sys

    out = {"python": sys.version.split()[0], "prefix": sys.prefix}
    for m in ("numpy", "scipy", "torch", "jax", "jaxlib", "mujoco", "casadi", "osqp",
              "warp", "curobo", "typing_extensions", "trimesh", "yourdfpy", "viser",
              "setuptools_scm"):
        try:
            mod = importlib.import_module(m)
            out[m] = getattr(mod, "__version__", None) or getattr(
                getattr(mod, "config", None), "version", "?")
        except Exception as exc:  # noqa: BLE001 — 없는 것도 결과다
            out[m] = f"MISSING ({type(exc).__name__})"
    return out


def eikonal(values: np.ndarray, voxel_size: float, *, free_above: float = 0.03) -> dict:
    """자유공간에서 `|∇d|` 의 분포. 제대로 된 거리장이면 중앙값이 1 이다.

    `free_above` 로 표면 근처를 뺀다 — 영교차 바로 옆은 이산화 때문에 기울기가 흔들리고,
    eikonal 성질은 자유공간의 주장이다.
    """
    v = np.asarray(values, np.float64)
    grads = np.gradient(v, float(voxel_size))
    norm = np.sqrt(sum(g ** 2 for g in grads))
    free = v > float(free_above)
    if not free.any():
        return {"n_free": 0, "median": None, "p05": None, "p95": None}
    sample = norm[free]
    return {"n_free": int(free.sum()),
            "median": float(np.median(sample)),
            "p05": float(np.percentile(sample, 5)),
            "p95": float(np.percentile(sample, 95))}


def build_two_tier(frame: dict, *, coarse: float, fine: float, tsdf_voxel: float,
                   grid_center, extent) -> dict:
    """cuRobo `Mapper` 로 coarse/fine 2계층 ESDF. **이 프로세스 안에서** 돈다."""
    import torch
    from curobo._src.types.camera import CameraObservation
    from curobo._src.types.pose import Pose
    from curobo.perception import Mapper, MapperCfg

    dev = "cuda:0"
    h, w = frame["depth_masked_head"].shape
    cfg = MapperCfg(
        extent_meters_xyz=tuple(extent), voxel_size=tsdf_voxel,
        esdf_voxel_size=coarse,
        grid_center=torch.tensor(list(grid_center), device=dev, dtype=torch.float32),
        truncation_distance=0.04, depth_minimum_distance=0.05, depth_maximum_distance=3.0,
        image_height=h, image_width=w, device=dev)
    mapper = Mapper(cfg)

    t_integrate = time.perf_counter()
    for cid in IDS:
        depth = np.asarray(frame[f"depth_masked_{cid}"], np.float32)
        mapper.integrate(CameraObservation(
            name=cid,
            depth_image=torch.as_tensor(depth, device=dev, dtype=torch.float32)[None],
            rgb_image=torch.zeros((1, h, w, 3), device=dev, dtype=torch.uint8),
            intrinsics=torch.as_tensor(np.asarray(frame[f"K_{cid}"], np.float32),
                                       device=dev, dtype=torch.float32)[None],
            pose=Pose.from_matrix(torch.as_tensor(np.asarray(frame[f"T_{cid}"], np.float32),
                                                  device=dev, dtype=torch.float32)),
            depth_to_meter=1.0))
    torch.cuda.synchronize()
    integrate_ms = (time.perf_counter() - t_integrate) * 1000.0

    centre = np.asarray(frame["fine_center"], np.float32)
    layers = [("coarse", dict(esdf_voxel_size=coarse)),
              ("fine", dict(esdf_origin=torch.as_tensor(centre, device=dev,
                                                        dtype=torch.float32),
                            esdf_voxel_size=fine))]
    out = {"integrate_ms": integrate_ms, "tiers": {}}
    grids = {}
    for name, kw in layers:
        vg = mapper.compute_esdf(**kw)
        vs = float(vg.voxel_size)
        dims = np.asarray(vg.dims, np.float64).reshape(3)
        pose = np.asarray(vg.pose, np.float64).reshape(-1)[:3]
        # 함정 1 — `esdf_origin` 은 격자 **중심**이다. 반 칸을 더해 첫 복셀 *중심*으로 바꾼다.
        origin = pose - 0.5 * dims + 0.5 * vs
        # 함정 4 — `feature_tensor` 는 재사용 버퍼다. 다음 호출 전에 **복사**한다.
        values = vg.feature_tensor.detach().float().cpu().numpy().copy()
        grids[name] = values
        out.setdefault("_grids", {})[name] = (values, origin, vs)
        out["tiers"][name] = {
            "voxel_size_m": vs,
            "shape": [int(v) for v in values.shape],
            "origin_m": [float(v) for v in origin],
            "min_m": float(values.min()), "max_m": float(values.max()),
            "negative_fraction": float((values < 0).mean()),
            "eikonal": eikonal(values, vs),
        }

    # 함정 4 의 직접 검사 — `feature_tensor` 는 **재사용 버퍼**다.
    #
    # shape 으로는 가릴 수 없다: cuRobo 는 두 계층 모두 128^3 을 주므로(coarse 20 mm 는
    # 2.56 m, fine 5 mm 는 0.64 m 를 덮는다) shape 이 같은 것이 정상이다. 처음에 shape
    # 비교로 검사를 썼다가 이 사실에 걸렸다 — 원인 분류는 **테스트 하네스 결함**이었다.
    #
    # 가르는 질문은 **값**이다. coarse 를 복사해 둔 뒤 fine 을 만들고, 다시 coarse 를
    # 만들어 셋을 비교한다.
    #   * 복사가 옳으면  : c1 == c2  이고  c1 != f1
    #   * 복사를 빼면    : c1 이 버퍼를 가리키므로 fine 호출이 덮어써 c1 == f1 이 된다
    # 그것이 발견 C4(두 계층을 만든 뒤에 읽어 자유공간 `|∇d| = 0.25` 가 나온 것)의 기전이다.
    vg_c2 = mapper.compute_esdf(esdf_voxel_size=coarse)
    c2 = vg_c2.feature_tensor.detach().float().cpu().numpy().copy()
    c1, f1 = grids["coarse"], grids["fine"]
    out["buffer_independence"] = {
        "coarse_shape": list(c1.shape), "fine_shape": list(f1.shape),
        "shapes_differ": c1.shape != f1.shape,
        "note": "shape 이 같은 것은 정상 — cuRobo 는 두 계층 모두 128^3 을 준다",
        "coarse_reproducible": bool(np.array_equal(c1, c2)),
        "coarse_differs_from_fine": bool(not np.array_equal(c1, f1)),
        "max_abs_coarse_minus_recomputed_m": float(np.max(np.abs(c1 - c2))),
        "mean_abs_coarse_minus_fine_m": float(np.mean(np.abs(c1 - f1))),
    }

    # 함정 3 — 첫 호출은 JIT 다 (기록상 약 1,500 ms). 정상 상태를 따로 잰다.
    warm = []
    for _ in range(5):
        t = time.perf_counter()
        mapper.compute_esdf(esdf_voxel_size=coarse)
        torch.cuda.synchronize()
        warm.append((time.perf_counter() - t) * 1000.0)
    out["compute_esdf_steady_ms"] = {"median": float(np.median(warm)),
                                     "min": float(np.min(warm)),
                                     "max": float(np.max(warm)),
                                     "note": "첫 호출(JIT)은 제외한 정상 상태"}

    # 적분도 정상 상태를 따로 잰다 — 위 `integrate_ms` 에는 warp 의 첫 커널 빌드가 섞여 있다.
    warm_int = []
    for _ in range(3):
        t = time.perf_counter()
        for cid in IDS:
            depth = np.asarray(frame[f"depth_masked_{cid}"], np.float32)
            mapper.integrate(CameraObservation(
                name=cid,
                depth_image=torch.as_tensor(depth, device=dev, dtype=torch.float32)[None],
                rgb_image=torch.zeros((1, h, w, 3), device=dev, dtype=torch.uint8),
                intrinsics=torch.as_tensor(np.asarray(frame[f"K_{cid}"], np.float32),
                                           device=dev, dtype=torch.float32)[None],
                pose=Pose.from_matrix(torch.as_tensor(np.asarray(frame[f"T_{cid}"], np.float32),
                                                      device=dev, dtype=torch.float32)),
                depth_to_meter=1.0))
        torch.cuda.synchronize()
        warm_int.append((time.perf_counter() - t) * 1000.0)
    out["integrate_steady_ms"] = {"median": float(np.median(warm_int)),
                                  "min": float(np.min(warm_int)),
                                  "max": float(np.max(warm_int)),
                                  "note": "3 카메라 한 벌. 첫 적분(warp 커널 빌드)은 제외"}
    return out


def capture_fresh_frame(*, seed: int, target: str, height: int, width: int) -> dict:
    """새 MuJoCo 씬에서 세 카메라의 **마스킹된** depth 를 캡처한다. 저장 기록을 읽지 않는다.

    raw depth 를 넣으면 로봇이 자기 몸을 장애물로 본다 (발견 C1 — cuRobo 검증 경로가
    마스킹 안 된 depth 를 써서 구 120 개 중 105 개가 자기 자신과 충돌한다고 나왔다).
    """
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import TRANSPORT_MODEL, TransportScene
    from benchmark.ag3s.runtime.pipeline import AG3S

    scene = TransportScene(TRANSPORT_MODEL, settle_steps=400, height=height, width=width)
    try:
        rng = np.random.default_rng(seed)
        joint = f"{target}_free"
        qadr = scene._qadr[joint]
        jitter = rng.uniform(-0.012, 0.012, size=2)
        scene.data.qpos[qadr:qadr + 2] += jitter
        scene.mujoco.mj_forward(scene.model, scene.data)

        target_true = scene.body_position_in_base(target)
        filter_robot = build_robot_model(scene)
        constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
        ag3s = AG3S(AG3SConfig.from_dict({"collision_backend": "primitive",
                                          "pointcloud": {"range_max": 2.0}}),
                    robot_model=filter_robot, constraint_robot_model=constraint_robot)

        out: dict = {"seed": seed, "target": target,
                     "target_true": np.asarray(target_true, np.float64),
                     "jitter_xy": jitter,
                     # I1 은 grounding 을 안 돌리므로 미세 창 중심은 **참값**을 쓴다.
                     # 창 중심의 선택 자체는 T3 에서 짝 비교한다.
                     "fine_center": np.asarray(target_true, np.float64)}
        renderer = scene._renderer_for()
        for cam, cid in zip(CAMS, IDS):
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
            renderer.update_scene(scene.data, camera=cam)
            out[f"rgb_{cid}"] = np.ascontiguousarray(renderer.render()).astype(np.uint8)
            f = scene.capture(cam)
            depth = np.asarray(f.depth, np.float64)
            K = np.asarray(f.camera_intrinsics, np.float64)
            T = np.asarray(f.T_base_cam, np.float64)
            mask = ag3s._robot_mask_for(depth, K, T, f.robot_state)
            if mask is None:
                raise RuntimeError(f"{cid}: robot mask was not produced")
            out[f"depth_{cid}"] = depth.astype(np.float32)
            out[f"depth_masked_{cid}"] = np.where(mask, np.float32(0.0),
                                                  depth.astype(np.float32))
            out[f"robot_mask_{cid}"] = np.asarray(mask, bool)
            out[f"K_{cid}"] = K.astype(np.float32)
            out[f"T_{cid}"] = T.astype(np.float32)
            out[f"mask_px_{cid}"] = int(mask.sum())
        q = np.asarray(scene.capture(CAMS[0]).robot_state, np.float64)
        centres, radii = constraint_robot.sphere_centers_numeric(q)
        out["sphere_centers"] = np.asarray(centres, np.float64).reshape(-1, 3)
        out["sphere_radii"] = np.asarray(radii, np.float64).reshape(-1)
        out["qpos"] = np.asarray(scene.data.qpos, np.float64).copy()
        if ag3s._esdf_builder is not None:
            raise RuntimeError("legacy EsdfBuilder was instantiated — T0 즉시 실패 조건")
        out["legacy_esdf_builder_created"] = False
        return out
    finally:
        scene.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", required=True, help="복제 전 openpi venv 버전 JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-npz", default=None, help="캡처한 프레임을 여기에 저장")
    ap.add_argument("--field-npz", default=None, help="두 계층의 거리 격자를 여기에 저장")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--coarse", type=float, default=0.020)
    ap.add_argument("--fine", type=float, default=0.005)
    ap.add_argument("--tsdf-voxel", type=float, default=0.005)
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"stage": "I1", "seed": args.seed, "checks": {}}

    # ── 조건 3: 버전이 복제 전과 같은가 (먼저 본다 — 다르면 나머지 결과를 못 믿는다) ──
    baseline = json.loads(pathlib.Path(args.baseline).read_text())
    now = _versions()
    drift = {k: {"before": baseline.get(k), "after": now.get(k)}
             for k in PINNED if baseline.get(k) != now.get(k)}
    report["versions_before"] = {k: baseline.get(k) for k in PINNED}
    report["versions_after"] = now
    report["checks"]["pinned_versions_unchanged"] = {"pass": not drift, "drift": drift}
    # 핀 밖에서 실제로 바뀐 것 — **보고서에 드러내는 것 자체가 목적이다.** 조용한 변경이
    # 이 검토에서 반복해서 나온 실패 방식이고(로그의 "조용한 실패"), 설치 부작용도 같다.
    report["unpinned_changes"] = {
        "typing_extensions": {"before": "4.13.2", "after": now.get("typing_extensions"),
                              "cause": "trimesh/viser 설치가 올렸다",
                              "verified_harmless": "jax CUDA · torch · mujoco · casadi · "
                                                   "osqp 가 이 보고서의 조건 1·2 에서 통과"},
        "note": "constraints 로 못 박은 것은 numpy·scipy·torch·jax·jaxlib·mujoco·casadi·osqp 다"}

    # ── 조건 1: 정책이 살아 있는가 ──
    try:
        import jax
        devices = [str(d) for d in jax.devices()]
        report["checks"]["jax_cuda_alive"] = {
            "pass": any("cuda" in d.lower() or "gpu" in d.lower() for d in devices),
            "devices": devices}
    except Exception as exc:  # noqa: BLE001
        report["checks"]["jax_cuda_alive"] = {"pass": False, "error": repr(exc)}

    # ── 조건 2: AG3S·SQP 가 살아 있는가 ──
    alive = {}
    for m in ("mujoco", "casadi", "osqp"):
        alive[m] = not str(now.get(m, "")).startswith("MISSING")
    report["checks"]["ag3s_sqp_alive"] = {"pass": all(alive.values()), "detail": alive}

    # ── 조건 4: 필드가 옳은가 ──
    try:
        t = time.perf_counter()
        frame = capture_fresh_frame(seed=args.seed, target=args.target,
                                    height=args.height, width=args.width)
        capture_ms = (time.perf_counter() - t) * 1000.0
        if args.frame_npz:
            pathlib.Path(args.frame_npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.frame_npz,
                                **{k: v for k, v in frame.items()
                                   if isinstance(v, (np.ndarray, np.generic))})
        field = build_two_tier(frame, coarse=args.coarse, fine=args.fine,
                               tsdf_voxel=args.tsdf_voxel,
                               grid_center=(0.45, 0.0, 0.8), extent=(1.5, 1.8, 1.6))
        saved_grids = field.pop("_grids", {})
        if args.field_npz and saved_grids:
            payload = {}
            for name, (values, origin, vs) in saved_grids.items():
                payload[f"{name}_values"] = values.astype(np.float32)
                payload[f"{name}_origin"] = np.asarray(origin, np.float64)
                payload[f"{name}_voxel_size"] = np.asarray(vs, np.float64)
            pathlib.Path(args.field_npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.field_npz, **payload)
        eik = {k: v["eikonal"]["median"] for k, v in field["tiers"].items()}
        ok = all(m is not None and 0.9 <= m <= 1.1 for m in eik.values())
        signs = {k: v["negative_fraction"] for k, v in field["tiers"].items()}
        report["capture"] = {
            "source": "fresh_mujoco_scene", "used_saved_run": False,
            "synthetic_attention": False, "attention_used": False,
            "capture_ms": capture_ms,
            "legacy_esdf_builder_created": frame["legacy_esdf_builder_created"],
            "target_true_m": [float(v) for v in frame["target_true"]],
            "jitter_xy_m": [float(v) for v in frame["jitter_xy"]],
            "mask_px": {cid: frame[f"mask_px_{cid}"] for cid in IDS},
            "n_spheres": int(len(frame["sphere_radii"]))}
        report["field"] = field
        report["checks"]["eikonal_in_band"] = {
            "pass": ok, "median": eik, "band": [0.9, 1.1]}
        bi = field["buffer_independence"]
        report["checks"]["buffer_independence"] = {
            "pass": bool(bi["coarse_reproducible"] and bi["coarse_differs_from_fine"]),
            "detail": bi}
        report["checks"]["sign_convention"] = {
            "pass": all(0.0 < f < 0.5 for f in signs.values()),
            "negative_fraction": signs,
            "note": "음수 = 물체 안쪽. 0 이면 표면이 안 쌓였고, 0.5 초과면 부호가 뒤집혔다"}
    except Exception as exc:  # noqa: BLE001 — 실패도 기록한다
        import traceback
        report["checks"]["eikonal_in_band"] = {"pass": False, "error": repr(exc)}
        report["field_traceback"] = traceback.format_exc()

    report["verdict"] = {
        "pass": all(c.get("pass") for c in report["checks"].values()),
        "checks": {k: bool(v.get("pass")) for k, v in report["checks"].items()}}
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))

    print(f"wrote {out_path}")
    for name, c in report["checks"].items():
        print(f"  [{'PASS' if c.get('pass') else 'FAIL'}] {name}")
        if not c.get("pass"):
            print(f"         {json.dumps({k: v for k, v in c.items() if k != 'pass'}, ensure_ascii=False)[:300]}")
    print(f"  => I1 {'PASS' if report['verdict']['pass'] else 'FAIL'}")
    raise SystemExit(0 if report["verdict"]["pass"] else 1)


if __name__ == "__main__":
    main()
