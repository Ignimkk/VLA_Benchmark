"""워밍업 후 정상 상태 측정 + 2계층 ESDF 의 값 검증."""
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
from curobo.perception import Mapper, MapperCfg

d = np.load("/tmp/rby1_frame.npz")
IDS = ("head", "left_wrist", "right_wrist")
dev = "cuda:0"
EXT = (1.5, 1.8, 1.6)
CENTER = torch.tensor([0.45, 0.0, 0.8], device=dev, dtype=torch.float32)

cfg = MapperCfg(extent_meters_xyz=EXT, voxel_size=0.005, esdf_voxel_size=0.020,
                grid_center=CENTER, truncation_distance=0.04,
                depth_minimum_distance=0.05, depth_maximum_distance=3.0,
                image_height=480, image_width=640, device=dev)
mapper = Mapper(cfg)

obs = []
for cid in IDS:
    T = np.asarray(d[f"T_{cid}"], np.float32)
    q = R.from_matrix(T[:3, :3]).as_quat()
    obs.append(CameraObservation(
        name=cid,
        depth_image=torch.as_tensor(d[f"depth_{cid}"], device=dev, dtype=torch.float32)[None],
        rgb_image=torch.zeros((1, 480, 640, 3), device=dev, dtype=torch.uint8),
        intrinsics=torch.as_tensor(d[f"K_{cid}"], device=dev, dtype=torch.float32)[None],
        pose=Pose(position=torch.as_tensor(T[:3, 3], device=dev, dtype=torch.float32)[None],
                  quaternion=torch.as_tensor(np.r_[q[3], q[:3]], device=dev,
                                             dtype=torch.float32)[None]),
        depth_to_meter=1.0))

tc = np.asarray(d["target_centroid"], np.float32)
# `esdf_origin` 은 코너가 아니라 격자 **중심**이다 (integrator_esdf.py:909 "Pose at center").
origin_fine = torch.as_tensor(tc, device=dev, dtype=torch.float32)


def bench(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts)), float(np.std(ts))


print("=" * 74)
print("정상 상태 (워밍업 3회 후 10회 평균)")
print("=" * 74)
m, s = bench(lambda: [mapper.integrate(o) for o in obs])
print(f"  TSDF 적분 (3 카메라)            {m:7.2f} ± {s:4.2f} ms")
m_c, s_c = bench(lambda: mapper.compute_esdf(esdf_voxel_size=0.020))
print(f"  ESDF 전체 20 mm (2.56 m 창)     {m_c:7.2f} ± {s_c:4.2f} ms")
m_f, s_f = bench(lambda: mapper.compute_esdf(esdf_origin=origin_fine, esdf_voxel_size=0.005))
print(f"  ESDF target 주변 5 mm (0.64 m)  {m_f:7.2f} ± {s_f:4.2f} ms")
print(f"  {'─'*52}")
print(f"  2계층 합계                      {m + m_c + m_f:7.2f} ms   / 예산 66.7 ms")
print(f"  (참고) 우리 numpy 구현 실측       ~2600 ms")

print("\n" + "=" * 74)
print("값 검증 — 두 계층이 같은 점에서 같은 거리를 답하는가")
print("=" * 74)
vg_c = mapper.compute_esdf(esdf_voxel_size=0.020)
fc = vg_c.feature_tensor.detach().float().cpu().numpy()
vg_f = mapper.compute_esdf(esdf_origin=origin_fine, esdf_voxel_size=0.005)
ff = vg_f.feature_tensor.detach().float().cpu().numpy()

print(f"  거친 격자 {fc.shape}  유한값 {np.isfinite(fc).mean():.1%}  "
      f"범위 [{np.nanmin(fc):.3f}, {np.nanmax(fc):.3f}] m")
print(f"  미세 격자 {ff.shape}  유한값 {np.isfinite(ff).mean():.1%}  "
      f"범위 [{np.nanmin(ff):.3f}, {np.nanmax(ff):.3f}] m")


def sample(feat, origin, vs, pts):
    idx = np.rint((pts - origin) / vs).astype(int)
    ok = np.all((idx >= 0) & (idx < 128), axis=1)
    out = np.full(len(pts), np.nan)
    i = idx[ok]
    out[ok] = feat[i[:, 0], i[:, 1], i[:, 2]]
    return out


# 중심 -> 코너로 바꿔 인덱싱
origin_c = CENTER.cpu().numpy() - 0.020 * 128 / 2
of = origin_fine.cpu().numpy() - 0.005 * 128 / 2
probes = np.array([tc, tc + [0.05, 0, 0], tc + [0, 0, 0.10], tc + [0.15, 0.15, 0]], np.float32)
dc = sample(fc, origin_c, 0.020, probes)
df = sample(ff, of, 0.005, probes)
print(f"\n  {'점 (target 기준)':<28}{'거친 20mm':>12}{'미세 5mm':>12}{'차이':>10}")
for p, a, b in zip(probes, dc, df):
    rel = np.round(p - tc, 2)
    diff = "" if (np.isnan(a) or np.isnan(b)) else f"{(b-a)*1000:+7.1f} mm"
    sa = "격자밖" if np.isnan(a) else f"{a:8.3f} m"
    sb = "격자밖" if np.isnan(b) else f"{b:8.3f} m"
    print(f"  {str(rel):<28}{sa:>12}{sb:>12}{diff:>10}")

print(f"\n  미세 계층이 덮는 범위: {of.round(2)} ~ {(of+0.64).round(2)}  (0.64 m 정육면체)")
print(f"  거친 계층이 덮는 범위: {origin_c.round(2)} ~ {(origin_c+2.56).round(2)}  (2.56 m)")
