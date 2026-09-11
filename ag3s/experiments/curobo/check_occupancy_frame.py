"""인덱싱 가정을 우회 — 점유 복셀을 직접 꺼내 실제 기하와 대조한다."""
import numpy as np, torch
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
from curobo.perception import Mapper, MapperCfg

d = np.load("/tmp/rby1_frame.npz"); dev="cuda:0"
cfg = MapperCfg(extent_meters_xyz=(1.5,1.8,1.6), voxel_size=0.005, esdf_voxel_size=0.005,
                grid_center=torch.tensor([0.45,0.0,0.8],device=dev,dtype=torch.float32),
                truncation_distance=0.04, depth_minimum_distance=0.05,
                depth_maximum_distance=3.0, image_height=480, image_width=640, device=dev)
m = Mapper(cfg)
for cid in ("head","left_wrist","right_wrist"):
    T = np.asarray(d[f"T_{cid}"], np.float32)
    m.integrate(CameraObservation(name=cid,
        depth_image=torch.as_tensor(d[f"depth_{cid}"],device=dev,dtype=torch.float32)[None],
        rgb_image=torch.zeros((1,480,640,3),device=dev,dtype=torch.uint8),
        intrinsics=torch.as_tensor(d[f"K_{cid}"],device=dev,dtype=torch.float32)[None],
        pose=Pose.from_matrix(torch.as_tensor(T, device=dev, dtype=torch.float32)),
        depth_to_meter=1.0))

ov = m.extract_occupied_voxels()
print("extract_occupied_voxels ->", type(ov).__name__, [a for a in dir(ov) if not a.startswith('_')][:10])
pos = None
for attr in ("positions","xyz","points","voxel_centers","centers"):
    if hasattr(ov, attr):
        pos = getattr(ov, attr); print("  using attr:", attr); break
if pos is None and torch.is_tensor(ov): pos = ov
p = np.asarray(pos.detach().cpu() if torch.is_tensor(pos) else pos).reshape(-1,3)
print(f"\n점유 복셀 {len(p):,} 개")
print(f"  x [{p[:,0].min():+.3f}, {p[:,0].max():+.3f}]")
print(f"  y [{p[:,1].min():+.3f}, {p[:,1].max():+.3f}]")
print(f"  z [{p[:,2].min():+.3f}, {p[:,2].max():+.3f}]")

# 테이블 상판 근처(x 0.4~0.9, y -0.5~0.5)의 z 히스토그램 -> 상판 높이가 봉우리로 나와야
sel = (p[:,0]>0.40)&(p[:,0]<0.90)&(np.abs(p[:,1])<0.50)
zz = p[sel,2]
print(f"\n테이블 영역 점유 복셀 {sel.sum():,} 개의 z 분포:")
h,e = np.histogram(zz, bins=np.arange(0.0,1.6,0.02))
for i in np.argsort(h)[::-1][:5]:
    print(f"   z {e[i]:.2f}~{e[i+1]:.2f} : {h[i]:6,} 개")
print(f"\n  우리 RANSAC 테이블 평면 : z = 0.823 m")
print(f"  우리 numpy ESDF 가 본 상판 인라이어 z 범위 : 0.81~0.83 m")
