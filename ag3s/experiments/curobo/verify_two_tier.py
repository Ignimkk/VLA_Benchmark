"""curobo 자신의 create_xyzr_tensor 로 좌표를 얻어 2계층 ESDF 를 검증한다."""
import numpy as np, torch
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
from curobo.perception import Mapper, MapperCfg

d = np.load("/tmp/rby1_frame.npz"); dev="cuda:0"
tc = np.asarray(d["target_centroid"], np.float32)
cfg = MapperCfg(extent_meters_xyz=(1.5,1.8,1.6), voxel_size=0.005, esdf_voxel_size=0.020,
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

def grid_points_and_vals(vg):
    """curobo 의 생성기로 (N,3) 월드좌표와 (N,) 거리."""
    xyzr = vg.create_xyzr_tensor(transform_to_origin=True)
    pts = xyzr[:, :3].detach().cpu().numpy()
    vals = vg.feature_tensor.detach().float().reshape(-1).cpu().numpy()
    return pts, vals

for label, kw, vs in (("거친 20 mm", dict(esdf_voxel_size=0.020), 0.020),
                      ("미세 5 mm (target 중심)",
                       dict(esdf_origin=torch.as_tensor(tc,device=dev,dtype=torch.float32),
                            esdf_voxel_size=0.005), 0.005)):
    vg = m.compute_esdf(**kw)
    pts, vals = grid_points_and_vals(vg)
    print(f"\n=== {label} ===")
    print(f"  pose(=중심?) {np.round(vg.pose[:3],3)}   dims {np.round(vg.dims,2)}")
    print(f"  좌표 범위  x[{pts[:,0].min():+.2f},{pts[:,0].max():+.2f}] "
          f"y[{pts[:,1].min():+.2f},{pts[:,1].max():+.2f}] z[{pts[:,2].min():+.2f},{pts[:,2].max():+.2f}]")
    # 테이블 상판 근처에서 거리가 0 에 가까운 복셀의 z 분포
    sel = (pts[:,0]>0.45)&(pts[:,0]<0.85)&(np.abs(pts[:,1])<0.4)&(np.abs(vals)<vs)
    if sel.sum():
        print(f"  |d|<{vs*1000:.0f}mm 인 복셀의 z 중앙값 : {np.median(pts[sel,2]):.3f} m "
              f"(n={sel.sum():,})   <- 테이블 0.823 과 비교")
    # target 중심에서의 값
    k = np.argmin(np.linalg.norm(pts - tc, axis=1))
    print(f"  target centroid 최근접 복셀 d = {vals[k]:+.3f} m  "
          f"(복셀 위치 {np.round(pts[k],3)})")
