"""Small public Mapper experiment; no checkpoints, training or robot execution."""
import json,time
from pathlib import Path
import numpy as np
import torch
from curobo.perception import Mapper,MapperCfg
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
OUT=Path('benchmark/ag3s/docs/figures/reuse-audit-20260915')
device='cuda:0'
cfg=MapperCfg(extent_meters_xyz=(2.,2.,2.),voxel_size=.02,esdf_voxel_size=.02,grid_center=torch.tensor([0.,0.,1.],device=device),truncation_distance=.08,depth_minimum_distance=.05,depth_maximum_distance=3.,image_height=16,image_width=16,feature_dim=1,feature_grid_height=4,feature_grid_width=4,feature_block_grid_size=4,device=device)
mapper=Mapper(cfg)
obs=CameraObservation(name='synthetic',depth_image=torch.ones((1,16,16),device=device),rgb_image=torch.zeros((1,16,16,3),device=device,dtype=torch.uint8),intrinsics=torch.tensor([[[20.,0.,7.5],[0.,20.,7.5],[0.,0.,1.]]],device=device),pose=Pose.from_matrix(torch.eye(4,device=device)),depth_to_meter=1.,feature_grid=torch.ones((1,4,4,1),device=device,dtype=torch.float16))
mapper.integrate(obs);vox=mapper.extract_occupied_voxels(surface_only=True);a=vox.features().clone();centers=vox.centers.clone()
obs.feature_grid.zero_();mapper.integrate(obs);vox2=mapper.extract_occupied_voxels(surface_only=True);b=vox2.features().clone()
vg=mapper.compute_esdf();values=vg.feature_tensor.detach().clone();torch.cuda.synchronize()
coords=vg.create_xyzr_tensor(transform_to_origin=True)[...,:3].reshape(-1,3)
q=torch.tensor([[0,0,.8],[0,0,1.0],[.8,.8,.2]],device=device,dtype=torch.float32)
ix=torch.cdist(q,coords).argmin(1)
np.savez(OUT/'gpu_surface.npz',centers=centers.cpu().numpy(),features_first=a.cpu().numpy(),features_second=b.cpu().numpy())
r={'gpu':torch.cuda.get_device_name(),'surface_count':len(vox),'feature_shape':list(a.shape),'first_feature_minmax':[float(a.min()),float(a.max())],'second_feature_minmax':[float(b.min()),float(b.max())],'second_feature_median':float(b.median()),'current_attention':0.,'esdf_queries':q.cpu().tolist(),'esdf_nearest_values':values.reshape(-1)[ix].cpu().tolist(),'all_esdf_finite':bool(torch.isfinite(values).all()),'tsdf_allocated_blocks':int(mapper.tsdf.data.num_allocated.item()),'note':'One synthetic plane; supports API feasibility, not segmentation accuracy. No timing claim (JIT included).'}
(OUT/'gpu_features.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
