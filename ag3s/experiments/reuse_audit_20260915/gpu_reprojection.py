"""Compare current-attention reprojection with stored spatial features on a synthetic plane."""
# Import also runs the original two-frame experiment to keep its settings identical.
from benchmark.ag3s.experiments.reuse_audit_20260915.gpu_features import *
mapper.reset()
obs.feature_grid.fill_(0);obs.feature_grid[:,:,:2,:]=1
mapper.integrate(obs)
v=mapper.extract_occupied_voxels(surface_only=True)
p=v.centers.detach().cpu().numpy();f=v.features().detach().cpu().numpy().reshape(-1)
# Camera at identity; image intrinsics and depth are precisely known in this fixture.
u=20*p[:,0]/p[:,2]+7.5;vv=20*p[:,1]/p[:,2]+7.5
valid=(u>=0)&(u<16)&(vv>=0)&(vv<16)&(np.abs(p[:,2]-1)<.03)
current=(u<7.5).astype(float)
interior=valid&(np.abs(p[:,0])>.15)
r={'surface_count':len(p),'visible_near_surface_count':int(valid.sum()),'interior_count':int(interior.sum()),'interior_binary_agreement':float(((f[interior]>.5)==(current[interior]>.5)).mean()),'feature_minmax':[float(f.min()),float(f.max())],'visibility_depth_tolerance_m':.03,'notes':'Synthetic planar depth; 4x4 features, left=1/right=0. Boundary excluded from agreement. Reprojection code is experiment-only; does not validate real instance segmentation.'}
np.savez(OUT/'gpu_reprojection.npz',centers=p,stored_feature=f,current_attention=current,visible=valid)
(OUT/'gpu_reprojection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
