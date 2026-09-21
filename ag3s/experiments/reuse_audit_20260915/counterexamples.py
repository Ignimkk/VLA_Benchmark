"""Read-only calls to production queries with analytic/synthetic counterexamples."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from benchmark.ag3s.fields.esdf import EsdfField,VoxelGrid,esdf_from_occupancy,UNKNOWN
from benchmark.ag3s.fields.curobo_field import CuroboEsdfField
from benchmark.trajopt.linearize import CollisionLinearizer
OUT=Path('benchmark/ag3s/docs/figures/reuse-audit-20260915')
lin=CollisionLinearizer.__new__(CollisionLinearizer);lin.n_spheres=1;lin.query_radii=np.zeros(2)
class TwoSurfaces:
 grid=NS(voxel_size=.001);has_labels=True
 def distance(self,p):return np.minimum(abs(p[:,0]+.030),abs(p[:,0]-.045))
 def is_label(self,p,name):return abs(p[:,0]+.030)<abs(p[:,0]-.045)
def clearance(field,p):
 scene=NS(esdf=field,esdf_margin=.05,destination_label='destination',destination_margin=.02,manipulated_link_margin=None)
 centers=np.stack([np.array([1.,1.,1.]),np.asarray(p)])[None,:,:]
 return float(lin._esdf_clearance(centers,scene)[0,1,0])
r={};r['unequal_margins']={'reported_m':clearance(TwoSurfaces(),[0,0,0]),'required_min_m':min(.030-.020,.045-.050),'nearest_destination_m':.030,'other_obstacle_m':.045}
assert r['unequal_margins']['reported_m']>0 and r['unequal_margins']['required_min_m']<0
# Distance interpolation and label classification use different spatial rules.
g=VoxelGrid(origin=np.array([0.,0.,0.]),shape=(2,2,2),voxel_size=.02)
d=np.stack([np.full((2,2),.03),np.full((2,2),.04)])
labels=np.stack([np.zeros((2,2),np.int32),np.ones((2,2),np.int32)])
f=EsdfField(grid=g,distance_grid=d,max_distance=.4,label_grid=labels,label_names=('destination','obstacle'))
p=np.array([[.0099,.01,.01],[.0101,.01,.01]])
r['label_boundary']={'points_m':p.tolist(),'distance_m':f.distance(p).tolist(),'label':f.label(p).tolist(),'clearance_m':[clearance(f,x) for x in p]}
r['missing_label']={'unknown_name_matches':bool(f.is_label(p,'missing').any()),'outside_label':int(f.label([[1,1,1]])[0])}
# Adapter clamps outside all layers and drops labels.
coarse=EsdfField(grid=VoxelGrid(np.array([-.1,-.1,-.1]),(11,11,11),.02),distance_grid=np.full((11,11,11),.04),max_distance=.4)
fine=EsdfField(grid=VoxelGrid(np.array([-.02,-.02,-.02]),(5,5,5),.01),distance_grid=np.full((5,5,5),.01),max_distance=.4)
composite=CuroboEsdfField((coarse,fine))
r['roi_boundary']={'query_m':[.0199,.0201], 'distance_m':composite.distance([[.0199,0,0],[.0201,0,0]]).tolist(),'outside_query_m':[2,0,0],'outside_distance_m':float(composite.distance([[2,0,0]])[0]),'outside_gradient':composite.gradient([[2,0,0]])[0].tolist(),'has_label_contract':hasattr(composite,'has_labels'),'unknown_fraction_without_metadata':composite.unknown_fraction}
# Actual legacy unknown-policy behavior.
occ=np.full((4,4,4),UNKNOWN,np.int8)
res=esdf_from_occupancy(occ,VoxelGrid(np.zeros(3),(4,4,4),.02),max_distance=.4,unknown_policy='free')
r['unknown_free']={'min_m':float(res[0].min()),'max_m':float(res[0].max())}
# Observed sampled surface: thin obstacle between samples, then between time steps.
pts=np.array([[-.01,0,0],[.01,0,0]])
rad=.003
r['sample_gap']={'sample_clearance_m':(np.linalg.norm(pts,axis=1)-rad).tolist(),'surface_midpoint_clearance_m':-rad,'sample_spacing_m':.02}
r['time_gap']={'endpoint_clearance_m':[.03-rad,.03-rad],'intermediate_clearance_m':-rad}
(OUT/'counterexamples.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
# Label-only changes are not included in the production incremental dirty set.
from benchmark.ag3s.fields.esdf import EsdfBuilder,FREE,OCCUPIED
from benchmark.ag3s.config import EsdfConfig
cfg=EsdfConfig(voxel_size=.02,bounds_lower=(0,0,0),bounds_upper=(.3,.06,.06),incremental=True,max_distance=.4)
def toy_builder():
 b=EsdfBuilder(cfg); oc=np.full(b.grid.shape,FREE,np.int8);oc[3,1,1]=OCCUPIED;oc[11,1,1]=OCCUPIED
 b.volume=NS(occupancy=lambda **kw:oc.copy())
 return b
b=toy_builder();pa=b.grid.origin+np.array([3,1,1])*.02;pb=b.grid.origin+np.array([11,1,1])*.02
f0=b.update([],labelled_points={'destination':pa[None,:],'obstacle':pb[None,:]});before=f0.label([pa,pb]).copy()
f1=b.update([],labelled_points={'destination':pb[None,:],'obstacle':pa[None,:]})
fresh=toy_builder().update([],labelled_points={'destination':pb[None,:],'obstacle':pa[None,:]})
r['label_only_update']={'before':before.tolist(),'incremental_after':f1.label([pa,pb]).tolist(),'full_rebuild_after':fresh.label([pa,pb]).tolist(),'recomputed_voxels':f1.stats['n_recomputed']}
assert r['label_only_update']['incremental_after']!=r['label_only_update']['full_rebuild_after']
b=toy_builder();oc=np.full(b.grid.shape,FREE,np.int8);oc[3,1,1]=OCCUPIED;oc[4,1,1]=OCCUPIED;b.volume=NS(occupancy=lambda **kw:oc.copy())
f=b.update([],labelled_points={'destination':pa[None,:]})
other=b.grid.origin+np.array([4,1,1])*.02
r['label_snap_leak']={'seed_m':pa.tolist(),'other_obstacle_m':other.tolist(),'other_assigned_destination':bool(f.is_label([other],'destination')[0]),'snap_distance_m':.04,'fixture':'Two differently owned occupied voxels, only destination supplies seeds.'}
(OUT/'counterexamples.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:r[k] for k in ['label_only_update','label_snap_leak']},indent=2))
