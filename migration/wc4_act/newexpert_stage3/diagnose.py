"""Diagnose saved stage-two failure; no new rollout or parameter search."""
import json,sys,os,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from newexpert_stage2.push_geometry import PadArm,PAD_LOCAL
from fullcycle_geometry import URDF
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
p=ROOT/'.repro/reviews/wc4-newexpert-stage2-20261025-01/new-linear-nominal/trace.jsonl'
rows=[json.loads(x) for x in p.open()];dep=[r for r in rows if r['phase']=='compact_rendezvous'];arm=PadArm(URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf'))
def detail(r):
 q=np.array(r['q'])[:6];F=arm.fk(q)
 def fn(q):
  T=arm.fk(q);return np.r_[T[:3,3]+T[:3,:3]@PAD_LOCAL,T[:3,0]]
 J=np.column_stack([(fn(q+np.eye(6)[i]*1e-6)-fn(q-np.eye(6)[i]*1e-6))/2e-6 for i in range(6)])
 return dict(t=r['t'],q=q.tolist(),pad=r['pad'],ik=r['ik'],joint_slack=np.minimum(q-arm.lo,arm.hi-q).tolist(),jacobian_singular_values=np.linalg.svd(J,compute_uv=False).tolist())
chosen=[dep[0]]+[next(r for r in dep if r['ik'].get('position_error_m',0)>threshold) for threshold in [.001,.01,.1]]+[dep[-1]]
x=dict(source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),evidence='saved failed synthetic trace, not new physics',samples=[detail(r) for r in chosen],
 explanation='Folded home lies on joint2 lower and joint3 upper limits; the Cartesian/axis warm-start sequence leaves that boundary on a different IK branch and later saturates other limits. An independently seeded hover solution exists. This supports a path/initialization failure, not proof that the endpoint is unreachable; singular values are local diagnostics only.',
 fix='Use one bounded smooth joint interpolation to the independently solved nominal hover; retain measured pad/axis arrival gate. No per-step Cartesian IK during deployment.')
(out/'diagnosis.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
