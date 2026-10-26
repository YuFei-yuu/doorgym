import sys,math,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
sys.path.insert(0,str(ROOT/'migration/wc4_act/close_inplace_stage3'))
from close_inplace_stage3.geometry import Geometry,Solver,baseT,HOME,NAMES,apply,rot,read,write,sha
from close_inplace_stage2.candidate import smooth

class FullPose(Solver):
    def __init__(self,robot):super().__init__(robot,'corner')
    def solve(self,B,p,R,seed,nfev):
        Tinv=np.linalg.inv(baseT(B));goal=apply(Tinv,np.array([p]))[0];target=Tinv[:3,:3]@R
        seed=np.clip(np.array(seed)[:6],self.lo+1e-9,self.hi-1e-9)
        def calc(q,jac=False):
            F,P,U,JP,JU,A,O=self.state(q)
            parts=[4*JP] if jac else [4*(P-goal)]
            for j in range(3):parts.append(np.cross(A,F[:3,j]).T if jac else F[:3,j]-target[:,j])
            parts.append(.001*np.eye(6) if jac else .001*(q-seed))
            return np.vstack(parts) if jac else np.concatenate(parts)
        r=least_squares(calc,seed,jac=lambda q:calc(q,True),bounds=(self.lo,self.hi),max_nfev=nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        F,P,*_=self.state(r.x)
        return r.x,dict(nfev=int(r.nfev),status=int(r.status),position_error_m=float(np.linalg.norm(P-goal)),orientation_error_rad=float(Rotation.from_matrix(target@F[:3,:3].T).magnitude()),joint_slack=float(np.min(np.r_[r.x-self.lo,self.hi-r.x])))
def normal(theta):return np.array([math.cos(theta),math.sin(theta),0.])
def point(cfg,theta,offset=0):return np.array([2.467-cfg['radius']*math.sin(theta)-(.002+offset)*math.cos(theta),-.463+cfg['radius']*math.cos(theta)-(.002+offset)*math.sin(theta),cfg['height']])
def pose_rotation(theta,roll_deg):return rot(normal(theta),math.radians(roll_deg))
def reference(cfg):
    rows=[json.loads(l) for l in (ROOT/cfg['source_stage3']/'single/trace.jsonl').open()]
    start=[r for r in rows if r['phase']=='turn'][-1]
    return start,rows
def evaluate(G,S,B,q,theta,p=None,R=None,contact=False,cfg=None):
    F=baseT(B)@S.fk(q);pad=F[:3,3]+F[:3,:3]@S.pad;n=normal(theta)
    g=G.check(B,q,theta);position=None if p is None else float(np.linalg.norm(pad-p));orientation=None if R is None else float(Rotation.from_matrix(R@F[:3,:3].T).magnitude())
    axis=float(np.arccos(np.clip((F[:3,:3]@n)@n,-1,1)))
    eps=cfg['geometry_epsilon_m'];exterior=g['minimums']['external']['gap_m']>=-eps and (contact or g['minimums']['intended']['gap_m']>=-eps)
    contact_surface=(not contact) or g['minimums']['intended']['gap_m']>=-eps
    good=bool(exterior and contact_surface and g['minimums']['self']['gap_m']>=-eps and g['far'] and (position is None or position<=cfg['position_gate_m']) and (orientation is None or orientation<=cfg['full_orientation_gate_rad']) and (not contact or axis<=cfg['normal_gate_rad']))
    return dict(position_error_m=position,orientation_error_rad=orientation,normal_error_rad=axis,actual_rotation=F[:3,:3].tolist(),pad=pad.tolist(),geometry=g,strict_local_pass=good,contact_allowed=contact)
