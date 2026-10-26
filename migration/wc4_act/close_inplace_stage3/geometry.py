"""Source geometry including runtime far-side convention and articulated self pairs."""
import sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from close_inplace_stage2.candidate import Solver,baseT,LEGS,NAMES,HOME,read,write,sha
from close_inplace_stage2.audit import fast_gap
from fullcycle_geometry import URDF,transform,apply,rot
from fullcycle_collision import collision_pieces
from newexpert_stage2.finite_walls import walls

class Geometry:
    def __init__(self):
        self.robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf',include_visual=True)
        self.door=URDF(ROOT/'high-level/data/asset/door_set/wc4/model.urdf')
        self.R=collision_pieces(self.robot.path);self.D=collision_pieces(self.door.path)
        effective=read(ROOT/'.repro/runs/wc4-newexpert-stage4-20261025/attempt_000/effective-config.json')
        self.W,self.wallmeta=walls(effective,read(self.door.path.parent/'bounding_box.json'),0.)
        parent={n:n for n in self.robot.links}
        def root(n):
            while parent[n]!=n:n=parent[n]
            return n
        for j in self.robot.joints:
            if j[1] in ('fixed','floating'):parent[root(j[3])]=root(j[2])
        self.components={n:root(n) for n in parent}
        adjacent={frozenset((root(j[2]),root(j[3]))) for j in self.robot.joints if j[1] in ('revolute','continuous')}
        self.selfpairs=[];self.excluded=[]
        for i,a in enumerate(self.R):
            for j,b in enumerate(self.R[:i]):
                arm=lambda n:n.startswith(('link','wrist','gripper'))
                if not (arm(a.link) or arm(b.link)):continue
                pair=frozenset((root(a.link),root(b.link)))
                if root(a.link)==root(b.link) or pair in adjacent:
                    self.excluded.append((a.link,a.name,b.link,b.name,'rigid assembly or directly joint-adjacent'))
                else:self.selfpairs.append((i,j))
    def check(self,B,q,theta,contact_piece='2'):
        poses=self.robot.fk(dict(LEGS,**dict(zip(NAMES,q))),baseT(B))
        dp=self.door.fk({'door_hinge_joint':theta,'handle_spindle_joint':0},transform([2.61,0,.01],[0,0,math.pi/2]))
        AW=[p.world(poses[p.link]) for p in self.R];pieces=self.D+[p for p,_ in self.W]
        BW=[p.world(dp[p.link]) for p in self.D]+[b for _,b in self.W]
        amin=np.array([x[0].min(0) for x in AW]);amax=np.array([x[0].max(0) for x in AW]);bmin=np.array([x[0].min(0) for x in BW]);bmax=np.array([x[0].max(0) for x in BW])
        broad=np.maximum(bmin[None]-amax[:,None],amin[:,None]-bmax[None]).max(2)
        best={k:dict(gap_m=100.) for k in ['intended','external','self']};overlaps=[]
        for i,p in enumerate(self.R):
            for j,d in enumerate(pieces):
                cat='intended' if p.link=='gripperStator' and p.name==contact_piece and d.link=='door_panel' else 'external'
                gap=float(broad[i,j])
                if gap>.05 and gap>=best[cat]['gap_m']:continue
                if gap<=.05:gap=fast_gap(AW[i],BW[j])
                x=dict(gap_m=gap,a=[p.link,p.name,p.kind],b=[d.link,d.name,d.kind])
                if gap<best[cat]['gap_m']:best[cat]=x
                if gap< -1e-7:overlaps.append(dict(category=cat,**x))
        for i,j in self.selfpairs:
            p=self.R[i];d=self.R[j];gap=float(np.maximum(amin[i]-amax[j],amin[j]-amax[i]).max())
            if gap>.05 and gap>=best['self']['gap_m']:continue
            if gap<=.05:gap=fast_gap(AW[i],AW[j])
            x=dict(gap_m=gap,a=[p.link,p.name,p.kind],b=[d.link,d.name,d.kind])
            if gap<best['self']['gap_m']:best['self']=x
            if gap< -1e-7:overlaps.append(dict(category='self',**x))
        cloud=self.robot.cloud(poses)
        return dict(minimums=best,overlaps=overlaps,whole_max_x=float(cloud[:,0].max()),far=bool(cloud[:,0].max()<2.5),
            far_semantics='runtime source: collision AND visual bounds, configured nominal leg q, not measured dynamics')
