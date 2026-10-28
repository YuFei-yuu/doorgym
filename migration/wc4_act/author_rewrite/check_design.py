"""CPU design checks: exact author DLS function, analytic URDF Jacobian.
Ideal joint tracking only. Never creates an Isaac Gym simulation.
"""
import argparse, ast, hashlib, json, math, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'migration/wc4_act'))
from fullcycle_geometry import URDF, transform, rot, apply
from author_rewrite.geometry import DoorFrame, closing_reference, smooth


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


class AuthorDLS:
    def __init__(self, robot, effective):
        import torch
        self.torch, self.robot = torch, robot
        self.names = ['joint%d'%i for i in range(1, 7)]
        limits = [robot.root.find("joint[@name='%s']/limit"%n) for n in self.names]
        scale=float(effective.get('range_scale',.75))
        self.lo = scale*np.array([float(e.get('lower')) for e in limits])
        self.hi = scale*np.array([float(e.get('upper')) for e in limits])
        child = {j[3]:j for j in robot.joints}
        current = 'ee_gripper_link'; self.chain = []
        while current in child:
            j=child[current]; self.chain.insert(0,j); current=j[2]
        source=ROOT/'high-level/float_ik/door_common.py'
        tree=ast.parse(source.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='update_arm_ik_targets_for_env')
        def orientation_error(torch, desired, current):
            # Same XYZW quaternion product and sign(w) as the source helper.
            a,b=desired,current.clone();b[:3]=-b[:3]
            v=a[3]*b[:3]+b[3]*a[:3]+torch.cross(a[:3],b[:3],dim=0)
            w=a[3]*b[3]-torch.dot(a[:3],b[:3])
            return v*torch.sign(w)
        ns={'np':np,'gymapi':SimpleNamespace(STATE_ALL=0),
            'base_ik':SimpleNamespace(torch_orientation_error=orientation_error)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        self.update=ns[node.name];self.args=SimpleNamespace(**effective)
        self.source_sha=hashlib.sha256(source.read_bytes()).hexdigest()

    def fk_jacobian(self, B, q, frame):
        T=frame.base_transform(B);axes=[];origins=[];values=dict(zip(self.names,q))
        for name,kind,parent,child,O,axis in self.chain:
            T=T@O
            if name in values:
                axes.append(T[:3,:3]@axis);origins.append(T[:3,3].copy())
                M=np.eye(4);M[:3,:3]=rot(axis,values[name]);T=T@M
        A,O=np.array(axes),np.array(origins)
        J=np.vstack((np.cross(A,T[:3,3]-O).T,A.T))
        return T,J

    def step(self, B, q, target, R, frame):
        torch=self.torch;F,J=self.fk_jacobian(B,q,frame)
        tensor=lambda x:torch.tensor(np.asarray(x),dtype=torch.float32)
        st=SimpleNamespace(torch=torch,rb_states=tensor([np.r_[F[:3,3],Rotation.from_matrix(F[:3,:3]).as_quat(),np.zeros(6)]]),
            eef_body_sim_index=0,eef_jacobian_index=0,jacobian=tensor(J[None,None]),
            control_indices=list(range(6)),target_pos=tensor(target),target_quat=None if R is None else tensor(Rotation.from_matrix(R).as_quat()),
            lower=tensor(self.lo),upper=tensor(self.hi))
        gym=SimpleNamespace(get_actor_dof_states=lambda *args:{'pos':q})
        out=np.array(q,dtype=np.float32)
        self.update(gym,None,None,0,out,st,self.args,6)
        return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--effective',type=Path,required=True)
    p.add_argument('--sample-manifest',type=Path);p.add_argument('--sample-index',type=int,default=0);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only: hide CUDA')
    a.out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(__file__).with_name('config.json').read_text());effective=json.loads(a.effective.read_text())
    sampling=None
    if a.sample_manifest:
        trial=json.loads(a.sample_manifest.read_text())['trials'][a.sample_index];sampling=trial['effective_sampling']
        frame=DoorFrame(transform([sampling['door_x']+.11,sampling['door_y'],.01],[0,0,math.pi/2]),
            base_z=sampling['robot_z'],base_pitch=sampling['robot_pitch'])
    else:frame=DoorFrame(transform([2.61,0,.01],[0,0,math.pi/2]))
    robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf',include_visual=True)
    ik=AuthorDLS(robot,effective);q=np.zeros(6);rows=[];dt=.02
    B0=frame.base(cfg['closing_start_hinge_xy'],math.radians(cfg['closing_start_yaw_deg']))
    F,_=ik.fk_jacobian(B0,q,frame);start=F[:3,3].copy();R0=F[:3,:3].copy()
    from scipy.spatial.transform import Slerp
    hover,R,_,_,_=frame.contact(cfg,math.pi/2,cfg['hover_m'])
    rotation=Slerp([0,1],Rotation.from_matrix([R0,R]))
    def sample(phase,t,B,p,R,theta,joints=None):
        nonlocal q
        prev=q.copy();q=ik.step(B,q,p,R,frame) if joints is None else joints.copy();F,_=ik.fk_jacobian(B,q,frame)
        row=dict(phase=phase,t=t,base=B.tolist(),q=q.tolist(),theta=theta,target=p.tolist(),R=R.tolist(),ee=F[:3,3].tolist(),
            position_error_m=float(np.linalg.norm(F[:3,3]-p)),orientation_error_rad=float(Rotation.from_matrix(R@F[:3,:3].T).magnitude()),
            joint_margin_rad=float(np.min(np.r_[q-ik.lo,ik.hi-q])),joint_rate=float(max(abs(q-prev))/dt))
        rows.append(row)
    for i in range(1,round(cfg['deploy_s']/dt)+1):
        t=i*dt;half=cfg['deploy_s']/2;way=cfg['deploy_waypoints_q']
        qa=np.zeros(6) if t<half else np.asarray(way[0]);qb=np.asarray(way[0] if t<half else way[1])
        s=smooth((t if t<half else t-half)/half);Q=(1-s)*qa+s*qb
        F,_=ik.fk_jacobian(B0,Q,frame);sample('deploy',t,B0,F[:3,3],F[:3,:3],math.pi/2,Q)
    F,_=ik.fk_jacobian(B0,q,frame);hover=F[:3,3].copy()
    contact,R,_,_,_=frame.contact(cfg,math.pi/2)
    for i in range(1,round(cfg['approach_s']/dt)+1):
        s=smooth(i*dt/cfg['approach_s']);sample('approach',i*dt,B0,(1-s)*hover+s*contact,R,math.pi/2)
    for i in range(1,round(cfg['close_s']/dt)+1):
        B,p,R,theta,_=closing_reference(cfg,frame,i*dt);sample('close',i*dt,B,p,R,theta)
    for i in range(1,101):sample('settle',i*dt,B,p,R,theta)
    F,_=ik.fk_jacobian(B,q,frame);release0=F[:3,3].copy();releaseR=F[:3,:3].copy()
    for i in range(1,round(cfg['release_s']/dt)+1):
        p=release0-smooth(i*dt/cfg['release_s'])*cfg['withdraw_m']*frame.normal
        sample('release',i*dt,B,p,releaseR,0.)
    q0=q.copy()
    for i in range(1,round(cfg['fold_s']/dt)+1):
        Q=(1-smooth(i*dt/cfg['fold_s']))*q0;F,_=ik.fk_jacobian(B,Q,frame)
        sample('fold',i*dt,B,F[:3,3],F[:3,:3],0.,Q)
    summary={name:{'samples':len(rr),'max_position_error_m':max(r['position_error_m'] for r in rr),
        'max_orientation_error_rad':max(r['orientation_error_rad'] for r in rr),
        'minimum_joint_margin_rad':min(r['joint_margin_rad'] for r in rr),'max_joint_rate_rad_s':max(r['joint_rate'] for r in rr),
        'end':rr[-1]} for name in ['deploy','approach','close','settle','release','fold'] if (rr:=[r for r in rows if r['phase']==name])}
    write(a.out/'summary.json',summary);write(a.out/'asset.json',{'hinge':frame.hinge.tolist(),'slab_lo':frame.slab_lo.tolist(),'slab_hi':frame.slab_hi.tolist(),'author_dls_sha':ik.source_sha,'assumption':'ideal joints, analytic source URDF Jacobian; no physics'})
    write(a.out/'binding.json',{'sampling':sampling,'sample_manifest':None if a.sample_manifest is None else str(a.sample_manifest),'configuration':cfg,
        'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*') if p.is_file()}})
    with (a.out/'trace.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    print(json.dumps({k:{n:v for n,v in d.items() if n!='end'} for k,d in summary.items()},indent=2))
if __name__=='__main__':main()
