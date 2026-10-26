"""Bounded nominal offline candidates. No simulator, torch, or policy imports.

Full attempted paths are retained after an unsolved IK waypoint, with rate-limited
synthetic q; later phases are diagnostics, not evidence the failed gate was met.
"""
import argparse, json, math, os, sys, time, hashlib, subprocess
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from fullcycle_geometry import URDF,transform,rot,apply
from fullcycle_compact import Arm
NAMES=['joint%d'%i for i in range(1,7)]+['jointGripper']
LEGS={side+s:v for side in ['FL','FR','RL','RR'] for s,v in [('_hip_joint',0),('_thigh_joint',.8),('_calf_joint',-1.5)]}
HOME=np.r_[np.zeros(6),-1.1775000095367432]
H=np.array([2.467,-.463,.01])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def smooth(a):a=np.clip(a,0,1);return a*a*a*(10+a*(-15+6*a))
def emit(x):print(json.dumps(x,allow_nan=False),flush=True)
def baseT(base):return transform([*base[:2],.5],[0,0,base[2]])
class Solver(Arm):
    def __init__(self,robot,face):
        super().__init__(robot)
        self.pad=np.array([.015,0,-.020]) if face=='x' else np.array([0,.030,-.020])
        self.axis=np.array([1.,0,0]) if face=='x' else np.array([0,1.,0])
    def state(self,q):
        T=np.eye(4);axes=[];origins=[];values=dict(zip(self.names,q))
        for name,kind,parent,child,O,axis in self.chain:
            T=T@O
            if name in values:
                origins.append(T[:3,3].copy());axes.append(T[:3,:3]@axis)
                M=np.eye(4);M[:3,:3]=rot(axis,values[name]);T=T@M
        p=T[:3,3]+T[:3,:3]@self.pad;u=T[:3,:3]@self.axis
        A=np.array(axes);O=np.array(origins)
        return T,p,u,np.cross(A,p-O).T,np.cross(A,u).T,A,O
    def solve_point(self,B,p,u,seed,nfev,full_rotation=None,ee=False):
        inv=np.linalg.inv(baseT(B));p=apply(inv,np.array([p]))[0]
        u=None if u is None else inv[:3,:3]@u
        full_rotation=None if full_rotation is None else inv[:3,:3]@full_rotation
        seed=np.clip(np.asarray(seed)[:6],self.lo+1e-9,self.hi-1e-9)
        def residual(q,jac=False):
            T,P,U,JP,JU,A,O=self.state(q)
            if ee:P=T[:3,3];JP=np.cross(A,P-O).T
            parts=[4*JP] if jac else [4*(P-p)]
            if full_rotation is not None:
                for j in range(3):parts.append(.25*np.cross(A,T[:3,j]).T if jac else .25*(T[:3,j]-full_rotation[:,j]))
            elif u is not None:parts.append(JU if jac else U-u)
            parts.append(.001*np.eye(6) if jac else .001*(q-seed))
            return np.vstack(parts) if jac else np.concatenate(parts)
        r=least_squares(residual,seed,jac=lambda q:residual(q,True),bounds=(self.lo,self.hi),max_nfev=nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        T,P,U,*_=self.state(r.x)
        return r.x,dict(nfev=r.nfev,solver_status=int(r.status),position_error_m=float(np.linalg.norm((T[:3,3] if ee else P)-p)),axis_error_rad=None if u is None else float(np.arccos(np.clip(U@u,-1,1))))

def generate(out,c,cfg):
    source=ROOT/cfg['source_run'];record=[json.loads(s) for s in (source/'states.jsonl').open()]
    b=next(r for r in record if r['passed']);start=next(r for r in record if r['phase']=='traverse_door')
    ref=read(source/'reference.json');effective=read(source/'effective-config.json')
    dt=cfg['dt'];robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');solver=Solver(robot,c['face'])
    endpoint=np.r_[ref['old_expert_final_base'],ref['old_expert_final_yaw']]
    base=np.array(b['actual_base']);base[2]=math.pi;q=np.array(b['arm_q']);theta=float(b['door_q'][0]);t=b['t'];rows=[];raw_solve=[]
    def save(phase,B,Q,angle,target=None,axis=None,ik=None,kind='synthetic',ee=False,rawq=None):
        nonlocal t,base,q,theta
        oldbase=base.copy();oldq=q.copy();t+=dt;base=np.array(B);q=np.array(Q);theta=float(angle)
        local=np.array([[math.cos(oldbase[2]),math.sin(oldbase[2])],[-math.sin(oldbase[2]),math.cos(oldbase[2])]])@(base[:2]-oldbase[:2])/dt
        F,P,U,*_=solver.state(q[:6]);T=baseT(base);pad=apply(T,np.array([P]))[0];u=T[:3,:3]@U;ee_world=apply(T,np.array([F[:3,3]]))[0]
        poses=robot.fk(dict(LEGS,**dict(zip(NAMES,q))),T)
        cloud=robot.cloud(poses)
        row=dict(t=t,phase=phase,kind=kind,base=base.tolist(),q=q.tolist(),theta=theta,z=.5,pitch=0.,
            action=np.r_[local[0],(base[2]-oldbase[2])/dt,q].tolist(),lateral_mps=float(local[1]),pad=pad.tolist(),ee=ee_world.tolist(),
            target=None if target is None else np.asarray(target).tolist(),target_axis=None if axis is None else np.asarray(axis).tolist(),
            tracking_m=None if target is None else float(np.linalg.norm((ee_world if ee else pad)-target)),
            tracking_axis_rad=None if axis is None else float(np.arccos(np.clip(u@axis,-1,1))),ik=ik,
            joint_rate_max=float(np.max(abs(q[:6]-oldq[:6]))/dt),joint_slack=float(np.min(np.r_[q[:6]-solver.lo,solver.hi-q[:6]])),
            whole_max_x=float(cloud[:,0].max()),raw_ik_joint_jump=None if rawq is None else float(np.max(abs(rawq-oldq[:6]))))
        rows.append(row)
    # Keep measured prefix as independent evidence; no synthetic endpoint claim.
    write(out/'measured-boundary.json',dict(first_pass=b,first_traverse=start,reference=ref,effective_source=str(source),record_sha256=sha(source/'states.jsonl')))
    fields=['step','t','phase','actual_base','arm_q','door_q','action','ee_actual','ee_target','ee_target_quat']
    with (out/'measured-prefix.jsonl').open('x') as f:
        for r in record[:b['step']+1]:f.write(json.dumps({k:r[k] for k in fields})+'\n')
    # Original linear traverse target held in world, from recorded first traverse.
    push_end=sum(effective[k] for k in ['walk_steps','initial_hold_steps','grasp_steps','grasp_hold_steps','gripper_close_steps','handle_rotate_steps','door_push_steps'])
    traverse_end=push_end+effective['traverse_steps'];pre=record[push_end-1]
    B0=np.array(pre['command_base']);B0[2]=math.pi
    target=np.array(start['ee_target']);R=Rotation.from_quat(start['ee_target_quat']).as_matrix()
    # One original-gain DLS update per 50Hz source step, analytic geometric Jacobian.
    # Perfect joint tracking after each update is a synthetic assumption.
    for step in range(b['step']+1,traverse_end):
        a=(step-push_end+1)/effective['traverse_steps'];B=(1-a)*B0+a*endpoint
        T,P,U,JP,JU,A,O=solver.state(q[:6]);world=baseT(B);p=apply(world,np.array([T[:3,3]]))[0]
        Jp=world[:3,:3]@np.cross(A,T[:3,3]-O).T;Jr=world[:3,:3]@A.T
        rv=Rotation.from_matrix(R@(world[:3,:3]@T[:3,:3]).T).as_rotvec();norm=np.linalg.norm(rv)
        orn=rv*(math.sin(norm/2)/norm if norm>1e-9 else .5)
        weight=effective['ik_rot_weight'];J=np.vstack([Jp,weight*Jr]);err=np.r_[effective['ik_pos_gain']*(target-p),weight*effective['ik_rot_gain']*orn]
        dq=J.T@np.linalg.solve(J@J.T+np.eye(6)*effective['ik_damping']**2,err)
        Q=q.copy();Q[:6]=np.clip(q[:6]+np.clip(dq,-effective['ik_max_step'],effective['ik_max_step']),solver.lo,solver.hi)
        Q[-1]=start['action'][-1]
        save('traverse_continuation',B,Q,theta,target=target,ee=True,kind='synthetic_original_targets_DLS')
    write(out/'handoff.json',dict(endpoint=endpoint.tolist(),last_source_step=traverse_end-1,synthetic_arrival=rows[-1],
        hypothesis='Door angle frozen at measured first-pass value; original world EE target held, original DLS gains with analytic source Jacobian, ideal joint tracking. Not simulator reproduction.',
        not_home_error=float(max(abs(q[:6]-HOME[:6]))),future_measured_checks=['arrival joint position/velocity and EE pose','door angle/rate after continuing traverse','source EE target tracking and contacts','arrival base braking and actual stopping position']))
    def segment(name,seconds,bfun=None,pfun=None,ufun=None,qfun=None,thetafun=None,ee=False):
        nonlocal q
        n=round(seconds/dt);Bstart=base.copy();Qstart=q.copy();thstart=theta;solution=q[:6].copy();diag=None
        for i in range(1,n+1):
            u=i/n;s=float(smooth(u));B=Bstart if bfun is None else bfun(s)
            th=thstart if thetafun is None else thetafun(s)
            p=None if pfun is None else pfun(s);axis=None if ufun is None else ufun(s)
            raw=None
            if qfun is not None:want=qfun(s)
            elif p is not None:
                if (i-1)%cfg['ik_period_steps']==0 or i==n:
                    solution,diag=solver.solve_point(B,p,axis,q,cfg['ik_max_nfev'],ee=ee);raw=solution
                    raw_solve.append(dict(t=t+dt,phase=name,base=np.asarray(B).tolist(),target=np.asarray(p).tolist(),q=solution.tolist(),**diag))
                want=np.r_[solution,HOME[-1]]
            else:want=Qstart
            Q=q+np.clip(want-q,-cfg['joint_speed']*dt,cfg['joint_speed']*dt)
            save(name,B,Q,th,p,axis,diag,ee=ee,rawq=raw)
        emit(dict(candidate=c['id'],phase=name,t=t,max_tracking_m=max([r['tracking_m'] for r in rows if r['phase']==name and r['tracking_m'] is not None] or [0])))
    d=cfg['durations'];segment('endpoint_hold',d['endpoint_hold'])
    # Move out of inherited contact toward the near side, then fold at fixed B.
    p0=rows[-1]['ee'];p0=np.array(p0)
    segment('withdraw',d['withdraw'],pfun=lambda s:p0+np.array([0,.14*s,0]),ee=True)
    Q0=q.copy();segment('fold',d['fold'],qfun=lambda s:(1-s)*Q0+s*HOME)
    yaw=math.radians(c['yaw_deg']);segment('turn',d['turn'],bfun=lambda s:np.r_[endpoint[:2],math.pi+s*(yaw-math.pi)])
    # One explicit endpoint IK seed, then a smooth joint arc. No multi-start search.
    pre=np.array(c['predeploy']);axis=np.array([0,1.,0])
    seed=np.array([0,1.7,-1.3,-.4,0,0]);hover,hovdiag=solver.solve_point(base,pre,axis,seed,cfg['ik_max_nfev'])
    write(out/'deployment-endpoint-solve.json',dict(seed=seed.tolist(),q=hover.tolist(),diagnostics=hovdiag))
    Q0=q.copy();Q1=np.r_[hover,HOME[-1]]
    segment('deploy_arc',d['deploy_arc'],qfun=lambda s:(1-s)*Q0+s*Q1)
    edge=np.array(c['edge']);P0=np.array(rows[-1]['pad'])
    segment('edge',d['edge'],pfun=lambda s:(1-s)*P0+s*edge,ufun=lambda s:axis)
    def point(angle,extra=0):return H+np.array([-c['radius']*math.sin(angle)-(.002+extra)*math.cos(angle),c['radius']*math.cos(angle)-(.002+extra)*math.sin(angle),c['height']-.01])
    P0=np.array(rows[-1]['pad']);P1=point(math.pi/2)
    segment('seek',d['seek'],pfun=lambda s:(1-s)*P0+s*P1,ufun=lambda s:axis)
    turnstart=math.radians(c['turn_start_deg']);follow=math.radians(c['follow_start_deg'])
    def closebase(s):
        a=(1-s)*math.pi/2
        y=yaw+(2*math.pi-yaw)*smooth((turnstart-a)/(turnstart-follow))
        x=endpoint[0]+c['follow_m']*smooth((follow-a)/follow)
        return np.array([x,endpoint[1],y])
    segment('close',d['close'],bfun=closebase,pfun=lambda s:point((1-s)*math.pi/2),ufun=lambda s:np.array([math.cos((1-s)*math.pi/2),math.sin((1-s)*math.pi/2),0]),thetafun=lambda s:(1-s)*math.pi/2)
    segment('settle',d['settle'])
    P0=np.array(rows[-1]['pad']);segment('release',d['release'],pfun=lambda s:P0-np.array([.12*s,0,0]))
    Q0=q.copy();segment('retract',d['retract'],qfun=lambda s:(1-s)*Q0+s*HOME)
    B0=base.copy();segment('restore',d['restore'],bfun=lambda s:np.r_[B0[:2],2*math.pi-s*math.pi])
    segment('hold',d['hold'])
    with (out/'trace.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    with (out/'ik-solves.jsonl').open('x') as f:
        for r in raw_solve:f.write(json.dumps(r,allow_nan=False)+'\n')
    a=np.array([r['action'] for r in rows]);poses=np.array([r['base'] for r in rows]);delta=np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1)
    suffix=[r for r in rows if r['phase']!='traverse_continuation'];inds=[i for i,r in enumerate(rows) if r['phase']!='traverse_continuation']
    byphase={}
    for phase in dict.fromkeys(r['phase'] for r in rows):
        rr=[r for r in rows if r['phase']==phase]
        byphase[phase]=dict(samples=len(rr),max_tracking_m=max([r['tracking_m'] for r in rr if r['tracking_m'] is not None] or [0]),
            max_axis_rad=max([r['tracking_axis_rad'] for r in rr if r['tracking_axis_rad'] is not None] or [0]),min_joint_slack=min(r['joint_slack'] for r in rr),max_joint_rate=max(r['joint_rate_max'] for r in rr),max_whole_x=max(r['whole_max_x'] for r in rr))
    summary=dict(candidate=c,stages=byphase,complete_attempted_path=True,physical_success=None,
        suffix_translation_m=float(sum(delta[inds[0]:])),suffix_max_displacement_m=float(max(np.linalg.norm(np.array(r['base'][:2])-endpoint[:2]) for r in suffix)),
        continuation_translation_m=float(np.linalg.norm(np.array(b['actual_base'][:2])-endpoint[:2])),
        maximum_lateral_mps=float(max(abs(r['lateral_mps']) for r in rows)),max_joint_rate_suffix=float(max(r['joint_rate_max'] for r in suffix)),
        max_base_speed=float(max(abs(r['action'][0]) for r in suffix)),max_yaw_rate=float(max(abs(r['action'][1]) for r in suffix)),
        max_v_jump=float(max(abs(np.diff(a[:,0])))),max_w_jump=float(max(abs(np.diff(a[:,1])))),
        max_raw_ik_jump=float(max(r['raw_ik_joint_jump'] or 0 for r in suffix)),
        final_home_error=float(max(abs(q[:6]-HOME[:6]))),constant_z=True,
        limitations=['50Hz ideal tracking with 10Hz warm-start IK references; no physical dynamics','Inherited endpoint linear traverse stops discontinuously; no relocated braking point','Full attempted suffix retained even after failed IK; not evidence of sequential gate completion'])
    write(out/'summary.json',summary);emit(summary)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--candidate',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden')
    cfg=read(Path(__file__).with_name('config.json'));c=next(x for x in cfg['candidates'] if x['id']==a.candidate)
    run=Path(a.output);run.mkdir(parents=True,exist_ok=True)
    if len(list(run.glob('c[0-9]')))>=3:raise RuntimeError('Three candidate budget exhausted')
    out=run/c['id'];out.mkdir(exist_ok=False)
    write(out/'config.json',dict(global_config=cfg,candidate=c))
    write(out/'receipt.json',dict(argv=sys.argv,python=sys.executable,cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),started=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('config.json')]}))
    try:generate(out,c,cfg)
    except Exception as e:write(out/'exception.json',dict(type=type(e).__name__,message=str(e)));raise
if __name__=='__main__':main()
