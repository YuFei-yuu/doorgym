"""One-shot CPU candidate. An approach gate failure cannot enter close."""
import argparse,os,json,math,sys,time,subprocess
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from geometry import Geometry,Solver,ROOT,HOME,NAMES,LEGS,baseT,apply,write,read,sha
from close_inplace_stage2.candidate import smooth

def generate(run):
    out=run/'single';out.mkdir(exist_ok=False)
    cfg=read(Path(__file__).with_name('config.json'));write(out/'config.json',cfg)
    G=Geometry();S=Solver(G.robot,'corner');dt=cfg['dt']
    source=ROOT/cfg['measured_source'];source_rows=[json.loads(l) for l in (source/'states.jsonl').open()]
    b=next(r for r in source_rows if r['passed']);ref=read(source/'reference.json')
    old=[json.loads(l) for l in (ROOT/cfg['reference_c2']/'trace.jsonl').open()]
    old_cont=[r for r in old if r['phase']=='traverse_continuation']
    local=read(ROOT/cfg['deployment_source']);hover=local['selected'];assert hover and hover['offset']==cfg['hover_offset_m']
    bindings=[Path(__file__).resolve(),Path(__file__).with_name('config.json').resolve(),ROOT/cfg['deployment_source'],ROOT/cfg['reference_c2']/'trace.jsonl',source/'states.jsonl',source/'reference.json']
    write(out/'receipt.json',dict(argv=sys.argv,python=sys.executable,cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),started=time.time(),source_sha256={str(p):sha(p) for p in bindings},complete_candidate_budget=1))
    endpoint=np.r_[ref['old_expert_final_base'],math.pi];B=np.array(b['actual_base']);B[2]=math.pi
    q=np.array(b['arm_q']);cmd=np.array(b['action'][2:]);theta0=float(b['door_q'][0]);theta=theta0;t=float(b['t']);rows=[];solves=[]
    S.axis=np.array([math.cos(theta0),math.sin(theta0),0])
    write(out/'measured-anchor.json',dict(row=b,previous_row=source_rows[b['step']-1],meaning='old measured physical sample; later states synthetic'))
    with (out/'measured-prefix.jsonl').open('x') as f:
        for r in source_rows[:b['step']+1]:f.write(json.dumps({k:r[k] for k in ['step','t','phase','actual_base','arm_q','action','door_q','ee_actual']})+'\n')
    stream=(out/'trace.jsonl').open('x');solve_stream=(out/'ik-solves.jsonl').open('x')
    def save(phase,base,desired,angle,limit,target=None,axis=None,ik=None,source_kind='synthetic',ee=False,analytic_v=None):
        nonlocal B,q,cmd,theta,t
        before=B.copy();qbefore=q.copy();cmd_before=cmd.copy();t+=dt;B=np.array(base);theta=float(angle)
        cmd=cmd+np.clip(np.asarray(desired)-cmd,-limit*dt,limit*dt)
        cmd[:6]=np.clip(cmd[:6],S.lo,S.hi)
        q=q+np.clip(cmd-q,-limit*dt,limit*dt) # explicitly ideal, bounded tracking; not dynamics
        delta=B[:2]-before[:2];v=float(delta@np.array([math.cos(before[2]),math.sin(before[2])])/dt);lat=float(delta@np.array([-math.sin(before[2]),math.cos(before[2])])/dt);w=(B[2]-before[2])/dt
        T=baseT(B)@S.fk(q[:6]);pad=T[:3,3]+T[:3,:3]@S.pad;normal=T[:3,:3]@S.axis
        poses=G.robot.fk(dict(LEGS,**dict(zip(NAMES,q))),baseT(B));cloud=G.robot.cloud(poses)
        row=dict(t=t,phase=phase,kind=source_kind,base=B.tolist(),z=.5,pitch=0.,q=q.tolist(),action=np.r_[v,w,cmd].tolist(),theta=theta,
            pad=pad.tolist(),ee=T[:3,3].tolist(),contact_local_axis=S.axis.tolist(),joint_speed_limit=limit,
            commanded_joint_rate=((cmd-cmd_before)/dt).tolist(),synthetic_joint_rate=((q-qbefore)/dt).tolist(),
            raw_desired_q=np.asarray(desired).tolist(),lateral_mps=lat,analytic_base_v=analytic_v,
            target=None if target is None else np.asarray(target).tolist(),target_axis=None if axis is None else np.asarray(axis).tolist(),
            tracking_m=None if target is None else float(np.linalg.norm((T[:3,3] if ee else pad)-target)),
            tracking_axis_rad=None if axis is None else float(np.arccos(np.clip(normal@axis,-1,1))),
            joint_slack=float(np.min(np.r_[q[:6]-S.lo,S.hi-q[:6]])),whole_max_x=float(cloud[:,0].max()),far=bool(cloud[:,0].max()<2.5),ik=ik)
        rows.append(row);stream.write(json.dumps(row)+'\n')
    # Same spatial traversal and same frozen virtual joint path; only its clock changes.
    startB=B.copy();distance=float(startB[0]-endpoint[0]);v0=float(b['action'][0]);N=math.ceil(2*distance/v0/dt);Tstop=N*dt
    a1=v0*Tstop;a3=10*distance-6*a1;a4=-15*distance+8*a1;a5=6*distance-3*a1
    progress=np.r_[0,[(startB[0]-r['base'][0])/distance for r in old_cont]];progress[-1]=1.
    qref=PchipInterpolator(progress,np.array([cmd.tolist()]+[r['q'] for r in old_cont]),axis=0)
    parking=[]
    for k in range(1,N+1):
        u=k/N;travel=a1*u+a3*u**3+a4*u**4+a5*u**5;vel=(a1+3*a3*u*u+4*a4*u**3+5*a5*u**4)/Tstop
        if k==N:travel=distance;vel=0.
        base=np.array([startB[0]-travel,startB[1],math.pi]);desired=qref(np.clip(travel/distance,0,1))
        save('retimed_traverse',base,desired,theta0,cfg['joint_speed_inherited'],target=np.array(b['ee_target']),ee=True,source_kind='synthetic_retimed_frozen_DLS',analytic_v=vel)
        parking.append(dict(elapsed=k*dt,progress=travel/distance,v=vel,x=base[0]))
    write(out/'parking.json',dict(start=startB.tolist(),endpoint=endpoint.tolist(),initial_speed=v0,duration=Tstop,polynomial=[0,a1,0,a3,a4,a5],samples=parking,
        meaning='C2 scalar position polynomial; unchanged straight spatial path. Frozen virtual joint-command samples re-timed by monotone PCHIP, then inherited rate bound. No re-simulation of opening.'))
    write(out/'handoff.json',dict(synthetic_arrival=rows[-1],prior_measured_anchor=b['step'],door_angle_hypothesis=theta0,not_home_error=float(max(abs(q[:6]-HOME[:6]))),
        future_measured_checks=['retimed door angle/rate and contacts','actual endpoint q/qd and EE','physical braking and drive tracking','full-body bound with actual leg joints'],source_kind='synthetic, not measured arrival'))
    def segment(name,seconds,bfun=None,pfun=None,ufun=None,qfun=None,thetafun=None,lafun=None,ee=False):
        nonlocal q
        n=round(seconds/dt);startbase=B.copy();startq=cmd.copy();starttheta=theta
        for i in range(1,n+1):
            s=float(smooth(i/n));base=startbase if bfun is None else bfun(s);angle=starttheta if thetafun is None else thetafun(s)
            if lafun is not None:S.axis=np.array(lafun(s))
            point=None if pfun is None else pfun(s);axis=None if ufun is None else ufun(s);ik=None
            if qfun is not None:desired=qfun(s)
            elif point is not None:
                sol,ik=S.solve_point(base,point,axis,q,cfg['ik_max_nfev'],ee=ee);desired=np.r_[sol,HOME[-1]]
                result=dict(t=t+dt,phase=name,q=sol.tolist(),target=np.asarray(point).tolist(),**ik);solves.append(result);solve_stream.write(json.dumps(result)+'\n')
            else:desired=startq
            save(name,base,desired,angle,cfg['joint_speed_suffix'],point,axis,ik,ee=ee)
        stream.flush();solve_stream.flush();print(json.dumps(dict(phase=name,t=t,max_error=max([r['tracking_m'] for r in rows if r['phase']==name and r['tracking_m'] is not None]or[0]))),flush=True)
    d=cfg['durations'];segment('endpoint_hold',d['endpoint_hold'])
    # Withdraw by one bounded joint arc to an explicitly solved endpoint, no abrupt IK commands.
    p0=np.array(rows[-1]['ee']);pwithdraw=p0+np.array([0,.14,0]);qwithdraw,e=S.solve_point(B,pwithdraw,None,q,120,ee=True);write(out/'withdraw-endpoint.json',dict(target=pwithdraw.tolist(),q=qwithdraw.tolist(),ik=e))
    qa=cmd.copy();qb=np.r_[qwithdraw,HOME[-1]];segment('withdraw',d['withdraw'],qfun=lambda s:(1-s)*qa+s*qb)
    qa=cmd.copy();segment('fold',d['fold'],qfun=lambda s:(1-s)*qa+s*HOME)
    yaw=math.radians(cfg['initial_yaw_deg']);segment('turn',d['turn'],bfun=lambda s:np.r_[endpoint[:2],math.pi+s*(yaw-math.pi)])
    qa=cmd.copy();qh=np.array(hover['q']);segment('deploy',d['deploy'],qfun=lambda s:(1-s)*qa+s*qh)
    def point(a,offset=0):return np.array([2.467-cfg['radius_m']*math.sin(a)-(.002+offset)*math.cos(a),-.463+cfg['radius_m']*math.cos(a)-(.002+offset)*math.sin(a),cfg['contact_height_m']])
    actual_pad=np.array(rows[-1]['pad']);goal=point(theta0);normal=np.array([math.cos(theta0),math.sin(theta0),0])
    segment('seek',d['seek'],pfun=lambda s:(1-s)*actual_pad+s*goal,ufun=lambda s:normal)
    last=rows[-1];geometry=G.check(B,q,theta0)
    gate=dict(position_error_m=last['tracking_m'],axis_error_rad=last['tracking_axis_rad'],joint_speed=max(abs(np.array(last['synthetic_joint_rate'])[:6])),base_speed=abs(last['action'][0]),base_yaw_rate=abs(last['action'][1]),far=last['far'],geometry=geometry)
    gate['passed']=bool(gate['position_error_m']<=cfg['position_gate_m'] and gate['axis_error_rad']<=cfg['axis_gate_rad'] and gate['joint_speed']<=cfg['gate_joint_speed_rad_s'] and gate['base_speed']<=cfg['gate_base_speed'] and gate['base_yaw_rate']<=cfg['gate_base_speed'] and gate['far'] and geometry['minimums']['external']['gap_m']>=0 and geometry['minimums']['self']['gap_m']>=0)
    gate['meaning']='Necessary geometric approach gate only; no real contact/force/latch feedback';write(out/'approach-gate.json',gate)
    if gate['passed']:
        turnstart=math.radians(cfg['turn_start_door_deg']);follow=math.radians(cfg['follow_start_door_deg'])
        def closebase(s):
            a=(1-s)*theta0;return np.array([endpoint[0]+cfg['follow_m']*smooth((follow-a)/follow),endpoint[1],yaw+(2*math.pi-yaw)*smooth((turnstart-a)/(turnstart-follow))])
        segment('close',d['close'],bfun=closebase,pfun=lambda s:point((1-s)*theta0),ufun=lambda s:np.array([math.cos((1-s)*theta0),math.sin((1-s)*theta0),0]),thetafun=lambda s:(1-s)*theta0,lafun=lambda s:np.array([math.cos((1-s)*theta0),math.sin((1-s)*theta0),0]))
        segment('settle',d['settle'])
        P=np.array(rows[-1]['pad']);segment('release',d['release'],pfun=lambda s:P-np.array([.12*s,0,0]),ufun=lambda s:np.array([1.,0,0]))
    else:
        write(out/'blocked-suffix.json',dict(reason='approach gate failed; close/settle/release NOT executed or claimed validated',stages_not_executed=['close','settle','release']))
    qa=cmd.copy();segment('retract' if gate['passed'] else 'abort_retract',d['retract'],qfun=lambda s:(1-s)*qa+s*HOME)
    start_yaw=B[2];segment('restore',d['restore'],bfun=lambda s:np.r_[B[:2],start_yaw+s*(math.pi-start_yaw)])
    segment('hold',d['hold']);stream.close();solve_stream.close()
    suffix=[r for r in rows if r['phase']!='retimed_traverse'];a=np.array([r['action'] for r in rows]);bases=np.array([r['base'] for r in rows])
    info=dict(approach_gate_passed=gate['passed'],close_evaluated=gate['passed'],samples=len(rows),total_reference_time_s=t,post_measured_duration_s=t-b['t'],parking_duration_s=Tstop,
        after_endpoint_translation_m=float(np.linalg.norm(np.diff(np.array([r['base'][:2] for r in suffix]),axis=0),axis=1).sum()),
        maximum_displacement_after_endpoint_m=max(float(np.linalg.norm(np.array(r['base'][:2])-endpoint[:2])) for r in suffix),
        maximum_lateral_mps=max(abs(r['lateral_mps']) for r in rows),minimum_far_margin_m=2.5-max(r['whole_max_x'] for r in rows),far_return_samples=sum(not r['far'] for r in rows),
        max_command_joint_rate_suffix=max(float(max(abs(np.array(r['commanded_joint_rate'])[:6]))) for r in suffix),
        max_command_joint_rate_inherited=max(float(max(abs(np.array(r['commanded_joint_rate'])[:6]))) for r in rows if r['phase']=='retimed_traverse'),
        max_v_step=float(abs(np.diff(a[:,0])).max()),max_w_step=float(abs(np.diff(a[:,1])).max()),final_home_error=float(max(abs(q[:6]-HOME[:6]))),
        full_candidate_attempts=1,physics=0,physical_success=None,source_meaning='Measured prefix separate; re-timed synthetic handoff; ideal bounded tracking; prescribed door motion after geometric approach gate. No actual latch or forces.')
    write(out/'summary.json',info);print(json.dumps(info),flush=True)
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden')
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();generate(a.run)
