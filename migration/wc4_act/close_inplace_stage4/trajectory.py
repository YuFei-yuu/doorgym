"""Exactly ONE fixed-base local trajectory; prefix-gated, failed suffix diagnostic."""
import argparse,os,json,math,itertools,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from common import *
from close_inplace_stage3.geometry import separating_gap,transform

def pairkey(x):return frozenset((tuple(x['a'][:2]),tuple(x['b'][:2])))
def generate(run):
    cfg=read(run/'config.json');plan=read(Path(__file__).with_name('trajectory_plan.json'));ledger=read(run/'local-ledger.json')
    assert ledger['used']==11 and all(x['status']=='completed' for x in ledger['groups'])
    out=run/'trajectory';out.mkdir(exist_ok=False)
    write(out/'plan.json',plan);write(out/'config.json',cfg)
    write(run/'trajectory-ledger.json',dict(limit=1,used=1,status='started',complete_candidates=0,physics=0))
    G=Geometry();S=FullPose(G.robot);start,_=reference(cfg);B=np.array(start['base']);theta0=start['theta'];q=np.array(start['q']);t=0.;dt=cfg['dt'];rows=[];geometry_rows=[];interval_rows=[];solves=[]
    write(out/'receipt.json',dict(argv=sys.argv,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={str(p):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('common.py').resolve(),Path(__file__).with_name('trajectory_plan.json').resolve(),run/'local-ledger.json',ROOT/cfg['source_stage3']/'single/trace.jsonl']},start=start,base_fixed=True))
    trace=(out/'trace.jsonl').open('x');geomlog=(out/'geometry.jsonl').open('x');interval_log=(out/'intervals.jsonl').open('x');iklog=(out/'ik-solves.jsonl').open('x')
    initial=G.check(B,q,theta0);baseline={pairkey(x):x['gap_m'] for x in initial['overlaps'] if x['category']=='self'};active=set(baseline);last_depth=dict(baseline)
    write(out/'baseline.json',dict(home=initial,source_third_round_baseline=str(ROOT/cfg['source_stage3']/'single/self-baseline.json'),self_exclusions=G.excluded,rule=plan['baseline']))
    allpairs=G.selfpairs[:];eps=cfg['geometry_epsilon_m'];issues=[];baseline_intervals=0;raw_cache={};critical_cache={}
    def raw_geometry(Q,a):
        key=(np.asarray(Q,dtype=float).tobytes(),float(a))
        if key not in raw_cache:raw_cache[key]=G.check(B,Q,a)
        return raw_cache[key]
    def critical_geometry(Q,a,exclusions):
        raw=raw_geometry(Q,a)
        if exclusions:
            key=(np.asarray(Q,dtype=float).tobytes(),float(a),tuple(sorted(tuple(sorted(x)) for x in exclusions)))
            if key in critical_cache:return raw,critical_cache[key]
            G.selfpairs=[(i,j) for i,j in allpairs if frozenset(((G.R[i].link,G.R[i].name),(G.R[j].link,G.R[j].name))) not in exclusions]
            try:other=G.check(B,Q,a)['minimums']['self']
            finally:G.selfpairs=allpairs
            critical_cache[key]=other
        else:other=raw['minimums']['self']
        return raw,other
    # Source box 2 corners in the EE frame. No deployment overlap exemption.
    vertices=np.array(list(itertools.product([-.015,.015],[-.03,.03],[-.03,-.01])))
    def selected_plane_certificate(q0,q1,a0,a1):
        dq=np.asarray(q1[:6])-q0[:6];da=a1-a0;M=10*(float(abs(dq).sum())+abs(da))**2
        values=[];derivative=None
        for u in [0,.5,1]:
            Q=(1-u)*np.asarray(q0)+u*np.asarray(q1);a=(1-u)*a0+u*a1;n=normal(a);F,P,U,JP,JU,A,O=S.state(Q[:6]);T=baseT(B);world=apply(T@F,vertices)
            values.append((world-np.array([2.467,-.463,.01]))@n+.002)
            if u==.5:
                pts=apply(F,vertices);derivative=np.array([n@(T[:3,:3]@np.cross(A,p-O).T@dq) for p in pts])+da*((world-np.array([2.467,-.463,.01]))@np.array([-math.sin(a),math.cos(a),0]))
        f0,_,f1=values;upper=np.where(derivative-M/2>=0,f1,np.where(derivative+M/2<=0,f0,np.maximum(f0,f1)+M/8))
        return bool(max(upper)<=eps),float(max(upper))
    def other_selected_gap(Q,a):
        F=baseT(B)@S.fk(Q);stator=F@transform([-.135,0,0]);piece=next(x for x in G.R if x.link=='gripperStator' and x.name=='2');A=piece.world(stator)
        dp=G.door.fk({'door_hinge_joint':a,'handle_spindle_joint':0},transform([2.61,0,.01],[0,0,math.pi/2]))
        return min(separating_gap(A,d.world(dp[d.link])) for d in G.D if d.link=='door_panel' and d.name!='door_slab')
    def check_interval(q0,q1,a0,a1,t0,t1,phase,exclusions,depth=0):
        raw,gself=critical_geometry(q0,a0,exclusions);dq=float(abs(np.asarray(q1)-q0).sum());da=abs(a1-a0);bound=2*dq+1.2*da;self_bound=4*dq
        plane,plane_upper=selected_plane_certificate(q0,q1,a0,a1)
        selected_ok=raw['minimums']['intended']['gap_m']>bound or (plane and other_selected_gap(q0,a0)>bound)
        external_ok=raw['minimums']['external']['gap_m']>bound
        self_ok=gself['gap_m']>self_bound
        far_ok=2.5-raw['whole_max_x']>2*dq
        rawbad=raw['minimums']['external']['gap_m']< -eps or raw['minimums']['intended']['gap_m']< -eps or gself['gap_m']< -eps
        okay=selected_ok and external_ok and self_ok and far_ok
        if not okay and not rawbad and depth<3:
            qm=(np.asarray(q0)+q1)/2;am=(a0+a1)/2;tm=(t0+t1)/2
            left=check_interval(q0,qm,a0,am,t0,tm,phase,exclusions,depth+1)
            right=check_interval(qm,q1,am,a1,tm,t1,phase,exclusions,depth+1)
            return left and right
        item=dict(start=t0,end=t1,phase=phase,certified_new_geometry=bool(okay),baseline_conditional=bool(exclusions),baseline_pairs=[sorted(list(x)) for x in exclusions],selected_plane_upper_m=plane_upper,selected_certified=bool(selected_ok),external_certified=bool(external_ok),self_certified=bool(self_ok),far_certified=bool(far_ok),depth=depth,minimums=dict(raw['minimums'],critical_self=gself))
        interval_rows.append(item);interval_log.write(json.dumps(item)+'\n')
        return bool(okay)
    prev_q=q.copy();prev_theta=theta0;history_ok=True;gate=None
    def record(phase,target_q,a,p,R,ik=None):
        nonlocal q,t,prev_q,prev_theta,history_ok,baseline_intervals
        t0=t;t+=dt;before=q.copy();q=q+np.clip(np.asarray(target_q)-q,-cfg['joint_speed']*dt,cfg['joint_speed']*dt)
        raw=raw_geometry(q,a);negative={pairkey(x):x for x in raw['overlaps'] if x['category']=='self' and x['gap_m']< -eps};allowed=set();baseline_new=[]
        for key in list(active):
            if key not in negative:active.remove(key);continue
            gap=negative[key]['gap_m']
            if gap<baseline[key]-eps or gap<last_depth[key]-eps:baseline_new.append(negative[key])
            else:allowed.add(key)
            last_depth[key]=gap
        novel_self=[x for key,x in negative.items() if key not in allowed]
        interval_exclusions=allowed|({k for k in baseline if k in active} if t0==0 else previous_allowed[0])
        inter=check_interval(before,q,prev_theta,a,t0,t,phase,interval_exclusions)
        if interval_exclusions:baseline_intervals+=1
        F=baseT(B)@S.fk(q);actual=F[:3,3]+F[:3,:3]@S.pad
        pos=float(np.linalg.norm(actual-p));ori=float(Rotation.from_matrix(R@F[:3,:3].T).magnitude());n=normal(a)
        normal_error=float(np.arccos(np.clip((F[:3,:3]@n)@n,-1,1)))
        rotq=Rotation.from_matrix(F[:3,:3]).as_quat();twist=2*math.atan2(float(rotq[:3]@n),float(rotq[3]));twist=(twist+math.pi)%(2*math.pi)-math.pi
        tarq=Rotation.from_matrix(R).as_quat();target_twist=2*math.atan2(float(tarq[:3]@n),float(tarq[3]));target_twist=(target_twist+math.pi)%(2*math.pi)-math.pi
        raw_ok=raw['minimums']['external']['gap_m']>=-eps and raw['minimums']['intended']['gap_m']>=-eps and not novel_self and raw['far']
        step_ok=bool(raw_ok and inter and pos<=cfg['position_gate_m'] and ori<=cfg['full_orientation_gate_rad'])
        if not step_ok:issues.append(dict(t=t,phase=phase,pose_position=pos,pose_orientation=ori,interval_certified=inter,external_gap=raw['minimums']['external']['gap_m'],selected_gap=raw['minimums']['intended']['gap_m'],new_self=novel_self))
        history_ok=history_ok and step_ok
        row=dict(t=t,source_aligned_t=start['t']+t,phase=phase,kind='synthetic_ideal_bounded_joint_tracking',base=B.tolist(),q=q.tolist(),theta=a,z=.5,action=np.r_[0.,0.,q].tolist(),target=p.tolist(),target_rotation=R.tolist(),actual_rotation=F[:3,:3].tolist(),pad=actual.tolist(),position_error_m=pos,orientation_error_rad=ori,normal_error_rad=normal_error,twist_rad=twist,target_twist_rad=target_twist,joint_rate=((q-before)/dt).tolist(),joint_slack=float(np.min(np.r_[q[:6]-S.lo,S.hi-q[:6]])),ik=ik,new_geometry_step_pass=step_ok,history_new_geometry_pass=history_ok,baseline_conditional=bool(allowed),status='candidate_reference' if history_ok else 'diagnostic_suffix_after_failed_or_unproved_prefix')
        rows.append(row);trace.write(json.dumps(row)+'\n');geometry_rows.append(dict(t=t,phase=phase,q=q.tolist(),theta=a,baseline_allowed=[sorted(list(x)) for x in allowed],**raw));geomlog.write(json.dumps(geometry_rows[-1])+'\n')
        previous_allowed[0]=allowed;prev_theta=a;prev_q=q.copy()
    previous_allowed=[set(baseline)]
    write(out/'geometry-method.json',dict(epsilon_m=eps,source_method='Same source collision/visual geometry and separating-axis lower bounds as local checks; no selected-piece overlap exemption during deployment or approach.',
        interpolation='Piecewise linear stored joint/door states, fixed base. Up to 3 bisections per 20ms interval; unresolved intervals cannot pass.',
        selected_box='All eight actual source box-2 vertices must stay behind the slab back plane, or ordinary SAT proves separation; other door-panel pieces checked separately.',
        plane_bound='Endpoint/midpoint derivative certificate with M=10*(sum(abs(delta_q6))+abs(delta_theta))^2; robot lever radius <2m, point-to-hinge <3m; not a physical tracking certificate.',
        baseline='Only connected non-deepening departure from recorded HOME overlap; recurrence or deepening is a new failure. Any resulting pass remains baseline-conditional, not unconditional.',
        later_translation_plan_m=.45,executed_translation_m=0.,stop_door_deg=70))
    def joint_arc(phase,goal,seconds):
        q0=q.copy()
        for i in range(1,round(seconds/dt)+1):
            desired=(1-smooth(i/round(seconds/dt)))*q0+smooth(i/round(seconds/dt))*goal;F=baseT(B)@S.fk(desired);p=F[:3,3]+F[:3,:3]@S.pad
            record(phase,desired,theta0,p,F[:3,:3])
        print(json.dumps(dict(phase=phase,t=t,history_new_geometry_pass=history_ok)),flush=True)
    d=plan['durations'];joint_arc('deploy_a',np.array(ledger['groups'][0]['q']),d['deploy_a']);joint_arc('deploy_b',np.array(ledger['groups'][1]['q']),d['deploy_b'])
    F=baseT(B)@S.fk(q);P0=F[:3,3]+F[:3,:3]@S.pad;R0=F[:3,:3];P1=point(cfg,theta0);R1=pose_rotation(theta0,plan['approach_contact_roll_deg']);slerp=Slerp([0,1],Rotation.from_matrix(np.array([R0,R1])))
    count=round(d['align_approach']/dt)
    for i in range(1,count+1):
        s=smooth(i/count);p=(1-s)*P0+s*P1;R=slerp(s).as_matrix();sol,ik=S.solve(B,p,R,q,cfg['continuous_ik_nfev']);iklog.write(json.dumps(dict(t=t+dt,phase='align_approach',q=sol.tolist(),**ik))+'\n');record('align_approach',np.r_[sol,HOME[-1]],theta0,p,R,ik)
    last=rows[-1];gate=dict(prefix_new_geometry_and_tracking_pass=history_ok,endpoint_position=last['position_error_m'],endpoint_orientation=last['orientation_error_rad'],endpoint_normal=last['normal_error_rad'],endpoint_joint_speed=max(abs(np.array(last['joint_rate'])[:6])),baseline_conditional=baseline_intervals>0)
    gate['passed_under_declared_baseline']=bool(history_ok and gate['endpoint_position']<=cfg['position_gate_m'] and gate['endpoint_orientation']<=cfg['full_orientation_gate_rad'] and gate['endpoint_normal']<=cfg['normal_gate_rad'] and gate['endpoint_joint_speed']<=.05)
    gate['unconditional_clearance_claim']=False if baseline_intervals else gate['passed_under_declared_baseline'];write(out/'approach-gate.json',gate)
    if not gate['passed_under_declared_baseline']:history_ok=False
    count=round(d['push_to_70']/dt);end=math.radians(70)
    for i in range(1,count+1):
        a=theta0-(theta0-end)*smooth(i/count);a=end if i==count else a
        psi=plan['approach_contact_roll_deg']+(plan['roll_at_80_and_70_deg']-plan['approach_contact_roll_deg'])*smooth((theta0-a)/(theta0-math.radians(80)))
        p=point(cfg,a);R=pose_rotation(a,psi);sol,ik=S.solve(B,p,R,q,cfg['continuous_ik_nfev']);iklog.write(json.dumps(dict(t=t+dt,phase='push_to_70',q=sol.tolist(),**ik))+'\n');record('push_to_70',np.r_[sol,HOME[-1]],a,p,R,ik)
    for f in [trace,geomlog,interval_log,iklog]:f.close()
    write(out/'issues.json',issues)
    write(out/'summary.json',dict(trajectory_count=1,local_groups=ledger['used'],complete_candidates=0,physics=0,ordered_completion_under_declared_baseline=bool(history_ok and gate['passed_under_declared_baseline']),unconditional_pass=False,gate=gate,first_issue=issues[0] if issues else None,issue_samples=len(issues),unproved_intervals=sum(not x['certified_new_geometry'] for x in interval_rows),baseline_intervals=baseline_intervals,samples=len(rows),geometry_intervals=len(interval_rows),duration=t,stop_door_deg=math.degrees(rows[-1]['theta']),base_translation_m=0.,base_yaw_change_rad=0.,preserved_later_translation_plan_m=.45,trace_sha256=sha(out/'trace.jsonl')))
    write(run/'trajectory-ledger.json',dict(limit=1,used=1,status='completed',complete_candidates=0,physics=0,diagnostic_suffix=not history_ok));print(json.dumps(read(out/'summary.json')),flush=True)
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('hide CUDA')
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();generate(a.run)
