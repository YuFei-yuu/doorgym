"""Bounded CPU design audit; never imports Gym, torch, policy or a simulator.

Reference sweeps and exact-tracking synthetic feedback are distinct evidence.
A negative conservative convex gap is an unresolved risk, not a contact claim.
"""
import argparse, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from fullcycle_compact import Arm, ClosingCurve, CompactExpert, rounded_route, wrap
from fullcycle_geometry import URDF, transform, apply
from fullcycle_scene import SceneFrame
BASE='5b782e0ba0374d6e2864a0e5863469ced1cdc146'
HOME=np.r_[np.zeros(6),-1.1775]
LEGS={side+suffix:val for side in ['FL','FR','RL','RR'] for suffix,val in [('_hip_joint',0),('_thigh_joint',.8),('_calf_joint',-1.5)]}
NAMES=['joint%d'%i for i in range(1,7)]+['jointGripper']
def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(x):print(json.dumps(x,allow_nan=False),flush=True)
def read(p):return json.loads(Path(p).read_text())
def boundary(path):
    with path.open() as f:
        for line in f:
            r=json.loads(line)
            if r['passed']:return r
    raise ValueError('No passed boundary')
def recipes():
    old=json.loads(subprocess.check_output(['git','show',BASE+':migration/wc4_act/fullcycle_compact.json'],cwd=ROOT,text=True))['compact_expert']
    new=read(Path(__file__).with_name('geometry.json'))['compact_expert']
    return {'old':old,'new':new}
def poses(robot,base,q,scene):
    return robot.fk(dict(LEGS,**dict(zip(NAMES,q))),scene.base_transform(base))
def reference(out):
    robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');arm=Arm(robot)
    src=ROOT/'.repro/runs/wc4-fullcycle-compact-r7-20261021/attempt_000/states.jsonl'
    b=boundary(src);write(out/'boundary.json',dict(source=str(src),meaning='Historical physical handoff only, not a new execution',row=b))
    trials=[];bindings={}
    for run in ['wc4-fulltask-pilot-r8-20261021','wc4-fulltask-expand-r9-20261021']:
        p=ROOT/'.repro/runs'/run/'manifest.json';m=read(p);bindings[str(p)]=sha(p)
        trials.extend(m['trials'])
    keys=['robot_x','robot_y','robot_z','robot_pitch','robot_yaw','door_x','door_y','door_wall_x_offset','door_push_distance']
    ranges={k:[min(t['effective_sampling'][k] for t in trials),max(t['effective_sampling'][k] for t in trials)] for k in keys}
    write(out/'historical-distribution.json',dict(manifest_sha256=bindings,trial_count=len(trials),observed_ranges=ranges,
        configured_ranges=dict(robot_x_offset=[-.70,0],robot_y_offset=[-.10,.04],robot_z=[.47,.53],robot_pitch_deg=[0,5],robot_yaw_offset=[-.03,.03],door_xy_offset=[-.03,.03],wall_x_offset=[-.03,.03],camera_position_each_axis=[-.02,.02],camera_rotation_each_axis_deg=[-5,5]),
        caveat='Observed manifest ranges are not continuous coverage. Initial x/y/yaw affect the unmodified opening and its handoff; they are not direct offsets to the fixed closing curve. Camera jitter does not change geometry.'))
    results={};all_samples={}
    for label,c in recipes().items():
        curve=ClosingCurve(c);theta=np.linspace(math.pi/2,0,1001);ref=np.array([curve.at(a) for a in theta])
        der=np.gradient(ref[:,:2],math.pi/2-theta,axis=0)
        lateral=-np.sin(ref[:,2])*der[:,0]+np.cos(ref[:,2])*der[:,1]
        # Same historical bypass pose for an apples-to-apples reference drawing.
        route=rounded_route([1.50075101852417,.18000000715255737,math.pi],ref[0,:3],c)
        stage=route[-1];dvec=ref[0,:2]-stage[:2]
        route_diff=np.diff(route[:,:2],axis=0);head=(route[:-1,2]+route[1:,2])/2
        route_lat=-np.sin(head)*route_diff[:,0]+np.cos(head)*route_diff[:,1]
        normal=np.c_[np.cos(theta),np.sin(theta)]
        panel_gap=np.abs(np.sum((ref[:,:2]-curve.point(theta,0))*normal,axis=1))
        cases=[('grid_p%g_z%g'%(pitch,z),math.radians(pitch),z,91) for pitch in [0,2.5,5] for z in [.47,.50,.53]]
        if label=='new':cases += [('manifest_%s'%t['seed'],t['scene_frame']['base_pitch'],t['scene_frame']['base_z_reference'],19) for t in trials]
        summaries=[];samples=[]
        for name,pitch,z,n in cases:
            q=np.array([0,1.7,-1.3,-.4,0,math.pi/2]);rows=[]
            for angle in np.linspace(math.pi/2,0,n):
                r=curve.at(angle);B=transform([*r[:2],z],[0,pitch,r[2]])
                q,err=arm.solve(np.linalg.inv(B)@curve.tool(angle),q,1.,120)
                cloud=robot.cloud(robot.fk(dict(LEGS,**dict(zip(NAMES,np.r_[q,0.]))),B))
                rows.append(dict(case=name,pitch=pitch,z=z,theta=float(angle),base=r[:3].tolist(),q=np.r_[q,0.].tolist(),**err,
                    joint_slack_rad=float(np.min(np.r_[q-arm.lo,arm.hi-q])),whole_max_x=float(cloud[:,0].max())))
            summaries.append(dict(case=name,max_position_error_m=max(x['position_error_m'] for x in rows),max_orientation_error_rad=max(x['orientation_error_rad'] for x in rows),min_joint_slack_rad=min(x['joint_slack_rad'] for x in rows),max_whole_x=max(x['whole_max_x'] for x in rows)))
            samples+=rows
        result=dict(route_length_m=float(np.linalg.norm(route_diff,axis=1).sum()),route_bounds=[route[:,:2].min(0).tolist(),route[:,:2].max(0).tolist()],second_radius_m=float((stage[0]-c['corridor_x'])/(1+math.sin(stage[2]))),
            approach_length_m=float(np.linalg.norm(dvec)),approach_lateral_residual_m=float(abs(np.dot(dvec,[-math.sin(stage[2]),math.cos(stage[2])]))),approach_heading_jump_rad=float(abs(wrap(stage[2]-ref[0,2]))),
            route_lateral_chord_residual_m=float(max(abs(route_lat))),close_lateral_derivative_residual=float(max(abs(lateral[2:-2]))),min_forward_per_progress=float(ref[:,6].min()),
            min_base_center_panel_plane_distance_m=float(panel_gap.min()),cases=summaries)
        results[label]=result;all_samples[label]=samples
        write(out/(label+'-reference.json'),dict(recipe=c,route=route.tolist(),closing=ref.tolist(),theta=theta.tolist(),panel_center_gap=panel_gap.tolist(),sweep_samples=samples))
        emit(dict(stage='reference',variant=label,summary={k:v for k,v in result.items() if k!='cases'},max_ik_m=max(x['max_position_error_m'] for x in summaries)))
    # Route-fit envelope in design coordinates, not a claimed reachable opening distribution.
    fits=[]
    for x in [1.4,1.5,1.6]:
        for y in [.04,.18,.32]:
            for label,c in recipes().items():
                try:r=rounded_route([x,y,math.pi],ClosingCurve(c).at(math.pi/2)[:3],c);fits.append(dict(variant=label,x=x,y=y,fit=True))
                except ValueError as e:fits.append(dict(variant=label,x=x,y=y,fit=False,error=str(e)))
    write(out/'route-fit-envelope.json',dict(meaning='Synthetic post-fold endpoint sensitivity; does not simulate randomized opening',cases=fits))
    write(out/'reference-summary.json',results)
    plot(out)
def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,7),gridspec_kw={'width_ratios':[1.25,1]})
    ax=axes[0]
    for label,color in [('old','#777777'),('new','#006eb8')]:
        d=read(out/(label+'-reference.json'));r=np.array(d['route']);c=np.array(d['closing']);curve=ClosingCurve(d['recipe'])
        ax.plot(r[:,0],r[:,1],color=color,label=label+' bypass')
        ax.plot([r[-1,0],c[0,0]],[r[-1,1],c[0,1]],':',color=color)
        ax.plot(c[:,0],c[:,1],'--',color=color,label=label+' closing base')
        for a in [90,45,0]:
            v=curve.at(math.radians(a));tool=curve.point(math.radians(a))
            # Illustrative base footprint only; source convex audit is separate.
            box=np.array([[-.35,-.24],[.35,-.24],[.35,.24],[-.35,.24],[-.35,-.24]])
            R=np.array([[math.cos(v[2]),-math.sin(v[2])],[math.sin(v[2]),math.cos(v[2])]])
            box=box@R.T+v[:2];ax.plot(box[:,0],box[:,1],color=color,alpha=.4)
            ax.plot([v[0],tool[0]],[v[1],tool[1]],color=color,alpha=.45);ax.scatter(*tool,color=color,s=14)
        axes[1].plot(np.degrees(d['theta']),d['panel_center_gap'],color=color,label=label)
    h=np.array([2.467,-.463])
    for a in [90,45,0]:
        end=h+1.0*np.array([-math.sin(math.radians(a)),math.cos(math.radians(a))])
        ax.plot([h[0],end[0]],[h[1],end[1]],color='#ad6221',alpha=.7)
        ax.text(end[0],end[1],str(a)+' deg',fontsize=9)
    ax.axvline(2.5,color='black',alpha=.25,label='door/wall reference plane')
    ax.set_aspect('equal');ax.set(xlabel='Reference x (m)',ylabel='Reference y (m)',title='Reference paths and nominal poses (NOT a rollout)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[1].set(xlabel='Nominal door angle (deg)',ylabel='Base-center to infinite panel plane (m)',title='Closing separation (NOT surface clearance)');axes[1].invert_xaxis();axes[1].legend();axes[1].grid(alpha=.2)
    fig.suptitle('WC4 stage 1 | old vs extended reach +5 cm | CPU geometry only',fontsize=14)
    fig.text(.5,.02,'Base rectangles are illustrative; IK and convex-source clearance are reported separately. Door tool target is unchanged.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(out/'trajectory-comparison.png',dpi=170);fig.savefig(out/'trajectory-comparison.svg');plt.close(fig)
def synthetic(out,label,pitch,z,drift):
    robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');c=recipes()[label]
    b=read(out.parent/'reference/boundary.json')['row'];base=np.array(b['actual_base']);q=np.array(b['arm_q']);theta=float(b['door_q'][0]);dt=.02
    scene=SceneFrame(base_z=z,base_pitch=math.radians(pitch));ctl=CompactExpert(robot,c,base,q,b['action'],theta,0.,HOME,math.pi,dt,scene=scene)
    traces=[];events=[];prev=None;hold=0.;latched=False;final_start=None
    with (out/'trace.jsonl').open('x') as stream:
        for k in range(9000):
            t=k*dt;contact=False;oldtheta=theta
            if drift and ctl.phase not in ('close','release','retract','compact_exit_turn','compact_exit_arc','restore_heading','final_hold'):
                theta=math.radians(90-45*np.clip((t-6)/19,0,1))
            if ctl.phase=='compact_seek':contact=ctl.seek_offset<=0 and abs(theta-math.pi/2)<.01
            if ctl.phase=='close' and ctl.close_parameter<=theta+.001:
                contact=True;theta=min(theta,max(0.,ctl.close_parameter))
            if ctl.phase=='close' and theta==0:
                hold+=dt
                if hold>=.54:latched=True
            before=base.copy();beforeq=q.copy();phase_before=ctl.phase
            action,diag=ctl.command(t,base,q,theta,(theta-oldtheta)/dt,contact,contact,latched)
            base[:2]+=action[0]*dt*np.array([math.cos(base[2]),math.sin(base[2])]);base[2]+=action[1]*dt;q=action[2:].copy()
            cloud=robot.cloud(poses(robot,base,q,scene))
            # The target is evaluated at pre-command actual q, so separate target
            # discontinuity/tracking residual from optimizer IK residual.
            target=diag['target_world'];ee=scene.base_transform(base)@ctl.arm.fk(q)
            tracking=None if target is None else float(np.linalg.norm(ee[:3,3]-np.array(target)[:3,3]))
            row=dict(t=t,phase=ctl.phase,phase_before=phase_before,base=base.tolist(),base_before=before.tolist(),q=q.tolist(),action=action.tolist(),theta=theta,pitch=scene.base_pitch,z=z,
                     ik=diag['ik'],target=target,ee=ee[:3,3].tolist(),target_tracking_m=tracking,whole_max_x=float(cloud[:,0].max()))
            stream.write(json.dumps(row,allow_nan=False)+'\n');traces.append(row)
            if ctl.phase!=prev:
                ev=dict(t=t,previous=prev,phase=ctl.phase,base=base.tolist(),theta=theta,joint_delta_rad=float(max(abs(q-beforeq))))
                events.append(ev);emit(ev);prev=ctl.phase
            if ctl.phase=='final_hold':
                if final_start is None:final_start=t
                if t-final_start>=2.02:break
    actions=np.array([r['action'] for r in traces]);da=np.diff(actions,axis=0)/dt
    ik=[r['ik'] for r in traces if r['ik']]
    phase_summary={}
    for phase in sorted({r['phase'] for r in traces}):
        rows=[r for r in traces if r['phase']==phase];errors=[r['ik'] for r in rows if r['ik']]
        phase_summary[phase]=dict(steps=len(rows),max_v=max(r['action'][0] for r in rows),max_abs_w=max(abs(r['action'][1]) for r in rows),
            max_target_tracking_m=max([r['target_tracking_m'] for r in rows if r['target_tracking_m'] is not None] or [0]),
            max_ik_m=max([r['position_error_m'] for r in errors] or [0]),max_ik_rad=max([r['orientation_error_rad'] for r in errors] or [0]))
    result=dict(evidence='Exact joint tracking, planar command integration, invented door/contact/latch feedback. NOT physics, not task success.',
        variant=label,pitch_deg=pitch,z=z,drift45=drift,complete_code_path=ctl.phase=='final_hold',suffix_seconds=traces[-1]['t']+dt,
        historical_prefix_seconds=b['t'],sum_reference_seconds=b['t']+traces[-1]['t']+dt,
        max_ik_m=max(e['position_error_m'] for e in ik),max_ik_rad=max(e['orientation_error_rad'] for e in ik),min_v=float(actions[:,0].min()),
        max_v=float(actions[:,0].max()),max_abs_w=float(abs(actions[:,1]).max()),max_joint_rate=float(abs(da[:,2:]).max()),max_yaw_acceleration=float(abs(da[:,1]).max()),
        max_v_acceleration=float(da[:,0].max()),max_v_deceleration=float(-da[:,0].min()),max_whole_x=max(r['whole_max_x'] for r in traces),phases=phase_summary,events=events,
        final_home_error_rad=float(max(abs(q[:6]-HOME[:6]))),final_yaw_error_rad=abs(wrap(base[2]-math.pi)),
        limitations=['Nominal historical handoff reused with varied pitch/z; does not reproduce randomized opening.', 'Inherited near-closed immediate braking is exempt from the acceleration ramp.', 'No contact forces, actuator lag, depth images, ACT inference, or task scoring.'])
    write(out/'summary.json',result);emit(result)
def clearance(out,trace):
    from fullcycle_collision import collision_pieces,signed_gap
    robotpath=ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf';doorpath=ROOT/'high-level/data/asset/door_set/wc4/model.urdf'
    robot=URDF(robotpath);door=URDF(doorpath);R=collision_pieces(robotpath);D=collision_pieces(doorpath)
    minima={};samples=[];previous=None
    with Path(trace).open() as f:
        for i,line in enumerate(f):
            r=json.loads(line);changed=r['phase']!=previous;previous=r['phase']
            if i%25 and not changed:continue
            scene=SceneFrame(base_z=r['z'],base_pitch=r['pitch']);rp=poses(robot,r['base'],r['q'],scene)
            dp=door.fk({'door_hinge_joint':r['theta'],'handle_spindle_joint':0},transform([2.61,0,.01],[0,0,math.pi/2]))
            worlds=[(d,d.world(dp[d.link])) for d in D];best={}
            for p in R:
                A=p.world(rp[p.link])
                for d,B in worlds:
                    cat=('end_panel' if d.link=='door_panel' and p.link in ('gripperStator','gripperMover') else 'wrist_panel' if d.link=='door_panel' and p.link in ('link06','wrist_down_mount','wrist_up_mount') else 'other_external')
                    gap=signed_gap(A,B)
                    if cat not in best or gap<best[cat]['gap_m']:best[cat]=dict(gap_m=gap,robot_link=p.link,door_link=d.link,robot_piece=p.name,door_piece=d.name,t=r['t'])
            # Whole-body x separation from the nearest possible wall face is
            # conservative even without assuming a finite wall opening.
            wall_gap=2.5-.03-.08/2-r['whole_max_x']
            best['worst_wall_plane']=dict(gap_m=wall_gap,t=r['t'])
            for cat,val in best.items():
                key=r['phase']+'/'+cat
                if key not in minima or val['gap_m']<minima[key]['gap_m']:minima[key]=val
            samples.append(dict(t=r['t'],phase=r['phase'],minimums=best))
    write(out/'clearance.json',dict(trace=str(trace),trace_sha256=sha(trace),sample_interval_s=.5,also_phase_boundaries=True,minimums=minima,samples=samples,
        limitations=['Source collision convex hulls and fixed nominal door transform; no cooked PhysX shapes.', 'Negative gap is unresolved conservative overlap; intended distal contact is listed separately.', 'No self-collision certification; no continuous swept-volume certification.', 'Wall plane uses worst configured x offset -0.03 m and half-thickness 0.04 m.']))
    emit(dict(stage='clearance',samples=len(samples),minimums=minima))
def behavior(out):
    robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');c=recipes()['new'];curve=ClosingCurve(c);goal=curve.at(math.pi/2)[:3]
    start=goal.copy();start[:2]-=c['staging_distance']*np.array([math.cos(goal[2]),math.sin(goal[2])])
    def make(phase):
        ctl=CompactExpert(robot,c,start,HOME,np.r_[0,0,HOME],math.pi/2,0,HOME,math.pi,.02);ctl.phase=phase;return ctl
    ctl=make('compact_rendezvous');a,d=ctl.command(100,start,HOME,math.pi/2,0,False,False,False)
    checks={'deployment_waits_for_measured_approach':bool(np.array_equal(a[2:],HOME))}
    c1=make('compact_rendezvous');c2=make('compact_rendezvous');p=start.copy();p[:2]+=.1*np.array([math.cos(goal[2]),math.sin(goal[2])])
    a1,d1=c1.command(3,p,HOME,math.pi/2,0,False,False,False);a2,d2=c2.command(3,p,HOME,math.pi/4,0,False,False,False)
    checks['fixed90_reference_independent_of_measured_angle']=bool(np.array_equal(a1,a2) and d1['target_world']==d2['target_world'])
    ctl=make('close');ctl.close_parameter=math.pi/2-.1;before=ctl.close_parameter;ctl.command(1,goal,HOME,math.pi/2,0,False,False,False)
    checks['existing_lag_guard_stops_progress']=ctl.close_parameter==before
    ctl=make('release');ctl.command(3,goal,HOME,0,0,False,True,True);checks['release_waits_for_contact_clearance']=ctl.phase=='release'
    write(out/'behavior.json',dict(checks=checks,all_pass=all(checks.values())));emit(checks)
    if not all(checks.values()):raise AssertionError(checks)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['reference','synthetic','clearance','behavior']);p.add_argument('--output',required=True);p.add_argument('--variant',default='new',choices=['old','new']);p.add_argument('--pitch',type=float,default=0);p.add_argument('--z',type=float,default=.5);p.add_argument('--drift45',action='store_true');p.add_argument('--trace');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Hide CUDA explicitly for CPU audit')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    receipt=dict(argv=sys.argv,interpreter=sys.executable,python=sys.version,cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),start_unix=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),status=subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT,text=True),source_sha256={str(x.resolve().relative_to(ROOT)):sha(x) for x in [Path(__file__),Path(__file__).with_name('geometry.json'),ROOT/'migration/wc4_act/fullcycle_compact.py']})
    write(out/'command.json',receipt)
    if a.mode=='reference':reference(out)
    elif a.mode=='synthetic':synthetic(out,a.variant,a.pitch,a.z,a.drift45)
    elif a.mode=='clearance':clearance(out,a.trace)
    else:behavior(out)
    write(out/'completed.json',dict(end_unix=time.time(),physical_episodes=0,training_updates=0,policy_evaluations=0))
if __name__=='__main__':main()
