"""Read actual physics records; never simulate/replay states."""
import json,sys,math,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
run=Path(sys.argv[1]);a=run/'attempt_000';out=run/'review';out.mkdir(exist_ok=False)
result=json.loads((a/'result.json').read_text());rows=[];events=[];previous=None;firstcontact=None;handoff=None;endpoint=None;follow=None;latch=None;zero_hinge=True;zlo=float('inf');zhi=-float('inf');loaded=collections.Counter();reference_end=None;stopped_end=None;contact_previous=False;contact_changes=[]
for line in (a/'states.jsonl').open():
    r=json.loads(line);p=r['phase'];d=r.get('compact_expert') or {};t=r['t']
    if p!=previous:events.append(dict(kind='phase',phase=p,step=r['step'],post_time_s=t));previous=p
    if p=='endpoint_hold' and endpoint is None:endpoint=dict(step=r['step'],time_s=t,base=r['actual_base'])
    if p in ['deploy_a','deploy_b','approach','close'] and r['nonzero_end_panel_contact']:
        loaded[p]+=1
        if firstcontact is None:firstcontact=dict(kind='first_postdeploy_loaded_contact',step=r['step'],post_time_s=t,phase=p,door_deg=math.degrees(r['door_q'][0]));events.append(firstcontact)
    if p in ['approach','close','release','retract','restore_heading','final_hold']:
        now=bool(r['nonzero_end_panel_contact'])
        if now!=contact_previous:
            ce=dict(kind='contact_on' if now else 'contact_off',step=r['step'],post_time_s=t,phase=p,door_deg=math.degrees(r['door_q'][0]));contact_changes.append(ce);events.append(ce)
        contact_previous=now
    if d.get('reference_complete') and reference_end is None:
        reference_end=dict(kind='reference_complete',step=r['step'],post_time_s=t,reference=d['reference'],door_deg=math.degrees(r['door_q'][0]),position_error_m=d.get('measured_position_error'),base=r['actual_base']);events.append(reference_end)
    if d.get('endpoint_stopped') and stopped_end is None:
        stopped_end=dict(kind='endpoint_stopped',step=r['step'],post_time_s=t,position_error_m=d.get('measured_position_error'));events.append(stopped_end)
    if d.get('dynamic_handoff') and handoff is None:
        handoff=dict(kind='dynamic_switch',step=r['step'],post_time_s=t,confirmation=d['dynamic_confirmation'],handoff=d['dynamic_handoff']);events.append(handoff)
    if endpoint and p=='close' and np.linalg.norm(np.array(r['actual_base'][:2])-np.array(endpoint['base'][:2]))>1e-5 and follow is None:
        follow=dict(kind='base_follow_start',step=r['step'],post_time_s=t,door_deg=math.degrees(r['door_q'][0]),base=r['actual_base']);events.append(follow)
    if r['latched'] is not None and latch is None:latch=dict(kind='latch',step=r['step'],post_time_s=t,latched_rad=r['latched']);events.append(latch)
    if r['unlocked'] and r['latched'] is None:zero_hinge &= float(r['door_effort'][0])==0
    z=float(r['root_rb']['pose']['p']['z']);zlo=min(zlo,z);zhi=max(zhi,z)
    rows.append({k:r[k] for k in ['step','t','phase','actual_base','action','arm_q','arm_qd','door_q','door_qd','whole_geometry_max_x','nonzero_end_panel_contact','latched','unlocked','compact_expert','task_monitor'] if k in r})
events.append(dict(kind='stop',post_time_s=result['sim_seconds'],reason=result['reason']))
T=np.array([r['t'] for r in rows]);B=np.array([r['actual_base'] for r in rows]);U=np.array([r['action'] for r in rows]);theta=np.rad2deg([r['door_q'][0] for r in rows]);idx=endpoint['step'] if endpoint else len(rows)-1
clock=[json.loads(l) for l in (a/'recording-clock.jsonl').open()]
report=dict(reference_completion=reference_end,endpoint_stopped=stopped_end,contact_transitions=contact_changes,result=result,first_loaded_contact=firstcontact,dynamic_handoff=handoff,base_follow=follow,latch=latch,endpoint=endpoint,
    final_phase=rows[-1]['phase'],final_door_deg=float(theta[-1]),actual_root_z_range=[zlo,zhi],unlocked_hinge_effort_all_zero=bool(zero_hinge),
    post_endpoint_translation_m=float(np.linalg.norm(np.diff(B[idx:,:2],axis=0),axis=1).sum()),max_endpoint_displacement_m=float(np.linalg.norm(B[idx:,:2]-B[idx,:2],axis=1).max()),
    actual_contact_steps_by_phase=dict(loaded),rows=len(rows),contact_rows=sum(1 for l in (a/'contacts.jsonl').open()),clock_frames=len(clock),
    state_contiguous=[r['step'] for r in rows]==list(range(result['steps'])),clock_contiguous=[r['source_step'] for r in clock]==list(range(0,result['steps'],2)))
if handoff:
    k=handoff['step'];dt=float(T[1]-T[0]);before=(U[k,2:]-U[k-1,2:])/dt;after=(U[k+1,2:]-U[k,2:])/dt
    report['actual_command_boundary']=dict(max_joint_increment=float(np.max(abs(U[k+1,2:]-U[k,2:]))),max_joint_velocity_change=float(np.max(abs(after-before))),before_velocity=before.tolist(),after_velocity=after.tolist())
(out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');(out/'event-index.json').write_text(json.dumps(events,indent=2)+'\n');(out/'compact-actual-trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
fig,ax=plt.subplots(4,1,figsize=(12,11),sharex=True)
ax[0].plot(T,theta,label='actual door');ax[0].plot(T,[math.degrees((r.get('compact_expert') or {}).get('reference',{}).get('path_angle_rad',float('nan'))) for r in rows],label='time reference');ax[0].legend();ax[0].set_ylabel('Actual door (deg)');ax[1].plot(T,B[:,0],label='x');ax[1].plot(T,B[:,1],label='y');ax[1].set_ylabel('Actual base (m)');ax[1].legend();ax[2].plot(T,np.unwrap(B[:,2]));ax[2].set_ylabel('Actual yaw (rad)');ax[3].plot(T,U[:,:2]);ax[3].set_ylabel('Joint9 v / w');ax[3].set_xlabel('Simulation time (s)')
for axis in ax:
    axis.grid(alpha=.3)
    for e in events:
        if e['kind'] in ['dynamic_switch','base_follow_start','reference_complete','latch']:axis.axvline(e['post_time_s'],ls='--',color='red',alpha=.5)
fig.tight_layout();fig.savefig(out/'measured-motion.png',dpi=160);plt.close(fig)
if handoff:
    chosen=[r for r in rows if handoff['post_time_s']-1<=r['t']<=handoff['post_time_s']+3];fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
    tt=[r['t'] for r in chosen];ax[0].plot(tt,np.array([r['action'][2:8] for r in chosen]));ax[0].set_ylabel('Issued joints (rad)')
    ax[1].plot(tt,[r['compact_expert'].get('measured_position_error',float('nan')) for r in chosen]);ax[1].axhline(.008,color='gray',ls='--');ax[1].set_ylabel('Actual pose error (m)')
    ax[2].plot(tt,[math.degrees(r['door_q'][0]) for r in chosen]);ax[2].set_ylabel('Actual door (deg)');ax[2].set_xlabel('Simulation time (s)')
    for axis in ax:axis.axvline(handoff['post_time_s'],ls='--',color='red');axis.grid(alpha=.3)
    fig.tight_layout();fig.savefig(out/'dynamic-handoff.png',dpi=160)
print(json.dumps({k:v for k,v in report.items() if k not in ['result','dynamic_handoff']}),flush=True)
