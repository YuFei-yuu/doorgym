import json,sys,math,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
run=Path(sys.argv[1]);out=run/'review';out.mkdir(exist_ok=False);a=run/'attempt_000'
result=json.loads((a/'result.json').read_text());rows=[];events=[];previous=None;firstcontact=None;firstpush=None;lasttheta=None;count=0;closes=[]
for line in (a/'states.jsonl').open():
    r=json.loads(line);count+=1;p=r['phase'];t=r['t'];theta=float(r['door_q'][0])
    if p!=previous:events.append(dict(event='phase',phase=p,step=r['step'],time_s=t));previous=p
    if p=='close' and r['nonzero_end_panel_contact']:
        if firstcontact is None:firstcontact=dict(event='first_loaded_closing_end_panel_contact',step=r['step'],time_s=t,door_deg=math.degrees(theta))
        if lasttheta is not None and theta<lasttheta and firstpush is None:firstpush=dict(event='first_contact_aligned_decrease',step=r['step'],time_s=t,door_deg=math.degrees(theta))
    lasttheta=theta
    rows.append({k:r[k] for k in ['step','t','phase','actual_base','action','arm_q','arm_qd','door_q','door_qd','whole_geometry_max_x','nonzero_end_panel_contact','latched','unlocked','compact_expert','task_monitor'] if k in r})
for e in [firstcontact,firstpush]:
    if e:events.append(e)
events.append(dict(event='stop',time_s=result['sim_seconds'],reason=result['reason']))
for key,phases in [('traversal_endpoint',['endpoint_hold']),('backside_contact',['close']),('latch',[]),('release',['release']),('home',['restore_heading','final_hold'])]:
    found=next((r for r in rows if (r['latched'] is not None if key=='latch' else r['phase'] in phases)),None)
    if key=='backside_contact':found=firstcontact
    events.append(dict(milestone=key,reached=found is not None,first=found))
# Keep compact events free of large repeated bodies.
(out/'event-index.json').write_text(json.dumps(events,indent=2)+'\n')
clock=[json.loads(l) for l in (a/'recording-clock.jsonl').open()];expected=list(range(0,result['steps'],2))
contact_count=sum(1 for _ in (a/'contacts.jsonl').open());T=np.array([r['t'] for r in rows]);B=np.array([r['actual_base'] for r in rows]);Q=np.array([r['arm_q'] for r in rows]);U=np.array([r['action'] for r in rows]);angles=np.rad2deg([r['door_q'][0] for r in rows])
idx=next((i for i,r in enumerate(rows) if r['phase']=='retimed_traverse'),len(rows)-1)
metrics=dict(result=result,state_rows=count,contact_rows=contact_count,clock_frames=len(clock),clock_contiguous=[r['source_step'] for r in clock]==expected,
    state_contiguous=[r['step'] for r in rows]==list(range(result['steps'])),post_pass_translation_m=float(np.linalg.norm(np.diff(B[idx:,:2],axis=0),axis=1).sum()),
    endpoint_error_m=float(np.linalg.norm(B[-1,:2]-[1.2599997520446777,.18000000715255737])),maximum_measured_joint_speed=float(np.max(abs(np.array([r['arm_qd'] for r in rows])))),
    last_phase=rows[-1]['phase'],last_door_deg=float(angles[-1]),last_task_monitor=rows[-1]['task_monitor'],first_closing_contact=firstcontact,first_closing_motion=firstpush,
    interpretation='Actual physics only. Phase entry is not completion; contact does not alone prove backside location or effective pushing.')
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
axes[0].plot(T,angles);axes[0].set_ylabel('Actual door (deg)')
axes[1].plot(T,B[:,0],label='x');axes[1].plot(T,B[:,1],label='y');axes[1].axhline(1.259999752,ls='--',color='gray');axes[1].set_ylabel('Base (m)');axes[1].legend()
axes[2].plot(T,U[:,:2]);axes[2].set_ylabel('Joint9 v / w')
axes[3].plot(T,Q[:,:6]);axes[3].set_ylabel('Measured joints (rad)');axes[3].set_xlabel('Actual simulation time (s)')
for ax in axes:
    ax.grid(True,alpha=.3)
    for ev in events:
        if ev.get('event')=='phase':ax.axvline(ev['time_s'],color='gray',alpha=.15)
fig.tight_layout();fig.savefig(out/'measured-motion.png',dpi=160);plt.close(fig)
(out/'compact-actual-trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print(json.dumps(metrics),flush=True)
