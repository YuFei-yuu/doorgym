"""Read-only timing, contact timing and foot comparison against accepted run."""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
root=Path('/data/zyf/video2door-wc4-act');sys.path.insert(0,str(root/'migration/wc4_act'))
from fullcycle_geometry import URDF,transform
from author_fast5.geometry import progress,smooth
run=Path(sys.argv[1]);review=run/'review_final';baseline=root/'.repro/runs/wc4-contact-inset-20261028'
read=lambda p:[json.loads(l) for l in p.open()]
old=read(baseline/'attempt_000/states.jsonl');rows=read(run/'attempt_000/states.jsonl')
def summary(rr):
 phases={}
 for phase in dict.fromkeys(r['phase'] for r in rr):
  a=[r for r in rr if r['phase']==phase]
  phases[phase]=dict(start_pre_s=a[0]['timing']['command_time_s'],end_post_s=a[-1]['t'],duration_s=a[-1]['t']-a[0]['timing']['command_time_s'],first_door_deg=math.degrees(a[0]['pre_physics']['door_q'][0]),last_door_deg=math.degrees(a[-1]['door_q'][0]),peak_actual_yaw_rate_rad_s=max(abs(r['base_measured_fd_yaw_rate']) for r in a))
 traverse=[r for r in rr if r['phase']=='traverse_door'];start=traverse[0]['timing']['command_time_s'];fold=next(r for r in traverse if r['expert_target'].get('mode')=='joint')
 fixed_geometry=[]
 for x in [1.8,1.6,1.4,1.26]:
  r=min(traverse,key=lambda r:abs(r['actual_base'][0]-x));fixed_geometry.append(dict(base_x=r['actual_base'][0],time_s=r['t'],door_deg=math.degrees(r['door_q'][0])))
 close=[r for r in rr if r['phase']=='close'];cfg=json.loads((run/'attempt_000/effective-expert.json').read_text())
 first=next(r for r in close if r['nonzero_end_panel_contact']);elapsed=first['expert_target']['reference_elapsed_s'];angle=90*(1-progress(elapsed,cfg['close_s'],cfg['progress_ramp_s']))
 return dict(phases=phases,fold_start_pre_s=fold['timing']['command_time_s'],withdrawal_real_s=fold['timing']['command_time_s']-start,
  door_at_matched_base_x=fixed_geometry,first_close_contact=dict(time_s=first['t'],reference_elapsed_s=elapsed,reference_angle_deg=angle,inset_m=cfg['contact_inset_m']*smooth((cfg['contact_inset_start_deg']-angle)/(cfg['contact_inset_start_deg']-cfg['contact_inset_end_deg'])),actual_angle_deg=math.degrees(first['door_q'][0])))
initial=read(root/'.repro/runs/wc4-traverse-hold-slow-turn-trial-20261028/attempt_000/states.jsonl');retry=read(root/'.repro/runs/wc4-traverse-hold-slow-turn-retry-20261028/attempt_000/states.jsonl')
prefix={}
for name,rr in [('initial_20s',initial),('debounce_15s',retry)]:
 prefix[name]={k:float(np.max(abs(np.asarray([r[k] for r in rows[:len(rr)]])-np.asarray([r[k] for r in rr])))) for k in ['actual_base','arm_q','ee_actual','door_q']}
result=dict(accepted=summary(old),tuned=summary(rows),engineering_retries_prefix_difference=prefix,
 limitation='Real observed intervals only. Same-position door samples separate changed speed from elapsed-time comparison. Contact timing association does not prove a single cause of wrist contact or leg deviation.')
(review/'timing-physical-audit.json').write_text(json.dumps(result,indent=2)+'\n')
robot=URDF(root/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf')
for name in ['wc4-traverse-hold-slow-turn-trial-20261028','wc4-traverse-hold-slow-turn-retry-20261028','wc4-traverse-hold-slow-turn-recording-20261028']:
 run1=root/'.repro/runs'/name;out=run1/'attempt_000';rev=run1/'review_final';rr=read(out/'states.jsonl')
 effective=json.loads((out/'effective-config.json').read_text());ref=json.loads((out/'reference.json').read_text());err=0.;count=0
 for r in rr:
  if r['expert_target'].get('mode')=='joint':
   B=r['command_base'];F=robot.fk(dict(zip(ref['dof_names'],r['arm_command'])),transform([*B[:2],effective['robot_z']],[0,effective['robot_pitch'],B[2]]))['ee_gripper_link'];err=max(err,float(np.max(abs(F[:3,3]-r['ee_target']))));count+=1
 assert err<1e-9
 (rev/'reference-target-audit.json').write_text(json.dumps(dict(raw_log_unchanged=True,explicit_target_mode=True,joint_mode_records=count,logged_fk_error_m=err),indent=2)+'\n')
 v=json.loads((rev/'video-check.json').read_text());v['sha256']=hashlib.sha256((out/'overview/env_0000/rollout.mp4').read_bytes()).hexdigest();(rev/'video-check.json').write_text(json.dumps(v,indent=2)+'\n')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(3,1,figsize=(12,9))
for label,rr in [('Accepted',old),('Tuned',rows)]:
 tr=[r for r in rr if r['phase'] in ['traverse_door','traverse_fold_finish']];t=np.array([r['t'] for r in tr]);x=np.array([r['actual_base'][0] for r in tr]);d=np.degrees([r['door_q'][0] for r in tr]);axs[0].plot(t,d,label=label);axs[1].plot(x,d,label=label)
 cl=[r for r in rr if r['phase']=='close'];t=np.array([r['expert_target']['reference_elapsed_s'] for r in cl]);axs[2].plot(t,[r['foot_guard']['deviations']['RR_hip_joint'] for r in cl],label=label)
axs[0].set(xlabel='Physical time (s)',ylabel='Door during traversal (deg)');axs[1].set(xlabel='Measured base x (m)',ylabel='Door during traversal (deg)');axs[1].invert_xaxis();axs[2].set(xlabel='Closing reference elapsed (s)',ylabel='RR hip deviation (rad)');axs[2].axhline(.2,color='red',ls=':',label='Existing stop threshold')
for ax in axs:ax.grid(alpha=.2);ax.legend()
fig.suptitle('Measured door opening retention and RR hip response; no extrapolation');fig.tight_layout();fig.savefig(review/'timing-door-feet-comparison.png',dpi=160)
print(json.dumps(result,indent=2))
