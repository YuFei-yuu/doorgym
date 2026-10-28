"""Read-only three-case review for the single-variable delay experiment."""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
root=Path('/data/zyf/video2door-wc4-act')
# Reuse the frozen comparison loader; execute only definitions before its CLI.
source=root/'migration/wc4_act/author_fast5/compare_inset.py'
exec(compile(source.read_text().split('\nnewrun=')[0],str(source),'exec'))
from author_fast5.geometry import progress,smooth
from fullcycle_geometry import URDF,transform
run=Path(sys.argv[1]);dest=run/'review_final'
paths=[root/'.repro/runs/wc4-contact-inset-20261028',root/'.repro/runs/wc4-traverse-hold-slow-turn-recording-20261028',run]
labels=['Accepted','Failed slow turns','Delay only'];colors=['#777777','#b26736','#087e9a']
cases=[load(p) for p in paths]
read=lambda p:[json.loads(l) for l in p.open()]
result={}
for label,path,d in zip(labels,paths,cases):
 rr=d['rows'];contacts=read(path/'attempt_000/contacts.jsonl');cfg=d['cfg']
 tr=[r for r in rr if r['phase']=='traverse_door'];first=tr[0]['timing']['command_time_s']
 fold=next((r for r in tr if r['expert_target'].get('mode')=='joint'),None)
 geometry=[]
 for x in [1.8,1.6,1.4,1.26]:
  r=min(tr,key=lambda r:abs(r['actual_base'][0]-x));geometry.append(dict(base_x=r['actual_base'][0],t=r['t'],door_deg=math.degrees(r['door_q'][0])))
 phase_metrics={}
 for ph in ['turn_for_close','close','restore_heading','final_hold']:
  a=[r for r in rr if r['phase']==ph]
  phase_metrics[ph]=None if not a else dict(start_pre_s=a[0]['timing']['command_time_s'],end_post_s=a[-1]['t'],duration_s=a[-1]['t']-a[0]['timing']['command_time_s'],peak_actual_yaw_rate_rad_s=max(abs(r['base_measured_fd_yaw_rate']) for r in a),
   RR_peak_rad={j:max(r['foot_guard']['deviations'][j] for r in a) for j in ['RR_hip_joint','RR_thigh_joint','RR_calf_joint']},RR_final_rad={j:a[-1]['foot_guard']['deviations'][j] for j in ['RR_hip_joint','RR_thigh_joint','RR_calf_joint']})
 opening=[r for r in rr if r['phase'] in ['push_door','traverse_door','traverse_fold_finish','turn_for_close']]
 last_loaded=next((r for r in reversed(opening) if r['nonzero_door_contact']),None)
 clear=next((r for r in rr if last_loaded is not None and r['step']>last_loaded['step'] and not r['nonzero_door_contact']),None)
 rawclear=next((r for r in tr if not r['any_door_contact']),None)
 firstnew=next((r for r in rr if r['phase'] in ['deploy','approach','close'] and r['nonzero_door_contact']),None)
 finger=next((r for r in rr if r['phase'] in ['deploy','approach','close'] and r['nonzero_end_panel_contact']),None)
 firstinfo=None
 if finger:
  elapsed=finger['expert_target'].get('reference_elapsed_s');angle=None if elapsed is None else 90*(1-progress(elapsed,cfg['close_s'],cfg['progress_ramp_s']))
  firstinfo=dict(t=finger['t'],phase=finger['phase'],actual_angle_deg=math.degrees(finger['door_q'][0]),reference_elapsed_s=elapsed,reference_angle_deg=angle,inset_m=None if angle is None else cfg['contact_inset_m']*smooth((cfg['contact_inset_start_deg']-angle)/(cfg['contact_inset_start_deg']-cfg['contact_inset_end_deg'])))
 pairs={}
 for r,c in zip(rr,contacts):
  if r['phase'] not in ['traverse_door','traverse_fold_finish','turn_for_close','deploy','approach','close','release','fold','restore_heading','final_hold']:continue
  for x in c['contacts']:
   if not x.get('lambda',0):continue
   door=next((z for z in x['names'] if z.get('actor')=='door_0'),None);robot=next((z for z in x['names'] if z.get('actor','').startswith(('b1_base','z1_arm'))),None)
   if not door or not robot:continue
   key=r['phase']+': '+robot['body']+' / '+door['body'];v=pairs.setdefault(key,dict(steps=set(),first_s=r['t'],last_s=r['t']));v['steps'].add(r['step']);v['last_s']=r['t']
 for v in pairs.values():v['step_count']=len(v['steps']);v['steps']=sorted(v['steps'])
 result[label]=dict(closing=d['summary'],door_at_same_position=geometry,fold_start_pre_s=None if fold is None else fold['timing']['command_time_s'],withdrawal_duration_s=None if fold is None else fold['timing']['command_time_s']-first,
  opening_contact_separation=dict(last_loaded_s=None if last_loaded is None else last_loaded['t'],first_unloaded_after_last_s=None if clear is None else clear['t'],first_no_raw_contact_during_traverse_s=None if rawclear is None else rawclear['t'],first_postdeployment_loaded_any_s=None if firstnew is None else firstnew['t'],meaning='Loaded means original lambda !=0. Single early clear samples can be followed by recontact; last-loaded/next-unloaded identify the final opening-contact separation. Raw contacts may have zero load.'),
  first_closing_finger_contact=firstinfo,phase_metrics=phase_metrics,loaded_pairs=pairs,
  full_episode_foot_max={j:max(r['foot_guard']['deviations'][j] for r in rr) for j in rr[0]['foot_guard']['deviations']},last_door_deg=math.degrees(rr[-1]['door_q'][0]),last_phase=rr[-1]['phase'])
shared=min(max(d['rows'][i]['expert_target']['reference_elapsed_s']/d['cfg']['close_s'] for i in d['idx']) for d in cases if len(d['idx']))
for label,d in zip(labels,cases):
 i=np.array([i for i in d['idx'] if d['rows'][i]['expert_target']['reference_elapsed_s']/d['cfg']['close_s']<=shared+1e-9],int)
 result[label]['matched_closing_progress']=dict(fraction=shared,position_error_m=stats(d['pose'][i]),orientation_error_rad=stats(d['orient'][i]))
prefix={}
for label,other,limit in [('accepted_opening',cases[0]['rows'],875),('failed_tune_until_preparation',cases[1]['rows'],1199)]:
 prefix[label]={k:float(np.max(abs(np.asarray([r[k] for r in cases[2]['rows'][:limit]])-np.asarray([r[k] for r in other[:limit]])))) for k in ['actual_base','arm_q','ee_actual','door_q']}
report=dict(cases=result,prefix_comparison=prefix,scope='Three original measured episodes; no extrapolation, no new contact force/coverage threshold. Phase foot peaks and matched closing progress separate different durations.')
(dest/'three-way-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axs=plt.subplots(2,1,figsize=(11,7))
for label,d,c in zip(labels,cases,colors):
 tr=[r for r in d['rows'] if r['phase']=='traverse_door'];axs[0].plot([r['actual_base'][0] for r in tr],np.degrees([r['door_q'][0] for r in tr]),label=label,color=c)
 i=d['allclose'];axs[1].plot(d['t'][i]-d['start'],np.degrees([d['rows'][k]['door_q'][0] for k in i]),label=label,color=c)
axs[0].invert_xaxis();axs[0].set(xlabel='Measured traversal base x (m)',ylabel='Door angle (deg)');axs[1].set(xlabel='Closing reference elapsed (s)',ylabel='Measured door angle (deg)')
for ax in axs:ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(dest/'three-way-door.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(15,4))
for ax,phase in zip(axs,['turn_for_close','close','restore_heading']):
 for label,d,c in zip(labels,cases,colors):
  a=[r for r in d['rows'] if r['phase']==phase]
  if a:ax.plot([r['t']-a[0]['timing']['command_time_s'] for r in a],[r['foot_guard']['deviations']['RR_hip_joint'] for r in a],label=label,color=c)
 ax.axhline(.2,color='red',ls=':');ax.set(title=phase,xlabel='Phase elapsed (s)',ylabel='RR hip deviation (rad)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Observed RR hip behavior; absent phases are not imputed');fig.tight_layout();fig.savefig(dest/'three-way-feet.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
for label,d,c in zip(labels,cases,colors):
 i=d['allclose'];t=d['t'][i]-d['start'];axs[0].plot(t,d['ev'][i],label=label,color=c);axs[0].plot(t,d['rv'][i],ls='--',color=c,alpha=.5);axs[1].plot(t,d['pose'][i]*1000,label=label,color=c);axs[2].step(t,d['wrist'][i],where='post',label=label,color=c)
for ax,y in zip(axs,['EE m/s (dashed: reference)','EE error (mm)','Loaded wrist present']):ax.set_ylabel(y);ax.grid(alpha=.2);ax.legend()
axs[-1].set_xlabel('Closing reference elapsed (s)');fig.tight_layout();fig.savefig(dest/'three-way-closing.png',dpi=160);plt.close(fig)
# Verify current joint-reference records, preserve all original logs.
out=run/'attempt_000';robot=URDF(root/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');eff=json.loads((out/'effective-config.json').read_text());ref=json.loads((out/'reference.json').read_text());err=0.;count=0
for r in cases[2]['rows']:
 if r['expert_target'].get('mode')=='joint':
  B=r['command_base'];F=robot.fk(dict(zip(ref['dof_names'],r['arm_command'])),transform([*B[:2],eff['robot_z']],[0,eff['robot_pitch'],B[2]]))['ee_gripper_link'];err=max(err,float(np.max(abs(F[:3,3]-r['ee_target']))));count+=1
assert err<1e-9
(dest/'reference-target-audit.json').write_text(json.dumps(dict(raw_log_unchanged=True,explicit_target_mode=True,joint_mode_records=count,logged_fk_error_m=err),indent=2)+'\n')
v=json.loads((dest/'video-check.json').read_text());v['sha256']=hashlib.sha256((out/'overview/env_0000/rollout.mp4').read_bytes()).hexdigest();(dest/'video-check.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(report,indent=2))
