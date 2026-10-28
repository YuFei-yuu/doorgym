"""Read-only comparison of two independently recorded physical episodes."""
import json,sys,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
root=Path('/data/zyf/video2door-wc4-act');sys.path.insert(0,str(root/'migration/wc4_act'))
from author_fast5.geometry import DoorFrame,closing_reference,transform
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
new_run=Path(sys.argv[1]);extra_run=Path(sys.argv[2]);dest=Path(sys.argv[3]);dest.mkdir(parents=True,exist_ok=False);old_run=root/'.repro/runs/wc4-fast5-repair-20261028'

def stats(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)]
 return None if not len(a) else dict(mean=float(np.mean(a)),p95=float(np.percentile(a,95)),max=float(np.max(a)))

def load(run):
 out=run/'attempt_000';rows=[json.loads(x) for x in (out/'states.jsonl').open()];cs=[json.loads(x) for x in (out/'contacts.jsonl').open()]
 cfg=json.loads((out/'effective-expert.json').read_text());dt=float(json.loads((run/'physics-start.json').read_text())['dt'])
 t=np.array([r['t'] for r in rows]);target=np.array([r['ee_target'] for r in rows]);ee=np.array([r['ee_actual'] for r in rows]);base=np.array([r['actual_base'] for r in rows])
 rv=np.r_[np.nan,np.linalg.norm(np.diff(target,axis=0),axis=1)/dt];ev=np.r_[np.nan,np.linalg.norm(np.diff(ee,axis=0),axis=1)/dt]
 bv=np.r_[np.nan,np.linalg.norm(np.diff(base[:,:2],axis=0),axis=1)/dt];bw=np.r_[np.nan,((np.diff(base[:,2])+math.pi)%(2*math.pi)-math.pi)/dt]
 start=next((r['timing']['command_time_s'] for r in rows if r['phase']=='close'),None)
 mask=np.array([r['phase']=='close' and not r['expert_target']['reference_complete'] for r in rows]);idx=np.flatnonzero(mask)
 allclose=np.flatnonzero([r['phase']=='close' for r in rows])
 pose=np.full(len(rows),np.nan);orient=np.full(len(rows),np.nan)
 for i in allclose:
  pose[i]=np.linalg.norm(target[i]-ee[i]);orient[i]=(Rotation.from_quat(rows[i]['ee_target_quat'])*Rotation.from_quat(rows[i]['ee_quat']).inv()).magnitude()
 end=np.array([r['nonzero_end_panel_contact'] for r in rows]);wrist=[];categories=[];pair_counts={}
 for r,c in zip(rows,cs):
  hit=False;other=False;ehit=False
  for x in c['contacts']:
   if not x.get('lambda',0):continue
   a=next((a for a in x['names'] if a.get('actor')=='door_0'),None);b=next((a for a in x['names'] if a.get('actor','').startswith(('b1_base','z1_arm'))),None)
   if a is None or b is None:continue
   if b['body'] in ['wrist_up_mount','wrist_down_mount']:hit=True
   if b['body'] in ['gripperStator','gripperMover'] and a['body']=='door_panel':ehit=True
   else:other=True
   if r['phase']=='close':pair_counts.setdefault(b['body']+' / '+a['body'],set()).add(r['step'])
  wrist.append(hit);categories.append('both' if ehit and other else 'end_only' if ehit else 'other_only' if other else 'none')
 first=next((r['t'] for r in rows if r['phase'] in ['deploy','approach','close'] and r['nonzero_end_panel_contact']),None)
 locks=[json.loads(x) for x in (out/'lock-writes.jsonl').open()];latch=next((x['time_s'] for x in locks if x['kind']=='relocked'),None)
 finish=next((r['t'] for r in rows if r['expert_target']['reference_complete']),None)
 result=json.loads((out/'result.json').read_text());r11=json.loads((run/'review_final/r11.json').read_text())
 group={}
 for i in allclose:
  g=group.setdefault(categories[i],dict(steps=0,signed_door_drop_deg=0.));g['steps']+=1;g['signed_door_drop_deg']+=math.degrees(rows[i]['pre_physics']['door_q'][0]-rows[i]['door_q'][0])
 summary=dict(reference_duration_s=cfg['close_s'],reference_start_pre_s=start,reference_end_post_s=finish,first_loaded_gripper_contact_post_s=first,latch_pre_s=latch,
  reference_start_to_latch_s=None if start is None or latch is None else latch-start,
  first_loaded_contact_to_latch_s=None if first is None or latch is None else latch-first,
  first_loaded_contact_to_reference_end_s=None if first is None or finish is None else finish-first,
  active_reference_samples=len(idx),active_reference_ee_speed_m_s=stats(rv[idx]),active_actual_ee_speed_m_s=stats(ev[idx]),
  active_actual_base_speed_m_s=stats(bv[idx]),active_actual_abs_yaw_rate_rad_s=stats(abs(bw[idx])),
  active_position_error_m=stats(pose[idx]),active_orientation_error_rad=stats(orient[idx]),
  close_loaded_pair_steps={k:len(v) for k,v in pair_counts.items()},close_contact_motion_groups=group,
  automatic_success=result['success'],r11_measured_screening=r11['measured_task_gates_pass'],sim_seconds=result['sim_seconds'],stop_reason=result['reason'])
 return dict(rows=rows,cfg=cfg,t=t,rv=rv,ev=ev,bv=bv,bw=bw,pose=pose,orient=orient,wrist=np.array(wrist),end=end,start=start,idx=idx,allclose=allclose,summary=summary)

cases=[load(old_run),load(new_run),load(extra_run)];labels=['5 s','15 s initial','15 s base-profile'];colors=['#999999','#c66a31','#167eaa']
frame=DoorFrame(transform([2.61,0,.01],[0,0,math.pi/2]))
fig,axs=plt.subplots(3,2,figsize=(14,11))
for d,label,col in zip(cases,labels,colors):
 if d['start'] is None:continue
 idx=d['allclose'];x=d['t'][idx]-d['start']
 axs[0,0].plot(x,d['rv'][idx],ls='--',color=col,label=label+' reference');axs[0,0].plot(x,d['ev'][idx],color=col,label=label+' measured')
 axs[0,1].plot(x,[d['rows'][i]['action'][0] for i in idx],ls='--',color=col,label=label+' command');axs[0,1].plot(x,d['bv'][idx],color=col,label=label+' measured')
 axs[1,0].plot(x,[d['rows'][i]['action'][1] for i in idx],ls='--',color=col,label=label+' command');axs[1,0].plot(x,d['bw'][idx],color=col,label=label+' measured')
 axs[1,1].plot(x,1000*d['pose'][idx],color=col,label=label)
 axs[2,0].plot(x,np.degrees(d['orient'][idx]),color=col,label=label)
 axs[2,1].plot(x,np.degrees([d['rows'][i]['door_q'][0] for i in idx]),color=col,label=label+' measured')
 elapsed=np.linspace(0,d['cfg']['close_s'],501);angle=[math.degrees(closing_reference(d['cfg'],frame,t)[3]) for t in elapsed]
 axs[2,1].plot(elapsed,angle,ls='--',color=col,label=label+' path angle')
for ax,title,ylabel in zip(axs.flat,['EE speed','Base forward speed','Base yaw rate','EE position error','EE orientation error','Door motion'],['m/s','m/s','rad/s','mm','deg','deg']):
 ax.set_title(title);ax.set_ylabel(ylabel);ax.set_xlabel('Seconds after close reference start');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('5 s / 15 s physical comparison — no smoothing or force threshold');fig.tight_layout();fig.savefig(dest/'5s-15s-speeds-errors.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(len(cases),1,figsize=(13,3*len(cases)),sharex=False)
for ax,d,label,col in zip(axs,cases,labels,colors):
 if d['start'] is None:continue
 i=d['allclose'];x=d['t'][i]-d['start'];ax.step(x,d['end'][i].astype(float),where='post',label='Loaded fixed finger / panel',color='#13764e');ax.step(x,1.4+d['wrist'][i].astype(float),where='post',label='Loaded wrist support / panel',color='#b84935')
 ax.set_yticks([0,1,1.4,2.4]);ax.set_yticklabels(['end off','end on','wrist off','wrist on']);ax.set_title(label+' original contact API');ax.set_xlabel('Seconds after close reference start');ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=8)
fig.tight_layout();fig.savefig(dest/'5s-15s-contact-timing.png',dpi=170);plt.close(fig)
# Same normalized progress: fair duration-independent comparison, no resampling of values.
fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True)
for d,label,col in zip(cases,labels,colors):
 i=d['idx'];x=np.array([d['rows'][k]['expert_target']['reference_elapsed_s']/d['cfg']['close_s'] for k in i]);axs[0].plot(x,1000*d['pose'][i],color=col,label=label);axs[1].plot(x,np.degrees(d['orient'][i]),color=col,label=label)
axs[0].set_ylabel('Position error (mm)');axs[1].set_ylabel('Orientation error (deg)');axs[1].set_xlabel('Normalized reference time (not measured door angle)')
for ax in axs:ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(dest/'5s-15s-normalized-errors.png',dpi=170)
# Match observed reference fractions; don't compare a stopped prefix to a full run.
fractions=[np.array([d['rows'][i]['expert_target']['reference_elapsed_s']/d['cfg']['close_s'] for i in d['idx']]) for d in cases]
shared=min(float(max(x)) for x in fractions if len(x))
for d,f in zip(cases,fractions):
 ids=d['idx'][f<=shared+1e-9]
 d['summary']['shared_reference_fraction']=shared
 d['summary']['matched_progress_position_error_m']=stats(d['pose'][ids])
 d['summary']['matched_progress_orientation_error_rad']=stats(d['orient'][ids])
 d['summary']['matched_progress_actual_ee_speed_m_s']=stats(d['ev'][ids])
 d['summary']['foot_maxima']={name:max(r['foot_guard']['deviations'][name] for r in d['rows']) for name in d['rows'][0]['foot_guard']['deviations']}
 d['summary']['foot_first_bad']=next(({'t':r['t'],'phase':r['phase'],'bad':r['foot_guard']['bad']} for r in d['rows'] if r['foot_guard']['bad']),None)
fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True)
for d,label,col in zip(cases,labels,colors):
 if d['start'] is None:continue
 idx=d['allclose'];x=(d['t'][idx]-d['start'])/d['cfg']['close_s']
 axs[0].plot(x,[r['foot_guard']['deviations']['RR_hip_joint'] for r in np.array(d['rows'],dtype=object)[idx]],label=label,color=col)
 axs[1].plot(x,d['bw'][idx],label=label+' measured',color=col)
 grid=np.linspace(0,d['cfg']['close_s'],601);yaw=np.array([closing_reference(d['cfg'],frame,t)[0][2] for t in grid]);vel=np.gradient(yaw,grid)
 axs[1].plot(grid/d['cfg']['close_s'],vel,ls='--',color=col,label=label+' reference')
axs[0].axhline(.20,color='red',ls=':',label='Existing RR hip stop level')
axs[0].set_ylabel('RR hip deviation (rad)');axs[1].set_ylabel('Base yaw rate (rad/s)');axs[1].set_xlabel('Normalized closing reference time')
for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(dest/'base-profile-feet-comparison.png',dpi=170)
summary=dict(cases={k:v['summary'] for k,v in zip(labels,cases)},scope='Three original physical runs; full observed intervals and matched normalized-reference windows are reported separately; speed/pose statistics cover active close reference only; contact counts include close confirmation holds; foot maxima use the whole observed run. First-loaded contact uses gripperStator/gripperMover to panel. Clock starts use physical pre-step samples; contact timestamps use post-step samples. Grouped door drops are timing diagnostics, not causal work attribution.')
(dest/'5s-15s-comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2))
