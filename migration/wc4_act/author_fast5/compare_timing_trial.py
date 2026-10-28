import json,sys,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
root=Path('/data/zyf/video2door-wc4-act');sys.path.insert(0,str(root/'migration/wc4_act'))
from author_fast5.geometry import DoorFrame,closing_reference,transform
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


newrun=Path(sys.argv[1]);out=newrun/'review_final';oldrun=root/'.repro/runs/wc4-contact-inset-20261028'
cases=[load(oldrun),load(newrun)];labels=['Accepted','Tuned'];colors=['#999999','#167eaa']
fig,axs=plt.subplots(3,2,figsize=(14,10))
for d,label,col in zip(cases,labels,colors):
 i=d['allclose'];x=d['t'][i]-d['start'];axs[0,0].plot(x,d['rv'][i],ls='--',color=col,label=label+' reference');axs[0,0].plot(x,d['ev'][i],color=col,label=label+' actual')
 axs[0,1].plot(x,d['bv'][i],color=col,label=label);axs[1,0].plot(x,d['bw'][i],color=col,label=label)
 axs[1,1].plot(x,1000*d['pose'][i],color=col,label=label);axs[2,0].plot(x,np.degrees(d['orient'][i]),color=col,label=label)
 axs[2,1].plot(x,np.degrees([d['rows'][k]['door_q'][0] for k in i]),color=col,label=label)
for ax,title,unit in zip(axs.flat,['EE speed','Actual base speed','Actual base yaw rate','EE position error','EE orientation error','Measured door angle'],['m/s','m/s','rad/s','mm','deg','deg']):
 ax.set_title(title);ax.set_ylabel(unit);ax.set_xlabel('Seconds after closing reference start');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Same closing path; slower traversal and turns — original measurements');fig.tight_layout();fig.savefig(out/'timing-tracking.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(13,6))
for d,label,col,ax in zip(cases,labels,colors,axs):
 i=d['allclose'];x=d['t'][i]-d['start'];ax.step(x,d['end'][i],where='post',label='Loaded fixed finger / panel',color='#167443');ax.step(x,1.4+d['wrist'][i],where='post',label='Loaded wrist / panel',color='#ba5035');ax.set_yticks([0,1,1.4,2.4]);ax.set_yticklabels(['end off','end on','wrist off','wrist on']);ax.set_title(label);ax.set_xlabel('Seconds after closing reference start');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(out/'timing-contact-timing.png',dpi=160);plt.close(fig)
shared=min(max(d['rows'][i]['expert_target']['reference_elapsed_s']/d['cfg']['close_s'] for i in d['idx']) for d in cases)
for d in cases:
 i=np.array([i for i in d['idx'] if d['rows'][i]['expert_target']['reference_elapsed_s']/d['cfg']['close_s']<=shared+1e-9],int)
 d['summary']['shared_fraction']=shared;d['summary']['matched_position_error_m']=stats(d['pose'][i]);d['summary']['matched_orientation_error_rad']=stats(d['orient'][i]);d['summary']['foot_maxima']={k:max(r['foot_guard']['deviations'][k] for r in d['rows']) for k in d['rows'][0]['foot_guard']['deviations']}
 d['summary']['last_door_deg']=math.degrees(d['rows'][-1]['door_q'][0]);d['summary']['last_phase']=d['rows'][-1]['phase']
# Actual selected fixed-finger corner (not the solver's exact contact point).
frame=DoorFrame(transform([2.61,0,.01],[0,0,math.pi/2]));pad=np.array([.015,.03,-.02])
fig,ax=plt.subplots(figsize=(12,4.5))
for d,label,col in zip(cases,labels,colors):
 i=d['allclose'];x=d['t'][i]-d['start'];rad=[];targetrad=[]
 for k in i:
  r=d['rows'][k];theta=r['door_q'][0];tangent=frame.R@np.array([-math.sin(theta),math.cos(theta),0.]);point=np.array(r['ee_actual'])+Rotation.from_quat(r['ee_quat']).as_matrix()@pad;rad.append(float((point-frame.hinge)@tangent))
  theta_ref=closing_reference(d['cfg'],frame,r['expert_target']['reference_elapsed_s'])[3];p=frame.contact(d['cfg'],theta_ref)[2];targetrad.append(float(np.linalg.norm((p-frame.hinge)[:2])))
 ax.plot(x,rad,color=col,label=label+' actual corner projected onto measured door');ax.plot(x,targetrad,ls='--',color=col,label=label+' reference radius')
 d['summary']['actual_corner_radial_projection_m']=stats(rad)
ax.axhline(frame.slab_hi[0],color='#b17a44',ls=':',label='Slab free-edge radial bound');ax.set_xlabel('Seconds after closing reference start');ax.set_ylabel('Distance along door from hinge (m)');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'timing-corner-projection.png',dpi=160)
result={'cases':{k:d['summary'] for k,d in zip(labels,cases)},'scope':'Raw physical episodes. Speed/pose summaries are active-reference intervals; contact counts include confirmation hold. Also report matched observed fractions. Selected-corner projections are geometry diagnostics, not exact solver contact locations or a force/coverage gate.'}
(out/'timing-comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
