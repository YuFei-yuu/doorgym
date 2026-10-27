"""Read-only actual-state, actual-DOF, rigid-body and live-video observation."""
import sys,json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from fullcycle_geometry import URDF
run=Path(sys.argv[1]).resolve();out=run/'observation';out.mkdir(exist_ok=False)
robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');report={};allrows={}
sources={'stage8':ROOT/'.repro/runs/wc4-close-inplace-stage8-20261027/attempt_000',**{'random_%02d'%i:run/('trial_%02d'%i)/'attempt_000' for i in range(2)}}
for label,p in sources.items():
 mapping=json.loads((p/'body-map.json').read_text());names=[v['body'] for k,v in sorted(mapping.items(),key=lambda kv:int(kv[0])) if v['actor_handle']==0];joints=[j for j in robot.joints if j[0].startswith(('FL_','FR_','RL_','RR_')) and j[1]=='revolute'];rows=[];phases=[];previous=None;dof_errors=[];events=[];firstcontact=None;latch=None;contactprev=False
 for l in (p/'states.jsonl').open():
  r=json.loads(l);states=dict(zip(names,r['base_body_states']));R={n:Rotation.from_quat(x[0][1]).as_matrix() for n,x in states.items()};q={};qd={}
  for n,kind,pa,ch,O,axis in joints:
   M=O[:3,:3].T@R[pa].T@R[ch];a=np.dot(axis,[M[2,1]-M[1,2],M[0,2]-M[2,0],M[1,0]-M[0,1]])/2;q[n]=float(np.arctan2(a,(np.trace(M)-1)/2));axisworld=R[pa]@O[:3,:3]@axis;qd[n]=float(axisworld@(np.array(states[ch][1][1])-np.array(states[pa][1][1])))
  actual={n:v for n,v in zip(r.get('base_dof_names',[]),r.get('base_dof_actual',[]))}
  for n,v in actual.items():
   if n in q and '_wheel_' not in n:dof_errors.append(abs((float(v[0])-q[n]+np.pi)%(2*np.pi)-np.pi))
  if previous!=r['phase']:events.append(dict(kind='phase',t=r['t'],phase=r['phase']));previous=r['phase']
  contact=r['nonzero_end_panel_contact'] and r['phase'] in ['approach','close','release']
  if contact and firstcontact is None:firstcontact=r['t'];events.append(dict(kind='first_back_contact',t=r['t'],phase=r['phase']))
  if contact!=contactprev:events.append(dict(kind='contact_on' if contact else 'contact_off',t=r['t'],phase=r['phase']));contactprev=contact
  if r['latched'] is not None and latch is None:latch=r['t'];events.append(dict(kind='latch',t=latch,angle=r['latched']))
  diag=r.get('compact_expert') or {}
  rows.append(dict(step=r['step'],t=r['t'],phase=r['phase'],actual_base=r['actual_base'],action=r['action'][:2],q=q,qd_from_bodies=qd,actual_dof=actual,door_rad=r['door_q'][0],door_velocity=r['door_qd'][0],end_contact=bool(contact),arm_qd_max=float(max(abs(x) for x in r['arm_qd'][:6])),tracking_error=diag.get('measured_position_error'),early_arm=r.get('early_arm'),root_z=r['root_rb']['pose']['p']['z']))
 T=np.array([r['t'] for r in rows]);yaw=np.unwrap([r['actual_base'][2] for r in rows]);w=np.array([r['action'][1] for r in rows]);v=np.array([r['action'][0] for r in rows]);feet={}
 for side in ['FL','FR','RL','RR']:
  feet[side]={}
  for part,home in [('hip',0),('thigh',.8),('calf',-1.5)]:
   n=side+'_'+part+'_joint';k=max(range(len(rows)),key=lambda i:abs(rows[i]['q'][n]-home));first=next((r for r in rows if abs(r['q'][n]-home)>(.2 if part=='hip' else .3)),None)
   feet[side][part]=dict(max_deviation_rad=abs(rows[k]['q'][n]-home),max_time_s=rows[k]['t'],phase=rows[k]['phase'],first_threshold=None if first is None else dict(t=first['t'],phase=first['phase'],value=first['q'][n]))
 for phase in dict.fromkeys(r['phase'] for r in rows):
  ix=np.array([i for i,r in enumerate(rows) if r['phase']==phase]);rr=[rows[i] for i in ix]
  phases.append(dict(phase=phase,start_s=rr[0]['t'],end_s=rr[-1]['t'],actual_yaw_change_rad=float(yaw[ix[-1]]-yaw[ix[0]]),command_yaw_rate_max=float(np.max(abs(w[ix]))),actual_yaw_rate_max=float(np.max(abs(np.diff(yaw[ix])/np.diff(T[ix])))) if len(ix)>1 else 0.,translation_m=float(np.sum(abs(v[ix]))*.019999999552965164),hips={side:float(max(abs(r['q'][side+'_hip_joint']) for r in rr)) for side in feet}))
 ground={side:dict(loaded_steps=0,max_lambda=0.) for side in feet}
 for line in (p/'contacts.jsonl').open():
  r=json.loads(line)
  for c in r['contacts']:
   if int(c['body0'])>=0 and int(c['body1'])>=0:continue
   for obj in c['names']:
    body=obj['body'];side=body[:2]
    if side in ground and body.endswith('_wheel') and float(c['lambda'])!=0:
     ground[side]['loaded_steps']+=1;ground[side]['max_lambda']=max(ground[side]['max_lambda'],abs(float(c['lambda'])))
 result=json.loads((p/'result.json').read_text());last=rows[-1];events.append(dict(kind='stop',t=result['sim_seconds'],reason=result['reason']))
 (out/(label+'-events.json')).write_text(json.dumps(events,indent=2)+'\n');(out/(label+'-actual-legs.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
 report[label]=dict(source=str(p),states_sha256=hashlib.sha256((p/'states.jsonl').read_bytes()).hexdigest(),result=result,feet=feet,phases=phases,ground=ground,actual_dof_vs_body_angle_max_error_rad=max(dof_errors) if dof_errors else None,first_back_contact_s=firstcontact,latch_s=latch,root_z_range=[min(r['root_z'] for r in rows),max(r['root_z'] for r in rows)],final_phase=last['phase'],final_door_deg=math.degrees(last['door_rad']),last_tracking_error=last['tracking_error'],last_arm_speed=last['arm_qd_max'],stages={p:'observed' if any(r['phase']==p for r in rows) else 'not_observed' for p in ['turn','close','restore_heading','final_hold']})
 fig,ax=plt.subplots(5,1,figsize=(12,14),sharex=True)
 for side in feet:
  for j,(part,home) in enumerate([('hip',0),('thigh',.8),('calf',-1.5)]):ax[j].plot(T,[r['q'][side+'_'+part+'_joint']-home for r in rows],label=side)
 ax[0].set_ylabel('Hip - home (rad)');ax[1].set_ylabel('Thigh - home (rad)');ax[2].set_ylabel('Calf - home (rad)');ax[3].plot(T,yaw,label='actual yaw unwrapped');ax[3].set_ylabel('Yaw (rad)');ax[4].plot(T,w,label='command yaw rate');ax[4].plot(T,v,label='forward velocity')
 for a in ax:
  a.grid(alpha=.3);a.legend(loc='upper left')
  for event in events:
   if event['kind']=='phase' and event['phase'] in ['turn','close','restore_heading']:a.axvline(event['t'],ls=':',color='black',alpha=.4)
 ax[-1].set_xlabel('Actual simulation time (s)');fig.suptitle(label+' | Actual rigid-body joints; no replay');fig.tight_layout();fig.savefig(out/(label+'-legs-turn.png'),dpi=140);plt.close(fig)
 allrows[label]=rows
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{n:v[n] for n in ['feet','final_phase','final_door_deg','stages','actual_dof_vs_body_angle_max_error_rad']} for k,v in report.items()},indent=2))
