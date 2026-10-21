"""Read-only full-task evidence review for completed frozen trials."""
import argparse,collections,hashlib,json,math,pathlib,shutil
import numpy as np
import av
from PIL import Image,ImageDraw
R=pathlib.Path('/data/zyf/video2door-wc4-act/.repro/runs/wc4-fulltask-pilot-r8-20261021')


def review(i):
 p=R/('trial_%02d'%i);out=p/'review';out.mkdir(exist_ok=True)
 s=[json.loads(l) for l in (p/'states.jsonl').open()];dt=s[0]['t'];r=json.loads((p/'result.json').read_text());v=json.loads((p/'record-validation.json').read_text())
 ref=json.loads((p/'reference.json').read_text());locks=[json.loads(l) for l in (p/'lock-writes.jsonl').open()];rel=[x for x in locks if x['kind']=='relocked']
 closed=lambda x:abs(x['door_q'][0])<=math.radians(1) and abs(x['door_qd'][0])<=math.radians(1)
 hold=None
 if rel:
  k=rel[0]['step'];b=k-1
  while b>0 and closed(s[b-1]) and s[b-1]['latched'] is None:b-=1
  hold=dict(first_post_s=s[b]['t'],latch_pre_s=rel[0]['time_s'],span_s=rel[0]['time_s']-s[b]['t'],measured_angle=rel[0]['latched'],no_snap=rel[0]['latched']==s[k-1]['door_q'][0],max_write_delta=max(abs(x['delta']) for x in rel))
 wrap=lambda a:(a+math.pi)%(2*math.pi)-math.pi
 good=lambda x:(x['whole_geometry_max_x']<x['pass_plane_x'] and x['passed'] and not x['any_door_contact'] and closed(x) and x['latched'] is not None and not x['unlocked']
  and abs(wrap(x['actual_base'][2]-ref['old_expert_final_yaw']))<=math.radians(2) and x['base_measured_fd_speed']<=.01 and x['base_measured_fd_yaw_rate']<=.01
  and np.all(abs(np.array(x['arm_q'][:6])-ref['home'][:6])<=.03) and np.all(abs(np.array(x['arm_qd'][:6]))<=.05))
 b=len(s)-1
 while b>0 and good(s[b-1]):b-=1
 window=s[-1]['t']-s[b]['t'] if good(s[-1]) else 0.
 F=s[b:] if good(s[-1]) else []
 newend=[];moving=[];nonend=[];pairs=collections.defaultdict(collections.Counter);rawrows=0
 for line in (p/'contacts.jsonl').open():
  c=json.loads(line);rawrows+=1;x=s[c['step']]
  for a in c['contacts']:
   f=a.get('review_flags',{});name=' / '.join(n['body'] for n in a['names'])
   if f.get('nonzero_door_contact'):pairs[x['phase']][name]+=1
   if x['passed'] and x['phase'] in ['close','compact_seek'] and f.get('nonzero_end_panel_contact'):newend.append(x['step'])
   if x['phase']=='close' and f.get('nonzero_end_panel_contact') and x['door_q'][0]<x['pre_physics']['door_q'][0]:moving.append(x['step'])
   if x['phase']=='close' and f.get('off_target_closing_contact'):nonend.append(x['step'])
 passed=next((x['step'] for x in s if x['passed']),None)
 no_return=passed is not None and all(x['whole_geometry_max_x']<x['pass_plane_x'] for x in s[passed:])
 phi=[x for x in s if x['phase'].startswith('compact_') and x['phase'] not in ['compact_exit_turn','compact_exit_arc']]
 closing=[x for x in s if x['phase']=='close']
 support=bool(v['recording_checks_pass'] and r.get('automatic_final_conditions_met') and r['independent_task_monitor']['accepted'] and hold and hold['span_s']>=.5 and hold['no_snap'] and window>=2 and no_return and moving and not nonend
  and all(x['action'][0]>=0 for x in s) and all(x['door_effort'][0]==0 for x in s if x['unlocked']) and rawrows==len(s))
 phase_first={}
 for x in s:phase_first.setdefault(x['phase'],x['step'])
 chosen=set([0,(len(s)-1)//2]);chosen.update(k//2 for k in phase_first.values())
 if newend:chosen.update([min(newend)//2,(min(newend)+1)//2])
 if closing:chosen.update(int(k)//2 for k in np.linspace(closing[0]['step'],closing[-1]['step'],4))
 frames=[]
 with av.open(str(p/'overview/env_0000/rollout.mp4')) as f:
  for n,frame in enumerate(f.decode(video=0)):
   if n in chosen:
    path=out/('original-%04d.png'%n);frame.to_image().save(path);frames.append(dict(frame=n,step=2*n,t=s[2*n]['t'],path=path.name))
 # Compact review sheet: actual original frames only, no state replay.
 keys=[phase_first.get('compact_withdraw',0),min(newend) if newend else phase_first.get('push_door',0),
       closing[len(closing)//2]['step'] if closing else len(s)-1,rel[0]['step'] if rel else len(s)-1,phase_first.get('retract',len(s)-1),len(s)-1]
 sheet=Image.new('RGB',(1920,3*345),'white');draw=ImageDraw.Draw(sheet)
 for j,k in enumerate(keys):
  frame=k//2;path=out/('original-%04d.png'%frame)
  if not path.exists():
   # Extract an already recorded frame, never render or replay a state.
   with av.open(str(p/'overview/env_0000/rollout.mp4')) as f:
    for n,im in enumerate(f.decode(video=0)):
     if n==frame:im.to_image().save(path);break
  im=Image.open(path);im.thumbnail((960,320));x=(j%2)*960;y=(j//2)*345
  sheet.paste(im,(x,y+25));draw.text((x+5,y+5),'trial %02d | original frame %d | t=%.2fs | %s'%(i,frame,s[2*frame]['t'],s[2*frame]['phase']),fill='black')
 sheet.save(out/'physical-review-sheet.jpg',quality=94)
 summary=dict(trial=i,seed=v['seed'],nominal=i==0,auto_success=r.get('automatic_final_conditions_met'),recording_checks_pass=v['recording_checks_pass'],execution_review_supports_full_success=support,
   candidate_training_eligible=support,external_independent_acceptance='awaiting',steps=len(s),frames=v['frames'],sim_seconds=r['sim_seconds'],reason=r['reason'],whole_body_pass_step=passed,no_return=no_return,
   hold=hold,final_common_span_s=window,first_new_end_contact_step=min(newend) if newend else None,first_new_end_contact_time_s=s[min(newend)]['t'] if newend else None,
   aligned_end_push_steps=len(set(moving)),nonend_close_steps=sorted(set(nonend)),nonzero_door_pairs={k:dict(z) for k,z in pairs.items()},
   approach_door_range_deg=[min(math.degrees(x['door_q'][0]) for x in phi),max(math.degrees(x['door_q'][0]) for x in phi)] if phi else None,
   final_metrics=dict(home_max_error=float(max(np.max(abs(np.array(x['arm_q'][:6])-ref['home'][:6])) for x in F)),joint_max_speed=float(max(np.max(abs(np.array(x['arm_qd'][:6]))) for x in F)),
     yaw_max_error_deg=max(abs(math.degrees(wrap(x['actual_base'][2]-ref['old_expert_final_yaw']))) for x in F),base_max_speed=max(x['base_measured_fd_speed'] for x in F),base_max_yaw_rate=max(x['base_measured_fd_yaw_rate'] for x in F)) if F else None,
   original_frame_map=frames,raw_npz=v['raw_npz'],raw_npz_bytes=v['raw_npz_bytes'],effective_sampling=json.loads((p/'frozen-trial.json').read_text())['effective_sampling'],
   limitations=['Raw result nominal_only is a legacy R7 scope field; the authoritative per-trial initial-state/camera distribution is frozen-trial.json and physical-camera-binding.json.',
    'Physical review uses contacts, complete measured states, code and original video frames; no force calibration or coverage threshold.'])
 (out/'review.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k not in ['frames','effective_sampling','nonzero_door_pairs']},indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('indices',type=int,nargs='+');a=p.parse_args()
 for i in a.indices:review(i)
