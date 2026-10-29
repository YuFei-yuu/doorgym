"""Actual state events only. No expert phase/stage labels, no motion output."""
import json,math,gzip
import numpy as np
from pathlib import Path

def audit(output,seed,step):
 output=Path(output)
 def lines(name):
  path=output/name
  opener=open if path.exists() else gzip.open
  if not path.exists():path=Path(str(path)+'.gz')
  with opener(path,'rt') as f:
   for line in f:yield json.loads(line)
 result=json.loads((output/'result.json').read_text());ref=json.loads((output/'reference.json').read_text())
 events={k:None for k in ['opened80','whole_body_pass','legal_relock','contact_separation','arm_fold','heading_restored','stable_hold','leg_posture_deviation']}
 maximum=0.;rows=0;final=None;closed_since=None;prev=None;last_locked=False;was_contact=False
 for s in lines('states.jsonl'):
  assert s['step']==rows;rows+=1;t=s['t'];assert abs(t-rows*.02)<.001
  maximum=max(maximum,math.degrees(abs(s['door_q'][0])))
  def record(k,condition):
   if condition and events[k] is None:events[k]=t
  record('opened80',abs(s['door_q'][0])>=math.radians(80))
  record('whole_body_pass',s['whole_geometry_max_x']<s['pass_plane_x'])
  record('contact_separation',was_contact and not s['any_door_contact'])
  was_contact|=s['any_door_contact']
  # Completion events evaluated after passage, independently of labels.
  after_pass=events['whole_body_pass'] is not None
  fold=np.all(abs(np.asarray(s['arm_q'][:6])-ref['home'][:6])<=.03) and np.all(abs(np.asarray(s['arm_qd'][:6]))<=.05)
  heading=abs((s['actual_base'][2]-ref['old_expert_final_yaw']+math.pi)%(2*math.pi)-math.pi)<=math.radians(2)
  record('arm_fold',after_pass and fold)
  record('heading_restored',after_pass and heading)
  record('stable_hold',s['final_hold_observed_span_s']>=2.)
  guard=s['foot_guard'];record('leg_posture_deviation',guard['since_s'] is not None and t-guard['since_s']>=.1-1e-9)
  if s['latched'] is not None and not last_locked:
   assert prev is not None and s['latched']==prev['door_q'][0]
   assert prev['task_monitor']['close_span_s']>=.5-1e-8 and not prev['task_monitor']['returned']
   events['legal_relock']=t-.02
  last_locked=s['latched'] is not None;prev=s;final=s
 assert rows==result['steps'] and rows<=6000
 locks=list(lines('lock-writes.jsonl'));relocks=[r for r in locks if r['kind']=='relocked']
 assert bool(relocks)==(events['legal_relock'] is not None)
 if relocks:assert all(r['delta']==0 for r in relocks)
 success=events['opened80'] is not None and events['legal_relock'] is not None and final['latched'] is not None and not final['unlocked']
 assert success==result['simplified_score']['success']==result['success']
 out=dict(seed=seed,step=step,success=success,max_open_deg=maximum,events=events,stop_reason=result['reason'],seconds=result['sim_seconds'],steps=rows,error=result['error'],failures=result['simplified_score']['failures'],source=str(output),criterion='actual opening >=80 deg AND final legal relock; other events diagnostics only',phase_labels_used=False)
 (output/'event-audit.json').write_text(json.dumps(out,indent=2)+'\n')
 return out
