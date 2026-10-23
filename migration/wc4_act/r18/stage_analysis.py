"""Post-produced-stage CPU analysis only; no model or simulator imports/calls."""
import os,sys,json,math
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];RUN=Path(os.environ.get('WC4_EXECUTION_RUN',str(ROOT/'.repro/runs/wc4-fulltask-act-r18-s1000-20261023')))
step=int(sys.argv[1]);assert step in [75000,100000]
base=RUN/'evaluation'/f'{step:06d}';assert json.loads((base/'complete.json').read_text())['consumed']==16
out=base/'offline-target-analysis';out.mkdir(exist_ok=False)

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi
records=[];chunks=[]
source=read(RUN/'baseline-050000.json')['records']
for tag in ['baseline050000',f'{step:06d}']:
 for i,prior in enumerate(source):
  d=Path(prior['original_directory']) if tag=='baseline050000' else base/f'trial_{i:02d}';score=read(d/'r11-score.json');target=read(d/'reference.json')['old_expert_final_yaw'];rows=[]
  with (d/'states.jsonl').open() as stream:
   for line in stream:
    s=json.loads(line);rows.append([s['t'],s['actual_base'][2],s['action'][0],s['action'][1],s['base_measured_fd_speed'],s['base_measured_fd_yaw_rate'],s['door_q'][0]])
  a=np.asarray(rows);dt=float(a[0,0]);latch=None
  if score['close_hold']:
   latch=next(json.loads(x)['step'] for x in (d/'lock-writes.jsonl').open() if json.loads(x)['kind']=='relocked')
  current=[]
  with (d/'policy-inputs-actions.jsonl').open() as stream:
   for line in stream:
    r=json.loads(line);e=r['worker_evidence'];assert e['executed_horizon']==25
    if not e['new_chunk']:continue
    begin=2*r['frame'];end=min(begin+50,len(a))
    if begin==0:continue
    pre=wrap(a[begin-1,1]-target);errors=np.asarray([wrap(y-target) for y in a[begin:end,1]]);inside=errors[:-1] if end<len(a) else errors
    allerrors=np.r_[pre,errors];cross=np.flatnonzero((allerrors[:-1]*allerrors[1:]<0)&(abs(np.diff(allerrors))<math.pi));omega=a[begin:end,3]
    row=dict(checkpoint=tag,seed=prior['seed'],frame=r['frame'],time_s=float(r['source_time_s']),post_pass=bool(score['whole_body_pass_step'] is not None and begin>score['whole_body_pass_step']),post_legal_relock=bool(latch is not None and begin>=latch),pre_heading_error_deg=math.degrees(pre),mean_applied_yaw_rate=float(omega.mean()),mean_abs_applied_yaw_rate=float(abs(omega).mean()),active_rotation_fraction=float((abs(omega)>.01).mean()),near_at_replan=bool(abs(pre)<=math.radians(2)),near_only_inside_segment=bool(abs(pre)>math.radians(2) and np.any(abs(inside)<=math.radians(2))),near_replan_continues_rotation=bool(abs(pre)<=math.radians(2) and abs(omega.mean())>.01),physical50hz_target_crossings_s=[[float((begin+int(k))*dt),float((begin+int(k)+1)*dt)] for k in cross],segment_rotation_deg=math.degrees(float(omega.sum()*dt)))
    current.append(row)
  scopes={}
  for scope in ['post_pass','post_legal_relock']:
   rr=[r for r in current if r[scope]];scopes[scope]=dict(eligible=bool(rr),chunks=len(rr),near_at_replan=sum(r['near_at_replan'] for r in rr),near_only_inside_segment=sum(r['near_only_inside_segment'] for r in rr),near_replan_continues_rotation=sum(r['near_replan_continues_rotation'] for r in rr),physical_crossings=sum(len(r['physical50hz_target_crossings_s']) for r in rr),absolute_applied_rotation_deg=sum(abs(r['segment_rotation_deg']) for r in rr))
  records.append(dict(checkpoint=tag,seed=prior['seed'],scopes=scopes,terminal_heading_error_deg=math.degrees(wrap(a[-1,1]-target)),terminal_speed_m_s=float(a[-1,4]),terminal_yaw_rate_rad_s=float(a[-1,5]),steps=len(a),simulation_s=float(a[-1,0])))
  chunks+=current
  np.savez_compressed(out/f'{tag}-s{prior["seed"]}.npz',rows=a,columns=np.asarray(['time_s','actual_yaw_rad','applied_vx_m_s','applied_yaw_rad_s','measured_speed_m_s','measured_yaw_rate_rad_s','door_angle_rad']),target_yaw=target)
  print('stage CPU existing-record analysis',tag,prior['seed'],flush=True)
write(out/'target-timing.json',dict(records=records,chunks=chunks,execution_horizon=25,scope='Same frozen16 initial states. Physical50Hz, +/-2deg, half-open committed25-action segments. Conditional post-pass/post-legal-relock denominators must be shown; no expert action truth transferred.',new_model_forward_calls=0,new_scenes=0))
print('Stage offline timing complete; no policy/model calls',flush=True)
