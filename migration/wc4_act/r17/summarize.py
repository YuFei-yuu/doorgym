"""Post-budget CPU paired analysis. No model, simulation, or new observation inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json,math,hashlib,fcntl,subprocess,concurrent.futures
import numpy as np
from common import *
FINAL=RUN/'FINAL'
def write(p,x):
 def native(v):
  if isinstance(v,np.generic):return v.item()
  if isinstance(v,np.ndarray):return v.tolist()
  raise TypeError(type(v).__name__)
 Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=native)+'\n')
def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi
def distribution(x):
 a=np.asarray(x,float)
 return dict(n=len(a),min=float(a.min()),p50=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max())) if len(a) else None

def one(args):
 directory,horizon,seed=args;d=Path(directory);score=read(d/'r11-score.json');target=read(d/'reference.json')['old_expert_final_yaw']
 # Compact physical state only; no use of phase or expert trajectory labels.
 states=[]
 with (d/'states.jsonl').open() as stream:
  for line in stream:
   s=json.loads(line);states.append([s['t'],s['actual_base'][2],s['action'][0],s['action'][1],s['base_measured_fd_speed'],s['base_measured_fd_yaw_rate'],s['door_q'][0]])
 states=np.asarray(states,float);n=len(states);dt=states[0,0]
 inputs=[]
 with (d/'policy-inputs-actions.jsonl').open() as stream:
  for line in stream:
   r=json.loads(line);e=r['worker_evidence'];assert e['executed_horizon']==horizon
   inputs.append(dict(frame=r['frame'],source_time_s=r['source_time_s'],action=r['applied_action'],new_chunk=e['new_chunk'],chunk_id=e['chunk_id'],worker_wall=e.get('worker_sample_wall_s'),parent_wall=r.get('parent_act_wall_s')))
 assert len(inputs)==(n+1)//2
 all_replans=[r for r in inputs if r['new_chunk']];assert [r['frame'] for r in all_replans]==list(range(0,len(inputs),horizon))
 pass_step=score['whole_body_pass_step'];latch_step=None
 if score['close_hold'] is not None:
  locks=[json.loads(s) for s in (d/'lock-writes.jsonl').open()];latch_step=next(s['step'] for s in locks if s['kind']=='relocked')
 chunks=[];max_integrator=0.
 for r in all_replans:
  frame=r['frame'];begin=frame*2;end=min(n,(frame+horizon)*2)
  if begin==0:continue # Initial orientation has no post-state predecessor; not a late window.
  previous=states[begin-1,1];preerr=wrap(previous-target)
  seg=states[begin:end];posterrs=np.asarray([wrap(v-target) for v in seg[:,1]])
  # Half-open segment: next segment owns its pre-observation boundary.
  inside=posterrs[:-1] if end<n else posterrs
  near=abs(preerr)<=math.radians(2);near_inside=bool(np.any(abs(inside)<=math.radians(2)))
  errors=np.r_[preerr,posterrs];cross=np.flatnonzero((errors[:-1]*errors[1:]<0)&(abs(np.diff(errors))<math.pi))
  brackets=[[float((begin+k)*dt),float((begin+k+1)*dt)] for k in cross]
  cmd=seg[:,3];integral=float(cmd.sum()*dt);max_integrator=max(max_integrator,abs(wrap(seg[-1,1]-previous-integral)))
  chunks.append(dict(seed=seed,horizon=horizon,chunk_id=r['chunk_id'],frame=frame,pre_time_s=r['source_time_s'],physical_begin=begin,physical_end_exclusive=end,post_pass=pass_step is not None and begin>pass_step,post_legal_relock=latch_step is not None and begin>=latch_step,pre_heading_error_deg=math.degrees(preerr),near_at_replan=near,near_only_inside_committed_segment=not near and near_inside,physical50hz_crossing_brackets_s=brackets,mean_yaw_command=float(cmd.mean()),mean_abs_yaw_command=float(abs(cmd).mean()),yaw_active_fraction=float((abs(cmd)>.01).mean()),segment_yaw_integral_deg=math.degrees(integral),post_heading_error_deg=math.degrees(float(posterrs[-1])),near_replan_continues_rotation=near and abs(cmd.mean())>.01,near_replan_mean_commands_stop=near and bool(np.all(abs(seg[:,2:4].mean(0))<=.01)),complete_horizon_actions=(end-begin)==2*horizon))
 scope={}
 for key in ['post_pass','post_legal_relock']:
  rows=[r for r in chunks if r[key]]
  scope[key]=dict(eligible=bool(rows),chunks=len(rows),near_at_replan=sum(r['near_at_replan'] for r in rows),near_only_inside_committed_segment=sum(r['near_only_inside_committed_segment'] for r in rows),near_replan_continues_rotation=sum(r['near_replan_continues_rotation'] for r in rows),near_replan_mean_commands_stop=sum(r['near_replan_mean_commands_stop'] for r in rows),crossings=sum(len(r['physical50hz_crossing_brackets_s']) for r in rows),sum_abs_segment_rotation_deg=sum(abs(r['segment_yaw_integral_deg']) for r in rows))
 sample=[r['worker_wall'] for r in all_replans if r['worker_wall'] is not None];parent=[r['parent_wall'] for r in all_replans if r['parent_wall'] is not None]
 serial=dict(seed=seed,horizon=horizon,steps=n,input_frames=len(inputs),replans=len(all_replans),partial_final_action=n%2==1,partial_final_chunk=len(inputs)%horizon!=0,terminal_heading_error_deg=math.degrees(wrap(states[-1,1]-target)),terminal_speed_m_s=float(states[-1,4]),terminal_yaw_rate_rad_s=float(states[-1,5]),max_abs_door_deg=float(np.max(abs(states[:,6]))*180/math.pi),scopes=scope,actual_segment_yaw_integrator_max_error_rad=max_integrator,simulation_s=float(states[-1,0]),dt_s=float(dt),worker_sample_wall_s=distribution(sample),parent_replan_wall_s=distribution(parent),cold_start_worker_s=sample[0] if sample else None,timing_baseline_unavailable=horizon==25)
 np.savez_compressed(FINAL/'timelines'/f'h{horizon}-s{seed}.npz',rows=states,columns=np.asarray(['time_s','actual_yaw_rad','applied_vx_m_s','applied_yaw_rad_s','measured_speed_m_s','measured_yaw_rate_rad_s','door_angle_rad']),target_yaw=target)
 return serial,chunks,sample,parent

def main():
 assert read(RUN/'R17-COMPLETE.json')['completed']==16
 assert read(RUN/'evaluation-exit.json')['returncode']==0
 assert not (RUN/'STOP_REQUEST.json').exists()
 p=read(RUN/'process.json');assert all(not Path('/proc',str(p[k])).exists() for k in ['supervisor','evaluation'])
 lock_free_now=True
 with open('/data/zyf/video2door/.repro/locks/'+p['gpu_uuid']+'.lock','a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
  except BlockingIOError:lock_free_now=False # another user may lease the already released GPU
 verify();FINAL.mkdir(exist_ok=False);(FINAL/'timelines').mkdir()
 claims=sorted((RUN/'evaluation-claims').glob('*.json'));assert len(claims)==16
 ledger=read(RUN/'ledger.json');assert ledger['completed']==16 and ledger['automatic_candidates']==0
 sourcebaseline=read(RUN/'baseline-h25.json')['records'];counts={5:{},25:{}};records=[];proofs=[]
 for i,seed in enumerate(SEEDS):
  d=RUN/'evaluation'/f'trial_{i:02d}';r=read(d/'evaluation-result.json');a=read(d/'deep-offline-audit.json');audit=read(d/'r17-runtime-audit.json')
  assert a['status']=='passed' and audit['runtime_evidence_passed']
  assert read(RUN/'evaluation'/f'trial_{i:02d}-exit.json')['returncode']==0
  assert read(d/'scene-creation-receipt.json')['scenes']==1
  assert read(d/'worker-model-binding.json')['model_weights_equal']==234 and read(d/'worker-model-binding.json')['normalizer_tensors_equal']==110
  assert read(d/'paired-initial-binding.json')['all_actual_initial_physics_camera_equal']
  assert r['full_task_success'] is False and not r['r11_measured_pass']
  score=read(d/'r11-score.json');assert score['failures']
  pair=read(d/'paired-comparison.json');assert pair['seed']==seed
  for h,gates in [(5,pair['gates']),(25,pair['baseline_gates'])]:
   for k,v in gates.items():counts[h][k]=counts[h].get(k,0)+int(v)
  baseline=sourcebaseline[i];assert baseline['seed']==seed
  for name,h in baseline['frozen_files'].items():assert sha(Path(baseline['original_directory'])/name)==h
  for name in ['initial-state.json','physical-camera-binding.json','frozen-trial.json']:assert read(d/name)==read(Path(baseline['original_directory'])/name)
  records.append(dict(seed=seed,h5=pair['gates'],h25=pair['baseline_gates'],improved=pair['improved_items'],regressed=pair['regressed_items'],full_success_h5=False,full_success_h25=baseline['full_task_success'],h5_failures=score['failures'],result=r,exit=read(RUN/'evaluation'/f'trial_{i:02d}-exit.json')))
  proofs.append(dict(seed=seed,**audit))
 write(FINAL/'paired-results.json',dict(denominator=16,model_step=50000,model_sha256=MODEL_SHA,counts=counts,automatic_candidates_h5=0,reviewed_full_successes_h5=0,review_basis='All16 independently audited R11 necessary gates failed; no success candidates awaiting visual acceptance',records=records,new_h25_calls=0,new_training=0))
 # Explicit paths prevent any evaluator/model import or new scene.
 work=[(str(RUN/'evaluation'/f'trial_{i:02d}') if h==5 else str(BASE/'evaluation/050000'/f'trial_{i:02d}'),h,seed) for h in [5,25] for i,seed in enumerate(SEEDS)]
 serial=[];chunk_rows=[];samples=[];parents=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  for s,c,ws,pa in pool.map(one,work):
   serial.append(s);chunk_rows+=c;samples+=ws;parents+=pa;print('CPU paired timing analyzed',s['horizon'],s['seed'],flush=True)
 scopes={}
 for h in [5,25]:
  rr=[r for r in serial if r['horizon']==h];scopes[h]={}
  for scope in ['post_pass','post_legal_relock']:
   fields=['chunks','near_at_replan','near_only_inside_committed_segment','near_replan_continues_rotation','near_replan_mean_commands_stop','crossings']
   scopes[h][scope]=dict(eligible_episodes=sum(r['scopes'][scope]['eligible'] for r in rr),**{f:sum(r['scopes'][scope][f] for r in rr) for f in fields})
 write(FINAL/'paired-target-timing.json',dict(scope_definitions=dict(post_pass='Replan after measured first whole-body passage; may include an eventual returned episode, disclosed in paired gates',post_legal_relock='Replan after R11-qualified physical close hold and measured angle latch',near='absolute wrapped measured heading error <=2deg; commands-stop is a descriptive mean vx/yaw threshold .01, not R11 full success',segments='Same half-open physical50Hz near windows for h5 and h25; physical crossing brackets include each applied-step interval'),scope_counts=scopes,episodes=serial,chunks=chunk_rows,no_expert_action_truth_transferred=True,new_model_forward_calls=0))
 cold=[s['cold_start_worker_s'] for s in serial if s['horizon']==5]
 write(FINAL/'inference-wall-timing.json',dict(h5_replans=len(samples),worker_sample_wall_s=distribution(samples),parent_replan_wall_s=distribution(parents),cold_start_each_episode_s=cold,cold_start_distribution=distribution(cold),worker_warm_only_s=distribution([x for x in samples]),h25_same_scope_timer_available=False,note='Worker sample includes model prediction, postprocessing and required CPU transfer synchronization; parent includes IPC. Warm-only distribution below recomputed excluding each episode first replan. No sim timestep/action-frequency compensation.',simulation_replan_interval_s=.2,control_hz=25,physics_hz=50))
 # Correctly exclude 16 cold starts without constructing policies.
 warm=[]
 for i in range(16):
  d=RUN/'evaluation'/f'trial_{i:02d}';first=True
  for line in (d/'policy-inputs-actions.jsonl').open():
   row=json.loads(line)
   if not row['worker_evidence']['new_chunk']:continue
   if first:first=False;continue
   warm.append(row['worker_evidence']['worker_sample_wall_s'])
 t=read(FINAL/'inference-wall-timing.json');t['worker_warm_only_s']=distribution(warm);write(FINAL/'inference-wall-timing.json',t)
 total={k:sum(p[k] for p in proofs) for k in ['emitted','new_chunks','commands','completed_physics_steps','full_chunks']}
 write(FINAL/'execution-budget-audit.json',dict(status='passed',claims=16,completed=16,scene_creation_receipts=16,evaluation_exit=read(RUN/'evaluation-exit.json'),lease_released=read(RUN/'LEASE_RELEASED.json'),live_processes_gone=True,UUID_lock_free_verified=lock_free_now,R17_lease_released=True,total_worker_application=total,proofs=proofs,source_models_frozen_files_and_baseline_unchanged=True,new_training_updates=0,new_collection=0,extra_smoke_scenes=0,extra_offline_policy_forward=0,partial_terminal_tails=[dict(seed=s['seed'],steps=s['steps'],input_frames=s['input_frames'],partial_action=s['partial_final_action'],partial_chunk=s['partial_final_chunk']) for s in serial if s['horizon']==5 and (s['partial_final_action'] or s['partial_final_chunk'])]))
 print('R17 CPU summary complete',json.dumps(counts),flush=True)
if __name__=='__main__':main()
