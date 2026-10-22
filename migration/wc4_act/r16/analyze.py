"""Post-prediction CPU evidence analysis; never constructs a policy or a scene."""
import sys,time,json,math
import numpy as np
from common import *
obs=read(RUN/'observations.json')['points'];done=read(RUN/'inference-complete.json');assert done['completed']==1536
ledger=[json.loads(s) for s in (RUN/'forward-budget.jsonl').open()]
claims=[s for s in ledger if s['event']=='forward_claim'];assert min(s['at'] for s in claims)>read(RUN/'FROZEN.json')['at'];completions=[s for s in ledger if s['event']=='forward_completed']
assert len(claims)==len(completions)==1536 and [s['sequence'] for s in claims]==list(range(1,1537))
assert len(set((s['model_step'],s['observation']) for s in claims))==1536
for s in completions:assert sha(RUN/s['prediction'])==s['prediction_sha256']
pred={}
for step in [30000,40000,50000]:
 pred[step]=[]
 for p in obs:
  with np.load(RUN/'predictions'/f'{step:06d}-{p["id"]}.npz') as a:pred[step].append(a['physical'].copy())
 pred[step]=np.stack(pred[step])
index={p['id']:i for i,p in enumerate(obs)};metrics={};point_metrics=[]
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 return x
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=serial)+'\n')
# Per-point errors keep valid horizons inside the source episode; padding is never a target.
for step in pred:
 groups={}
 for i,p in enumerate(obs):
  a=pred[step][i];r=dict(model_step=step,id=p['id'],kind=p['kind'],group=p['group'])
  for lo,hi,label in [(0,25,'first25'),(25,100,'tail75'),(0,100,'full100')]:
   r[label]=dict(mean_vx=float(a[lo:hi,0].mean()),mean_yaw_rate=float(a[lo:hi,1].mean()),mean_abs_yaw_rate=float(abs(a[lo:hi,1]).mean()),active_yaw_fraction=float((abs(a[lo:hi,1])>.01).mean()),integrated_yaw_rad=float(a[lo:hi,1].sum()/25),first_stop_offset=next((j+lo for j,x in enumerate(a[lo:hi]) if abs(x[0])<=.01 and abs(x[1])<=.01),None))
  if p['kind']=='expert':
   with np.load(RUN/p['truth']) as z:y=z['action'];valid=z['valid']
   for lo,hi,label in [(0,25,'first25'),(25,100,'tail75'),(0,100,'full100')]:
    mask=valid[lo:hi];delta=a[lo:hi][mask]-y[lo:hi][mask]
    truthstop=(abs(y[lo:hi,0])<=.01)&(abs(y[lo:hi,1])<=.01)&mask
    predictedstop=(abs(a[lo:hi,0])<=.01)&(abs(a[lo:hi,1])<=.01)
    key=(p['group'],label);g=groups.setdefault(key,{'errors':[],'truth_stop':0,'predicted_stop_on_true_stop':0,'valid_count':0,'false_stop_on_moving':0,'moving_count':0,'moving_errors':[]})
    if len(delta):g['errors'].append(delta)
    if (mask&~truthstop).any():g['moving_errors'].append(a[lo:hi][mask&~truthstop]-y[lo:hi][mask&~truthstop])
    g['truth_stop']+=int(truthstop.sum());g['predicted_stop_on_true_stop']+=int((truthstop&predictedstop).sum());g['valid_count']+=int(mask.sum());g['false_stop_on_moving']+=int((mask&~truthstop&predictedstop).sum());g['moving_count']+=int((mask&~truthstop).sum())
    r[label]['valid_count']=int(mask.sum());r[label]['physical_mae']=abs(delta).mean(0).tolist() if len(delta) else None
    r[label]['truth_mean_vx_yaw']=y[lo:hi][mask,:2].mean(0).tolist() if mask.any() else None
    r[label]['prediction_mean_vx_yaw_valid']=a[lo:hi][mask,:2].mean(0).tolist() if mask.any() else None
  point_metrics.append(r)
 result=[]
 for (group,horizon),g in groups.items():
  errors=np.concatenate(g.pop('errors')) if g['valid_count'] else np.empty((0,9))
  moving_parts=g.pop('moving_errors');moving_errors=np.concatenate(moving_parts) if moving_parts else np.empty((0,9))
  result.append(dict(group=group,horizon=horizon,physical_mae=abs(errors).mean(0).tolist() if len(errors) else None,physical_rmse=np.sqrt((errors**2).mean(0)).tolist() if len(errors) else None,moving_only_mae=abs(moving_errors).mean(0).tolist() if len(moving_errors) else None,**g))
 metrics[step]=result
write(RUN/'expert-physical-errors.json',dict(action_order=['vx_m_s','yaw_rad_s','joint1_rad','joint2_rad','joint3_rad','joint4_rad','joint5_rad','joint6_rad','gripper_rad'],models=metrics,padding_excluded=True,not_vae_training_loss=True,episode_count=72,points_per_group=72))
write(RUN/'point-prediction-analysis.json',point_metrics)
# Reconstruct full historical state/receipt/application paths for the four actual failure trajectories.
rollouts=[];comparisons=[];timelines=[]
for seed in SEEDS:
 d=R15/'evaluation/050000'/f'trial_{seed-62200:02d}'
 rows=[json.loads(x) for x in (d/'policy-inputs-actions.jsonl').open()]
 full=[]
 for line in (d/'states.jsonl').open():
  s=json.loads(line);full.append({k:s[k] for k in ['step','t','actual_base','command_base','action','observation_pre','base_measured_fd_yaw_rate','base_measured_fd_speed','arm_q']})
 lim=read(d/'joint-limits.json');lower=np.asarray(lim['physical_lower'],np.float32);upper=np.asarray(lim['physical_upper'],np.float32)
 assert len(full)==2*len(rows)
 expected=None;commands=acks=emitted=0;prevseq=0;lastbase=np.zeros(2,np.float32);maxclock=maxstate=maxapply=0.
 for i,row in enumerate(rows):
  r=row['worker_evidence'];assert row['frame']==i and row['applied_steps']==[2*i,2*i+1]
  assert r['new_chunk']==(i%25==0) and r['action_index']==i%25 and r['executed_horizon']==25
  assert r['emission_seq']==r['observation_seq']==i+1 and r['sampled_observation_seq']==(i//25)*25+1
  assert r['queue_after']==24-i%25 and r['queue_before']==(0 if i%25==0 else 25-i%25)
  if r['new_chunk']:
   expected=np.asarray(r['selected_chunk_actions'],np.float32);assert expected.shape==(25,9) and ahash(expected)==r['chunk_sha256']
  raw=np.asarray(row['policy_output'],np.float32);assert np.array_equal(raw,expected[i%25]) and np.array_equal(raw,np.asarray(r['action'],np.float32))
  applied=raw.copy();applied[2:]=np.clip(applied[2:],lower,upper)
  assert np.array_equal(applied,np.asarray(row['applied_action'],np.float32))
  state=np.asarray(row['state'],np.float32);assert np.array_equal(state[:2],lastbase);lastbase=applied[:2]
  assert np.array_equal(state[2:],np.asarray(full[2*i]['observation_pre']['state'],np.float32)[2:])
  maxclock=max(maxclock,abs(row['source_time_s']-i/25));maxstate=max(maxstate,float(abs(state[2:]-np.asarray(full[2*i]['observation_pre']['state'],np.float32)[2:]).max()))
  for s in full[2*i:2*i+2]:assert np.array_equal(np.asarray(s['action'],np.float32),applied)
 for line in (d/'worker-applied-evidence.jsonl').open():
  j=json.loads(line)
  if j['event']=='emitted':
   emitted+=1;seq=j['receipt']['emission_seq'];assert seq==emitted and commands==acks==2*(emitted-1)
  elif j['event']=='command_applied':
   commands+=1;seq=j['emission_seq'];assert seq==emitted and j['substep']==(commands-1)%2 and j['step']==commands-1 and acks==commands-1
   assert np.array_equal(np.asarray(j['action'],np.float32),np.asarray(rows[seq-1]['applied_action'],np.float32))
  elif j['event']=='physics_completed':
   acks+=1;assert j['emission_seq']==emitted and j['step']==acks-1 and j['substep']==(acks-1)%2 and acks==commands
 assert emitted==len(rows) and commands==acks==2*len(rows)
 ref=read(d/'reference.json');target=ref['old_expert_final_yaw'];tl=read(RUN/'truth'/f'failure-timeline-{seed}.json');byframe={r['frame']:r for r in tl['rows']};selected={p['frame']:p for p in obs if p['kind']=='failure' and p['seed']==seed}
 angle_integrator=[];chunk_rows=[]
 for f in range(0,len(rows),25):
  block=rows[f:f+25];actions=np.asarray([r['applied_action'] for r in block]);before=byframe[f]['actual_base'][2]
  after=full[min(2*(f+len(block))-1,len(full)-1)]['actual_base'][2]
  angle_integrator.append(abs(float(wrap((after-before)-actions[:,1].sum()/25))))
  physical_errors=np.r_[wrap(before-target),[wrap(s['actual_base'][2]-target) for s in full[2*f:2*(f+len(block))]]]
  physical_cross=np.flatnonzero((physical_errors[:-1]*physical_errors[1:]<0)&(abs(np.diff(physical_errors))<np.pi))
  physical_cross_brackets=[[f/25+k*.02,f/25+(k+1)*.02] for k in physical_cross.tolist()]
  preerr=byframe[f]['heading_error_rad'];cross=[x for x in tl['target_crossing_frames'] if f<=x<min(f+25,len(rows))];nears=[x for x in range(f,min(f+25,len(rows))) if abs(byframe[x]['heading_error_rad'])<=math.radians(2)]
  if f>=tl['tail_start']:
   chunk_rows.append(dict(seed=seed,frame=f,time_s=f/25,chunk_id=block[0]['worker_evidence']['chunk_id'],pre_error_deg=math.degrees(preerr),near_at_replan=abs(preerr)<=math.radians(2),near_inside_committed_chunk=bool(nears),near_only_inside_chunk=bool(nears) and abs(preerr)>math.radians(2),target_crossing_frames=cross,physical_50hz_crossing_brackets_s=physical_cross_brackets,first25_mean_yaw_rate=float(actions[:,1].mean()),first_yaw_rate=float(actions[0,1]),first25_mean_abs_yaw_rate=float(abs(actions[:,1]).mean()),first25_active_yaw_fraction=float((abs(actions[:,1])>.01).mean()),first25_mean_vx=float(actions[:,0].mean()),segment_yaw_integral_deg=math.degrees(float(actions[:,1].sum()/25)),post_error_deg=math.degrees(float(wrap(after-target))),continued_rotation_near_at_replan=abs(preerr)<=math.radians(2) and abs(actions[:,1].mean())>.01,selected_for_R16=f in selected))
 for f,p in selected.items():
  i=index[p['id']];archive=np.asarray(rows[f]['worker_evidence']['selected_chunk_actions'],np.float32);a=pred[50000][i,:25];delta=abs(a-archive)
  comparisons.append(dict(seed=seed,frame=f,id=p['id'],max_abs_all9=float(delta.max()),max_abs_base2=float(delta[:,:2].max()),mean_abs_yaw=float(delta[:,1].mean()),historical_device='cuda',new_device='cpu',same_original_input=True,original_worker_chunk_id=rows[f]['worker_evidence']['chunk_id'],new_mean_yaw_first25=float(a[:,1].mean()),worker_mean_yaw_first25=float(archive[:,1].mean())))
 rollouts.append(dict(seed=seed,frames=len(rows),chunks=len(rows)//25,worker_emitted=emitted,physical_commands=commands,physical_acks=acks,base_previous_25hz_equal=True,state_arm_pre_equal=True,physical_only_joint_clamp_equal=True,queue25_equal=True,clock_max_error_s=maxclock,measured_segment_yaw_integral_max_error_rad=max(angle_integrator),tail_start_s=tl['tail_start']/25,terminal_heading_error_deg=math.degrees(byframe[len(rows)-1]['heading_error_rad']),post_relock_near_replan_chunks=sum(r['near_at_replan'] for r in chunk_rows),post_relock_near_only_inside_chunks=sum(r['near_only_inside_chunk'] for r in chunk_rows),post_relock_continue_near_replan_chunks=sum(r['continued_rotation_near_at_replan'] for r in chunk_rows)))
 timelines.extend(chunk_rows)
write(RUN/'actual-execution-audit.json',dict(rollouts=rollouts,CPU_vs_historical_GPU= dict(points=len(comparisons),max_abs_all9=max(r['max_abs_all9'] for r in comparisons),max_abs_base2=max(r['max_abs_base2'] for r in comparisons)),comparisons=comparisons,no_extra_forward=True))
write(RUN/'chunk-target-timelines.json',timelines)
# Descriptive observation similarity only. No expert-action labels are assigned to failures.
features=[];states=[];full_images=[]
for p in obs:
 with np.load(RUN/p['input']) as a:
  states.append(a['state'].copy());features.append(np.concatenate([a[v][::16,::16,0].reshape(-1) for v in ['front','wrist']]).astype(np.float32)/255)
features=np.stack(features);states=np.stack(states);experts=np.asarray([i for i,p in enumerate(obs) if p['kind']=='expert']);fails=np.asarray([i for i,p in enumerate(obs) if p['kind']=='failure']);scale=np.asarray([.01,.01]+[.03]*7)
nearest=[];ambiguities=[];half=features.shape[1]//2
for i in fails:
 sd=np.max(abs(states[experts]-states[i])/scale,axis=1);delta=features[experts]-features[i]
 rms=np.stack([np.sqrt((delta[:,:half]**2).mean(1)),np.sqrt((delta[:,half:]**2).mean(1))],1)
 rank=np.maximum(sd,rms.max(1)/.02);j0=int(np.argmin(rank));j=int(experts[j0])
 nearest.append(dict(failure_id=obs[i]['id'],expert_id=obs[j]['id'],expert_group=obs[j]['group'],state_max_scaled=float(sd[j0]),coarse_depth_rmse=rms[j0].tolist(),jointly_near=bool(sd[j0]<=1 and rms[j0].max()<=.02),no_transferred_expert_action_truth=True))
for ai,i in enumerate(experts):
 js=experts[ai+1:]
 if not len(js):continue
 sd=np.max(abs(states[js]-states[i])/scale,axis=1);delta=features[js]-features[i];rms=np.stack([np.sqrt((delta[:,:half]**2).mean(1)),np.sqrt((delta[:,half:]**2).mean(1))],1)
 for k in np.flatnonzero((sd<=1)&(rms.max(1)<=.02)):
  j=int(js[k]);
  if obs[i]['group']==obs[j]['group']:continue
  with np.load(RUN/obs[i]['truth']) as z:ya=z['action'];va=z['valid']
  with np.load(RUN/obs[j]['truth']) as z:yb=z['action'];vb=z['valid']
  valid=va&vb;yd=abs(ya[valid]-yb[valid]);amb=dict(a=obs[i]['id'],b=obs[j]['id'],groups=[obs[i]['group'],obs[j]['group']],state_max_scaled=float(sd[k]),coarse_depth_rmse=rms[k].tolist(),valid_common=int(valid.sum()),mean_future_abs_base_difference=yd[:,:2].mean(0).tolist())
  ambiguities.append(amb)
# Full resolution confirmation for ranked descriptive examples only, no new predictions.
ranked=sorted(ambiguities,key=lambda x:-x['mean_future_abs_base_difference'][1])[:12]
for r in ranked:
 a=obs[index[r['a']]];b=obs[index[r['b']]]
 with np.load(RUN/a['input']) as za,np.load(RUN/b['input']) as zb:
  r['full_resolution_depth_rmse']=[float(np.sqrt(np.mean(((za[v].astype(np.float32)-zb[v])/255)**2))) for v in ['front','wrist']]
write(RUN/'observation-similarity.json',dict(scope='Only 512 pre-frozen observations; no exhaustive state-space search. Descriptive 32x40 depth thumbnails and original R11 diagnostic scales; not policy input or success criteria.',state_scale=scale.tolist(),depth_rmse_scale=.02,failure_nearest=nearest,expert_cross_group_coarse_near_pairs=len(ambiguities),ranked_expert_examples=ranked,no_failure_expert_action_truth=True,not_proof_of_unlearnability=True))
# Final integrity and strictly bounded completion.
for rel,h in read(RUN/'FROZEN.json')['files'].items():assert sha(RUN/rel)==h,rel
for model in read(RUN/'model-binding.json').values():
 for path,h in model['files'].items():assert sha(path)==h,path
for path,h in read(RUN/'source-data-binding.json').items():assert sha(path)==h,path
write(RUN/'ANALYSIS-COMPLETE.json',dict(at=time.time(),forward_claims=len(claims),forward_completed=len(completions),models=3,observations=512,training=0,simulation=0,closed_loop=0,source_inputs_models_unchanged=True,status='offline_budget_complete_no_next_stage'))
print('ANALYSIS COMPLETE; forward budget 1536/1536',flush=True)
