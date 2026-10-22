"""Freeze points using ONLY expert phases/truth and measured failed-rollout state.
No policy constructed or forwarded; inputs are sealed before any R16 prediction.
"""
import os,sys,time,json,subprocess,platform
import numpy as np
from common import *
from scipy.spatial.transform import Rotation
from lerobot.datasets.factory import make_dataset
import torch,lerobot
assert not torch.cuda.is_available() and not os.environ.get('CUDA_VISIBLE_DEVICES')
RUN.mkdir(exist_ok=False,parents=True);(RUN/'inputs').mkdir();(RUN/'truth').mkdir()
write(RUN/'contract.json',dict(max_observations=512,max_model_observation_forwards=BUDGET,models=[30000,40000,50000],chunk=100,execution_horizon=25,device='cpu',training=0,collection=0,simulation=0,closed_loop=0,repeated_predictions_charged=True,automatic_resume=False,selection_uses_new_predictions=False,expert_points=288,failure_points=224,expert_all_72_episodes=True,selection_rule='Four points/episode: exit turn peak; moving restore-heading closest to final heading; first final_hold (stop switch); middle final_hold. 56 true replan inputs per failure: neighborhoods of measured post-relock heading approach/crossing/continued rotation and measured stopping; uniform measured-tail fill, no action-output selection. Deterministic unique frame tie-breaks.',diagnostic_thresholds=dict(heading_near_deg=2,base_speed_m_s=.01,yaw_rate_rad_s=.01,tail_similarity='descriptive only, not labels or acceptance gates')))
write(RUN/'environment.json',dict(python=sys.executable,version=sys.version,torch=torch.__version__,lerobot=lerobot.__file__,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_available=torch.cuda.is_available(),cpu_threads=2,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),platform=platform.platform()))
source={str(p.relative_to(ROOT)):sha(p) for d in [ROOT/'migration/wc4_act/r16',ROOT/'migration/wc4_act/r13'] for p in d.glob('*.py')}
for p in [ROOT/'high-level/dp/door_policy_backend.py',ROOT/'high-level/lerobot/src/lerobot/policies/act/modeling_act.py',ROOT/'high-level/lerobot/src/lerobot/datasets/video_utils.py',ROOT/'migration/wc4_act/r11/CRITERIA.md']:
 source[str(p.relative_to(ROOT))]=sha(p)
write(RUN/'source-binding.json',source)
models={}
for step,base in MODELS.items():
 official=base/'train/checkpoints'/f'{step:06d}'/'pretrained_model';export=base/'checkpoints'/f'{step:06d}'/'door-checkpoint'
 paths={str(p):sha(p) for p in official.iterdir() if p.is_file()}
 for p in [export/'model_latest.pt',export/'model_latest/door_policy_meta.json']:paths[str(p)]=sha(p)
 expected={30000:'fdd5ad12f657a78dd7d9b483e7b657d9757bff4c829af6671017682543768159',40000:'c6a9d92406cbe0a5614c2b33f961fb48d4b945bfdcf29975f9e727b2e16903de',50000:'76178cc3248521158535923c02ecbe8fd8085e08f47d57b8c7c3852e871293b5'}[step]
 assert sha(official/'model.safetensors')==expected
 models[step]=dict(official=str(official),export=str(export/'model_latest.pt'),files=paths)
write(RUN/'model-binding.json',models)
ds=make_dataset(dataset_config());assert len(ds)==159529
prov=read(R11/'unified-provenance.json')['episodes'];points=[];sources={};offset=0
for ep,e in enumerate(prov):
 p=R11/'candidates'/f'episode_{ep:06d}.npz';assert sha(p)==e['sha256'];sources[str(p)]=e['sha256']
 with np.load(p,allow_pickle=True) as a:
  action=a['action'];state=a['state'];phase=a['subtask_index'].reshape(-1);names=list(a['phase_names']);n=len(action)
  assert action.dtype==state.dtype==np.float32 and n==e['frames']
  q=a['replay_root_state'][:,3:7];yaw=Rotation.from_quat(q).as_euler('xyz')[:,2];target=float(yaw[-1]);err=wrap(yaw-target)
  def ids(names_wanted):return np.flatnonzero(np.isin(phase,[names.index(x) for x in names_wanted if x in names]))
  turn=ids(['compact_exit_turn','compact_exit_arc']);restore=ids(['restore_heading']);hold=ids(['final_hold']);assert len(turn) and len(restore) and len(hold)
  moving=restore[np.abs(action[restore,1])>.01];moving=moving if len(moving) else restore
  choices=[('exit_turn',int(turn[np.argmax(abs(action[turn,1]))])),('heading_restore',int(moving[np.argmin(abs(err[moving]))])),('stop_switch',int(hold[0])),('final_hold',int(hold[len(hold)//2]))]
  assert len(set(f for _,f in choices))==4
  for label,f in choices:
   sample=ds[offset+f];assert np.array_equal(sample['observation.state'].numpy(),state[f]);assert np.array_equal(sample['action'][0].numpy(),action[f])
   data={'state':state[f].copy()}
   image_hash={};raw_decode={}
   for view in ['front','wrist']:
    k=f'observation.images.{view}_masked_depth';x=sample[k].numpy();assert x.shape==(3,480,640) and x.dtype==np.float32
    im=np.rint(x.transpose(1,2,0)*255).astype(np.uint8);assert np.array_equal(im.transpose(2,0,1).astype(np.float32)/255,x)
    data[view]=im;image_hash[view]=ahash(im)
   valid=min(100,n-f);truth=np.zeros((100,9),np.float32);truth[:valid]=action[f:f+valid];mask=np.arange(100)<valid
   assert np.array_equal(sample['action_is_pad'].numpy(),~mask)
   assert np.array_equal(sample['action'].numpy()[:valid],truth[:valid])
   pid=f'expert-e{ep:02d}-f{f:04d}';inp=RUN/'inputs'/(pid+'.npz');np.savez_compressed(inp,**data)
   np.savez_compressed(RUN/'truth'/(pid+'.npz'),action=truth,valid=mask)
   points.append(dict(id=pid,kind='expert',episode=ep,seed=e['seed'],frame=f,index=offset+f,group=label,phase=names[int(phase[f])],valid_future=valid,target_yaw=target,measured_yaw=float(yaw[f]),heading_error_rad=float(err[f]),time_s=f/25,input=str(inp.relative_to(RUN)),input_sha256=sha(inp),state_sha256=ahash(data['state']),images_sha256=image_hash,source=str(p),source_sha256=e['sha256'],truth=str(Path('truth')/(pid+'.npz')),image_source='actual LeRobot factory H264 decode, float32 quantized /255; deployed uint8 preprocessing verified exact',truth_cross_episode=False))
 offset+=n
 print('expert frozen',ep,len(points),flush=True)
assert offset==159529
for seed in SEEDS:
 d=R15/'evaluation/050000'/f'trial_{seed-62200:02d}';ref=read(d/'reference.json');target=ref['old_expert_final_yaw']
 sp=d/'states.jsonl';pp=d/'policy-inputs-actions.jsonl';sources[str(sp)]=sha(sp);sources[str(pp)]=sha(pp)
 states=[]
 with sp.open() as stream:
  for line in stream:
   s=json.loads(line);states.append({k:s[k] for k in ['step','t','actual_base','base_measured_fd_speed','base_measured_fd_yaw_rate','passed','latched','door_q','arm_q']})
 rows=[]
 with pp.open() as stream:
  for line in stream:
   s=json.loads(line);rows.append({k:s[k] for k in ['frame','source_physics_step','source_time_s','state','images_sha256']}|{'new_chunk':s['worker_evidence']['new_chunk'],'chunk_id':s['worker_evidence']['chunk_id']})
 assert all(r['new_chunk']==(r['frame']%25==0) for r in rows)
 # Observation at f is BEFORE physical step 2f; never use row 2f post-state as if it were pre-state.
 pre=[states[max(0,2*f-1)] for f in range(len(rows))];err=np.asarray([wrap(s['actual_base'][2]-target) for s in pre]);omega=np.asarray([s['base_measured_fd_yaw_rate'] for s in pre]);speed=np.asarray([s['base_measured_fd_speed'] for s in pre]);passed=np.asarray([s['passed'] for s in pre]);latched=np.asarray([s['latched'] for s in pre])
 tail=np.flatnonzero(passed&latched);assert len(tail);start=int(tail[0]);near=np.flatnonzero((np.arange(len(rows))>=start)&(abs(err)<=np.deg2rad(2)))
 # Crossings require small wrapped delta, to avoid treating the +/-pi branch cut as a target crossing.
 cross=np.flatnonzero((np.arange(len(err)-1)>=start)&(err[:-1]*err[1:]<0)&(abs(err[:-1]-err[1:])<np.pi))+1
 groups=[]
 for f in near:
  if not groups or f>groups[-1][-1]+1:groups.append([int(f)])
  else:groups[-1].append(int(f))
 anchors=[(start,'relock'),(start+int(np.argmin(abs(err[start:]))),'closest_heading')]
 for g in groups:anchors.append((g[len(g)//2],'heading_near'))
 for f in cross:anchors.append((int(f),'target_crossing'))
 still=(abs(omega)<=.01)&(speed<=.01);stable=[f for f in range(start,len(rows)-50) if np.all(still[f:f+51])]
 if stable:anchors.append((stable[0],'measured_stop'))
 anchors.append((len(rows)-1,'final_state'))
 picked={}
 def add(f,label):
  f=int(np.clip(round(f/25)*25,0,((len(rows)-1)//25)*25))
  if len(picked)<56:picked.setdefault(f,[]).append(label)
 # Cover every central anchor (including final state/stop) before expanding neighborhoods.
 for f,label in anchors:add(f,label)
 for off in [-25,25,-50,50,-100,100]:
  for f,label in anchors:add(f+off,label)
 for f in np.linspace(start,len(rows)-1,200):add(f,'uniform_tail')
 assert len(picked)==56
 write(RUN/'truth'/f'failure-timeline-{seed}.json',dict(seed=seed,target_yaw=target,rows=[dict(frame=f,time_s=r['source_time_s'],heading_error_rad=float(err[f]),actual_base=pre[f]['actual_base'],measured_speed=float(speed[f]),measured_yaw_rate=float(omega[f]),passed=bool(passed[f]),latched=bool(latched[f]),chunk_id=r['chunk_id'],replan=r['new_chunk']) for f,r in enumerate(rows)],near_groups=groups,target_crossing_frames=cross.tolist(),tail_start=start))
 loaded_path=None;depth=None
 for f,labels in sorted(picked.items()):
  r=rows[f];assert r['new_chunk'];src=d/'policy-depth'/f'{(f//100)*100:06d}.npz'
  if src!=loaded_path:
   sources[str(src)]=sha(src)
   with np.load(src) as a:depth={v:a[v].copy() for v in ['front','wrist']}
   loaded_path=src
  data={v:depth[v][f%100].copy() for v in ['front','wrist']};data['state']=np.asarray(r['state'],np.float32)
  assert all(ahash(data[v])==r['images_sha256'][v] for v in ['front','wrist'])
  pid=f'failure-s{seed}-f{f:04d}';inp=RUN/'inputs'/(pid+'.npz');np.savez_compressed(inp,**data)
  points.append(dict(id=pid,kind='failure',seed=seed,frame=f,group='failure',selection_labels=labels,chunk_id=r['chunk_id'],target_yaw=target,measured_yaw=pre[f]['actual_base'][2],heading_error_rad=float(err[f]),time_s=r['source_time_s'],input=str(inp.relative_to(RUN)),input_sha256=sha(inp),state_sha256=ahash(data['state']),images_sha256=r['images_sha256'],source=str(d),replan=True,expert_action_truth_available=False,image_source='original deployed raw uint8 front/wrist, exact archived hash'))
 print('failure frozen',seed,len(points),flush=True)
assert len(points)==512 and len(set(p['id'] for p in points))==512
write(RUN/'source-data-binding.json',sources);write(RUN/'observations.json',dict(frozen_at=time.time(),points=points,point_count=len(points),forwards_per_model=512,total_model_observation_forwards=1536,no_new_prediction_seen=True))
files={str(p.relative_to(RUN)):sha(p) for sub in ['inputs','truth'] for p in (RUN/sub).glob('*')}
for n in ['contract.json','environment.json','source-binding.json','model-binding.json','source-data-binding.json','observations.json']:files[n]=sha(RUN/n)
write(RUN/'FROZEN.json',dict(at=time.time(),files=files,status='sealed_before_any_R16_policy_forward',forward_claims=0));print('FROZEN',sha(RUN/'observations.json'),flush=True)
