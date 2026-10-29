"""Read-only completed-stage audit; no model construction or simulation."""
import sys,json,time,gzip,hashlib
from pathlib import Path
sys.path.insert(0,'/data/zyf/video2door-wc4-act/migration/wc4_act/act200k')
from common import *
import numpy as np,torch
sys.path[:0]=[str(SUPPORT),str(ROOT/'high-level/dp')]
from r14 import audit as runtime_audit
stage=int(sys.argv[1]);assert stage in STAGES;STAGES=[stage]
assert read(RUN/'evaluation'/f'{stage:06d}'/'complete.json')['trials']==16

def stream(path):
 path=Path(path)
 f=path.open() if path.exists() else gzip.open(str(path)+'.gz','rt')
 with f:
  for line in f:yield json.loads(line)
runtime_audit.stream=stream
from safetensors.torch import load_file
from door_policy_backend import load_lerobot_policy_normalizer_stats
new_stats=read(DATA/'meta/stats.json');checkpoint_audits=[];trial_audits=[]
for stage in STAGES:
 cp=RUN/'train/checkpoints'/f'{stage:06d}'
 assert read(cp/'training_state/training_step.json')['step']==stage
 model=load_file(str(cp/'pretrained_model/model.safetensors'))
 optimizer=load_file(str(cp/'training_state/optimizer_state.safetensors'))
 steps=[float(v) for n,v in optimizer.items() if n.endswith('/step')]
 assert steps and set(steps)=={float(stage)},set(steps)
 assert all(torch.isfinite(v).all() for v in model.values())
 assert all(torch.isfinite(v).all() for v in optimizer.values())
 rng=load_file(str(cp/'training_state/rng_state.safetensors'));assert 'torch_cuda_rng_state' in rng
 stats=load_lerobot_policy_normalizer_stats(cp/'pretrained_model');compared=0
 for key,values in stats.items():
  for name,value in values.items():
   if name not in ['mean','std']:continue
   expected=torch.tensor(new_stats[key][name],dtype=value.dtype)
   assert torch.equal(expected.reshape(value.shape),value.cpu()),(key,name)
   compared+=1
 checkpoint_audits.append(dict(step=stage,optimizer_step_tensors=len(steps),all_optimizer_steps_equal=stage,model_optimizer_finite=True,full_rng_keys=list(rng),fresh_dataset_stat_tensors_equal=compared,export=read(RUN/'checkpoints'/f'{stage:06d}'/'export-consistency.json')))
 summary=read(RUN/'evaluation'/f'{stage:06d}'/'summary.json');assert len(summary['results'])==16
 for index,r in enumerate(summary['results']):
  trial=Path(r['source']);proof=runtime_audit.verify_journal(trial/'worker-applied-evidence.jsonl')
  assert proof['completed_physics_steps']==r['steps'] and proof['emitted']==(r['steps']+1)//2
  assert proof['new_chunks']==(proof['emitted']+24)//25
  inputs=iter(stream(trial/'policy-inputs-actions.jsonl'));frame=0
  for path in sorted((trial/'policy-depth').glob('*.npz')):
   with np.load(path) as npz:
    data={name:npz[name] for name in ['front','wrist']}
    assert data['front'].shape==data['wrist'].shape and data['front'].dtype==data['wrist'].dtype==np.uint8
    for j in range(len(data['front'])):
     row=next(inputs);assert row['frame']==frame and abs(row['source_time_s']-frame*.04)<.001
     for name in ['front','wrist']:assert hashlib.sha256(data[name][j].tobytes()).hexdigest()==row['images_sha256'][name]
     frame+=1
  assert next(inputs,None) is None and frame==proof['emitted']
  assert read(trial/'paired-initial-binding.json')['all_actual_initial_physics_camera_equal']
  assert read(trial/'worker-model-binding.json')['execution_horizon']==25
  assert not list(trial.rglob('*.mp4'))
  original=[]
  for receipt in read(trial/'lossless-log-compression.json'):
   h=hashlib.sha256()
   with gzip.open(trial/receipt['archive'],'rb') as f:
    for b in iter(lambda:f.read(8388608),b''):h.update(b)
   assert h.hexdigest()==receipt['sha256'] and sha(trial/receipt['archive'])==receipt['archive_sha256']
  trial_audits.append(dict(stage=stage,seed=r['seed'],depth_frames_per_view=frame,depth_observation_hashes_equal=True,paired_actual_state=True,worker_queue=proof,lossless_logs=True))
  print(json.dumps(dict(stage=stage,seed=r['seed'],runtime_depth_audit='passed')),flush=True)

files={}
for base in [RUN/'train/checkpoints'/f'{stage:06d}',RUN/'checkpoints'/f'{stage:06d}',RUN/'evaluation'/f'{stage:06d}']:
 for p in sorted(base.rglob('*')):
  if not p.is_file() or p.is_symlink() or 'wandb' in p.relative_to(base).parts:continue
  files[str(p.relative_to(RUN))]=dict(bytes=p.stat().st_size,sha256=sha(p))
write(RUN/f'stage-audit-{stage:06d}.json',dict(stage=stage,passed=True,checkpoints=checkpoint_audits,trials=trial_audits,files=files,auditor_source_sha256=sha(Path(__file__)),at=time.time(),no_model_constructed=True,no_simulation=True))
print(json.dumps(dict(stage=stage,passed=True,trials=len(trial_audits),files=len(files))),flush=True)
