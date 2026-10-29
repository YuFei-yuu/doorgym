"""Read-only full evidence audit, then hardlink a frozen review tree."""
from common import *
import numpy as np,torch,gzip,subprocess,shutil,hashlib,csv
sys.path[:0]=[str(SUPPORT),str(ROOT/'high-level/dp')]
from r14 import audit as runtime_audit

def stream(path):
 path=Path(path)
 if path.exists():f=path.open()
 else:f=gzip.open(str(path)+'.gz','rt')
 with f:
  for line in f:yield json.loads(line)
runtime_audit.stream=stream
assert read(RUN/'complete.json')['updates']==50000
updates=list(stream(RUN/'updates.jsonl'))
# Interruption rollbacks are explicit separate attempts; the uninterrupted path is exact.
assert [r['update'] for r in updates]==list(range(1,50001)), 'Review resumed/rollback lineage explicitly'
assert all(np.isfinite([r['loss'],r['lr'],r['grad_norm']]).all() and r['lr']==1e-5 for r in updates)
seen=np.zeros(77356,dtype=np.int64)
for row in updates:
 ids=np.asarray(row['indices'],dtype=np.int64)
 assert len(ids)==row['batch_size_actual'] and np.all((ids>=0)&(ids<77356))
 seen+=np.bincount(ids,minlength=77356)
assert seen.min()>0
write(RUN/'training-audit.json',dict(updates=50000,first_update=1,last_update=50000,finite_all_updates=True,optimizer_lr=1e-5,total_sample_presentations=int(seen.sum()),sampling='author full-dataset uniform shuffle; no phase/category weights or frame exclusions',per_frame_min=int(seen.min()),per_frame_max=int(seen.max()),last_loss=updates[-1]['loss'],initialization=read(RUN/'initialization/provenance.json')))
np.savez_compressed(RUN/'uniform-sampling-counts.npz',counts=seen)
from safetensors.torch import load_file
from door_policy_backend import load_lerobot_policy_normalizer_stats
new_stats=read(DATA/'meta/stats.json');checkpoint_audits=[];trial_audits=[]
for stage in STAGES:
 cp=RUN/'train/checkpoints'/f'{stage:06d}'
 assert read(cp/'training_state/training_step.json')['step']==stage
 model=load_file(str(cp/'pretrained_model/model.safetensors'))
 optimizer=load_file(str(cp/'training_state/optimizer_state.safetensors'))
 steps=[float(v) for n,v in optimizer.items() if n.endswith('.step')]
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
   with np.load(path) as data:
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
write(RUN/'full-checkpoint-audit.json',checkpoint_audits)
write(RUN/'full-evaluation-runtime-audit.json',trial_audits)
all_results=[read(RUN/'evaluation'/f'{s:06d}'/'summary.json') for s in STAGES]
write(RUN/'five-stage-comparison.json',[dict(step=x['step'],successes=x['successes'],denominator=16,event_counts=x['event_counts'],wandb=read(RUN/'evaluation'/f"{x['step']:06d}"/'wandb-run.json')['url']) for x in all_results])
with (RUN/'per-trial-events.csv').open('w') as f:
 keys=['step','seed','success','max_open_deg','seconds','stop_reason','opened80','whole_body_pass','legal_relock','contact_separation','arm_fold','heading_restored','stable_hold','leg_posture_deviation']
 out=csv.DictWriter(f,fieldnames=keys);out.writeheader()
 for summary in all_results:
  for r in summary['results']:out.writerow({**{k:r[k] for k in keys if k in r},**r['events']})
lines=['ACT 50k paired evaluation — frozen delivery','',f"Code: {subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}",f'Data: {DATA}',f'Repo ID: {REPO}','Dataset freeze: 197a169; 50 episodes / 77,356 frames.',f"Training W&B: {read(RUN/'wandb-run.json')['url']}",'','Criterion: actual opening >=80 degrees AND final legal relock. Other events do not veto success.','Legal lock: unchanged fullcycle_task_monitor.TaskMonitor; measured-angle latch, no zero snap.','Fixed evaluation seeds 62300–62315; one unfiltered original-author draw per seed; paired actual states/cameras checked across all stages.','No execution video or replay; exact observed depth buffers and original physics/action logs retained.','','Step | Success / 16 | Open80 | Passage | Relock | Separation | Fold | Heading | Stable | Leg diagnostic','--- | --- | --- | --- | --- | --- | --- | --- | --- | ---']
for x in all_results:
 c=x['event_counts'];lines.append(f"{x['step']} | {x['successes']}/16 | "+' | '.join(str(c[k]) for k in ['opened80','whole_body_pass','legal_relock','contact_separation','arm_fold','heading_restored','stable_hold','leg_posture_deviation']))
lines+=['','See per-trial-events.csv and each event-audit.json for actual stop reasons and event times.','No minimum success threshold; no failed tasks replaced, no budget extended.','Logs ending in .jsonl.gz are lossless; individual original/decoded SHA receipts are next to each trial.','Complete original-author checkpoints include pretrained_model and training_state; exports include manifest AND model_latest bundle directory.','Initializer is a new ACT with ImageNet V1 backbone only; new dataset normalization, no old-task policy weights.','No uncompleted authorized work remains after this audit passes.']
(RUN/'RESULT.txt').write_text('\n'.join(lines)+'\n')
# Review tree shares model/data bytes safely; do not duplicate large artifacts on /data.
final=ROOT/'.repro/reviews/wc4-act50k-20261029-FINAL';assert not final.exists();final.mkdir()
exclude={'wandb','__pycache__'}
def linktree(src,dst):
 dst.mkdir(exist_ok=True)
 for item in sorted(src.iterdir()):
  if item.name in exclude:continue
  target=dst/item.name
  if item.is_symlink():continue
  if item.is_dir():linktree(item,target)
  elif item.is_file():os.link(item,target)
linktree(RUN,final/'evidence')
# W&B original run files/logs remain on server; retain all of them losslessly in a single package.
wandb_dirs=[p for p in RUN.rglob('wandb') if p.is_dir()]
if wandb_dirs:
 import tarfile
 with tarfile.open(final/'wandb-logs.tar.gz','w:gz') as archive:
  for d in wandb_dirs:archive.add(d,arcname=str(d.relative_to(RUN)),recursive=True)
code=final/'code';code.mkdir()
for name in subprocess.check_output(['git','ls-files','migration/wc4_act/act50k'],text=True).splitlines():shutil.copyfile(ROOT/name,code/Path(name).name)
shutil.copyfile(RUN/'RESULT.txt',final/'RESULT.txt')
write(final/'INDEX.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),input_commit='197a169133c847e23bbf0f4ea5c389bf1234fae4',dataset=str(DATA),repo_id=REPO,training=str(RUN),local_delivery='/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act50k-20261029/FINAL',checkpoints=[str((RUN/'train/checkpoints'/f'{s:06d}')) for s in STAGES],exports=[str(RUN/'checkpoints'/f'{s:06d}'/'door-checkpoint/model_latest.pt') for s in STAGES],complete=True,all_80_runtime_audits_pass=True))
files=[dict(path=str(p.relative_to(final)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(final.rglob('*')) if p.is_file()]
write(final/'SHA256.json',files)
for p in final.rglob('*'):
 if p.is_file():p.chmod(0o444)
final.chmod(0o555)
print(json.dumps(dict(final=str(final),files=len(files),bytes=sum(r['bytes'] for r in files),stages=[dict(step=r['step'],successes=r['successes']) for r in all_results])),flush=True)
