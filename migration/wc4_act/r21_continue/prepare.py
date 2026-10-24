"""Prepare explicit150155→200000 suffix continuation, no models/forward/scenes."""
import sys,os,pathlib,json,hashlib,subprocess,shutil,secrets,time,ast
ROOT=pathlib.Path(__file__).resolve().parents[3];SUPPORT=ROOT/'migration/wc4_act';sys.path[:0]=[str(pathlib.Path(__file__).parent),str(SUPPORT)]
from runtime import read,write,sha,DATA
OLD=ROOT/'.repro/runs/wc4-fulltask-act-r21-r20-200k-s1000-20261024';RUN=ROOT/'.repro/runs/wc4-fulltask-act-r21-cont150155-s1000-20261024';CP=OLD/'stopped-checkpoints/stop-150155'
assert not RUN.exists();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
required=260*1024**3;free=shutil.disk_usage(ROOT).free;assert free>=required,'Remaining16 capacity below260GiB; preserve history/no launch'
assert read(OLD/'STOPPED.json')['completed_updates']==150155 and not read(OLD/'update-boundary.json')['inflight']
for pid in [read(OLD/'process.json')[k] for k in ['trainer','monitor','supervisor']]:assert not pathlib.Path('/proc',str(pid)).exists()
committed=read(CP/'COMMITTED.json');assert committed['step']==150155 and len(committed['files'])==11
for n,h in committed['files'].items():assert sha(CP/n)==h,n
assert sha(CP/'pretrained_model/model.safetensors')=='f9f696c0a93dad2fbec925d82a0a6693dbd41c01792255b17562fac2dadc280e'
assert read(CP/'training_state/training_step.json')['step']==150155
for n,h in read(OLD/'dataset-sha256.json').items():assert sha(DATA/n)==h,n
source=read(OLD/'source-binding.json')
for n,h in source.items():assert sha(ROOT/n)==h,n
for p in pathlib.Path(__file__).parent.iterdir():
 if p.is_file():
  if p.suffix=='.py':ast.parse(p.read_text())
  source[str(p.relative_to(ROOT))]=sha(p)
RUN.mkdir();(RUN/'evaluation-claims').mkdir();write(RUN/'source-binding.json',source)
protected=read(OLD/'protected-checkpoints.json')
for directory in [OLD/'train/checkpoints/100000',OLD/'train/checkpoints/150000',CP]:
 for p in directory.rglob('*'):
  if p.is_file():protected[str(p)]=sha(p)
for n,h in protected.items():assert sha(pathlib.Path(n))==h,n
write(RUN/'protected-checkpoints.json',protected)
for n in ['windows.json','WINDOWS-FROZEN.json','route-reference.json','frozen-r7-physics.yaml','geometry.json','dataset-sha256.json','collection-disjoint.json','SAMPLING-FROZEN.json','SAMPLING-RNG-CONTINUATION.json','sampling-schedule.npz','future-supervision-coverage.json','historical-closing-records.json','baseline-r20-075000.json','baseline-050000.json']:
 shutil.copy2(OLD/n,RUN/n)
for step in [100000,150000]:shutil.copy2(OLD/f'evaluation/{step}/paired-summary.json',RUN/f'baseline-r21-{step}.json')
import numpy as np
assert sha(RUN/'sampling-schedule.npz')=='5ba7dbebf406a02c0c3e3e68e4df73f5c81ab2991b430a8b91ec98f6a65f81d7'
with np.load(RUN/'sampling-schedule.npz') as z:ii=z['indices'];gg=z['groups'];assert ii.shape==(125000,16);tail=ii[75155:];groups=gg[75155:];assert tail.shape==(49845,16)
write(RUN/'SAMPLING-SUFFIX.json',dict(original_run=str(OLD),original_schedule_sha256=sha(RUN/'sampling-schedule.npz'),start_original_zero_based_row=75155,next_global_update=150156,rows=49845,samples=797520,group_counts=np.bincount(groups.ravel(),minlength=5).tolist(),suffix_indices_sha256=hashlib.sha256(tail.tobytes()).hexdigest(),prefix_not_replayed=True,no_rng_reseed_or_redraw=True,original_prefetch_cursor_not_restored=True))
master=read(OLD/'manifest.json');master['source_sha256']=source;write(RUN/'manifest.json',master)
for t in master['trials']:
 d=RUN/'frozen'/str(t['seed']);d.mkdir(parents=True);write(d/'manifest.json',dict(master,trials=[t],seeds=[t['seed']]))
oldindex=read(OLD/'FINAL/server-original-evidence-index-150000.json')['files'];trialbytes=[sum(v['bytes'] for v in oldindex if f'/trial_{i:02d}/' in v['path']) for i in range(16)]
write(RUN/'capacity-estimate.json',dict(at=time.time(),observed150k_trial_bytes=trialbytes,max_observed=max(trialbytes),rounded_trial_GiB=10,remaining_episodes=16,evidence_budget_GiB=200,evidence_formula='16*10GiB*1.25',model_delivery_metadata_GiB=16,ongoing_headroom_GiB=40,calculated_GiB=256,required_start_bytes=required,free_at_prep=free,no_cleanup=True))
contract=dict(task='R21 authorized stop150155 continuation',training_start=150155,training_end=200000,new_optimizer_updates=49845,evaluation_rounds={'200000':16},total_round_budget=16,prior_rounds=32,total_R21_round_budget=48,diagnostic_seeds=list(range(62200,62216)),execution_horizon=25,chunk_size=100,training_n_action_steps=50,data_episodes=72,data_frames=159529,required_start_capacity_bytes=required,ongoing_headroom_bytes=40*1024**3,prior_run=str(OLD),stop='No automatic restart after true stop; no recipe changes,extra rounds or budgets')
write(RUN/'contract.json',contract);wid=secrets.token_hex(4)
files={n:dict(bytes=(CP/n).stat().st_size,sha256=h) for n,h in committed['files'].items()}
order=dict(authorized_by_new_user_task=True,r12_consumed_budget_acknowledged=True,seed=1000,steps=200000,start_update=150155,new_optimizer_updates=49845,resume_checkpoint=str(CP),resume_state_files=files,resume_model_sha256=files['pretrained_model/model.safetensors']['sha256'],wandb_run_id=wid,prior_wandb_run_id='9c85ca7c',evaluation_budget=16,evaluation_rounds=contract['evaluation_rounds'],diagnostic_seeds=contract['diagnostic_seeds'],evaluation_horizon=25,chunk_size=100,author_DataLoader_cursor_not_serialized=True,no_automatic_resume=True,checkpoint_save_freq=50000,required_start_capacity_bytes=required)
write(RUN/'TRAINING_ORDER.json',order)
write(RUN/'prior-budget.json',dict(prior_updates=75155,prior_samples=1202480,prior_rounds=32,old_run=str(OLD),stop_receipt_sha256=sha(OLD/'STOPPED.json'),old_completed_ledger_sha256=sha(OLD/'completed-updates.jsonl'),old_updates_sha256=sha(OLD/'updates.jsonl'),prior_completed_steps=[100000,150000],stop_marker_retained=True))
cmd=[sys.executable,'-B',str(pathlib.Path(__file__).with_name('resume_entry.py')),'--config_path='+str(CP/'pretrained_model/train_config.json'),'--resume=true','--output_dir='+str(RUN/'train'),'--steps=200000','--save_freq=50000','--job_name='+RUN.name,'--wandb.run_id='+wid];write(RUN/'prepared-command.json',cmd)
import lerobot
assert ROOT/'high-level/lerobot/src' in pathlib.Path(lerobot.__file__).parents
write(RUN/'startup-binding.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.executable,lerobot=lerobot.__file__,source=str(CP),source_manifest_checked=11,protected_files=len(protected),no_models_scenes=True))
names=[p for p in RUN.iterdir() if p.is_file() and p.name not in ['source-binding.json','prepared-command.json','startup-binding.json']]+sorted((RUN/'frozen').glob('*/manifest.json'));write(RUN/'frozen-files.json',{str(p.relative_to(RUN)):sha(p) for p in names})
print(json.dumps(dict(run=str(RUN),wandb=wid,remaining_updates=49845,remaining_evaluations=16,protected=len(protected),free_GiB=free/1024**3)))
