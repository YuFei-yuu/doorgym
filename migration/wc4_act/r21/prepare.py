"""R21 completed R20-75k full state + fixed48 + fresh continued sampling; zero models/scenes."""
import sys,subprocess,shutil,time,secrets,ast,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];SUPPORT=ROOT/'migration/wc4_act';sys.path[:0]=[str(Path(__file__).parent),str(SUPPORT/'r13'),str(SUPPORT)]
from runtime import read,write,sha,DATA
BASE=ROOT/'.repro/runs/wc4-fulltask-act-r20-bypass-s1000-20261024';R15=ROOT/'.repro/runs/wc4-fulltask-act-r15-s1000-20261022-cont001';R18=ROOT/'.repro/runs/wc4-fulltask-act-r18-s1000-20261023';RUN=ROOT/'.repro/runs/wc4-fulltask-act-r21-r20-200k-s1000-20261024';checkpoint=BASE/'train/checkpoints/075000'
assert not RUN.exists() and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
free=shutil.disk_usage(ROOT).free;required=300*1024**3;assert free>=required,'48-episode capacity below300GiB; no cleanup or launch'
audit=read(BASE/'checkpoint-audit-075000.json');assert audit['status']=='passed' and audit['optimizer_step_values']==[75000.] and audit['optimizer_step_tensors']==153 and audit['model_sha256']=='0aed867224e411fe22bf23e22b5c5575ebc81534348d4b2909337072aededc86'
for rel,v in audit['state_files'].items():assert sha(checkpoint/rel)==v['sha256'],rel
for rel,h in read(BASE/'dataset-sha256.json').items():assert sha(DATA/rel)==h,rel
assert read(DATA/'meta/info.json')['total_frames']==159529 and read(DATA/'meta/info.json')['total_episodes']==72
assert read(BASE/'sampling-actual-final-audit.json')['actual_new_updates']==25000 and read(BASE/'sampling-actual-final-audit.json')['all_actual_batch_indices_equal_frozen']
source=read(BASE/'source-binding.json')
for rel,h in source.items():assert sha(ROOT/rel)==h,rel
for p in Path(__file__).parent.iterdir():
 if p.is_file():
  if p.suffix=='.py':ast.parse(p.read_text())
  source[str(p.relative_to(ROOT))]=sha(p)
RUN.mkdir();(RUN/'evaluation-claims').mkdir();write(RUN/'source-binding.json',source)
protected=read(BASE/'protected-checkpoints.json')
for rel,v in audit['state_files'].items():protected[str(checkpoint/rel)]=v['sha256']
for p,h in protected.items():assert sha(Path(p))==h,p
write(RUN/'protected-checkpoints.json',protected)
for n in ['windows.json','WINDOWS-FROZEN.json','route-reference.json','frozen-r7-physics.yaml','geometry.json','dataset-sha256.json','collection-disjoint.json']:shutil.copy2(BASE/n,RUN/n)
assert sha(RUN/'windows.json')=='d0cb01abcde64acc13cf870acf547d84cc7fa8429e257edfe6fc1a365c9ebedb'
master=read(BASE/'manifest.json');assert master['seeds']==list(range(62200,62216));master['source_sha256']=source;write(RUN/'manifest.json',master)
for t in master['trials']:
 d=RUN/'frozen'/str(t['seed']);d.mkdir(parents=True);write(d/'manifest.json',dict(master,trials=[t],seeds=[t['seed']]))
for label,p in [('r20-075000',BASE/'evaluation/075000/paired-summary.json'),('r18-075000',R18/'evaluation/075000/paired-summary.json'),('r18-100000',R18/'evaluation/100000/paired-summary.json')]:shutil.copy2(p,RUN/f'baseline-{label}.json')
shutil.copy2(BASE/'baseline-050000.json',RUN/'baseline-050000.json');shutil.copy2(BASE/'FINAL/closing-analysis/all-records.json',RUN/'historical-closing-records.json')
raw=read(BASE/'server-original-evidence-index.json')['files'];trialbytes=[sum(v['bytes'] for k,v in raw.items() if k.startswith(f'evaluation/075000/trial_{i:02d}/')) for i in range(16)]
write(RUN/'capacity-estimate.json',dict(free_bytes_at_prep=free,R20_trial_bytes=trialbytes,max_observed_trial_bytes=max(trialbytes),rounded_trial_budget_bytes=4*1024**3,episodes=48,evidence_budget_bytes=240*1024**3,evidence_formula='48*ceil(max_R20_trial_GiB=3.44→4GiB)*1.25=240GiB',training_exports_delivery_metadata_bytes=16*1024**3,ongoing_headroom_bytes=40*1024**3,calculated_total_bytes=296*1024**3,required_start_bytes=required,no_history_cleanup=True))
contract=dict(task='R21 fixed R20 distribution,75k→200k',training_start=75000,training_end=200000,new_optimizer_updates=125000,evaluation_rounds={'100000':16,'150000':16,'200000':16},total_round_budget=48,diagnostic_seeds=list(range(62200,62216)),execution_horizon=25,chunk_size=100,training_n_action_steps=50,data_episodes=72,data_frames=159529,required_start_capacity_bytes=required,ongoing_headroom_bytes=40*1024**3,stop='No true-stop automatic restart,extra trials,controls rerun or budget increase; ordinary failures count normally',comparisons='R20_75k baseline; at100k R18_uniform100k equal-update reference;150k/200k branch trend only, no nonexistent equal-budget superiority claims')
write(RUN/'contract.json',contract);wid=secrets.token_hex(4)
order=dict(authorized_by_new_user_task=True,r12_consumed_budget_acknowledged=True,seed=1000,steps=200000,start_update=75000,new_optimizer_updates=125000,resume_checkpoint=str(checkpoint),resume_state_files=audit['state_files'],resume_model_sha256=audit['model_sha256'],wandb_run_id=wid,evaluation_budget=48,evaluation_rounds=contract['evaluation_rounds'],diagnostic_seeds=contract['diagnostic_seeds'],evaluation_horizon=25,chunk_size=100,author_DataLoader_cursor_not_serialized=True,no_automatic_resume=True,checkpoint_save_freq=50000,checkpoint_cadence_is_artifact_scheduling_not_optimizer_recipe=True,required_start_capacity_bytes=required)
write(RUN/'TRAINING_ORDER.json',order)
cmd=[sys.executable,'-B',str(Path(__file__).with_name('resume_entry.py')),'--config_path='+str(checkpoint/'pretrained_model/train_config.json'),'--resume=true','--output_dir='+str(RUN/'train'),'--steps=200000','--save_freq=50000','--job_name='+RUN.name,'--wandb.run_id='+wid];write(RUN/'prepared-command.json',cmd)
from sampling import freeze_schedule
freeze_schedule(RUN)
import lerobot
assert ROOT/'high-level/lerobot/src' in Path(lerobot.__file__).parents
write(RUN/'startup-binding.json',dict(at=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.executable,lerobot=lerobot.__file__,source_checkpoint=str(checkpoint),source_model_sha256=audit['model_sha256'],source_full_state_files_checked=11,protected_state_files=len(protected),no_model_forward_optimizer_or_scene=True,free_disk_bytes=shutil.disk_usage(ROOT).free,sampling_distribution_exact_R20=True,artifact_only_save_freq=50000))
names=['TRAINING_ORDER.json','contract.json','capacity-estimate.json','protected-checkpoints.json','windows.json','WINDOWS-FROZEN.json','route-reference.json','frozen-r7-physics.yaml','geometry.json','dataset-sha256.json','collection-disjoint.json','manifest.json','sampling-schedule.npz','SAMPLING-FROZEN.json','SAMPLING-RNG-CONTINUATION.json','future-supervision-coverage.json','baseline-r20-075000.json','baseline-r18-075000.json','baseline-r18-100000.json','baseline-050000.json','historical-closing-records.json']
files=[RUN/n for n in names]+sorted((RUN/'frozen').glob('*/manifest.json'));write(RUN/'frozen-files.json',{str(p.relative_to(RUN)):sha(p) for p in files})
print(__import__('json').dumps(dict(prepared=str(RUN),new_updates=125000,end=200000,evaluation_budget=48,wandb_run_id=wid,required_start_GiB=300,no_model_or_scene=True)),flush=True)
