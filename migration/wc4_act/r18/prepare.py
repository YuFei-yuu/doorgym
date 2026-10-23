"""R18 source/data/full50k-state/paired16 freeze; no policy forward or scene."""
import sys,os,json,subprocess,shutil,secrets,ast,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];SUPPORT=ROOT/'migration/wc4_act'
sys.path[:0]=[str(Path(__file__).parent),str(SUPPORT/'r13'),str(SUPPORT)]
from runtime import read,write,sha,DATA
BASE=ROOT/'.repro/runs/wc4-fulltask-act-r15-s1000-20261022-cont001'
RUN=ROOT/'.repro/runs/wc4-fulltask-act-r18-s1000-20261023'
checkpoint=BASE/'train/checkpoints/050000'
assert not RUN.exists() and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
assert shutil.disk_usage(ROOT).free>=120*1024**3,'80GiB evidence reserve plus40GiB headroom; no historical cleanup'
audit=read(BASE/'checkpoint-audit-050000.json');assert audit['status']=='passed' and audit['optimizer_step_values']==[50000.] and audit['optimizer_step_tensors']==153
assert audit['model_sha256']=='76178cc3248521158535923c02ecbe8fd8085e08f47d57b8c7c3852e871293b5'
for rel,v in audit['state_files'].items():assert sha(checkpoint/rel)==v['sha256'],rel
for rel,h in read(BASE/'dataset-sha256.json').items():assert sha(DATA/rel)==h,rel
info=read(DATA/'meta/info.json');assert info['total_frames']==159529 and info['total_episodes']==72
source=read(BASE/'source-binding.json')
for rel,h in source.items():assert sha(ROOT/rel)==h,rel
for p in Path(__file__).parent.iterdir():
 if p.is_file():
  if p.suffix=='.py':ast.parse(p.read_text())
  source[str(p.relative_to(ROOT))]=sha(p)
RUN.mkdir();(RUN/'evaluation-claims').mkdir()
write(RUN/'source-binding.json',source)
master=read(BASE/'manifest.json');trials=[t for t in master['trials'] if 62200<=t['seed']<=62215];assert [t['seed'] for t in trials]==list(range(62200,62216))
master.update(source_sha256=source,trials=trials,seeds=list(range(62200,62216)));write(RUN/'manifest.json',master)
baseline=[]
for i,t in enumerate(trials):
 d=RUN/'frozen'/str(t['seed']);d.mkdir(parents=True)
 prior=read(BASE/'frozen'/str(t['seed'])/'manifest.json')['trials'][0];assert t==prior
 write(d/'manifest.json',dict(master,trials=[t],seeds=[t['seed']]))
 original=BASE/'evaluation/050000'/f'trial_{i:02d}';assert read(original/'frozen-trial.json')==t
 r=read(original/'paired-diagnostics.json');r.update(original_directory=str(original),full_task_success=read(original/'evaluation-result.json')['full_task_success'],frozen_files={n:sha(original/n) for n in ['initial-state.json','physical-camera-binding.json','frozen-trial.json','r11-score.json','evaluation-result.json','paired-diagnostics.json']});baseline.append(r)
write(RUN/'baseline-050000.json',dict(model_step=50000,execution_horizon=25,model_sha256=audit['model_sha256'],records=baseline,no_baseline_rerun=True))
for n in ['frozen-r7-physics.yaml','geometry.json','dataset-sha256.json','collection-disjoint.json']:shutil.copy2(BASE/n,RUN/n)
contract=read(BASE/'contract.json');contract.update(task='R18 original recipe continuation50k to100k',diagnostic_seeds=list(range(62200,62216)),evaluation_rounds={'075000':16,'100000':16},execution_horizon=25,chunk_size=100,training_n_action_steps=50,training_start=50000,training_end=100000,new_optimizer_updates=50000)
write(RUN/'contract.json',contract)
wid=secrets.token_hex(4)
order=dict(authorized_by_new_user_task=True,r12_consumed_budget_acknowledged=True,seed=1000,steps=100000,start_update=50000,new_optimizer_updates=50000,resume_checkpoint=str(checkpoint),resume_state_files=audit['state_files'],wandb_run_id=wid,r12_separate_loss=[1122,1123],evaluation_budget=32,evaluation_rounds={'075000':16,'100000':16},diagnostic_seeds=list(range(62200,62216)),evaluation_horizon=25,chunk_size=100,paired_R15_records=str(BASE/'evaluation/050000'),no_new_50000_evaluation=True,no_extra_64=True,author_DataLoader_cursor_not_serialized=True,no_automatic_resume=True,original_rgb_keep_all_32_for_candidate_and_anomaly_review=True,checkpoint_save_freq=25000,checkpoint_cadence_is_artifact_scheduling_not_optimizer_recipe=True)
write(RUN/'TRAINING_ORDER.json',order)
cfg=read(checkpoint/'pretrained_model/train_config.json');assert cfg['dataloader_seed'] is None
cmd=[sys.executable,'-B',str(Path(__file__).with_name('resume_entry.py')),'--config_path='+str(checkpoint/'pretrained_model/train_config.json'),'--resume=true','--output_dir='+str(RUN/'train'),'--steps=100000','--save_freq=25000','--job_name='+RUN.name,'--wandb.run_id='+wid]
write(RUN/'prepared-command.json',cmd)
import lerobot
assert ROOT/'high-level/lerobot/src' in Path(lerobot.__file__).parents
write(RUN/'startup-binding.json',dict(at=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.executable,lerobot=lerobot.__file__,source_checkpoint=str(checkpoint),source_state_files_checked=len(audit['state_files']),no_model_forward_optimizer_or_scene=True,frozen16_equal_R15=True,free_disk_bytes=shutil.disk_usage(ROOT).free,reserved_evidence_bytes=80*1024**3,stop_headroom_bytes=40*1024**3,original_recipe=cfg,only_scheduling_changes=['steps100000','save_freq25000','new output and W&B id'],horizon25_not_R17_h5=True))
files=[RUN/n for n in ['TRAINING_ORDER.json','manifest.json','baseline-050000.json','contract.json','dataset-sha256.json','frozen-r7-physics.yaml','geometry.json','collection-disjoint.json']]+sorted((RUN/'frozen').glob('*/manifest.json'))
write(RUN/'frozen-files.json',{str(p.relative_to(RUN)):sha(p) for p in files})
print(json.dumps(dict(prepared=str(RUN),new_updates=50000,evaluation_budget=32,execution_horizon=25,wandb_run_id=wid,no_model_or_scene=True)),flush=True)
