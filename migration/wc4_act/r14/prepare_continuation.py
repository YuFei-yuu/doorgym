"""Latest user explicitly authorizes continuing; retain pause and all spent calls."""
import sys,subprocess,shutil,ast,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];SUPPORT=ROOT/'migration/wc4_act';sys.path.insert(0,str(SUPPORT/'r13'))
from runtime import read,write,sha,validate_checkpoint
root=ROOT/'.repro/runs/wc4-fulltask-act-r14-s1000-20261022';checkpoint=root/'stopped-checkpoints/stop-010000'
assert validate_checkpoint(checkpoint)==10000
assert read(root/'STOPPED.json')['uncertain_updates']==0
assert len(list((root/'evaluation-claims').glob('*.json')))==2
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
source=read(root/'source-binding.json')
for name,digest in source.items():assert sha(ROOT/name)==digest,name
for p in (SUPPORT/'r14').iterdir():
 if p.is_file():
  if p.suffix=='.py':ast.parse(p.read_text())
  source[str(p.relative_to(ROOT))]=sha(p)
master=read(root/'manifest.json');trials=master['trials'];master['source_sha256']=source
for name,role in [('continuation-train','training 10000 to 30000 plus 020000/030000 fixed evaluation'),
 ('continuation-mid010000','six untouched 010000 initial states 62302 to 62307')]:
 run=root/name;run.mkdir(exist_ok=False)
 write(run/'source-binding.json',source);write(run/'manifest.json',master)
 for trial in trials:
  d=run/'frozen'/str(trial['seed']);d.mkdir(parents=True)
  write(d/'manifest.json',dict(master,trials=[trial],seeds=[trial['seed']]))
 write(run/'TRAINING_ORDER.json',dict(authorized_by_new_user_task=True,r12_consumed_budget_acknowledged=True,
  seed=1000,steps=30000,resume_checkpoint=str(checkpoint),start_update=10000,new_optimizer_updates=20000 if name=='continuation-train' else 0,
  authorization='Latest user: robot outside overview is policy failure, continue training; no stop for this condition',
  r12_separate_loss=[1122,1123],role=role,total_R14_evaluation_budget=80,no_retry_of_seed62301=True,
  author_resume_DataLoader_cursor_not_serialized=True))
 for file in ['contract.json','dataset-sha256.json','wandb-run.json']:
  shutil.copyfile(root/file,run/file)
 write(run/'frozen-files.json',{str(p.relative_to(run)):sha(p) for p in [run/'TRAINING_ORDER.json',run/'manifest.json',run/'contract.json',run/'dataset-sha256.json',*sorted((run/'frozen').glob('*/manifest.json'))]})
mid=root/'continuation-mid010000';(mid/'checkpoints').mkdir()
(mid/'checkpoints/010000').symlink_to(root/'checkpoints/010000',target_is_directory=True)
write(root/'USER-CONTINUATION-ORDER.json',dict(latest_instruction='Continue training; leaving overview in an ordinary failed rollout is not an engineering stop',
 prior_pause_preserved=True,update_start=10000,update_stop=30000,new_updates=20000,training_seed=1000,
 remaining_mid_seeds=list(range(62302,62308)),middle_020000=list(range(62300,62308)),final_030000=list(range(62200,62264)),
 prior_calls=2,prior_completed=1,prior_interrupted=1,remaining_calls=78,total_calls=80,
 no_replacement_or_extra_call=True,policy_inputs_physics_criteria_unchanged=True,
 data_cursor_limitation='Original author resume does not serialize DataLoader sampler/prefetch cursor; no exact sample-order continuity claim',
 code_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
print('Explicit continuation prepared: same checkpoint/optimizer/RNG, +20000 updates; 6+8+64 remaining calls; 80 total')
