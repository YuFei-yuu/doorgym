"""R15 startup binding only: no model, diagnostic forward, or scene."""
import sys,os,json,subprocess,shutil,secrets,ast,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];SUPPORT=ROOT/'migration/wc4_act'
sys.path[:0]=[str(Path(__file__).parent),str(SUPPORT/'r13'),str(SUPPORT)]
from runtime import read,write,sha,DATA
R14=ROOT/'.repro/runs/wc4-fulltask-act-r14-s1000-20261022'
RUN=Path(os.environ.get('WC4_R15_NEW_RUN',str(ROOT/'.repro/runs/wc4-fulltask-act-r15-s1000-20261022'))).resolve()
assert RUN.parent==ROOT/'.repro/runs' and RUN.name.startswith('wc4-fulltask-act-r15-s1000-20261022')
checkpoint=R14/'continuation-train/train/checkpoints/030000'
assert not RUN.exists() and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
assert shutil.disk_usage(ROOT).free>=120*1024**3,'Need 80GiB evidence reserve plus40GiB stop headroom; preserve history'
audit=read(R14/'continuation-train/checkpoint-audit-030000.json')
assert audit['optimizer_step_values']==[30000.] and audit['status']=='passed'
for rel,v in audit['state_files'].items():assert sha(checkpoint/rel)==v['sha256'],rel
for rel,digest in read(R14/'dataset-sha256.json').items():assert sha(DATA/rel)==digest,rel
assert read(DATA/'meta/info.json')['total_frames']==159529 and read(DATA/'meta/info.json')['total_episodes']==72
source=read(R14/'continuation-train/source-binding.json')
for rel,digest in source.items():assert sha(ROOT/rel)==digest,rel
for p in Path(__file__).parent.iterdir():
 if p.is_file():
  if p.suffix=='.py':ast.parse(p.read_text())
  source[str(p.relative_to(ROOT))]=sha(p)
RUN.mkdir()
write(RUN/'source-binding.json',source)
master=read(R14/'manifest.json');trials=[t for t in master['trials'] if t['seed'] in range(62200,62216)]
assert [t['seed'] for t in trials]==list(range(62200,62216))
master.update(source_sha256=source,trials=trials,seeds=list(range(62200,62216)))
write(RUN/'manifest.json',master)
baseline=read(R14/'combined-summary-030000.json')
baseline['trials']=[t for t in baseline['trials'] if 62200<=t['seed']<=62215]
assert [t['seed'] for t in baseline['trials']]==list(range(62200,62216))
write(RUN/'baseline-030000.json',dict(trials=baseline['trials'],source=str(R14/'combined-summary-030000.json'),source_sha256=sha(R14/'combined-summary-030000.json'),model_sha256=audit['model_sha256']))
for t in trials:
 d=RUN/'frozen'/str(t['seed']);d.mkdir(parents=True)
 old=read(R14/'frozen'/str(t['seed'])/'manifest.json')['trials'][0];assert t==old
 write(d/'manifest.json',dict(master,trials=[t],seeds=[t['seed']]))
for name in ['frozen-r7-physics.yaml','geometry.json','dataset-sha256.json','collection-disjoint.json']:
 shutil.copy2(R14/name,RUN/name)
contract=read(R14/'contract.json');contract.update(task='R15 +20000 to cumulative50000 paired learning trend',diagnostic_seeds=list(range(62200,62216)),evaluation_rounds={'040000':16,'050000':16})
write(RUN/'contract.json',contract)
wid=secrets.token_hex(4)
order=dict(authorized_by_new_user_task=True,r12_consumed_budget_acknowledged=True,seed=1000,steps=50000,start_update=30000,new_optimizer_updates=20000,
 resume_checkpoint=str(checkpoint),resume_state_files=audit['state_files'],wandb_run_id=wid,r12_separate_loss=[1122,1123],evaluation_budget=32,
 evaluation_rounds={'040000':16,'050000':16},diagnostic_seeds=list(range(62200,62216)),paired_R14_records=str(R14/'continuation-train/evaluation/030000'),
 no_new_30000_evaluation=True,no_extra_64=True,author_DataLoader_cursor_not_serialized=True,no_automatic_resume=True,original_rgb_keep_all_32_for_collision_review=True)
write(RUN/'TRAINING_ORDER.json',order)
prior=ROOT/'.repro/runs/wc4-fulltask-act-r15-s1000-20261022'
if RUN!=prior:
 assert read(prior/'STOPPED.json')['completed_updates']==30000 and read(prior/'STOPPED.json')['uncertain_updates']==0
 assert not (prior/'updates.jsonl').read_text() and not list((prior/'evaluation-claims').glob('*.json'))
 order.update(prior_stopped_run=str(prior),prior_R15_new_updates=0,prior_R15_evaluation_invocations=0,
  explicit_user_continuation_authorization='Use fixed version in new continuation directory from original R14 true030000; unchanged20000 updates plus32 episodes; restore then directly train, no additional diagnostics/evaluations')
 write(RUN/'TRAINING_ORDER.json',order)
cfg=read(checkpoint/'pretrained_model/train_config.json')
cmd=[sys.executable,'-B',str(Path(__file__).with_name('resume_entry.py')),'--config_path='+str(checkpoint/'pretrained_model/train_config.json'),
 '--resume=true','--output_dir='+str(RUN/'train'),'--steps=50000','--job_name='+RUN.name,'--wandb.run_id='+wid]
write(RUN/'prepared-command.json',cmd)
import lerobot
assert ROOT/'high-level/lerobot/src' in Path(lerobot.__file__).parents
write(RUN/'startup-binding.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.executable,lerobot=lerobot.__file__,source_checkpoint=str(checkpoint),no_model_optimizer_forward_or_scene=True,frozen16_equal_R14=True,
 free_disk_bytes=shutil.disk_usage(ROOT).free,reserved_evidence_bytes=80*1024**3,stop_headroom_bytes=40*1024**3,original_recipe=cfg))
files=[RUN/'TRAINING_ORDER.json',RUN/'manifest.json',RUN/'baseline-030000.json',RUN/'contract.json',RUN/'dataset-sha256.json',RUN/'frozen-r7-physics.yaml',RUN/'geometry.json',*sorted((RUN/'frozen').glob('*/manifest.json'))]
write(RUN/'frozen-files.json',{str(p.relative_to(RUN)):sha(p) for p in files})
print(json.dumps(dict(prepared=str(RUN),new_updates=20000,evaluation_budget=32,wandb_run_id=wid,no_diagnostic_phase=True)))
