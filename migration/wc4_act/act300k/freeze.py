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
verify()
assert read(RUN/'complete.json')['updates']==300000
trainer_pid=read(RUN/'phase.json')['pid']
proc=Path('/proc')/str(trainer_pid)/'cmdline'
assert not proc.exists() or b'act300k/train.py' not in proc.read_bytes(), 'Trainer has not exited'
journals=list((RUN/'trainer-attempts').glob('*/completed-updates.jsonl'))
assert len(journals)==1
assert [r['completed'] for r in stream(journals[0])]==list(range(200001,300001))
execution_commit=read(RUN/'manifest.json')['start_commit']
checkpoint_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
updates=list(stream(RUN/'updates.jsonl'))
# Interruption rollbacks are explicit separate attempts; the uninterrupted path is exact.
assert [r['update'] for r in updates]==list(range(200001,300001)), 'Review resumed/rollback lineage explicitly'
assert all(np.isfinite([r['loss'],r['lr'],r['grad_norm']]).all() and r['lr']==1e-5 for r in updates)
seen=np.zeros(77356,dtype=np.int64)
for row in updates:
 ids=np.asarray(row['indices'],dtype=np.int64)
 assert len(ids)==row['batch_size_actual'] and np.all((ids>=0)&(ids<77356))
 seen+=np.bincount(ids,minlength=77356)
assert seen.min()>0
write(RUN/'training-audit.json',dict(updates=300000,parent_updates=200000,new_updates=100000,first_update=200001,last_update=300000,finite_all_updates=True,optimizer_lr=1e-5,total_sample_presentations=int(seen.sum()),sampling='author full-dataset uniform shuffle; no phase/category weights or frame exclusions',per_frame_min=int(seen.min()),per_frame_max=int(seen.max()),last_loss=updates[-1]['loss'],continuation=read(RUN/'parent-checkpoint-binding.json'),scheduler=None))
np.savez_compressed(RUN/'uniform-sampling-counts.npz',counts=seen)
checkpoint_audits=[];trial_audits=[]
for stage in STAGES:
 proof=read(RUN/f'stage-audit-{stage:06d}.json')
 assert proof['passed'] and proof['stage']==stage and len(proof['trials'])==16
 assert proof['auditor_source_sha256']==sha(Path(__file__).with_name('audit_stage.py'))
 for name,row in proof['files'].items():
  p=RUN/name;assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],name
 checkpoint_audits.extend(proof['checkpoints']);trial_audits.extend(proof['trials'])
 groups=read(RUN/'train/checkpoints'/f'{stage:06d}'/'training_state/optimizer_param_groups.json')
 assert all(g['lr']==1e-5 for g in groups)
assert len(checkpoint_audits)==5 and len(trial_audits)==80
for row in read(RUN/'input-dataset-binding.json')['files']:
 p=DATA/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
for row in read(RUN/'parent-checkpoint-binding.json')['files']:
 assert sha(PARENT/row['path'])==row['sha256']
shutil.copyfile(DATA/'meta/stats.json',RUN/'dataset-statistics.json')
shutil.copyfile(RUN/'train/checkpoints/300000/pretrained_model/train_config.json',RUN/'final-effective-training-config.json')
shutil.copyfile(RUN/'train/checkpoints/300000/pretrained_model/config.json',RUN/'final-effective-policy-config.json')
write(RUN/'full-checkpoint-audit.json',checkpoint_audits)
write(RUN/'full-evaluation-runtime-audit.json',trial_audits)
all_results=[read(RUN/'evaluation'/f'{s:06d}'/'summary.json') for s in STAGES]
write(RUN/'sixteen-stage-comparison.json',read(PARENT_FINAL/'evidence/eleven-stage-comparison.json')+[dict(step=x['step'],successes=x['successes'],denominator=16,event_counts=x['event_counts'],wandb=read(RUN/'evaluation'/f"{x['step']:06d}"/'wandb-run.json')['url']) for x in all_results])
with (RUN/'per-trial-events.csv').open('w') as f:
 keys=['step','seed','success','max_open_deg','seconds','stop_reason','failure_reasons','opened80','whole_body_pass','legal_relock','contact_separation','arm_fold','heading_restored','stable_hold','leg_posture_deviation']
 out=csv.DictWriter(f,fieldnames=keys);out.writeheader()
 for summary in all_results:
  for r in summary['results']:out.writerow({**{k:r[k] for k in keys if k in r},**r['events'],'failure_reasons':';'.join(r['failures'])})
lines=['ACT从200k续训至累计300k与五节点固定初态评测：冻结交付','',f'执行提交：{execution_commit}',f'冻结提交：{checkpoint_commit}',f'数据：{DATA}',f'repo_id：{REPO}',f'父checkpoint：{PARENT}',f"训练W&B：{read(RUN/'wandb-run.json')['url']}",'','父checkpoint累计200,000次更新；本轮新增100,000次；终点累计300,000次。','原作者全状态恢复，模型、优化器、参数组、步数与RNG逐项核对；学习率始终1e-5，scheduler=None。','checkpoint未包含DataLoader游标，按原作者恢复方式重新开始全量均匀shuffle。','五节点各16回合，共80回合；无新增执行视频或回放。','成功仅按实际开门≥80°且最终合法重锁；其他事件为诊断。','固定seed62300—62315、25Hz、chunk100、执行25步、单回合最多120秒。','','累计更新 | 成功/16 | 事件计数']
for x in read(RUN/'sixteen-stage-comparison.json'):lines.append(str(x['step'])+' | '+str(x['successes'])+'/16 | '+json.dumps(x['event_counts'],ensure_ascii=False))
lines+=['','逐回合详见per-trial-events.csv；与10k—300k完整对照见sixteen-stage-comparison.json。','原始日志、完整checkpoint与导出bundle、有效配置、实际命令、父/数据SHA及W&B日志均冻结在服务器。','本地仅同步轻量审查包，不是完整镜像；实际存放路径与SHA见INDEX.json、SHA256.json。','达到预算后停止，未依结果调整配方、换样或追加回合；交回独立验收。']
(RUN/'RESULT.txt').write_text('\n'.join(lines)+'\n')
# Review tree shares model/data bytes safely; do not duplicate large artifacts on /data.
final=ROOT/'.repro/reviews/wc4-act300k-20261030-FINAL';assert not final.exists();final.mkdir()
exclude={'wandb','__pycache__','freeze-process.log'}
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
for name in subprocess.check_output(['git','ls-files','migration/wc4_act/act300k'],text=True).splitlines():shutil.copyfile(ROOT/name,code/Path(name).name)
shutil.copyfile(RUN/'RESULT.txt',final/'RESULT.txt')
subprocess.run(['git','bundle','create',str(final/'code-lineage.bundle'),'197a169..HEAD'],cwd=ROOT,check=True)
write(final/'INDEX.json',dict(finalizer='migration/wc4_act/act300k/freeze.py',execution_commit=execution_commit,completion_meaning='100000 new optimizer updates from200000 to300000 plus five complete16-trial evaluations; no minimum success threshold',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parent_freeze_commit='b80a9540d66718a1ce12bfbb62cec1ef9999ec36',parent_checkpoint=str(PARENT),parent_updates=200000,new_updates=100000,total_updates=300000,input_commit='197a169133c847e23bbf0f4ea5c389bf1234fae4',dataset=str(DATA),repo_id=REPO,training=str(RUN),local_review_metadata='/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act300k-20261030/FINAL-METADATA',local_is_complete_mirror=False,full_payload_location='server',checkpoints=[str((RUN/'train/checkpoints'/f'{s:06d}')) for s in STAGES],exports=[str(RUN/'checkpoints'/f'{s:06d}'/'door-checkpoint/model_latest.pt') for s in STAGES],complete=True,all_80_runtime_audits_pass=True))
files=[dict(path=str(p.relative_to(final)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(final.rglob('*')) if p.is_file()]
write(final/'SHA256.json',files)
for p in final.rglob('*'):
 if p.is_file():p.chmod(0o444)
for p in sorted([p for p in final.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
final.chmod(0o555)
print(json.dumps(dict(final=str(final),files=len(files),bytes=sum(r['bytes'] for r in files),stages=[dict(step=r['step'],successes=r['successes']) for r in all_results])),flush=True)
