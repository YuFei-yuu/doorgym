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
assert read(RUN/'complete.json')['updates']==50000
trainer_pid=read(RUN/'phase.json')['pid']
proc=Path('/proc')/str(trainer_pid)/'cmdline'
assert not proc.exists() or b'act50k/train.py' not in proc.read_bytes(), 'Trainer has not exited'
journals=list((RUN/'trainer-attempts').glob('*/completed-updates.jsonl'))
assert len(journals)==1
assert [r['completed'] for r in stream(journals[0])]==list(range(1,50001))
execution_commit=read(RUN/'manifest.json')['start_commit']
checkpoint_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
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
checkpoint_audits=[];trial_audits=[]
for stage in STAGES:
 proof=read(RUN/f'stage-audit-{stage:06d}.json')
 assert proof['passed'] and proof['stage']==stage and len(proof['trials'])==16
 assert proof['auditor_source_sha256']==sha(RUN/'tooling/audit_stage.py')
 for name,row in proof['files'].items():
  p=RUN/name;assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],name
 checkpoint_audits.extend(proof['checkpoints']);trial_audits.extend(proof['trials'])
 groups=read(RUN/'train/checkpoints'/f'{stage:06d}'/'training_state/optimizer_param_groups.json')
 assert all(g['lr']==1e-5 for g in groups)
assert len(checkpoint_audits)==5 and len(trial_audits)==80
for row in read(RUN/'input-dataset-binding.json')['files']:
 p=DATA/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
assert sha(RUN/'initialization/model.safetensors')==read(RUN/'initialization/provenance.json')['initial_model_sha256']
shutil.copyfile(DATA/'meta/stats.json',RUN/'dataset-statistics.json')
shutil.copyfile(RUN/'train/checkpoints/050000/pretrained_model/train_config.json',RUN/'final-effective-training-config.json')
shutil.copyfile(RUN/'train/checkpoints/050000/pretrained_model/config.json',RUN/'final-effective-policy-config.json')
write(RUN/'full-checkpoint-audit.json',checkpoint_audits)
write(RUN/'full-evaluation-runtime-audit.json',trial_audits)
all_results=[read(RUN/'evaluation'/f'{s:06d}'/'summary.json') for s in STAGES]
write(RUN/'five-stage-comparison.json',[dict(step=x['step'],successes=x['successes'],denominator=16,event_counts=x['event_counts'],wandb=read(RUN/'evaluation'/f"{x['step']:06d}"/'wandb-run.json')['url']) for x in all_results])
with (RUN/'per-trial-events.csv').open('w') as f:
 keys=['step','seed','success','max_open_deg','seconds','stop_reason','failure_reasons','opened80','whole_body_pass','legal_relock','contact_separation','arm_fold','heading_restored','stable_hold','leg_posture_deviation']
 out=csv.DictWriter(f,fieldnames=keys);out.writeheader()
 for summary in all_results:
  for r in summary['results']:out.writerow({**{k:r[k] for k in keys if k in r},**r['events'],'failure_reasons':';'.join(r['failures'])})
lines=['ACT训练50k与五节点固定初态评测：冻结交付','',f'执行提交：{execution_commit}',f'冻结提交：{checkpoint_commit}',f'数据：{DATA}',f'repo_id：{REPO}','输入冻结197a169：50回合／77,356帧；本批统计；新ACT仅使用ResNet18 ImageNet V1骨干初始化。',f"训练W&B：{read(RUN/'wandb-run.json')['url']}",'','新运行完整执行50,000次优化器更新；五节点各16回合，共80回合。','成功仅按同回合实际开门≥80°且最终合法重锁；合法锁闭门控与原物理机制保持。','固定seed62300—62315；与训练来源62100—62149不重叠；五节点实际初态、物理与相机绑定逐回合一致。','控制25Hz，预测chunk100、每次执行25步；每回合最多120秒；无新增执行视频或回放。','','更新节点 | 成功/16 | 开门80° | 全身越过门平面 | 合法重锁 | 接触分离 | 收臂条件 | 朝向条件 | 稳定保持 | 姿态诊断','--- | --- | --- | --- | --- | --- | --- | --- | --- | ---']
for x in all_results:
 c=x['event_counts'];lines.append(f"{x['step']} | {x['successes']}/16 | "+' | '.join(str(c[k]) for k in ['opened80','whole_body_pass','legal_relock','contact_separation','arm_fold','heading_restored','stable_hold','leg_posture_deviation']))
lines+=['','事件按实测条件首次满足记录，具体阈值与范围见SCOPE.txt；事件时间、最大门角、停止原因及失败原因见per-trial-events.csv。','全身通过为URDF全几何越过门平面的事件；收臂与朝向为通过后满足既有目标阈值的条件。','本轮无成功率最低门槛；实际任务失败均保留，未换样、追加预算或续训。','完整checkpoint均保留pretrained_model与training_state；对应导出同时保留manifest和完整bundle目录。','全部80回合的动作队列、原始双路观测、完整状态与本批归一化统计核验通过；与任务成功率分开统计。','原始JSONL已无损gzip；逐项解压SHA回执随回合保留；无用传输包已在核验后清理。','实际命令见train-command.json、launch-binding.json及各回合command文件；W&B链接见各节点wandb-run.json。','授权范围已完成；交回独立验收。']
(RUN/'RESULT.txt').write_text('\n'.join(lines)+'\n')
# Review tree shares model/data bytes safely; do not duplicate large artifacts on /data.
final=ROOT/'.repro/reviews/wc4-act50k-20261029-r2-FINAL';assert not final.exists();final.mkdir()
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
subprocess.run(['git','bundle','create',str(final/'code-lineage.bundle'),'197a169..HEAD'],cwd=ROOT,check=True)
write(final/'INDEX.json',dict(finalizer='migration/wc4_act/act50k/freeze.py',execution_commit=execution_commit,completion_meaning='50000 optimizer updates plus five complete16-trial evaluations; no minimum success threshold',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),input_commit='197a169133c847e23bbf0f4ea5c389bf1234fae4',dataset=str(DATA),repo_id=REPO,training=str(RUN),local_delivery='/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act50k-20261029-r2/FINAL',checkpoints=[str((RUN/'train/checkpoints'/f'{s:06d}')) for s in STAGES],exports=[str(RUN/'checkpoints'/f'{s:06d}'/'door-checkpoint/model_latest.pt') for s in STAGES],complete=True,all_80_runtime_audits_pass=True))
files=[dict(path=str(p.relative_to(final)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(final.rglob('*')) if p.is_file()]
write(final/'SHA256.json',files)
for p in final.rglob('*'):
 if p.is_file():p.chmod(0o444)
for p in sorted([p for p in final.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
final.chmod(0o555)
print(json.dumps(dict(final=str(final),files=len(files),bytes=sum(r['bytes'] for r in files),stages=[dict(step=r['step'],successes=r['successes']) for r in all_results])),flush=True)
