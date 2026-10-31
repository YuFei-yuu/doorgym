"""Final source/optimizer/runtime audit and immutable100-data200k evidence."""
from common import *
import numpy as np,torch,gzip,subprocess,shutil,hashlib,csv,tarfile,time
verify()
assert read(RUN/'complete.json')['updates']==200000
assert read(RUN/'monitor-state.json')['state']=='stopped_completed'
assert all(read(RUN/f'stage-audit-{s:06d}.json')['passed'] for s in STAGES)
def stream(p):
 p=Path(p)
 with (p.open() if p.exists() else gzip.open(str(p)+'.gz','rt')) as f:
  for line in f:yield json.loads(line)
updates=list(stream(RUN/'updates.jsonl'));assert [r['update'] for r in updates]==list(range(1,200001))
journal=[]
for p in sorted((RUN/'trainer-attempts').glob('*/completed-updates.jsonl')):journal.extend(r['completed'] for r in stream(p))
assert journal==list(range(1,200001)), 'No duplicate, replayed or uncertain budgeted updates accepted'
assert all(np.isfinite([r['loss'],r['lr'],r['grad_norm']]).all() and r['lr']==1e-5 for r in updates)
seen=np.zeros(FRAMES,np.int64)
for r in updates:
 ids=np.asarray(r['indices'],np.int64);assert len(ids)==r['batch_size_actual'] and np.all((ids>=0)&(ids<FRAMES));np.add.at(seen,ids,1)
assert seen.min()>0
assert sha(RUN/'initialization/model.safetensors')==read(RUN/'initialization/provenance.json')['initial_model_sha256']
assert not read(RUN/'initialization/provenance.json')['old_task_weights_loaded']
for r in read(RUN/'input-dataset-binding.json')['files']:assert sha(DATA/r['path'])==r['sha256']
np.savez_compressed(RUN/'uniform-sampling-counts.npz',counts=seen)
write(RUN/'training-audit.json',dict(updates=200000,first_update=1,last_update=200000,actual_completed_update_journals_exact=True,
 finite_all_updates=True,optimizer_lr=1e-5,scheduler=None,total_sample_presentations=int(seen.sum()),frames=FRAMES,episodes=100,
 sampling='Original full-dataset uniform shuffle, no phase/category weights or frame exclusions',per_frame_min=int(seen.min()),per_frame_max=int(seen.max()),
 last_loss=updates[-1]['loss'],initialization=read(RUN/'initialization/provenance.json'),attempts=len(list((RUN/'trainer-attempts').iterdir())),dataloader_cursor_restored=False))
cp_audits=[];trial_audits=[];summaries=[]
for step in STAGES:
 audit=read(RUN/f'stage-audit-{step:06d}.json');assert audit['passed'] and len(audit['trials'])==16
 assert audit['auditor_source_sha256']==sha(Path(__file__).with_name('audit_stage.py'))
 for rel,v in audit['files'].items():p=RUN/rel;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256']
 cp_audits.extend(audit['checkpoints']);trial_audits.extend(audit['trials'])
 groups=read(RUN/'train/checkpoints'/f'{step:06d}'/'training_state/optimizer_param_groups.json');assert all(g['lr']==1e-5 for g in groups)
 summary=read(RUN/'evaluation'/f'{step:06d}'/'summary.json');assert len(summary['results'])==16
 summaries.append(summary)
assert len(cp_audits)==10 and len(trial_audits)==160
write(RUN/'full-checkpoint-audit.json',cp_audits);write(RUN/'full-evaluation-runtime-audit.json',trial_audits)
shutil.copyfile(RUN/'train/checkpoints/200000/pretrained_model/train_config.json',RUN/'final-effective-training-config.json')
shutil.copyfile(RUN/'train/checkpoints/200000/pretrained_model/config.json',RUN/'final-effective-policy-config.json')
old=read(RUN/'old50-stage-comparison.json');old_by_step={r['step']:r for r in old}
compare=[]
for s in summaries:
 prior=old_by_step.get(s['step'])
 compare.append(dict(step=s['step'],new100_successes=s['successes'],denominator=16,new100_events=s['event_counts'],new100_training_relation='fresh0to200k;100 episodes/154667frames',
  old50_successes=prior['successes'] if prior else None,old50_events=prior['event_counts'] if prior else None,
  old50_training_relation='fresh0to50k then full-state continuation50kto200k;50episodes/77356frames',old50_node_available=prior is not None,
  fixed16_numeric_initials_reused=True,causal_improvement_claim=False,new100_wandb=read(RUN/'evaluation'/f"{s['step']:06d}"/'wandb-run.json')['url']))
write(RUN/'ten-stage-comparison.json',compare)
with (RUN/'per-trial-events.csv').open('w') as f:
 keys=['step','seed','success','max_open_deg','seconds','stop_reason','failure_reasons','opened80','whole_body_pass','legal_relock','contact_separation','arm_fold','heading_restored','stable_hold','leg_posture_deviation']
 writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
 for s in summaries:
  for r in s['results']:writer.writerow({**{k:r[k] for k in keys if k in r},**r['events'],'failure_reasons':';'.join(r['failures'])})
with (RUN/'node-comparison.csv').open('w') as f:
 keys=['step','new100_successes','old50_successes','denominator','old50_node_available','new100_training_relation','old50_training_relation','new100_wandb']
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in compare)
text=['100条数据ACT从头200k及十节点固定16评测：冻结交付','',f"实际执行提交：{read(RUN/'manifest.json')['start_commit']}",
 f'输入冻结426feb7；repo_id={REPO}；数据100条/154667帧，统计SHA1fdfb23cfac8591bf328490ec28c88e8e667683b34d3cb16ac0c043191531db7。',
 '从0初始化ACT，未加载旧任务策略，仅ResNet18 ImageNet V1初始化，实际初始模型和骨干张量映射/SHA保留。',
 'seed1000、batch16、lr1e-5、chunk100、训练n_action_steps50、3读取worker、原作者全量均匀shuffle；scheduler=None。',
 '完成200000次优化器更新，十个完整checkpoint和训练状态/处理器/RNG及对应导出均保留。',
 '十节点各16回合，共160回合；25Hz、预测100执行25、每回合最多120秒，固定62300—62315原实际初态无重抽。',
 '成功只按同回合实际开门>=80度且最终合法重锁；穿越、分离、收臂、朝向、保持及姿态诊断单列事件。',
 '原始评测状态/接触/锁写/动作队列与双深度观测保留；JSONL逐项gzip无损回读SHA验证。无新增执行录像/回放。',
 '中断只从本轮最新一致完整状态恢复，已知完成更新总账严格1..200000，无静默回滚重做；原作者checkpoint无DataLoader游标，不声称跨进程位级复演。',
 f"训练W&B：{read(RUN/'wandb-run.json')['url']}",'','节点 | 新100条成功/16 | 旧50条成功/16 | 新事件计数']
for r in compare:text.append(str(r['step'])+' | '+str(r['new100_successes'])+'/16 | '+(str(r['old50_successes'])+'/16' if r['old50_node_available'] else '无对应节点')+' | '+json.dumps(r['new100_events'],ensure_ascii=False))
text+=['','旧50条为原0→50k后续训到200k，新100条为独立0→200k，不加载旧任务模型；仅作固定16初态描述性比较，不将数据规模差异称为继续旧训练。',
 '逐回合事件详见per-trial-events.csv；节点表node-comparison.csv/ten-stage-comparison.json。',
 f'完整服务器成果：{ROOT}/.repro/reviews/wc4-act100-fresh200k-20261031-FINAL',
 '本地默认轻量审查子集，具体下载文件及远端完整字节位置由LOCAL_CONTENTS.json说明；不能将本地索引当作完整模型/原始日志。',
 '实际命令、有效配置、来源及数据SHA、初始化证据、W&B原始日志与所有训练/评测原始指标均冻结。',
 '运行监控在发布前已停止，不自动扩量。未改旧数据/模型/main/环境，交回独立验收。']
(RUN/'RESULT.txt').write_text('\n'.join(text)+'\n')
final=ROOT/'.repro/reviews/wc4-act100-fresh200k-20261031-FINAL';final.mkdir(exist_ok=False)
exclude={'wandb','__pycache__','cache','tmp'}
def linktree(src,dst):
 dst.mkdir(exist_ok=True)
 for p in sorted(src.iterdir()):
  if p.name in exclude or p.is_symlink():continue
  q=dst/p.name
  if p.is_dir():linktree(p,q)
  elif p.is_file():os.link(p,q)
linktree(RUN,final/'evidence')
# Control continues to receive only the final process exit receipt; copy stable
# completed launch/training logs before publishing, never hardlink live files.
ctl=RUN.with_name(RUN.name+'-control');shutil.copytree(ctl,final/'control',ignore=shutil.ignore_patterns('freeze.log','driver.log','supervisor.lock','driver.exit','driver.end'))
with tarfile.open(final/'wandb-logs.tar.gz','w:gz') as t:
 for p in RUN.rglob('wandb'):
  if p.is_dir():t.add(p,arcname=str(p.relative_to(RUN)),recursive=True)
code=final/'code';code.mkdir()
for rel in subprocess.check_output(['git','ls-files','migration/wc4_act/act100_200k'],cwd=ROOT,text=True).splitlines():shutil.copyfile(ROOT/rel,code/Path(rel).name)
shutil.copyfile(RUN/'RESULT.txt',final/'RESULT.txt')
subprocess.run(['git','bundle','create',str(final/'code-lineage.bundle'),'426feb7..HEAD'],cwd=ROOT,check=True)
write(final/'INDEX.json',dict(execution_commit=read(RUN/'manifest.json')['start_commit'],input_freeze='426feb7473e2bc08bbe2d3754a0f75f09dd9bf45',dataset=str(DATA),repo_id=REPO,
 training=str(RUN),episodes=100,frames=154667,initialization='fresh; ImageNet ResNet18 only',updates=200000,evaluation_trials=160,stages=STAGES,
 completion_meaning='exact200000 optimizer updates plus10 completed16-trial evaluations and all runtime/export audits; no minimum success rate threshold',
 full_checkpoints=[str(RUN/'train/checkpoints'/f'{s:06d}') for s in STAGES],exports=[str(RUN/'checkpoints'/f'{s:06d}'/'door-checkpoint/model_latest.pt') for s in STAGES],
 monitoring=read(RUN/'monitor-state.json'),local_delivery='/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act100-fresh200k-20261031',local_scope='review subset; consult LOCAL_CONTENTS.json',main=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()))
files=[dict(path=str(p.relative_to(final)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(final.rglob('*')) if p.is_file()]
write(final/'SHA256.json',files)
for p in final.rglob('*'):
 if p.is_file():p.chmod(0o444)
for p in sorted([p for p in final.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
final.chmod(0o555)
print(json.dumps(dict(final=str(final),files=len(files)+1,sha256_manifest=sha(final/'SHA256.json'),stages=[dict(step=s['step'],successes=s['successes']) for s in summaries]),ensure_ascii=False),flush=True)
