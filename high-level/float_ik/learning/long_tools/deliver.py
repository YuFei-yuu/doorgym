"""CPU-only formal experiment report, early-loop figures and artifact hashes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from resume_state import patience_state

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--local-root',required=True);a=p.parse_args()
r, local = Path(a.root).resolve(), Path(a.local_root)
plots=r/'plots';plots.mkdir(exist_ok=True)

def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def save(fig,name):
    fig.tight_layout();fig.savefig(plots/(name+'.png'),dpi=180);fig.savefig(plots/(name+'.pdf'));plt.close(fig)

def interval(k,n):
    if not n:return None
    z=1.96;q=k/n;d=1+z*z/n;c=(q+z*z/(2*n))/d;h=z*math.sqrt(q*(1-q)/n+z*z/(4*n*n))/d
    return [max(0,c-h),min(1,c+h)]

status=load(r/'status.json',{})
resume=load(r/'resume_state.json',{})
selection=load(r/'selection.json',{})
config=load(r/'config.json',{})
ledger=rows(r/'budget.jsonl')
train=rows(r/'training/train.jsonl')
points=[load(x) for x in sorted(r.glob('validation_point_*.json'))]
if points:
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    xx=[x['step'] for x in points]
    axs[0].plot(xx,[100*x['successes']/x['attempts'] for x in points],'-o')
    axs[0].set(xlabel='Optimizer step',ylabel='Final task success (%)',ylim=(-3,103))
    axs[1].plot(xx,[x['mean_final_angle_deg'] for x in points],'-o')
    axs[1].set(xlabel='Optimizer step',ylabel='Mean final signed opening (deg)')
    fig.suptitle('Complete fixed-seed validation; stopping metrics')
    save(fig,'validation_stopping_metrics')
audit=load(r/'audit_early.json',load(r/'audit_data.json',{}))
video=load(r/'video_checks.json',[])
episodes=[]
for category in ('diagnostic','validation','test','videos'):
    for path in sorted((r/category).glob('*/summary.json')):
        s=load(path);s['category']=category;s['directory']=str(path.parent.relative_to(r));episodes.append(s)

# Preserve unsmoothed channel values used by all early-loop plots.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from schema import STATE_NAMES, ACTION_NAMES
early_rows=[]
for path in sorted((r/'diagnostic').glob('*/early_diagnostic.npz')):
    with np.load(path) as z:
        for i in range(len(z['state'])):
            row=dict(episode=path.parent.name,frame=int(z['frame'][i]),sim_time_sec=i/60,decision=bool(z['decision'][i]),decision_sec=float(z['decision_sec'][i]))
            for prefix,key,names in [('state','state',STATE_NAMES),('normalized','normalized_state',STATE_NAMES),('raw','raw_action',ACTION_NAMES),('applied','action',ACTION_NAMES),('clipped','clipped_channels',ACTION_NAMES)]:
                if key in z:
                    row.update({prefix+'_'+name:float(z[key][i,j]) for j,name in enumerate(names)})
            early_rows.append(row)
if early_rows:
    fields=sorted(set().union(*(x.keys() for x in early_rows)))
    with (r/'early_diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(early_rows)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for path in sorted((r/'diagnostic').glob('policy*/early_diagnostic.npz')):
        with np.load(path) as z:
            t=np.arange(len(z['state']))/60
            label=path.parent.name.replace('policy_5000_','')
            values=[z['clipped_channels'].mean(1)*100,np.max(abs(z['normalized_state']),1),np.linalg.norm(z['state'][:,14:21],axis=1),np.linalg.norm(z['raw_action']-z['action'],axis=1)]
            for ax,value in zip(axs.flat,values):ax.plot(t,value,label=label,lw=1)
    for ax,title,unit in zip(axs.flat,['Clipped action channels','Largest absolute input z-score','Joint velocity norm','Raw minus applied action'],['%','standard deviations','rad/s','mixed units']):
        ax.set(xlabel='Simulation time (s)',ylabel=unit,title=title);ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7,ncol=2);save(fig,'early_closed_loop_diagnostics')

continuation_early=[]
fig,axs=plt.subplots(2,2,figsize=(12,8))
for path in sorted((r/'validation').glob('*/early_diagnostic.npz')):
    step=int(path.parent.name.split('_')[1]);seed=int(path.parent.name.rsplit('_',1)[1])
    with np.load(path) as z:
        clip=z['clipped_channels'].mean(1)
        hit=np.flatnonzero(clip>=.5)
        ood=(z['state']<np.asarray(audit['state_min'])-1e-3)|(z['state']>np.asarray(audit['state_max'])+1e-3)
        outside=np.flatnonzero(ood.any(1))
        continuation_early.append(dict(step=step,seed=seed,recorded_steps=len(clip),
                                       first_half_clipped_s=float(hit[0]/60) if len(hit) else None,
                                       first_outside_training_range_s=float(outside[0]/60) if len(outside) else None,
                                       mean_first_second_clipped_fraction=float(clip.mean()),
                                       max_abs_z=float(abs(z['normalized_state']).max())))
        if seed==4000:
            t=np.arange(len(clip))/60
            for ax,value in zip(axs.flat,[clip*100,np.max(abs(z['normalized_state']),axis=1),np.linalg.norm(z['state'][:,14:21],axis=1),np.linalg.norm(z['raw_action']-z['action'],axis=1)]):
                ax.plot(t,value,label='step %d'%step,lw=1)
if continuation_early:
    with (r/'continuation_early.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(continuation_early[0]));w.writeheader();w.writerows(continuation_early)
    for ax,title,unit in zip(axs.flat,['Clipped channels, seed 4000','Largest input z-score, seed 4000','Joint velocity norm, seed 4000','Raw minus applied, seed 4000'],['%','standard deviations','rad/s','mixed units']):
        ax.set(xlabel='Simulation time (s)',ylabel=unit,title=title);ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8);save(fig,'continuation_early_comparison')
else:plt.close(fig)

groups=[]
test_plan=load(r/'test_plan.json',{})
planned=test_plan.get('attempts',[])
test_by_name={Path(ep['directory']).name:ep for ep in episodes if ep['category']=='test'}
test_attempts=[]
for item in planned:
    ep=test_by_name.get(f'{item["method"]}_seed_{item["seed"]}')
    test_attempts.append(dict(method=item['method'],seed=item['seed'],status=ep['status'] if ep else 'not_started',
        success=ep.get('success') if ep else None,steps_completed=ep.get('steps_completed') if ep else None,
        failure_type=ep.get('failure_type') if ep else None,directory=ep['directory'] if ep else None))
(r/'planned_test_attempts.json').write_text(json.dumps(test_attempts,indent=2))
if test_attempts:
    with (r/'planned_test_attempts.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(test_attempts[0]));w.writeheader();w.writerows(test_attempts)
for method in dict.fromkeys(x['method'] for x in planned):
    ee=[x for x in episodes if x['category']=='test' and Path(x['directory']).name.startswith(method+'_seed_')]
    n=len(ee);k=sum(bool(x.get('success')) for x in ee)
    def mean(key):
        values=[x[key] for x in ee if x.get(key) is not None]
        return statistics.mean(values) if values else None
    clip=[];infer=[];errors=[]
    for ep in ee:
        path=r/ep['directory']
        if (path/'metrics.csv').exists():
            with (path/'metrics.csv').open() as f:
                mm=list(csv.DictReader(f))
            clip.extend(float(x['clipped_fraction']) for x in mm)
            errors.extend(float(x['ee_to_handle_m']) for x in mm)
        if (path/'trajectory.npz').exists():
            with np.load(path/'trajectory.npz') as z:infer.extend(z['decision_sec'].tolist())
    groups.append(dict(method=method,planned=30,attempts=n,not_started=30-n,completed=sum(x['status']=='completed' for x in ee),
                       successes=k,success_rate=k/n if n else None,wilson95=interval(k,n),
                       success_with_reach_and_grasp=sum(bool(x.get('success') and x.get('reach') and x.get('grasp')) for x in ee),
                       reach=sum(x.get('reach',False) for x in ee),grasp=sum(x.get('grasp',False) for x in ee),unlocked=sum(x.get('unlocked',False) for x in ee),
                       mean_final_angle_deg=mean('signed_angle_deg'),mean_final_ee_error_m=mean('ee_to_handle_m'),mean_step_ee_error_m=statistics.mean(errors) if errors else None,
                       final_angle_samples=sum(x.get('signed_angle_deg') is not None for x in ee),final_ee_samples=sum(x.get('ee_to_handle_m') is not None for x in ee),recorded_transition_samples=len(clip),
                       mean_clipped_fraction=statistics.mean(clip) if clip else None,
                       inference_p50_ms=float(np.median(infer)*1000) if infer else None,inference_p95_ms=float(np.percentile(infer,95)*1000) if infer else None,
                       mean_elapsed_sec=mean('elapsed_sec'),failure_types={name:sum(x.get('failure_type')==name for x in ee) for name in sorted(set(x.get('failure_type','unknown') for x in ee))}))
(r/'test_aggregate.json').write_text(json.dumps(groups,indent=2))
validation_groups=[]
for point in points:
    ee=[ep for ep in episodes if ep['category']=='validation' and ep.get('checkpoint_step')==point['step']]
    clip=[];infer=[]
    for ep in ee:
        path=r/ep['directory']
        with (path/'metrics.csv').open() as f:clip.extend(float(row['clipped_fraction']) for row in csv.DictReader(f))
        with np.load(path/'trajectory.npz') as z:infer.extend(z['decision_sec'].tolist())
    validation_groups.append(dict(step=point['step'],attempts=len(ee),mean_clipped_fraction=statistics.mean(clip) if clip else None,
        mean_final_ee_error_m=statistics.mean(ep['ee_to_handle_m'] for ep in ee),
        inference_p50_ms=float(np.median(infer)*1000) if infer else None,inference_p95_ms=float(np.percentile(infer,95)*1000) if infer else None))
(r/'validation_aggregate.json').write_text(json.dumps(validation_groups,indent=2))
if validation_groups:
    with (r/'validation_aggregate.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(validation_groups[0]));w.writeheader();w.writerows(validation_groups)
if groups:
    with (r/'test_aggregate.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(groups[0]));w.writeheader();w.writerows(groups)

checks=dict(data_contract=bool(audit.get('passed')),no_new_training_data=config.get('collect_new_training_episodes') is False,
            reused_data_models_logs_unchanged=bool(load(r/'reused_data_provenance.json',{}).get('all_historically_sealed_files_match')),
            total_budget_within_cap=sum(x['elapsed_sec'] for x in ledger)<=21600,
            new_optimizer_steps_within_cap=(max([5000]+[x['step'] for x in train])-5000)<=20000,
            selection_before_new_test=all(x['started']>=selection.get('frozen_at',float('inf')) for x in episodes if x['category']=='test' and 'started' in x),
            policy_expert_guard=all(x.get('expert_call_guard_passed',False) for x in episodes if x['mode']=='policy' and x['status']=='completed'),
            no_automatic_open_torque=all(x.get('config',{}).get('door_auto_open_force')==0 for x in episodes if x['status']=='completed'),
            videos_complete=bool(video) and all(x['passed'] for x in video))
scenarios={}
for x in episodes:
    if x['category']=='test':scenarios.setdefault(x['seed'],[]).append(x.get('randomization'))
checks['test_scenarios_match']=bool(scenarios) and all(v[0] is not None and all(x==v[0] for x in v) for v in scenarios.values())
checks['all_planned_tests_completed']=bool(groups) and all(x['attempts']==30 and x['completed']==30 for x in groups)
checks['test_seeds_isolated']=set(config.get('test_seeds',[])).isdisjoint(config.get('validation_seeds',[]))
checks['resume_rng_audits']=all(all(v for k,v in load(x).items() if k!='step') for x in list((r/'training').glob('resume_*.json'))+list((r/'resume-audit').glob('resume_*.json')))
complete_video_dirs={str(Path(item['path']).parent.relative_to(r)) for item in video if item['passed']}
media_coverage=[]
for group in groups:
    if group['method']=='untrained':continue
    matching=[ep for ep in episodes if (ep['category']=='test' and Path(ep['directory']).name.startswith(group['method']+'_seed_')) or (ep['category']=='videos' and Path(ep['directory']).name.startswith(group['method']+'_'))]
    media_coverage.append(any(ep['directory'] in complete_video_dirs for ep in matching))
    if group['method']!='expert' and any(not ep.get('success') for ep in matching if ep['category']=='test'):
        media_coverage.append(any(not ep.get('success') and ep['directory'] in complete_video_dirs for ep in matching))
checks['required_videos_covered']=bool(media_coverage) and all(media_coverage)
if resume:
    checks['preserved_checkpoint_hashes']=all(digest(r/name)==h for name,h in resume['protected_checkpoints'].items())
    checks['preserved_original_budget_prefix']=hashlib.sha256((r/'budget.jsonl').read_bytes()[:resume['budget_prefix_bytes']]).hexdigest()==resume['budget_prefix_sha256']
    checks['budget_command_ids_unique']=len({x['command_id'] for x in ledger})==len(ledger)
    checks['training_steps_contiguous']=all(row['step']==5001+i for i,row in enumerate(train))
    checks['experiment_config_unchanged']=digest(r/'config.json')==resume['config_sha256']
    checks['scientific_modules_unchanged_on_resume']=bool(load(r/'resume_source_audit.json',{}).get('all_scientific_modules_unchanged'))
    checks['preexisting_validation_points_unchanged']=all(digest(r/name)==h for name,h in resume.get('validation_completed_before',{}).items())
for x in episodes:
    if x['category']=='test' and x['status']=='completed':
        expected=bool(x['unlocked'] and x['signed_angle_deg']>=80 and x['rear_clearance_m']>=.52)
        assert expected==x['success'], x['directory']
atomic=dict(checks=checks,all_passed=all(checks.values()),expected_video_frame_to_step=dict(first=[0,2],middle=[712,1426],last=[1424,2850]))
(r/'delivery_checks.json').write_text(json.dumps(atomic,indent=2))

def fmt(x, digits=3):return '未完成' if x is None else ('%.*f'%(digits,x))
spent=sum(x['elapsed_sec'] for x in ledger)
last=max([5000]+[x['step'] for x in train])
stop_reason=status.get('reason') if status.get('phase')=='stopped' else selection.get('stop_reason',status.get('stop_reason','unknown'))
stale=patience_state(points)[0] if points else 0
lines=[f'# 浮动底座推门：受控续训实验报告\n',f'实验 `{r.name}`；状态：`{status.get("phase","unknown")}`。停止原因：`{stop_reason}`。\n',
       f'本轮累计占卡进程墙钟 **{spent/3600:.3f} GPU 小时**（上限 6 小时），新增优化 **{last-5000} 步**，累计到 **{last} 步**。完整进程计时包含加载、保存、采集、评估和视频；账本也保守包含 CPU 预检命令。\n',
       '## 独立闭环结论\n',
       '| 方法 | 成功/已启动 | Wilson 95% | 完成/计划 | 最终平均门角 | reach / grasp / 解锁 |\n|---|---:|---|---:|---:|---|']
if selection.get('status')=='validation_only_resource_stopped':
    lines.insert(2,f'停止类型：资源门槛触发。已完成阶段见下文，未将未启动测试计为完成；停止时连续无改善计数为 {stale}。\n')
if resume:
    completed_before='validation_point_021000.json' in resume.get('validation_completed_before',{})
    resume_action='沿用已完成的 21000 步验证，直接从该 checkpoint 继续优化，不重复旧验证' if completed_before else '先补齐 21000 步完整验证'
    lines.insert(3,f'资源停止后经用户明确授权接续：保留原 15000/21000 步模型文件，{resume_action}；原预算前缀、模型 SHA256 与配置核对见 resume_state.json。恢复前已计费 {resume["used_seconds_before"]/3600:.3f} GPU 小时，本次接续新增计费 {(spent-resume["used_seconds_before"])/3600:.3f} 小时，二者累计受同一 6 小时上限约束。恢复时沿用历史连续无改善计数 {resume["consecutive_no_improvement_before"]}，随后按完整验证结果更新为 {stale}。旧选模与交付记录封存在 `{resume["archive"]}`，此前恢复事件见 prior_resume_events。\n')
    updates_after=len(train)-resume['new_optimizer_steps_before']
    lines.insert(4,f'本次接续新增优化更新 **{updates_after} 步**。'+('训练阶段资源检查未通过，未执行新的优化器更新。\n' if updates_after==0 else '新增训练从原 checkpoint 恢复，恢复审计与逐步日志保留。\n'))
if selection.get('stop_reason')=='three_complete_points_without_task_improvement':
    lines.insert(3,'优化停止原因：连续三个完整验证点未提高主成功次数或历史最佳平均最终有向门角，按原规则提前结束优化；之后冻结选模并执行独立测试。\n')
for g in groups:
    ci=g['wilson95'];interval_text=f'[{100*ci[0]:.1f}%, {100*ci[1]:.1f}%]' if ci else '未评估'
    lines.append(f'| {g["method"]} | {g["successes"]}/{g["attempts"]} | {interval_text} | {g["completed"]}/30 | {fmt(g["mean_final_angle_deg"])}° | {g["reach"]} / {g["grasp"]} / {g["unlocked"]} |')
if not any(g['attempts'] for g in groups):lines.append('独立测试尚未启动；表中 0/0 不代表成功率为 0，不能据验证集报告新测试成绩。')
if groups and not checks['all_planned_tests_completed']:
    lines.append(f'\n原计划 {len(planned)} 回合，实际启动 {sum(g["attempts"] for g in groups)} 回合、完成 {sum(g["completed"] for g in groups)} 回合、未启动 {sum(g["not_started"] for g in groups)} 回合。逐回合状态见 planned_test_attempts.csv/JSON；各组未启动数为 '+ '、'.join(f'{g["method"]} {g["not_started"]}' for g in groups)+'。异常回合保留在成功率分母中，未重试。')
    lines.append('\n这些比例和 Wilson 区间仅描述已启动的部分测试；因基础设施故障提前终止且各组样本数不等，不能当作预定每组 30 回合的完整性能或证明稳定闭环成功。')
stop_diagnostics=load(r/'stop_diagnostics.json',{})
if stop_diagnostics:
    lines.append('\n停止证据：未训练模型 seed 5007 已保存 294 个完整执行步；下一步后观测有限性断言失败。日志先出现 PhysX 碰撞阶段错误码 700，继而出现 CUDA illegal memory access 和求解器 kernel 失败。已保存轨迹及待执行动作均为有限值；故障后状态未保存，无法定位具体非有限通道或确定 CUDA 故障触发原因。仅能确认该回合发生仿真基础设施故障，不能归因于输入契约、显存不足或硬件故障。见 stop_diagnostics.json 与 logs/0166_test_untrained_seed_5007.log。按原停止规则退出，剩余预算没有继续使用。')
if selection.get('stop_reason')=='max_optimizer_steps':
    lines.append(f'\n优化已达到累计 {last} 步上限，连续无改善计数为 {stale}/3；选模已在独立测试首次启动前冻结。独立测试的提前停止原因与训练步数上限分别记录。')
if selection.get('best_equals_last'):lines.append(f'\n最佳与最后模型均为 {selection["best_step"]} 步，测试合并为 best 组。')
elif selection:lines.append(f'\n最佳模型 {selection["best_step"]} 步；最后模型 {selection["last_step"]} 步。')
if selection.get('status')=='validation_only_resource_stopped':
    lines.append(f'\n最佳仅为完整验证集选出的候选，尚未经新测试集检验。最后模型有完整闭环验证：{selection["last_has_complete_closed_loop_validation"]}。测试种子 5000–5029 全部未使用；本轮没有录制新的 MP4。')
    if train and 'val_loss' in train[-1]:
        meaning='已完成对应十回合闭环验证，并按主成功率、门角、验证 loss 的顺序参与候选排序。' if selection['last_has_complete_closed_loop_validation'] else '这不构成完整闭环验证点，也不参与本轮最佳模型选择。'
        lines.append(f'\n最后 {last} 步模型的离线噪声验证 MSE 为 {train[-1]["val_loss"]:.6f}；{meaning}')
for g in groups:
    if g['method'] in ('best','last') and g['successes']:
        lines.append(f'\n{g["method"]} 的 {g["successes"]} 个数值主成功回合中，同时通过 reach 与 grasp 保持指标的有 {g["success_with_reach_and_grasp"]} 个。主成功判据未要求稳定抓取，必须结合阶段指标与视频解释，不能将数值成功直接称为稳定抓把推门。')
lines += ['\n主成功：已解锁、最终正确方向门角 ≥80°、尾端净距离 ≥0.52 m。所有已启动回合均进入分母；未启动回合不计为已评估。原 `open_80deg` 兼容字段实际默认 20°，另行保留，不参与主成功判断。\n',
          '## 前 1 秒诊断\n',f'同步、坐标、上一执行动作、模型历史/队列、深度及归一化契约检查：**{audit.get("passed",False)}**。检查明细和逐通道原始数据保留。\n',
          '| 5000 步模型回合 | 首次超训练状态范围 | 首次 ≥50% 通道限幅 |\n|---|---:|---:|']
for x in audit.get('early_episodes',[]):
    if x['mode']=='policy':lines.append(f'| {x["episode"]} | {fmt(x.get("first_outside_training_range_s"))} s | {fmt(x.get("first_half_clipped_s"))} s |')
diagnostic_experts=[ep for ep in episodes if ep['category']=='diagnostic' and ep['mode']=='expert']
if diagnostic_experts:
    lines.append(f'\n本轮诊断专家完成 {len(diagnostic_experts)} 回合，主成功 {sum(bool(ep.get("success")) for ep in diagnostic_experts)}/{len(diagnostic_experts)}；这是控制与观测参照，不是预定 30 回合独立专家测试。')
lines += ['\n25/100 去噪对照使用相同 checkpoint 与种子 4000–4002，仅改变采样步数；100 步结果仅作诊断。录制专家观测下的离线动作误差保存在 offline_diagnostics.json，不能替代真实闭环表现。训练集包含夹爪启动速度瞬态，单个较大 z-score 本身不能证明实现错误。状态名 q0/dq0 到 q5/dq5 对应 joint1 到 joint6，q6/dq6 对应夹爪。\n',
          f'![前 1 秒诊断]({local}/plots/early_closed_loop_diagnostics.png)\n',
          '## 受控续训与选模\n',
          '沿用原 14 条训练、4 条验证演示，manifest 与归一化不变，不新增训练数据。schema=float_push_joint_v1；batch=16，AdamW lr=1e-4、weight_decay=0.01，梯度裁剪=1，观察/预测/执行窗口=4/16/4，DDPM 训练 100 步、推理 25 步、clip_sample=False。恢复模型、优化器、采样器及 Python/NumPy/Torch/CUDA RNG。每 250 步固定离线验证，每 500 步不可覆盖保存。\n',
          '| 累计优化步 | 完整验证成功 | Wilson 95% | 最终平均门角 | reach / grasp / 解锁 | 验证噪声 MSE |\n|---:|---:|---|---:|---|---:|']
# Add paired diagnostics next to the evidence, rather than assigning a cause
# merely from a clipped action or a high normalized state value.
paired=['\n| 去噪步数（诊断） | 闭环成功 | 平均最终门角 | 专家观测离线底座 vx RMSE | 机械臂关节 RMSE 范围 |\n|---:|---:|---:|---:|---:|']
offline=load(r/'offline_diagnostics.json',[])
for inference in (25,100):
    pp=[x for x in episodes if x['category']=='diagnostic' and x['mode']=='policy' and x.get('inference_steps')==inference]
    oo=next((x for x in offline if x['inference_steps']==inference),None)
    rmse=oo['rmse_per_channel'] if oo else None
    paired.append(f'| {inference} | {sum(x.get("success",False) for x in pp)}/{len(pp)} | {fmt(statistics.mean(x.get("signed_angle_deg",0) for x in pp) if pp else None)}° | {fmt(rmse[0] if rmse else None,5)} m/s | {fmt(min(rmse[3:9]) if rmse else None)}–{fmt(max(rmse[3:9]) if rmse else None)} rad |')
sampling=load(r/'sampling_metadata.json',{})
if sampling:
    ss=sampling['schedules']
    paired.append(f'\n实装 DDPM 的 25/100 步时间表首步分别是 t={ss[0]["first_timestep"]}/{ss[1]["first_timestep"]}；预测 x0 对 epsilon 误差的首步放大系数分别约为 {ss[0]["first_epsilon_error_to_x0_gain"]:.1f}/{ss[1]["first_epsilon_error_to_x0_gain"]:.1f}。100 步引入更低信噪比的终端步，在 clip_sample=False 下可能放大尚未充分拟合的噪声预测误差。此为配置与公式支持的解释，尚未单独干预验证因果。')
hundred=[x for x in audit.get('early_episodes',[]) if x['mode']=='policy' and x['inference_steps']==100]
if len(hundred)==3 and all(x['first_half_clipped_s']==0 for x in hundred):
    paired.append('\n100 步诊断在三个回合的首个动作即出现多数通道限幅，且专家观测离线预测误差同样很大；因此它的异常不能仅归咎于环境状态偏离。25 步则先出现关节速度越界，再出现大范围限幅。')
early_detail=audit.get('early_episodes',[])
if early_detail:
    description=[]
    for detail in early_detail:
        if detail['mode']=='policy':
            description.append(f'{detail["episode"]} 首次越过训练范围的通道：'+', '.join(detail.get('first_outside_channels',[])))
    paired.append('\n'+ '；'.join(description)+'。')
    paired.append('\n当状态越界先于多数动作限幅且契约检查通过时，证据支持闭环偏离后的反馈放大；这仍是时序证据，不能单凭本轮将根因唯一归结为关节速度、采样器或训练量。')
insert_at=next(i for i,x in enumerate(lines) if x=='## 受控续训与选模\n')
lines[insert_at:insert_at]=paired
for x in points:
    vv=[ep for ep in episodes if ep['category']=='validation' and ep.get('checkpoint_step')==x['step']]
    phases=' / '.join(str(sum(ep.get(key,False) for ep in vv)) for key in ('reach','grasp','unlocked'))
    ci=interval(x['successes'],10)
    lines.append(f'| {x["step"]} | {x["successes"]}/10 | [{100*ci[0]:.1f}%, {100*ci[1]:.1f}%] | {x["mean_final_angle_deg"]:.6f}° | {phases} | {x["val_loss"]:.6f} |')
lines.append('\n上述 Wilson 区间描述固定验证集的回合比例；该集合已反复用于选模，不能把这些区间或验证成功率当作独立测试性能。')
last_validation=[ep for ep in episodes if ep['category']=='validation' and ep.get('checkpoint_step')==selection.get('last_step')]
if len(last_validation)==10:
    opened=sum(ep['signed_angle_deg']>=80 for ep in last_validation)
    short=sum(ep['signed_angle_deg']>=80 and ep['rear_clearance_m']<.52 for ep in last_validation)
    lines.append(f'\n最后 {selection["last_step"]} 步模型有 {opened}/10 回合最终门角达到 80°，其中 {short} 回合尾端净距离不足 0.52 m；reach/grasp 分别为 {sum(ep["reach"] for ep in last_validation)}/10、{sum(ep["grasp"] for ep in last_validation)}/10。平均门角提高不能代替完整过门成功。')
    if resume and stale==0 and points[-1]['mean_final_angle_deg']>resume['best_mean_angle_before']+.001:
        lines.append(f'\n本次完整验证的平均最终门角超过此前历史最佳 {resume["best_mean_angle_before"]:.6f}°，因此按预定规则重置早停计数；候选排序仍首先比较主成功率，最佳模型为 {selection["best_step"]} 步。')
selected_validation=[ep for ep in episodes if ep['category']=='validation' and ep.get('checkpoint_step')==selection.get('best_step')]
selected_success=[ep for ep in selected_validation if ep.get('success')]
if selected_success:
    held=sum(bool(ep.get('reach') and ep.get('grasp')) for ep in selected_success)
    lines.append(f'\n候选 {selection["best_step"]} 步的 {len(selected_success)} 个数值主成功回合中，同时通过 reach 与 grasp 指标的只有 {held} 个。主成功判据没有要求稳定抓取；这些结果不能直接称为稳定抓把推门。阶段指标与原始轨迹一并保留。')
if continuation_early:
    lines.append('\n| 累计优化步 | 前 1 秒触发多数限幅的回合 | 已触发者平均首次时间 | 前 1 秒平均限幅比例 |\n|---:|---:|---:|---:|')
    for step in sorted(set(x['step'] for x in continuation_early)):
        ee=[x for x in continuation_early if x['step']==step];tt=[x['first_half_clipped_s'] for x in ee if x['first_half_clipped_s'] is not None]
        first_text=fmt(statistics.mean(tt))+' s' if tt else '—（全删失）'
        lines.append(f'| {step} | {len(tt)}/{len(ee)} | {first_text} | {100*statistics.mean(x["mean_first_second_clipped_fraction"] for x in ee):.2f}% |')
    lines.append('\n未在前 60 步触发的回合按右删失保留，不填为 1 秒或无穷大；早期限幅指标只用于诊断。')
    lines.append(f'\n![续训早期闭环对照]({local}/plots/continuation_early_comparison.png)\n')
lines += ['\n连续三个完整验证点相较历史最佳成功次数和最终平均门角均无提高则早停（门角容差 0.001°）；loss、reach 和限幅时间不重置计数。最佳按成功率、最终门角、验证 loss 依次选择；同分取较早 checkpoint。测试种子 5000–5029 在选模冻结后使用，旧 3000–3009 未作为新测试。\n',
          f'![完整验证停止指标]({local}/plots/validation_stopping_metrics.png)\n',
          f'![训练与离线验证]({local}/plots/training_validation.png)\n',f'![闭环验证]({local}/plots/closed_loop_success.png)\n',f'![独立测试]({local}/plots/test_comparison.png)\n',
          f'![实际回合门角、把手与末端误差]({local}/plots/rollout_metrics.png)\n',
          '## 动作与运行诊断\n','| 方法 | 平均限幅比例 | 最终 EE 误差 | 决策 p50 / p95 |\n|---|---:|---:|---:|']
for g in groups:
    clip=100*g['mean_clipped_fraction'] if g['mean_clipped_fraction'] is not None else None
    lines.append(f'| {g["method"]} | {fmt(clip,2)}% | {fmt(g["mean_final_ee_error_m"])} m | {fmt(g["inference_p50_ms"],2)} / {fmt(g["inference_p95_ms"],2)} ms |')
for g in groups:
    if g['final_angle_samples']<g['attempts'] or g['final_ee_samples']<g['attempts']:
        lines.append(f'\n{g["method"]} 最终门角/末端误差仅有 {g["final_angle_samples"]}/{g["final_ee_samples"]} 个有效末态样本（已启动 {g["attempts"]} 回合），异常回合没有伪造末态。限幅统计按 {g["recorded_transition_samples"]} 个已记录执行步平均，包含异常回合保存的前 294 步；时延包含异常回合已发生的网络决策。reach/grasp/解锁仅计已记录确认事件，异常缺失不当作确认成功。')
if validation_groups:
    lines.append('\n已完成固定验证的全回合统计（限幅按所有执行步/通道平均，时延按网络决策汇总）：\n\n| 累计步 | 全回合平均限幅 | 平均最终 EE 误差 | 决策 p50 / p95 |\n|---:|---:|---:|---:|')
    for g in validation_groups:
        lines.append(f'| {g["step"]} | {100*g["mean_clipped_fraction"]:.2f}% | {g["mean_final_ee_error_m"]:.3f} m | {g["inference_p50_ms"]:.2f} / {g["inference_p95_ms"]:.2f} ms |')
lines += ['\n动作、末端误差与决策时延的原始记录保留在对应 metrics.csv、trajectory.npz 和 early_diagnostic.npz 中。未启动测试显示为未完成。推理时延仅统计真正网络决策，四步队列的取动作不算推理；首调用包含在原始记录中。\n',
          '## 视频、预算与交付核验\n',f'完整预定测试：{checks["all_planned_tests_completed"]}；视频全部完整解码：{checks["videos_complete"]}。每 2 仿真步采样，30 fps；完整视频应为 1425 帧、47.5 秒。首/中/尾帧分别对应仿真步 2/1426/2850，抽帧与逐步日志关联保存。12 个抽帧画面与对应时刻/门角/成功标注的人工核验：{load(r/"visual_review.json",{}).get("all_12_video_anchors_match_metrics",False)}；明细见 visual_review.json 与 video_frame_alignment.json。\n']
lines.append('| 预算阶段 | 完整进程墙钟 GPU 小时 |\n|---|---:|')
for stage in dict.fromkeys(x['stage'] for x in ledger):
    lines.append(f'| {stage} | {sum(x["elapsed_sec"] for x in ledger if x["stage"]==stage)/3600:.4f} |')
lines.append(f'\n工程核验：预算正常退出、TERM 超时计费、互斥、最终预留、死进程恢复及不重复记账测试通过；25/100 步缓存推理与完整参考前向的最大误差均为 0（默认推理仍为 25）。完整 RNG 恢复审计={checks["resume_rng_audits"]}；模型专家函数调用保护={checks["policy_expert_guard"]}；自动开门扭矩为零={checks["no_automatic_open_torque"]}。明细见 logs、resume-audit 和 training/resume_*.json。')
if resume:
    lines.append('\n本次恢复检查：原预算前缀、配置和 15000/21000 步模型哈希保持一致，预算命令 ID 不重复、训练步日志连续，既有完整验证点哈希不变；见 delivery_checks.json 与 logs/resume_cpu_checks*.log。')
supplemental=load(r/'supplemental_video_status.json',{})
if supplemental.get('exit_code'):
    lines.append(f'\n补录视频阶段未全部完成，原因见 {supplemental["log"]}；已完成独立测试不因此重新抽样，已有视频完整性逐一报告。')
if train:
    old_train=rows(Path(config['old_run'])/'training/train.jsonl')
    prior=old_train[-1]['elapsed_sec'] if old_train else 0
    lines.append(f'\n本轮新增训练循环累计约 {train[-1]["elapsed_sec"]-prior:.1f} 秒（含循环内离线验证/保存）；与上表包含初始化和全部仿真的占卡进程墙钟分开报告。\n')
if 'No eligible GPU' in status.get('reason',''):
    samples_path=sorted([p for p in r.glob('resource_*.json') if not load(p).get('candidates')],key=lambda p:p.stat().st_mtime)
    if samples_path:
        resources=load(samples_path[-1])['samples']
        lines.append('\n资源停止证据（三次采样，间隔 2 秒；要求空闲 ≥16 GiB 且平均利用率 <70%）：\n\n| GPU | 三次最小空闲显存 | 平均利用率 |\n|---:|---:|---:|')
        for index,item in enumerate(resources[0]):
            lines.append(f'| {item["index"]} | {min(s[index]["free_mib"] for s in resources)/1024:.2f} GiB | {statistics.mean(s[index]["util"] for s in resources):.2f}% |')
for x in video:
    relative=Path(x['path']).relative_to(r)
    lines.append(f'- [{relative}]({local/relative})：{x["decoded_frames"]} 帧，{x["fps"]} fps，{fmt(x["duration_sec"],1)} 秒，解码核验 {x["passed"]}。')
reproductions=[x for x in episodes if x['category']=='videos']
if reproductions:
    lines.append(f'\n补录视频共启动 {len(reproductions)} 回合，完成 {sum(x["status"]=="completed" for x in reproductions)} 回合，数值主成功 {sum(bool(x.get("success")) for x in reproductions)}/{len(reproductions)}。这些回合重复已冻结测试种子，仅用于展示，不作为额外独立样本，也不参与选模；所有补录失败、异常或与原测试不同的结果均保留。')
    with (r/'video_reproduction_attempts.json').open('w') as f:json.dump(reproductions,f,indent=2)
    for x in reproductions:
        lines.append(f'- `{x["directory"]}`：原测试期望成功={x.get("intended_success")}，实际成功={x.get("success")}，状态={x["status"]}。')
video_note=f'本轮已核验 {len(video)} 个 MP4，完整性逐一列于上文。' if video else '本轮未生成 MP4。'
if video and status.get('phase')=='stopped' and any(g['method']=='best' and g['successes'] for g in groups):
    lines.append('\n四个视频均为 seed 5000：专家成功，未训练/最佳/最后模型失败，保留了完整代表性失败。测试中最佳模型的数值成功回合未开启视频；基础设施停止后未补录，不能用已有失败视频展示其成功。异常 seed 5007 未录视频，保存了部分轨迹和完整错误日志。视频抽帧与日志对照见 video_frame_alignment.json 和 video_contact_sheet.png。')
lines += [f'\n服务器目录：`{r}`；本地目录：`{local}`。配置、源码快照、环境清单、原始 CSV/JSONL、分步轨迹与 PNG/PDF 图表均按 run_id 保存。{video_note}必要模型包括未训练、5000 步起点、最佳和最后模型；全部分步 checkpoint 保留服务器。SHA256 清单用于同步校验。\n',
          f'原始 20 条数据、manifest、训练日志、预算日志及起点/未训练模型共 25 个文件与历史 SHA256 封存逐一核对：{checks["reused_data_models_logs_unchanged"]}；见 reused_data_provenance.json。\n',
          '旧数据缺少未经处理的 Gym 原始深度与显式末步后状态；本轮仅复用其训练输入，不追补或伪造。新诊断专家回合和前 60 步记录保存原始 Gym 深度、处理深度、动作前观测、执行动作与显式后状态，且不进入训练 manifest。\n',
          '## 结论边界\n',
          '脚本专家成功、训练接口可运行、噪声 loss 下降和独立闭环成功分别以对应证据判断。此实验只覆盖门 99650089960001 和小范围初态随机化；浮动底座由运动学直接移动，未加载低层行走或平衡策略。模型使用 Gym 语义分割辅助的双相机 mask/depth。尾端净距离判据未检查门洞横向容纳或足接触，不能外推真实机器人或其他门泛化。\n']
trained_groups=[g for g in groups if g['method'] in ('best','last')]
if trained_groups and all(g['attempts']==30 and g['successes']==0 for g in trained_groups):
    lines.append('\n本轮受控续训在既定范围内未获得独立闭环成功；增加原数据上的优化量尚未获得能完成整个任务的策略。早期稳定性是否改善另见前 1 秒诊断，不能与完整任务成功混为一谈。该结论不证明更长训练一定无效，也不支持继续无限延长。本轮按预定条件停止。后续若继续，应另立单因素实验验证动作变化约束、速度输入敏感性或恢复状态覆盖，不能把这些改动混入本轮结果。\n')
failures=[x for x in ledger if x.get('exit_code') not in (0,None)]
if failures:
    lines.append(f'保留 {len(failures)} 次非零退出记录，均已计费；逐项见 budget.jsonl。历史预检曾因 diffusers 内部默认字段列表顺序触发配置误报，修正为比较实际扩散参数后重跑；本次独立测试的 PhysX CUDA 故障另见上述停止证据，未重跑。\n')
if status.get('phase')!='gpu_work_complete':lines.append(f'本轮未完成全部实验：{status.get("reason","见 status.json")}。已保存实际结果与未完成项，不宣称收敛或完整测试。\n')
rendered=[]
for line in lines:
    if line.startswith('!['):
        target=line.rsplit('](',1)[1].strip().rstrip(')')
        if not (r/Path(target).relative_to(local)).exists():
            rendered.append('对应图未生成：该实验阶段尚未执行。\n')
            continue
    rendered.append(line)
(r/'REPORT.md').write_text('\n'.join(rendered))
hashes={str(path.relative_to(r)):dict(bytes=path.stat().st_size,sha256=digest(path)) for path in r.rglob('*') if path.is_file() and path.name not in ('artifact_sha256.json','budget.lock','driver.lock') and not path.name.endswith('.tmp')}
(r/'artifact_sha256.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps(dict(report=str(r/'REPORT.md'),checks=checks,used_gpu_hours=spent/3600,test_groups=groups,hashed_files=len(hashes))),flush=True)
