"""Postprocess already completed30k/6 runs; no policy inference or simulation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main(root,repo):
    sys.path.insert(0,str(repo/'high-level/float_ik/learning'))
    sys.path.insert(0,str(repo/'high-level/dp'))
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from smallbatch30k import NODES,verify
    from branch16040_analysis import angle_error
    from stop_data_checks import measured_motion
    from diagnostic_integrity import sha256
    from forward_report import write_csv
    from stop_tail_online import smoothed,metric_rows
    torch.set_num_threads(2);verify(root)
    assert json.loads((root/'state.json').read_text())['phase']=='completed'
    out=root/'final_analysis';out.mkdir(exist_ok=False)
    def save(name,value):
        (out/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        if isinstance(value,list):write_csv(out/(name+'.csv'),value)
    stats=torch.load(root/'training_only_stats.pt',map_location='cpu',weights_only=False)
    hashes=[];tasks=[];offline=[];tracking=[];trajectory=[];timing=[];initial=[]
    with np.load(root/'raw/episode_000000.npz',allow_pickle=True) as z:
        raw={k:z[k].copy() for k in ['state','action','wrist_handle_mask','wrist_masked_depth','front_handle_mask','front_masked_depth']}
    for step in [0,*NODES]:
        p=root/'training/checkpoints'/('step_%06d.pt'%step);c=torch.load(p,map_location='cpu',weights_only=False)
        assert c['config']['completed_updates']==step and c['config']['batch_size']==8 and c['config']['seed']==42
        assert c['config']['initialize_weights'] is None and c['config']['initialization']['mode']=='random_seeded_training_data_stats'
        assert all(torch.equal(c['stats'][k],v) for k,v in stats.items())
        values=list(c['model'].values())+list(c['stats'].values())+[v for state in c['optimizer']['state'].values() for v in state.values() if isinstance(v,torch.Tensor)]
        assert all(torch.isfinite(v).all() for v in values if v.is_floating_point())
        if step:assert len(c['optimizer']['state'])==130 and all(int(v['step'])==step for v in c['optimizer']['state'].values())
        assert all(g['lr']==1e-4 and g['weight_decay']==1e-6 for g in c['optimizer']['param_groups'])
        hashes.append(dict(updates=step,path=str(p),sha256=sha256(p),bytes=p.stat().st_size,
            optimizer_parameter_states=len(c['optimizer']['state']),sampler_cursor=c['training_state']['sampler']['cursor'],sampler_epoch=c['training_state']['sampler']['epoch']))
        if not step:continue
        folder=root/'pilot/evaluation'/('step_%06d'%step)/'seed_16040';result=json.loads((folder/'result.json').read_text());stop=result['stopping'];checks=stop['checks']
        rows=[json.loads(x) for x in (folder/'execution.jsonl').read_text().splitlines()]
        dt=json.loads((folder/'config.json').read_text())['dt'];_,_,distance,speed,omega=measured_motion(rows,dt)
        with np.load(folder/'trajectory.npz',allow_pickle=True) as z:
            exact=dict(updates=step,state=np.array_equal(raw['state'][0],z['state'][0]),
                images=all(np.array_equal(raw[k][0],z[k][0]) for k in raw if k.endswith('mask') or k.endswith('depth')))
        assert exact['state'] and exact['images'];initial.append(exact)
        passed=bool(result['pass_done']);candidate=stop.get('stop_step');complete=bool(checks.get('two_second_hold_complete',False))
        p=result['first_pass_frame'];span=(len(rows)-1-p)*dt if p is not None else None
        tasks.append(dict(updates=step,frames=result['frames'],execution_ok=result['execution_ok'],opened=result['opened'],
            max_signed_door_deg=result['max_signed_door_deg'],first_open_sec=result['first_open_frame']*dt if result['first_open_frame'] is not None else None,
            passed=passed,first_pass_sec=p*dt if p is not None else None,pass_event_distance_m=result.get('pass_event_distance_m'),
            stop_evaluable=passed,stopped=result['stopped'] if passed else None,joint=result['success'],stop_status=result['stop_status'],
            post_pass_real_span_sec=span,post_pass_actual_min_speed_m_s=float(speed[p:].min()) if passed else None,
            post_pass_actual_max_speed_m_s=float(speed[p:].max()) if passed else None,
            post_pass_actual_max_angular_speed_rad_s=float(omega[p:].max()) if passed else None,
            post_pass_total_path_m=stop.get('post_pass_total_path_m') if passed else None,
            low_speed_candidate_latency_sec=stop.get('settle_latency_sec') if passed else None,
            low_speed_candidate_real_control_steps=len(rows)-candidate if candidate is not None else None,
            low_speed_candidate_physical_span_sec=(len(rows)-candidate)*dt if candidate is not None else None,
            two_second_hold_complete=complete if passed else None,
            verified_two_second_drift_m=stop.get('max_two_second_drift_m') if complete else None,
            raw_partial_candidate_drift_m=stop.get('max_two_second_drift_m') if not complete else None,
            qualified_stop_latency_sec=stop.get('settle_latency_sec') if result['stopped'] else None,
            extra_path_until_candidate_m=stop.get('extra_path_until_stopped_m'),
            remaining_candidate_max_position_drift_m=stop.get('max_remaining_position_drift_m'),
            last_actual_speed_m_s=float(speed[-1]),backward_travel_after80_m=result.get('backward_travel_after80_m'),
            yaw_travel_after80_deg=result.get('yaw_travel_after80_deg'),snap_events=result['snap_events'],video=str(folder/'rollout.mp4')))
        assert result['execution_ok'] and result['frames']==2500
        d=json.loads((root/'offline'/('step_%06d'%step)/'summary.json').read_text())
        offline.extend(dict(m,updates=step) for m in d['metrics'])
        comp=json.loads((root/'comparison'/('step_%06d'%step)/'summary.json').read_text())
        tracking.append(dict(comp['full_record_tracking'],updates=step))
        trajectory.extend(dict(m,updates=step) for m in comp['metrics'])
        timing.extend(dict(m,updates=step) for m in comp['stage_timing'])
    rows=smoothed(metric_rows([root/'training/metrics.jsonl']))
    assert len(rows)==3000 and [v['train/step'] for v in rows]==list(range(10,30001,10))
    assert all(v['train/lr']==1e-4 for v in rows)
    for name,value in [('checkpoint_hashes',hashes),('tasks',tasks),('offline_physical_errors',offline),
        ('full_record_tracking',tracking),('trajectory_errors',trajectory),('stage_timing',timing),('initial_conditions',initial),('training_metrics',rows)]:save(name,value)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,key,title in zip(axes,['train/loss','train/lr','train/elapsed_sec'],['Noise MSE','Learning rate','Optimization seconds']):
        ax.plot([v['train/step'] for v in rows],[v[key] for v in rows],label='Raw')
        for node in NODES:ax.axvline(node,color='grey',alpha=.2)
        ax.set(xlabel='Actual optimizer updates (random start)',ylabel=title);ax.grid(alpha=.2)
    axes[0].plot([v['train/step'] for v in rows],[v['train/loss_ema'] for v in rows],label='EWMA alpha0.1 display only');axes[0].legend()
    fig.savefig(out/'training_curves.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for stage in ['all','conversion','hold']:
        values=[v for v in offline if v['stage']==stage and v['horizon']==16]
        for ax,key,title in zip(axes.flat,['vx_rmse_m_s','yaw_rmse_rad_s','ee_position_rmse_m','ee_orientation_rmse_deg','gripper_rmse_rad'],
            ['vx RMSE m/s','yaw RMSE rad/s','position RMSE m','sign-invariant orientation RMSE deg','gripper RMSE rad']):
            ax.plot([v['updates'] for v in values],[v[key] for v in values],marker='o',label=stage);ax.set(xlabel='Updates',ylabel=title);ax.grid(alpha=.2);ax.legend()
    axes[1,2].axis('off');fig.savefig(out/'diagnostic_curves_prefix16.png',dpi=140);plt.close(fig)
    save('definitions',dict(expert_diagnostic='Fixed19 windows times3 diffusion seeds, not whole-episode time average; full32 and executed16 scored separately.',
        trajectory='True-time uses real reference common prefix only. Stage alignment is supplementary; delays/durations/missing stages retained.',
        tracking='Actual post-step EE reconstructed from next_state and recorded yaw-only base transform; no future observations used for inference.',
        partial_stop='Raw legacy max_two_second_drift_m may be a shorter low-speed candidate. Verified value isnull unless complete2-second hold; qualified latency only for accepted stop.',
        no_pass='Stopped/stop latency/drift/path metrics areNA without the original pass; joint fails.',
        no_imitation_threshold='Continuous errors, no new imitation success threshold.',
        scope='Single training episode and one training seed, one rollout per different checkpoint; not six repetitions of final policy or generalization test.'))
    final=tasks[-1];all1=next(v for v in offline if v['updates']==1000 and v['stage']=='all' and v['horizon']==16);all30=next(v for v in offline if v['updates']==30000 and v['stage']=='all' and v['horizon']==16)
    lines=['# 单回合B、batch8、随机初始化、30k完整交付','',
        '执行完整性：30000次真实优化、仅指定1446帧/1415窗口、训练数据统计、完整续训、6次2500步闭环均完成。',
        '动作模仿：固定专家观测下部分物理误差下降；不将连续误差自动判为模仿通过。',
        '闭环：最终开门/原通过，但未自主停稳，联合失败；六个不同节点的联合均未通过。','',
        '| 更新 | 正向门角° | 原通过 | 通过秒 | 停车 | 联合 |','|---:|---:|---|---:|---|---|']
    for v in tasks:lines.append('|%d|%.5f|%s|%s|%s|%s|'%(v['updates'],v['max_signed_door_deg'],'是' if v['passed'] else '否',
        '%.2f'%v['first_pass_sec'] if v['first_pass_sec'] is not None else '—','失败' if v['stop_evaluable'] else 'NA','通过' if v['joint'] else '失败'))
    lines+=['','| 前16步固定窗口RMSE | 1k | 30k |','|---|---:|---:|']
    for key,name in [('vx_rmse_m_s','vx m/s'),('yaw_rmse_rad_s','yaw rad/s'),('ee_position_rmse_m','末端位置m'),('ee_orientation_rmse_deg','姿态°'),('gripper_rmse_rad','夹爪rad')]:lines.append('|%s|%.6f|%.6f|'%(name,all1[key],all30[key]))
    lines+=['','最终通过后实际速度范围 %.6f–%.6fm/s，额外路径%.6fm，无合格停止；真实剩余记录%.2fs。'%(final['post_pass_actual_min_speed_m_s'],final['post_pass_actual_max_speed_m_s'],final['post_pass_total_path_m'],final['post_pass_real_span_sec']),
        '10k末尾候选仅3个控制步；原短片段漂移不作为验证过的2秒漂移，详见tasks.csv。',
        '完整32/16与各阶段误差、轨迹/自身跟踪、真实延迟/缺失阶段、模型哈希及训练曲线均在本目录。',
        '原27D/10D方案在这组条件下未实现可靠闭环轨迹模仿与自主停车；不据此声称不可学习或单独归因batch。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    verify(root);print(json.dumps(dict(execution_complete=True,updates=30000,rollouts=6,joint_successes=sum(v['joint'] for v in tasks),files=len(list(out.iterdir())))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repo',required=True);a=p.parse_args();main(Path(a.root).resolve(),Path(a.repo).resolve())
