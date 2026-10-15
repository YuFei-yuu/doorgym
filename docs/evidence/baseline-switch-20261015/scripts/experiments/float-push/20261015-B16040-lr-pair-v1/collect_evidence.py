"""Postprocess preserved artifacts only: no training, neural inference, or simulation."""
import argparse,json,sys,subprocess,shutil,time,math
from pathlib import Path

def main(root,repo):
    sys.path[:0]=[str(repo/'high-level/float_ik/learning'),str(repo/'high-level/dp')]
    import numpy as np,torch
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lr_pair import OLD,NODES,LRS,SOURCE,EXPECTED
    from smallbatch30k import verify
    from diagnostic_integrity import sha256,materialize_external_links
    from train_door_dp import exact_state
    from stop_data_checks import measured_motion
    from stop_audit import inspect_video
    from forward_report import write_csv
    torch.set_num_threads(2);verify(root)
    assert json.loads((root/'state.json').read_text())['phase']=='completed'
    assert sha256(SOURCE)==EXPECTED
    out=root/'final_analysis';out.mkdir(exist_ok=True)
    def save(name,value):
        (out/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        if isinstance(value,list):write_csv(out/(name+'.csv'),value)
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    checkpoint=[];tasks=[];offline=[];variation=[];tracking=[];trajectory=[];timing=[];training=[];initial=[];video_checks={};online={}
    base_off=json.loads((root/'baseline/summary.json').read_text())['metrics']
    base_var=json.loads((root/'baseline/bias_variation.json').read_text())['metrics']
    offline.extend(dict(m,branch='baseline',cumulative_updates=30000,added_updates=0) for m in base_off)
    variation.extend(dict(m,branch='baseline',cumulative_updates=30000,added_updates=0) for m in base_var)
    with np.load(OLD/'raw/episode_000000.npz',allow_pickle=True) as z:
        raw={k:z[k][0].copy() for k in ['state','wrist_handle_mask','wrist_masked_depth','front_handle_mask','front_masked_depth']}
    reference_first=json.loads((OLD/'reference/execution.jsonl').read_text().splitlines()[0])
    eval_configs=[]
    for label,lr in LRS.items():
        r=root/label;verify(r)
        freeze=json.loads((r/'training_source_freeze.json').read_text())['source_hashes']
        assert all(sha256(repo/n)==h for n,h in freeze.items())
        assert json.loads((r/'online/finished.json').read_text())['verified']
        for step in [30000,*NODES]:
            p=r/('restore_audit' if step==30000 else 'training')/'checkpoints'/('step_%06d.pt'%step)
            c=torch.load(p,map_location='cpu',weights_only=False)
            assert c['config']['completed_updates']==step and c['config']['lr']==lr and c['config']['batch_size']==8
            assert c['config']['num_workers']==0 and c['config']['seed']==42
            assert exact_state(c['stats'],source['stats']) and len(c['optimizer']['state'])==130
            assert all(int(v['step'])==step for v in c['optimizer']['state'].values())
            assert all(g['lr']==lr and g['weight_decay']==1e-6 for g in c['optimizer']['param_groups'])
            vals=list(c['model'].values())+[v for q in c['optimizer']['state'].values() for v in q.values() if isinstance(v,torch.Tensor)]
            assert all(torch.isfinite(v).all() for v in vals)
            checkpoint.append(dict(branch=label,cumulative_updates=step,added_updates=step-30000,sha256=sha256(p),path=str(p),lr=lr,
                optimizer_parameter_states=130,sampler_cursor=c['training_state']['sampler']['cursor'],sampler_epoch=c['training_state']['sampler']['epoch']))
            if step==30000:continue
            folder=r/'pilot/evaluation'/('step_%06d'%step)/'seed_16040';result=json.loads((folder/'result.json').read_text());stop=result['stopping'];checks=stop['checks']
            rows=[json.loads(x) for x in (folder/'execution.jsonl').read_text().splitlines()]
            dt=json.loads((folder/'config.json').read_text())['dt'];_,_,distance,speed,omega=measured_motion(rows,dt)
            with np.load(folder/'trajectory.npz',allow_pickle=True) as z:
                exact=dict(branch=label,cumulative_updates=step,state=np.array_equal(raw['state'],z['state'][0]),images=all(np.array_equal(v,z[k][0]) for k,v in raw.items() if k!='state'))
            exact['base_root_before_exact']=np.array_equal(rows[0]['root_before'],reference_first['root_before'])
            cfg=json.loads((folder/'config.json').read_text())
            exact['float_seed']=cfg['args']['float_seed'];exact['DDPM_steps']=cfg['args']['dp_inference_steps'];exact['execution_horizon']=cfg['args']['dp_action_horizon']
            assert exact['state'] and exact['images'] and exact['base_root_before_exact'] and exact['float_seed']==16040 and exact['DDPM_steps']==100 and exact['execution_horizon']==16
            initial.append(exact)
            eval_configs.append(dict(branch=label,cumulative_updates=step,args={k:v for k,v in cfg['args'].items() if k not in ['dp_policy_checkpoint','float_output']}))
            passed=bool(result['pass_done']);p=result['first_pass_frame'];candidate=stop.get('stop_step');complete=bool(checks.get('two_second_hold_complete',False))
            tasks.append(dict(branch=label,cumulative_updates=step,added_updates=step-30000,frames=result['frames'],execution_ok=result['execution_ok'],
                opened=result['opened'],max_signed_door_deg=result['max_signed_door_deg'],first_open_sec=result['first_open_frame']*dt if result['first_open_frame'] is not None else None,
                passed=passed,first_pass_sec=p*dt if p is not None else None,pass_event_distance_m=result.get('pass_event_distance_m'),stop_evaluable=passed,
                stopped=result['stopped'] if passed else None,joint=result['success'],stop_status=result['stop_status'],
                post_pass_real_span_sec=(len(rows)-1-p)*dt if passed else None,
                post_pass_actual_min_speed_m_s=float(speed[p:].min()) if passed else None,post_pass_actual_max_speed_m_s=float(speed[p:].max()) if passed else None,
                post_pass_actual_max_angular_speed_rad_s=float(omega[p:].max()) if passed else None,post_pass_total_path_m=stop.get('post_pass_total_path_m') if passed else None,
                low_speed_candidate_latency_sec=stop.get('settle_latency_sec') if passed else None,low_speed_candidate_control_steps=len(rows)-candidate if candidate is not None else None,
                two_second_hold_complete=complete if passed else None,verified_two_second_drift_m=stop.get('max_two_second_drift_m') if complete else None,
                raw_partial_candidate_drift_m=stop.get('max_two_second_drift_m') if not complete else None,extra_path_until_candidate_m=stop.get('extra_path_until_stopped_m'),
                qualified_stop_latency_sec=stop.get('settle_latency_sec') if result['stopped'] else None,last_actual_speed_m_s=float(speed[-1]),
                backward_travel_after80_m=result.get('backward_travel_after80_m'),yaw_travel_after80_deg=result.get('yaw_travel_after80_deg'),snap_events=result['snap_events'],
                video=str(folder/'rollout.mp4'),expert_pair_video=str(r/'comparison'/('step_%06d'%step)/'expert_policy.mp4')))
            assert result['execution_ok'] and result['frames']==2500
            for kind,pv in [('raw',folder/'rollout.mp4'),('expert_pair',r/'comparison'/('step_%06d'%step)/'expert_policy.mp4')]:
                check=inspect_video(pv,1250);assert check['passed'];video_checks['%s_%d_%s'%(label,step,kind)]=check
            offline.extend(dict(m,branch=label,cumulative_updates=step,added_updates=step-30000) for m in json.loads((r/'offline'/('step_%06d'%step)/'summary.json').read_text())['metrics'])
            variation.extend(dict(m,branch=label,cumulative_updates=step,added_updates=step-30000) for m in json.loads((r/'offline'/('step_%06d'%step)/'bias_variation.json').read_text())['metrics'])
            comp=json.loads((r/'comparison'/('step_%06d'%step)/'summary.json').read_text())
            tracking.append(dict(comp['full_record_tracking'],branch=label,cumulative_updates=step,added_updates=step-30000))
            trajectory.extend(dict(m,branch=label,cumulative_updates=step,added_updates=step-30000) for m in comp['metrics'])
            timing.extend(dict(m,branch=label,cumulative_updates=step,added_updates=step-30000) for m in comp['stage_timing'])
        native=[json.loads(x) for x in (r/'training/metrics.jsonl').read_text().splitlines()]
        assert len(native)==1000 and [v['train/step'] for v in native]==list(range(30010,40001,10))
        assert all(v['train/lr']==lr and math.isfinite(v['train/loss']) for v in native)
        training.extend(dict(v,branch=label,added_updates=v['train/step']-30000,branch_optimization_sec=v['train/elapsed_sec']-source['training_state']['elapsed_sec']) for v in native)
        import wandb
        info=json.loads((r/'online/run.json').read_text());cloud=wandb.Api(timeout=30).run('fi-yu-individual/door-dp-stop-tail/'+info['id'])
        history=list(cloud.scan_history(keys=['train/step','train/loss','train/lr','branch/added_updates'],page_size=500))
        mapping={int(v['train/step']):v for v in history};assert len(mapping)==1000
        for row in native:
            found=mapping[row['train/step']]
            assert found['branch/added_updates']==row['train/step']-30000 and found['train/lr']==lr and math.isclose(found['train/loss'],row['train/loss'],rel_tol=1e-7,abs_tol=1e-10)
        online[label]=dict(url=info['url'],cloud_unique_training_rows=len(mapping),native_metrics_match=True,mode='online',sync_anomalies=len((r/'online/sync_anomalies.jsonl').read_text().splitlines()) if (r/'online/sync_anomalies.jsonl').exists() else 0)
    assert all(v['args']==eval_configs[0]['args'] for v in eval_configs)
    save('evaluation_config_pairing',eval_configs)
    for step in [30000,*NODES]:assert json.loads((root/('pair_%06d.json'%step)).read_text())['passed']
    for name,value in [('checkpoint_hashes',checkpoint),('tasks',tasks),('offline_physical_errors',offline),('bias_variation',variation),
        ('full_record_tracking',tracking),('trajectory_errors',trajectory),('stage_timing',timing),('initial_conditions',initial),('training_metrics',training)]:save(name,value)
    save('online_verification',online);save('videos',video_checks)
    differences=[]
    for step in NODES:
        for stage in dict.fromkeys(v['stage'] for v in offline):
            for horizon in [16,32]:
                c=next(v for v in offline if v['branch']=='C' and v['cumulative_updates']==step and v['stage']==stage and v['horizon']==horizon)
                l=next(v for v in offline if v['branch']=='L' and v['cumulative_updates']==step and v['stage']==stage and v['horizon']==horizon)
                b=next(v for v in offline if v['branch']=='baseline' and v['stage']==stage and v['horizon']==horizon)
                row=dict(cumulative_updates=step,added_updates=step-30000,stage=stage,horizon=horizon)
                for k in ['vx_rmse_m_s','yaw_rmse_rad_s','ee_position_rmse_m','ee_orientation_rmse_deg','gripper_rmse_rad']:
                    row[k]=dict(C=c[k],L=l[k],baseline=b[k],L_minus_C=l[k]-c[k],C_minus_baseline=c[k]-b[k],L_minus_baseline=l[k]-b[k])
                differences.append(row)
    save('paired_differences',differences)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for label in LRS:
        vals=[v for v in training if v['branch']==label];x=[v['added_updates'] for v in vals]
        for ax,key in zip(axes,['train/loss','train/lr','branch_optimization_sec']):ax.plot(x,[v[key] for v in vals],label=label,alpha=.75);ax.set(xlabel='Branch added optimizer updates',ylabel=key);ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'training_curves.png',dpi=140);plt.close(fig)
    for horizon in [16,32]:
        fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
        for ax,key in zip(axes.flat,['vx_rmse_m_s','yaw_rmse_rad_s','ee_position_rmse_m','ee_orientation_rmse_deg','gripper_rmse_rad']):
            for stage,style in [('all','-'),('conversion','--'),('hold',':')]:
                b=next(v for v in offline if v['branch']=='baseline' and v['stage']==stage and v['horizon']==horizon)
                for label,color in [('C','tab:blue'),('L','tab:orange')]:
                    vals=[v for v in offline if v['branch']==label and v['stage']==stage and v['horizon']==horizon]
                    ax.plot([0,*[v['added_updates'] for v in vals]],[b[key],*[v[key] for v in vals]],style,marker='o',color=color,label=label+' '+stage)
            ax.set(xlabel='Added updates',ylabel=key);ax.grid(alpha=.2);ax.legend(fontsize=7)
        axes[1,2].axis('off');fig.savefig(out/('physical_errors_h%d.png'%horizon),dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for row,channel in enumerate(['vx','yaw']):
        for col,key in enumerate(['mean_prediction_rmse','seed_variation_rms_std']):
            ax=axes[row,col]
            for label,color in [('C','tab:blue'),('L','tab:orange')]:
                for stage,style in [('all','-'),('conversion','--'),('hold',':')]:
                    b=next(v for v in variation if v['branch']=='baseline' and v['stage']==stage and v['horizon']==16)
                    vals=[v for v in variation if v['branch']==label and v['stage']==stage and v['horizon']==16]
                    ax.plot([0,*[v['added_updates'] for v in vals]],[b[channel][key],*[v[channel][key] for v in vals]],style,marker='o',color=color,label=label+' '+stage)
            ax.set(xlabel='Added updates',ylabel=channel+' '+key);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'bias_vs_seed_variation_h16.png',dpi=140);plt.close(fig)
    save('definitions',dict(source_updates=30000,added_each=10000,baseline_reused=True,no_new_baseline_sim=True,
        bias_variation='Mean over same3 diffusion seeds at each observation; RMSE of mean and RMS population standard deviation separately. Three fixed seeds, not a confidence interval.',
        physical_errors='Fixed19 windows,3 seeds; executed16/full32 separately; quaternion sign invariant angle.',
        temporal='Reference real common prefix only; task stage alignment supplementary, missing stages and delays retained.',
        stop='Original acceptance unchanged; qualified2-second drift null without full100-control-step hold; missing pass => stopNA; all six planned new episodes in denominator.',
        imitation_threshold=None,scope='One training episode, one paired training seed, one rollout per branch/node; not a generalization or repeatability test.'))
    lines=['# 学习率配对续训：完整证据','', '完整恢复同一30k状态，C=1e-4、L=1e-5，各新增10000真实优化；6个新闭环全部保留。','',
        '|组|累计/新增|门角°|原通过|通过秒|停车|联合|','|---|---:|---:|---|---:|---|---|']
    for v in sorted(tasks,key=lambda v:(v['cumulative_updates'],v['branch'])):lines.append('|%s|%d/+%d|%.5f|%s|%s|%s|%s|'%(v['branch'],v['cumulative_updates'],v['added_updates'],v['max_signed_door_deg'],v['passed'],v['first_pass_sec'],v['stopped'],v['joint']))
    lines+=['','不以总loss或单次任务事件替代动作/轨迹模仿。无预先新增误差阈值；连续误差、均值/种子波动、真实延迟、缺失阶段及末端退化分别审查。',
        '完整物理量及配对差值见offline_physical_errors与paired_differences，预测均值偏差及三固定seed波动见bias_variation。',
        '真实时间轨迹、动作后跟踪和阶段耗时分别见trajectory_errors、full_record_tracking、stage_timing；原始及参考对照视频见tasks。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    verify(root);print(json.dumps(dict(completed=True,updates_each=10000,new_rollouts=len(tasks),joint_successes=sum(v['joint'] for v in tasks),online=online),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repo',required=True);a=p.parse_args();main(Path(a.root).resolve(),Path(a.repo).resolve())
