"""Independent online monitor with local event journal and explicit reconnects."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
import uuid


def run(root):
    import wandb
    from stop_tail_online import atomic,metric_rows,smoothed,MetricIntegrityError
    from lr_pair import NODES,LRS
    folder=root/'online';folder.mkdir(exist_ok=True)
    prior=json.loads((folder/'run.json').read_text()) if (folder/'run.json').exists() else None
    accepted=set();last_train=0;last_offline=0;last_eval=0;last_comparison=0
    if (folder/'events.jsonl').exists():
        for line in (folder/'events.jsonl').read_text().splitlines():
            event=json.loads(line);accepted.add(event['id']);step=event['values']['train/step']
            if event['id'].startswith('train:'):last_train=max(last_train,step)
            elif event['id'].startswith('offline:'):last_offline=max(last_offline,step)
            elif event['id'].startswith('eval:'):last_eval=max(last_eval,step)
            elif event['id'].startswith('comparison:'):last_comparison=max(last_comparison,step)
    active=None;last_probe=0.;token=uuid.uuid4().hex;ready=False
    while True:
        try:
            if active is None:
                active=wandb.init(entity='fi-yu-individual',project='door-dp-stop-tail',group=root.parent.name,
                    id=prior['id'] if prior else None,resume='allow' if prior else None,name=root.parent.name+'-'+root.name,mode='online',dir=str(folder),save_code=False,
                    config=dict(experiment_kind='paired_complete_resume_lr_only',branch=root.name,source_updates=30000,added_updates=10000,
                        initialization='FULL model/AdamW/sampler/RNG from same30k; LR only explicit branch difference',
                        batch_size=8,num_workers=0,lr=LRS[root.name],seed=42,horizons=[16,32,16],DDPM_steps=100),
                    settings=wandb.Settings(disable_git=True,disable_code=True,x_disable_stats=True,init_timeout=60))
                if active.settings.mode!='online':raise RuntimeError('Online mode not established')
                active.define_metric('train/step')
                active.define_metric('branch/added_updates')
                for name in ['train/*','offline/*','eval/*','closed/*','stage_timing/*','variation/*']:active.define_metric(name,step_metric='branch/added_updates')
                prior=dict(id=active.id,url=active.url,experiment_kind='paired_complete_resume_lr_only',pid=os.getpid(),mode='online')
                atomic(folder/'run.json',prior);active.summary['monitor/preflight_token']=token
            events=[]
            for row in smoothed(metric_rows([root/'training/metrics.jsonl'])):events.append(('train:%d'%row['train/step'],row))
            for step in [30000,*NODES]:
                p=(root.parent/'baseline/summary.json') if step==30000 else root/'offline'/('step_%06d'%step)/'summary.json'
                if p.exists():
                    values={'train/step':step}
                    for m in json.loads(p.read_text())['metrics']:
                        prefix='offline/%s/h%d/'%(m['stage'],m['horizon'])
                        for k in ['vx_rmse_m_s','yaw_rmse_rad_s','ee_position_rmse_m','ee_orientation_rmse_deg','gripper_rmse_rad']:values[prefix+k]=m[k]
                        for index,k in enumerate(['ee_x_rmse_m','ee_y_rmse_m','ee_z_rmse_m']):values[prefix+k]=m['ee_xyz_rmse_m'][index]
                    events.append(('offline:%d'%step,values))
                p=root/'pilot/evaluation'/('step_%06d'%step)/'seed_16040/result.json'
                if p.exists():
                    m=json.loads(p.read_text());values={'train/step':step,'eval/opened':int(m['opened']),'eval/passed':int(m['pass_done']),
                        'eval/stop_evaluable':int(m['stop_evaluable']),'eval/joint':int(m['success']),'eval/max_signed_door_deg':m['max_signed_door_deg'],
                        'eval/complete_2500':int(m['execution_ok'])}
                    if m['stop_evaluable']:values['eval/stopped']=int(bool(m['stopped']))
                    if m.get('first_pass_frame') is not None:values['eval/first_pass_time_sec']=m['first_pass_frame']*.02
                    for k,v in m['stopping'].items():
                        if isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v):
                            if k=='max_two_second_drift_m' and not m['stopping'].get('two_second_hold_complete',False):continue
                            values['eval/'+k]=v
                    events.append(('eval:%d'%step,values))
                p=(root.parent/'baseline/bias_variation.json') if step==30000 else root/'offline'/('step_%06d'%step)/'bias_variation.json'
                if p.exists():
                    values={'train/step':step}
                    for row in json.loads(p.read_text())['metrics']:
                        for channel in ['vx','yaw']:
                            for k,v in row[channel].items():
                                if isinstance(v,(float,int)):values['variation/%s/h%d/%s/%s'%(row['stage'],row['horizon'],channel,k)]=v
                    events.append(('variation:%d'%step,values))
                p=root/'comparison'/('step_%06d'%step)/'summary.json'
                if p.exists():
                    m=json.loads(p.read_text());values={'train/step':step}
                    for k,v in m['full_record_tracking'].items():
                        if isinstance(v,(float,int)) and not isinstance(v,bool):values['closed/full50/'+k]=v
                    for row in m['metrics']:
                        if row['alignment']=='true_time':
                            for k,v in row.items():
                                if isinstance(v,(float,int)) and not isinstance(v,bool):values['closed/true_time/'+k]=v
                    for row in m['stage_timing']:
                        prefix='stage_timing/phase_%d/'%row['stage'];values[prefix+'missing']=int(row['model_missing_stage'])
                        for k in ['stage_start_delay_sec','model_duration_sec','reference_duration_sec']:
                            if k in row:values[prefix+k]=row[k]
                    events.append(('comparison:%d'%step,values))
            for key,values in sorted(events,key=lambda event:event[1]['train/step']):
                if key in accepted:continue
                if values['train/step']==30000:values['baseline/reused_source_result']=1
                values['branch/added_updates']=values['train/step']-30000
                values['train/cumulative_optimizer_updates']=values['train/step']
                active.log(values)
                with (folder/'events.jsonl').open('a') as out:out.write(json.dumps(dict(id=key,values=values,at=time.time()),allow_nan=False)+'\n')
                accepted.add(key);step=values['train/step']
                if key.startswith('train:'):last_train=max(last_train,step)
                elif key.startswith('offline:'):last_offline=max(last_offline,step)
                elif key.startswith('eval:'):last_eval=max(last_eval,step)
                elif key.startswith('comparison:'):last_comparison=max(last_comparison,step)
            active.summary.update(dict(last_train_step=last_train,last_offline_step=last_offline,last_eval_step=last_eval,last_comparison_step=last_comparison))
            if time.time()-last_probe>25:
                cloud=wandb.Api(timeout=15).run('fi-yu-individual/door-dp-stop-tail/'+active.id)
                connected=cloud.summary.get('monitor/preflight_token')==token
                remote=int(cloud.summary.get('last_train_step',0));last_probe=time.time()
                atomic(folder/'status.json',dict(phase='connected' if connected and last_train-remote<=100 else 'sync_lag',
                    local_updates=last_train,cloud_updates=remote,mode='online',url=active.url,at=time.time()))
                if connected and not ready:
                    atomic(folder/'ready.json',dict(passed=True,url=active.url,cloud_token_verified=True));ready=True
                    print('ONLINE_READY '+active.url,flush=True)
            p=root/'completion.json'
            if p.exists():
                done=json.loads(p.read_text());active.summary['completion']=done['phase'];active.finish(exit_code=0 if done['phase']=='completed' else 1)
                active=None
                cloud=wandb.Api(timeout=20).run('fi-yu-individual/door-dp-stop-tail/'+prior['id'])
                verified=all(int(cloud.summary.get(k,-1))==v for k,v in [('last_train_step',last_train),('last_offline_step',last_offline),('last_eval_step',last_eval),('last_comparison_step',last_comparison)])
                atomic(folder/'finished.json',dict(verified=verified,url=cloud.url,train_step=last_train,offline_step=last_offline,eval_step=last_eval,comparison_step=last_comparison))
                if not verified:raise RuntimeError('Final online state not yet readable')
                return
        except (FloatingPointError,MetricIntegrityError):raise
        except Exception as exc:
            with (folder/'sync_anomalies.jsonl').open('a') as out:out.write(json.dumps(dict(at=time.time(),error=repr(exc),offline_fallback=False))+'\n')
            atomic(folder/'status.json',dict(phase='sync_error',mode='online',error=repr(exc),at=time.time(),offline_fallback=False))
            print('ONLINE_RETRY '+type(exc).__name__,flush=True);time.sleep(10)
        time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root).resolve();(root/'online').mkdir(exist_ok=True)
    with (root/'online/logger.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(root)
