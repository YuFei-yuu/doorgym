"""Independent online W&B reader: never imports the trainer or changes its RNG."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import uuid
import fcntl


class MetricIntegrityError(ValueError):
    pass


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def metric_rows(paths):
    result={}
    for path in paths:
        if not path.exists():continue
        content=path.read_text()
        lines=content.splitlines()
        if content and not content.endswith('\n'):lines=lines[:-1]
        for line in lines:
            try:value=json.loads(line);step=int(value['train/step'])
            except (ValueError,KeyError,TypeError) as exc:raise MetricIntegrityError('Malformed complete metric record') from exc
            if not all(math.isfinite(value[key]) for key in ['train/loss','train/lr','train/elapsed_sec']):
                raise FloatingPointError('Nonfinite native metric at update %d'%step)
            if step in result and result[step]!=value:raise MetricIntegrityError('Conflicting metrics for same actual update')
            result[step]=value
    return [result[step] for step in sorted(result)]


def smoothed(rows,alpha=.1):
    ema=None;out=[]
    for row in rows:
        ema=row['train/loss'] if ema is None else alpha*row['train/loss']+(1-alpha)*ema
        out.append(dict(row,**{'train/loss_raw':row['train/loss'],'train/loss_ema':ema}))
    return out


def run(root):
    import wandb
    spec=json.loads((root/'online_spec.json').read_text());folder=root/'online';folder.mkdir(exist_ok=True)
    ids=set(json.loads((folder/'accepted_ids.json').read_text())) if (folder/'accepted_ids.json').exists() else set()
    active=None;last_probe=0.;api=None;ready=(folder/'ready.json').exists()
    token=spec['preflight_token'];last_error=None
    while True:
        try:
            if active is None:
                active=wandb.init(project='door-dp-stop-tail',entity=spec['entity'],id=spec['run_id'],resume='allow',
                    name=root.name,dir=str(folder),mode='online',save_code=False,
                    config=dict(resume_update=1000,target_update=5000,new_updates=4000,batch_size=64,num_workers=0,
                        lr=1e-4,action_contract='forward_v3',horizons=[16,32,16],dataset_frames=4321,dataset_windows=4228,
                        smoothing='EWMA alpha0.1; display only',checkpoint_sha256=spec['checkpoint_sha256']),
                    settings=wandb.Settings(disable_git=True,disable_code=True,x_disable_stats=True,init_timeout=60))
                if active.settings.mode!='online':raise RuntimeError('Online mode was not established')
                active.define_metric('train/step')
                for name in ['train/*','eval/*','diagnostic/*','resume/*']:
                    active.define_metric(name,step_metric='train/step')
                active.summary['monitor/preflight_token']=token
                atomic(folder/'run.json',dict(entity=spec['entity'],project='door-dp-stop-tail',id=active.id,url=active.url,mode='online',pid=os.getpid()))
                api=wandb.Api(timeout=10)
            events=[]
            events.append(('baseline:0',{'train/step':0,'resume/baseline_marker':1}))
            rows=smoothed(metric_rows([Path(spec['baseline_metrics']),root/'pilot/training/metrics.jsonl']))
            for row in rows:events.append(('train:%d'%row['train/step'],row))
            events.append(('resume:1000',{'train/step':1000,'resume/start_marker':1}))
            latest=int(rows[-1]['train/step']) if rows else 0
            for step in [1000,2000,3000,5000]:
                suite=Path(spec['baseline_evaluation']) if step==1000 else root/'pilot/evaluation'/('step_%06d'%step)
                if (suite/'aggregate.json').exists():
                    a=json.loads((suite/'aggregate.json').read_text());n=a['attempts']
                    events.append(('eval:%d'%step,{'train/step':step,'eval/open_rate':a['opened']/n,
                        'eval/pass_rate':a['passed']/n,'eval/stop_rate':a['stopped']/n,'eval/joint_rate':a['successes']/n,
                        'eval/stop_evaluable':a['stop_evaluable'],'eval/attempts':n}))
                    for row in json.loads((suite/'results.json').read_text()):
                        stopping=row['stopping'];seed=row['seed'];prefix='eval/seed_%d/'%seed
                        data={'train/step':step,prefix+'opened':int(row['opened']),prefix+'passed':int(row['pass_done']),
                              prefix+'joint':int(row['success']),prefix+'stop_evaluable':int(row['pass_done'])}
                        if row['pass_done']:
                            data[prefix+'stopped']=int(row['stopped'])
                            for key in ['settle_latency_sec','post_pass_total_path_m','extra_path_until_stopped_m','max_two_second_drift_m','max_remaining_position_drift_m']:
                                if stopping.get(key) is not None:data[prefix+key]=stopping[key]
                        events.append(('case:%d:%d'%(step,seed),data))
                diagnostic=root/'diagnostics'/('step_%06d/summary.json'%step)
                if diagnostic.exists():
                    value=json.loads(diagnostic.read_text());data={'train/step':step}
                    for stage,metrics in value['stages'].items():
                        data['diagnostic/'+stage+'/vx_rmse']=metrics['vx_rmse'];data['diagnostic/'+stage+'/yaw_rmse']=metrics['yaw_rmse']
                    events.append(('diagnostic:%d'%step,data))
            events.sort(key=lambda item:item[1]['train/step'])
            for key,event in events:
                if key in ids:continue
                with (folder/'events.jsonl').open('a') as journal:journal.write(json.dumps(dict(event_id=key,values=event,at=time.time()),allow_nan=False)+'\n')
                active.log(event)
                ids.add(key);atomic(folder/'accepted_ids.json',sorted(ids))
            active.summary['monitor/last_train_step']=latest
            last_eval=max([int(key.split(':')[1]) for key in ids if key.startswith('eval:')]+[0])
            last_diagnostic=max([int(key.split(':')[1]) for key in ids if key.startswith('diagnostic:')]+[0])
            active.summary['monitor/last_eval_step']=last_eval
            active.summary['monitor/last_diagnostic_step']=last_diagnostic
            if time.time()-last_probe>=20:
                # Public Run.summary is cached separately from Run.load().
                # A fresh API/Run object is necessary for a fresh summary read.
                cloud=wandb.Api(timeout=10).run('%s/door-dp-stop-tail/%s'%(spec['entity'],spec['run_id']))
                remote=int(cloud.summary.get('monitor/last_train_step',0))
                connected=cloud.summary.get('monitor/preflight_token')==token
                status=dict(phase='connected' if connected and latest-remote<=30 else 'sync_lag',mode='online',
                    latest_local_step=latest,latest_cloud_step=remote,url=active.url,last_cloud_check=time.time(),last_error=None)
                atomic(folder/'status.json',status);last_probe=time.time()
                last_error=None
                if not ready and connected and remote>=1000:
                    atomic(folder/'ready.json',dict(passed=True,url=active.url,verified_step=remote,at=time.time()));ready=True
                print(json.dumps(status),flush=True)
            complete=root/'completion.json'
            if complete.exists():
                done=json.loads(complete.read_text())
                if done.get('phase') in ['completed_5000_update_resume','failed_stopped']:
                    active.summary['monitor/completion_phase']=done['phase'];active.finish(exit_code=0 if done['phase'].startswith('completed') else 1)
                    active=None
                    final=wandb.Api(timeout=20).run('%s/door-dp-stop-tail/%s'%(spec['entity'],spec['run_id']))
                    remote=int(final.summary.get('monitor/last_train_step',0))
                    atomic(folder/'finished.json',dict(phase='online_finished',verified_cloud_step=remote,expected_step=latest,
                        verified=remote>=latest and int(final.summary.get('monitor/last_eval_step',0))>=last_eval and
                            int(final.summary.get('monitor/last_diagnostic_step',0))>=last_diagnostic,url=final.url,at=time.time()))
                    return
        except (FloatingPointError,MetricIntegrityError) as exc:
            atomic(folder/'status.json',dict(phase='input_error',mode='online',error_type=type(exc).__name__,error=str(exc),at=time.time()))
            raise
        except Exception as exc:
            last_error=type(exc).__name__
            atomic(folder/'status.json',dict(phase='sync_error',mode='online',error_type=last_error,at=time.time(),offline_fallback=False))
            print(json.dumps(dict(phase='sync_error',error_type=last_error,retry_seconds=15)),flush=True)
        time.sleep(15 if last_error else 5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args();root=Path(args.root).resolve()
    (root/'online').mkdir(exist_ok=True)
    with (root/'online/logger.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(root)
