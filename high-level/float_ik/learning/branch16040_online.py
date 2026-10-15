"""Independent A/B online metrics; branch new-update axis, no training RNG use."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid


def run(root,label):
    import wandb
    from stop_tail_online import atomic,metric_rows
    folder=root/label/'online';folder.mkdir(exist_ok=True)
    prior=json.loads((folder/'run.json').read_text()) if (folder/'run.json').exists() else None
    token=uuid.uuid4().hex
    active=wandb.init(entity='fi-yu-individual',project='door-dp-stop-tail',group=root.name,
        id=prior['id'] if prior else None,resume='allow' if prior else None,
        name=root.name+'-'+label,mode='online',dir=str(folder),save_code=False,
        config=dict(source_updates=5000,new_updates=2000,branch=label,initialization='model weights plus fixed stats; optimizer/sampler/RNG reset',
            observation_source=label,action_source='original16040',seed=42,batch_size=64,workers=0,lr=1e-4,horizons=[16,32,16]),
        settings=wandb.Settings(disable_git=True,disable_code=True,x_disable_stats=True,init_timeout=60))
    if active.settings.mode!='online':raise RuntimeError('Online W&B unavailable')
    active.define_metric('train/step')
    for key in ['train/*','offline/*','eval/*']:active.define_metric(key,step_metric='train/step')
    active.summary['preflight_token']=token
    atomic(folder/'run.json',dict(id=active.id,url=active.url,group=root.name,mode='online',pid=os.getpid()))
    deadline=time.time()+100
    while True:
        remote=wandb.Api(timeout=15).run('fi-yu-individual/door-dp-stop-tail/'+active.id)
        if remote.summary.get('preflight_token')==token:break
        if time.time()>deadline:raise RuntimeError('Cloud preflight token not readable')
        time.sleep(3)
    atomic(folder/'ready.json',dict(passed=True,url=active.url))
    accepted=set();last_train=0;last_offline=-1;last_eval=0
    if (folder/'events.jsonl').exists():
        for line in (folder/'events.jsonl').read_text().splitlines():
            event=json.loads(line);accepted.add(event['id']);step=event['event']['train/step']
            if event['id'].startswith('train:'):last_train=max(last_train,step)
            if event['id'].startswith('offline:'):last_offline=max(last_offline,step)
            if event['id'].startswith('eval:'):last_eval=max(last_eval,step)
    while True:
        events=[]
        for row in metric_rows([root/label/'training/metrics.jsonl']):events.append(('train:%d'%row['train/step'],row))
        for step in [0,500,1000,2000]:
            p=root/'offline'/('step_%06d'%step)/'summary.json'
            if p.exists():
                data={'train/step':step}
                for m in json.loads(p.read_text())['metrics']:
                    if m['model']!=label:continue
                    prefix='offline/on_%s/%s/h%d/'%(m['observation'],m['stage'],m['horizon'])
                    for k,v in m.items():
                        if isinstance(v,(float,int)) and k not in ['horizon','draws']:data[prefix+k]=v
                events.append(('offline:%d'%step,data))
            p=root/label/'pilot/evaluation'/('step_%06d'%step)/'seed_16040/result.json'
            if p.exists():
                m=json.loads(p.read_text());data={'train/step':step,'eval/opened':int(m['opened']),'eval/passed':int(m['pass_done']),
                    'eval/stop_evaluable':int(m['stop_evaluable']),'eval/joint':int(m['success'])}
                if m['stop_evaluable']:data['eval/stopped']=int(m['stopped'])
                for k,v in m['stopping'].items():
                    if isinstance(v,(float,int)) and not isinstance(v,bool):data['eval/'+k]=v
                events.append(('eval:%d'%step,data))
        for key,event in sorted(events,key=lambda x:x[1]['train/step']):
            if key in accepted:continue
            active.log(event);accepted.add(key)
            with (folder/'events.jsonl').open('a') as out:out.write(json.dumps(dict(id=key,event=event))+'\n')
            step=event['train/step']
            if key.startswith('train:'):last_train=max(last_train,step)
            if key.startswith('offline:'):last_offline=max(last_offline,step)
            if key.startswith('eval:'):last_eval=max(last_eval,step)
        active.summary.update(dict(last_train_step=last_train,last_offline_step=last_offline,last_eval_step=last_eval))
        completion=root/'completion.json'
        if completion.exists():
            c=json.loads(completion.read_text());active.summary['completion']=c['phase'];active.finish(exit_code=0 if c['phase']=='completed' else 1)
            cloud=wandb.Api(timeout=20).run('fi-yu-individual/door-dp-stop-tail/'+active.id)
            verified=int(cloud.summary.get('last_train_step',-1))==last_train and int(cloud.summary.get('last_offline_step',-2))==last_offline and int(cloud.summary.get('last_eval_step',-1))==last_eval
            atomic(folder/'finished.json',dict(verified=verified,url=cloud.url,last_train_step=last_train,last_offline_step=last_offline,last_eval_step=last_eval))
            if not verified:raise RuntimeError('Final online metrics not verified')
            return
        time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--label',choices=['A','B'],required=True);a=p.parse_args();run(Path(a.root).resolve(),a.label)
