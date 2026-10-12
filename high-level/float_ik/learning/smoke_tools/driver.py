#!/usr/bin/env python3
"""Sequential, resumable experiment driver. No formal training is dispatched."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import os
import shlex
import time

p=argparse.ArgumentParser()
p.add_argument('phase', choices=['prepare','collect','train','final'])
p.add_argument('--root',required=True)
p.add_argument('--target',type=int,default=100)
p.add_argument('--episodes',type=int,default=20)
a=p.parse_args()
r=Path(a.root).resolve()
repo=Path('/data/zyf/video2door')
rollout=repo/'high-level/float_ik/learning/rollout.py'
trainer=repo/'high-level/float_ik/learning/train.py'
(r/'logs').mkdir(exist_ok=True)

samples=[]
for _ in range(3):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    samples.append([[float(v.strip()) for v in line.split(',')] for line in raw.strip().splitlines()])
    time.sleep(2)
choices=[]
for i in range(len(samples[0])):
    free=min(s[i][1] for s in samples)
    load=sum(s[i][2] for s in samples)/3
    if free>=16384 and load<70:
        choices.append((load,int(samples[0][i][0])))
(r/('resource_%s_%s_%d.json'%(a.phase,a.target,int(time.time())))).write_text(json.dumps(samples))
if not choices:
    raise SystemExit('No eligible GPU; stop without touching other jobs')
os.environ['VIDEO2DOOR_GPU']=str(min(choices)[1])
print('SELECT_GPU',os.environ['VIDEO2DOOR_GPU'],flush=True)

def used(stage):
    rows=[json.loads(x) for x in (r/'budget.jsonl').read_text().splitlines()]
    return sum(x['elapsed_sec'] for x in rows if x['stage']==stage)

def run(label,stage,limit,timeout,command):
    if Path(command[1]).name in ('rollout.py', 'train.py'):
        command = list(command) + ['--schema', 'float_push_joint_v1']
    with (r/'logs'/(label+'.log')).open('x') as f:
        cmd=[sys.executable,str(r/'tools/bounded.py'),'--root',str(r),'--stage',stage,'--limit',str(limit),'--timeout',str(timeout),'--','bash','-c','source /data/zyf/video2door/.repro/blackwell/activate.sh; exec nice -n 10 '+shlex.join(list(map(str,command)))]
        print('START',label,flush=True)
        rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=str(repo))
        print('END',label,rc,'stage_used',used(stage),flush=True)
    return rc

def ep(label,mode,seed,stage,limit,timeout=150,extras=()):
    target=r/label
    if (target/'summary.json').exists():
        return json.loads((target/'summary.json').read_text())
    cmd=[sys.executable,rollout,'--output',target,'--mode',mode,'--seed',seed]+list(extras)
    run(label.replace('/','_'),stage,limit,timeout,cmd)
    if not (target/'summary.json').exists():
        target.mkdir(parents=True,exist_ok=True)
        (target/'summary.json').write_text(json.dumps(dict(success=False,status='launch_error',seed=seed,mode=mode)))
    s=json.loads((target/'summary.json').read_text())
    if s.get('status')=='started':
        s.update(status='timeout',success=False,failure_type='timeout')
        (target/'summary.json').write_text(json.dumps(s,indent=2))
    return s

if a.phase=='prepare':
    s=ep('adapter-replay','replay',0,'baseline',300,extras=['--nominal','--replay',r/'adapter-nominal/episode.npz'])
    assert s['success'],s
    assert abs(s['signed_angle_deg']-89.98025)<.1
elif a.phase=='collect':
    manifest=[]
    for i in range(a.episodes):
        if used('data')>1400:
            break
        split='train' if i%10<7 else 'val' if i%10<9 else 'test'
        label='data/episode_%03d'%i
        s=ep(label,'expert',1000+i,'data',1500,timeout=100,extras=['--record','--split',split])
        manifest.append(dict(episode=str(r/label/'episode.npz'),seed=1000+i,split=split,success=bool(s.get('success')),status=s['status']))
        (r/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if s['status']!='completed':
            raise SystemExit('Collection infrastructure error; stop for diagnosis')
    print('COLLECT',json.dumps(manifest),flush=True)
elif a.phase=='train':
    latest=r/'training/latest.json'
    command=[sys.executable,trainer,'--manifest',r/'manifest.json','--output',r/'training','--steps',a.target,'--seconds',max(1,3500-used('train'))]
    if latest.exists():
        command+=['--resume',json.loads(latest.read_text())['checkpoint']]
    rc=run('train_to_%06d'%a.target,'train',3600,max(1,3550-used('train')),command)
    if rc:
        raise SystemExit(rc)
    path=r/'training/checkpoints'/('step_%06d.pt'%a.target)
    if not path.exists():
        raise SystemExit('Target not reached within budget')
    points=[a.target] if a.target==100 else ([0,a.target] if a.target==500 else [a.target])
    for point in points:
        for j in range(1 if point==100 else 5):
            if used('train')>3450:
                raise SystemExit('Training/evaluation budget reserved for shutdown')
            s=ep('validation/step_%06d_seed_%d'%(point,2000+j),'policy',2000+j,'train',3600,timeout=150,extras=['--checkpoint',r/'training/checkpoints'/('step_%06d.pt'%point)])
            if s['status']!='completed':
                raise SystemExit('Evaluation execution failed')
elif a.phase=='final':
    groups={}
    for path in (r/'validation').glob('*/summary.json'):
        s=json.loads(path.read_text())
        if s.get('checkpoint_step',0)>0:
            groups.setdefault(s['checkpoint_step'],[]).append(s)
    train_rows=[json.loads(x) for x in (r/'training/train.jsonl').read_text().splitlines()]
    loss={x['step']:x.get('val_loss',float('inf')) for x in train_rows}
    eligible={k:v for k,v in groups.items() if len(v)>=5 and all(x['status']=='completed' for x in v)}
    assert eligible,'No complete validation set'
    best=max(eligible,key=lambda k:(sum(x.get('success',False) for x in eligible[k]),sum(x.get('signed_angle_deg',0) for x in eligible[k])/len(eligible[k]),-loss.get(k,float('inf'))))
    (r/'selection.json').write_text(json.dumps(dict(best_step=best,rule='final_success_then_final_angle_then_val_loss',validation_seeds=list(range(2000,2005)),test_seeds=list(range(3000,3010))),indent=2))
    # Round robin avoids spending all final budget on just one method.
    for j in range(10):
        for mode,point in [('expert',None),('policy',0),('policy',best)]:
            if used('delivery')>1680:
                raise SystemExit('Delivery budget reserve reached')
            label='test/%s_%s_seed_%d'%(mode,point,3000+j)
            extra=[] if mode=='expert' else ['--checkpoint',r/'training/checkpoints'/('step_%06d.pt'%point)]
            if j==0:
                extra+=['--video']
            ep(label,mode,3000+j,'delivery',1800,timeout=120,extras=extra)
