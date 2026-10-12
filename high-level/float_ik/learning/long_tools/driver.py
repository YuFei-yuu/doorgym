#!/usr/bin/env python3
"""Single-GPU formal continuation; immutable outputs and mandatory final reserve."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import time
import traceback
from bounded import atomic_json, read_rows, reconcile, run, used
from resume_state import prepare_resume, patience_state

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--old', required=True)
p.add_argument('--repo', default='/data/zyf/video2door')
p.add_argument('--resume-continuation',type=int,metavar='STEP',help='Explicitly reopen a resource-stopped provisional selection; preserve original ledger and checkpoints')
a = p.parse_args()
r, old, repo = Path(a.root).resolve(), Path(a.old).resolve(), Path(a.repo)
learning = repo/'high-level/float_ik/learning'
r.mkdir(parents=True, exist_ok=True)
(r/'logs').mkdir(exist_ok=True)
driver_lock = (r/'driver.lock').open('a')
fcntl.flock(driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
reconcile(r)
TOTAL = 21600
gpu = uuid = None
sequence = len(read_rows(r))
config = dict(run_id=r.name, old_run=str(old), total_seconds=TOTAL,
              max_new_optimizer_steps=20000, start_step=5000,max_step=25000,
              collect_new_training_episodes=False,inference_steps=25,
              diagnostic_inference_steps=[25,100],validation_seeds=list(range(4000,4010)),
              test_seeds=list(range(5000,5030)),validation_every_new_steps=2000,
              angle_improvement_tolerance_deg=.001,patience=3,
              min_final_reserve_seconds=7200,seed=42,batch_size=16,lr=1e-4,
              weight_decay=.01,nice=10,threads=2,environment=str(repo/'.repro/blackwell/activate.sh'))
if (r/'config.json').exists():
    assert json.loads((r/'config.json').read_text()) == config
else:
    atomic_json(r/'config.json', config)


def status(phase, **kwargs):
    value = dict(phase=phase, updated=time.time(),used_seconds=used(r),**kwargs)
    atomic_json(r/'status.json',value)
    print('STATUS',json.dumps(value),flush=True)


def resource(phase):
    global gpu, uuid
    samples=[]
    for i in range(3):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        samples.append([dict(index=int(v[0]),uuid=v[1],free_mib=float(v[2]),util=float(v[3])) for v in ([x.strip() for x in line.split(',')] for line in raw.strip().splitlines())])
        if i<2:time.sleep(2)
    candidates=[]
    for item in samples[0]:
        entries=[next(x for x in sample if x['uuid']==item['uuid']) for sample in samples]
        load=statistics.mean(x['util'] for x in entries)
        if min(x['free_mib'] for x in entries)>=16384 and load<70:
            candidates.append((load,item['index'],item['uuid']))
    path=r/('resource_%s_%d.json'%(phase,time.time_ns()))
    atomic_json(path,dict(samples=samples,candidates=candidates))
    if not candidates:
        raise RuntimeError('No eligible GPU; resource stop without waiting')
    _,gpu,uuid=min(candidates)
    status(phase,gpu=gpu,gpu_uuid=uuid)


def launch(label,phase,args,timeout=240,reserve=0):
    global sequence
    if Path(args[1]).name in ('rollout.py', 'train.py'):
        args = list(args) + ['--schema', 'float_push_joint_v1']
    sequence+=1
    shell='export VIDEO2DOOR_GPU=%d; source %s || exit; exec nice -n 10 %s'%(gpu,shlex.quote(str(repo/'.repro/blackwell/activate.sh')),shlex.join(list(map(str,args))))
    return run(r,phase,['bash','-c',shell],timeout,total=TOTAL,reserve=reserve,
               gpu=gpu,gpu_uuid=uuid,log_name='%04d_%s.log'%(sequence,label))


def timings():
    expert,policy=40.314,60.089
    for row in read_rows(r):
        cmd=' '.join(row['command'])
        if 'rollout.py' not in cmd or row['exit_code']!=0:continue
        if '--mode expert' in cmd:expert=max(expert,row['elapsed_sec'])
        elif '--inference-steps 25' in cmd:policy=max(policy,row['elapsed_sec'])
    return expert,policy


def final_reserve():
    expert,policy=timings()
    value=max(7200,1.25*(30*(expert+3*policy)+4*max(expert,policy)))
    atomic_json(r/'reserve.json',dict(seconds=value,expert_process_sec=expert,policy25_process_sec=policy,
                                    formula='max(7200,1.25*(30*(E+3*P)+4*max(E,P)))'))
    return value


def episode(label,mode,seed,phase,checkpoint=None,inference=25,video=False,record=False,reserve=0,diagnostic=0):
    output=r/label
    summary=output/'summary.json'
    if summary.exists():
        value=json.loads(summary.read_text())
        if value['status']=='started':
            value.update(status='interrupted',success=False,failure_type='interrupted')
            atomic_json(summary,value)
        return value
    args=['python',learning/'rollout.py','--output',output,'--mode',mode,'--seed',seed,
          '--inference-steps',inference,'--diagnostic-steps',diagnostic,'--split',phase]
    if checkpoint:args+=['--checkpoint',checkpoint]
    if video:args+=['--video']
    if record:args+=['--record']
    # Full-process limits include initialization and serialization. Use measured
    # 100-step timings only for diagnostics, never change formal test length.
    e,p=timings()
    timeout=max(180,3*e+30) if mode=='expert' else max(240,3*p+30)
    if inference==100:timeout=max(420,4*p+90)
    rc=launch(label.replace('/','_'),phase,args,timeout,reserve)
    if not summary.exists():
        output.mkdir(exist_ok=True)
        atomic_json(summary,dict(status='launch_error',success=False,mode=mode,seed=seed,
                                checkpoint=str(checkpoint) if checkpoint else None,
                                inference_steps=inference,failure_type='launch_error'))
    value=json.loads(summary.read_text())
    if rc or value['status']=='started':
        value.update(status='timeout' if rc==124 else 'error',success=False,
                     failure_type='timeout' if rc==124 else 'infrastructure_or_invalid_action')
        atomic_json(summary,value)
    print('EPISODE',json.dumps(dict(label=label,status=value['status'],success=value.get('success'),angle=value.get('signed_angle_deg'))),flush=True)
    return value


def checkpoint(step):
    return r/'training/checkpoints'/('step_%06d.pt'%step)


def val(step):
    resource('validation_%d'%step)
    reserve=final_reserve()
    values=[]
    for seed in config['validation_seeds']:
        value=episode('validation/step_%06d_seed_%d'%(step,seed),'policy',seed,'validation',checkpoint(step),reserve=reserve,diagnostic=60)
        values.append(value)
        if value['status']!='completed':raise RuntimeError('Validation infrastructure failure')
    old_loss=json.loads((old/'training/train.jsonl').read_text().splitlines()[-1])['val_loss']
    losses={5000:old_loss}
    if (r/'training/train.jsonl').exists():
        for row in map(json.loads,(r/'training/train.jsonl').read_text().splitlines()):
            if 'val_loss' in row:losses[row['step']]=row['val_loss']
    point=dict(step=step,attempts=10,successes=sum(x['success'] for x in values),
               mean_final_angle_deg=statistics.mean(x['signed_angle_deg'] for x in values),
               val_loss=losses.get(step),seeds=config['validation_seeds'])
    atomic_json(r/('validation_point_%06d.json'%step),point)
    return point


def selection(reason):
    points=[json.loads(x.read_text()) for x in sorted(r.glob('validation_point_*.json'))]
    if not points:raise RuntimeError('No complete validation set; test selection unavailable')
    best=max(points,key=lambda x:(x['successes'],x['mean_final_angle_deg'],-x['val_loss'],-x['step']))
    last=max(int(x.stem.split('_')[1]) for x in (r/'training/checkpoints').glob('step_*.pt'))
    result=dict(best_step=best['step'],last_step=last,rule='success_then_final_angle_then_loss_then_earlier',
                validation_seeds=config['validation_seeds'],test_seeds=config['test_seeds'],
                stop_reason=reason,frozen_at=time.time(),best_equals_last=best['step']==last,
                status='frozen_before_independent_test',last_has_complete_closed_loop_validation=any(x['step']==last for x in points))
    if (r/'selection.json').exists():return json.loads((r/'selection.json').read_text())
    atomic_json(r/'selection.json',result)
    return result


reason='max_optimizer_steps'
try:
    if a.resume_continuation:
        resumed=prepare_resume(r,a.resume_continuation)
        status('resume_audited',resume_step=a.resume_continuation,consecutive_no_improvement=resumed['consecutive_no_improvement_before'])
    if not (r/'preflight_passed.json').exists():
        resource('preflight')
        if (r/'provenance/learning-source.tar.gz').exists():
            shutil.copytree(r/'provenance',r/('preflight_revision_%d'%time.time_ns()))
        (r/'provenance').mkdir(exist_ok=True)
        shutil.copy2(old/'manifest.json',r/'manifest.json')
        (r/'training/checkpoints').mkdir(parents=True,exist_ok=True)
        for step in (0,5000):
            if not checkpoint(step).exists():shutil.copy2(old/'training/checkpoints'/checkpoint(step).name,checkpoint(step))
        for label,args in [
            ('budget_tests',['python',learning/'long_tools/test_budget.py']),
            ('provenance',['python',learning/'smoke_tools/provenance.py',r]),
            ('audit_data',['python',learning/'long_tools/audit.py','--root',r,'--old',old,'--phase','data']),
            ('resume_no_updates',['python',learning/'train.py','--manifest',r/'manifest.json','--output',r/'resume-audit','--steps',5000,'--resume',checkpoint(5000),'--seconds',1]),
        ]:
            if launch(label,'preflight',args,timeout=180,reserve=final_reserve()):raise RuntimeError('Preflight failed: '+label)
        for inference in (25,100):
            args=['python',learning/'check_policy.py','--checkpoint',checkpoint(5000),'--episode',old/'recording-audit/episode.npz','--inference-steps',inference]
            if launch('policy_equivalence_%d'%inference,'preflight',args,180,final_reserve()):raise RuntimeError('Policy equivalence failed')
        atomic_json(r/'preflight_passed.json',dict(passed=True,time=time.time()))

    if not (r/'audit_early.json').exists():
        resource('diagnostic')
        e=episode('diagnostic/expert_seed_4000','expert',4000,'diagnostic',record=True,diagnostic=60,reserve=final_reserve())
        if e['status']!='completed' or not e['success']:raise RuntimeError('Diagnostic expert failed')
        for seed in range(4000,4003):
            for inference in (25,100):
                label='diagnostic/policy_5000_d%d_seed_%d'%(inference,seed)
                value=episode(label,'policy',seed,'diagnostic',checkpoint(5000),inference=inference,diagnostic=60,reserve=final_reserve())
                if value['status']!='completed':raise RuntimeError('Diagnostic infrastructure failure')
                if inference==25:
                    dest=r/('validation/step_005000_seed_%d'%seed)
                    dest.parent.mkdir(exist_ok=True)
                    if not dest.exists():dest.symlink_to(Path('../diagnostic')/Path(label).name)
        args=['python',learning/'long_tools/audit.py','--root',r,'--old',old,'--phase','early']
        if launch('audit_early','diagnostic',args,180,final_reserve()):raise RuntimeError('Early contract audit failed')
        args[-1]='offline'
        if launch('offline_paired','diagnostic',args,180,final_reserve()):raise RuntimeError('Offline diagnostics failed')
    if not json.loads((r/'audit_early.json').read_text())['passed']:raise RuntimeError('Early contract audit failed')

    if not (r/'selection.json').exists():
        initial=r/'validation_point_005000.json'
        points=[json.loads(initial.read_text()) if initial.exists() else val(5000)]
        stale=0
        best_success=points[0]['successes']; best_angle=points[0]['mean_final_angle_deg']
        for target in range(7000,25001,2000):
            prior=target-2000
            if not checkpoint(prior).exists():raise RuntimeError('Missing preceding checkpoint')
            if not (r/('validation_point_%06d.json'%target)).exists():
                e,p=timings()
                if checkpoint(target).exists():
                    if TOTAL-used(r)<final_reserve()+1.25*10*p:
                        reason='final_test_reserve_before_pending_validation';break
                    status('pending_validation',checkpoint_step=target)
                else:
                    trained_updates=len((r/'training/train.jsonl').read_text().splitlines()) if (r/'training/train.jsonl').exists() else 0
                    if trained_updates+2000>config['max_new_optimizer_steps']:
                        reason='max_new_optimizer_updates_including_retries';break
                    cycle=1.25*(10*p+300)
                    if TOTAL-used(r)<final_reserve()+cycle:
                        reason='final_test_reserve';break
                    resource('train_%d'%target)
                    allowed=min(900,TOTAL-used(r)-final_reserve()-1.25*10*p)
                    args=['python',learning/'train.py','--manifest',r/'manifest.json','--output',r/'training','--steps',target,'--resume',checkpoint(prior),'--seconds',max(1,allowed-90)]
                    rc=launch('train_to_%06d'%target,'train',args,allowed,final_reserve()+1.25*10*p)
                    if rc or not checkpoint(target).exists():raise RuntimeError('Training failed or stopped before target')
                point=val(target)
            else:point=json.loads((r/('validation_point_%06d.json'%target)).read_text())
            points.append(point)
            stale,best_success,best_angle=patience_state(points)
            status('validation_complete',point=point,consecutive_no_improvement=stale)
            if stale>=3:
                reason='three_complete_points_without_task_improvement';break
        selection(reason)

    selected=json.loads((r/'selection.json').read_text())
    resource('final_test')
    methods=[('expert','expert',None),('untrained','policy',0),('best','policy',selected['best_step'])]
    if not selected['best_equals_last']:methods.append(('last','policy',selected['last_step']))
    planned=[dict(method=name,mode=mode,step=step,seed=seed) for seed in config['test_seeds'] for name,mode,step in methods]
    atomic_json(r/'test_plan.json',dict(attempts=planned,alias={'last':'best'} if selected['best_equals_last'] else {}))
    for seed in config['test_seeds']:
        for name,mode,step in methods:
            value=episode('test/%s_seed_%d'%(name,seed),mode,seed,'test',checkpoint(step) if step is not None else None,video=seed==5000,diagnostic=60)
            if value['status']!='completed':raise RuntimeError('Final test infrastructure failure')
        status('final_test_progress',completed_seed=seed)
    status('gpu_work_complete',stop_reason=selected['stop_reason'],best_step=selected['best_step'],last_step=selected['last_step'])
except BaseException as exc:
    reason=str(exc)
    atomic_json(r/'stop_error.json',dict(reason=reason,traceback=traceback.format_exc(),time=time.time()))
    status('stopped',reason=reason)
    # A failed contract never dispatches further policy tests. A valid complete
    # selection can be inspected and resumed; completed attempts stay immutable.
    traceback.print_exc()
    raise SystemExit(1)
