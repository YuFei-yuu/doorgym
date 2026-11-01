"""Bounded launcher/recovery supervisor. No hidden rollback or extra updates."""
from common import *
import subprocess,time,signal,shutil,math,fcntl
sys.path.insert(0,str(SUPPORT/'r13'))
from stopping import request_stop
from gpu_admission import probe
CTL=RUN.with_name(RUN.name+'-control');CTL.mkdir(exist_ok=True)
lease=(CTL/'supervisor.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
train_gpu=sys.argv[1];eval_gpus=sys.argv[2];assert int(train_gpu) not in [int(g) for g in eval_gpus.split(',')]

def completed():
    values=[0]
    for p in (RUN/'trainer-attempts').glob('*/update-boundary.json'):
        values.append(int(read(p)['completed']))
    return max(values)

def checkpoint_at_known_boundary():
    known=completed();candidates=[]
    for p in list((RUN/'train/checkpoints').glob('[0-9]*'))+list((RUN/'trainer-attempts').glob('*/stopped-checkpoints/stop-*')):
        state=p/'training_state/training_step.json'
        if state.exists() and (p/'pretrained_model/model.safetensors').exists() and (p/'training_state/optimizer_state.safetensors').exists() and (p/'training_state/rng_state.safetensors').exists():
            candidates.append((int(read(state)['step']),p))
    if not candidates:return None
    step,path=max(candidates,key=lambda x:x[0])
    if step!=known:raise RuntimeError('No complete state at latest known optimizer boundary; no silent replay: '+str((known,step)))
    return path

def stage_recovery(step):
    if step not in STAGES:return
    proof=RUN/f'stage-audit-{step:06d}.json'
    if proof.exists() and read(proof)['passed']:return
    if not (RUN/'train/checkpoints'/f'{step:06d}'/'training_state/training_step.json').exists():return
    env=dict(os.environ,WC4_ACT100_200K_RUN=str(RUN),WC4_ACT100_200K_EVAL_GPUS=eval_gpus,VIDEO2DOOR_GPU=train_gpu)
    for name in ['evaluate','audit_stage']:
        command=[sys.executable,'-B',str(Path(__file__).with_name(name+'.py')),str(step)]
        actual_env=dict(env,CUDA_VISIBLE_DEVICES='') if name=='audit_stage' else env
        log=CTL/(name+'-recovery-'+str(step)+'-'+str(time.time_ns())+'.log')
        with log.open('x') as out:subprocess.run(command,cwd=ROOT,env=actual_env,stdout=out,stderr=subprocess.STDOUT,check=True)
    write(RUN/f'evaluation-recovered-{step:06d}.json',dict(step=step,complete=True,same_training_process=False,full_checkpoint_preserved=True,training_resumes_from_saved_state=True,no_additional_optimizer_updates=True))

attempt=0
while True:
    verify()
    known=completed();assert 0<=known<=200000
    cp=checkpoint_at_known_boundary() if known else None
    if known:
        stage_recovery(known)
    if known==200000:
        assert all(read(RUN/f'stage-audit-{s:06d}.json')['passed'] for s in STAGES)
        if not (RUN/'complete.json').exists():write(RUN/'complete.json',dict(updates=200000,evaluation_trials=160,stages=STAGES,recovered_at_final_boundary=True))
        break
    if attempt and cp is None:raise RuntimeError('No full state for automatic recovery; startup evidence retained')
    # Resource waiting is not an execution failure and never consumes retries.
    pool=list(dict.fromkeys([int(train_gpu),*[int(g) for g in eval_gpus.split(',')]]))
    selected=None
    while selected is None:
        snapshots=[]
        for gpu in pool:
            item=probe(gpu);snapshots.append(item)
            if item['available']:selected=gpu;break
        if selected is None:
            free=shutil.disk_usage(ROOT).free
            write(RUN/'monitor-state.json',dict(state='waiting_training_gpu',completed_updates=known,free_bytes=free,candidates=pool,gpu_admission=snapshots,at=time.time()))
            time.sleep(30)
    train_gpu=str(selected);eval_gpus=','.join(str(g) for g in pool if g!=selected)
    write(CTL/('training-gpu-selection-'+str(time.time_ns())+'.json'),dict(selected_training_gpu=selected,evaluation_candidates=eval_gpus,completed_updates=known,admission=snapshots,no_optimizer_updates_for_wait=True,at=time.time()))
    verify()
    cmd=['bash',str(Path(__file__).with_name('launch.sh')),train_gpu,eval_gpus]
    if cp:cmd.append(str(cp))
    token=str(time.time_ns());log=CTL/('training-process-'+token+'.log')
    write(CTL/('launch-'+token+'.json'),dict(argv=cmd,known_completed_updates=known,resume_checkpoint=str(cp) if cp else None,at=time.time()))
    with log.open('x') as out:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
        write(RUN/'monitor-state.json',dict(state='running',supervisor_pid=os.getpid(),trainer_pid=proc.pid,attempt=attempt,training_gpu=train_gpu,eval_candidates=eval_gpus,at=time.time()))
        while proc.poll() is None:
            free=shutil.disk_usage(ROOT).free
            snap=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,capture_output=True)
            with (RUN/'resource-monitor.jsonl').open('a') as f:f.write(json.dumps(dict(at=time.time(),free_bytes=free,trainer_pid=proc.pid,phase=read(RUN/'phase.json') if (RUN/'phase.json').exists() else None,compute_apps=snap.stdout,query_returncode=snap.returncode))+'\n')
            if free<40*1024**3:
                request_stop(RUN,'capacity_below40GiB',free_bytes=free)
                active=sorted((RUN/'trainer-attempts').glob('*/update-boundary.json'))
                if active:request_stop(active[-1].parent,'capacity_below40GiB',free_bytes=free)
            time.sleep(20)
    write(CTL/('training-exit-'+token+'.json'),dict(code=proc.returncode,at=time.time(),known_completed_updates=completed()))
    if proc.returncode==0:
        assert read(RUN/'complete.json')['updates']==200000
        break
    if proc.returncode==3 and completed()==known:
        write(RUN/'monitor-state.json',dict(state='waiting_training_gpu',resource_admission_race=True,completed_updates=known,at=time.time()))
        time.sleep(20);continue
    attempt+=1
    write(RUN/'monitor-state.json',dict(state='interrupted',returncode=proc.returncode,completed_updates=completed(),at=time.time()))
    if attempt>3:raise RuntimeError('Repeated infrastructure interruption; evidence retained for engineering review')
    # Only resume when the original latest completed state exists. Never load
    # an older state to recreate already-budgeted updates.
    cp=checkpoint_at_known_boundary()
    if cp is None:raise RuntimeError('No complete boundary checkpoint; no automatic fresh restart')
    capacity_stop=(RUN/'STOP_REQUEST.json').exists() and read(RUN/'STOP_REQUEST.json')['reason']=='capacity_below40GiB'
    required_free=(65 if capacity_stop else 40)*1024**3
    while shutil.disk_usage(ROOT).free<required_free:
        write(RUN/'monitor-state.json',dict(state='waiting_capacity',completed_updates=completed(),free_bytes=shutil.disk_usage(ROOT).free,at=time.time()));time.sleep(30)
    root_stop=RUN/'STOP_REQUEST.json'
    if root_stop.exists():
        assert read(root_stop)['reason']=='capacity_below40GiB'
        dest=RUN/'archived-stop-requests'/('stop-'+str(time.time_ns())+'.json');dest.parent.mkdir(exist_ok=True);root_stop.rename(dest)
    time.sleep(10)
# Runtime monitoring is stopped BEFORE immutable evidence is published.
write(RUN/'monitor-state.json',dict(state='stopped_completed',completed_updates=200000,evaluation_trials=160,supervisor_pid=os.getpid(),at=time.time(),automatic_expansion=False))
command=[sys.executable,'-B',str(Path(__file__).with_name('freeze.py'))]
with (CTL/'freeze.log').open('x') as out:
    code=subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=out,stderr=subprocess.STDOUT).returncode
write(CTL/'finalization.json',dict(code=code,at=time.time(),complete=code==0,final=str(ROOT/'.repro/reviews/wc4-act100-fresh200k-20261031-FINAL'),runtime_monitor_stopped=True))
raise SystemExit(code)
