"""Two leased GPUs: remaining fixed mid seeds and authorized 10k->30k training."""
import os,sys,subprocess,time,signal,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];R13=ROOT/'migration/wc4_act/r13'
sys.path[:0]=[str(R13),str(ROOT/'migration/wc4_act')]
from stopping import request_stop,atomic_json
root=ROOT/'.repro/runs/wc4-fulltask-act-r14-s1000-20261022';train=root/'continuation-train';mid=root/'continuation-mid010000'
for fd,selector in [(9,os.environ['R14_TRAIN_GPU']),(10,os.environ['R14_MID_GPU'])]:
    uuid=subprocess.check_output(['nvidia-smi','-i',selector,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert 'FLOCK' in Path('/proc/self/fdinfo/'+str(fd)).read_text()
    assert Path(os.readlink('/proc/self/fd/'+str(fd))).name==uuid+'.lock'

def stop_all(reason,**details):
    for run in [train,mid]:request_stop(run,reason,**details)
for number in [signal.SIGINT,signal.SIGTERM]:signal.signal(number,lambda n,f:stop_all('continuation supervisor signal',signal=n))
processes={};monitors={};handles=[]
for name,run,gpu,script,args in [('train',train,os.environ['R14_TRAIN_GPU'],'resume_entry.py',None),('mid',mid,os.environ['R14_MID_GPU'],'remaining_mid_entry.py',[])]:
    env=dict(os.environ,WC4_EXECUTION_RUN=str(run),VIDEO2DOOR_GPU=gpu,CUDA_VISIBLE_DEVICES=gpu,WANDB_RUN_GROUP=root.name)
    if args is None:
        cp=root/'stopped-checkpoints/stop-010000'
        args=['--config_path='+str(cp/'pretrained_model/train_config.json'),'--resume=true','--output_dir='+str(run/'train'),'--steps=30000']
    cmd=[sys.executable,'-B',str(Path(__file__).with_name(script)),*args]
    log=(run/'execution.log').open('x');mlog=(run/'monitor.log').open('x');handles += [log,mlog]
    atomic_json(run/'train-command.json',dict(argv=cmd,continued_from_update=10000,role=name,new_user_authorized=True))
    processes[name]=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    monitors[name]=subprocess.Popen([sys.executable,'-B',str(R13/'monitor.py'),str(processes[name].pid)],cwd=ROOT,
        env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=mlog,stderr=subprocess.STDOUT)
    atomic_json(run/'process.json',dict(pid=processes[name].pid,monitor_pid=monitors[name].pid,gpu=gpu))
reported=set();released=set()
def release(name):
    if name in released:return
    fd=9 if name=='train' else 10
    run=train if name=='train' else mid
    lease=os.readlink('/proc/self/fd/'+str(fd))
    os.close(fd);released.add(name)
    atomic_json(run/'LEASE_RELEASED.json',dict(at=time.time(),lease=lease))
while any(p.poll() is None for p in processes.values()):
    for name,process in processes.items():
        run=train if name=='train' else mid
        if process.poll() is None and monitors[name].poll() is not None:stop_all('continuation monitor exited',role=name,code=monitors[name].returncode)
        if (run/'STOP_REQUEST.json').exists():
            reason=json.loads((run/'STOP_REQUEST.json').read_text());stop_all(reason['reason'],origin=name)
        if process.poll() is not None and name not in reported:
            code=process.returncode;atomic_json(run/'execution-exit.json',dict(returncode=code,at=time.time()));reported.add(name)
            if code!=0:stop_all('continuation process exited nonzero',role=name,code=code)
            if monitors[name].poll() is None:monitors[name].terminate()
            release(name)
    claims=2+sum(len(list((d/'evaluation-claims').glob('*.json'))) for d in [train,mid] if (d/'evaluation-claims').exists())
    if claims>80:stop_all('Global evaluation budget exceeded')
    atomic_json(root/'continuation-status.json',dict(at=time.time(),training_pid=processes['train'].pid,
        training_returncode=processes['train'].poll(),remaining_mid_pid=processes['mid'].pid,
        remaining_mid_returncode=processes['mid'].poll(),evaluation_invocations=claims,total_budget=80))
    time.sleep(2)
for name,process in processes.items():
    run=train if name=='train' else mid
    if name not in reported:atomic_json(run/'execution-exit.json',dict(returncode=process.returncode,at=time.time()))
    if monitors[name].poll() is None:monitors[name].terminate()
    monitors[name].wait(timeout=30)
    release(name)
for f in handles:f.close()
if any(p.returncode!=0 for p in processes.values()):raise SystemExit(75)
assert json.loads((train/'training-complete.json').read_text())['updates']==30000
assert len(list((train/'evaluation-claims').glob('*.json')))==72 and len(list((mid/'evaluation-claims').glob('*.json')))==6
atomic_json(root/'R14-COMPLETE.json',dict(completed_optimizer_updates=30000,evaluation_invocations=80,
 completed_evaluation_trials=79,controlled_interrupted_trials=1,no_replacement_or_extra_budget=True))
