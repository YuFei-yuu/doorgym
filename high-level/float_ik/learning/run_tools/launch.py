"""One bounded GPU child with resource sampling; caller supplies all budget limits."""
import argparse
from pathlib import Path
import shlex
import statistics
import subprocess
import time
import fcntl
import os
from bounded import atomic_json, run


def launch(root, stage, args, timeout=None, total=None, reserve=0, environment='gym'):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    resource=root/'resources';resource.mkdir(exist_ok=True)
    record=resource/(stage+'.json')
    if record.exists():raise FileExistsError('Stage already attempted: '+str(record))
    repo=Path(__file__).resolve().parents[4]
    locks=repo/'.repro/locks';locks.mkdir(parents=True,exist_ok=True)
    gpu_lock=None
    while gpu_lock is None:
        samples=[]
        for i in range(3):
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
            samples.append([[x.strip() for x in line.split(',')] for line in raw.splitlines()])
            if i<2:time.sleep(2)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        occupied=set(apps.splitlines())
        eligible=[]
        for j,row in enumerate(samples[0]):
            if (min(float(s[j][2]) for s in samples)>=16384 and
                    max(float(s[j][4]) for s in samples)<1024 and
                    max(float(s[j][3]) for s in samples)<10 and row[1] not in occupied):
                eligible.append((statistics.mean(float(s[j][3]) for s in samples),int(row[0]),row[1]))
        for _,gpu,uuid in sorted(eligible):
            candidate=(locks/(uuid+'.lock')).open('a')
            try:
                fcntl.flock(candidate,fcntl.LOCK_EX | fcntl.LOCK_NB)
                gpu_lock=candidate
                break
            except BlockingIOError:
                candidate.close()
        if gpu_lock is None:
            atomic_json(resource/(stage+'.waiting.json'),dict(at=time.time(),samples=samples,eligible=eligible))
            print('WAITING_FOR_IDLE_GPU '+stage,flush=True)
            time.sleep(30)
    atomic_json(record,dict(samples=samples,eligible=eligible,physical_gpu=gpu,gpu_uuid=uuid))
    activation = (repo/'.repro/blackwell/activate.sh' if environment == 'gym'
                  else Path(__file__).with_name('activate_train.sh'))
    increment=max(0,10-os.getpriority(os.PRIO_PROCESS,0))
    shell='export VIDEO2DOOR_GPU=%d; source %s || exit; exec nice -n %d '%(gpu,shlex.quote(str(activation)),increment)+shlex.join(list(map(str,args)))
    try:
        return run(root,stage,['bash','-c',shell],timeout=timeout,total=total,reserve=reserve,gpu=gpu,gpu_uuid=uuid,log_name=stage+'.log')
    finally:
        gpu_lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',required=True);p.add_argument('--timeout',type=float);p.add_argument('--total-seconds',type=float);p.add_argument('--environment',choices=['gym','train'],default='gym');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if not cmd:p.error('Missing child command')
    raise SystemExit(launch(a.root,a.stage,cmd,a.timeout,a.total_seconds,environment=a.environment))
