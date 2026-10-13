"""One bounded GPU child with resource sampling; caller supplies all budget limits."""
import argparse
from pathlib import Path
import shlex
import statistics
import subprocess
import time
from bounded import atomic_json, run


def launch(root, stage, args, timeout, total, reserve=0, environment='gym'):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    samples=[]
    for i in range(3):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        samples.append([[x.strip() for x in line.split(',')] for line in raw.splitlines()])
        if i<2:time.sleep(2)
    eligible=[]
    for j,row in enumerate(samples[0]):
        if min(float(s[j][2]) for s in samples)>=16384 and statistics.mean(float(s[j][3]) for s in samples)<70:
            eligible.append((statistics.mean(float(s[j][3]) for s in samples),int(row[0]),row[1]))
    resource=root/'resources';resource.mkdir(exist_ok=True)
    record=resource/(stage+'.json')
    if record.exists():raise FileExistsError('Stage already attempted: '+str(record))
    atomic_json(record,dict(samples=samples,eligible=eligible))
    if not eligible:raise RuntimeError('No eligible GPU')
    _,gpu,uuid=min(eligible)
    repo=Path(__file__).resolve().parents[4]
    activation = (repo/'.repro/blackwell/activate.sh' if environment == 'gym'
                  else Path(__file__).with_name('activate_train.sh'))
    shell='export VIDEO2DOOR_GPU=%d; source %s || exit; exec nice -n 10 '%(gpu,shlex.quote(str(activation)))+shlex.join(list(map(str,args)))
    return run(root,stage,['bash','-c',shell],timeout=timeout,total=total,reserve=reserve,gpu=gpu,gpu_uuid=uuid,log_name=stage+'.log')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',required=True);p.add_argument('--timeout',required=True,type=float);p.add_argument('--total-seconds',required=True,type=float);p.add_argument('--environment',choices=['gym','train'],default='gym');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if not cmd:p.error('Missing child command')
    raise SystemExit(launch(a.root,a.stage,cmd,a.timeout,a.total_seconds,environment=a.environment))
