#!/usr/bin/env python3
"""Run once after final tests; diagnostic only, within remaining baseline budget."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

r=Path(sys.argv[1]).resolve()
repo=Path('/data/zyf/video2door')
samples=[]
for _ in range(3):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    samples.append([[float(v) for v in line.split(',')] for line in raw.strip().splitlines()])
    time.sleep(2)
(r/'resource_post_audit.json').write_text(json.dumps(samples))
choices=[]
for i in range(len(samples[0])):
    load=sum(s[i][2] for s in samples)/3
    if min(s[i][1] for s in samples)>=16384 and load<70:
        choices.append((load,int(samples[0][i][0])))
if not choices:
    raise SystemExit('No eligible GPU for audit; preserve completed results')
os.environ['VIDEO2DOOR_GPU']=str(min(choices)[1])
print('SELECT_GPU',os.environ['VIDEO2DOOR_GPU'],flush=True)
for name,timeout,args in [
    ('offline-action',70,['offline_eval.py','--root',str(r)]),
    ('recording-audit',65,['rollout.py','--schema','float_push_joint_v1','--output',str(r/'recording-audit'),'--mode','expert','--nominal','--record']),
]:
    command=[sys.executable,str(repo/'high-level/float_ik/learning'/args[0])]+args[1:]
    wrapped=[sys.executable,str(r/'tools/bounded.py'),'--root',str(r),'--stage','baseline','--limit','300','--timeout',str(timeout),'--','bash','-c','source /data/zyf/video2door/.repro/blackwell/activate.sh; exec nice -n 10 '+shlex.join(command)]
    with (r/'logs'/(name+'.log')).open('x') as log:
        rc=subprocess.call(wrapped,cwd=str(repo),stdout=log,stderr=subprocess.STDOUT)
    print(name,rc,flush=True)
    if rc:raise SystemExit(rc)
