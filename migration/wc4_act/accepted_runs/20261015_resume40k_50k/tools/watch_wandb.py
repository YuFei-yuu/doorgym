import json
import os
from pathlib import Path
import signal
import sys
import time
import wandb

from common import RUN
pid=int(sys.argv[1]);failures=0;previous_remote=-1;last_remote_change=time.time()
for _ in range(180):
    if (RUN/'wandb-run.json').exists():break
    try:os.kill(pid,0)
    except ProcessLookupError:raise SystemExit(0)
    time.sleep(2)
receipt=json.loads((RUN/'wandb-run.json').read_text());path=f"{receipt['entity']}/{receipt['project']}/{receipt['id']}"
while True:
    try:os.kill(pid,0)
    except ProcessLookupError:break
    progress=json.loads((RUN/'train-progress.json').read_text()) if (RUN/'train-progress.json').exists() else {'update':40000}
    try:
        run=wandb.Api(timeout=30).run(path)
        remote_step=int(run.summary.get('train/steps',run.summary.get('train/step',run.summary.get('_step',-1))))
        failures=0
        if remote_step>previous_remote:previous_remote=remote_step;last_remote_change=time.time()
        event={'at':time.time(),'online':True,'local_update':progress['update'],'remote_step':remote_step,'state':run.state}
    except Exception as exc:
        failures+=1;event={'at':time.time(),'online':False,'consecutive_failures':failures,'error':str(exc)}
    with (RUN/'wandb-monitor.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    if failures>=6 or (event.get('online') and progress['update']>event['remote_step']+100 and time.time()-last_remote_change>600):
        (RUN/'stop-monitor.json').write_text(json.dumps({'reason':'Persistent W&B disconnection or stale remote history','event':event},indent=2)+'\n')
        os.kill(pid,signal.SIGTERM);break
    time.sleep(60)
