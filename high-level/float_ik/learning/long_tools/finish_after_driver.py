"""Preserve delivery after the owned GPU driver releases its lock."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time
from bounded import atomic_json

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--local-root',required=True);a=p.parse_args()
r=Path(a.root).resolve();tool=Path(__file__).resolve().parent;repo=tool.parents[3]
with (r/'resume_finish.lock').open('a') as own:
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (r/'driver.lock').open('a') as driver:
        while True:
            try:fcntl.flock(driver,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(10)
    state=json.loads((r/'status.json').read_text())
    event=Path(json.loads((r/'resume_state.json').read_text())['archive']).name
    if state['phase']=='gpu_work_complete':
        video_log='logs/resume_video_delivery_%s.log'%event
        with (r/video_log).open('x') as log:
            result=subprocess.run(['python3',str(tool/'video_delivery.py'),'--root',str(r)],stdout=log,stderr=subprocess.STDOUT)
        atomic_json(r/'supplemental_video_status.json',dict(exit_code=result.returncode,finished=time.time(),log=video_log))
    elif not (r/'selection.json').exists() and 'No eligible GPU' in state.get('reason',''):
        subprocess.run(['python3',str(tool/'seal_selection.py'),'--root',str(r)],check=True)
    command=['python',str(tool/'finalize.py'),'--root',str(r),'--local-root',a.local_root,'--refresh']
    shell='source %s && exec nice -n 10 %s'%(shlex.quote(str(repo/'.repro/blackwell/activate.sh')),shlex.join(command))
    with (r/('logs/resume_final_delivery_%s.log'%event)).open('x') as log:
        result=subprocess.run(['bash','-c',shell],cwd=str(repo),stdout=log,stderr=subprocess.STDOUT)
    atomic_json(r/'resume_finish_status.json',dict(exit_code=result.returncode,finished=time.time(),phase='complete' if result.returncode==0 else 'error'))
    if result.returncode:raise SystemExit(result.returncode)
