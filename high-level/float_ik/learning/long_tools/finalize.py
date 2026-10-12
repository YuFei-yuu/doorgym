"""Wait for the owned driver, then preserve CPU-only delivery on every exit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from bounded import atomic_json

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--local-root',required=True);p.add_argument('--refresh',action='store_true');a=p.parse_args()
r=Path(a.root).resolve();learning=Path(__file__).resolve().parents[1];repo=learning.parents[2]
with (r/'finalize.lock').open('a') as final_lock:
    fcntl.flock(final_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not a.refresh and (r/'delivery_status.json').exists() and json.loads((r/'delivery_status.json').read_text()).get('phase')=='complete':
        raise SystemExit('Already finalized')
    with (r/'driver.lock').open('a') as driver_lock:
        while True:
            try:
                fcntl.flock(driver_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(10)
        commands=[
            ['tar','--exclude=._*','--exclude=__pycache__','-czf',str(r/'provenance/final-learning-source.tar.gz'),'-C',str(repo),'high-level/float_ik/learning'],
            [sys.executable,str(learning/'analyze.py'),'--root',str(r),'--validation-count','10','--test-seed','5000','--validation-seed','4000'],
        ]
        try:
            (r/'provenance').mkdir(exist_ok=True)
            atomic_json(r/'delivery_status.json',dict(phase='cpu_analysis',started=time.time(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
            for i,command in enumerate(commands):
                with (r/'logs'/('delivery_%d.log'%i)).open('w') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError('CPU delivery command failed: '+str(command))
            # Set completion before hashing so the recorded status is stable.
            atomic_json(r/'delivery_status.json',dict(phase='complete',finished=time.time(),cpu_only=True))
            command=[sys.executable,str(learning/'long_tools/deliver.py'),'--root',str(r),'--local-root',a.local_root]
            with (r/'logs/delivery_report.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Chinese report generation failed')
        except BaseException as exc:
            atomic_json(r/'delivery_status.json',dict(phase='error',error=repr(exc),traceback=traceback.format_exc(),finished=time.time()))
            traceback.print_exc()
            raise
print('CPU_DELIVERY_COMPLETE',str(r),flush=True)
