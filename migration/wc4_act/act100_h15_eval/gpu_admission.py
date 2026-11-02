"""Read-only healthy-idle GPU admission under the existing UUID lease."""
import subprocess,fcntl
from pathlib import Path
LOCKS=Path('/data/zyf/video2door/.repro/locks')
def probe(gpu):
    result=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True)
    row=dict(gpu=int(gpu),available=False,query_returncode=result.returncode,snapshot=result.stdout.strip())
    if result.returncode or any(s in result.stdout for s in ['N/A','ERR']):return row
    try:uuid,memory,util=[s.strip() for s in result.stdout.strip().split(',')];memory=int(memory);util=int(util)
    except ValueError:return row
    row.update(uuid=uuid,memory_used_mib=memory,utilization=util)
    if memory>=1024 or util>=10:return row
    try:
        with (LOCKS/(uuid+'.lock')).open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            p=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True,capture_output=True)
            row['compute_contexts']=p.stdout.strip();row['available']=p.returncode==0 and not p.stdout.strip()
    except BlockingIOError:row['lease_busy']=True
    return row
