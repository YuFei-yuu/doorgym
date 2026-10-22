"""Own-group cooperative resource supervisor with shared UUID lease."""
import sys,os,time,signal,subprocess,traceback
from common import *
sys.path.insert(0,str(SUPPORT));from r13.stopping import request_stop
verify();assert 'FLOCK' in Path('/proc/self/fdinfo/9').read_text()
uuid=subprocess.check_output(['nvidia-smi','-i',os.environ['VIDEO2DOOR_GPU'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert Path(os.readlink('/proc/self/fd/9')).name==uuid+'.lock'
for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda s,f:request_stop(RUN,'R17 supervisor signal',signal=s))
cmd=[sys.executable,'-B',str(Path(__file__).with_name('evaluate.py'))];write(RUN/'evaluation-command.json',dict(argv=cmd,cwd=str(ROOT)))
try:
 with (RUN/'evaluation.log').open('x') as log:
  p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  write(RUN/'process.json',dict(supervisor=os.getpid(),evaluation=p.pid,gpu_uuid=uuid))
  while p.poll() is None:
   if shutil.disk_usage(ROOT).free<40*1024**3:request_stop(RUN,'R17 capacity stop')
   # Read-only ownership inventory. Never touch other users/processes.
   children={os.getpid()};changed=True
   ps=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines();pairs=[list(map(int,s.split())) for s in ps if len(s.split())==2]
   while changed:
    found={pid for pid,ppid in pairs if ppid in children};changed=not found.issubset(children);children|=found
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
   foreign=[s for s in apps.splitlines() if uuid in s and int(s.split(',')[0]) not in children]
   if foreign:request_stop(RUN,'R17 shared GPU resource conflict',foreign=foreign)
   with (RUN/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(at=time.time(),free_bytes=shutil.disk_usage(ROOT).free,gpu_uuid=uuid,own_pids=sorted(children),foreign_same_gpu=foreign))+'\n')
   time.sleep(15)
  code=p.wait()
 write(RUN/'evaluation-exit.json',dict(returncode=code,at=time.time()))
 if code!=0 and not (RUN/'STOP_REQUEST.json').exists():request_stop(RUN,'R17 evaluation exited',returncode=code)
finally:
 os.close(9);write(RUN/'LEASE_RELEASED.json',dict(gpu_uuid=uuid,at=time.time()))
raise SystemExit(code)
