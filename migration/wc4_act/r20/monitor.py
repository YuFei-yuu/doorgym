"""Read-only W&B/resource monitor; persistent failure terminates only this trainer."""
import sys,os,time,signal,subprocess
from pathlib import Path
# Keep R20 common/runtime first; stopping remains the accepted R13 module.
sys.path.append(str(Path(__file__).resolve().parents[1]/'r13'))
from common import *
from stopping import request_stop
import wandb
pid=int(sys.argv[1]);failures=0;previous=-1;last_change=time.time()
while not (RUN/'wandb-run.json').exists():
 try:os.kill(pid,0)
 except ProcessLookupError:raise SystemExit(0)
 if time.time()-(RUN/'train-command.json').stat().st_mtime>360:raise RuntimeError('W&B initialization exceeded 360s')
 time.sleep(2)
r=read(RUN/'wandb-run.json');path=f"{r['entity']}/{r['project']}/{r['id']}"
uuid=subprocess.check_output(['nvidia-smi','-i',os.environ['VIDEO2DOOR_GPU'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
def stop(reason,event):
 request_stop(RUN,reason,event=event)
 raise SystemExit(75)
while True:
 try:os.kill(pid,0)
 except ProcessLookupError:break
 progress=read(RUN/'train-progress.json') if (RUN/'train-progress.json').exists() else {'update':0}
 event=dict(at=time.time(),local_update=progress['update'])
 try:
  remote=wandb.Api(timeout=30).run(path)
  remote_step=int(remote.summary.get('train/steps',remote.summary.get('train/step',remote.summary.get('_step',-1))))
  failures=0
  if remote_step>previous:previous=remote_step;last_change=time.time()
  event.update(online=True,remote_step=remote_step,state=remote.state)
 except Exception as exc:
  failures+=1;event.update(online=False,consecutive_failures=failures,error=repr(exc))
 resource=subprocess.check_output(['nvidia-smi','-i',os.environ['VIDEO2DOOR_GPU'],'--query-gpu=uuid,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader'],text=True).strip()
 event.update(resource=resource,free_disk_bytes=__import__('shutil').disk_usage(ROOT).free)
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
 event['compute_apps']=compute
 meminfo=Path('/proc/meminfo').read_text().splitlines();event['host_mem_available_kib']=int(next(l.split()[1] for l in meminfo if l.startswith('MemAvailable:')))
 table=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True)
 owned={pid};pairs=[tuple(map(int,l.split())) for l in table.splitlines()]
 for _ in range(12):owned|={p for p,parent in pairs if parent in owned}
 def owned_now(current):
  seen=set()
  while current>1 and current not in seen:
   if current==pid:return True
   seen.add(current);status=Path('/proc')/str(current)/'status'
   try:current=int(next(x.split(':',1)[1] for x in status.read_text().splitlines() if x.startswith('PPid:')))
   except FileNotFoundError:return True
  return False
 foreign=[l for l in compute.splitlines() if l.startswith(uuid) and not owned_now(int(l.split(',')[1]))]
 event['owned_pids_snapshot']=sorted(owned);event['foreign_same_gpu']=foreign
 with (RUN/'monitor.jsonl').open('a') as f:f.write(__import__('json').dumps(event)+'\n')
 if foreign:stop('Resource conflict on leased GPU',event)
 if event['free_disk_bytes']<40*1024**3:stop('Capacity below 40GiB headroom',event)
 if failures>=6:stop('Persistent W&B disconnection: six checks',event)
 if event.get('online') and progress['update']>remote_step+100 and time.time()-last_change>600:stop('W&B remote history stale >600s',event)
 time.sleep(60)
