"""One UUID lease; R13 consistent-boundary supervision; never auto-resumes."""
import os,sys,subprocess,signal,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];SUPPORT=ROOT/'migration/wc4_act'
sys.path[:0]=[str(Path(__file__).parent),str(SUPPORT/'r13'),str(SUPPORT)]
from runtime import authorized_run,read
from stopping import atomic_json,request_stop
from supervision import supervise
RUN,ORDER=authorized_run()
import shutil
free=shutil.disk_usage(ROOT).free
atomic_json(RUN/'launch-capacity.json',dict(free_bytes=free,required_start_bytes=80*1024**3,ongoing_headroom_bytes=40*1024**3,at=time.time()))
assert free>=80*1024**3,'R20 pretraining capacity below80GiB; no trainer launched; preserve history'
assert 'FLOCK' in Path('/proc/self/fdinfo/9').read_text()
uuid=subprocess.check_output(['nvidia-smi','-i',os.environ['VIDEO2DOOR_GPU'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert Path(os.readlink('/proc/self/fd/9')).name==uuid+'.lock'
assert not (RUN/'train-command.json').exists()
cmd=read(RUN/'prepared-command.json');atomic_json(RUN/'train-command.json',dict(argv=cmd,authorization_order=ORDER))
for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda n,f:request_stop(RUN,'R20 launcher signal',signal=n))
p=monitor=None;code=1
try:
 with (RUN/'train.log').open('x') as log,(RUN/'monitor.log').open('x') as mlog:
  p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  monitor=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('monitor.py')),str(p.pid)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=mlog,stderr=subprocess.STDOUT)
  atomic_json(RUN/'process.json',dict(trainer=p.pid,monitor=monitor.pid,supervisor=os.getpid(),trainer_session=p.pid,gpu_uuid=uuid))
  code=supervise(p,monitor,RUN)
except BaseException as exc:
 request_stop(RUN,'R20 supervisor failure',error=repr(exc),traceback=traceback.format_exc())
 if p is not None and p.poll() is None:p.wait()
 atomic_json(RUN/'supervisor-error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
 raise
finally:
 if p is not None and p.poll() is None:p.wait()
 if p is not None and p.returncode!=0:request_stop(RUN,'R20 trainer exited',returncode=p.returncode)
 # A crashed trainer may leave its evaluation children alive. Keep the UUID
 # lease while our own isolated session cooperatively exits; never kill them.
 waited=time.monotonic()
 while p is not None:
  table=subprocess.check_output(['ps','-eo','pid=,sid=,stat='],text=True)
  remaining=[int(parts[0]) for line in table.splitlines() if len(parts:=line.split())==3 and int(parts[1])==p.pid and not parts[2].startswith('Z')]
  if not remaining:break
  request_stop(RUN,'R20 waiting for owned child session to stop',owned_remaining=remaining)
  if time.monotonic()-waited>120:atomic_json(RUN/'CHILD_STOP_TIMEOUT.json',dict(remaining=remaining,lease_must_remain_held=True,forced_kill=False))
  time.sleep(1)
 if monitor is not None:
  if monitor.poll() is None:monitor.terminate() # only our CPU monitor after trainer has exited
  monitor.wait(timeout=30)
 atomic_json(RUN/'train-exit.json',dict(returncode=p.returncode if p is not None else code,at=time.time()))
 os.close(9);atomic_json(RUN/'LEASE_RELEASED.json',dict(gpu_uuid=uuid,at=time.time()))
if code==0:
 assert read(RUN/'training-complete.json')['updates']==75000
 assert len(list((RUN/'evaluation-claims').glob('*.json')))==16
 atomic_json(RUN/'R20-COMPLETE.json',dict(cumulative_updates=75000,new_updates=25000,evaluation_invocations=16))
raise SystemExit(code)
