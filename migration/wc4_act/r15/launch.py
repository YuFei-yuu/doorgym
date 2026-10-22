"""One leased GPU; existing R13 cooperative supervision and stop serialization."""
import os,sys,subprocess,signal,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];SUPPORT=ROOT/'migration/wc4_act'
sys.path[:0]=[str(Path(__file__).parent),str(SUPPORT/'r13'),str(SUPPORT)]
from runtime import authorized_run,read
from stopping import atomic_json,request_stop
from supervision import supervise
RUN,ORDER=authorized_run()
assert 'FLOCK' in Path('/proc/self/fdinfo/9').read_text()
uuid=subprocess.check_output(['nvidia-smi','-i',os.environ['VIDEO2DOOR_GPU'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert Path(os.readlink('/proc/self/fd/9')).name==uuid+'.lock'
assert not (RUN/'train-command.json').exists()
cmd=read(RUN/'prepared-command.json')
atomic_json(RUN/'train-command.json',dict(argv=cmd,authorization_order=ORDER))
for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda n,f:request_stop(RUN,'R15 launcher signal',signal=n))
with (RUN/'train.log').open('x') as log,(RUN/'monitor.log').open('x') as mlog:
 p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 monitor=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('monitor.py')),str(p.pid)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=mlog,stderr=subprocess.STDOUT)
 atomic_json(RUN/'process.json',dict(trainer=p.pid,monitor=monitor.pid,supervisor=os.getpid(),gpu_uuid=uuid))
 code=supervise(p,monitor,RUN)
 assert p.poll() is not None
 if monitor.poll() is None:monitor.terminate()
 monitor.wait(timeout=30)
atomic_json(RUN/'train-exit.json',dict(returncode=code,at=time.time()))
os.close(9);atomic_json(RUN/'LEASE_RELEASED.json',dict(gpu_uuid=uuid,at=time.time()))
if code==0:
 assert read(RUN/'training-complete.json')['updates']==50000
 assert len(list((RUN/'evaluation-claims').glob('*.json')))==32
 atomic_json(RUN/'R15-COMPLETE.json',dict(cumulative_updates=50000,new_updates=20000,evaluation_invocations=32))
raise SystemExit(code)
