"""Future launch with a later task's order and existing shared lease; never automatic."""
import os,sys,subprocess,signal
from runtime import *
sys.path.insert(0,str(SUPPORT))
from supervision import supervise
from stopping import request_stop,atomic_json
RUN,ORDER=authorized_run()
if not Path('/proc/self/fd/9').exists() or 'FLOCK' not in Path('/proc/self/fdinfo/9').read_text():raise RuntimeError('Shared GPU UUID lease FD9 must be inherited from parent shell')
if (RUN/'train-command.json').exists():raise RuntimeError('No automatic replacement/retry')
cmd=training_argv(RUN)
if ORDER.get('resume_checkpoint'):
 cmd=[sys.executable,'-B',str(SUPPORT/'r13/train.py'),'--config_path='+str(Path(ORDER['resume_checkpoint'])/'pretrained_model/train_config.json'),
     '--resume=true','--output_dir='+str(RUN/'train'),'--steps=30000']
for number in [signal.SIGINT,signal.SIGTERM]:signal.signal(number,lambda n,f:request_stop(RUN,'launcher signal',signal=n))
atomic_json(RUN/'train-command.json',dict(argv=cmd,authorization_order=ORDER))
with (RUN/'train.log').open('x') as log,(RUN/'monitor.log').open('x') as mlog:
 process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 monitor=subprocess.Popen([sys.executable,'-B',str(SUPPORT/'r13/monitor.py'),str(process.pid)],cwd=ROOT,
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=mlog,stderr=subprocess.STDOUT)
 try:code=supervise(process,monitor,RUN)
 finally:
  if process.poll() is None:raise RuntimeError('Do not release shared lease while trainer is alive')
  if monitor.poll() is None:monitor.terminate()
  monitor.wait(timeout=30)
atomic_json(RUN/'train-exit.json',dict(returncode=code,checkpointed_stop=(RUN/'STOPPED.json').exists()))
raise SystemExit(code)
