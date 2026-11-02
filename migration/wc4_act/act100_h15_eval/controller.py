from common import *
import time,fcntl,shutil
ctl=RUN.with_name(RUN.name+'-control');ctl.mkdir(exist_ok=True)
with (ctl/'controller.lock').open('a+') as lease:
 fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=dict(os.environ,WC4_EXECUTION_RUN=str(RUN),WC4_ACT100_200K_EVAL_GPUS='7,4,5,6')
 for attempt in range(4):
  verify()
  if (RUN/'STOP_REQUEST.json').exists():raise RuntimeError('Capacity/controlled stop retained; review before resume')
  with (ctl/f'evaluation-{attempt}.log').open('xb') as out:
   p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('evaluate.py')),'120000'],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
   while p.poll() is None:
    free=shutil.disk_usage(ROOT).free
    write(RUN/'monitor-state.json',dict(state='evaluating',pid=p.pid,at=time.time(),free_bytes=free,official_results=len(list((RUN/'evaluation/120000').glob('trial_*/event-result.json'))),optimizer_updates=0))
    if free<40*1024**3:write(RUN/'STOP_REQUEST.json',dict(reason='capacity_below40GiB',at=time.time(),free_bytes=free))
    time.sleep(20)
   p.returncode=p.wait()
  write(ctl/f'evaluation-{attempt}.exit.json',dict(code=p.returncode,at=time.time()))
  if p.returncode==0:break
  if attempt==3:raise RuntimeError('Repeated infrastructure error; all attempts preserved')
  time.sleep(10)
 for name in ['audit_stage','finalize']:
  with (ctl/f'{name}.log').open('xb') as out:subprocess.run([sys.executable,'-B',str(Path(__file__).with_name(name+'.py')),*( ['120000'] if name=='audit_stage' else [])],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=out,stderr=subprocess.STDOUT,check=True)
 write(ctl/'completed.json',dict(complete=True,official_trials=16,optimizer_updates=0,at=time.time()))
