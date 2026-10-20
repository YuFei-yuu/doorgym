"""Final 50k checkpoint only: exactly four batches of 16, no reruns."""
import json,os,signal,subprocess,sys,time,traceback
from common import *
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from commands import build
STEPS=[50000];assert len(sys.argv)==1
def healthy():
 stops=list(RUN.glob('stop*.json'))
 assert not stops,stops
 if (RUN/'train-result.json').exists():assert json.loads((RUN/'train-result.json').read_text())['complete'],'training exited early'
def execute(cmd,log,env=None):
 with log.open('w') as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while proc.poll() is None:healthy();time.sleep(3)
  except BaseException:
   if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
   raise
  code=proc.wait()
  if code:raise RuntimeError(f'Command failed ({code}): {cmd}; log={log}')
forbid=RUN/f'evaluation-worker-started-{STEPS[0]:06d}.json';assert not forbid.exists();save(forbid,{'pid':os.getpid(),'at':time.time(),'checkpoint_steps':STEPS,'trials_per_step':64,'total_policy_trials':64,'first_base_seed':615455575,'video_policy':'Keep earliest success and earliest failure; temporary synchronous candidates only until both represented; no reruns'})
try:
 for step in STEPS:
  healthy();source_check()
  while not (RUN/f'checkpoint-ready-{step:06d}.json').exists():healthy();time.sleep(5)
  CP=RUN/'checkpoints'/f'{step:06d}';CP.mkdir(parents=True,exist_ok=True)
  cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=str(ROOT/'.repro/envs/door-act/lib'))
  execute([str(ACT),'-B',str(RUN/'tools/export_and_check.py'),str(step)],CP/'export-check.log',cpuenv)
  assert json.loads((CP/'export-consistency.json').read_text())['status']=='passed'
  with GPULease(int(os.environ['VIDEO2DOOR_GPU']),f'eval-{step:06d}'):
   os.environ['DOOR_DP_LEROBOT_PYTHON']=str(ACT)+' '+str(RUN/'tools/guarded_policy_worker.py')
   execute([sys.executable,'-B',str(RUN/'tools/check_gpu_bridge.py'),str(step)],CP/'gpu-bridge-check.log')
   ledger=[];ledgerfile=CP/'eval-batches.json';assert not ledgerfile.exists()
   for i in range(4):
    healthy();selected=json.loads((CP/'video-selection.json').read_text()) if (CP/'video-selection.json').exists() else {}
    record=not {'success','failure'}.issubset(selected)
    directory=CP/'evaluation'/f'batch_{i:03d}';assert not directory.exists();directory.mkdir(parents=True)
    cmd=build('evaluate',ID,gpu=int(os.environ['VIDEO2DOOR_GPU']),seed=615455575+i,trials=16)['argv']
    cmd=[sys.executable,*cmd[1:]]
    if not record:cmd[1]='high-level/dp/eval/eval_door_policy_success.py'
    cmd[cmd.index('--checkpoint')+1]=str(CP/'door-checkpoint/model_latest.pt');cmd[cmd.index('--run_root')+1]=str(directory)
    sep=cmd.index('--');cmd[sep:sep]=['--save_failure_rollouts','--no_progress']
    cmd+=['--door_twin_log_dir',str(directory/'door-twin'),'--save_failed_rollouts']
    entry={'step':step,'batch':i,'seed':615455575+i,'trials':16,'record_candidates':record,'argv':cmd,'output':str(directory),'status':'running','started_at':time.time()}
    ledger.append(entry);save(ledgerfile,ledger)
    execute(cmd,directory/'driver.log')
    check=[str(ACT),'-B',str(RUN/'tools/check_batch.py'),str(step),str(i)]+(['--recorded'] if record else [])
    execute(check,directory/'check.log',cpuenv)
    result=json.loads((directory/'independent-check.json').read_text());entry.update(status='passed',successes=result['successes'],finished_at=time.time());save(ledgerfile,ledger)
   trials=[]
   for i in range(4):trials+=json.loads((CP/'evaluation'/f'batch_{i:03d}'/'independent-check.json').read_text())['results']
   assert len(trials)==64 and [v['trial_index'] for v in trials]==list(range(64))
   selected=json.loads((CP/'video-selection.json').read_text());observed={'success' if v['success'] else 'failure' for v in trials};assert set(selected)==observed
   result={'status':'passed','checkpoint_step':step,'trials':64,'successes':sum(v['success'] for v in trials),'success_rate':sum(v['success'] for v in trials)/64,'seeds':[615455575+i for i in range(4)],'criterion':'max absolute hinge angle >=80 degrees','selected_videos':selected,'missing_video_classes':sorted({'success','failure'}-observed),'results':trials}
   save(CP/'evaluation-results.json',result)
   print(json.dumps({'step':step,'successes':result['successes'],'trials':64,'success_rate':result['success_rate'],'video_classes':sorted(selected)}),flush=True)
 # Training should have completed before final export; wait only for its final online flush/exit.
 while not (RUN/'train-result.json').exists():healthy();time.sleep(3)
 healthy();save(RUN/'evaluation-complete.json',{'steps':STEPS,'total_policy_trials':64,'at':time.time()})
except BaseException as e:
 save(RUN/'stop-evaluation.json',{'at':time.time(),'error':repr(e),'traceback':traceback.format_exc(),'no_reruns':True})
 if (RUN/'train-pid.json').exists() and not (RUN/'training-complete.json').exists():
  pid=json.loads((RUN/'train-pid.json').read_text())['pid']
  try:
   assert b'guarded_train.py' in Path(f'/proc/{pid}/cmdline').read_bytes();os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
 raise
