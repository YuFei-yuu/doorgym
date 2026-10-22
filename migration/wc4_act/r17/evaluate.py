"""One sequential fixed16 group. Stop on engineering errors; never repeat a claim."""
import sys,os,time,subprocess,gc,traceback
from common import *
sys.path.insert(0,str(SUPPORT));from r11.criteria import score
from r15.paired import record_diagnostics
from audit import audit_trial
from deep_audit import audit_one
from r13.stopping import request_stop
import numpy as np

def compare(output,seed):
 row=read(output/'paired-diagnostics.json');base=next(x for x in read(RUN/'baseline-h25.json')['records'] if x['seed']==seed)
 row.update(execution_horizon=5,baseline_horizon=25,baseline_gates=base['gates'],improved_items=[k for k,v in row['gates'].items() if v and not base['gates'][k]],regressed_items=[k for k,v in row['gates'].items() if not v and base['gates'][k]],baseline_full_success=base['full_task_success'])
 write(output/'paired-comparison.json',row);return row

def main():
 verify();results=[];paired=[];selected={};best=-1
 for i,seed in enumerate(SEEDS):
  verify();assert not (RUN/'STOP_REQUEST.json').exists()
  assert shutil.disk_usage(ROOT).free>40*1024**3
  claims=RUN/'evaluation-claims';assert len(list(claims.glob('*.json')))<16
  output=RUN/'evaluation'/f'trial_{i:02d}'
  with (claims/f'{i:02d}.json').open('x') as f:json.dump(dict(index=i,seed=seed,at=time.time(),execution_horizon=5,model_step=50000,output=str(output)),f)
  manifest=RUN/'frozen'/str(seed)/'manifest.json';trial=read(manifest)['trials'][0]
  command=['bash','-c','source migration/wc4_act/activate.sh gym && exec python -B "$@"','r17',str(SUPPORT/'r17/rollout.py'),'--output',str(output),'--manifest',str(manifest),'--checkpoint',str(CHECKPOINT),'--record-video',*trial['argv']]
  env=dict(os.environ,WC4_R17_BASELINE_TRIAL=str(BASE/'evaluation/050000'/f'trial_{i:02d}'))
  write(RUN/'evaluation'/f'trial_{i:02d}-command.json',dict(argv=command,at=time.time(),checkpoint=str(CHECKPOINT),horizon=5))
  started=time.time()
  with (RUN/'evaluation'/f'trial_{i:02d}.log').open('x') as log:code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  write(RUN/'evaluation'/f'trial_{i:02d}-exit.json',dict(returncode=code,wall_s=time.time()-started,at=time.time()))
  assert code==0,'Rollout stopped/error; no further episode'
  result=read(output/'result.json');assert not result['error']
  states=[json.loads(s) for s in (output/'states.jsonl').open()];locks=[json.loads(s) for s in (output/'lock-writes.jsonl').open()]
  scored=score(states,locks,read(output/'reference.json'));write(output/'r11-score.json',scored)
  result.update(seed=seed,checkpoint_step=50000,execution_horizon=5,r11_measured_pass=scored['measured_task_gates_pass'],full_task_success='pending_original_review' if scored['measured_task_gates_pass'] else False,failures=scored['failures'])
  write(output/'evaluation-result.json',result)
  import av
  video=output/'overview/env_0000/rollout.mp4'
  with av.open(str(video)) as container:
   assert float(container.streams.video[0].average_rate)==25;count=sum(1 for _ in container.decode(video=0))
  assert count==read(video.parent/'recording.json')['frames']==(result['steps']+1)//2
  write(output/'video-validation.json',dict(sha256=sha(video),decoded_frames=count,fps=25,kept=True,original_execution=True,physical_duration_s=result['sim_seconds'],encoded_duration_s=count/25))
  proof=audit_trial(output);assert proof['new_chunks']==(proof['emitted']+4)//5
  record_diagnostics(output,states,scored,seed,50000);pair=compare(output,seed)
  wall=[];parent=[];near=inside=0
  for line in (output/'policy-inputs-actions.jsonl').open():
   row=json.loads(line)
   if row['worker_evidence']['new_chunk']:
    wall.append(row['worker_evidence']['worker_sample_wall_s']);parent.append(row['parent_act_wall_s'])
  write(output/'inference-timing.json',dict(replans=len(wall),worker_sample_wall_s={k:float(v) for k,v in zip(['min','median','p95','max'],[np.min(wall),np.median(wall),np.percentile(wall,95),np.max(wall)])},parent_replan_act_wall_s={k:float(v) for k,v in zip(['min','median','p95','max'],[np.min(parent),np.median(parent),np.percentile(parent,95),np.max(parent)])},simulated_replan_interval_s=.2,action_hz=25,physics_hz=50,no_walltime_compensation=True))
  del states,locks;gc.collect()
  # Independent raw-depth/contact/receipt evidence before authorizing the next scene.
  audit_one(output)
  category='success_candidate' if scored['measured_task_gates_pass'] else 'failure'
  if category=='success_candidate':selected.setdefault(category,str(video))
  else:
   opened=scored['opening_max_abs_deg']>=80;passed=scored['whole_body_pass_step'] is not None
   priority=int(opened)+int(opened and passed)+int(opened and passed and (scored['first_aligned_end_push_step'] is not None or scored['close_hold'] is not None))
   if priority>best:selected['failure']=str(video);best=priority
  results.append(result);paired.append(pair)
  write(RUN/'ledger.json',dict(budget=16,claims=len(list(claims.glob('*.json'))),completed=len(results),automatic_candidates=sum(bool(r['r11_measured_pass']) for r in results),reviewed_successes=0 if not any(r['r11_measured_pass'] for r in results) else None,results=results,paired=paired,representative_videos=selected))
  print(json.dumps(dict(event='episode_complete',index=i,seed=seed,completed=len(results),automatic_candidate=result['r11_measured_pass'],gates=pair['gates'],runtime_evidence_passed=True)),flush=True)
 verify();assert len(list((RUN/'evaluation-claims').glob('*.json')))==16
 write(RUN/'R17-COMPLETE.json',dict(at=time.time(),evaluation_invocations=16,completed=16,model_step=50000,horizon=5,training_updates=0,automatic_candidates=sum(bool(r['r11_measured_pass']) for r in results),selected=selected,manual_review_pending=True))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if isinstance(exc,SystemExit) and exc.code in [None,0]:raise
  request_stop(RUN,'R17 implementation/resource/evidence stop',error=repr(exc),traceback=traceback.format_exc());write(RUN/'STOPPED.json',dict(at=time.time(),error=repr(exc),traceback=traceback.format_exc(),claims=len(list((RUN/'evaluation-claims').glob('*.json'))),no_automatic_restart=True));raise
