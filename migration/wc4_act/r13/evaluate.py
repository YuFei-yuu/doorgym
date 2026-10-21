"""Exactly 8/8/64 claimed scenes, immutable checkpoints and complete receipts."""
import sys,subprocess,os,time,gc
from common import *
sys.path.insert(0,str(SUPPORT))
verify()
step=int(sys.argv[1]);assert step in [10000,20000,30000]
base=RUN/'evaluation'/f'{step:06d}';base.mkdir(parents=True,exist_ok=False)
# CPU-only export and comparison do not modify or consume trainer RNG.
with (base/'export.log').open('x') as log:
 cmd=[sys.executable,'-B',str(SUPPORT/'r13/export.py'),str(step)]
 write(base/'export-command.json',cmd)
 code=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
write(base/'export-exit.json',dict(returncode=code))
assert code==0,'Export consistency failed'
checkpoint=RUN/'checkpoints'/f'{step:06d}'/'door-checkpoint/model_latest.pt'
seeds=read(RUN/'contract.json')['final_seeds' if step==30000 else 'intermediate_seeds']
selected={};results=[]
for index,seed in enumerate(seeds):
 verify()
 if (RUN/'STOP_REQUEST.json').exists():raise SystemExit(75)
 assert not (RUN/'SYSTEMIC_STOP.json').exists()
 manifest=RUN/'frozen'/str(seed)/'manifest.json';trial=read(manifest)['trials'][0]
 output=base/f'trial_{index:02d}';record=True
 claims=RUN/'evaluation-claims';claims.mkdir(exist_ok=True)
 assert len(list(claims.glob('*.json')))<80
 claim=claims/f'{step:06d}-{index:02d}.json'
 with claim.open('x') as f:json.dump(dict(step=step,index=index,seed=seed,at=time.time(),output=str(output)),f)
 command=['bash','-c','source migration/wc4_act/activate.sh gym && exec python -B "$@"','r13',
   str(SUPPORT/'r13/rollout.py'),'--output',str(output),'--manifest',str(manifest),'--checkpoint',str(checkpoint)]
 if record:command.append('--record-video')
 command+=trial['argv']
 write(base/f'trial_{index:02d}-command.json',dict(argv=command,checkpoint_sha256=sha(checkpoint)))
 with (base/f'trial_{index:02d}.log').open('x') as log:
  code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
 write(base/f'trial_{index:02d}-exit.json',dict(returncode=code,at=time.time()))
 if code==75 and (RUN/'STOP_REQUEST.json').exists():raise SystemExit(75)
 assert code==0,'Systemic rollout process failure; no further scenes'
 result=read(output/'result.json')
 assert not result['error'] and 'nonfinite' not in result['reason'] and 'auxiliary' not in result['reason'],'Systemic runtime failure'
 assert (output/'scene-creation-receipt.json').exists()
 import json
 states=[json.loads(s) for s in (output/'states.jsonl').open()]
 locks=[json.loads(s) for s in (output/'lock-writes.jsonl').open()]
 from r11.criteria import score
 scored=score(states,locks,read(output/'reference.json'))
 write(output/'r11-score.json',scored)
 # Automatic gates are separate from independent original-video/code review.
 result.update(checkpoint_step=step,seed=seed,r11_measured_pass=scored['measured_task_gates_pass'],
     full_task_success='pending_original_review' if scored['measured_task_gates_pass'] else False,
     failures=scored['failures'])
 write(output/'evaluation-result.json',result)
 category='success_candidate' if scored['measured_task_gates_pass'] else 'failure'
 if record:
  import av
  video=output/'overview/env_0000/rollout.mp4'
  with av.open(str(video)) as container:
   stream=container.streams.video[0];assert float(stream.average_rate)==25
   count=sum(1 for _ in container.decode(video=0))
  receipt=read(video.parent/'recording.json')
  assert count==receipt['frames']==(result['steps']+1)//2 and count>0
  digest=sha(video)
  keep=category=='success_candidate' or category not in selected
  write(output/'video-validation.json',dict(sha256=digest,decoded_frames=count,fps=25,kept=keep,
      physical_duration_s=result['sim_seconds'],encoded_duration_s=count/25,original_execution=True))
  if keep and category not in selected:selected[category]=str(video)
  if not keep:video.unlink() # Only unselected RGB generated inside this run, SHA/receipt retained.
 results.append(result)
 write(base/'ledger.json',dict(checkpoint_step=step,budget=len(seeds),consumed=len(results),
     measured_pass=sum(r['r11_measured_pass'] for r in results),results=results,representative_videos=selected))
 del states,locks;gc.collect()
 import wandb
 receipt=read(RUN/'wandb-run.json')
 # Separate evaluation run, explicitly bound to the training run and checkpoint.
 # Never resume the trainer run from another process.
 if index==len(seeds)-1:
  with wandb.init(entity=receipt['entity'],project=receipt['project'],group=ID,
      name=f'{ID}-eval-{step:06d}',job_type='fulltask-evaluation',mode='online',dir=str(base),
      config=dict(training_run=receipt['id'],checkpoint_step=step,checkpoint_sha256=sha(checkpoint),seeds=seeds)) as wr:
   wr.log(dict(checkpoint_step=step,episodes=len(results),r11_measured_pass=sum(r['r11_measured_pass'] for r in results),
       physical_review_pending=sum(r['r11_measured_pass'] for r in results)))
   write(base/'wandb-run.json',dict(id=wr.id,url=wr.url))
write(base/'complete.json',dict(checkpoint_step=step,consumed=len(results),selected=selected,
     missing_categories=sorted({'success_candidate','failure'}-set(selected)),automatic_only=True))
