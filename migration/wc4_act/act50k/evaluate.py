"""Paired fixed16 policy-only rollouts. Actual task failures stay in denominator."""
from common import *
import subprocess,time,concurrent.futures,gzip,shutil,fcntl,traceback
step=int(sys.argv[1]);assert step in STAGES
verify();base=RUN/'evaluation'/f'{step:06d}';base.mkdir(parents=True,exist_ok=True)
if (base/'complete.json').exists():raise SystemExit(0)
export=RUN/'checkpoints'/f'{step:06d}'
if not (export/'export-consistency.json').exists():
 with (base/('export-'+str(time.time_ns())+'.log')).open('x') as log:
  subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('export.py')),str(step)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
checkpoint=export/'door-checkpoint/model_latest.pt'
rows=read(RUN/'fixed16.json')['trials']

def trial(index,gpu):
 row=rows[index];seed=row['seed'];target=base/f'trial_{index:02d}';target.mkdir(exist_ok=True)
 if (target/'event-result.json').exists():return read(target/'event-result.json')
 prior=list(target.glob('attempt_*'))
 # Only an infrastructure attempt without an actual-state task result may retry.
 if prior:
  latest=sorted(prior)[-1]
  if (latest/'result.json').exists() and not read(latest/'result.json')['error']:
   from events import audit
   result=audit(latest,seed,step);write(target/'event-result.json',result);return result
 attempt=target/f'attempt_{len(prior):03d}'
 cmd=['bash','-c','export VIDEO2DOOR_GPU="$1"; shift; source migration/wc4_act/activate.sh gym && exec python -B "$@"','act50k',str(gpu),str(Path(__file__).with_name('rollout.py')),'--output',str(attempt),'--manifest',row['manifest'],'--checkpoint',str(checkpoint),*row['trial']['argv']]
 write(target/f'command-{len(prior):03d}.json',dict(argv=cmd,seed=seed,checkpoint_sha256=sha(checkpoint),official_model_sha256=sha(RUN/'train/checkpoints'/f'{step:06d}'/'pretrained_model/model.safetensors')))
 env=dict(os.environ,WC4_ACT50K_RUN=str(RUN),WC4_ACT50K_STAGE=str(step),CUDA_VISIBLE_DEVICES=str(gpu))
 with (target/f'process-{len(prior):03d}.log').open('x') as log:
  code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 write(target/f'exit-{len(prior):03d}.json',dict(code=code,at=time.time()))
 if not (attempt/'result.json').exists():raise RuntimeError(f'Infrastructure exception seed {seed}: see {target}')
 from events import audit
 result=audit(attempt,seed,step)
 write(target/'event-result.json',result)
 # Lossless compression after scoring with original/decoded SHA receipt; no evidence deletion.
 receipt=[]
 for name in ['states.jsonl','contacts.jsonl','lock-writes.jsonl','policy-inputs-actions.jsonl','worker-applied-evidence.jsonl']:
  path=attempt/name
  if not path.exists():continue
  digest=sha(path);compressed=path.with_suffix(path.suffix+'.gz')
  with path.open('rb') as src,gzip.open(compressed,'wb',compresslevel=3) as out:shutil.copyfileobj(src,out)
  import hashlib
  h=hashlib.sha256()
  with gzip.open(compressed,'rb') as f:
   for b in iter(lambda:f.read(8388608),b''):h.update(b)
  assert h.hexdigest()==digest
  receipt.append(dict(original=name,bytes=path.stat().st_size,sha256=digest,archive=compressed.name,archive_sha256=sha(compressed),decoded_sha_equal=True));path.unlink()
 write(attempt/'lossless-log-compression.json',receipt)
 if read(attempt/'result.json')['error']:raise RuntimeError(f'Infrastructure error retained seed {seed}')
 return result

def lane(gpu,indices):
 uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 with open('/data/zyf/video2door/.repro/locks/'+uuid+'.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  write(base/f'gpu-{gpu}-lock.json',dict(gpu=gpu,uuid=uuid,pid=os.getpid(),at=time.time()))
  return [trial(i,gpu) for i in indices]
# Independent simulations; every stage reuses each exact manifest once.
gpus=[int(x) for x in os.environ.get('WC4_ACT50K_EVAL_GPUS','0,2,5,6').split(',')]
with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
 futures=[pool.submit(lane,gpu,list(range(i,16,len(gpus)))) for i,gpu in enumerate(gpus)]
 results=[]
 for future in futures:results.extend(future.result())
results.sort(key=lambda x:x['seed']);assert len(results)==16
counts={k:sum(r['events'][k] is not None for r in results) for k in results[0]['events']}
summary=dict(step=step,denominator=16,successes=sum(r['success'] for r in results),event_counts=counts,results=results)
write(base/'summary.json',summary)
import wandb
receipt=read(RUN/'wandb-run.json')
with wandb.init(entity=receipt['entity'],project=receipt['project'],group=RUN.name,name=f'{RUN.name}-eval-{step:06d}',job_type='paired-evaluation',dir=str(base),config=dict(training_run=receipt['url'],step=step,fixed16_sha256=sha(RUN/'fixed16.json'),checkpoint_sha256=sha(checkpoint))) as wr:
 wr.log(dict(successes=summary['successes'],episodes=16,**{'events/'+k:v for k,v in counts.items()}))
 write(base/'wandb-run.json',dict(id=wr.id,url=wr.url))
write(base/'complete.json',dict(step=step,trials=16,successes=summary['successes']))
print(json.dumps(summary),flush=True)
