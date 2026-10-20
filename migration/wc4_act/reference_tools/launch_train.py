import json,os,subprocess,sys,time
from common import *
source_check();save(RUN/'input-dataset-manifest.json',data_check())
assert sha(SOURCE/'pretrained_model/model.safetensors')=='88fa37325c84e070d8f122bf098b006896fe75ec38e9cba3883edf202be3bcf5'
assert not (RUN/'train').exists()
with GPULease(int(os.environ['VIDEO2DOOR_GPU']),'train'):
 import wandb
 with wandb.init(entity='fi-yu-individual',project='door-dp-stop-tail',mode='online',group=ID,name=ID,job_type='train-resume',dir=str(RUN),config={'parent_run':'69gd85mm','resume_checkpoint':str(SOURCE),'start_step':40000,'stop_step':50000,'save_freq':2000,'seed':1000}) as bootstrap:
  receipt={'id':bootstrap.id,'url':bootstrap.url,'entity':bootstrap.entity,'project':bootstrap.project,'group':bootstrap.group};save(RUN/'wandb-bootstrap.json',receipt)
 # Author resume="must" can now resume this new, independently created online run.
 cmd=[sys.executable,'-B',str(RUN/'tools/guarded_train.py'),'--config_path='+str(SOURCE/'pretrained_model/train_config.json'),'--resume=true','--steps=50000','--save_freq=2000','--output_dir='+str(RUN/'train'),'--job_name='+ID,'--wandb.run_id='+receipt['id'],'--wandb.mode=online','--wandb.enable=true']
 save(RUN/'train-command.json',{'argv':cmd,'source_head':HEAD,'checkpoint':str(SOURCE),'planned_new_updates':10000,'steps':[42000,44000,46000,48000,50000]})
 env=dict(os.environ,WANDB_RUN_GROUP=ID,WANDB_MODE='online')
 for k in ['WANDB_RUN_ID','WANDB_RESUME','RANK','LOCAL_RANK','WORLD_SIZE']:env.pop(k,None)
 with (RUN/'train.log').open('w') as log:
  process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);save(RUN/'train-pid.json',{'pid':process.pid})
  with (RUN/'wandb-watch.log').open('w') as mlog:
   monitor=subprocess.Popen([sys.executable,'-B',str(RUN/'tools/watch_wandb.py'),str(process.pid)],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=mlog,stderr=subprocess.STDOUT)
   while process.poll() is None:
    if monitor.poll() is not None and not (RUN/'training-complete.json').exists():
     save(RUN/'stop-monitor-process.json',{'at':time.time(),'returncode':monitor.returncode});process.terminate();break
    time.sleep(3)
   code=process.wait()
   if monitor.poll() is None:monitor.terminate()
   monitor.wait(timeout=30)
 save(RUN/'train-result.json',{'returncode':code,'complete':(RUN/'training-complete.json').exists(),'at':time.time()})
 if code:raise SystemExit(code)
