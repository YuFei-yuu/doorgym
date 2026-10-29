"""Full-state continuation ACT via unmodified author training loop; stage callbacks are read-only."""
from common import *
import time,subprocess,math,traceback,shutil
sys.path[:0]=[str(SUPPORT),str(SUPPORT/'r13')]
from stopping import BoundaryStop,request_stop
from training_guard import TrainingGuard,metric_values
import torch
from accelerate import Accelerator
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.train import TrainPipelineConfig
from overview import load
verify();holder={}
validate=TrainPipelineConfig.validate

def checked(cfg):
 result=validate(cfg)
 assert (cfg.steps,cfg.seed,cfg.batch_size,cfg.save_freq,cfg.num_workers)==(300000,1000,16,20000,3)
 assert (cfg.policy.optimizer_lr,cfg.policy.chunk_size,cfg.policy.n_action_steps)==(1e-5,100,50)
 assert cfg.policy.pretrained_backbone_weights=='ResNet18_Weights.IMAGENET1K_V1'
 assert str(cfg.dataset.root)==str(DATA) and cfg.dataset.episodes is None and not cfg.dataset.use_imagenet_stats
 assert cfg.env is None and cfg.keyframe_sampling_ratio==0 and cfg.recovery_sampling_ratio==0 and not cfg.use_rabc
 assert not getattr(cfg.policy,'drop_n_last_frames',0)
 for key in ['plucker_conditioning','interaction_state_conditioning','camera_input_gating','handle_latent_aux','point_cloud_conditioning','end_signal_prediction','use_action_loss_weight']:assert not getattr(cfg.policy,key)
 assert cfg.resume and cfg.scheduler is None and Path(cfg.output_dir)==RUN/'train'
 baseline=read(PARENT/'pretrained_model/train_config.json');actual=cfg.to_dict()
 inherited=['dataset','optimizer','scheduler','seed','batch_size','num_workers','dataloader_seed','separate_auxiliary_grad_clip','optimizer_foreach','strict_determinism','tolerance_s','use_policy_training_preset','use_rabc','keyframe_sampling_ratio','recovery_sampling_ratio','rename_map','eval_freq','log_freq']
 for key in inherited:assert actual[key]==baseline[key],('Recipe changed',key,actual[key],baseline[key])
 policy_old=dict(baseline['policy']);policy_new=dict(actual['policy']);policy_old.pop('pretrained_path');policy_new.pop('pretrained_path')
 assert policy_old==policy_new, 'Inherited policy recipe changed'
 write(RUN/'inherited-recipe-check.json',dict(passed=True,fields_equal=inherited,policy_fields_equal_except_checkpoint_path=True,parent_config_sha256=sha(PARENT/'pretrained_model/train_config.json')))
 holder['cfg']=cfg;write(RUN/'effective-training-config.json',actual)
 return result
TrainPipelineConfig.validate=checked
forward=ACTPolicy.forward

def finite_forward(*a,**k):
 result=forward(*a,**k)
 if not torch.isfinite(result[0]).all():raise FloatingPointError('Nonfinite loss before backward')
 return result
ACTPolicy.forward=finite_forward
clip=Accelerator.clip_grad_norm_
def finite_clip(*a,**k):
 result=clip(*a,**k)
 if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite gradient before optimizer step')
 return result
Accelerator.clip_grad_norm_=finite_clip
m=load('act50k_author_train','high-level/lerobot/src/lerobot/scripts/lerobot_train.py')
logger_init=m.WandBLogger.__init__
def init_logger(self,cfg):
 import wandb
 original_init=wandb.init
 first=not (RUN/'wandb-run.json').exists()
 if first:
  cfg.wandb.run_id="new-continuation-run"
  def new_init(*a,**k):
   assert k['resume']=='must';k.update(resume=None,id=None);return original_init(*a,**k)
  wandb.init=new_init
 else:cfg.wandb.run_id=read(RUN/'wandb-run.json')['id']
 try:logger_init(self,cfg)
 finally:wandb.init=original_init
 assert wandb.run.id!=read(PARENT_FINAL/'evidence/wandb-run.json')['id']
 wandb.config.update({'parent_checkpoint':str(PARENT),'parent_step':200000,'target_cumulative_updates':300000,'new_update_budget':100000},allow_val_change=True)
 assert wandb.run is not None and not wandb.run.offline
 write(RUN/'wandb-run.json',dict(id=wandb.run.id,url=wandb.run.url,entity=wandb.run.entity,project=wandb.run.project))
m.WandBLogger.__init__=init_logger
logger_log=m.WandBLogger.log_dict
def log(self,d,step,*a,**k):
 result=logger_log(self,d,step,*a,**k);self._wandb.log({'train/steps':step},step=step,commit=True);return result
m.WandBLogger.log_dict=log
make=m.make_policy

def make_policy(*a,**k):
 policy=make(*a,**k);holder['policy']=policy
 assert holder['cfg'].resume, 'Continuation cannot initialize a fresh policy'
 return policy
m.make_policy=make_policy
metrics=(RUN/'updates.jsonl').open('a',buffering=1)
# The guard wraps only the author optimizer/scheduler call. Post-update audits
# run AFTER its completed boundary, so an evidence error can save full state.
guardroot=RUN/'trainer-attempts'/str(time.time_ns());guardroot.mkdir(parents=True)
guard=TrainingGuard(m,guardroot,lambda:holder['cfg']);guard.boundary.install_signals()
guarded_update=m.update_policy

def logged_update(*a,**k):
 assert guard.boundary.completed<300000
 if holder.get('restored_checkpoint') and not holder.get('restored_before_forward'):
  # Loader/model setup may consume RNG after the original load call.
  from lerobot.utils.random_utils import deserialize_rng_state,serialize_rng_state
  from safetensors.torch import load_file
  cp=holder['restored_checkpoint']
  saved_rng=load_file(str(cp/'training_state/rng_state.safetensors'))
  deserialize_rng_state(saved_rng)
  actual_rng=serialize_rng_state()
  assert all(torch.equal(v.cpu(),actual_rng[n].cpu()) for n,v in saved_rng.items())
  holder['restored_before_forward']=True
  write(guardroot/'restored-before-first-forward.json',dict(step=guard.boundary.completed,model_optimizer_RNG_verified=True,RNG_reapplied_after_loader_setup=True,dataloader_cursor_restored=False))
 result=guarded_update(*a,**k)
 assert not guard.boundary.inflight
 row=dict(update=guard.boundary.completed,new_updates=guard.boundary.completed-200000,at=time.time(),**metric_values(result[0]))
 assert all(math.isfinite(row[n]) for n in ['loss','lr','grad_norm'])
 assert row['lr']==1e-5 and k.get('lr_scheduler') is None
 if row['update']==200001:write(RUN/'first-continued-update.json',dict(**row,optimizer_param_group_lrs=[g['lr'] for g in a[3].param_groups],scheduler=None))
 row.update(batch_size_actual=int(a[2]['observation.state'].shape[0]),indices=a[2]['index'].detach().cpu().reshape(-1).tolist())
 metrics.write(json.dumps(row)+'\n');metrics.flush()
 if row['update']%100==0 or row['update']==200001:
  os.fsync(metrics.fileno());write(RUN/'train-progress.json',row);verify()
 return result
m.update_policy=logged_update
restore=m.load_training_state

def restored(*a,**k):
 result=restore(*a,**k);guard.boundary.completed=result[0]
 from safetensors.torch import load_file
 from lerobot.optim.optimizers import flatten_dict
 cp=Path(a[0]);saved=load_file(str(cp/'training_state/optimizer_state.safetensors'))
 live=flatten_dict({'state':result[1].state_dict()['state']})
 assert set(saved)==set(live) and all(torch.equal(v,live[n].cpu()) for n,v in saved.items())
 model_saved=load_file(str(cp/'pretrained_model/model.safetensors'))
 assert set(model_saved)==set(holder['policy'].state_dict())
 assert all(torch.equal(v,holder['policy'].state_dict()[n].detach().cpu()) for n,v in model_saved.items())
 assert result[2] is None
 groups=json.loads(json.dumps(result[1].state_dict()['param_groups']))
 assert groups==read(cp/'training_state/optimizer_param_groups.json')
 assert all(g['lr']==1e-5 for g in groups)
 assert 200000<=result[0]<300000
 holder['restored_checkpoint']=cp
 write(guardroot/'restored.json',dict(step=result[0],checkpoint=str(cp),optimizer_tensors_equal=True,model_tensors_equal=True,param_groups_equal=True,scheduler=None,effective_lrs=[g['lr'] for g in groups],original_author_resume=True,dataloader_cursor_restored=False))
 return result
m.load_training_state=restored
save=m.save_checkpoint
def scheduled_save(*a,**k):
 assert k['cfg'].save_freq==20000
 assert k['scheduler'] is None
 return save(*a,**k)
m.save_checkpoint=scheduled_save
last=m.update_last_checkpoint

def ready(directory):
 result=last(directory);step=int(directory.name);assert step in STAGES and step==guard.boundary.completed
 verify()
 from lerobot.utils.random_utils import serialize_rng_state
 before=serialize_rng_state()
 write(RUN/'phase.json',dict(phase='evaluating',step=step,pid=os.getpid()))
 command=[sys.executable,'-B',str(Path(__file__).with_name('evaluate.py')),str(step)]
 write(RUN/f'evaluate-command-{step:06d}.json',command)
 subprocess.run(command,cwd=ROOT,check=True)
 after=serialize_rng_state();assert all(torch.equal(before[n],after[n]) for n in before)
 subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('audit_stage.py')),str(step)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
 write(RUN/f'evaluation-RNG-preserved-{step:06d}.json',dict(step=step,same_model_optimizer_process=True,rng_equal=True))
 write(RUN/'phase.json',dict(phase='training',step=step,pid=os.getpid()))
 return result
m.update_last_checkpoint=ready
try:
 write(RUN/'phase.json',dict(phase='training',pid=os.getpid()))
 m.main()
 assert guard.boundary.completed==300000
 assert all((RUN/'evaluation'/f'{step:06d}'/'complete.json').exists() for step in STAGES)
 write(RUN/'complete.json',dict(updates=300000,parent_updates=200000,new_updates=100000,evaluation_trials=80,stages=STAGES))
except BoundaryStop:
 raise SystemExit(75)
except BaseException as exc:
 write(guardroot/'exception.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed=guard.boundary.completed,inflight=guard.boundary.inflight))
 if not guard.boundary.inflight and guard.context:
  request_stop(guardroot,'interruption',error=repr(exc))
  try:guard.boundary.checkpoint_if_requested()
  except BoundaryStop:pass
 raise
finally:
 metrics.close();guard.boundary.close()
 import wandb
 if wandb.run:wandb.finish()
