"""Future authorized trainer entry; R13 never invokes it or creates its order."""
import sys,os,time,subprocess,math
from runtime import *
sys.path.insert(0,str(SUPPORT))
from stopping import BoundaryStop,atomic_json,request_stop
from training_guard import TrainingGuard,metric_values
RUN,ORDER=authorized_run()
import torch
from accelerate import Accelerator
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.train import TrainPipelineConfig
from overview import load
cfg_holder={}
original_validate=TrainPipelineConfig.validate

def validate(cfg):
    result=original_validate(cfg)
    assert cfg.steps==30000 and cfg.seed==1000 and cfg.batch_size==16 and cfg.save_freq==10000 and cfg.num_workers==3
    assert cfg.policy.optimizer_lr==1e-5 and cfg.policy.chunk_size==100 and cfg.policy.n_action_steps==50 and cfg.policy.n_obs_steps==1
    assert cfg.policy.vision_backbone=='resnet18' and cfg.policy.pretrained_backbone_weights=='ResNet18_Weights.IMAGENET1K_V1'
    assert str(cfg.dataset.root)==str(DATA) and cfg.dataset.episodes is None and not cfg.dataset.use_imagenet_stats
    assert cfg.env is None and cfg.keyframe_sampling_ratio==0
    for key in ['plucker_conditioning','interaction_state_conditioning','camera_input_gating','handle_latent_aux','point_cloud_conditioning','end_signal_prediction','use_action_loss_weight']:assert not getattr(cfg.policy,key)
    if cfg.resume:
        assert str(cfg.checkpoint_path)==ORDER.get('resume_checkpoint')
    else:assert cfg.policy.pretrained_path is None and not ORDER.get('resume_checkpoint')
    cfg_holder['cfg']=cfg;atomic_json(RUN/'effective-training-config.json',cfg.to_dict())
    return result
TrainPipelineConfig.validate=validate
forward=ACTPolicy.forward
def finite_forward(*a,**k):
    r=forward(*a,**k)
    if not torch.isfinite(r[0]).all():raise FloatingPointError('Nonfinite loss before backward')
    return r
ACTPolicy.forward=finite_forward
clip=Accelerator.clip_grad_norm_
def finite_clip(*a,**k):
    r=clip(*a,**k)
    if not torch.isfinite(r).all():raise FloatingPointError('Nonfinite gradient before optimizer step')
    return r
Accelerator.clip_grad_norm_=finite_clip
m=load('r13_author_train','high-level/lerobot/src/lerobot/scripts/lerobot_train.py')
init=m.WandBLogger.__init__
def logger_init(self,cfg):
    init(self,cfg)
    import wandb
    if wandb.run is None or wandb.run.offline:raise RuntimeError('Online W&B required')
    atomic_json(RUN/'wandb-run.json',dict(id=wandb.run.id,url=wandb.run.url,entity=wandb.run.entity,project=wandb.run.project))
m.WandBLogger.__init__=logger_init
# Metrics wrapper is inside the boundary: the completed receipt includes all
# author optimizer/scheduler work plus finite metrics, never merely a start.
update=m.update_policy;metrics=(RUN/'updates.jsonl').open('x',buffering=1);started=time.time()
def logged_update(*a,**k):
    if guard.boundary.completed>=30000:raise RuntimeError('Update budget exhausted')
    result=update(*a,**k);t=result[0]
    row=dict(update=guard.boundary.completed+1,at=time.time())
    row.update(metric_values(t))
    if not all(math.isfinite(row[k]) for k in ['loss','lr','grad_norm']):raise FloatingPointError('Nonfinite metric after update; unsafe boundary')
    metrics.write(__import__('json').dumps(row)+'\n');metrics.flush();os.fsync(metrics.fileno())
    if row['update']%100==0 or row['update']==1:atomic_json(RUN/'train-progress.json',row)
    return result
m.update_policy=logged_update
guard=TrainingGuard(m,RUN,lambda:cfg_holder['cfg']);guard.boundary.install_signals()
restore=m.load_training_state
def restored(*a,**k):
    result=restore(*a,**k)
    expected=validate_checkpoint(Path(ORDER['resume_checkpoint']))
    if result[0]!=expected:raise RuntimeError('Restored update differs from committed boundary')
    guard.boundary.completed=result[0]
    return result
m.load_training_state=restored
last=m.update_last_checkpoint
def ready(directory):
    result=last(directory);step=int(directory.name)
    if step not in [10000,20000,30000] or step!=guard.boundary.completed:raise RuntimeError('Checkpoint/update boundary mismatch')
    guard.boundary.checkpoint_if_requested()
    from lerobot.utils.random_utils import serialize_rng_state
    before=serialize_rng_state()
    # Same trainer process/model/optimizer; no automatic checkpoint selection.
    evaluated=subprocess.run([sys.executable,'-B',str(SUPPORT/'r13/evaluate.py'),str(step)],cwd=ROOT)
    guard.boundary.checkpoint_if_requested()
    evaluated.check_returncode()
    after=serialize_rng_state()
    if not all(torch.equal(before[k],after[k]) for k in before):raise RuntimeError('Evaluation changed trainer RNG')
    guard.boundary.checkpoint_if_requested()
    return result
m.update_last_checkpoint=ready
exit_code=0
try:
    m.main()
    guard.boundary.checkpoint_if_requested()
    if guard.boundary.completed!=30000:raise RuntimeError('Incomplete update budget')
    atomic_json(RUN/'training-complete.json',dict(updates=30000))
except BoundaryStop:
    exit_code=75
except BaseException as exc:
    exit_code=1;atomic_json(RUN/'SYSTEMIC_STOP.json',dict(reason=repr(exc),completed=guard.boundary.completed,
        unsafe_update=guard.boundary.inflight,recovery_claim=False));raise
finally:
    metrics.close();guard.boundary.close()
    import wandb
    if wandb.run is not None:wandb.finish(exit_code=exit_code)
if exit_code:raise SystemExit(exit_code)
