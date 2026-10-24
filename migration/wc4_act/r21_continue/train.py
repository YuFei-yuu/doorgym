"""Future authorized trainer entry; R13 never invokes it or creates its order."""
import sys,os,time,subprocess,math
from runtime import *
sys.path[:0]=[str(Path(__file__).parent),str(SUPPORT/'r13'),str(SUPPORT)]
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
    assert cfg.steps==200000 and cfg.seed==1000 and cfg.batch_size==16 and cfg.save_freq==50000 and cfg.num_workers==3
    assert cfg.policy.optimizer_lr==1e-5 and cfg.policy.chunk_size==100 and cfg.policy.n_action_steps==50 and cfg.policy.n_obs_steps==1
    assert cfg.policy.vision_backbone=='resnet18' and cfg.policy.pretrained_backbone_weights=='ResNet18_Weights.IMAGENET1K_V1'
    assert str(cfg.dataset.root)==str(DATA) and cfg.dataset.episodes is None and not cfg.dataset.use_imagenet_stats
    assert cfg.env is None and cfg.keyframe_sampling_ratio==0
    assert getattr(cfg,'recovery_sampling_ratio',0)==0 and not getattr(cfg,'use_rabc',False)
    assert getattr(cfg.policy,'drop_n_last_frames',0) in [0,None]
    assert cfg.resume and cfg.wandb.run_id==ORDER['wandb_run_id']
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
    wandb.config.update({'R21_sampling':read(RUN/'SAMPLING-FROZEN.json') | {'per_frame_counts':'SAMPLING-FROZEN.json'},'new_optimizer_updates':49845,'prior_training_run':'9c85ca7c','prior_stop_step':150155,'source_checkpoint':ORDER['resume_checkpoint'],'execution_horizon':25},allow_val_change=True)
    atomic_json(RUN/'wandb-run.json',dict(id=wandb.run.id,url=wandb.run.url,entity=wandb.run.entity,project=wandb.run.project))
m.WandBLogger.__init__=logger_init
# Metrics wrapper is inside the boundary: the completed receipt includes all
# author optimizer/scheduler work plus finite metrics, never merely a start.
update=m.update_policy;metrics=(RUN/'updates.jsonl').open('x',buffering=1);started=time.time()
def logged_update(*a,**k):
    if guard.boundary.completed>=200000:raise RuntimeError('Update budget exhausted')
    from sampling import audit_batch
    sampling_proof=audit_batch(a[2],guard.boundary.completed-75000)
    result=update(*a,**k);t=result[0]
    row=dict(update=guard.boundary.completed+1,at=time.time())
    row.update(metric_values(t));row['sampling']=sampling_proof
    row['batch_size_actual']=int(a[2]['observation.state'].shape[0])
    row['dataloading_s']=float(getattr(getattr(t,'dataloading_s',0),'val',getattr(t,'dataloading_s',0)))
    denominator=row['update_s']+row['dataloading_s']
    row['samples_per_s_actual']=row['batch_size_actual']/denominator if denominator>0 else 0.
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
    if step not in [200000] or step!=guard.boundary.completed:raise RuntimeError('Checkpoint/update boundary mismatch')
    guard.boundary.checkpoint_if_requested()
    from lerobot.utils.random_utils import serialize_rng_state
    before=serialize_rng_state()
    # Same trainer process/model/optimizer; no automatic checkpoint selection.
    child=subprocess.Popen([sys.executable,'-B',str(SUPPORT/'r21_continue/evaluate_entry.py'),str(step)],cwd=ROOT)
    atomic_json(RUN/'monitor-phase.json',dict(phase='evaluating',step=step,trainer_pid=os.getpid(),child_pid=child.pid,at=time.time()))
    evaluated_code=child.wait()
    atomic_json(RUN/'monitor-phase.json',dict(phase='training',step=step,trainer_pid=os.getpid(),at=time.time()))
    guard.boundary.checkpoint_if_requested()
    if evaluated_code:raise subprocess.CalledProcessError(evaluated_code,child.args)
    after=serialize_rng_state()
    if not all(torch.equal(before[k],after[k]) for k in before):raise RuntimeError('Evaluation changed trainer RNG')
    atomic_json(RUN/f'evaluation-RNG-preserved-{step:06d}.json',dict(step=step,all_RNG_keys_equal=True,keys=list(before),same_trainer_process=True,checkpoint_reloaded_after_evaluation=False))
    guard.boundary.checkpoint_if_requested()
    return result
m.update_last_checkpoint=ready
exit_code=0
try:
    atomic_json(RUN/'monitor-phase.json',dict(phase='training',step=150155,trainer_pid=os.getpid(),at=time.time()))
    m.main()
    guard.boundary.checkpoint_if_requested()
    if guard.boundary.completed!=200000:raise RuntimeError('Incomplete update budget')
    atomic_json(RUN/'training-complete.json',dict(updates=200000))
except BoundaryStop:
    exit_code=75
except BaseException as exc:
    exit_code=1
    atomic_json(RUN/'SYSTEMIC_STOP.json',dict(reason=repr(exc),completed=guard.boundary.completed,unsafe_update=guard.boundary.inflight,recovery_claim=False))
    request_stop(RUN,'R21 trainer contract/numeric/evidence exception',error=repr(exc))
    if not guard.boundary.inflight and guard.context is not None:
        try:guard.boundary.checkpoint_if_requested()
        except BoundaryStop:exit_code=75
        except BaseException as saving_error:
            atomic_json(RUN/'STOP_SAVE_ERROR.json',dict(error=repr(saving_error),completed=guard.boundary.completed,consistent_snapshot_published=False))
    if exit_code!=75:raise
finally:
    metrics.close();guard.boundary.close()
    import wandb
    if wandb.run is not None:wandb.finish(exit_code=exit_code)
if exit_code:raise SystemExit(exit_code)
