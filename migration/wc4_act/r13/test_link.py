"""Real author update/save/load and CLI handoffs, with toy policy only."""
import os,sys,json,signal,ast
from pathlib import Path
from types import SimpleNamespace
import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from runtime import *
sys.path.insert(0,str(SUPPORT))
from training_guard import TrainingGuard
from stopping import BoundaryStop,request_stop
from test_stopping import Toy,Config,Processor
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.train_utils import save_checkpoint,load_training_state
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from overview import load
import draccus
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
# No real ACT constructor may run during any handoff check.
ACTPolicy.__init__=lambda *a,**k:(_ for _ in ()).throw(AssertionError('ACT construction forbidden in R13'))
class Policy(Toy):
 def forward(self,batch):
  os.kill(os.getpid(),signal.SIGTERM)
  return ((self.w-batch['target'])**2).sum(),{}
module=SimpleNamespace(update_policy=update_policy,save_checkpoint=save_checkpoint,make_pre_post_processors=lambda *a,**k:(Processor('pre'),Processor('post')))
run=out/'real-author-update';run.mkdir()
guard=TrainingGuard(module,run,lambda:Config());guard.boundary.install_signals()
module.make_pre_post_processors(None)
model=Policy();opt=torch.optim.AdamW(model.parameters(),lr=1e-3);acc=Accelerator(cpu=True)
try:module.update_policy(SimpleNamespace(),model,{'target':torch.zeros(2)},opt,1.,accelerator=acc,lr_scheduler=None)
except BoundaryStop:pass
else:raise AssertionError('Author update ignored stop')
finally:guard.boundary.close()
stopped=read(run/'STOPPED.json');assert stopped['completed_updates']==1
cp=Path(stopped['checkpoint']['path'])
restore=Toy();restore.load_state_dict(load_file(str(cp/'pretrained_model/model.safetensors')))
restore_opt=torch.optim.AdamW(restore.parameters(),lr=1e-3)
step,restore_opt,sched=load_training_state(cp,restore_opt,None);assert step==1 and sched is None
assert all(float(v['step'])==1. for v in restore_opt.state.values())
# The real argv builders/parsers agree on the original recipe and checkpoint paths.
cmd=training_argv(out/'future-run-never-started');cfg=draccus.parse(TrainPipelineConfig,args=cmd[3:]);cfg.validate()
assert cfg.steps==30000 and cfg.seed==1000 and cfg.batch_size==16 and cfg.policy.chunk_size==100 and cfg.policy.n_action_steps==50
assert cfg.policy.pretrained_path is None and cfg.policy.pretrained_backbone_weights=='ResNet18_Weights.IMAGENET1K_V1'
assert str(cfg.dataset.root)==str(DATA) and cfg.dataset.episodes is None and not cfg.dataset.use_imagenet_stats
exporter=load('r13_export_parser','high-level/dp/export_official_lerobot_act_to_door_checkpoint.py')
sys.argv=['export','--official_checkpoint',str(cp/'pretrained_model'),'--root',str(DATA.parents[1]),'--repo_id',REPO,
 '--out_dir',str(out/'never-exported/model_latest'),'--depth_only','--device','cpu','--action_horizon','25']
parsed=exporter.parse_args();assert parsed.action_horizon==25 and parsed.depth_only
assert not (out/'never-exported').exists()
# Synthetic readiness is not a valid runtime authorization or an old R12 replacement.
os.environ['WC4_EXECUTION_RUN']=str(R12)
try:authorized_run()
except RuntimeError:pass
else:raise AssertionError('Historical run accepted')
write(out/'result.json',dict(status='passed',real_author_update_with_toy_policy=True,
 model_optimizer_processors_saved_same_boundary=True,author_loader_restored_step=step,
 training_cli_original_recipe=True,export_cli_parsed=True,actual_export=False,real_policy_inference=0,
 ACT_training_updates=0,simulation=0,old_run_refused=True))
print('PASS: author update->cooperative boundary->author serializer/loader; unchanged training/export CLI contracts; no ACT model')
