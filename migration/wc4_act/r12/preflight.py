"""CPU configuration and actual joint-limit contracts; never constructs a scene."""
import sys,json,ast,subprocess
from pathlib import Path
from common import *
sys.path.insert(0,str(SUPPORT))
import draccus,lerobot
from commands import build
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
cmd=build('train',ID,gpu=0,seed=1000,train_steps=30000)['argv']
cmd=[('--dataset.repo_id='+REPO if x.startswith('--dataset.repo_id=') else '--dataset.root='+str(DATA) if x.startswith('--dataset.root=') else x) for x in cmd]
cfg=draccus.parse(TrainPipelineConfig,args=cmd[2:]);cfg.validate()
assert not cfg.resume and cfg.policy.pretrained_path is None
assert cfg.policy.pretrained_backbone_weights=='ResNet18_Weights.IMAGENET1K_V1'
assert cfg.policy.chunk_size==100 and cfg.policy.n_action_steps==50 and cfg.policy.n_obs_steps==1
assert cfg.steps==30000 and cfg.batch_size==16 and cfg.seed==1000
assert cfg.policy.optimizer_lr==1e-5 and cfg.dataset.episodes is None
assert cfg.env is None and cfg.keyframe_sampling_ratio==0
assert ROOT/'high-level/lerobot/src' in Path(lerobot.__file__).parents
for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
import importlib.metadata as meta
receipt=dict(no_simulation=True,no_model_or_optimizer_constructed=True,python=sys.executable,
 lerobot=lerobot.__file__,config=cfg.to_dict(),
 versions={k:meta.version(k) for k in ['torch','torchvision','numpy','lerobot','av','wandb','torchcodec']})
if RUN.exists():
 import xml.etree.ElementTree as ET
 stats=read(DATA/'meta/stats.json')
 joint=next(j for j in ET.parse(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf').findall('joint') if j.attrib['name']=='jointGripper')
 limits=joint.find('limit').attrib
 write(RUN/'gripper-contract.json',dict(data_min=stats['action']['min'][-1],data_max=stats['action']['max'][-1],
     urdf_limits=limits,ik_range_scale=.75,ik_soft_lower=.75*float(limits['lower']),
     policy_target_limits='actual physical actor limits; no soft clipping or gripper latch'))
 write(RUN/'preflight-act.json',receipt)
else:print(json.dumps(receipt,indent=2))
print('PASS: fresh ACT recipe and complete 72-episode input; no model, optimizer or scene created')
