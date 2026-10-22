"""R16 fixed CPU-only offline diagnostic contract. No experiment runner imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'.repro/runs/wc4-act-terminal-diagnostic-r16-20261023'
R11=ROOT/'.repro/runs/wc4-fulltask-reaudit-r11-20261021'
R14=ROOT/'.repro/runs/wc4-fulltask-act-r14-s1000-20261022/continuation-train'
R15=ROOT/'.repro/runs/wc4-fulltask-act-r15-s1000-20261022-cont001'
DATA=R11/'lerobot/local/wc4_fulltask_reaudit_r11_20261021'
MODELS={30000:R14,40000:R15,50000:R15}
SEEDS=[62202,62206,62210,62214]
BUDGET=1536

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def ahash(a):return hashlib.sha256(a.tobytes()).hexdigest()
def wrap(x):
 import numpy as np
 return (x+np.pi)%(2*np.pi)-np.pi

def dataset_config():
 from lerobot.configs.train import TrainPipelineConfig
 from lerobot.configs.default import DatasetConfig
 from lerobot.policies.act.configuration_act import ACTConfig
 return TrainPipelineConfig(dataset=DatasetConfig(repo_id='local/wc4_fulltask_reaudit_r11_20261021',root=str(DATA),use_imagenet_stats=False),policy=ACTConfig(device='cpu',chunk_size=100,n_action_steps=50,use_action_loss_weight=False),batch_size=1,num_workers=0)
