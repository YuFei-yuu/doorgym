"""CPU-only fixed R9 dataset and new R10 evidence paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,pathlib
ROOT=pathlib.Path('/data/zyf/video2door-wc4-act')
OLD=ROOT/'.repro/runs/wc4-fulltask-reaudit-r11-20261021'
RUN=ROOT/'.repro/runs/wc4-fulltask-reaudit-r11-20261021'
DATA=OLD/'lerobot/local/wc4_fulltask_reaudit_r11_20261021'
REPO='local/wc4_fulltask_reaudit_r11_20261021'

def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def config():
 from lerobot.configs.train import TrainPipelineConfig
 from lerobot.configs.default import DatasetConfig
 from lerobot.policies.act.configuration_act import ACTConfig
 return TrainPipelineConfig(dataset=DatasetConfig(repo_id=REPO,root=str(DATA),use_imagenet_stats=False),
  policy=ACTConfig(device='cpu',chunk_size=100,n_action_steps=50,use_action_loss_weight=False,plucker_conditioning=False,interaction_state_conditioning=False,camera_input_gating=False,handle_latent_aux=False,point_cloud_conditioning=False,end_signal_prediction=False),batch_size=16,num_workers=3,seed=1000)

INDEX=read(RUN/'candidate-index.json') if (RUN/'candidate-index.json').exists() else None
EPISODES=len(INDEX['episodes']) if INDEX else 0
FRAMES=sum(e['frames'] for e in INDEX['episodes']) if INDEX else 0
