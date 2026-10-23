"""Read real dataset samples at bypass endpoints and episode ends; no policy built."""
import os,sys,json,pathlib
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3];R=pathlib.Path(os.environ['WC4_EXECUTION_RUN']);sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.datasets.factory import make_dataset
D=ROOT/'.repro/runs/wc4-fulltask-reaudit-r11-20261021/lerobot/local/wc4_fulltask_reaudit_r11_20261021'
cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id='local/wc4_fulltask_reaudit_r11_20261021',root=str(D),use_imagenet_stats=False),policy=ACTConfig(device='cpu',chunk_size=100,n_action_steps=50,use_action_loss_weight=False),batch_size=1,num_workers=0)
ds=make_dataset(cfg);assert len(ds)==159529;w=json.loads((R/'windows.json').read_text());checked=[]
for e in w['segments']:
 ep=e['episode'];n=e['episode_frames'];off=e['episode_from_index'];p=D.parents[2]/'candidates'/f'episode_{ep:06d}.npz'
 with np.load(p,allow_pickle=True) as a:actions=a['action'].copy();states=a['state'].copy()
 for f in [e['lo'],e['hi_inclusive'],n-1]:
  s=ds[off+f];valid=min(100,n-f);assert np.array_equal(s['observation.state'].numpy(),states[f]);assert s['action'].shape==(100,9)
  assert np.array_equal(s['action'].numpy()[:valid],actions[f:f+valid]);assert np.array_equal(s['action_is_pad'].numpy(),np.arange(100)>=valid)
  if f!=n-1:assert valid==100
  checked.append(dict(episode=ep,frame=f,index=off+f,valid_future=valid,endpoint=('episode_end' if f==n-1 else 'bypass_endpoint'),phase_boundary_did_not_truncate=True))
 print('chunks',ep,flush=True)
x=dict(status='passed',checks=len(checked),episodes=72,records=checked,original_100future_actions_equal=True,original_episode_padding_equal=True,new_model_forwards=0,new_optimizer_updates=0,new_scenes=0)
(R/'future-chunk-interface-fixture.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:v for k,v in x.items() if k!='records'}))
