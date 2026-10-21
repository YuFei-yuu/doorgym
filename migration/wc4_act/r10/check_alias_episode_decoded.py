"""Confirm diagnostic image proximity through the actual training factory."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,pathlib,numpy as np
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.datasets.factory import make_dataset
R=pathlib.Path('/data/zyf/video2door-wc4-act/.repro/runs/wc4-fulltask-expand-r9-20261021');from common import RUN
F=RUN/'dataset-check'
report=json.loads((F/'learnability-by-episode.json').read_text());pairs={}
for group in report['reviewed_by_horizon'].values():
    for c in group:pairs[(c['a'],c['b'])]=c
c=report['r8_anchor'];pairs[(c['a'],c['b'])]=c
cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id='local/wc4_fulltask_expand_r9_20261021',root=str(R/'lerobot/local/wc4_fulltask_expand_r9_20261021'),use_imagenet_stats=False),
    policy=ACTConfig(device='cpu',chunk_size=100,n_action_steps=50,use_action_loss_weight=False,plucker_conditioning=False,interaction_state_conditioning=False,camera_input_gating=False,handle_latent_aux=False,point_cloud_conditioning=False,end_signal_prediction=False),batch_size=16,num_workers=3)
ds=make_dataset(cfg);keys=[k for k in ds.features if k.startswith('observation.images.')]
result=[]
for (a,b),c in sorted(pairs.items()):
    x=ds[a];y=ds[b];rmse={k:float(np.sqrt(np.mean((x[k].numpy()-y[k].numpy())**2))) for k in keys}
    result.append(dict(a=a,b=b,raw_depth_rmse=c['full_depth_rmse'],decoded_depth_rmse=rmse,decoded_near_both=max(rmse.values())<=.02,
        significant_by_horizon=c['significant_by_horizon'],raw_near_both=c['near_full_depth']))
(F/'learnability-decoded-by-episode.json').write_text(json.dumps(dict(factory='lerobot.datasets.factory.make_dataset',pairs=result,checked_pairs=len(result),
    no_model_constructed=True,diagnostic_only=True,threshold_normalized_rmse=.02,scope='Training H264 decode for the bounded raw-image diagnostic pairs; not an exhaustive search or an acceptance threshold.'),indent=2)+'\n')
print('Training-factory decoded diagnostic pairs checked:',len(result))
