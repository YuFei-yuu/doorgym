"""Accepted 72-episode stats -> actual processors -> direct backend, no model."""
import sys,json
from pathlib import Path
import torch
from runtime import *
sys.path.insert(0,str(ROOT/'high-level/dp'))
from door_policy_backend import DoorPolicyNormalizer,load_lerobot_policy_normalizer_stats
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.utils import dataset_to_policy_features
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
ACTPolicy.__init__=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No ACT construction in R13'))
features=dataset_to_policy_features(read(DATA/'meta/info.json')['features'])
keys=['observation.state','observation.images.front_masked_depth','observation.images.wrist_masked_depth']
cfg=ACTConfig(device='cpu',chunk_size=100,n_action_steps=50,push_to_hub=False,use_action_loss_weight=False)
cfg.input_features={k:features[k] for k in keys};cfg.output_features={'action':features['action']}
stats={k:{n:torch.as_tensor(v,dtype=torch.float32) for n,v in values.items()} for k,values in read(DATA/'meta/stats.json').items()}
pre,post=make_pre_post_processors(cfg,dataset_stats=stats)
pre.save_pretrained(out/'processors');post.save_pretrained(out/'processors')
saved_stats=load_lerobot_policy_normalizer_stats(out/'processors');assert saved_stats
normalizer=DoorPolicyNormalizer(saved_stats,cfg.normalization_mapping,keys[1:],torch.device('cpu'))
batch={'observation.state':torch.linspace(-1,1,18).reshape(2,9),'action':torch.zeros(2,100,9),
    keys[1]:torch.linspace(0,1,120).reshape(2,3,4,5),keys[2]:torch.ones(2,3,4,5)}
expected=pre({k:v.clone() for k,v in batch.items()})
actual=normalizer.normalize_batch(batch)
errors={k:float((expected[k]-actual[k]).abs().max()) for k in batch}
assert max(errors.values())<=1e-6,errors
z=torch.zeros(2,100,9)
a=post(z.clone());b=normalizer.denormalize_action(z.clone())
error=float((a-b).abs().max());assert error<=1e-6
pre2,post2=make_pre_post_processors(cfg,pretrained_path=str(out/'processors'))
again=pre2({k:v.clone() for k,v in batch.items()})
assert all(torch.equal(expected[k],again[k]) for k in batch)
assert all(expected[k].dtype==torch.float32 for k in batch)
write(out/'result.json',dict(status='passed',data_stats_sha256=sha(DATA/'meta/stats.json'),
    accepted_data_episodes=72,accepted_data_frames=159529,normalization_max_errors=errors,
    denormalization_max_error=error,processor_save_reload_equal=True,float32_preserved=True,
    synthetic_tensor_inputs=True,real_policy_inference=0,ACT_training_updates=0,model_constructed=False))
print('PASS: real 72-episode stats, official processors save/reload and direct backend normalization/denormalization; no model')
