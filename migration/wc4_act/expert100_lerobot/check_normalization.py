"""Actual ACT processors with new dataset stats, without constructing a policy."""
from common import *
import numpy as np,torch
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import NormalizerProcessorStep,UnnormalizerProcessorStep

def main():
 cfg=config();ds=make_dataset(cfg);features=dataset_to_policy_features(ds.features)
 cfg.policy.output_features={k:v for k,v in features.items() if v.type is FeatureType.ACTION}
 cfg.policy.input_features={k:v for k,v in features.items() if k not in cfg.policy.output_features}
 assert cfg.dataset.use_imagenet_stats is False and not cfg.dataset.image_transforms.enable
 assert cfg.policy.pretrained_path is None and not cfg.resume
 pre,post=make_pre_post_processors(policy_cfg=cfg.policy,pretrained_path=None,dataset_stats=ds.meta.stats)
 normal=next(s for s in pre.steps if isinstance(s,NormalizerProcessorStep))
 unnormal=next(s for s in post.steps if isinstance(s,UnnormalizerProcessorStep))
 keys=list(cfg.policy.input_features)+list(cfg.policy.output_features)
 for processor in [normal,unnormal]:
  for key in processor.features:
   for field,tensor in processor._tensor_stats[key].items():
    torch.testing.assert_close(tensor,torch.as_tensor(ds.meta.stats[key][field],dtype=tensor.dtype),rtol=0,atol=0)
 errors={k:0. for k in keys}
 for i in [0,len(ds)//2,len(ds)-1]:
  sample=ds[i];before={k:sample[k].clone() for k in keys}
  processed=pre(sample)
  for key in keys:
   st=normal._tensor_stats[key]
   expected=(before[key]-st['mean'])/(st['std']+normal.eps)
   expected=torch.broadcast_to(expected,processed[key].shape)
   torch.testing.assert_close(processed[key],expected,rtol=0,atol=0)
   errors[key]=max(errors[key],float(torch.max(abs(processed[key]-expected))))
   assert torch.isfinite(processed[key]).all()
 result=post(torch.zeros((1,100,9),dtype=torch.float32))
 expected=torch.broadcast_to(unnormal._tensor_stats['action']['mean'],result.shape)
 torch.testing.assert_close(result,expected,rtol=0,atol=0)
 write(RUN/'normalization-check.json',dict(passed=True,processor_factory='make_pre_post_processors',new_stats_loaded_exactly=True,
  stats_sha256=sha(DATA/'meta/stats.json'),use_imagenet_stats=False,image_transforms=False,normalization_mapping={k:v.value for k,v in cfg.policy.normalization_mapping.items()},
  maximum_errors=errors,postprocessor_zero_action_equals_new_mean=True,model_constructed=False,optimizer_constructed=False,training=False))
 print('ACT processors loaded new stats and normalized inputs/actions exactly; no model created',flush=True)
if __name__=='__main__':main()
