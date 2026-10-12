"""Compare cached observation/time inference to the original full model forward."""
import argparse
import json
import numpy as np
import torch
from policy import Policy, images_to_torch, scheduler
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--episode',required=True);p.add_argument('--inference-steps',type=int,choices=[25,100],default=25);a=p.parse_args()
policy=Policy(a.checkpoint,seed=123,inference_steps=a.inference_steps)
z=np.load(a.episode)
obs={'state':z['state'][400], 'images':z['images'][400]}
got,_=policy.act(obs)
state=torch.as_tensor(np.repeat(obs['state'][None,None],4,axis=1),device='cuda')
state=(state-policy.stats['state_mean'])/policy.stats['state_std']
imgs=images_to_torch(np.repeat(obs['images'][None,None],4,axis=1),'cuda')
generator=torch.Generator(device='cuda').manual_seed(123)
schedule=scheduler();schedule.set_timesteps(a.inference_steps,device='cuda')
with torch.no_grad():
    sample=torch.randn((1,16,10),device='cuda',generator=generator)
    for t in schedule.timesteps:
        noise=policy.model(sample,t.reshape(1),state,*imgs)
        sample=schedule.step(noise,t,sample,generator=generator).prev_sample
    expected=(sample[0,0]*policy.stats['action_std']+policy.stats['action_mean']).cpu().numpy()
error=float(np.max(np.abs(got-expected)))
print(json.dumps(dict(inference_steps=a.inference_steps,cached_vs_reference_max_abs_error=error,passed=error<1e-5)),flush=True)
assert error<1e-5
