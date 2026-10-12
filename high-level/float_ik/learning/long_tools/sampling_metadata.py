"""CPU-only exact DDPM timetable/SNR audit for the installed scheduler."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from policy import scheduler
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
result=[]
for count in (25,100):
    schedule=scheduler();schedule.set_timesteps(count,device='cpu')
    first=int(schedule.timesteps[0]);alpha=float(schedule.alphas_cumprod[first])
    result.append(dict(inference_steps=count,timesteps=schedule.timesteps.tolist(),first_timestep=first,
                       first_alpha_cumprod=alpha,first_epsilon_error_to_x0_gain=((1-alpha)/alpha)**.5,
                       formula='x0=(xt-sqrt(1-alpha_bar)*epsilon_pred)/sqrt(alpha_bar)',
                       note='Noise-error amplification in predicted x0, not proof of the full closed-loop failure cause'))
payload=dict(schedules=result,scheduler_source=inspect.getfile(type(schedule)),cuda_initialized=torch.cuda.is_initialized())
assert not payload['cuda_initialized']
(Path(a.root)/'sampling_metadata.json').write_text(json.dumps(payload,indent=2))
print(json.dumps(payload),flush=True)
