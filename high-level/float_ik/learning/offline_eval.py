"""Held-out teacher-observation inference diagnostics; NOT closed-loop success."""
import argparse
import json
from pathlib import Path
import numpy as np
from artifacts import resolve_recorded_path, write_new_json
from dataset_contract import validate_episode
from policy import Policy
from schema import DEFAULT_SCHEMA

p = argparse.ArgumentParser()
p.add_argument('--root', required=True); p.add_argument('--output', required=True)
p.add_argument('--checkpoint', action='append', help='Repeat for selected checkpoints; default: all checkpoints in this run')
p.add_argument('--device', default='cuda:0'); p.add_argument('--inference-steps', type=int, choices=[25,100], default=25)
p.add_argument('--episodes', type=int, default=2); p.add_argument('--windows', type=int, default=10)
a=p.parse_args(); r=Path(a.root).resolve()
if Path(a.output).exists(): raise FileExistsError(a.output)
if a.episodes < 1 or a.windows < 1: raise ValueError('Positive episode/window counts required')
manifest=json.loads((r/'manifest.json').read_text())
val=[x for x in manifest if x['split']=='val' and x['success']][:a.episodes]
paths=[resolve_recorded_path(r,x) for x in a.checkpoint] if a.checkpoint else sorted((r/'training/checkpoints').glob('step_*.pt'))
if not val or not paths: raise ValueError('Need accepted validation episodes and checkpoints')
result=[]
for path in paths:
    policy=Policy(path,device=a.device,seed=1234,profile=a.device.startswith('cuda'),inference_steps=a.inference_steps,schema=DEFAULT_SCHEMA)
    errors=[]; predictions=[]; rotations=[]; windows=[];timings=[]
    for entry in val:
        with np.load(resolve_recorded_path(r,entry['episode'])) as z:
            validate_episode(z,DEFAULT_SCHEMA)
            for center in np.unique(np.linspace(0,len(z['action'])-1,a.windows,dtype=int)):
                policy.history.clear(); policy.queue.clear()
                for idx in range(center-policy.config['obs_horizon']+1,center):
                    i=max(0,idx);policy.history.append({'state':z['state'][i],'images':z['images'][i]})
                action,_=policy.act({'state':z['state'][center],'images':z['images'][center]})
                # Float64 diagnostics must not overflow when finite float32 predictions are huge.
                action=action.astype(float); target=z['action'][center].astype(float)
                errors.append(action-target);predictions.append(action);timings.append(policy.last_timing.copy())
                norm=np.linalg.norm(action[6:10]);rotations.append(2*np.arccos(np.clip(abs(np.dot(action[6:10]/norm,target[6:10])),0,1)) if norm>1e-8 else np.nan)
                windows.append(dict(seed=entry['seed'],frame=int(center)))
    errors=np.asarray(errors);rmse=np.sqrt(np.mean(errors**2,axis=0))
    result.append(dict(checkpoint=str(path),step=policy.step,schema=policy.schema,inference_steps=a.inference_steps,samples=len(errors),windows=windows,action_names=policy.action_names,rmse_per_channel=rmse,position_vector_rmse_m=np.linalg.norm(rmse[3:6]),orientation_geodesic_mean_rad=np.mean(rotations),invalid_rotation_count=int((~np.isfinite(rotations)).sum()),max_abs_prediction=float(np.max(np.abs(predictions))),timing_sec=timings,meaning='Teacher-observation action error; not closed-loop success. Raw quaternion component RMSE is not rotation distance.'))
    del policy
write_new_json(a.output,result)
print(json.dumps({'checkpoints':len(result),'output':a.output}))
