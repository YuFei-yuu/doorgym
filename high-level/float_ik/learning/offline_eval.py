"""Held-out teacher-observation inference diagnostics; NOT closed-loop success."""
import argparse
import json
from pathlib import Path
import numpy as np
from policy import Policy
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
r=Path(a.root)
manifest=json.loads((r/'manifest.json').read_text())
val=[x for x in manifest if x['split']=='val' and x['success']][:2]
result=[]
for step in [0,100,500,2000,5000]:
    path=r/'training/checkpoints'/('step_%06d.pt'%step)
    if not path.exists():continue
    policy=Policy(path,seed=1234,profile=True)
    errors=[];ranges=[];timings=[]
    for entry in val:
        z=np.load(entry['episode'])
        for center in [0,300,500,700,950,1250,1600,2000,2300,2700]:
            policy.history.clear();policy.queue.clear()
            for idx in range(center-3,center):
                i=max(0,idx)
                policy.history.append({'state':z['state'][i],'images':z['images'][i]})
            action,_=policy.act({'state':z['state'][center],'images':z['images'][center]})
            timings.append(policy.last_timing.copy())
            errors.append(action-z['action'][center]);ranges.append(action)
    errors=np.asarray(errors)
    result.append(dict(step=step,samples=len(errors),rmse_per_channel=np.sqrt(np.mean(errors**2,axis=0)).tolist(),max_abs_prediction=float(np.max(np.abs(ranges))),timing_sec=timings,meaning='offline action error and synchronized timing with recorded expert observations; not policy rollout'))
    del policy
(r/'offline_action_diagnostics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
