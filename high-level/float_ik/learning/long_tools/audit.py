"""Data/early-loop contract audit and paired teacher-observation diagnostics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import STATE_NAMES

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--old', required=True)
p.add_argument('--phase', choices=['data','early','offline'], required=True)
a = p.parse_args()
r, old = Path(a.root), Path(a.old)
manifest = json.loads((old/'manifest.json').read_text())
checkpoint = old/'training/checkpoints/step_005000.pt'

if a.phase == 'offline':
    from policy import Policy
    result = []
    for steps in (25,100):
        controller = Policy(checkpoint, seed=1234, inference_steps=steps, profile=True)
        errors, timings = [], []
        centers = list(range(0,61,4)) + [300,500,950,1600,2300,2700]
        for entry in [x for x in manifest if x['split']=='val' and x['success']][:2]:
            with np.load(entry['episode']) as z:
                states, images, actions = z['state'], z['images'], z['action']
            for center in centers:
                controller.history.clear(); controller.queue.clear()
                for idx in range(center-3,center):
                    i = max(0,idx)
                    controller.history.append(dict(state=states[i],images=images[i]))
                action, _ = controller.act(dict(state=states[center],images=images[center]))
                errors.append(action-actions[center]); timings.append(controller.last_timing.copy())
        errors = np.asarray(errors)
        result.append(dict(inference_steps=steps, centers=centers, samples=len(errors),
                           rmse_per_channel=np.sqrt(np.mean(errors**2,axis=0)).tolist(),
                           errors=errors.tolist(),timings=timings,
                           meaning='Recorded expert observations, not closed-loop success'))
        del controller
    (r/'offline_diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    raise SystemExit()

ck = torch.load(checkpoint, map_location='cpu')
states, actions = [], []
previous_errors = []
for entry in manifest:
    if entry['split']=='train' and entry['success']:
        with np.load(entry['episode']) as z:
            s, act = z['state'], z['action']
        states.append(s); actions.append(act)
        previous_errors.append(float(np.max(np.abs(s[1:,-10:]-act[:-1]))))
s, act = np.concatenate(states), np.concatenate(actions)
expected = dict(state_mean=torch.from_numpy(s).mean(0),state_std=torch.from_numpy(s).std(0).clamp_min(.01),action_mean=torch.from_numpy(act).mean(0),action_std=torch.from_numpy(act).std(0).clamp_min(.01))
checks = dict(manifest_equal=ck['manifest']==manifest,
              previous_applied_action=max(previous_errors)==0,
              normalization_train_only=all(torch.allclose(expected[k],ck['stats'][k],rtol=1e-6,atol=1e-7) for k in expected),
              finite_training_state_action=bool(np.isfinite(s).all() and np.isfinite(act).all()))
stats = {k:v.numpy() for k,v in ck['stats'].items()}
result = dict(checks=checks, state_names=STATE_NAMES,
              state_min=s.min(0).tolist(),state_max=s.max(0).tolist(),
              state_std=stats['state_std'].tolist(),
              action_min=act.min(0).tolist(),action_max=act.max(0).tolist(),
              old_data_missing=['gym_depth_m','next_state'], training_manifest_changed=False)
if a.phase == 'early':
    details = []
    for path in sorted((r/'diagnostic').glob('*/early_diagnostic.npz')):
        with np.load(path) as z:
            d = {k:z[k] for k in z.files}
        summary = json.loads((path.parent/'summary.json').read_text())
        n = len(d['state'])
        checks[path.parent.name+'_post_state'] = bool(np.array_equal(d['next_state'][:-1],d['state'][1:]) and np.array_equal(d['next_state'][:,-10:],d['action']))
        checks[path.parent.name+'_finite'] = bool(np.isfinite(d['state']).all() and np.isfinite(d['raw_action']).all())
        processed = np.nan_to_num(abs(d['gym_depth_m']),nan=0.,posinf=0.,neginf=0.)
        processed[processed<.02] = 0
        checks[path.parent.name+'_depth'] = bool(np.array_equal(np.clip(processed,0,2),d['depth_m']))
        # Initial-frame axes: integrate the body command using relative yaw.
        yaw = np.arctan2(d['state'][:,2],d['state'][:,3])
        velocities = d['action'][:,:2]
        delta = np.stack([np.cos(yaw)*velocities[:,0]-np.sin(yaw)*velocities[:,1],np.sin(yaw)*velocities[:,0]+np.cos(yaw)*velocities[:,1]],axis=1)/60
        checks[path.parent.name+'_base_coordinates'] = bool(np.allclose(d['next_state'][:,:2]-d['state'][:,:2],delta,atol=3e-6))
        checks[path.parent.name+'_quaternion_continuity'] = bool(np.all(np.sum(d['state'][1:,24:28]*d['state'][:-1,24:28],axis=1)>=-1e-6))
        detail = dict(episode=path.parent.name,mode=summary['mode'],inference_steps=summary['inference_steps'],
                      first_half_clipped_s=next((i/60 for i,x in enumerate(d['clipped_channels'].mean(1)) if x>=.5),None),
                      dq_finite_difference_rmse=np.sqrt(np.mean((np.diff(d['state'][:,7:14],axis=0)*60-d['state'][1:,14:21])**2,axis=0)).tolist())
        if 'history_state' in d:
            checks[path.parent.name+'_history'] = all(np.array_equal(d['history_state'][i],d['state'][np.maximum(np.arange(i-3,i+1),0)]) for i in range(n))
            checks[path.parent.name+'_history_images'] = all(np.array_equal(d['history_images'][i],d['images'][np.maximum(np.arange(i-3,i+1),0)]) for i in range(n))
            checks[path.parent.name+'_normalized_input'] = bool(np.allclose(d['normalized_history'],(d['history_state']-stats['state_mean'])/stats['state_std'],atol=1e-5))
            checks[path.parent.name+'_decision_queue'] = bool(np.array_equal(d['decision'],np.arange(n)%4==0))
            ood = (d['state']<s.min(0)-1e-3)|(d['state']>s.max(0)+1e-3)
            first = next((i for i,x in enumerate(ood.any(1)) if x),None)
            detail.update(first_outside_training_range_s=first/60 if first is not None else None,
                          first_outside_channels=[STATE_NAMES[j] for j in np.flatnonzero(ood[first])] if first is not None else [],
                          max_abs_z_per_channel=np.max(abs(d['normalized_state']),0).tolist())
        details.append(detail)
    result['early_episodes'] = details
    checks['paired_six_policy_episodes'] = sum(x['mode']=='policy' for x in details)==6
    checks['corrected_expert_recording'] = any(x['mode']=='expert' for x in details)
result['passed'] = all(checks.values())
(r/('audit_%s.json'%a.phase)).write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
assert result['passed'],checks
