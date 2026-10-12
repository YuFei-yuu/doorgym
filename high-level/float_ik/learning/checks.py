#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--root',required=True)
p.add_argument('--reset',action='store_true')
a=p.parse_args()
r=Path(a.root)
from schema import task_success
checks={}
previous_checks = json.loads((r/'checks.json').read_text()).get('checks',{}) if (r/'checks.json').exists() else {}
checks.update(previous_checks)
checks['success_boundary_and_direction'] = (task_success(80.,True,.52) and not task_success(79.999,True,.52) and not task_success(80.,True,.51999) and not task_success(-90.,True,.55) and not task_success(90.,False,.55))
with np.load(r/'adapter-nominal/episode.npz') as z:
    checks['state_38_action_10']=z['state'].shape==(2850,38) and z['action'].shape==(2850,10)
    checks['synchronous_frame_ids']=bool(np.array_equal(z['frame'],np.arange(2850)) and np.array_equal(z['next_frame'],np.arange(1,2851)))
    checks['all_finite']=bool(np.isfinite(z['state']).all() and np.isfinite(z['action']).all() and np.isfinite(z['depth_m']).all())
    checks['previous_applied_action']=bool(np.allclose(z['state'][1:,-10:],z['action'][:-1]))
    checks['both_cameras_observe_handle']=bool(z['images'][:,0].max()>0 and z['images'][:,2].max()>0)
expert=json.loads((r/'adapter-nominal/summary.json').read_text())
replay=json.loads((r/'adapter-replay/summary.json').read_text())
checks['independent_action_replay_success']=replay['success'] and abs(expert['signed_angle_deg']-replay['signed_angle_deg'])<.1
x=np.load(r/'adapter-nominal/trajectory.npz')['action']
y=np.load(r/'adapter-replay/trajectory.npz')['action']
checks['replay_actions_equal']=bool(np.allclose(x,y,atol=1e-6))
manifest = json.loads((r/'manifest.json').read_text()) if (r/'manifest.json').exists() else []
if manifest:
    seeds = [x['seed'] for x in manifest]
    paths = [x['episode'] for x in manifest]
    checks['episode_seed_disjoint'] = len(set(seeds)) == len(seeds) and len(set(paths)) == len(paths)
    checks['valid_split_names'] = all(x['split'] in ('train','val','test') for x in manifest)
if (r/'recording-audit/episode.npz').exists():
    with np.load(r/'recording-audit/episode.npz') as z:
        checks['audit_explicit_post_state_alignment']=bool(z['next_state'].shape==(2850,38) and np.array_equal(z['next_state'][:-1],z['state'][1:]))
        raw=z['gym_depth_m']
        processed=np.nan_to_num(np.abs(raw),nan=0.,posinf=0.,neginf=0.)
        processed[processed<.02]=0
        processed=np.clip(processed,0,2)
        checks['audit_original_depth_preserved']=bool(raw.shape==(2850,2,54,96) and np.array_equal(processed,z['depth_m']))
    audit=json.loads((r/'recording-audit/summary.json').read_text())
    checks['audit_nominal_expert_unchanged']=bool(audit['success'] and abs(audit['signed_angle_deg']-expert['signed_angle_deg'])<.1)
if (r/'selection.json').exists():
    selection=json.loads((r/'selection.json').read_text())
    scenarios={}
    test_seeds=set(selection['test_seeds'])
    checks['test_seeds_disjoint_from_training_validation']=not bool(test_seeds & (set(x['seed'] for x in manifest)|set(selection['validation_seeds'])))
    for path in (r/'test').glob('*/summary.json'):
        s=json.loads(path.read_text())
        scenarios.setdefault(s['seed'],[]).append(s.get('randomization'))
    checks['fixed_scenarios_match_across_three_methods']=bool(len(scenarios)==10 and all(len(v)==3 and v[0]==v[1]==v[2] for v in scenarios.values()))
if a.reset:
    from float_env import FloatEnv, rotate
    from schema import JOINT_SCHEMA
    e=FloatEnv(schema=JOINT_SCHEMA)
    o1=e.reset(7)
    first={k:v.copy() for k,v in o1.items() if hasattr(v,'copy')}
    for _ in range(3):
        o1,_,_,_=e.step(np.r_[[.1,0,0],e.targets])
    o2=e.reset(7)
    checks['reset_same_seed_state']=bool(np.allclose(first['state'],o2['state'],atol=1e-6))
    checks['reset_same_seed_images']=bool(np.array_equal(first['images'],o2['images']))
    checks['coordinate_roundtrip']=bool(np.allclose(rotate(rotate([.3,-.1],1.2),-1.2),[.3,-.1]))
    try:
        e.expert_action()
        checks['expert_disabled']=False
    except RuntimeError:
        checks['expert_disabled']=True
    e.close()
# Match checkpoint normalization against train-only frames; test/val cannot leak into stats.
if (r/'training/latest.json').exists():
    import torch
    ck=torch.load(json.loads((r/'training/latest.json').read_text())['checkpoint'],map_location='cpu')
    states=[];actions=[]
    for entry in manifest:
        if entry['split']=='train' and entry['success']:
            with np.load(entry['episode']) as z:
                states.append(z['state']);actions.append(z['action'])
    ts=torch.from_numpy(np.concatenate(states));ta=torch.from_numpy(np.concatenate(actions))
    checks['normalization_train_only'] = bool(torch.allclose(ts.mean(0),ck['stats']['state_mean']) and torch.allclose(ta.mean(0),ck['stats']['action_mean']) and torch.allclose(ts.std(0).clamp_min(.01),ck['stats']['state_std']) and torch.allclose(ta.std(0).clamp_min(.01),ck['stats']['action_std']))
import cv2
cap=cv2.VideoCapture(str(r/'adapter-nominal/rollout.mp4'))
fps=cap.get(cv2.CAP_PROP_FPS)
count=0
while True:
    ok,frame=cap.read()
    if not ok:
        break
    if count in [0,712,1424]:
        cv2.imwrite(str(r/('expert_video_frame_%d.png'%count)),frame)
    count+=1
cap.release()
checks['video_full_decode_1425_at30fps']=count==1425 and abs(fps-30)<1e-5
(r/'checks.json').write_text(json.dumps(dict(checks=checks,passed=all(checks.values())),indent=2))
print(json.dumps(checks),flush=True)
assert all(checks.values()),checks
