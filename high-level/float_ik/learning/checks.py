#!/usr/bin/env python3
"""Read-only recording, split and upstream checkpoint normalization checks."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from artifacts import resolve_recorded_path,write_new_json
from dataset_contract import validate_episode


def audit_run(root,checkpoint=None,expected_updates=None):
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text());entries=[x for x in manifest if x.get('success',False)]
    seeds=[x['seed'] for x in manifest];paths=[str(x['episode']) for x in manifest]
    checks=dict(nonempty_manifest=bool(entries),disjoint_seeds=len(set(seeds))==len(seeds),disjoint_paths=len(set(paths))==len(paths),valid_splits=all(x['split'] in ('train','val','test') for x in manifest))
    details=[];train=[];contracts=set()
    for item in entries:
        for field,hash_field in [('episode','sha256'),('raw_episode','raw_sha256')]:
            if hash_field in item:
                path=resolve_recorded_path(root,item[field])
                checks['recording_hashes']=checks.get('recording_hashes',True) and hashlib.sha256(path.read_bytes()).hexdigest()==item[hash_field]
        with np.load(resolve_recorded_path(root,item['episode'])) as z:
            meta=validate_episode(z);contracts.add((tuple(meta['state_names']),tuple(meta['action_names']),meta['schema'],meta['execution_profile']))
            details.append(dict(seed=item['seed'],split=item['split'],**meta))
            if item['split']=='train':train.append({k:z[k].copy() for k in ('state','action')})
    checks['one_recording_contract']=len(contracts)==1
    if checkpoint:
        import torch
        ck=torch.load(resolve_recorded_path(root,checkpoint),map_location='cpu',weights_only=False);cfg=ck['config'];meta=details[0]
        checks['checkpoint_dimensions']=cfg['state_dim']==len(meta['state_names']) and cfg['action_dim']==len(meta['action_names'])
        if expected_updates is not None:
            checks['checkpoint_updates']=cfg.get('completed_updates')==expected_updates
        checks['normalization_train_only']=bool(train)
        for key in ('state','action'):
            if not train:break
            values=torch.from_numpy(np.concatenate([x[key] for x in train]))
            checks['normalization_train_only'] &= bool(torch.allclose(values.mean(0),ck['stats'][key+'_mean'],rtol=1e-5,atol=1e-6) and torch.allclose(values.std(0).clamp_min(1e-6),ck['stats'][key+'_std'],rtol=1e-5,atol=1e-6))
    return dict(checks=checks,episodes=details,passed=all(checks.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--checkpoint');p.add_argument('--expected-updates',type=int);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit_run(a.root,a.checkpoint,a.expected_updates);write_new_json(a.output,result);print(json.dumps(result['checks']))
    if not result['passed']:raise SystemExit(1)
