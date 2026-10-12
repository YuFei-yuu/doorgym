#!/usr/bin/env python3
"""CPU audit of current Cartesian recordings and train-only checkpoint statistics."""
import argparse
import json
from pathlib import Path
import numpy as np
from artifacts import resolve_recorded_path, write_new_json
from dataset_contract import validate_episode
from schema import DEFAULT_SCHEMA, contract, task_success


def audit_run(root, checkpoint=None):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    names, states = contract(DEFAULT_SCHEMA)
    seeds = [x['seed'] for x in manifest]
    paths = [resolve_recorded_path(root, x['episode']) for x in manifest]
    checks = dict(nonempty_manifest=bool(manifest), disjoint_seeds=len(set(seeds)) == len(seeds),
                  disjoint_paths=len(set(paths)) == len(paths),
                  valid_splits=all(x['split'] in ('train', 'val', 'test') for x in manifest),
                  success_boundary=task_success(80, True, .52) and not task_success(-80, True, .52) and not task_success(80, False, .52))
    train = []; details = []
    for item, path in zip(manifest, paths):
        with np.load(path) as z:
            validate_episode(z, DEFAULT_SCHEMA)
            n = len(z['action'])
            ok = (n > 0 and z['next_state'].shape == z['state'].shape and
                  np.isfinite(z['next_state']).all() and np.array_equal(z['state'][1:], z['next_state'][:-1]) and
                  np.array_equal(z['frame'], np.arange(n)) and np.array_equal(z['next_frame'], np.arange(1, n+1)))
            details.append(dict(path=str(path), seed=item['seed'], frames=n, aligned=bool(ok)))
            if item['split'] == 'train' and item['success']:
                train.append({k: z[k] for k in ('state', 'action')})
    checks['all_recordings_aligned'] = all(x['aligned'] for x in details)
    if checkpoint:
        import torch
        ck = torch.load(resolve_recorded_path(root, checkpoint), map_location='cpu')
        checks['checkpoint_contract'] = ck['schema'] == DEFAULT_SCHEMA and ck['action_names'] == names and ck['state_names'] == states
        # Compare original manifest contents before remapping file paths for local mirrors.
        checks['checkpoint_manifest'] = ck['manifest'] == manifest
        floor = ck['training']['norm_std_floor']
        checks['normalization_train_only'] = bool(train)
        for key in ('state', 'action'):
            if not train: break
            values = torch.from_numpy(np.concatenate([x[key] for x in train]))
            checks['normalization_train_only'] &= bool(torch.allclose(values.mean(0), ck['stats'][key+'_mean']) and torch.allclose(values.std(0).clamp_min(floor), ck['stats'][key+'_std']))
    return dict(schema=DEFAULT_SCHEMA, checks=checks, episodes=details, passed=all(checks.values()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--checkpoint'); p.add_argument('--output', required=True); a = p.parse_args()
    result = audit_run(a.root, a.checkpoint); write_new_json(a.output, result)
    print(json.dumps(dict(passed=result['passed'], checks=result['checks'])))
    if not result['passed']: raise SystemExit(1)
