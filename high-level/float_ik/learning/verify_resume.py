"""Compare real source-model continuous and interrupted/resumed training updates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def equal(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--continuous',required=True)
    p.add_argument('--resumed',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    first=torch.load(a.continuous,map_location='cpu',weights_only=False)
    second=torch.load(a.resumed,map_location='cpu',weights_only=False)
    checks={key:equal(first[key],second[key]) for key in ('model','optimizer','stats','feature_names','action_names')}
    checks['sampler']=equal(first['training_state']['sampler'],second['training_state']['sampler'])
    checks['rng']=equal(first['training_state']['rng'],second['training_state']['rng'])
    checks['updates']=first['config']['completed_updates']==second['config']['completed_updates']==4
    result=dict(checks=checks,passed=all(checks.values()))
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
    if not result['passed']: raise SystemExit(1)
