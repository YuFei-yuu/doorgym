"""Explicit paired diagnostic only; the existing limiter is not a recommended executor."""
import argparse,hashlib,json,sys
from pathlib import Path
here=Path(__file__).resolve().parents[1];sys.path.insert(0,str(here/'run_tools'));sys.path.insert(0,str(here))
from launch import launch
from artifacts import resolve_recorded_path,write_new_json
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--timeout',required=True,type=float);p.add_argument('--total-seconds',required=True,type=float);a=p.parse_args()
r=Path(a.root).resolve();frozen=json.loads((r/'frozen_model.json').read_text());ck=resolve_recorded_path(r,frozen['checkpoint'])
if hashlib.sha256(ck.read_bytes()).hexdigest()!=frozen['sha256']:raise ValueError('Frozen model changed')
for mode in ['expert','policy']:
    name='limited_%s_seed_%d'%(mode,a.seed);out=r/'execution_diagnostic'/name
    if out.exists():raise FileExistsError(out)
    args=['python',here/'short_tools/limited_rollout.py','--mode',mode,'--seed',a.seed,'--split','execution_diagnostic','--record','--video','--output',out]
    if mode=='policy':args+=['--checkpoint',ck]
    rc=launch(r,name,args,a.timeout,a.total_seconds)
    if rc:raise SystemExit(rc)
