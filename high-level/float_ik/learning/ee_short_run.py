#!/usr/bin/env python3
"""One bounded Cartesian smoke run; explicit stages, immutable attempts, no tuning."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE / 'run_tools'))
from bounded import atomic_json, run, used

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--plan', help='Required for init: explicit new-run JSON plan; no inherited seeds/budget')
p.add_argument('stage', choices=['init', 'pilot', 'collect', 'train', 'evaluate'])
a = p.parse_args()
r = Path(a.root).resolve()
r.mkdir(parents=True, exist_ok=True)
lock = (r/'driver.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if a.stage == 'init':
    if not a.plan: p.error('--plan is required for init')
    plan = json.loads(Path(a.plan).read_text())
else:
    plan = json.loads((r/'plan.json').read_text())
required = {'schema','total_seconds','optimizer_steps','train_seeds','offline_val_seeds','rollout_val_seeds','pilot_seed','inference_steps','train_seconds','teacher','checkpoint_selection'}
if not required.issubset(plan): raise ValueError('Missing plan keys: %s' % (required-set(plan)))
if plan['schema'] != 'float_push_ee_world_v1': raise ValueError('Only Cartesian plans supported')
seeds = plan['train_seeds']+plan['offline_val_seeds']+plan['rollout_val_seeds']+[plan['pilot_seed']]
if len(seeds)!=len(set(seeds)): raise ValueError('Plan seed sets overlap')
if min(plan['total_seconds'],plan['optimizer_steps'],plan['train_seconds']) <= 0: raise ValueError('Positive budget and update limits required')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source_hashes():
    files=list(HERE.rglob('*.py')) + [HERE.parent/'isaacgym_float_ik_b1z1_basearn_push_door.py', HERE.parent/'isaacgym_visualize_b1z1_basearn.py', REPO/'high-level/dp/models/door_diffusion_policy.py']
    return {str(f.relative_to(REPO)):digest(f) for f in files}

if a.stage == 'init':
    if (r/'plan.json').exists():
        raise RuntimeError('Existing experiment; do not reinitialize')
    atomic_json(r/'plan.json', plan)
    (r/'budget.jsonl').touch(exist_ok=False)
    atomic_json(r/'source_sha256.json', source_hashes())
    atomic_json(r/'provenance.json', dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(), initialized=time.time()))
    (r/'source.diff').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=REPO))
    (r/'source').mkdir()
    import shutil
    for name in source_hashes():
        dest=r/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest)
    print(json.dumps(plan, indent=2));sys.exit(0)
assert json.loads((r/'plan.json').read_text()) == plan
assert json.loads((r/'source_sha256.json').read_text()) == source_hashes(), 'Run source changed'

def launch(name, args, timeout, reserve=0):
    # A directory or log is an attempted episode; never silently replay it.
    samples=[]
    for i in range(3):
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        samples.append([[x.strip() for x in row.split(',')] for row in rows.splitlines()])
        if i<2:time.sleep(2)
    eligible=[]
    for j,row in enumerate(samples[0]):
        if min(float(s[j][2]) for s in samples)>=16384 and statistics.mean(float(s[j][3]) for s in samples)<70:
            eligible.append((statistics.mean(float(s[j][3]) for s in samples),int(row[0]),row[1]))
    resource=r/'resources';resource.mkdir(exist_ok=True)
    if (resource/(name+'.json')).exists():raise RuntimeError('Attempt already dispatched: '+name)
    atomic_json(resource/(name+'.json'),dict(samples=samples,eligible=eligible))
    if not eligible:raise RuntimeError('No eligible GPU')
    _,gpu,uuid=min(eligible)
    cmd='export VIDEO2DOOR_GPU=%d; source %s || exit; exec nice -n 10 '%(gpu,shlex.quote(str(REPO/'.repro/blackwell/activate.sh')))+shlex.join(['python']+list(map(str,args)))
    code=run(r,name,['bash','-c',cmd],timeout=timeout,total=plan['total_seconds'],reserve=reserve,gpu=gpu,gpu_uuid=uuid,log_name=name+'.log')
    if code:raise RuntimeError('Stage failed; retained attempt: '+name)

def episode(name, mode, seed, folder, nominal=False, checkpoint=None, video=False):
    out=r/folder/name
    cmd=[HERE/'rollout.py','--schema',plan['schema'],'--mode',mode,'--seed',seed,'--split',folder,'--record','--diagnostic-steps',60,'--output',out]
    if nominal:cmd+=['--nominal']
    if checkpoint:cmd+=['--checkpoint',checkpoint,'--inference-steps',plan['inference_steps']]
    if video:cmd+=['--video']
    launch(name,cmd,timeout=180,reserve=1200 if folder=='data' else 0)
    print('EPISODE',name,json.loads((out/'summary.json').read_text())['success'],flush=True)

if a.stage == 'pilot':
    episode('expert_nominal','expert',plan['pilot_seed'],'pilot',nominal=True,video=True)
elif a.stage == 'collect':
    gate=json.loads((r/'pilot_quality.json').read_text())
    assert gate['passed'], 'Expert pilot quality must pass before collection'
    for split,seeds in [('train',plan['train_seeds']),('val',plan['offline_val_seeds'])]:
        for seed in seeds:
            name='expert_%s_%d'%(split,seed)
            if (r/'data'/name/'summary.json').exists():
                # Retain completed/failed attempts, without retrying/replacing.
                continue
            episode(name,'expert',seed,'data',video=seed==plan['train_seeds'][0])
elif a.stage == 'train':
    gate=json.loads((r/'dataset_quality.json').read_text())
    assert gate['passed'], 'Dataset quality must pass before optimizer work'
    launch('train',[HERE/'train.py','--schema',plan['schema'],'--manifest',r/'manifest.json','--output',r/'training','--steps',plan['optimizer_steps'],'--seconds',plan['train_seconds']],timeout=plan['train_seconds']+100,reserve=900)
    latest=json.loads((r/'training/latest.json').read_text())
    assert latest['step']==plan['optimizer_steps'], 'Did not reach planned checkpoint; do not silently change selection'
    ck=Path(latest['checkpoint'])
    atomic_json(r/'frozen_model.json',dict(checkpoint=str(ck),step=plan['optimizer_steps'],sha256=digest(ck),selection=plan['checkpoint_selection']))
elif a.stage == 'evaluate':
    frozen=json.loads((r/'frozen_model.json').read_text());ck=Path(frozen['checkpoint'])
    assert digest(ck)==frozen['sha256']
    for seed in plan['rollout_val_seeds']:
        for mode in ['expert','policy']:
            name='%s_seed_%d'%(mode,seed)
            if (r/'validation'/name/'summary.json').exists():continue
            episode(name,mode,seed,'validation',checkpoint=ck if mode=='policy' else None,video=True)
print('ACCOUNTED_SECONDS',used(r),flush=True)
