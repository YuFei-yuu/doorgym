"""Prepare one authorized nominal attempt; no Gym imports or scene creation."""
import argparse, copy, hashlib, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE='58e488be8ab5f9b6f358aa803e750d5e64af3088'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=False)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    old=ROOT/'.repro/runs/wc4-fulltask-pilot-r8-20261021'
    r7=ROOT/'.repro/runs/wc4-fullcycle-compact-r7-20261021/attempt_000'
    manifest=json.loads((old/'manifest.json').read_text())
    trial=copy.deepcopy(manifest['trials'][0]);assert trial['seed']==62000 and trial['nominal']
    yaml=run/'frozen-r7-physics.yaml';yaml.write_bytes((old/'frozen-r7-physics.yaml').read_bytes())
    cli=trial['argv'];cli[cli.index('--steps')+1]='7500'
    cli[cli.index('--a2w_float_ik_config')+1]=str(yaml)
    assert cli[cli.index('--num_envs')+1]=='1'
    assert '--no_ikpush_env_randomization' in cli and '--no_enable_depth_camera_randomization' in cli
    frozen=ROOT/'migration/wc4_act/newexpert_stage3/geometry.json'
    geom=json.loads(frozen.read_text());geom['status']='candidate_found'
    geom['live_joint9_recording']=True
    geom['execution_authorization']={'stage':4,'episodes':1,'maximum_sim_seconds':150,
        'basis':'User fourth-round task order; unchanged stage-three expert and geometry'}
    # Offline-only metadata is retained verbatim as historical design metadata.
    write(run/'geometry.json',geom)
    write(run/'geometry-diff.json',{'source':str(frozen),'source_sha256':sha(frozen),
        'compact_expert_identical':geom['compact_expert']==json.loads(frozen.read_text())['compact_expert'],
        'changes':{'status':['offline_review_only','candidate_found'],
                   'live_joint9_recording':[None,True],'execution_authorization':geom['execution_authorization']}})
    old_sources=json.loads((ROOT/'.repro/reviews/wc4-r7-design-20261021/consistency.json').read_text())['source_sha256']
    protected={n:h for n,h in old_sources.items() if n.startswith(('high-level/data/','high-level/float_ik/'))}
    assert all(sha(ROOT/n)==h for n,h in protected.items()),'Protected R7 asset/opening source differs'
    files=set(old_sources)
    for folder in ['migration/wc4_act','migration/wc4_act/newexpert_stage2',
                   'migration/wc4_act/newexpert_stage3','migration/wc4_act/newexpert_stage4',
                   'migration/wc4_act/r11']:
        files.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).glob('*') if p.is_file())
    sources={n:sha(ROOT/n) for n in sorted(files)}
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.check_call(['git','merge-base','--is-ancestor',BASE,head],cwd=ROOT)
    # The approved runtime itself must be byte-identical to the handoff.
    for n in ['migration/wc4_act/fullcycle_compact.py','migration/wc4_act/newexpert_stage2/push_geometry.py',
              'migration/wc4_act/newexpert_stage3/geometry.json']:
        assert hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+n],cwd=ROOT)).hexdigest()==sources[n]
    new={'start_commit':head,'maximum_scenes':1,'maximum_sim_seconds':150,'seeds':[62000],
         'trials':[trial],'r7_physics_after_asset_overrides':manifest['r7_physics_after_asset_overrides'],
         'yaml':str(yaml),'yaml_sha256':sha(yaml),'source_sha256':sources,
         'provenance':{'nominal_manifest':str(old/'manifest.json'),'sha256':sha(old/'manifest.json'),
                       'r7_initial_state':str(r7/'initial-state.json'),'r7_effective_config':str(r7/'effective-config.json')}}
    write(run/'manifest.json',new)
    write(run/'consistency.json',{'status':'source_route_consistent','geometry_sha256':sha(run/'geometry.json'),
         'source_sha256':sources,'physical_execution_verified':False,
         'basis':'Third-round frozen CPU evidence plus explicit fourth-round single-live-attempt authorization'})
    for n in ['initial-state.json','effective-config.json','command.json','body-map.json']:
        (run/('r7-'+n)).write_bytes((r7/n).read_bytes())
    write(run/'source-binding.json',{'head':head,'base':BASE,'main':subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip(),
          'protected_r7_files':protected,'all_protected_identical':True})
    write(run/'budget.json',{'authorized_episodes':1,'maximum_sim_seconds':150,'scene_claims':0,
        'training_updates':0,'policy_evaluations':0,'collection_campaign':0,'retry_allowed':False,
        'reserved_run_bytes':10*1024**3,'headroom_bytes':40*1024**3})
    print(json.dumps({'prepared':str(run),'head':head,'episodes':1,'steps':7500}),flush=True)

if __name__=='__main__':main()
