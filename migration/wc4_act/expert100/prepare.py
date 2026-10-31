"""Bind old50, all exclusions and exact implementation; CPU only, no scene."""
import argparse,json,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from expert100.common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--exclusions',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    for folder in ['author_fast5','legdiag_random5','success_random5','newexpert_stage7/adapters.py','fullcycle_task_monitor.py']:
        assert not subprocess.check_output(['git','diff',BASE,'--','migration/wc4_act/'+folder],cwd=ROOT,text=True)
    old=ROOT/'.repro/reviews/wc4-expert50-20261029-FINAL';prior=read(old/'manifest.json');index=read(old/'source-index.json');e=old/'evidence';prev=read(e/'manifest.json')
    assert prior['final_commit']==OLD_FREEZE and index['actual']==50
    # Check every old execution-source binding before admitting reuse.
    for n,h in prev['source_sha256'].items():assert sha(ROOT/n)==h,n
    assert shutil.disk_usage(ROOT).free>46*1024**3
    excluded=read(a.exclusions);assert not excluded['parse_errors']
    used=set(excluded['used_seeds']);assert set(x['seed'] for x in index['episodes'])<=used
    assert set(range(62300,62316))<=used,'fixed ACT evaluation exclusions missing'
    run.mkdir(parents=True,exist_ok=False)
    for name in ['attempts','reused','qualification','raw']:(run/name).mkdir()
    for name in ['frozen-physics.yaml','geometry.json']:(run/name).write_bytes((e/name).read_bytes())
    (run/'seed-exclusions.json').write_bytes(a.exclusions.read_bytes())
    files=subprocess.check_output(['git','ls-files','migration/wc4_act','high-level/float_ik','high-level/dp','high-level/data'],cwd=ROOT,text=True).splitlines()
    start=62150
    while start in used:start+=1
    m=dict(start_commit=head(),baseline=BASE,reused_freeze=OLD_FREEZE,previous_evidence=str(e),
        target_successes=100,reused_successes=50,new_target_successes=50,maximum_sim_seconds=120,
        start_new_seed=start,workers=list(range(8)),failure_streak_stop=3,
        yaml=str(run/'frozen-physics.yaml'),yaml_sha256=sha(run/'frozen-physics.yaml'),geometry_sha256=sha(run/'geometry.json'),
        source_sha256={n:sha(ROOT/n) for n in files if (ROOT/n).is_file()},
        r7_physics_after_asset_overrides=prev['r7_physics_after_asset_overrides'],author_argv_template=prev['author_argv_template'],
        seed_exclusions_sha256=sha(run/'seed-exclusions.json'),
        sampling='Sequential previously unused seeds >=62150, author make_env_args(args,0) and original camera jitter; no initial-state filtering or redraw',
        success='same episode actual opening >=80deg AND final legal relock',video=False,replay=False,conversion=False,training=False)
    write(run/'manifest.json',m);reuse=[]
    for x in index['episodes']:
        source=e/x['evidence'];target=run/'reused'/('seed%d'%x['seed']);hardlink_tree(source,target)
        before={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
        after={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()};assert before==after
        q=qualification(target,'reused',run);assert q['success'] and q['raw_sha256']==x['raw_sha256']
        q.update(original_evidence=str(source),original_file_sha256=before,original_execution_commit=x['execution_commit'],accepted_freeze=OLD_FREEZE,original_episode_id=x['episode_id'])
        write(run/'qualification'/('seed%d.json'%q['seed']),q)
        reuse.append(dict(seed=q['seed'],source=str(source),target=str(target),original_episode_id=x['episode_id'],all_original_bytes_identical=True))
        print(json.dumps(dict(event='reused_verified',seed=q['seed'],frames=q['frames'])),flush=True)
    write(run/'reuse-binding.json',reuse)
    print('READY: all50 original records requalified; no scene created',flush=True)
if __name__=='__main__':main()
