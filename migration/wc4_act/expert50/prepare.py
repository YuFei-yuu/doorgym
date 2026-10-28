"""Bind baseline and verify byte-identical reused records, without new scenes."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from expert50.common import BASE,sha,read,write,head,hardlink_tree,qualification

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','diff',BASE,'--','migration/wc4_act/author_fast5','migration/wc4_act/legdiag_random5','migration/wc4_act/success_random5','migration/wc4_act/fullcycle_task_monitor.py'],cwd=ROOT,text=True)
    old=ROOT/'.repro/reviews/wc4-legdiag-random5-20261029-FINAL/evidence';prior=read(old/'manifest.json')
    run.mkdir(parents=True,exist_ok=False)
    for name in ['attempts','reused','qualification','raw']: (run/name).mkdir()
    yaml=run/'frozen-physics.yaml';yaml.write_bytes((old/'frozen-physics.yaml').read_bytes())
    (run/'geometry.json').write_bytes((old/'geometry.json').read_bytes())
    files=subprocess.check_output(['git','ls-files','migration/wc4_act','high-level/float_ik','high-level/dp','high-level/data'],cwd=ROOT,text=True).splitlines()
    m=dict(start_commit=head(),baseline=BASE,target_successes=50,reused_successes=5,new_target_successes=45,
        maximum_sim_seconds=120,start_new_seed=62105,workers=[1,2,5,6],failure_streak_stop=3,
        yaml=str(yaml),yaml_sha256=sha(yaml),geometry_sha256=sha(run/'geometry.json'),
        source_sha256={n:sha(ROOT/n) for n in files if (ROOT/n).is_file()},
        r7_physics_after_asset_overrides=prior['r7_physics_after_asset_overrides'],
        author_argv_template=prior['trials'][0]['argv'],previous_evidence=str(old),
        sampling='Sequential seeds from 62105, author make_env_args(args,0), original frozen pose distribution and camera jitter; no filtering or redraw',
        success='actual opening >=80deg AND final legal relock; unchanged',video=False,replay=False,conversion=False,training=False)
    write(run/'manifest.json',m)
    reused=[]
    for i,seed in enumerate(range(62100,62105)):
        source=old/('trial_%02d'%i)/'attempt_000';target=run/'reused'/('seed%d'%seed)
        hardlink_tree(source,target)
        before={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
        after={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}
        assert before==after
        q=qualification(target,'reused',run);assert q['success'] and q['seed']==seed
        q.update(original_evidence=str(source),original_file_sha256=before,original_execution_commit=read(source/'command.json')['head'],accepted_freeze=BASE)
        write(run/'qualification'/('seed%d.json'%seed),q);reused.append(dict(seed=seed,source=str(source),evidence=str(target),all_original_bytes_identical=True))
        print(json.dumps(dict(event='reused_verified',seed=seed,frames=q['frames'])),flush=True)
    write(run/'reuse-binding.json',reused)
    print('Ready: five immutable reused episodes qualified; no new physical scene created.',flush=True)
if __name__=='__main__':main()
