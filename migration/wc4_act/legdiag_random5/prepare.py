"""Inherit all five frozen initial states without new draws or filtering."""
import argparse,copy,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from success_random5.common import sha,write,read,head
BASE='1505ca0ff32e0a5438ef20f6c91896dc5a673fb6'

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','diff',BASE,'--','migration/wc4_act/author_fast5','migration/wc4_act/success_random5','migration/wc4_act/fullcycle_task_monitor.py'],cwd=ROOT,text=True)
    old=ROOT/'.repro/reviews/wc4-success-random5-20261029-FINAL/evidence'
    prior=read(old/'manifest.json');m=copy.deepcopy(prior)
    run.mkdir(parents=True,exist_ok=False)
    yaml=run/'frozen-physics.yaml';yaml.write_bytes((old/'frozen-physics.yaml').read_bytes())
    (run/'geometry.json').write_bytes((old/'geometry.json').read_bytes())
    for i,t in enumerate(m['trials']):
        t['argv'][t['argv'].index('--a2w_float_ik_config')+1]=str(yaml)
        expected=read(old/('trial_%02d'%i)/'attempt_000/initial-state.json')
        t['expected_actual_initial_state']=expected
        (run/('trial_%02d'%i)).mkdir()
    files=subprocess.check_output(['git','ls-files','migration/wc4_act','high-level/float_ik','high-level/dp','high-level/data'],cwd=ROOT,text=True).splitlines()
    m.update(start_commit=head(),previous_freeze=BASE,previous_run=str(old),
        yaml=str(yaml),yaml_sha256=sha(yaml),geometry_sha256=sha(run/'geometry.json'),
        source_sha256={n:sha(ROOT/n) for n in files if (ROOT/n).is_file()},
        sampling='No new draws. Exact five frozen author trials inherited from prior FINAL; full actual initial state checked before physics.',
        leg_posture_abort=False,representative_state_replay='Only earliest trial with a diagnostic event; one render-only replay, no integration',
        initial_state_draw_calls=0)
    write(run/'manifest.json',m);write(run/'initial-list.json',m['trials'])
    write(run/'inheritance.json',dict(previous_manifest_sha256=sha(old/'manifest.json'),
        previous_initial_list_sha256=sha(old/'initial-list.json'),physics_yaml_identical=True,expert_geometry_identical=True,
        seeds=m['seeds'],sampling_calls=0,scene_count=0,changes=['YAML output path','expected actual initial-state evidence']))
    print('Frozen original five initial states; no new draws or scenes.')
if __name__=='__main__':main()
