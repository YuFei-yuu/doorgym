"""Read-only raw-contract fixture using accepted bytes; never a source episode."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from newexpert_stage7.common import ROOT,read,write,sha
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.output
old=ROOT/'.repro/runs/wc4-newexpert-stage6-20261025';source=old/'trial_00';raw=source/'attempt_000'
r.mkdir(parents=True,exist_ok=False);out=r/'attempt_000';out.mkdir()
for name in ['states.jsonl','contacts.jsonl','recording-clock.jsonl','lock-writes.jsonl','result.json','reference.json',
             'initial-frozen-comparison.json','physical-camera-binding.json']:(out/name).symlink_to(raw/name)
(out/'raw').symlink_to(raw/'raw',target_is_directory=True)
trial=read(raw/'frozen-trial.json');trial['source_index']=-1;trial['fixture_only_not_source_episode']=True
write(out/'frozen-trial.json',trial)
(r/'manifest.json').write_bytes((old/'manifest.json').read_bytes())
(r/'depth-encoding-receipt.json').write_bytes((source/'depth-encoding-receipt.json').read_bytes())
clock=[json.loads(x) for x in (raw/'recording-clock.jsonl').open()]
write(r/'raw-render-receipt.json',{'rendered_steps':[c['source_step'] for c in clock],
    'fixture_only':'Counter derived from accepted source clock for CPU contract test; no new rendering or scene.'})
write(r/'fixture-provenance.json',{'source_dir':str(source),'source_npz_sha256':sha(raw/'raw/episode_000000.npz'),
    'scene_creations':0,'physics_steps':0,'is_source_episode':False,'purpose':'CPU reader/reviewer smoke test only'})
