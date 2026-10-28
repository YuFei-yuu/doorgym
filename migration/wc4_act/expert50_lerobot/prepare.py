"""Bind the frozen b2dfd5e source index; no old cache or data is removed."""
from common import *
import csv,shutil,subprocess,time

def main():
    base='b2dfd5ec685865e3b5d1676bc76fc1938237301d'
    source=ROOT/'.repro/reviews/wc4-expert50-20261029-FINAL'
    index=read(source/'source-index.json');origin=source/index['paths_relative_to'];episodes=index['episodes']
    assert len(episodes)==50 and index['total_frames']==77356
    assert [e['episode_id'] for e in episodes]==list(range(50))
    assert sum(e['origin_kind']=='reused' for e in episodes)==5 and sum(e['origin_kind']=='new' for e in episodes)==45
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','diff',base,'--','high-level','migration/wc4_act/newexpert_stage8'],cwd=ROOT,text=True)
    RUN.mkdir(exist_ok=False)
    procs=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    write(RUN/'process-snapshot.json',dict(at=time.time(),processes=procs))
    for line in procs.splitlines():
        if str(ROOT) in line and any(x in line for x in ['lerobot_train','full_loader.py','convert_door_raw_to_lerobot.py']):
            raise RuntimeError('Active conflicting project data process: '+line)
    free=shutil.disk_usage(ROOT).free;assert free>=72*2**30
    write(RUN/'capacity-preflight.json',dict(free_bytes=free,incremental_peak_estimate_bytes=32*2**30,reserve_bytes=40*2**30,passed=True,
        cleanup='No historical data or cache deleted. New HF cache/tmp isolated inside this run.'))
    raw=RUN/'raw-links';raw.mkdir();mapped=[];offset=0
    for i,item in enumerate(episodes):
        src=origin/item['raw_path'];assert src.is_file() and sha(src)==item['raw_sha256']
        alias=raw/('episode_%06d.npz'%i);alias.symlink_to(src)
        mapped.append(dict(**item,source_id='seed%d'%item['seed'],candidate_npz=str(alias),
            source_npz=str(src),source_npz_sha256=item['raw_sha256'],episode_index=i,
            dataset_from_index=offset,dataset_to_index=offset+item['frames'],full_episode=True,frame_slice=None,
            source_clock=str(origin/item['evidence']/'recording-clock.jsonl'),source_clock_sha256=item['source_bindings']['recording-clock.jsonl']))
        offset+=item['frames']
    assert offset==77356
    write(RUN/'candidate-index.json',dict(episodes=mapped,frames=offset,input_index=str(source/'source-index.json'),input_index_sha256=sha(source/'source-index.json'),input_manifest_sha256=sha(source/'manifest.json'),accepted_input_commit=base))
    with (RUN/'SOURCE_TABLE.csv').open('x') as f:
        keys=['episode_index','origin_kind','seed','frames','dataset_from_index','dataset_to_index','source_npz','source_npz_sha256','source_clock','source_clock_sha256']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:e[k] for k in keys} for e in mapped)
    files=['high-level/dp/convert_door_raw_to_lerobot.py','high-level/dp/door_dp_common.py',
        'high-level/lerobot/src/lerobot/datasets/lerobot_dataset.py','high-level/lerobot/src/lerobot/datasets/video_utils.py',
        'high-level/lerobot/src/lerobot/datasets/utils.py','high-level/lerobot/src/lerobot/datasets/compute_stats.py',
        'high-level/lerobot/src/lerobot/datasets/factory.py','high-level/lerobot/src/lerobot/policies/act/configuration_act.py',
        'high-level/lerobot/src/lerobot/policies/act/processor_act.py','high-level/lerobot/src/lerobot/processor/normalize_processor.py']
    files+=[str(p.relative_to(ROOT)) for p in (ROOT/'migration/wc4_act/expert50_lerobot').glob('*.py')]
    write(RUN/'source-binding.json',dict(actual_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={p:sha(ROOT/p) for p in files},repo_id=REPO,dataset_path=str(DATA),raw_root=str(raw),
        input_index_sha256=sha(source/'source-index.json'),training=False,simulation=False,policy_inference=False,device='cpu'))
    print('Prepared 50 original episodes / 77356 frames in exact source-index order; no raw rewrite',flush=True)
if __name__=='__main__':main()
