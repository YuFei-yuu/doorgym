"""Bind ONLY the final 011870a root index. Preserve all100 original inputs."""
from common import *
import csv,subprocess,shutil,time

def main():
    base='011870a4c91358f468e79bbcedfdde298fbff501'
    source=ROOT/'.repro/reviews/wc4-expert100-20261031-FINAL'
    assert sha(source/'SHA256SUMS')=='9d9e664a54e3b8ec8d4f827b298e00bb6af1ec248988f939313053741d366013'
    expected={n:h for h,n in (s.split('  ',1) for s in (source/'SHA256SUMS').read_text().splitlines())}
    for name in ['manifest.json','source-index.json']:assert sha(source/name)==expected[name]
    final=read(source/'manifest.json');assert final['final_commit']==base
    index=read(source/'source-index.json');origin=source/index['paths_relative_to'];episodes=index['episodes']
    assert len(episodes)==100 and index['total_frames']==154667
    assert [e['episode_id'] for e in episodes]==list(range(100))
    assert sum(e['origin_kind']=='reused' for e in episodes)==50 and sum(e['origin_kind']=='new' for e in episodes)==50
    assert len({e['seed'] for e in episodes})==len({e['raw_sha256'] for e in episodes})==100
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','diff','197a169','--','high-level','migration/wc4_act/newexpert_stage8'],cwd=ROOT,text=True)
    RUN.mkdir(exist_ok=False)
    procs=subprocess.check_output(['ps','-eo','pid,args'],text=True);write(RUN/'process-snapshot.json',dict(at=time.time(),processes=procs))
    for line in procs.splitlines():
        if str(ROOT) in line and any(k in line for k in ['lerobot_train','full_loader.py','convert_door_raw_to_lerobot.py']):raise RuntimeError('Conflicting data process: '+line)
    free=shutil.disk_usage(ROOT).free;assert free>=104*2**30
    write(RUN/'capacity-preflight.json',dict(free_bytes=free,incremental_peak_estimate_bytes=64*2**30,reserve_bytes=40*2**30,passed=True,cleanup='No historical data/cache removed; independent new run tmp/HF cache.'))
    raw=RUN/'raw-links';raw.mkdir();mapped=[];offset=0
    for i,item in enumerate(episodes):
        assert item['raw_path'].startswith('raw/') and 'phases/' not in item['raw_path']
        src=origin/item['raw_path'];assert src.is_file() and sha(src)==item['raw_sha256']
        clock=origin/item['evidence']/'recording-clock.jsonl';assert sha(clock)==item['source_bindings']['recording-clock.jsonl']
        alias=raw/('episode_%06d.npz'%i);alias.symlink_to(src)
        mapped.append(dict(**item,source_id='seed%d'%item['seed'],candidate_npz=str(alias),source_npz=str(src),source_npz_sha256=item['raw_sha256'],episode_index=i,
          dataset_from_index=offset,dataset_to_index=offset+item['frames'],full_episode=True,frame_slice=None,source_clock=str(clock),source_clock_sha256=item['source_bindings']['recording-clock.jsonl']))
        offset+=item['frames'];print('bound source',i,item['seed'],item['execution_commit'],item['frames'],flush=True)
    assert offset==154667
    write(RUN/'candidate-index.json',dict(episodes=mapped,frames=offset,input_index=str(source/'source-index.json'),input_index_sha256=sha(source/'source-index.json'),input_manifest_sha256=sha(source/'manifest.json'),accepted_input_commit=base,partial_phase_index_used=False))
    with (RUN/'SOURCE_TABLE.csv').open('x') as f:
        keys=['episode_index','episode_id','origin_kind','seed','env_seed','execution_commit','frames','dataset_from_index','dataset_to_index','source_npz','source_npz_sha256','source_clock','source_clock_sha256','evidence']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:e[k] for k in keys} for e in mapped)
    provenance=RUN/'input-provenance';provenance.mkdir();receipts=[]
    names=['manifest.json','source-index.json','SHA256SUMS','data-acceptance.json','summary.json','RESULT.txt','evidence/audit-recovery.json','evidence/manifest.json','evidence/attempt-ledger.jsonl','evidence/frozen-physics.yaml','evidence/geometry.json']
    for rel in names:
        p=source/rel
        if rel!='SHA256SUMS':assert sha(p)==expected[rel]
        q=provenance/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());receipts.append(dict(source=str(p),local=str(q.relative_to(RUN)),sha256=sha(p)))
    supplement=source.with_name('wc4-expert100-20261031-resource-review')
    for rel in ['review.json','review.py','SHA256SUMS']:
        p=supplement/rel;q=provenance/'resource-review'/rel;q.parent.mkdir(exist_ok=True);q.write_bytes(p.read_bytes());receipts.append(dict(source=str(p),local=str(q.relative_to(RUN)),sha256=sha(p)))
    assert read(supplement/'review.json')['parent_sha256sums_sha256']==sha(source/'SHA256SUMS')
    for row in (supplement/'SHA256SUMS').read_text().splitlines():h,n=row.split('  ',1);assert sha(supplement/n)==h
    write(RUN/'input-provenance-receipts.json',receipts)
    files=['high-level/dp/convert_door_raw_to_lerobot.py','high-level/dp/door_dp_common.py',
      'high-level/lerobot/src/lerobot/datasets/lerobot_dataset.py','high-level/lerobot/src/lerobot/datasets/video_utils.py',
      'high-level/lerobot/src/lerobot/datasets/utils.py','high-level/lerobot/src/lerobot/datasets/compute_stats.py',
      'high-level/lerobot/src/lerobot/datasets/factory.py','high-level/lerobot/src/lerobot/policies/act/configuration_act.py',
      'high-level/lerobot/src/lerobot/policies/act/processor_act.py','high-level/lerobot/src/lerobot/processor/normalize_processor.py']
    files+=[str(p.relative_to(ROOT)) for p in (ROOT/'migration/wc4_act/expert100_lerobot').glob('*.py')]
    write(RUN/'source-binding.json',dict(actual_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256={p:sha(ROOT/p) for p in files},repo_id=REPO,dataset_path=str(DATA),raw_root=str(raw),
      input_index_sha256=sha(source/'source-index.json'),original_input_commit=base,training=False,simulation=False,policy_inference=False,device='cpu'))
    print('Prepared100 sources /154667 frames, final index only; all original versions and sampling provenance retained',flush=True)
if __name__=='__main__':main()
