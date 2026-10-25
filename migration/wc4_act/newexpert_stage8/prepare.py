"""Freeze the accepted sources, bounded cache cleanup and conversion inputs."""
from common import *
import shutil,subprocess,time

def main():
    RUN.mkdir(exist_ok=False)
    old=ROOT/'.repro/runs/wc4-newexpert-stage7-20261025'
    candidate=read(old/'candidate-index.json'); decisions=read(old/'source-review-decisions.json')
    assert len(candidate['episodes'])==len(decisions['sources'])==50
    assert sum(e['frames'] for e in candidate['episodes'])==121305
    assert sha(old/'unified-source-index.json')==candidate['source_index_sha256']
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    procs=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    write(RUN/'process-snapshot.json',{'time':time.time(),'ps':procs})
    # Only rebuildable HF Arrow copies, never datasets, weights, raw records or historical evidence.
    caches=sorted((ROOT/'.repro/cache/act/huggingface/datasets/parquet').rglob('*.arrow'))
    entries=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in caches]
    write(RUN/'cleanup-plan.json',dict(files=entries,reason='Disposable HF Arrow cache of retained Parquet datasets; no active project reader or converter. All raw and historical datasets/evidence protected.'))
    # An existing project training/reader process would make cache deletion unsafe.
    for line in procs.splitlines():
        if str(ROOT) in line and any(x in line for x in ['lerobot_train','full_loader.py','check_dataset.py','convert_door_raw_to_lerobot.py']):
            raise RuntimeError('Active project data process: '+line)
    before=shutil.disk_usage(ROOT).free
    for p in caches:p.unlink()
    after=shutil.disk_usage(ROOT).free
    write(RUN/'cleanup-receipt.json',dict(files=entries,logical_bytes_removed=sum(x['bytes'] for x in entries),free_before=before,free_after=after))
    estimate=dict(output_videos_and_parquet=16*2**30,episode_temporary_images_and_encoding=8*2**30,cache_and_audit=8*2**30,required_remaining=40*2**30)
    write(RUN/'capacity-preflight.json',dict(free_bytes=after,estimate_bytes=estimate,passed=after>=sum(estimate.values()),method='Conservative 32GiB incremental peak; historical same-code H264 dataset 331MiB, two per-episode PNG trees cleared after encoding; no raw duplication.'))
    assert after>=sum(estimate.values()),'Capacity must preserve 40GiB reserve'
    raw=RUN/'raw-links';raw.mkdir()
    mapped=[];start=0
    for i,(item,decision) in enumerate(zip(candidate['episodes'],decisions['sources'])):
        assert i==item['source_index']==decision['source_index']
        assert item['source_id']==decision['source_id'] and decision['admitted'] and decision['complete_task_qualified']
        src=pathlib.Path(item['raw_npz']['path']);h=sha(src)
        assert h==item['raw_npz']['sha256']==decision['raw_npz_sha256']
        assert src.stat().st_size==item['raw_npz']['bytes'] and item['full_episode'] and item['frame_slice'] is None
        alias=raw/('episode_%06d.npz'%i);alias.symlink_to(src)
        mapped.append(dict(**item,candidate_npz=str(alias),episode_index=i,dataset_from_index=start,dataset_to_index=start+item['frames']))
        start+=item['frames']
    write(RUN/'candidate-index.json',dict(episodes=mapped,input_candidate_sha256=sha(old/'candidate-index.json'),input_review_sha256=sha(old/'source-review-decisions.json'),input_source_index_sha256=sha(old/'unified-source-index.json'),frames=start))
    files=['high-level/dp/convert_door_raw_to_lerobot.py','high-level/dp/door_dp_common.py','high-level/lerobot/src/lerobot/datasets/lerobot_dataset.py','high-level/lerobot/src/lerobot/datasets/video_utils.py','high-level/lerobot/src/lerobot/datasets/compute_stats.py','high-level/lerobot/src/lerobot/datasets/factory.py']
    files+= [str(p.relative_to(ROOT)) for p in (ROOT/'migration/wc4_act/newexpert_stage8').glob('*.py')]
    write(RUN/'source-binding.json',dict(actual_commit=head,source_sha256={p:sha(ROOT/p) for p in files},repo_id=REPO,dataset_path=str(DATA),raw_root=str(raw),training=False,simulation=False,policy_inference=False,device='cpu'))
    print('Prepared',len(mapped),start,'free GiB',after/2**30,flush=True)

if __name__=='__main__':main()
