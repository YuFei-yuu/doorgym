"""Success-count bounded parallel author collection. No sample filtering."""
import argparse,collections,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from expert50.common import read,write,sha,verify,qualification,raw_path
from expert50.sample import Sampler

def can_launch(successes,inflight,target=45):return successes+inflight<target

def finish_index(run,stop):
    qualified=[read(p) for p in sorted((run/'qualification').glob('seed*.json'))]
    qualified.sort(key=lambda x:x['seed']);assert len({q['seed'] for q in qualified})==len(qualified)
    assert len({q['raw_sha256'] for q in qualified})==len(qualified)
    index=[]
    for i,q in enumerate(qualified):
        assert q['success']
        source=run/q['raw_path'];dest=run/'raw'/('episode_%06d.npz'%i)
        os.link(source,dest);assert sha(dest)==q['raw_sha256']
        item=dict(episode_id=i,seed=q['seed'],env_seed=q['effective_randomization']['env_seed'],
            origin_kind=q['origin_kind'],raw_path=str(dest.relative_to(run)),raw_sha256=q['raw_sha256'],
            source_raw_path=q['raw_path'],evidence=q['evidence'],
            qualification='qualification/seed%d.json'%q['seed'],frames=q['frames'],steps=q['steps'],sim_seconds=q['sim_seconds'],
            max_abs_open_deg=q['max_abs_open_deg'],legal_relock_time_s=q['legal_relock_time_s'],
            execution_commit=read(run/q['evidence']/'command.json')['head'],source_bindings=q['bindings'])
        index.append(item)
    write(run/'source-index.json',dict(target=50,actual=len(index),reused=sum(x['origin_kind']=='reused' for x in index),
        new=sum(x['origin_kind']=='new' for x in index),total_frames=sum(x['frames'] for x in index),episodes=index,
        raw_byte_policy='Raw NPZ internal episode identifiers are unchanged. Unique dataset IDs are external filenames/index only; no source-byte rewrite.'))
    attempts=[json.loads(x) for x in (run/'attempt-ledger.jsonl').read_text().splitlines()] if (run/'attempt-ledger.jsonl').exists() else []
    summary=dict(complete=len(index)==50,stop_reason=stop,new_attempts=len(attempts),
        new_successes=sum(x['status']=='success' for x in attempts),
        task_failures=sum(x['status']=='task_failure' for x in attempts),
        anomalies=sum(x['status'] not in ['success','task_failure'] for x in attempts),
        failure_reasons=dict(collections.Counter(x['reason'] for x in attempts if x['status']!='success')),
        dataset_episodes=len(index),reused=5,total_frames=sum(x['frames'] for x in index),
        total_steps=sum(x['steps'] for x in index),free_bytes_end=shutil.disk_usage(ROOT).free,
        videos=0,replays=0,conversion=False,training=False)
    write(run/'summary.json',summary);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    manifest=verify(run);assert len(list((run/'qualification').glob('*.json')))==5
    write(run/'collection-start.json',dict(at=time.time(),pid=os.getpid(),gpus=manifest['workers'],free_bytes=shutil.disk_usage(ROOT).free))
    sampler=Sampler(run);successes=0;inflight={};next_seed=manifest['start_new_seed'];stop=None
    streak_reason=None;streak=0;raw_hashes={read(p)['raw_sha256'] for p in (run/'qualification').glob('*.json')}
    def event(x):
        x=dict(at=time.time(),**x)
        with (run/'collection-events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
        print(json.dumps(x),flush=True)
    while inflight or (successes<45 and stop is None):
        free=shutil.disk_usage(ROOT).free
        if stop is None and free<40*1024**3+(len(inflight)+1)*3*1024**3:
            stop='capacity_reserve_insufficient';event(dict(event='stop_launching',reason=stop,free_bytes=free))
        for gpu in manifest['workers']:
            if stop is not None or not can_launch(successes,len(inflight)):break
            if gpu in inflight:continue
            target=sampler.prepare(next_seed);seed=next_seed;next_seed+=1
            log=(target/'launcher.log').open('x')
            proc=subprocess.Popen(['bash',str(ROOT/'migration/wc4_act/expert50/launch_one.sh'),str(run),str(seed),str(gpu)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            inflight[gpu]=dict(proc=proc,seed=seed,target=target,log=log)
            event(dict(event='launched',seed=seed,gpu=gpu,new_successes=successes))
        done=[]
        for gpu,job in inflight.items():
            rc=job['proc'].poll()
            if rc is None:continue
            done.append(gpu);job['log'].close();seed=job['seed'];out=job['target']/'attempt_000'
            result=read(out/'result.json') if (out/'result.json').exists() else {}
            entry=dict(seed=seed,gpu=gpu,returncode=rc,evidence=str(out.relative_to(run)),reason=result.get('reason','entry_failed_before_result'))
            if rc!=0 or result.get('error'):
                entry['status']='execution_anomaly';stop=stop or 'execution_anomaly'
            else:
                try:
                    q=qualification(out,'new',run)
                    q['trial_manifest_sha256']=sha(job['target']/'manifest.json')
                    write(job['target']/'audit.json',q)
                    if q['success']:
                        assert q['raw_sha256'] not in raw_hashes,'Duplicate source raw bytes'
                        raw_hashes.add(q['raw_sha256']);successes+=1
                        write(run/'qualification'/('seed%d.json'%seed),q);entry['status']='success';entry['frames']=q['frames']
                    else:entry['status']='task_failure'
                except BaseException:
                    entry['status']='data_contract_anomaly';stop=stop or 'data_contract_anomaly'
                    (job['target']/'audit-error.txt').write_text(traceback.format_exc())
            if entry['status']=='success':streak=0;streak_reason=None
            elif entry['status']=='task_failure':
                signature=entry['reason'].split(':',1)[0]
                streak=streak+1 if signature==streak_reason else 1;streak_reason=signature
                if streak>=3:stop=stop or 'three_consecutive_same_task_failures:'+signature
            with (run/'attempt-ledger.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            event(dict(event='attempt_done',new_successes=successes,**entry))
        for gpu in done:del inflight[gpu]
        progress=dict(new_successes=successes,total_qualified=successes+5,inflight=[dict(gpu=g,seed=j['seed']) for g,j in inflight.items()],next_seed=next_seed,stop_reason=stop,free_bytes=shutil.disk_usage(ROOT).free)
        temp=run/'progress.tmp';temp.write_text(json.dumps(progress,indent=2)+'\n');temp.replace(run/'progress.json')
        if inflight:time.sleep(1)
    summary=finish_index(run,stop or 'target_50_reached')
    event(dict(event='collection_finished',**summary))
    if not summary['complete']:sys.exit(2)
if __name__=='__main__':main()
