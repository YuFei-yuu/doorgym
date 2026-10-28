"""Bounded CPU conversion and checks; first failing command freezes this run."""
from common import *
import subprocess,shutil,time,sys,traceback

def main():
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:8])
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='2'
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    py=str(ROOT/'.repro/envs/door-act/bin/python');directory=ROOT/'migration/wc4_act/expert50_lerobot'
    subprocess.run([py,'-B',str(directory/'prepare.py')],cwd=ROOT,env=env,check=True)
    for d in ['tmp','cache','commands']:(RUN/d).mkdir()
    env.update(TMPDIR=str(RUN/'tmp'),HF_DATASETS_CACHE=str(RUN/'cache'),HF_HUB_CACHE=str(RUN/'cache/hub'))
    steps=[('convert',[py,'-B','high-level/dp/convert_door_raw_to_lerobot.py','--raw_root',str(RUN/'raw-links'),'--root',str(RUN/'lerobot'),'--repo_id',REPO,'--state_action_mode','joint_state9','--depth_only','--fps','25','--image_storage','video','--video_codec','h264','--num_workers','4'])]
    steps += [(n,[py,'-B',str(directory/(n+'.py'))]) for n in ['precision_guard','check_dataset','check_stats','check_normalization','full_loader']]
    binding=read(RUN/'source-binding.json');ledger=[]
    for n,cmd in steps:
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==binding['actual_commit']
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
        for p,h in binding['source_sha256'].items():assert sha(ROOT/p)==h
        assert shutil.disk_usage(ROOT).free>=40*2**30,'Capacity reserve violated before command'
        receipt=dict(name=n,argv=cmd,cwd=str(ROOT),environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONNOUSERSITE','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','HF_DATASETS_CACHE','TMPDIR','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},cpu_affinity=sorted(os.sched_getaffinity(0)),start_time=time.time())
        write(RUN/'commands'/(n+'.json'),receipt)
        with (RUN/(n+'.log')).open('x') as out:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            receipt['pid']=proc.pid;write(RUN/'active-command.json',receipt)
            while proc.poll() is None:
                usage=shutil.disk_usage(ROOT)
                with (RUN/'resource-monitor.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),stage=n,pid=proc.pid,free_bytes=usage.free))+'\n')
                if usage.free<40*2**30:
                    import signal
                    os.killpg(proc.pid,signal.SIGTERM);proc.wait()
                    raise RuntimeError('Stopped command to preserve 40GiB reserve')
                time.sleep(10)
        receipt.update(exit_code=proc.returncode,end_time=time.time());ledger.append(receipt);write(RUN/'command-ledger.json',ledger)
        assert proc.returncode==0,(n,proc.returncode)
    write(RUN/'CHECKS-COMPLETE.json',dict(passed=True,frames=77356,episodes=50,commands=len(ledger),training=False,simulation=False,policy_inference=False))
    print('Expert50 conversion checks complete; no training started.',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if RUN.exists():write(RUN/'STOP.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()))
        raise
