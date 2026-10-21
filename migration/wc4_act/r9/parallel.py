"""Bounded independent-scene scheduling under exclusive idle-GPU leases.

No changes to expert, dt, physics, cameras, or raw recorder. CPU evidence tools
use the existing ACT environment (the bare Gym conda Python lacks PyAV).
"""
import argparse, concurrent.futures, fcntl, hashlib, json, os, signal, subprocess, sys, threading, time
from pathlib import Path
from collect import ROOT, read, write, verify, sha

RUN=ROOT/'.repro/runs/wc4-fulltask-expand-r9-20261021'
ACTIVE={}; ACTIVE_LOCK=threading.Lock(); AUDIT_LOCK=threading.Lock()


def command(stage,index,gpu,group):
    if stage in ['preflight','attempt']:
        script='export VIDEO2DOOR_GPU="$1"; source migration/wc4_act/activate.sh gym; python -B migration/wc4_act/r9/collect.py --run "$2" --mode "$3" --index "$4"'
        args=['bash','-c',script,'_',str(gpu),str(RUN),stage,str(index)]
    else:
        target={'validate':'validate.py','review':'review.py','screen':'audit.py'}[stage]
        flags=[str(index)] if stage=='review' else ['--run',str(RUN),'--index',str(index)]
        if stage=='screen':flags+=['--mode','screen']
        script='source migration/wc4_act/activate.sh act; exec python -B "$@"'
        args=['bash','-c',script,'_',str(ROOT/'migration/wc4_act/r9'/target),*flags]
    write(group/('%02d-%s-command.json'%(index,stage)),dict(argv=args,wall_time=time.time(),gpu=gpu,stage=stage,index=index))
    with (group/('%02d-%s.log'%(index,stage))).open('x') as out:
        process=subprocess.Popen(args,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        with ACTIVE_LOCK: ACTIVE[process.pid]=process
        code=process.wait()
        with ACTIVE_LOCK: ACTIVE.pop(process.pid,None)
    write(group/('%02d-%s-exit.json'%(index,stage)),dict(returncode=code,wall_time=time.time()))
    if code:
        if not (RUN/'SYSTEMIC_STOP.json').exists():
            write(RUN/'SYSTEMIC_STOP.json',dict(index=index,stage=stage,reason='Process/record audit failed; stop group and inspect',returncode=code))
        raise RuntimeError('%s %d exited %d'%(stage,index,code))


def one(index,gpu,group):
    for stage in ['preflight','attempt','validate','review','screen']:
        if (RUN/'SYSTEMIC_STOP.json').exists():return
        if stage=='screen':
            with AUDIT_LOCK: command(stage,index,gpu,group)
        else:command(stage,index,gpu,group)


def main():
    p=argparse.ArgumentParser();p.add_argument('--group',type=int,required=True);p.add_argument('--indices',type=int,nargs='+',required=True);p.add_argument('--gpus',type=int,nargs='+',default=[1,2]);a=p.parse_args()
    verify(RUN);ids=a.indices
    assert len(set(ids))==len(ids) and 1<=len(ids)<=8 and all(0<=i<64 for i in ids)
    assert len(set(a.gpus))==len(a.gpus)
    assert len(read(RUN/'candidate-index.json')['episodes'])+len(ids)<=50, 'Never launch beyond remaining candidate capacity'
    assert all(not (RUN/'claims'/('%03d.json'%i)).exists() for i in ids)
    assert not (RUN/'SYSTEMIC_STOP.json').exists()
    group=RUN/'groups'/('%03d'%a.group);group.mkdir(parents=True,exist_ok=False)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(group/'binding.json',dict(indices=ids,gpus=a.gpus,head=head,scheduler_sha256=sha(Path(__file__)),num_envs_per_process=1,expert_source_binding=str(RUN/'consistency.json'),maximum_concurrent_scenes=len(ids),wall_time=time.time()))
    leases=[]
    try:
        for gpu in a.gpus:
            uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            fd=open('/data/zyf/video2door/.repro/locks/'+uuid+'.lock','w');fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(fd)
            for k in range(3):
                snap=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True)
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv'],text=True)
                (group/('gpu%d-check%d.txt'%(gpu,k))).write_text(snap+'\n'+apps)
                assert uuid not in apps, 'GPU already occupied'
                mem,util=map(float,subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(','))
                assert mem<1024 and util<10
                time.sleep(1)
        (group/'cpu-memory-before.txt').write_text(subprocess.check_output(['free','-h'],text=True))
        errors=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ids)) as pool:
            futures=[pool.submit(one,i,a.gpus[n%len(a.gpus)],group) for n,i in enumerate(ids)]
            interrupted=False
            while not all(f.done() for f in futures):
                if (RUN/'SYSTEMIC_STOP.json').exists() and not interrupted:
                    with ACTIVE_LOCK:
                        for process in ACTIVE.values():
                            try:os.killpg(process.pid,signal.SIGINT)
                            except ProcessLookupError:pass
                    interrupted=True
                time.sleep(.5)
            for f in futures:
                try:f.result()
                except BaseException as e:errors.append(repr(e))
        # Rebuild the canonical ledger once after concurrent screens complete.
        from audit import ledger
        ledger(RUN)
        write(group/'completion.json',dict(errors=errors,wall_time=time.time(),systemic_stop=(RUN/'SYSTEMIC_STOP.json').exists(),progress=read(RUN/'progress.json')))
        print(json.dumps(read(group/'completion.json')),flush=True)
        if errors or (RUN/'SYSTEMIC_STOP.json').exists():raise SystemExit(5)
    finally:
        for fd in leases:fd.close()


if __name__=='__main__':main()
