"""Freeze stable evidence and package; no numerical dataset changes."""
from common import *
import subprocess,sys,shutil,time

def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
    summary=read(RUN/'acceptance-summary.json')
    assert summary['status']=='FROZEN_DATA_ACCEPTANCE_COMPLETE'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    main_head=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()
    assert main_head=='87c5a8665f78b0627c95ea9be6a0e756fc359b11'
    # Driver and its worker subprocesses have exited before this freeze.
    ledger=read(RUN/'command-ledger.json')
    for e in ledger:
        assert e['exit_code']==0
        assert not pathlib.Path('/proc/%d'%e['pid']).exists()
    snapshot=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    for line in snapshot.splitlines():
        if str(ROOT/'migration/wc4_act/newexpert_stage8/run.py') in line or 'python -B migration/wc4_act/newexpert_stage8/run.py' in line:
            raise RuntimeError('Driver still active: '+line)
    write(RUN/'freeze-command.json',dict(argv=[sys.executable,'-B',str(pathlib.Path(__file__).resolve())],cwd=str(ROOT),actual_commit=head,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),time=time.time()))
    write(RUN/'freeze-receipt.json',dict(status='FROZEN_DATA_ACCEPTANCE_COMPLETE',head=head,actual_conversion_and_checks_commit=summary['actual_execution_commit'],actual_seal_commit=summary['actual_seal_commit'],branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),main=main_head,worktree='',episodes=50,frames=121305,all_checks_passed=True,training_updates=0,simulation_episodes=0,policy_inference=0,gpu_locks_held=False,run_monitor_active=False,dataset_path=str(DATA),repo_id=REPO,dataset_index_sha256=sha(RUN/'dataset-SHA256.json'),statistics_sha256=sha(DATA/'meta/stats.json'),free_bytes=shutil.disk_usage(ROOT).free,time=time.time()))
    excluded={'SHA256.json','review-package.tar.gz','review-package-receipt.json'}
    files={str(p.relative_to(RUN)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(RUN.rglob('*')) if p.is_file() and not p.is_symlink() and p.relative_to(RUN).parts[0] not in ['cache','tmp'] and p.name not in excluded}
    write(RUN/'SHA256.json',dict(files=files,excluded=['self','cache','tmp','raw-links (input source hashes in candidate-index)','review package and its receipt (separate SHA)']))
    names=[n for n in files if not n.startswith('lerobot/') and n!='lerobot-dataset.tar']+['SHA256.json']
    names.sort()
    cmd=['tar','-czf',str(RUN/'review-package.tar.gz'),'-C',str(RUN),*names]
    subprocess.run(cmd,check=True)
    write(RUN/'review-package-receipt.json',dict(argv=cmd,cwd=str(ROOT),exit_code=0,sha256=sha(RUN/'review-package.tar.gz'),bytes=(RUN/'review-package.tar.gz').stat().st_size,members=names,actual_commit=head))
    print('Frozen',head,'audit files',len(names),'dataset files',len(read(RUN/'dataset-SHA256.json')['files']),flush=True)

if __name__=='__main__':main()
