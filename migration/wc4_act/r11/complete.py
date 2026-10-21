"""Finish only authorized read-only checks after the current data pipeline."""
import os,pathlib,subprocess,time,hashlib,json
from common import ROOT,RUN,read,write


def main():
    while not (RUN/'pipeline-result.json').exists():
        if (RUN/'DATA_AUDIT_STOP.json').exists():raise SystemExit('Pipeline stopped; no automatic retry')
        time.sleep(30)
    assert read(RUN/'pipeline-result.json')['passed']
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    python=ROOT/'.repro/envs/door-act/bin/python';tools=ROOT/'migration/wc4_act/r11'
    stages=[('statistics-check',[str(python),'-B',str(tools/'check_stats.py')],ROOT)]
    for label,source in [('r8','wc4-fulltask-pilot-r8-20261021'),('r9','wc4-fulltask-expand-r9-20261021'),('r10','wc4-fulltask-reader-r10-20261021')]:
        stages.append((label+'-after-hashcheck',['sha256sum','-c','SHA256SUMS'],ROOT/'.repro/runs'/source))
    stages.append(('finalize',[str(python),'-B',str(tools/'finalize.py')],ROOT))
    for name,command,cwd in stages:
        start=time.time()
        write(RUN/(name+'-command.json'),dict(argv=command,cwd=str(cwd),CUDA_VISIBLE_DEVICES='',start_time=start,
            source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        with (RUN/(name+'.log')).open('x') as f:
            done=subprocess.run(command,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
        write(RUN/(name+'-exit.json'),dict(returncode=done.returncode,elapsed_s=time.time()-start))
        if done.returncode:raise SystemExit(done.returncode)
    print('All authorized offline data checks finished; no training follows',flush=True)


if __name__=='__main__':main()
