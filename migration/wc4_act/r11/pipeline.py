"""New sealed dataset, unchanged author conversion and fixed TorchCodec reader."""
import json,pathlib,os,subprocess,time,hashlib
from common import ROOT,RUN,REPO,read,write


def main():
    seal=read(RUN/'candidate-seal.json');assert seal['episodes']>0
    act=ROOT/'.repro/envs/door-act/bin/python'
    gym=pathlib.Path('/data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONNOUSERSITE']='1'
    clean=env.copy();clean.pop('LD_LIBRARY_PATH',None);clean.pop('PYTHONPATH',None)
    tools=ROOT/'migration/wc4_act/r11'
    stages=[('convert',[str(act),'-B',str(ROOT/'high-level/dp/convert_door_raw_to_lerobot.py'),
        '--raw_root',str(RUN/'candidates'),'--root',str(RUN/'lerobot'),'--repo_id',REPO,
        '--state_action_mode','joint_state9','--depth_only','--fps','25','--image_storage','video','--video_codec','h264','--num_workers','4'],env),
        ('dataset-check',[str(act),'-B',str(tools/'check_dataset.py')],env),
        ('full-loader',[str(act),'-B',str(tools/'full_loader.py')],env),
        ('ambiguity-by-episode',[str(gym),'-B',str(tools/'check_alias_episode.py')],clean),
        ('ambiguity-decoded',[str(act),'-B',str(tools/'check_alias_episode_decoded.py')],env)]
    receipts=[]
    for name,command,child in stages:
        assert hashlib.sha256((RUN/'candidate-index.json').read_bytes()).hexdigest()==seal['sha256']
        start=time.time()
        write(RUN/(name+'-command.json'),dict(argv=command,cwd=str(ROOT),CUDA_VISIBLE_DEVICES='',
            source_sha256=hashlib.sha256(pathlib.Path(command[2]).read_bytes()).hexdigest(),
            start_time=start,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        with (RUN/(name+'.log')).open('x') as f:
            done=subprocess.run(command,cwd=ROOT,env=child,stdout=f,stderr=subprocess.STDOUT)
        receipt=dict(stage=name,returncode=done.returncode,elapsed_s=time.time()-start)
        write(RUN/(name+'-exit.json'),receipt);receipts.append(receipt)
        print(receipt,flush=True)
        if done.returncode:
            write(RUN/'DATA_AUDIT_STOP.json',receipt);raise SystemExit(done.returncode)
    write(RUN/'pipeline-result.json',dict(passed=True,stages=receipts,collection=0,training=0,policy_inference=0))


if __name__=='__main__':main()
