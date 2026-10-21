"""One sealed candidate conversion and CPU-only data acceptance pipeline."""
import json, os, pathlib, shutil, subprocess, sys, time
from collect import ROOT,read,write,sha,verify
R=ROOT/'.repro/runs/wc4-fulltask-expand-r9-20261021'
ACT=ROOT/'.repro/envs/door-act/bin/python'
GYM=pathlib.Path('/data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python')


def main():
    verify(R)
    progress=read(R/'progress.json');index=read(R/'candidate-index.json')
    assert not (R/'SYSTEMIC_STOP.json').exists() and not progress['pending_visual']
    assert progress['scenes']<=64 and (progress['scenes']==64 or len(index['episodes'])==50), 'Collection must stop at target/budget first'
    seal=dict(candidate_index_sha256=sha(R/'candidate-index.json'),episodes=len(index['episodes']),frames=sum(e['frames'] for e in index['episodes']),
        retained_r8=6,retained_r8_frames=sum(e['frames'] for e in index['episodes'][:6]),new_training=False,policy_inference=False)
    assert seal['retained_r8_frames']==13301
    write(R/'collection-sealed.json',seal)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONNOUSERSITE']='1'
    clean=env.copy();clean.pop('LD_LIBRARY_PATH',None);clean.pop('PYTHONPATH',None)
    converter=ROOT/'high-level/dp/convert_door_raw_to_lerobot.py'
    command=[str(ACT),'-B',str(converter),'--raw_root',str(R/'candidates'),'--root',str(R/'lerobot'),
        '--repo_id','local/wc4_fulltask_expand_r9_20261021','--state_action_mode','joint_state9','--depth_only','--fps','25',
        '--image_storage','video','--video_codec','h264','--num_workers','4']
    write(R/'convert-command.json',dict(argv=command,cwd=str(ROOT),source_sha256=sha(converter),input_binding=seal))
    path=ROOT/'migration/wc4_act/r9'
    stages=[('aggregate-before',[str(GYM),'-B',str(path/'aggregate.py')],clean),('convert',command,env),
        ('dataset-check',[str(ACT),'-B',str(path/'check_dataset.py')],env),
        ('feature-check',[str(ACT),'-B',str(path/'check_features.py')],env),
        ('ambiguity-check',[str(GYM),'-B',str(path/'check_alias.py')],clean),
        ('ambiguity-decoded-check',[str(ACT),'-B',str(path/'check_alias_decoded.py')],env),
        ('figures',[str(GYM),'-B',str(path/'plot.py')],clean),
        ('aggregate-final',[str(GYM),'-B',str(path/'aggregate.py')],clean)]
    receipts=[]
    for name,cmd,child_env in stages:
        assert sha(R/'candidate-index.json')==seal['candidate_index_sha256']
        start=time.time();write(R/(name+'-command.json'),dict(argv=cmd,cwd=str(ROOT),start_time=start,CUDA_VISIBLE_DEVICES='',no_model_or_training=True))
        with (R/(name+'.log')).open('x') as f:
            result=subprocess.run(cmd,cwd=ROOT,env=child_env,stdout=f,stderr=subprocess.STDOUT)
        receipt=dict(stage=name,returncode=result.returncode,elapsed_s=time.time()-start,end_time=time.time())
        write(R/(name+'-exit.json'),receipt);receipts.append(receipt)
        print(json.dumps(receipt),flush=True)
        if result.returncode:
            write(R/'DATA_AUDIT_STOP.json',receipt);raise SystemExit(result.returncode)
        if name=='convert':
            dest=R/'lerobot/local/wc4_fulltask_expand_r9_20261021'
            shutil.copyfile(R/'candidate-index.json',dest/'collection-episode-parameters.json')
    write(R/'data-pipeline-result.json',dict(complete=True,stages=receipts,collection=seal,training=False,policy_inference=False))


if __name__=='__main__':main()
