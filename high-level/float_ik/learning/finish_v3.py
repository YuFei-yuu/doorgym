"""Original LeRobot conversion, 5000-update training and frozen-checkpoint evaluation."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE/'run_tools'))
from launch import launch
from bounded import atomic_json


def main(root,total):
    root=Path(root).resolve()
    state_path=root/'pipeline_state.json'
    def state(phase,**extra):
        atomic_json(state_path,dict(phase=phase,**extra));print(json.dumps(dict(phase=phase,**extra)),flush=True)
    def cpu(script,args,log,training=True):
        prefix=REPO/'.repro/envs'/('door-dp-train' if training else 'b1z1-blackwell')
        env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
                 MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        env.pop('PYTHONPATH',None)
        env['LD_LIBRARY_PATH']=str(prefix/'lib')
        path=root/'logs'/log;path.parent.mkdir(exist_ok=True)
        with path.open('x') as out:
            subprocess.run(['nice','-n','10',str(prefix/'bin/python'),str(script),*map(str,args)],
                           cwd=REPO,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    try:
        result=json.loads((root/'collection_result.json').read_text())
        if result != dict(complete=True,episodes=20,train=16,val=4):raise ValueError('Collection incomplete')
        state('converting')
        for split in ['train','val']:
            cpu(REPO/'high-level/dp/convert_door_raw_to_lerobot.py',
                ['--raw_root',root/'data'/('raw_'+split),'--root',root/'data/lerobot'/split,
                 '--repo_id','local/'+split+'_v3'],'convert_'+split+'.log')
        cpu(HERE/'checks.py',['--root',root,'--output',root/'data_checks.json'],'data_checks.log')
        def train_args(steps,folder,repo_id,dataset):
            return ['python',str(REPO/'high-level/dp/train_door_dp.py'),'--root',str(dataset),
                    '--repo_id',repo_id,'--run_name','upstream-float-v3-'+str(steps),
                    '--steps',str(steps),'--batch_size','64','--lr','1e-4','--num_workers','0',
                    '--obs_horizon','16','--pred_horizon','32','--action_horizon','16',
                    '--num_diffusion_iters','100','--save_interval',str(steps),'--seed','42',
                    '--output_dir',str(folder),'--recording_metadata',str(root/'recording_contract.json')]
        state('fit20')
        command=train_args(20,root/'fit/training','local/fit_v3',root/'fit/lerobot/local/fit_v3')+['--log_interval','1']
        if launch(root,'fit__original_loop20',command,timeout=300,total=total,environment='train'):
            raise RuntimeError('Original-loop fit/throughput check failed')
        state('training5000')
        (root/'training').mkdir(exist_ok=True)
        shutil.copy2(REPO/'high-level/dp/train_door_dp.py',root/'training/train_source.py')
        command=train_args(5000,root/'training','local/train_v3',root/'data/lerobot/train/local/train_v3')
        if launch(root,'training__random5000',command,timeout=14400,total=total,environment='train'):
            raise RuntimeError('5000-update training failed or exceeded budget')
        latest=root/'training/checkpoints/model_latest.pt'
        frozen=root/'training/frozen_step_005000.pt'
        if frozen.exists():raise FileExistsError(frozen)
        shutil.copy2(latest,frozen);frozen.chmod(0o444)
        digest=hashlib.sha256(frozen.read_bytes()).hexdigest()
        cpu(HERE/'checks.py',['--root',root,'--checkpoint',frozen,'--expected-updates','5000','--output',root/'checkpoint_checks.json'],'checkpoint_checks.log')
        provenance=dict(checkpoint=str(frozen),sha256=digest,completed_updates=5000,seed=42,
            training_source_sha256=hashlib.sha256((root/'training/train_source.py').read_bytes()).hexdigest(),
            recording_contract=json.loads((root/'recording_contract.json').read_text()))
        atomic_json(root/'training/checkpoint_provenance.json',provenance)
        for name,expected in provenance['recording_contract']['source_hashes'].items():
            if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen execution source changed: '+name)
        state('evaluating',checkpoint_sha256=digest)
        (root/'evaluation').mkdir(exist_ok=True)
        if launch(root,'evaluation__offline',
                  ['python',str(HERE/'offline_eval.py'),'--root',str(root),'--checkpoint',str(frozen),
                   '--output',str(root/'evaluation/offline.json'),'--episodes','4','--windows','32'],
                  timeout=300,total=total):raise RuntimeError('Offline inference failed')
        cases=[]
        for steps in [60,300,900]:cases.append(('short_%d'%steps,14300,steps,True))
        for seed in range(14300,14303):cases.append(('validation_%d'%seed,seed,2500,True))
        for seed in range(14400,14403):cases.append(('test_%d'%seed,seed,2500,True))
        cases.append(('expert_14400',14400,2500,False))
        for name,seed,steps,policy in cases:
            folder=root/'evaluation'/name
            command=['python',str(HERE/'wholebody_float.py'),'--headless','--no_show_seg',
                     '--steps',str(steps),'--camera_seg','--camera_depth','--float_seed',str(seed),
                     '--stop_distance','0.25','--float_execution','source',
                     '--float_output',str(folder),'--float_video']
            if policy:command += ['--dp_policy_checkpoint',str(frozen),'--dp_inference_steps','100','--dp_action_horizon','16']
            code=launch(root,'evaluation__'+name,command,timeout=300,total=total)
            if code:raise RuntimeError('Evaluation process failed: '+name)
        state('delivering',checkpoint_sha256=digest)
        folders=[p.parent for p in (root/'evaluation').glob('*/execution.jsonl')]
        cpu(HERE/'analyze_v3.py',folders,'evaluation_plots.log',training=False)
        cpu(HERE/'deliver_v3.py',['--root',root],'delivery.log',training=False)
        state('complete',checkpoint_sha256=digest)
    except BaseException as exc:
        state('error',error=repr(exc));traceback.print_exc();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--total-seconds',required=True,type=float);a=p.parse_args()
    main(a.root,a.total_seconds)
