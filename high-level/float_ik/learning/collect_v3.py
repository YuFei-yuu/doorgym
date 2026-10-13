"""Collect the frozen source-API expert on the predeclared disjoint seed sets."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).parent/'run_tools'))
from launch import launch
from bounded import atomic_json
from dataset_contract import validate_episode


def collect(root, reference, total=7200):
    root=Path(root).resolve();reference=Path(reference).resolve()
    frozen=json.loads((reference/'config.json').read_text())
    if not json.loads((reference/'summary.json').read_text())['success']:
        raise ValueError('Reference expert has not completed the task')
    repo=Path(__file__).resolve().parents[3]
    for name,digest in frozen['source_hashes'].items():
        if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen execution source changed: '+name)
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    contract=dict(schema=frozen['schema'],execution_profile=frozen['execution_profile'],
                  source_hashes=frozen['source_hashes'],tensor_pipeline=frozen['tensor_pipeline'],
                  fps=50,split_seeds=dict(train=list(range(14100,14116)),val=list(range(14200,14204))),
                  expert_args={k:frozen['args'][k] for k in ['robot_x','robot_y','robot_z','robot_yaw','stop_distance','push_base_vx','lever_step_size','float_execution','float_push_target']})
    atomic_json(root/'recording_contract.json',contract)
    for split,seeds in contract['split_seeds'].items():
        raw_out=root/'data'/('raw_'+split);raw_out.mkdir(parents=True,exist_ok=True)
        for index,seed in enumerate(seeds):
            previous=[x for x in manifest if x['seed']==seed]
            if previous:
                if len(previous)!=1 or not previous[0]['success']:raise ValueError('Prior failed/duplicate seed requires review')
                continue
            folder=root/'dataset'/('%s_seed_%d'%(split,seed))
            command=['python',str(Path(__file__).with_name('wholebody_float.py')),
                     '--headless','--no_show_seg','--steps','2500','--camera_seg','--camera_depth',
                     '--float_seed',str(seed),'--float_output',str(folder)]
            for key,value in contract['expert_args'].items():command += ['--'+key,str(value)]
            code=launch(root,'collection__%s_%d'%(split,seed),command,timeout=150,total=total)
            summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {'success':False,'end_reason':'process_failure'}
            accepted=code==0 and summary['success'] and summary.get('invalid_image_frames',0)==0
            validation_error=None
            if accepted:
                try:
                    cfg=json.loads((folder/'config.json').read_text())
                    if cfg['source_hashes']!=frozen['source_hashes']:raise ValueError('Executor drift during collection')
                    with np.load(folder/'trajectory.npz') as z:validate_episode(z)
                except Exception as exc:
                    accepted=False;validation_error=repr(exc)
            entry=dict(seed=seed,split=split,success=accepted,episode=str((folder/'trajectory.npz').relative_to(root)),
                       raw_episode=str((folder/'raw/episode_000000.npz').relative_to(root)),result=summary,exit_code=code,validation_error=validation_error)
            manifest.append(entry);atomic_json(manifest_path,manifest)
            if not accepted:raise RuntimeError('Expert collection attempt failed; inspect seed %d'%seed)
            shutil.copy2(folder/'raw/episode_000000.npz',raw_out/('episode_%06d.npz'%index))
            sidecar=folder/'raw/door_dp_feature_names.json'
            destination=raw_out/sidecar.name
            if destination.exists() and destination.read_bytes()!=sidecar.read_bytes():raise ValueError('Feature sidecar drift')
            shutil.copy2(sidecar,destination)
            print(json.dumps(dict(split=split,seed=seed,frames=summary['frames'],max_angle=summary['max_signed_door_deg'],accepted=True)),flush=True)
    result=dict(complete=True,episodes=len(manifest),train=sum(x['split']=='train' for x in manifest),val=sum(x['split']=='val' for x in manifest))
    atomic_json(root/'collection_result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--reference',required=True);p.add_argument('--total-seconds',type=float,default=7200);a=p.parse_args()
    collect(a.root,a.reference,a.total_seconds)
