"""Predeclared forward expert collection, exact staged DP training, and video review."""
import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE/'run_tools'))
from bounded import atomic_json
from launch import launch
from dataset_contract import validate_episode
from forward_results import episode_result, aggregate

BASELINE = '0acc2f050479fb54a0670ccb90bc72b98b52a54a'
POSITIONS = list(itertools.product([3.9,4.1,4.3],[-.06,0.,.06]))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def cases(start, repeats):
    return [dict(seed=start+i*repeats+j, robot_x=x, robot_y=y, repeat=j)
            for i,(x,y) in enumerate(POSITIONS) for j in range(repeats)]


class Pipeline:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock = (self.root/'pipeline.lock').open('a')
        fcntl.flock(self.lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.plan_path = self.root/'plan.json'

    def state(self, phase, **extra):
        value = dict(phase=phase, at=time.time(), pid=os.getpid(), **extra)
        atomic_json(self.root/'pipeline_state.json', value)
        print(json.dumps(value),flush=True)

    def initialize(self):
        if self.plan_path.exists():
            self.plan = json.loads(self.plan_path.read_text())
            self.verify_sources()
            return
        splits = dict(train=cases(16000,10), val=cases(16100,2), test=cases(16200,2),
                      nominal=[dict(seed=16300+i,robot_x=4.1,robot_y=0.,repeat=i) for i in range(10)])
        target = {c['seed'] for group in splits.values() for c in group}
        conflicts=[]
        for pattern in ('summary.json','manifest.json','plan.json','recording_contract.json'):
            for path in (REPO/'experiments/float-push').rglob(pattern):
                if self.root in path.parents: continue
                try: value=json.loads(path.read_text())
                except (ValueError,OSError): continue
                def seeds(v):
                    if isinstance(v,dict):
                        for k,x in v.items():
                            if k in ('seed','float_seed') and isinstance(x,int): yield x
                            elif k in ('seeds','split_seeds'):
                                def numbers(z):
                                    if isinstance(z,int): yield z
                                    elif isinstance(z,list):
                                        for y in z: yield from numbers(y)
                                    elif isinstance(z,dict):
                                        for y in z.values(): yield from numbers(y)
                                yield from numbers(x)
                            else: yield from seeds(x)
                    elif isinstance(v,list):
                        for x in v: yield from seeds(x)
                overlap=target.intersection(seeds(value))
                if overlap: conflicts.append(dict(path=str(path),seeds=sorted(overlap)))
        if conflicts: raise ValueError('Planned seeds previously used: '+str(conflicts))
        reference=REPO/'experiments/float-push/20261013-forward-expert-grid18/seed_14508'
        cfg=json.loads((reference/'config.json').read_text())
        if not json.loads((reference/'summary.json').read_text())['success']:
            raise ValueError('Reference expert failed')
        expert={k:cfg['args'][k] for k in ('robot_z','robot_yaw','stop_distance','push_base_vx',
                    'lever_step_size','float_execution','float_push_target','float_pass_mode')}
        if expert['float_pass_mode'] != 'forward': raise ValueError('Reference is not forward')
        paths=set(cfg['source_hashes'])
        paths.update(str(p.relative_to(REPO)) for directory in (HERE,REPO/'high-level/dp')
                     for p in directory.rglob('*.py'))
        source_hashes={name:digest(REPO/name) for name in sorted(paths)}
        for name,expected in cfg['source_hashes'].items():
            if source_hashes[name]!=expected: raise ValueError('Reference execution changed: '+name)
        self.plan=dict(source_baseline=BASELINE,repo_commit=subprocess.check_output(
                      ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),splits=splits,
                      expert_args=expert,source_hashes=source_hashes,created=time.time(),
                      authorization='User approved full implementation on 2026-10-13',
                      checkpoints=[1000,2000,3000,4000,5000],nominal_required_successes=9,
                      nominal_attempts=10,total_time_limit=None,video_review_at=5000)
        for name in source_hashes:
            out=self.root/'sources'/name;out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(REPO/name,out)
        (self.root/'source_changes.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=REPO))
        atomic_json(self.plan_path,self.plan)
        atomic_json(self.root/'recording_contract.json',dict(schema=cfg['schema'],
                    execution_profile=cfg['execution_profile'],source_hashes=cfg['source_hashes'],
                    tensor_pipeline=cfg['tensor_pipeline'],fps=50,
                    split_seeds={k:[c['seed'] for c in v] for k,v in splits.items()},
                    expert_args=dict(expert,robot_x=4.1,robot_y=0.),cases=splits))

    def verify_sources(self):
        for name,expected in self.plan['source_hashes'].items():
            if digest(REPO/name)!=expected: raise ValueError('Frozen source drift: '+name)

    def cpu(self, script, args, stage, environment='train'):
        self.verify_sources()
        log=self.root/'logs'/(stage+'.log');log.parent.mkdir(exist_ok=True)
        prefix=REPO/'.repro/envs'/('door-dp-train' if environment=='train' else 'b1z1-blackwell')
        env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
                 MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                 WANDB_MODE='disabled',LD_LIBRARY_PATH=str(prefix/'lib'))
        env.pop('PYTHONPATH',None)
        with log.open('x') as out:
            increment=max(0,10-os.getpriority(os.PRIO_PROCESS,0))
            subprocess.run(['nice','-n',str(increment),str(prefix/'bin/python'),str(script),*map(str,args)],
                           cwd=REPO,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)

    def gpu(self, stage, command, environment='gym', timeout=None):
        self.verify_sources()
        return launch(self.root,stage,command,timeout=timeout,total=None,environment=environment)

    def rollout_command(self, folder, case, checkpoint=None, steps=2500):
        command=['python','-u',str(HERE/'wholebody_float.py'),'--headless','--no_show_seg',
                 '--camera_seg','--camera_depth','--steps',str(steps),'--float_output',str(folder),
                 '--float_seed',str(case['seed']),'--robot_x',str(case['robot_x']),
                 '--robot_y',str(case['robot_y']),'--float_video']
        for key,value in self.plan['expert_args'].items(): command += ['--'+key,str(value)]
        if checkpoint: command += ['--dp_policy_checkpoint',str(checkpoint),
                                  '--dp_inference_steps','100','--dp_action_horizon','16']
        return command

    def collect(self, selection):
        manifest_path=self.root/'manifest.json'
        manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        for split,case in selection:
            prior=[r for r in manifest if r['seed']==case['seed']]
            if prior:
                if len(prior)!=1 or not prior[0]['success']:
                    raise ValueError('Prior failed/unfinished collection requires diagnosis: '+str(case))
                continue
            self.state('collecting',split=split,case=case,completed=len(manifest),target=108)
            folder=self.root/'dataset'/('%s_seed_%d'%(split,case['seed']))
            entry=dict(case,split=split,success=False,started=time.time(),
                       episode=str((folder/'trajectory.npz').relative_to(self.root)),
                       raw_episode=str((folder/'raw/episode_000000.npz').relative_to(self.root)))
            manifest.append(entry);atomic_json(manifest_path,manifest)
            code=self.gpu('collection__%s_%d'%(split,case['seed']),self.rollout_command(folder,case),timeout=600)
            result=episode_result(folder,case,code,policy=False)
            entry.update(exit_code=code,result=result)
            if result['success'] and result['invalid_image_frames']==0:
                config=json.loads((folder/'config.json').read_text())
                if config['source_hashes']!=json.loads((self.root/'recording_contract.json').read_text())['source_hashes']:
                    raise ValueError('Collection executor drift')
                with np.load(folder/'trajectory.npz') as z:
                    meta=validate_episode(z)
                    if len(meta['state_names'])!=27 or len(meta['action_names'])!=10:
                        raise ValueError('Expected original 27D/10D interface')
                    for key in ('wrist_handle_mask','wrist_masked_depth','front_handle_mask','front_masked_depth'):
                        if z[key].shape!=(meta['frames'],54,96,3) or z[key].dtype!=np.uint8:
                            raise ValueError('Invalid camera tensor: '+key)
                entry.update(success=True,frames=meta['frames'],sha256=digest(folder/'trajectory.npz'),
                             raw_sha256=digest(folder/'raw/episode_000000.npz'))
            atomic_json(manifest_path,manifest)
            if not entry['success']: raise RuntimeError('Expert attempt failed: '+str(case))
            raw=self.root/'data'/('raw_'+split);raw.mkdir(parents=True,exist_ok=True)
            index=next(i for i,c in enumerate(self.plan['splits'][split]) if c['seed']==case['seed'])
            shutil.copy2(folder/'raw/episode_000000.npz',raw/('episode_%06d.npz'%index))
            sidecar=folder/'raw/door_dp_feature_names.json'
            if (raw/sidecar.name).exists() and (raw/sidecar.name).read_bytes()!=sidecar.read_bytes():
                raise ValueError('Sidecar drift')
            shutil.copy2(sidecar,raw/sidecar.name)
            print(json.dumps(dict(collected=len(manifest),seed=case['seed'],frames=entry['frames'])),flush=True)
        return manifest

    def train_command(self, steps, output, dataset, repo_id, resume=None, log_interval=100):
        command=['python','-u',str(REPO/'high-level/dp/train_door_dp.py'),'--root',str(dataset),
                 '--repo_id',repo_id,'--steps',str(steps),'--batch_size','64','--lr','1e-4',
                 '--num_workers','0','--obs_horizon','16','--pred_horizon','32','--action_horizon','16',
                 '--num_diffusion_iters','100','--save_interval','1000','--seed','42',
                 '--log_interval',str(log_interval),'--output_dir',str(output),
                 '--recording_metadata',str(self.root/'recording_contract.json')]
        if resume: command += ['--resume',str(resume)]
        return command

    def preflight(self):
        marker=self.root/'preflight.json'
        if marker.exists(): return
        self.state('preflight')
        self.cpu(HERE/'run_tools/test_budget.py',[],'test_budget',environment='gym')
        self.cpu(HERE/'test_forward_pass.py',[],'test_forward_pass',environment='gym')
        self.cpu(HERE/'test_forward_results.py',[],'test_forward_results',environment='gym')
        self.cpu(REPO/'high-level/dp/test_training_state.py',[],'test_training_state')
        atomic_json(marker,dict(passed=True,at=time.time()))

    def smoke(self, manifest):
        marker=self.root/'smoke/result.json'
        if marker.exists(): return
        self.state('smoke')
        raw=self.root/'smoke/raw';raw.mkdir(parents=True,exist_ok=True)
        for i,item in enumerate(manifest):
            source=self.root/item['raw_episode']
            shutil.copy2(source,raw/('episode_%06d.npz'%i))
            shutil.copy2(source.parent/'door_dp_feature_names.json',raw/'door_dp_feature_names.json')
        self.cpu(REPO/'high-level/dp/convert_door_raw_to_lerobot.py',
                 ['--raw_root',raw,'--root',self.root/'smoke/lerobot','--repo_id','local/smoke_forward'], 'smoke_conversion')
        dataset=self.root/'smoke/lerobot/local/smoke_forward'
        for name,target,resume in [('continuous',4,None),('split',2,None),
                    ('resumed',4,self.root/'smoke/split/checkpoints/step_000002.pt')]:
            output=self.root/'smoke'/('split' if name=='resumed' else name)
            code=self.gpu('smoke__'+name,self.train_command(target,output,dataset,
                          'local/smoke_forward',resume=resume,log_interval=1),environment='train')
            if code: raise RuntimeError('Smoke training failed: '+name)
        self.cpu(HERE/'verify_resume.py',['--continuous',self.root/'smoke/continuous/checkpoints/step_000004.pt',
                 '--resumed',self.root/'smoke/split/checkpoints/step_000004.pt',
                 '--output',self.root/'smoke/resume_check.json'],'verify_resume')
        case=dict(seed=16108,robot_x=4.1,robot_y=0.,repeat=0)
        checkpoint=self.root/'smoke/continuous/checkpoints/step_000004.pt'
        folder=self.root/'smoke/rollout'
        code=self.gpu('smoke__rollout60',self.rollout_command(folder,case,checkpoint,steps=60),timeout=600)
        result=episode_result(folder,case,code)
        if code or not result['complete_log'] or not result['autonomous']:
            raise RuntimeError('Smoke policy execution did not remain autonomous')
        atomic_json(marker,dict(passed=True,rollout=result,at=time.time()))

    def convert(self, split):
        marker=self.root/('conversion_'+split+'.json')
        if marker.exists(): return
        self.state('converting',split=split)
        self.cpu(REPO/'high-level/dp/convert_door_raw_to_lerobot.py',
            ['--raw_root',self.root/'data'/('raw_'+split),'--root',self.root/'data/lerobot'/split,
             '--repo_id','local/forward_'+split],'conversion_'+split)
        atomic_json(marker,dict(complete=True,at=time.time()))

    def evaluate(self, checkpoint, selection, name):
        out=self.root/'evaluation'/name;out.mkdir(parents=True,exist_ok=True)
        attempts_path=out/'attempts.json';rows_path=out/'results.json'
        attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else []
        rows=json.loads(rows_path.read_text()) if rows_path.exists() else []
        expected_hash=digest(checkpoint)
        for case in selection:
            if any(r['seed']==case['seed'] for r in rows): continue
            folder=out/('seed_%d'%case['seed'])
            interrupted=any(r['seed']==case['seed'] for r in attempts)
            self.state('evaluating',checkpoint=str(checkpoint),suite=name,completed=len(rows),case=case)
            error=None
            if interrupted:
                # An interrupted attempt stays a failure; no silent rerun.
                code=125;error='Interrupted prior attempt; retained without retry'
            else:
                attempts.append(dict(case,started=time.time(),checkpoint_sha256=expected_hash))
                atomic_json(attempts_path,attempts)
                try:
                    code=self.gpu('evaluation__%s_%d'%(name,case['seed']),
                                  self.rollout_command(folder,case,checkpoint),timeout=600)
                except Exception as exc:
                    code=125;error=repr(exc)
            try: result=episode_result(folder,case,code)
            except Exception as exc:
                result=episode_result(out/'missing',case,125);error=repr(exc)
            if error: result.update(error=error,success=False)
            rows.append(result);atomic_json(rows_path,rows)
            atomic_json(out/'aggregate.json',aggregate(rows,len(selection)))
            self.report()
            if digest(checkpoint)!=expected_hash: raise ValueError('Frozen checkpoint changed')
        self.cpu_once(HERE/'forward_report.py',['--root',self.root,'--suite',name], 'report_'+name)
        return rows

    def cpu_once(self, script, args, stage, environment='train'):
        marker=self.root/'completed_stages'/(stage+'.json')
        if marker.exists(): return
        self.cpu(script,args,stage,environment)
        marker.parent.mkdir(exist_ok=True);atomic_json(marker,dict(complete=True,at=time.time()))

    def report(self):
        # Small CPU-only report; all expensive per-episode plots are made after a suite.
        self.cpu_once(HERE/'forward_report.py',['--root',self.root],
                      'progress_report_'+str(time.time_ns()))

    def run(self):
        self.initialize();self.preflight()
        pilot=[('train',c) for c in self.plan['splits']['train'] if c['robot_x']==4.1 and c['robot_y']==0.][:3]
        manifest=self.collect(pilot);self.smoke(manifest)
        manifest=self.collect([(split,c) for split in ('train','val') for c in self.plan['splits'][split]])
        atomic_json(self.root/'collection_result.json',dict(complete=True,episodes=len(manifest),
             train=sum(x['split']=='train' for x in manifest),val=sum(x['split']=='val' for x in manifest),
             train_frames=sum(x['frames'] for x in manifest if x['split']=='train'),
             val_frames=sum(x['frames'] for x in manifest if x['split']=='val')))
        for split in ('train','val'): self.convert(split)
        self.cpu_once(HERE/'checks.py',['--root',self.root,'--output',self.root/'data_checks.json'],'data_checks')
        previous=None
        for step in self.plan['checkpoints']:
            checkpoint=self.root/'training/checkpoints'/('step_%06d.pt'%step)
            if not checkpoint.exists():
                self.state('training',target_updates=step,resume=str(previous) if previous else None)
                code=self.gpu('training__%06d'%step,self.train_command(step,self.root/'training',
                             self.root/'data/lerobot/train/local/forward_train','local/forward_train',
                             resume=previous),environment='train')
                if code: raise RuntimeError('Training failed: '+str(step))
            self.cpu_once(HERE/'checks.py',['--root',self.root,'--checkpoint',checkpoint,
                          '--expected-updates',step,'--output',self.root/('checkpoint_checks_%06d.json'%step)],
                          'checkpoint_checks_%06d'%step)
            atomic_json(checkpoint.with_suffix('.json'),dict(checkpoint=str(checkpoint),
                        sha256=digest(checkpoint),completed_updates=step,source_baseline=BASELINE))
            validation=self.plan['splits']['val'] if step==5000 else self.plan['splits']['val'][::2]
            self.evaluate(checkpoint,validation,'step_%06d'%step)
            previous=checkpoint
        offline=self.root/'evaluation/step_005000/offline.json'
        if not offline.exists():
            code=self.gpu('offline__5000',['python','-u',str(HERE/'offline_eval.py'),'--root',str(self.root),
                          '--checkpoint',str(previous),'--output',str(offline),'--episodes','4','--windows','32'])
            if code: raise RuntimeError('Offline diagnostics failed')
        self.cpu_once(HERE/'forward_review.py',['--root',self.root],'review5000',environment='gym')
        self.state('awaiting_video_review',completed_updates=5000,checkpoint=str(previous),
                   checkpoint_sha256=digest(previous),objective_complete=False,
                   next_action='User reviews full videos and agrees next training stage or candidate testing')
        self.report()

    def final_test(self, checkpoint):
        self.initialize()
        if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
        checkpoint_hash=digest(checkpoint)
        candidate=self.root/'candidate.json'
        if candidate.exists() and json.loads(candidate.read_text())['sha256']!=checkpoint_hash:
            raise ValueError('Final test already assigned to another candidate; use fresh held-out seeds')
        atomic_json(candidate,dict(checkpoint=str(checkpoint),sha256=checkpoint_hash,
                    authorization='Explicit --final-checkpoint requested after video review',at=time.time()))
        self.evaluate(checkpoint,self.plan['splits']['test'],'final_test')
        rows=self.evaluate(checkpoint,self.plan['splits']['nominal'],'nominal_acceptance')
        count=sum(r['success'] for r in rows)
        self.state('awaiting_final_video_review',nominal_successes=count,nominal_attempts=len(rows),
                   numerical_gate_passed=count>=9 and len(rows)==10,objective_complete=False)
        self.report()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--final-checkpoint',type=Path);a=p.parse_args()
    pipeline=Pipeline(a.root)
    try:
        if a.final_checkpoint: pipeline.final_test(a.final_checkpoint.resolve())
        else: pipeline.run()
    except BaseException as exc:
        pipeline.state('error',error=repr(exc),objective_complete=False)
        traceback.print_exc();raise
