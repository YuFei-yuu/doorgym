"""Authorized observation-source A/B branches; no collection or automatic expansion."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO/'high-level/dp'))
import collect_stop_tail as shared
from diagnostic_integrity import sha256, verify_frozen, phase_lock
from artifacts import finite_json

SOURCE = REPO/'experiments/float-push/20261014-stop-tail-resume5000-v1/pilot/training/checkpoints/step_005000.pt'
EXPECTED = '2c0b9634465c37d15ff9d461d79f8e5268b1956076d1ec72ba1792b887338026'
REF_A = REPO/'experiments/float-push/20261014-stop-tail-small3-v1/reference/seed_16040'
REF_B = REPO/'experiments/float-push/20261014-stop-tail-freeze-ab-v1/replay/B'
IMAGES = shared.IMAGES


def save(path, value):
    shared.save(path, finite_json(value))


def completed_stage(root, name, operation):
    """Never repeat successful computation after a later report/monitor repair."""
    folder=root/'stages';folder.mkdir(exist_ok=True)
    marker=folder/(name+'.json')
    if marker.exists() and json.loads(marker.read_text())['passed']:
        return 0
    attempt=len(list(folder.glob(name+'.attempt_*.json')))+1
    record=folder/('%s.attempt_%03d.json'%(name,attempt))
    start=time.time()
    try:
        result=operation('%s_attempt_%03d'%(name,attempt))
        if result not in (None,0):raise RuntimeError('Stage %s exit %s'%(name,result))
        value=dict(passed=True,attempt=attempt,started=start,ended=time.time())
        save(record,value);save(marker,value);return 0
    except BaseException as exc:
        save(record,dict(passed=False,attempt=attempt,started=start,ended=time.time(),error=repr(exc)))
        raise


def cpu_stage(root, name, arguments):
    return completed_stage(root,name,lambda actual:shared.cpu(root,actual,arguments))


def gpu_stage(root, name, command, **kwargs):
    return completed_stage(root,name,lambda actual:shared.launch(root,actual,command,**kwargs))


def register_repair(root):
    """Archive active source changes, retaining initial snapshot and prior verdicts."""
    plan=json.loads((root/'plan.json').read_text())
    changed={n:sha256(REPO/n) for n,h in plan['source_hashes'].items() if sha256(REPO/n)!=h}
    trained=any((root/l/'training/metrics.jsonl').exists() for l in ['A','B'])
    allowed={str((HERE/n).relative_to(REPO)) for n in ['branch16040_analysis.py','branch16040_online.py']}
    if trained and set(changed)-allowed:
        raise ValueError('After training, repair must be monitoring/analysis only')
    if changed:
        parent=root/'source_revisions';parent.mkdir(exist_ok=True)
        revision=parent/('%04d'%(len(list(parent.iterdir()))+1));revision.mkdir()
        save(revision/'prior_plan.json',plan)
        for name in changed:
            target=revision/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,target)
        save(revision/'manifest.json',dict(changed=changed,training_started=trained,at=time.time(),
            scope='Authorized recoverable engineering repair; experiment variables unchanged'))
        (revision/'worktree.diff').write_bytes(shared.git('diff','--binary','HEAD'))
        plan['source_hashes'].update(changed);save(root/'plan.json',plan)
    archive=root/'failure_attempts';archive.mkdir(exist_ok=True)
    for name in ['failure.json','completion.json','state.json']:
        path=root/name
        if path.exists():
            target=archive/(str(time.time_ns())+'_'+name);path.rename(target)
    verify(root)


def verify(root):
    plan = verify_frozen(root, REPO)
    if not shared.verify_inputs(plan)['passed']:
        raise ValueError('Protected input changed')


def verify_raw_trajectory(raw, trajectory):
    # RawDoorDPRecorder stores pre-action fields. next_state is recorded by the
    # floating backend in trajectory/execution only and is checked there below.
    for key in ['state', 'action', 'subtask_index', *IMAGES]:
        if not np.array_equal(raw[key], trajectory[key]):
            raise ValueError('Raw/trajectory mismatch '+key)


def prepare(root):
    import torch
    if sha256(SOURCE) != EXPECTED:
        raise ValueError('Source checkpoint hash mismatch')
    baseline = torch.load(SOURCE, map_location='cpu', weights_only=False)
    inputs, dirs, sources = {}, {}, {}
    accepted=REPO/'experiments/float-push/20261014-stop-tail-small3-v1/episodes/seed_16040/data_acceptance.json'
    if not json.loads(accepted.read_text())['passed']:raise ValueError('Accepted original source verdict changed')
    for p in [SOURCE, REF_A, REF_B, accepted,
              REPO/'experiments/float-push/20261015-observation-branch16040-v1',
              REPO/'experiments/float-push/20261015-observation-branch16040-v2']:
        shared.register_input(p, inputs, dirs)
    for name in shared.git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0'):
        p = REPO/name
        if name.startswith(('high-level/', 'low-level/')) and p.suffix in ('.py', '.sh', '.json', '.urdf', '.yaml', '.yml') and p.is_file():
            sources[name] = sha256(p)
            dst = root/'sources'/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)
    for p in [REPO/'.repro/blackwell/activate.sh']:
        name = str(p.relative_to(REPO)); sources[name] = sha256(p)
        dst = root/'sources'/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)
    save(root/'plan.json', dict(source_hashes=sources, input_hashes=inputs, input_directories=dirs,
        source_checkpoint=str(SOURCE), source_sha256=EXPECTED, source_updates=5000,
        server_head=shared.git('rev-parse', 'HEAD').decode().strip(), nodes=[0,500,1000,2000],
        branch_new_updates=2000, observation_source={'A':str(REF_A), 'B':str(REF_B)},
        label_source=str(REF_A), fixed_stats_source=str(SOURCE), seed=42, batch_size=64,
        num_workers=0, lr=1e-4, horizons=[16,32,16], branch_initialization='weights and statistics only',
        no_holdout=True, no_recollection=True, no_stop_override=True,
        causal_pre_action_observations=True, inference_seed=16040,
        limitation='Observation source includes motion and natural snap differences; no isolated freeze causality.'))
    (root/'server_initial_status.txt').write_bytes(shared.git('status','--short','--branch'))
    (root/'task.diff').write_bytes(shared.git('diff','--binary','HEAD'))
    data = {}
    rows = {}
    for label, ref in [('A',REF_A),('B',REF_B)]:
        with np.load(ref/'trajectory.npz', allow_pickle=True) as z:
            data[label] = {k:z[k].copy() for k in z.files}
        with np.load(ref/'raw/episode_000000.npz', allow_pickle=True) as raw:
            verify_raw_trajectory(raw, data[label])
        rows[label] = [json.loads(x) for x in (ref/'execution.jsonl').read_text().splitlines()]
        d = data[label]
        if len(rows[label]) != 1446 or d['state'].shape != (1446,27) or d['action'].shape != (1446,10):
            raise ValueError('Incomplete branch input')
        if not np.array_equal(d['frame'],np.arange(1446)) or not np.array_equal(d['next_frame'],np.arange(1,1447)):
            raise ValueError('Noncontiguous frame alignment')
        if not all(np.isfinite(d[k]).all() for k in ['state','next_state','action']) or not d['image_valid'].all():
            raise ValueError('Nonfinite/missing observation')
        for i,r in enumerate(rows[label]):
            if r['step'] != i or r['completed_physical_steps'] != 4*(i+1) or r['sim_time'] != float(d['sim_time'][i]):
                raise ValueError('Observation/execution time alignment mismatch')
            if not np.array_equal(np.asarray(r['next_state'],dtype=np.float32),d['next_state'][i]):
                raise ValueError('Next state execution mismatch')
            if r['extra']['command_source'] != ('script' if label=='A' else 'replay'):
                raise ValueError('Unexpected input execution source')
            if label=='B' and (r['freeze_arm_zero'] or r['freeze_arm_default']):
                raise ValueError('Pure replay restored freeze')
            if label=='B' and not np.array_equal(np.array(r['diagnostic']['requested_action'],dtype=np.float32),data['A']['action'][i]):
                raise ValueError('Replay requested labels differ from original')
        # next_state and next pre-observation can differ only at an explicitly recorded snap.
        bad = [i for i in range(1,1446) if not np.array_equal(d['state'][i],d['next_state'][i-1]) and rows[label][i]['pre_observation_event']=='none']
        if bad: raise ValueError('Unexplained state continuity break '+str(bad[:5]))
    a,b = data['A'],data['B']
    for k in ['frame','next_frame','sim_time','state_feature_names','action_names']:
        if not np.array_equal(a[k],b[k]):raise ValueError('A/B timing/features differ: '+k)
    applied = torch.from_numpy(a['action'].copy())
    applied[:,5:9] /= torch.clamp(torch.norm(applied[:,5:9],dim=-1,keepdim=True),min=1e-6)
    if not np.array_equal(applied.numpy(),b['action']):
        raise ValueError('B applied action not exactly original Torch quaternion normalization')
    save(root/'input_alignment.json',dict(passed=True,frames=1446,window_count=1415,
        all_requested_B_actions_exact_to_A=True,applied_B_exact_to_original_torch_normalization=True,
        applied_action_max_difference=float(np.abs(a['action']-b['action']).max()),
        aligned_frame_time_and_pre_action_observations=True,labels_shifted=False,synthetic_states=False,
        snap_frames={k:[i for i,r in enumerate(rows[k]) if r['pre_observation_event']!='none'] for k in rows}))
    # Actual controller generation centers are on the 16-step queue clock.
    replan_source = REPO/'experiments/float-push/20261014-stop-tail-small3-v1/pilot/evaluation/step_001000/seed_16040/execution.jsonl'
    replans = [r['step'] for r in map(json.loads,replan_source.read_text().splitlines()) if r['extra']['dp_replanned']]
    shared.register_input(replan_source, inputs, dirs)
    stages = [('start',0,58),('approach',58,458),('grasp',458,978),('push',978,1149),
              ('cross',1149,1280),('conversion',1280,1296),('hold',1296,1415)]
    windows = []
    for name,lo,hi in stages:
        choices = [c for c in replans if lo<=c<hi and c<=1414]
        if not choices:raise ValueError('No causal replan window in '+name)
        for c in sorted(set([choices[0],choices[len(choices)//2],choices[-1]])):
            windows.append(dict(stage=name,center=c,history_last=c,stop_offset=1295-c))
    plan=json.loads((root/'plan.json').read_text());plan.update(input_hashes=inputs,input_directories=dirs,
        windows=windows,diffusion_seeds=[4201,4202,4203],replan_source=str(replan_source));save(root/'plan.json',plan)
    for label in ['A','B']:
        branch=root/label;branch.mkdir()
        (branch/'raw').mkdir()
        # B's observations are unchanged; supervision is the original requested 10D sequence.
        z=dict(data[label]);z['action']=a['action'].copy()
        np.savez_compressed(branch/'raw/episode_000000.npz',**z)
        save(branch/'recording_contract.json',dict(schema='float_push_upstream_ee_v3',
            execution_profile='upstream_wholebody_float_v3_forward_pass',tensor_pipeline='cpu',fps=50,
            source_expert_seeds=[16040],preprocessing_version='door_action_minmax_continuous_quat_v3',
            observation_source=str(REF_A if label=='A' else REF_B),action_source=str(REF_A),
            change='Observation-source branch with original common action labels; no alignment shift or padding'))
    verify(root)


def converted_audit(root):
    from action_contract import FORWARD_V3, canonicalize_sequence
    from train_door_dp import DoorDPSequenceDataset, _as_chw_u8
    for label in ['A','B']:
        with np.load(root/label/'raw/episode_000000.npz',allow_pickle=True) as z:
            original_state=z['state'].copy();original_action=z['action'].copy()
            state=canonicalize_sequence(original_state,slice(-5,-1));actions=canonicalize_sequence(original_action,slice(5,9))
            images={k:z[k].copy() for k in IMAGES}
        ds=DoorDPSequenceDataset(root/label/'lerobot/local/branch16040','local/branch16040',16,32,FORWARD_V3)
        if len(ds)!=1415 or ds.length!=1446:raise ValueError('Converted frame/window count mismatch')
        for i in range(1446):
            frame=ds.dataset[i]
            if not np.array_equal(frame['observation.state'].numpy(),original_state[i]) or not np.array_equal(frame['action'].numpy(),original_action[i]):
                raise ValueError('Converted original frame differs')
            if int(frame['episode_index'])!=0 or int(frame['frame_index'])!=i:raise ValueError('Converted frame index differs')
            for key in IMAGES:
                if not np.array_equal(_as_chw_u8(frame['observation.images.'+key]).numpy(),images[key][i].transpose(2,0,1)):
                    raise ValueError('Converted original image differs')
        for j,c in enumerate(ds.indices):
            item=ds[j];idx=np.maximum(0,np.arange(c-15,c+1))
            if not np.array_equal(item['state'].numpy(),state[idx]) or not np.array_equal(item['action'].numpy(),actions[c:c+32]):
                raise ValueError('Converted labels/history mismatch')
            for k,out in zip(IMAGES,['mask','masked_depth','front_mask','front_masked_depth']):
                if not np.array_equal(item[out].numpy(),images[k][idx].transpose(0,3,1,2)):
                    raise ValueError('Converted image mismatch '+k)
        save(root/label/'converted_audit.json',dict(passed=True,all_original_frames_read=1446,all_actual_windows_read=1415))
    with np.load(root/'A/raw/episode_000000.npz') as a,np.load(root/'B/raw/episode_000000.npz') as b:
        if not np.array_equal(a['action'],b['action']):raise ValueError('Branch common supervision changed')


def train_cmd(root,label,step):
    branch=root/label
    cmd=['python','-u',str(REPO/'high-level/dp/train_door_dp.py'),'--root',str(branch/'lerobot/local/branch16040'),
        '--repo_id','local/branch16040','--output_dir',str(branch/'training'),'--run_name',root.name+'-'+label,
        '--steps',str(step),'--batch_size','64','--num_workers','0','--lr','1e-4','--seed','42',
        '--action_contract','forward_v3','--obs_horizon','16','--pred_horizon','32','--action_horizon','16',
        '--num_diffusion_iters','100','--log_interval','10','--save_interval',str(step),
        '--recording_metadata',str(branch/'recording_contract.json'),'--stop_on_nonfinite_loss']
    cmd += ['--initialize_weights',str(SOURCE)] if step==500 else ['--resume',str(branch/'training/checkpoints'/('step_%06d.pt'%({1000:500,2000:1000}[step])))]
    return cmd


def initialization_audit(root,step):
    import torch
    baseline=torch.load(SOURCE,map_location='cpu',weights_only=False)
    checkpoints=[]
    for label in ['A','B']:
        path=root/label/('initialization' if step==0 else 'training')/'checkpoints'/('step_%06d.pt'%step)
        c=torch.load(path,map_location='cpu',weights_only=False);checkpoints.append(c)
        if not all(torch.equal(c['stats'][k],baseline['stats'][k]) for k in baseline['stats']):
            raise ValueError('Normalization changed')
        if c['config']['completed_updates']!=step or c['config']['initialization']['source_sha256']!=EXPECTED:
            raise ValueError('Branch provenance/update count mismatch')
        if step==0:
            if c['optimizer']['state'] or c['training_state']['sampler']['order'] is not None:
                raise ValueError('Initialization did not reset optimizer/sampler')
            if not all(torch.equal(c['model'][k],baseline['model'][k]) for k in baseline['model']):
                raise ValueError('Initial weights changed')
        elif not c['optimizer']['state'] or any(int(v['step'])!=step for v in c['optimizer']['state'].values()):
            raise ValueError('Branch optimizer count mismatch')
        save(root/label/('checkpoint_%06d.json'%step),dict(passed=True,sha256=sha256(path),new_updates=step,
            source_updates=5000,fixed_stats_exact=True,mode='weight_initialized_branch'))
    a,b=checkpoints
    if a['training_state']['rng']['python']!=b['training_state']['rng']['python']:raise ValueError('A/B Python RNG differs')
    na,nb=a['training_state']['rng']['numpy'],b['training_state']['rng']['numpy']
    if na[0]!=nb[0] or not np.array_equal(na[1],nb[1]) or na[2:]!=nb[2:]:raise ValueError('A/B NumPy RNG differs')
    if step==0 and any(c['training_state']['sampler'][k]!=0 for c in checkpoints for k in ['cursor','epoch']):
        raise ValueError('Initial sampler cursor/epoch not reset')
    if a['optimizer']['param_groups']!=b['optimizer']['param_groups']:raise ValueError('A/B optimizer groups differ')
    for k in ['order','generator']:
        x,y=a['training_state']['sampler'][k],b['training_state']['sampler'][k]
        if (x is None)!=(y is None) or (x is not None and not torch.equal(x,y)):
            raise ValueError('A/B sample order mismatch')
    for k in ['cursor','epoch']:
        if a['training_state']['sampler'][k]!=b['training_state']['sampler'][k]:raise ValueError('A/B cursor differs')
    # Training stochastic consumption stays paired; inference runs in separate processes.
    for k in ['torch','cuda']:
        x,y=a['training_state']['rng'][k],b['training_state']['rng'][k]
        if k=='cuda':
            if len(x)!=len(y) or not all(torch.equal(i,j) for i,j in zip(x,y)):raise ValueError('A/B CUDA RNG differs')
        elif not torch.equal(x,y):raise ValueError('A/B Torch RNG differs')
    save(root/('paired_initialization_%06d.json'%step),dict(passed=True,same_sample_order_and_cursor=True,
        same_torch_rng=True,source_weights_and_stats_exact=step==0,new_updates=step))


def evaluation(root,label,step):
    branch=root/label;folder=branch/'pilot/evaluation'/('step_%06d'%step)/'seed_16040'
    cfg=json.loads((REF_A/'config.json').read_text())['args']
    cmd=['python','-u',str(HERE/'wholebody_float.py'),'--headless','--no_show_seg','--camera_seg','--camera_depth',
        '--float_video','--float_seed','16040','--steps','2500','--float_output',str(folder),
        '--dp_policy_checkpoint',str(branch/'training/checkpoints'/('step_%06d.pt'%step)),
        '--dp_inference_steps','100','--dp_action_horizon','16']
    for k in ['robot_x','robot_y','robot_z','robot_yaw','stop_distance','push_base_vx','lever_step_size','float_execution','float_pass_mode','float_push_target']:
        cmd+=['--'+k,str(cfg[k])]
    code=gpu_stage(root,'eval_%s_%d'%(label,step),cmd,timeout=600,total=None,environment='gym')
    if code:raise RuntimeError('Closed-loop execution exit '+str(code))
    cpu_stage(root,'eval_metrics_%s_%d'%(label,step),[__file__,'--root',root,'--worker','eval_metrics','--label',label,'--step',step])


def run(root,continue_repair=False):
    if root.exists() and not continue_repair:raise FileExistsError('Independent new run required')
    if continue_repair:
        register_repair(root)
    else:
        root.mkdir(parents=True);(root/'logs').mkdir()
    loggers=[]
    try:
        completed_stage(root,'prepare',lambda _:prepare(root))
        with phase_lock(root,'branch16040'):
            cpu_stage(root,'branch_tests',['-m','unittest','discover','-s',REPO/'high-level/dp','-p','test_branch_initialization.py','-v'])
            for label in ['A','B']:
                cpu_stage(root,'convert_'+label,[REPO/'high-level/dp/convert_door_raw_to_lerobot.py',
                    '--raw_root',root/label/'raw','--root',root/label/'lerobot','--repo_id','local/branch16040'])
            cpu_stage(root,'converted_audit',[__file__,'--root',root,'--worker','converted'])
            for label in ['A','B']:
                cmd=train_cmd(root,label,500)
                cmd[cmd.index('--output_dir')+1]=str(root/label/'initialization')
                cmd+=['--initialization_only']
                code=gpu_stage(root,'initialize_'+label,cmd,environment='train')
                if code:raise RuntimeError('Branch initialization failed')
            cpu_stage(root,'audit_0',[__file__,'--root',root,'--worker','audit','--step','0'])
            # Online processes have separate RNG and no access to the optimizer.
            for label in ['A','B']:
                log= (root/'logs'/('online_'+label+'_'+str(time.time_ns())+'.log')).open('x')
                proc=subprocess.Popen([str(REPO/'.repro/envs/door-dp-train/bin/python'),'-u',str(HERE/'branch16040_online.py'),
                    '--root',str(root),'--label',label],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
                loggers.append((proc,log))
            deadline=time.time()+150
            while not all((root/x/'online/ready.json').exists() for x in ['A','B']):
                if any(p.poll() is not None for p,_ in loggers) or time.time()>deadline:raise RuntimeError('A/B online preflight failed')
                time.sleep(3)
            # Node 0 uses the common source weights and fixed stats; no optimization.
            code=gpu_stage(root,'offline_0',['python','-u',str(HERE/'branch16040_analysis.py'),'--root',root,'--step','0'],environment='gym')
            if code:raise RuntimeError('Baseline offline checks failed')
            verify(root)
            if not (root/'training_source_freeze.json').exists():
                save(root/'training_source_freeze.json',dict(source_hashes=json.loads((root/'plan.json').read_text())['source_hashes'],at=time.time(),
                    preparation_passed=True,source_updates=5000,new_updates_each=2000))
            for step in [500,1000,2000]:
                for label in ['A','B']:
                    verify(root);save(root/'state.json',dict(phase='training',label=label,new_updates_target=step,pid=os.getpid()))
                    code=gpu_stage(root,'train_%s_%d'%(label,step),train_cmd(root,label,step),environment='train')
                    if code:raise RuntimeError('Branch training exit '+str(code))
                cpu_stage(root,'audit_%d'%step,[__file__,'--root',root,'--worker','audit','--step',step])
                verify(root)
                code=gpu_stage(root,'offline_%d'%step,['python','-u',str(HERE/'branch16040_analysis.py'),'--root',root,'--step',str(step)],environment='gym')
                if code:raise RuntimeError('Cross offline checks failed')
                if step in [1000,2000]:
                    for label in ['A','B']:evaluation(root,label,step)
                    cpu_stage(root,'compare_%d'%step,[HERE/'branch16040_analysis.py','--root',root,'--step',step,'--closed-loop'])
            verify(root)
            save(root/'final_preservation.json',shared.verify_inputs(json.loads((root/'plan.json').read_text())))
            save(root/'completion.json',dict(phase='completed',branches=['A','B'],new_updates_each=2000,closed_loop_attempts=4,no_holdout=True,auto_expand=False))
            for proc,_ in loggers:
                proc.wait(timeout=120)
                if proc.returncode:raise RuntimeError('Online logger failed to finish')
            save(root/'state.json',dict(phase='completed',new_updates_each=2000))
    except BaseException as exc:
        save(root/'failure.json',dict(error=repr(exc),at=time.time(),auto_retry=False))
        save(root/'completion.json',dict(phase='failed_stopped',error=repr(exc),auto_expand=False))
        save(root/'state.json',dict(phase='failed_stopped',error=repr(exc)))
        raise
    finally:
        for proc,log in loggers:
            if proc.poll() is None:
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=15)
            log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--worker',choices=['converted','audit','eval_metrics']);p.add_argument('--step',type=int,default=0);p.add_argument('--label',choices=['A','B']);p.add_argument('--continue-repair',action='store_true');args=p.parse_args();root=Path(args.root).resolve()
    if args.worker=='converted':converted_audit(root)
    elif args.worker=='audit':initialization_audit(root,args.step)
    elif args.worker=='eval_metrics':
        from stop_tail_small3 import evaluation_metrics
        evaluation_metrics(root/args.label,args.step,16040)
    else:run(root,args.continue_repair)
