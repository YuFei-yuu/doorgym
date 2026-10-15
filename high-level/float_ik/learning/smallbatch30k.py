"""Single accepted B episode, fresh stats/random seed42, batch8, exactly30k updates."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'high-level/dp'))
import collect_stop_tail as shared
from artifacts import finite_json
from diagnostic_integrity import sha256,verify_frozen,phase_lock
from branch16040 import completed_stage,cpu_stage,gpu_stage

OLD=REPO/'experiments/float-push/20261015-observation-branch16040-v3'
RAW=OLD/'B/raw/episode_000000.npz'
REFERENCE=OLD/'reference/pure10D_B'
NODES=[1000,3000,5000,10000,20000,30000]
IMAGES=shared.IMAGES


def save(path,value):shared.save(path,finite_json(value))


def verify(root):
    plan=verify_frozen(root,REPO)
    if not shared.verify_inputs(plan)['passed']:raise ValueError('Protected input changed')
    data=root/'data/lerobot/local/single_B16040'
    if (root/'data_hashes.json').exists():
        hashes=json.loads((root/'data_hashes.json').read_text())
        if any(sha256(data/n)!=h for n,h in hashes.items()):raise ValueError('Converted training data changed')


def prepare(root):
    import torch
    old_manifest=json.loads((OLD/'delivery_sha256.json').read_text())
    if sha256(RAW)!=old_manifest['B/raw/episode_000000.npz']:raise ValueError('Accepted B/raw source hash changed')
    inputs,dirs,sources={},{},{}
    shared.register_input(OLD,inputs,dirs)
    for name in shared.git('ls-files','-z','--cached','--others','--exclude-standard').decode().split('\0'):
        p=REPO/name
        if name.startswith(('high-level/','low-level/')) and p.suffix in ('.py','.sh','.json','.yaml','.yml','.urdf') and p.is_file():
            sources[name]=sha256(p);target=root/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    p=REPO/'.repro/blackwell/activate.sh';name=str(p.relative_to(REPO));sources[name]=sha256(p)
    target=root/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    old_plan=json.loads((OLD/'plan.json').read_text())
    save(root/'plan.json',dict(source_hashes=sources,input_hashes=inputs,input_directories=dirs,
        experiment_kind='single_B_random_batch8_30k',raw_source=str(RAW),raw_sha256=sha256(RAW),
        nodes=NODES,max_updates=30000,batch_size=8,num_workers=0,seed=42,lr=1e-4,weight_decay=1e-6,gradient_clip=1.,
        frames=1446,sequences=1415,horizons=[16,32,16],DDPM_steps=100,
        stats_source='This training episode only, original compute_stats coverage rules',
        initialization='Random; no old model/optimizer/sampler/statistics loaded',
        windows=old_plan['windows'],diffusion_seeds=old_plan['diffusion_seeds'],
        replan_clock_source=old_plan['replan_source'],first_stop_label_frame=1295,
        no_recollection=True,no_heldout=True,no_tail_padding=True,no_stop_override=True,
        server_head=shared.git('rev-parse','HEAD').decode().strip(),created=time.time()))
    rows=[json.loads(x) for x in (REFERENCE/'execution.jsonl').read_text().splitlines()]
    with np.load(RAW,allow_pickle=True) as z,np.load(REFERENCE/'trajectory.npz',allow_pickle=True) as trajectory:
        for k in ['state','next_state','frame','next_frame','sim_time','subtask_index','state_feature_names','action_names',*IMAGES]:
            if not np.array_equal(z[k],trajectory[k]):raise ValueError('Accepted raw/record time or observation mismatch: '+k)
        requested=np.asarray([r['diagnostic']['requested_action'] for r in rows],dtype=np.float32)
        if not np.array_equal(z['action'],requested):raise ValueError('Common approved action labels changed')
        normalized=torch.from_numpy(z['action'].copy());normalized[:,5:9]/=torch.clamp(torch.norm(normalized[:,5:9],dim=-1,keepdim=True),min=1e-6)
        if not np.array_equal(normalized.numpy(),trajectory['action']):raise ValueError('Quaternion execution meaning differs')
        if z['state'].shape!=(1446,27) or z['action'].shape!=(1446,10) or len(rows)!=1446:raise ValueError('Wrong single-episode input size')
        if not all(np.isfinite(z[k]).all() for k in ['state','action','next_state']) or not z['image_valid'].all():raise ValueError('Nonfinite/invalid source')
        for i,row in enumerate(rows):
            if row['step']!=i or row['completed_physical_steps']!=4*(i+1) or row['sim_time']!=float(z['sim_time'][i]):raise ValueError('Input execution time mismatch')
            if not np.array_equal(np.asarray(row['next_state'],dtype=np.float32),z['next_state'][i]):raise ValueError('Next state log mismatch')
            if row['extra']['command_source']!='replay' or row['freeze_arm_zero'] or row['freeze_arm_default']:raise ValueError('Input not pure10D replay')
            if i and row['pre_observation_event']=='none' and not np.array_equal(z['state'][i],z['next_state'][i-1]):raise ValueError('Unlogged continuity break')
    (root/'raw').mkdir();shutil.copy2(RAW,root/'raw/episode_000000.npz')
    if sha256(root/'raw/episode_000000.npz')!=sha256(RAW):raise ValueError('Input copy changed')
    shutil.copytree(REFERENCE,root/'reference',ignore=shutil.ignore_patterns('source','assets'))
    save(root/'input_acceptance.json',dict(passed=True,frames=1446,raw_byte_exact=True,
        shared_action_labels_exact=True,normalized_quaternion_execution_exact=True,
        snap_frames=[r['step'] for r in rows if r['pre_observation_event']!='none'],
        pre_action_time_alignment=True,no_synthetic_states_or_tail=True))
    save(root/'recording_contract.json',dict(schema='float_push_upstream_ee_v3',execution_profile='upstream_wholebody_float_v3_forward_pass',
        fps=50,tensor_pipeline='cpu',source_expert_seeds=[16040],preprocessing_version='door_action_minmax_continuous_quat_v3',
        training_input_sha256=sha256(RAW),observation_source='accepted pure10D replay B',action_source='approved common16040 action sequence',
        statistics='computed from this training data only',random_initialization=True))
    (root/'initial_git_status.txt').write_bytes(shared.git('status','--short','--branch'))
    (root/'initial_worktree.diff').write_bytes(shared.git('diff','--binary','HEAD'))
    verify(root)


def data_audit(root):
    import torch
    from action_contract import FORWARD_V3,canonicalize_sequence
    from train_door_dp import DoorDPSequenceDataset,compute_stats,_as_chw_u8
    data=root/'data/lerobot/local/single_B16040'
    ds=DoorDPSequenceDataset(data,'local/single_B16040',16,32,FORWARD_V3)
    if len(ds)!=1415 or ds.length!=1446:raise ValueError('Wrong converted frame/window counts')
    with np.load(root/'raw/episode_000000.npz',allow_pickle=True) as z:
        original_state=z['state'].copy();original_action=z['action'].copy();images={k:z[k].copy() for k in IMAGES}
    states=canonicalize_sequence(original_state,slice(-5,-1));actions=canonicalize_sequence(original_action,slice(5,9))
    for i in range(1446):
        frame=ds.dataset[i]
        if not np.array_equal(frame['observation.state'].numpy(),original_state[i]) or not np.array_equal(frame['action'].numpy(),original_action[i]):raise ValueError('Converted original frame mismatch')
        if int(frame['episode_index'])!=0 or int(frame['frame_index'])!=i:raise ValueError('Converted frame index mismatch')
        for key in IMAGES:
            if not np.array_equal(_as_chw_u8(frame['observation.images.'+key]).numpy(),images[key][i].transpose(2,0,1)):raise ValueError('Converted original image mismatch')
    for j,c in enumerate(ds.indices):
        item=ds[j];ids=np.maximum(0,np.arange(c-15,c+1))
        if not np.array_equal(item['state'].numpy(),states[ids]) or not np.array_equal(item['action'].numpy(),actions[c:c+32]):raise ValueError('Actual window/history/labels mismatch')
        for key,out in zip(IMAGES,['mask','masked_depth','front_mask','front_masked_depth']):
            if not np.array_equal(item[out].numpy(),images[key][ids].transpose(0,3,1,2)):raise ValueError('Actual window image mismatch')
    stats=compute_stats(ds)
    state_values=torch.from_numpy(states[ds.state_stats_mask]);action_values=torch.from_numpy(actions[ds.action_stats_mask])
    independent=dict(state_mean=state_values.mean(0),state_std=state_values.std(0).clamp_min(1e-6),
        action_mean=action_values.mean(0),action_std=action_values.std(0).clamp_min(1e-6),action_min=action_values.amin(0),action_max=action_values.amax(0))
    if any(not torch.equal(stats[k],v) for k,v in independent.items()):raise ValueError('Original stats rule differs from direct training-only calculation')
    torch.save(stats,root/'training_only_stats.pt')
    save(root/'statistics.json',dict(stats={k:v.tolist() for k,v in stats.items()},state_unique_frames=int(ds.state_stats_mask.sum()),
        action_unique_frames=int(ds.action_stats_mask.sum()),all_actual_frames=1446,all_actual_windows=1415,
        independent_training_only_check=True,old_stats_loaded=False))
    save(root/'data_hashes.json',{str(p.relative_to(data)):sha256(p) for p in sorted(data.rglob('*')) if p.is_file()})
    save(root/'data_acceptance.json',dict(passed=True,all_frames_read=1446,all_windows_read=1415,labels_unshifted=True,
        original_loader=True,no_image_fallback_used=True,statistics_from_training_data_only=True))


def train_command(root,step,initialization=False):
    if step not in NODES:raise ValueError('Unauthorized optimizer update target')
    args=['python','-u',str(REPO/'high-level/dp/train_door_dp.py'),'--root',str(root/'data/lerobot/local/single_B16040'),
        '--repo_id','local/single_B16040','--output_dir',str(root/('initialization' if initialization else 'training')),
        '--run_name',root.name,'--steps',str(step),'--batch_size','8','--num_workers','0','--lr','1e-4','--seed','42',
        '--action_contract','forward_v3','--obs_horizon','16','--pred_horizon','32','--action_horizon','16','--num_diffusion_iters','100',
        '--log_interval','10','--save_interval',str(step),'--recording_metadata',str(root/'recording_contract.json'),'--stop_on_nonfinite_loss']
    if initialization:args+=['--initialization_only']
    elif step==1000:args+=['--verify_random_initialization',str(root/'initialization/checkpoints/step_000000.pt')]
    else:args+=['--resume',str(root/'training/checkpoints'/('step_%06d.pt'%NODES[NODES.index(step)-1]))]
    return args


def audit_checkpoint(root,step):
    import torch
    from training_state import ResumableBatchSampler
    p=root/('initialization' if step==0 else 'training')/'checkpoints'/('step_%06d.pt'%step)
    c=torch.load(p,map_location='cpu',weights_only=False);cfg=c['config']
    expected=torch.load(root/'training_only_stats.pt',map_location='cpu',weights_only=False)
    if cfg['completed_updates']!=step or cfg['batch_size']!=8 or cfg['dataset_frames']!=1446 or cfg['dataset_sequences']!=1415:raise ValueError('Update/batch/data contract changed')
    if cfg['initialization']['mode']!='random_seeded_training_data_stats' or cfg.get('initialize_weights') is not None:raise ValueError('Not random initialization')
    if any(not torch.equal(c['stats'][k],expected[k]) for k in expected):raise ValueError('Training-only normalization changed')
    values=list(c['model'].values())+list(c['stats'].values())+[v for state in c['optimizer']['state'].values() for v in state.values() if isinstance(v,torch.Tensor)]
    if not all(torch.isfinite(v).all() for v in values if v.is_floating_point()):raise FloatingPointError('Nonfinite checkpoint state')
    sampler=c['training_state']['sampler']
    if step==0:
        if c['optimizer']['state'] or sampler['order'] is not None or sampler['cursor']!=0 or sampler['epoch']!=0 or cfg.get('resume'):raise ValueError('Random initial optimizer/sampler not empty')
        reset=ResumableBatchSampler(1415,8,43)
        if not torch.equal(reset.state_dict()['generator'],sampler['generator']):raise ValueError('Random sampler generator not reset')
    else:
        probe=ResumableBatchSampler(1415,8,43);probe.load_state_dict(sampler)
        if sampler['cursor']!=(step%176)*8 or any(int(v['step'])!=step for v in c['optimizer']['state'].values()):raise ValueError('Incomplete optimized state/position')
        previous=None if step==1000 else str(root/'training/checkpoints'/('step_%06d.pt'%NODES[NODES.index(step)-1]))
        if cfg.get('resume')!=previous:raise ValueError('Resume is not this run previous node')
        init=torch.load(root/'initialization/checkpoints/step_000000.pt',map_location='cpu',weights_only=False)
        if step==1000:
            first=torch.load(root/'training/checkpoints/step_000000.pt',map_location='cpu',weights_only=False)
            if not all(torch.equal(first['model'][k],init['model'][k]) for k in init['model']):raise ValueError('Actual fresh starting weights differ from audit')
        from stop_tail_resume5000 import canonical_scheduler
        if canonical_scheduler(cfg['diffusion_scheduler'])!=canonical_scheduler(init['config']['diffusion_scheduler']):raise ValueError('Frozen DDPM configuration changed')
    save(root/('checkpoint_%06d.json'%step),dict(passed=True,step=step,sha256=sha256(p),stats_exact_to_training_only=True,
        optimizer_parameter_states=len(c['optimizer']['state']),sampler_cursor=sampler['cursor'],sampler_epoch=sampler['epoch'],
        initialization='random_seed42',full_resume=step>1000))


def evaluate(root,step):
    folder=root/'pilot/evaluation'/('step_%06d'%step)/'seed_16040'
    cfg=json.loads((root/'reference/config.json').read_text())['args']
    command=['python','-u',str(HERE/'wholebody_float.py'),'--headless','--no_show_seg','--camera_seg','--camera_depth',
        '--float_seed','16040','--steps','2500','--float_video','--float_output',str(folder),
        '--dp_policy_checkpoint',str(root/'training/checkpoints'/('step_%06d.pt'%step)),'--dp_inference_steps','100','--dp_action_horizon','16']
    for k in ['robot_x','robot_y','robot_z','robot_yaw','stop_distance','push_base_vx','lever_step_size','float_execution','float_pass_mode','float_push_target']:
        command+=['--'+k,str(cfg[k])]
    gpu_stage(root,'closed_%d'%step,command,environment='gym',timeout=600,total=None)
    cpu_stage(root,'closed_metrics_%d'%step,[__file__,'--root',root,'--worker','evaluation','--step',step])


def evaluation_metrics(root,step):
    from stop_tail_small3 import evaluation_metrics as original
    original(root,step,16040)
    folder=root/'pilot/evaluation'/('step_%06d'%step)/'seed_16040';p=folder/'result.json';r=json.loads(p.read_text())
    rows=[json.loads(x) for x in (folder/'execution.jsonl').read_text().splitlines()]
    if not all(rows[w['center']]['extra']['dp_replanned'] for w in json.loads((root/'plan.json').read_text())['windows']):raise ValueError('Selected causal queue clocks changed')
    cfg=json.loads((folder/'config.json').read_text());first=r['first_pass_frame']
    span=(len(rows)-1-first)*cfg['dt'] if first is not None else None
    r['post_pass_real_span_sec']=span
    if not r['pass_done']:r.update(stopped=None,stop_evaluable=False,stop_status='not_evaluable_no_pass')
    elif not r['stopped'] and (len(rows)-first)*cfg['dt']<2.-1e-6:
        r.update(stop_status='insufficient_stop_hold_evidence',success=False)
    else:r['stop_status']='passed' if r['stopped'] else 'failed'
    save(p,r)


def recover(root):
    plan=json.loads((root/'plan.json').read_text());changed={n:sha256(REPO/n) for n,h in plan['source_hashes'].items() if sha256(REPO/n)!=h}
    trained=(root/'training/metrics.jsonl').exists()
    allowed={str((HERE/n).relative_to(REPO)) for n in ['smallbatch30k_analysis.py','smallbatch30k_online.py']}
    if trained and set(changed)-allowed:raise ValueError('Training already started; core/orchestration changes prohibited')
    if changed:
        parent=root/'source_revisions';parent.mkdir(exist_ok=True);rev=parent/('%04d'%(len(list(parent.iterdir()))+1));rev.mkdir()
        save(rev/'prior_plan.json',plan)
        for name in changed:
            target=rev/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,target)
        save(rev/'manifest.json',dict(changed=changed,training_started=trained,scope='Authorized engineering-only repair',at=time.time()))
        plan['source_hashes'].update(changed);save(root/'plan.json',plan)
    old=root/'failure_attempts';old.mkdir(exist_ok=True)
    for name in ['failure.json','completion.json','state.json']:
        p=root/name
        if p.exists():p.rename(old/(str(time.time_ns())+'_'+name))
    verify(root)


def run(root,repair=False):
    if root.exists() and not repair:raise FileExistsError('Independent new run required')
    if repair:recover(root)
    else:root.mkdir(parents=True);(root/'logs').mkdir()
    logger=None;log=None
    try:
        completed_stage(root,'prepare',lambda _:prepare(root))
        with phase_lock(root,'smallbatch30k'):
            cpu_stage(root,'initialization_tests',['-m','unittest','discover','-s',REPO/'high-level/dp','-p','test_random_initialization.py','-v'])
            cpu_stage(root,'convert',[REPO/'high-level/dp/convert_door_raw_to_lerobot.py','--raw_root',root/'raw',
                '--root',root/'data/lerobot','--repo_id','local/single_B16040'])
            cpu_stage(root,'data_audit',[__file__,'--root',root,'--worker','data'])
            gpu_stage(root,'random_initialization',train_command(root,1000,initialization=True),environment='train')
            cpu_stage(root,'audit_0',[__file__,'--root',root,'--worker','audit','--step','0'])
            log=(root/'logs'/('online_'+str(time.time_ns())+'.log')).open('x')
            logger=subprocess.Popen([str(REPO/'.repro/envs/door-dp-train/bin/python'),'-u',str(HERE/'smallbatch30k_online.py'),'--root',str(root)],
                stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            deadline=time.time()+180
            while not (root/'online/ready.json').exists():
                if logger.poll() is not None or time.time()>deadline:raise RuntimeError('Online preflight failed')
                time.sleep(3)
            save(root/'state.json',dict(phase='ready_for_online_link_delivery',pid=os.getpid(),
                url=json.loads((root/'online/ready.json').read_text())['url']))
            while not (root/'online/link_delivered.json').exists():
                if logger.poll() is not None:raise RuntimeError('Online logger exited before link delivery')
                time.sleep(2)
            verify(root)
            if not (root/'training_source_freeze.json').exists():save(root/'training_source_freeze.json',dict(at=time.time(),source_hashes=json.loads((root/'plan.json').read_text())['source_hashes'],preparation_passed=True))
            for step in NODES:
                verify(root);save(root/'state.json',dict(phase='training',target_updates=step,max_updates=30000,pid=os.getpid()))
                gpu_stage(root,'train_%d'%step,train_command(root,step),environment='train')
                cpu_stage(root,'audit_%d'%step,[__file__,'--root',root,'--worker','audit','--step',step])
                gpu_stage(root,'offline_%d'%step,['python','-u',str(HERE/'smallbatch30k_analysis.py'),'--root',root,'--step',str(step),'--offline'],environment='gym')
                evaluate(root,step)
                cpu_stage(root,'comparison_%d'%step,[HERE/'smallbatch30k_analysis.py','--root',root,'--step',step,'--closed'])
                if step==1000 and not (root/'runtime_estimate_1k.json').exists():
                    progress=json.loads((root/'training/progress.json').read_text());per=progress['elapsed_sec']/1000
                    save(root/'runtime_estimate_1k.json',dict(seconds_per_update=per,remaining_optimization_seconds=29000*per,
                        remaining_planned_rollouts=5,basis='Measured batch8 first1000 optimization; estimate only, no wall-time training cutoff'))
            verify(root);save(root/'final_input_preservation.json',shared.verify_inputs(json.loads((root/'plan.json').read_text())))
            save(root/'completion.json',dict(phase='completed',optimizer_updates=30000,planned_evaluations=6,no_heldout=True,auto_expand=False))
            logger.wait(timeout=120)
            if logger.returncode:raise RuntimeError('Online final verification failed')
            save(root/'state.json',dict(phase='completed',optimizer_updates=30000))
    except BaseException as exc:
        save(root/'failure.json',dict(error=repr(exc),at=time.time(),auto_retry=False));save(root/'completion.json',dict(phase='failed_stopped',error=repr(exc)))
        save(root/'state.json',dict(phase='failed_stopped',error=repr(exc)));raise
    finally:
        if logger and logger.poll() is None:
            try:logger.wait(timeout=30)
            except subprocess.TimeoutExpired:logger.terminate();logger.wait(timeout=15)
        if log:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--worker',choices=['data','audit','evaluation']);p.add_argument('--step',type=int,default=0);p.add_argument('--continue-repair',action='store_true');a=p.parse_args();r=Path(a.root).resolve()
    if a.worker=='data':data_audit(r)
    elif a.worker=='audit':audit_checkpoint(r,a.step)
    elif a.worker=='evaluation':evaluation_metrics(r,a.step)
    else:run(r,a.continue_repair)
