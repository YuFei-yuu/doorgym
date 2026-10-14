"""New authorized1000->5000 continuation. Old pilot limits and outputs stay read-only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'high-level/dp'))
import collect_stop_tail as shared
import stop_tail_small3 as prior
from artifacts import finite_json
from diagnostic_integrity import sha256,source_paths,verify_frozen,phase_lock
from forward_results import aggregate
from stop_tail_diagnostics import select_windows
from stop_tail_online import metric_rows,smoothed

OLD=REPO/'experiments/float-push/20261014-stop-tail-small3-v1'
NODES=[2000,3000,5000];SEEDS=[16040,16041,16042]


def canonical_scheduler(value):
    result=dict(value)
    if '_use_default_values' in result:result['_use_default_values']=sorted(result['_use_default_values'])
    return result


def save(path,value):shared.save(path,finite_json(value))


def state(root,phase,**extra):
    value=dict(phase=phase,pid=os.getpid(),at=time.time(),max_updates=5000,new_updates=4000,holdout_test_used=False,auto_retry=False,**extra)
    save(root/'state.json',value);print(json.dumps(finite_json(value)),flush=True)


def initialize(root):
    import torch
    from training_state import ResumableBatchSampler
    checkpoint=OLD/'pilot/training/checkpoints/step_001000.pt'
    if sha256(checkpoint)!='b7c49cce0c350f743bc08fe01a6732830951bf1eb765c6f2d188cc04dd3f79ec':raise ValueError('Starting model hash changed')
    manifest=json.loads((OLD/'delivery_sha256.json').read_text())
    if not all(sha256(OLD/name)==value for name,value in manifest.items()):raise ValueError('Old pilot evidence changed')
    c=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=c['config']
    if cfg['completed_updates']!=1000 or cfg['dataset_frames']!=4321 or cfg['dataset_sequences']!=4228:raise ValueError('Wrong baseline model/data')
    s=c['training_state'];sampler=ResumableBatchSampler(4228,64,43);sampler.load_state_dict(s['sampler'])
    if set(s['rng'])!={'python','numpy','torch','cuda'} or len(s['rng']['cuda'])!=1:raise ValueError('Incomplete single-GPU RNG state')
    optimizer=c['optimizer']
    if not optimizer['state'] or any(int(value['step'])!=1000 for value in optimizer['state'].values()):raise ValueError('Incomplete optimizer update state')
    for value in list(c['model'].values())+list(c['stats'].values())+[v for item in optimizer['state'].values() for v in item.values() if isinstance(v,torch.Tensor)]:
        if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('Nonfinite starting state')
    old_data_plan=json.loads((OLD/'pilot/plan.json').read_text())
    if not shared.verify_inputs(old_data_plan)['passed']:raise ValueError('Original training dataset changed')
    inputs,dirs={},{};shared.register_input(OLD,inputs,dirs)
    sources={}
    paths=set(source_paths(REPO,json.loads((OLD/'pilot/reference/seed_16040/config.json').read_text())['source_hashes']))
    for name in shared.git('ls-files','-z','--cached','--others','--exclude-standard').decode().split('\0'):
        p=REPO/name
        if name.startswith(('high-level/','low-level/')) and p.suffix in ('.py','.sh','.yaml','.yml','.json','.urdf') and p.is_file():paths.add(p)
    for p in sorted(paths):
        name=str(p.relative_to(REPO));sources[name]=sha256(p);target=root/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    target=root/'pilot';target.mkdir()
    shutil.copytree(OLD/'pilot/reference',target/'reference')
    save(root/'recording_contract.json',cfg['recording_metadata'])
    save(root/'baseline_config.json',cfg)
    save(root/'plan.json',dict(source_hashes=sources,input_hashes=inputs,input_directories=dirs,
        starting_checkpoint=str(checkpoint),starting_sha256=sha256(checkpoint),resume_update=1000,max_updates=5000,new_updates=4000,nodes=NODES,
        train_frames=4321,train_sequences=4228,batch_size=64,num_workers=0,lr=1e-4,seed=42,horizons=[16,32,16],DDPM_steps=100,
        original_dataset_path=cfg['root'],complete_state_resume=True,nonfinite_guard=True,old_limit_unchanged=True,
        server_head=shared.git('rev-parse','HEAD').decode().strip(),source_commit='0acc2f050479fb54a0670ccb90bc72b98b52a54a',created=time.time()))
    save(target/'plan.json',dict(source_hashes=sources,input_hashes=old_data_plan['input_hashes'],input_directories=old_data_plan['input_directories']))
    diagnostics=[]
    for seed in SEEDS:
        folder=OLD/'pilot/reference'/('seed_%d'%seed)
        expert=[json.loads(line) for line in (folder/'execution.jsonl').read_text().splitlines()]
        p=next(i for i,row in enumerate(expert) if row['extra']['pass_done'])
        actual=[json.loads(line) for line in (OLD/'pilot/evaluation/step_001000'/('seed_%d'%seed)/'execution.jsonl').read_text().splitlines()]
        replans=[row['step'] for row in actual if row['extra']['dp_replanned']]
        diagnostics.append(dict(seed=seed,trajectory=str(folder/'trajectory.npz'),first_expert_stop=p,
            actual_replan_source=str(OLD/'pilot/evaluation/step_001000'/('seed_%d'%seed)/'execution.jsonl'),
            windows=select_windows(p,replans,len(expert)-32)))
    save(root/'diagnostic_plan.json',dict(baseline_checkpoint=str(checkpoint),diffusion_seeds=[4201,4202,4203],episodes=diagnostics,
        input_policy='Streaming expert observations from episode start through selected replan; no later observation used'))
    save(root/'resume_state_audit.json',dict(passed=True,optimizer_states=len(optimizer['state']),sampler_cursor=sampler.cursor,
        sampler_epoch=sampler.epoch,cuda_rng_states=1,prior_optimization_seconds=s['elapsed_sec'],cuda_initialized=torch.cuda.is_initialized()))
    (root/'server_initial_status.txt').write_bytes(shared.git('status','--short','--branch'))
    (root/'server_worktree.diff').write_bytes(shared.git('diff','--binary','HEAD'))


def verify(root):
    p=verify_frozen(root,REPO)
    if not shared.verify_inputs(p)['passed']:raise ValueError('Old evidence/data changed')
    if not shared.verify_inputs(json.loads((root/'pilot/plan.json').read_text()))['passed']:raise ValueError('Training data changed')


def train_command(root,target,previous):
    if (target,previous) not in [(2000,1000),(3000,2000),(5000,3000)]:raise ValueError('Unauthorized resume schedule')
    cfg=json.loads((root/'baseline_config.json').read_text())
    checkpoint=Path(json.loads((root/'plan.json').read_text())['starting_checkpoint']) if previous==1000 else root/'pilot/training/checkpoints'/('step_%06d.pt'%previous)
    cmd=['python','-u',str(REPO/'high-level/dp/train_door_dp.py'),'--root',cfg['root'],'--repo_id',cfg['repo_id'],
        '--run_name',root.name,'--output_dir',str(root/'pilot/training'),'--steps',str(target),'--batch_size','64','--num_workers','0',
        '--lr','1e-4','--action_contract','forward_v3','--obs_horizon','16','--pred_horizon','32','--action_horizon','16',
        '--num_diffusion_iters','100','--seed','42','--log_interval','10','--save_interval',str(target),
        '--recording_metadata',str(root/'recording_contract.json'),'--resume',str(checkpoint),'--stop_on_nonfinite_loss']
    return cmd


def checkpoint_audit(root,step):
    import torch
    from training_state import ResumableBatchSampler
    file=root/'pilot/training/checkpoints'/('step_%06d.pt'%step);c=torch.load(file,map_location='cpu',weights_only=False)
    baseline=torch.load(json.loads((root/'plan.json').read_text())['starting_checkpoint'],map_location='cpu',weights_only=False)
    cfg=c['config'];previous={2000:1000,3000:2000,5000:3000}[step]
    expected=str(Path(json.loads((root/'plan.json').read_text())['starting_checkpoint'])) if previous==1000 else str(root/'pilot/training/checkpoints'/('step_%06d.pt'%previous))
    if cfg['completed_updates']!=step or cfg['resume']!=expected or not all(torch.equal(c['stats'][k],baseline['stats'][k]) for k in baseline['stats']):raise ValueError('Checkpoint/normalization mismatch')
    keys=['root','repo_id','batch_size','num_workers','lr','seed','obs_horizon','pred_horizon','action_horizon','num_diffusion_iters','dataset_frames','dataset_sequences','recording_metadata','action_normalization_version']
    if any(cfg.get(k)!=baseline['config'].get(k) for k in keys):raise ValueError('Checkpoint contract drift')
    if canonical_scheduler(cfg['diffusion_scheduler'])!=canonical_scheduler(baseline['config']['diffusion_scheduler']):raise ValueError('Effective scheduler contract drift')
    sampler=ResumableBatchSampler(4228,64,43);sampler.load_state_dict(c['training_state']['sampler'])
    if sampler.cursor!=(step%66)*64 or any(int(v['step'])!=step for v in c['optimizer']['state'].values()):raise ValueError('Incomplete resumed update state')
    save(root/'pilot'/('checkpoint_%06d.json'%step),dict(passed=True,step=step,sha256=sha256(file),resume=expected,stats_exact_to_1000=True,sampler_cursor=sampler.cursor,sampler_epoch=sampler.epoch))


def evaluation_metrics(root,step,seed):
    prior.evaluation_metrics(root,step,seed)
    folder=root/'pilot/evaluation'/('step_%06d'%step)/('seed_%d'%seed)
    path=folder/'result.json';r=json.loads(path.read_text())
    if not r['pass_done']:r.update(stopped=None,stop_evaluable=False,success=False,stop_status='not_evaluable_no_pass')
    else:r['stop_status']='passed' if r['stopped'] else 'failed'
    save(path,r)
    rows=[json.loads(line) for line in (folder/'execution.jsonl').read_text().splitlines()]
    episode=next(e for e in json.loads((root/'diagnostic_plan.json').read_text())['episodes'] if e['seed']==seed)
    if not all(rows[w['center']]['extra']['dp_replanned'] for w in episode['windows']):raise ValueError('Actual queue replan anchors changed')
    p=r['first_pass_frame'];queues=[]
    if p is not None:
        centers=[row['step'] for row in rows if row['extra']['dp_replanned']]
        before=max(c for c in centers if c<=p);after=next((c for c in centers if c>p),None)
        with np.load(folder/'trajectory.npz',allow_pickle=True) as z:
            for c in [before,after]:
                if c is None:continue
                queues.append(dict(generation_step=c,observation_cutoff=c,history_start=max(0,c-15),pass_offset=p-c,
                    actual_executed_vx_yaw=z['action'][c:min(c+16,len(rows)),:2].tolist()))
    save(folder/'queue_alignment.json',dict(first_pass=p,queues=queues,earlier_queue_not_explained_by_later_observation=True))


def evaluate(root,step):
    target=root/'pilot';suite=target/'evaluation'/('step_%06d'%step);suite.mkdir(parents=True)
    results=[]
    for seed in SEEDS:
        verify(root);cfg=json.loads((target/'reference'/('seed_%d/config.json'%seed)).read_text())['args']
        cmd=['python','-u',str(HERE/'wholebody_float.py'),'--headless','--no_show_seg','--camera_seg','--camera_depth','--float_video',
            '--float_seed',str(seed),'--steps','2500','--float_output',str(suite/('seed_%d'%seed)),
            '--dp_policy_checkpoint',str(target/'training/checkpoints'/('step_%06d.pt'%step)),'--dp_inference_steps','100','--dp_action_horizon','16']
        for key in ['robot_x','robot_y','robot_z','robot_yaw','stop_distance','push_base_vx','lever_step_size','float_execution','float_pass_mode','float_push_target']:cmd+=['--'+key,str(cfg[key])]
        code=shared.launch(target,'evaluation_%06d_%d'%(step,seed),cmd,timeout=600,total=None,environment='gym')
        if code:raise RuntimeError('Evaluation process exit %d'%code)
        shared.cpu(root,'evaluation_metrics_%06d_%d'%(step,seed),[__file__,'--root',root,'--worker','evaluation','--step',step,'--seed',seed])
        results.append(json.loads((suite/('seed_%d/result.json'%seed)).read_text()));save(suite/'attempts.json',results)
    save(suite/'results.json',results)
    a=aggregate(results,3);a.update(stopped=sum(bool(r['stopped']) for r in results),stop_evaluable=sum(r['pass_done'] for r in results));save(suite/'aggregate.json',a)


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from forward_report import rollout_plots,write_csv
    baseline=OLD/'pilot/training/metrics.jsonl';rows=smoothed(metric_rows([baseline,root/'pilot/training/metrics.jsonl']))
    out=root/'analysis';out.mkdir(exist_ok=True);write_csv(out/'training_metrics.csv',rows)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,key,label in zip(axes,['train/loss','train/lr','train/elapsed_sec'],['Noise MSE','Learning rate','Optimization seconds']):
        ax.plot([r['train/step'] for r in rows],[r[key] for r in rows]);ax.axvline(1000,color='purple',linestyle='--');ax.set(xlabel='Actual update',ylabel=label);ax.grid(alpha=.2)
    axes[0].plot([r['train/step'] for r in rows],[r['train/loss_ema'] for r in rows],label='EWMA alpha0.1');axes[0].legend()
    fig.savefig(out/'training_curves.png',dpi=140);plt.close(fig)
    summaries=[]
    for step in NODES:
        suite=root/'pilot/evaluation'/('step_%06d'%step)
        if not (suite/'aggregate.json').exists():continue
        a=json.loads((suite/'aggregate.json').read_text());summaries.append(dict(step=step,**a))
        for r in json.loads((suite/'results.json').read_text()):
            folder=Path(r['folder'])
            if not (folder/'door_and_path.png').exists():rollout_plots(folder)
    write_csv(out/'checkpoint_success.csv',summaries)
    lines=['# Stop-tail continuation1000 to5000','', 'Complete optimizer/sampler/RNG resume; original data and contracts retained. Three training initial states only.', '', '| Total update | Opened | Passed | Stop successes / evaluable | Joint |','|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append('|%d|%d/3|%d/3|%d/%d|%d/3|'%(r['step'],r['opened'],r['passed'],r['stopped'],r['stop_evaluable'],r['successes']))
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def finish(root,error=None):
    completed=json.loads((root/'pilot/training/progress.json').read_text())['step'] if (root/'pilot/training/progress.json').exists() else 1000
    if error is None:
        verify(root)
        if completed!=5000 or len(list((root/'pilot/training/checkpoints').glob('step_*.pt')))!=3:raise ValueError('Continuation cap/checkpoint mismatch')
    save(root/'completion.json',dict(phase='completed_5000_update_resume' if error is None else 'failed_stopped',completed_updates=completed,new_updates=completed-1000,error=error,auto_expand=False,heldout_test_used=False))
    state(root,'completed_5000_update_resume' if error is None else 'failed_stopped',completed_updates=completed,error=error)


def run(root):
    if any((root/name).exists() for name in ['plan.json','state.json','sources']):raise FileExistsError('New run required; old cap is not changed')
    if not (root/'online/ready.json').exists() or not json.loads((root/'online/ready.json').read_text())['passed']:raise RuntimeError('Online link must be verified before training')
    try:
        initialize(root)
        with phase_lock(root,'stop_tail_resume5000'):
            for stage,folder,pattern in [('resume_tests',REPO/'high-level/dp','test_training_state.py'),('finite_tests',REPO/'high-level/dp','test_finite_loss.py'),('monitor_diagnostic_tests',HERE,'test_stop_tail_resume.py')]:
                shared.cpu(root,stage,['-m','unittest','discover','-s',folder,'-p',pattern,'-v'])
            state(root,'baseline_diagnostic',completed_updates=1000)
            code=shared.launch(root,'diagnostic_001000',['python','-u',str(HERE/'stop_tail_diagnostics.py'),'--root',root,'--step','1000'],timeout=None,total=None,environment='gym')
            if code:raise RuntimeError('Baseline diagnostic failed')
            previous=1000
            for step in NODES:
                verify(root);state(root,'training',target_updates=step,completed_updates=previous)
                code=shared.launch(root/'pilot','training_%06d'%step,train_command(root,step,previous),timeout=None,total=None,environment='train')
                verify(root)
                if code:raise RuntimeError('Training exit %d'%code)
                shared.cpu(root,'checkpoint_%06d'%step,[__file__,'--root',root,'--worker','checkpoint','--step',step])
                state(root,'evaluating',completed_updates=step);evaluate(root,step)
                state(root,'parking_diagnostic',completed_updates=step)
                code=shared.launch(root,'diagnostic_%06d'%step,['python','-u',str(HERE/'stop_tail_diagnostics.py'),'--root',root,'--step',str(step)],timeout=None,total=None,environment='gym')
                if code:raise RuntimeError('Parking diagnostic failed')
                shared.cpu(root,'report_%06d'%step,[__file__,'--root',root,'--worker','report']);previous=step
            finish(root)
    except BaseException as exc:
        save(root/'failure.json',dict(error=repr(exc),auto_retry=False));finish(root,repr(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--worker',choices=['checkpoint','evaluation','report'])
    parser.add_argument('--step',type=int,choices=NODES);parser.add_argument('--seed',type=int,choices=SEEDS)
    a=parser.parse_args();root=Path(a.root).resolve()
    if a.worker=='checkpoint':checkpoint_audit(root,a.step)
    elif a.worker=='evaluation':evaluation_metrics(root,a.step,a.seed)
    elif a.worker=='report':report(root)
    else:run(root)
