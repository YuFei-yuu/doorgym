"""Authorized completion of two experts and a 250/500/1000-update stop pilot."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO/'high-level/dp'))
import collect_stop_tail as shared
import small_forward_v3 as pilot
from artifacts import finite_json
from diagnostic_integrity import sha256, source_paths, verify_frozen, phase_lock
from forward_results import episode_result, aggregate
from stop_data_checks import stop_acceptance, measured_motion
from stop_audit import inspect_video

PAIR = REPO/'experiments/float-push/20261014-stop-tail-freeze-ab-v1'
ACCEPTED = REPO/'experiments/float-push/20261014-stop-tail-offline-16040-v1'
OLD_SMALL = REPO/'experiments/float-push/20261014-forward-v3-small3'
SEEDS, STEPS = [16040, 16041, 16042], [250, 500, 1000]


def save(path, value):
    shared.save(path, finite_json(value))


def state(root, phase, **extra):
    value = dict(phase=phase, at=time.time(), auto_retry=False, holdout_test_used=False,
        training_started=(root/'pilot/training').exists() or phase == 'training', **extra)
    save(root/'state.json', value); print(json.dumps(finite_json(value)), flush=True)


def source(root, seed):
    return root/'reference/seed_16040' if seed == 16040 else root/'episodes'/('seed_%d' % seed)


def initialize(root):
    if not json.loads((PAIR/'acceptance.json').read_text())['passed']:
        raise ValueError('A/B replay gate has not passed')
    hashes, directories = {}, {}
    for prior in [PAIR, ACCEPTED]:
        manifest = json.loads((prior/'delivery_sha256.json').read_text())
        if not all(sha256(prior/name) == value for name, value in manifest.items()):
            raise ValueError('Input delivery changed: '+str(prior))
        shared.register_input(prior, hashes, directories)
    for path in (OLD_SMALL/'training/checkpoints').glob('step_*.pt'):
        shared.register_input(path, hashes, directories)
    for name in ['plan.json', 'state.json', 'completion.json', 'reference']:
        shared.register_input(OLD_SMALL/name, hashes, directories)
    old_plan = json.loads((OLD_SMALL/'plan.json').read_text())
    old_root = REPO/'experiments/float-push/20261013-forward-dp-v1'
    for name, expected in old_plan['protected_original_hashes'].items():
        if sha256(old_root/name) != expected: raise ValueError('Protected original changed: '+name)
        shared.register_input(old_root/name, hashes, directories)
    shutil.copytree(ACCEPTED/'inputs/seed_16040', root/'reference/seed_16040')
    for seed in [16041, 16042]:
        target = root/'reference'/('seed_%d' % seed); target.mkdir(parents=True)
        for name in ['config.json', 'summary.json']:
            shutil.copy2(OLD_SMALL/'reference'/('seed_%d' % seed)/name, target/name)
    cfg = json.loads((root/'reference/seed_16040/config.json').read_text())
    paths = set(source_paths(REPO, cfg['source_hashes']))
    for name in shared.git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0'):
        path = REPO/name
        if name.startswith(('high-level/', 'low-level/')) and path.suffix in ('.py', '.sh', '.yaml', '.yml', '.json', '.urdf') and path.is_file(): paths.add(path)
    sources = {}
    for path in sorted(paths):
        name = str(path.relative_to(REPO)); sources[name] = sha256(path)
        target = root/'sources'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    save(root/'plan.json', dict(source_hashes=sources, input_hashes=hashes, input_directories=directories,
        reuse_seed=16040, collect_seeds=[16041,16042], train_seeds=SEEDS, evaluation_seeds=SEEDS,
        planned={str(seed): dict(planned_frames=json.loads((root/'reference'/('seed_%d/summary.json' % seed)).read_text())['frames']+(0 if seed == 16040 else 150)) for seed in SEEDS},
        max_updates=1000, checkpoints=STEPS, batch_size=64, num_workers=0, training_seed=42,
        random_initialization=True, action_contract='door_action_minmax_continuous_quat_v3', horizons=[16,32,16],
        evaluation_steps=2500, continue_after_ordinary_task_failure=True, simulation_timeout_seconds=600,
        force_stop_override=False, holdout_test_used=False, auto_expand=False,
        data_change='Real 3-second post-pass recordings only; forward_v3 preprocessing and all model/control contracts unchanged',
        replay_gate=str(PAIR), server_head=shared.git('rev-parse','HEAD').decode().strip(), created=time.time()))
    (root/'server_initial_status.txt').write_bytes(shared.git('status','--short','--branch'))
    (root/'server_worktree.diff').write_bytes(shared.git('diff','--binary','HEAD'))


def verify(root, training=False):
    plan = verify_frozen(root, REPO)
    if not shared.verify_inputs(plan)['passed']: raise ValueError('Protected input changed')
    if training:
        p = json.loads((root/'pilot/plan.json').read_text())
        if not shared.verify_inputs(p)['passed']: raise ValueError('Frozen pilot dataset changed')


def expert_command(root, seed):
    cfg = json.loads((root/'reference'/('seed_%d/config.json' % seed)).read_text())['args']
    cmd = ['python','-u',str(HERE/'wholebody_float.py'),'--headless','--no_show_seg','--camera_seg','--camera_depth',
        '--float_video','--float_seed',str(seed),'--float_expert_post_pass_seconds','3','--steps','2500',
        '--float_output',str(root/'episodes'/('seed_%d' % seed))]
    for key in ['robot_x','robot_y','robot_z','robot_yaw','stop_distance','push_base_vx','lever_step_size','float_execution','float_pass_mode','float_push_target']:
        cmd += ['--'+key,str(cfg[key])]
    return cmd


def prepare_training(root):
    target = root/'pilot'; target.mkdir()
    raw = target/'data/raw'; raw.mkdir(parents=True)
    frames, sequences, manifest = 0, 0, []
    for i, seed in enumerate(SEEDS):
        src = source(root, seed); record = root/'episodes'/('seed_%d' % seed)
        for name in ['data_acceptance.json','video_verification.json','window_acceptance.json']:
            if not json.loads((record/name).read_text())['passed']: raise ValueError('Three-episode data gate failed')
        summary = json.loads((src/'summary.json').read_text()); n = summary['frames']
        frames += n; sequences += n-31
        ref = target/'reference'/('seed_%d' % seed); ref.mkdir(parents=True)
        for name in ['config.json','summary.json','execution.jsonl','trajectory.npz','rollout.mp4']:
            shutil.copy2(src/name, ref/name)
        shutil.copy2(src/'raw/episode_000000.npz', raw/('episode_%06d.npz' % i))
        shutil.copy2(src/'raw/door_dp_feature_names.json', raw/'door_dp_feature_names.json')
        manifest.append(dict(seed=seed, frames=n, windows=n-31, raw_sha256=sha256(raw/('episode_%06d.npz' % i)), source=str(src)))
    shared.cpu(root, 'combined_conversion', [REPO/'high-level/dp/convert_door_raw_to_lerobot.py', '--raw_root',raw,
        '--root',target/'data/lerobot','--repo_id','local/small_forward_v3'])
    from train_door_dp import DoorDPSequenceDataset, _as_chw_u8
    from action_contract import FORWARD_V3, canonicalize_sequence
    combined = DoorDPSequenceDataset(target/'data/lerobot/local/small_forward_v3','local/small_forward_v3',16,32,FORWARD_V3)
    if len(combined) != sequences or combined.length != frames or set(combined.episodes) != {0,1,2}:
        raise ValueError('Combined training conversion/window boundaries mismatch')
    # All per-episode tensors were actually read and checked before this merge.
    # Verify every combined frame and every window's episode boundary as well.
    offset, oracles = 0, []
    for i, seed in enumerate(SEEDS):
        with np.load(raw/('episode_%06d.npz' % i),allow_pickle=True) as z:
            for j in range(len(z['action'])):
                frame = combined.dataset[offset+j]
                if not np.array_equal(frame['action'].numpy(),z['action'][j]) or not np.array_equal(frame['observation.state'].numpy(),z['state'][j]) or int(frame['episode_index']) != i:
                    raise ValueError('Combined dataset frame mismatch')
                for key in shared.IMAGES:
                    if not np.array_equal(_as_chw_u8(frame['observation.images.'+key]).numpy(),z[key][j].transpose(2,0,1)):
                        raise ValueError('Combined dataset image mismatch')
            oracle = {key:z[key].copy() for key in shared.IMAGES}
            oracle.update(state=canonicalize_sequence(z['state'],slice(-5,-1)),action=canonicalize_sequence(z['action'],slice(5,9)),start=offset)
            oracles.append(oracle)
        offset += manifest[i]['frames']
    if not all(combined.episodes[c] == combined.episodes[c+31] for c in combined.indices):
        raise ValueError('Combined future window crosses episodes')
    image_map = dict(mask='wrist_handle_mask',masked_depth='wrist_masked_depth',front_mask='front_handle_mask',front_masked_depth='front_masked_depth')
    for index, center in enumerate(combined.indices):
        item = combined[index]; oracle = oracles[combined.episodes[center]]; local = center-oracle['start']
        history = np.maximum(np.arange(local-15,local+1),0)
        exact = np.array_equal(item['state'].numpy(),oracle['state'][history]) and np.array_equal(item['action'].numpy(),oracle['action'][local:local+32])
        for key, original in image_map.items():
            exact &= np.array_equal(item[key].numpy(),oracle[original][history].transpose(0,3,1,2))
        if not exact: raise ValueError('Combined actual window mismatch at center %d' % center)
        if index % 200 == 0: print(json.dumps(dict(combined_windows_read=index+1,total_windows=sequences)),flush=True)
    save(root/'combined_window_verification.json',dict(passed=True,frames_actually_read=frames,windows_actually_read=sequences,
        state_images_actions_exact=True,no_cross_episode=True,training_frame_cache_enabled=False))
    inputs, directories = {}, {}
    for path in [target/'data',target/'reference']:
        shared.register_input(path,inputs,directories)
    cfg = json.loads((target/'reference/seed_16040/config.json').read_text())
    save(target/'plan.json',dict(source_hashes=json.loads((root/'plan.json').read_text())['source_hashes'],
        input_hashes=inputs,input_directories=directories,train_frames=frames,train_sequences=sequences,
        train_seeds=SEEDS,evaluation_seeds=SEEDS,max_updates=1000,checkpoints=STEPS,batch_size=64,num_workers=0,
        training_seed=42,random_initialization=True,contract=FORWARD_V3,obs_horizon=16,pred_horizon=32,action_horizon=16,
        source_snapshot_parent=str(root/'sources'),holdout_test_used=False,auto_expand=False))
    save(target/'manifest.json',manifest)
    save(target/'recording_contract.json',dict(schema=cfg['schema'],execution_profile=cfg['execution_profile'],
        tensor_pipeline=cfg['tensor_pipeline'],fps=50,source_expert_seeds=SEEDS,preprocessing_version=FORWARD_V3,
        recording_change='Real post-pass 3-second expert tail; no padding/mask/input/execution change'))
    save(root/'three_episode_data_gate.json',dict(passed=True,frames=frames,windows=sequences,seeds=SEEDS,manifest=manifest))


def evaluation_metrics(root, step, seed):
    target = root/'pilot'; folder = target/'evaluation'/('step_%06d' % step)/('seed_%d' % seed)
    cfg = json.loads((folder/'config.json').read_text()); summary = json.loads((folder/'summary.json').read_text())
    rows = [json.loads(line) for line in (folder/'execution.jsonl').read_text().splitlines()]
    result = episode_result(folder,dict(seed=seed,robot_x=4.1,robot_y=0.,checkpoint_updates=step),0,policy=True)
    stopping = stop_acceptance(rows,cfg['dt'],cfg['door_runtime']['door_motion_sign'],cfg['args']['pass_align_pos_tol'],
        expected_source='dp',required_post_pass_seconds=0.)
    with np.load(folder/'trajectory.npz',allow_pickle=True) as z:
        finite = all(np.isfinite(z[key]).all() for key in ['state','action','next_state'])
    video = inspect_video(folder/'rollout.mp4',1250); save(folder/'video_verification.json',video)
    execution_ok = (len(rows) == 2500 and result['complete_log'] and result['autonomous'] and finite and
        summary['error'] is None and video['passed'] and
        all(not row['freeze_arm_default'] and not row['freeze_arm_zero'] for row in rows))
    park_keys = ['settles_within_half_second','two_second_hold_complete','maximum_two_second_drift','extra_path_until_stopped','no_later_movement']
    stopped = all(stopping['checks'].get(key,False) for key in park_keys)
    strict_pass = result['pass_done'] and result['pass_event_distance_m'] <= cfg['args']['pass_align_pos_tol']
    result.update(execution_ok=bool(execution_ok),finite_state_action=bool(finite),pass_done=bool(strict_pass),
        stopped=bool(stopped),stop_evaluable=stopping['first_pass_step'] is not None,
        opening_and_pass_success=result['success'],success=bool(result['opened'] and strict_pass and stopped and execution_ok),
        stopping=stopping,termination_reason='max_steps' if len(rows) == 2500 else 'incomplete',
        complete_video=video['passed'],DP_stop_override_used=False,all_freeze_flags_cleared=True)
    save(folder/'result.json',result)
    print(json.dumps(finite_json(dict(step=step,seed=seed,opened=result['opened'],passed=result['pass_done'],stopped=stopped,joint=result['success'],execution_ok=execution_ok))),flush=True)
    if not execution_ok: raise RuntimeError('DP evaluation execution anomaly at %d/%d' % (step,seed))


def evaluate(root, step):
    target = root/'pilot'; suite = target/'evaluation'/('step_%06d' % step); suite.mkdir(parents=True)
    results = []
    for seed in SEEDS:
        verify(root,training=True)
        cfg = json.loads((target/'reference'/('seed_%d/config.json' % seed)).read_text())['args']
        cmd = ['python','-u',str(HERE/'wholebody_float.py'),'--headless','--no_show_seg','--camera_seg','--camera_depth',
            '--float_video','--float_seed',str(seed),'--steps','2500','--float_output',str(suite/('seed_%d' % seed)),
            '--dp_policy_checkpoint',str(target/'training/checkpoints'/('step_%06d.pt' % step)),
            '--dp_inference_steps','100','--dp_action_horizon','16']
        for key in ['robot_x','robot_y','robot_z','robot_yaw','stop_distance','push_base_vx','lever_step_size','float_execution','float_pass_mode','float_push_target']:
            cmd += ['--'+key,str(cfg[key])]
        code = shared.launch(target,'evaluation_%06d_%d' % (step,seed),cmd,timeout=600,total=None,environment='gym')
        if code: raise RuntimeError('Evaluation process exit %d' % code)
        shared.cpu(root,'evaluation_metrics_%06d_%d' % (step,seed),[__file__,'--root',root,'--worker','evaluation','--step',step,'--seed',seed])
        results.append(json.loads((suite/('seed_%d/result.json' % seed)).read_text())); save(suite/'attempts.json',results)
    save(suite/'results.json',results)
    summary = aggregate(results,3); summary.update(stopped=sum(row['stopped'] for row in results),stop_evaluable=sum(row['stop_evaluable'] for row in results))
    save(suite/'aggregate.json',summary)


def report(root):
    from forward_report import write_csv, rollout_plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target = root/'pilot'; output = root/'analysis'; output.mkdir(exist_ok=True)
    metrics_path = target/'training/metrics.jsonl'
    if metrics_path.exists():
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        write_csv(target/'training/metrics.csv',metrics)
        fig, axes = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for axis,key,label in zip(axes,['train/loss','train/lr','train/elapsed_sec'],['Noise MSE','Learning rate','Optimization seconds']):
            axis.plot([r['train/step'] for r in metrics],[r[key] for r in metrics]); axis.set(xlabel='Updates',ylabel=label); axis.grid(alpha=.2)
        fig.savefig(output/'training_curves.png',dpi=140); plt.close(fig)
    summaries, all_results = [], []
    for step in STEPS:
        suite = target/'evaluation'/('step_%06d' % step)
        if not (suite/'results.json').exists(): continue
        values = json.loads((suite/'results.json').read_text()); all_results.extend(values)
        summaries.append(dict(step=step,**json.loads((suite/'aggregate.json').read_text())))
        for value in values:
            folder = Path(value['folder'])
            if (folder/'stopping_motion.png').exists(): continue
            rollout_plots(folder)
            rows = [json.loads(line) for line in (folder/'execution.jsonl').read_text().splitlines()]
            dt = json.loads((folder/'config.json').read_text())['dt']
            _,_,_,speed,omega = measured_motion(rows,dt)
            t = np.arange(len(rows))*dt
            fig, axes = plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
            for axis,signal,threshold,label in zip(axes,[speed,omega],[.01,.02],['Measured translation speed (m/s)','Measured angular speed (rad/s)']):
                axis.plot(t,signal);axis.axhline(threshold,color='red',linestyle='--');axis.set_ylabel(label);axis.grid(alpha=.2)
                if value['first_pass_frame'] is not None: axis.axvline(value['first_pass_frame']*dt,color='green',linestyle=':')
            axes[-1].set_xlabel('Simulation seconds');fig.savefig(folder/'stopping_motion.png',dpi=140);plt.close(fig)
    write_csv(output/'checkpoint_success.csv',summaries)
    write_csv(output/'all_results.csv',[{k:v for k,v in row.items() if k != 'stopping'} for row in all_results])
    lines=['# Three-training-state stopping pilot','', 'Random initialization seed42, forward_v3, 16/32/16, batch64 workers0. No final held-out test. Ordinary task failures never stop the authorized 1000 updates.', '', '| Updates | Opened | Passed | Stopped after pass | Joint | Attempts |','|---:|---:|---:|---:|---:|---:|']
    for row in summaries: lines.append('|%d|%d|%d|%d|%d|%d|' % (row['step'],row['opened'],row['passed'],row['stopped'],row['successes'],row['attempts']))
    lines += ['', 'Stopping uses measured motion and unchanged thresholds after the original pass event. All 2500 control steps are recorded. Commands are DP outputs with cleared freeze flags; no terminal zero-speed override.', 'See per-episode result.json/stopping_motion.png/rollout.mp4, data gates, checkpoint audits, source snapshots and hashes. This is fitting of three training initial states, not generalization or final stability acceptance.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def finish(root,error=None):
    preserved = shared.verify_inputs(json.loads((root/'plan.json').read_text())) if (root/'plan.json').exists() else dict(passed=False)
    save(root/'preservation.json',preserved)
    target = root/'pilot'; completed = json.loads((target/'training/progress.json').read_text())['step'] if (target/'training/progress.json').exists() else 0
    summaries = [dict(step=step,**json.loads((target/'evaluation'/('step_%06d/aggregate.json' % step)).read_text())) for step in STEPS if (target/'evaluation'/('step_%06d/aggregate.json' % step)).exists()]
    if error is None:
        verify(root,training=True)
        if completed != 1000 or len(list((target/'training/checkpoints').glob('step_*.pt'))) != 3 or len(summaries) != 3:
            raise ValueError('Authorized completion/checkpoint schedule mismatch')
    save(root/'completion.json',dict(phase='completed_1000_update_pilot' if error is None else 'failed_stopped',
        error=error,completed_updates=completed,checkpoint_summaries=summaries,preservation_passed=preserved['passed'],
        auto_expand=False,holdout_test_used=False,final_stability_acceptance=False))
    state(root,'completed_1000_update_pilot' if error is None else 'failed_stopped',error=error,completed_updates=completed)
    save(root/'delivery_sha256.json',{str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ['delivery_sha256.json','supervisor.log','diagnostics.lock']})


def run(root):
    if any((root/name).exists() for name in ['plan.json','state.json','sources']): raise FileExistsError('Use a new independent run')
    root.mkdir(parents=True,exist_ok=True)
    try:
        initialize(root)
        with phase_lock(root,'stop_tail_small3'):
            shared.cpu(root,'training_cap_tests',['-m','unittest','discover','-s',HERE,'-p','test_small_forward_v3.py','-v'])
            for seed in SEEDS:
                verify(root)
                if seed != 16040:
                    state(root,'collecting_tail',seed=seed)
                    code = shared.launch(root,'expert_%d' % seed,expert_command(root,seed),timeout=600,total=None,environment='gym')
                    if code: raise RuntimeError('Expert collection exit %d' % code)
                shared.cpu(root,'data_%d' % seed,[__file__,'--root',root,'--worker','data','--seed',seed])
                shared.cpu(root,'conversion_%d' % seed,[REPO/'high-level/dp/convert_door_raw_to_lerobot.py','--raw_root',source(root,seed)/'raw',
                    '--root',root/'episodes'/('seed_%d/lerobot' % seed),'--repo_id','local/stop_tail'])
                shared.cpu(root,'windows_%d' % seed,[__file__,'--root',root,'--worker','windows','--seed',seed])
            state(root,'preparing_three_episode_training')
            shared.cpu(root,'prepare_training',[__file__,'--root',root,'--worker','prepare'])
            previous = None
            for step in STEPS:
                verify(root,training=True); state(root,'training',target_updates=step,maximum_updates=1000)
                code = shared.launch(root/'pilot','training_%06d' % step,pilot.train_command(root/'pilot',step,previous),timeout=None,total=None,environment='train')
                verify(root,training=True)
                if code: raise RuntimeError('Training exit %d' % code)
                shared.cpu(root,'checkpoint_%06d' % step,[__file__,'--root',root,'--worker','checkpoint','--step',step])
                state(root,'evaluating',completed_updates=step)
                evaluate(root,step)
                shared.cpu(root,'report_%06d' % step,[__file__,'--root',root,'--worker','report'])
                previous = step
            finish(root)
    except BaseException as exc:
        save(root/'failure.json',dict(error=repr(exc),auto_retry=False))
        finish(root,repr(exc));raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--worker',choices=['data','windows','prepare','checkpoint','evaluation','report'])
    parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--step',type=int,choices=STEPS)
    args = parser.parse_args();root = Path(args.root).resolve()
    if args.worker == 'data': shared.audit_episode(root,args.seed,source_folder=source(root,args.seed),output_folder=root/'episodes'/('seed_%d' % args.seed))
    elif args.worker == 'windows': shared.audit_windows(root,args.seed,source_folder=source(root,args.seed))
    elif args.worker == 'prepare': prepare_training(root)
    elif args.worker == 'checkpoint': pilot.audit_checkpoint(root/'pilot',args.step)
    elif args.worker == 'evaluation': evaluation_metrics(root,args.step,args.seed)
    elif args.worker == 'report': report(root)
    else: run(root)
