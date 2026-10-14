"""Three fixed expert recordings and data acceptance. No training or replay entry."""
import argparse
import csv
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
sys.path.insert(0, str(HERE/'run_tools'))
sys.path.insert(0, str(REPO/'high-level/dp'))
from launch import launch
from bounded import atomic_json
from diagnostic_integrity import sha256, verify_frozen, phase_lock, source_paths
from stop_data_checks import stop_acceptance
from stop_audit import compare_actions, inspect_video

SEEDS = [16040, 16041, 16042]
OLD = REPO/'experiments/float-push/20261014-forward-v3-small3'
FAILED_RUN = REPO/'experiments/float-push/20261014-forward-stop-tail3s-v1'
IMAGES = ['wrist_handle_mask', 'wrist_masked_depth', 'front_handle_mask', 'front_masked_depth']


def save(path, value):
    atomic_json(path, value)


def state(root, phase, **extra):
    value = dict(phase=phase, at=time.time(), pid=os.getpid(), auto_retry=False,
                 training_started=False, holdout_test_used=False, **extra)
    save(root/'state.json', value)
    print(json.dumps(value), flush=True)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


def directory_entries(path):
    return sorted(str(p.relative_to(path)) + ('/' if p.is_dir() else '')
                  for p in Path(path).rglob('*'))


def register_input(path, hashes, directories):
    path = Path(path)
    if path.is_file():
        hashes[str(path)] = sha256(path)
    elif path.is_dir():
        directories[str(path)] = directory_entries(path)
        for child in sorted(path.rglob('*')):
            if child.is_file():
                hashes[str(child)] = sha256(child)
    else:
        raise FileNotFoundError('Missing required input: '+str(path))


def verify_inputs(plan):
    files = {name: Path(name).is_file() and sha256(name) == value
             for name, value in plan.get('input_hashes', {}).items()}
    directories = {name: Path(name).is_dir() and directory_entries(Path(name)) == entries
                   for name, entries in plan.get('input_directories', {}).items()}
    return dict(passed=bool(plan) and all(files.values()) and all(directories.values()),
                inputs=files, directories=directories)


def initialize(root):
    # The directory may contain only the read-only preparation captured before upload.
    if (root/'plan.json').exists():
        raise FileExistsError('This run has already been initialized; no automatic restart')
    root.mkdir(parents=True, exist_ok=True)
    paths = set(source_paths(REPO))
    for name in git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0'):
        path = REPO/name
        if name.startswith(('high-level/', 'low-level/')) and path.suffix in ('.py', '.sh', '.yaml', '.yml', '.json', '.urdf') and path.is_file():
            paths.add(path)
    hashes = {}
    for source in sorted(paths):
        name = str(source.relative_to(REPO))
        hashes[name] = sha256(source)
        target = root/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    inputs, directories, planned = {}, {}, {}
    for seed in SEEDS:
        folder = OLD/'reference'/('seed_%d' % seed)
        cfg = json.loads((folder/'config.json').read_text())
        summary = json.loads((folder/'summary.json').read_text())
        target = root/'reference'/('seed_%d' % seed)
        target.mkdir(parents=True)
        for name in ['config.json', 'summary.json']:
            shutil.copy2(folder/name, target/name)
        for path in folder.iterdir():
            if path.is_file(): register_input(path, inputs, directories)
        planned[str(seed)] = dict(old_frames=summary['frames'], planned_frames=summary['frames']+150,
                                 planned_windows=summary['frames']+150-31)
    for path in sorted((OLD/'training/checkpoints').glob('step_*.pt')):
        register_input(path, inputs, directories)
    previous = REPO/'experiments/float-push/20261013-forward-dp-v1'
    for name in ['pipeline_state.json', 'user_stop_3000', 'stop_after_updates.json', 'training/checkpoints/step_003000.pt']:
        register_input(previous/name, inputs, directories)
    for name in ['state.json', 'completion.json', 'plan.json']:
        register_input(OLD/name, inputs, directories)
    register_input(FAILED_RUN, inputs, directories)
    save(root/'plan.json', dict(source_hashes=hashes, input_hashes=inputs, input_directories=directories, seeds=SEEDS,
        source_commit='0acc2f050479fb54a0670ccb90bc72b98b52a54a', server_head=git('rev-parse', 'HEAD').decode().strip(),
        local_base_commit='5dd64f2', planned=planned, expert_post_pass_seconds=3., steps_limit=2500,
        action_contract='door_action_minmax_continuous_quat_v3', horizons=[16, 32, 16],
        new_raw_schema=False, changes='Expert recording end only; existing script control preserved',
        no_training=True, no_holdout=True, no_replay=True, no_tail_padding=True, no_mask=True,
        no_policy_stop_override=True, created=time.time()))
    (root/'server_worktree.diff').write_bytes(git('diff', '--binary', 'HEAD'))
    (root/'server_status.txt').write_bytes(git('status', '--short', '--branch'))
    (root/'server_untracked.txt').write_bytes(git('ls-files', '--others', '--exclude-standard'))
    state(root, 'initialized')


def cpu(root, stage, arguments):
    verify_frozen(root, REPO)
    prefix = REPO/'.repro/envs/door-dp-train'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        WANDB_MODE='disabled', XDG_CACHE_HOME=str(REPO/'.repro/cache/train'), LD_LIBRARY_PATH=str(prefix/'lib'))
    env.pop('PYTHONPATH', None)
    (root/'logs').mkdir(exist_ok=True)
    with (root/'logs'/(stage+'.log')).open('x') as output:
        code = subprocess.call(['nice', '-n', str(max(0, 10-os.getpriority(os.PRIO_PROCESS, 0))),
            str(prefix/'bin/python'), '-u', *map(str, arguments)], cwd=REPO, env=env,
            stdout=output, stderr=subprocess.STDOUT)
    verify_frozen(root, REPO)
    if code: raise RuntimeError('CPU stage %s failed: exit %d' % (stage, code))


def audit_episode(root, seed, source_folder=None, output_folder=None):
    folder = Path(source_folder) if source_folder is not None else root/'episodes'/('seed_%d' % seed)
    output = Path(output_folder) if output_folder is not None else folder
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((folder/'config.json').read_text())
    summary = json.loads((folder/'summary.json').read_text())
    rows = [json.loads(line) for line in (folder/'execution.jsonl').read_text().splitlines()]
    result = stop_acceptance(rows, cfg['dt'], cfg['door_runtime']['door_motion_sign'], cfg['args']['pass_align_pos_tol'])
    checks = result['checks']
    raw = folder/'raw/episode_000000.npz'
    with np.load(raw, allow_pickle=True) as z, np.load(folder/'trajectory.npz', allow_pickle=True) as trajectory:
        n = len(rows)
        checks['raw_shapes'] = z['state'].shape == (n, 27) and z['action'].shape == (n, 10)
        checks['raw_finite'] = bool(np.isfinite(z['state']).all() and np.isfinite(z['action']).all())
        action_audit = compare_actions(z['action'], trajectory['action'], rows)
        result['action_audit'] = action_audit
        checks['raw_execution_action_original_serialization'] = action_audit['log_matches_original_serialization']
        checks['raw_trajectory_full_precision_action_exact'] = action_audit['raw_trajectory_full_precision_exact']
        checks['raw_trajectory_exact'] = all(np.array_equal(z[key], trajectory[key]) for key in ['state', 'action', 'subtask_index']+IMAGES)
        # The preserved source snap is explicitly logged before its observation.
        regular = np.asarray([r['pre_observation_event'] == 'none' for r in rows[1:]])
        checks['state_next_state_continuity_except_logged_snap'] = bool(np.allclose(z['state'][1:][regular], trajectory['next_state'][:-1][regular], rtol=0, atol=1e-6))
        checks['raw_images'] = all(z[key].shape == (n, 54, 96, 3) and z[key].dtype == np.uint8 for key in IMAGES)
        checks['no_phase_or_parking_input'] = len(z['state_feature_names']) == 27 and not any('phase_' in str(v) or 'pass_done' in str(v) for v in z['state_feature_names'])
        result['image_audit'] = {key: dict(shape=list(z[key].shape),
            post_pass_distinct_pixel_arrays=int(len(np.unique(z[key][result['first_pass_step']:].reshape(-1, 54*96*3), axis=0)))) for key in IMAGES}
    checks['summary_complete'] = summary.get('recording_complete') is True and summary['error'] is None
    checks['summary_frame_count'] = summary['frames'] == len(rows)
    p = result['first_pass_step']
    if p is not None:
        q = np.asarray([r['q_before'][:6] for r in rows])
        ik = np.asarray([r['ik_delta'] for r in rows])
        servo = np.asarray([r['servo_target'][:6] for r in rows])
        no_freeze = np.clip(q+ik, cfg['joint_lower'][:6], cfg['joint_upper'][:6])
        gap = np.abs(no_freeze-servo)
        result['hidden_control_audit'] = dict(
            freeze_zero_post_pass_frames=sum(r['freeze_arm_zero'] for r in rows[p:]),
            freeze_default_post_pass_frames=sum(r['freeze_arm_default'] for r in rows[p:]),
            snap_events=[r['step'] for r in rows if r['pre_observation_event'] != 'none'],
            max_post_pass_servo_difference_without_freeze_rad=float(gap[p:].max()),
            post_pass_servo_difference_frames_gt_1e_6=int(np.any(gap[p:] > 1e-6, axis=1).sum()),
            post_pass_action_ee_position_range_m=np.ptp(np.asarray([r['applied_action'][2:5] for r in rows[p:]]), axis=0).tolist(),
            description='Expert freeze flags override IK joint targets. apply_door_dp_action clears flags. No replay was authorized or run; 10D-only hold equivalence is unproven.')
    result.update(seed=seed, planned_frames=json.loads((root/'plan.json').read_text())['planned'][str(seed)]['planned_frames'])
    result['frame_difference_from_plan'] = len(rows)-result['planned_frames']
    result['passed'] = all(checks.values())
    save(output/'data_acceptance.json', result)
    print(json.dumps(result), flush=True)
    if not result['passed']: raise RuntimeError('Data acceptance failed for seed %d' % seed)
    video = inspect_video(folder/'rollout.mp4', len(rows)//2)
    save(output/'video_verification.json', video)
    if not video['passed']: raise RuntimeError('Video verification failed')


def audit_windows(root, seed, source_folder=None):
    import torch
    from train_door_dp import DoorDPSequenceDataset, compute_stats, normalize_batch, _as_chw_u8
    from action_contract import FORWARD_V3, canonicalize_sequence, normalize_action, denormalize_action
    folder = root/'episodes'/('seed_%d' % seed)
    source = Path(source_folder) if source_folder is not None else folder
    acceptance = json.loads((folder/'data_acceptance.json').read_text())
    p, n = acceptance['first_pass_step'], acceptance['frames']
    dt = json.loads((source/'config.json').read_text())['dt']
    data = folder/'lerobot/local/stop_tail'
    seq = DoorDPSequenceDataset(data, 'local/stop_tail', 16, 32, contract_version=FORWARD_V3)
    with np.load(source/'raw/episode_000000.npz', allow_pickle=True) as z:
        raw_state, raw_action = z['state'].copy(), z['action'].copy()
        state = canonicalize_sequence(z['state'], slice(-5, -1))
        action = canonicalize_sequence(z['action'], slice(5, 9))
        images = {key: z[key].copy() for key in IMAGES}
    # Read every converted frame directly, without the loader's image fallback.
    for i in range(n):
        frame = seq.dataset[i]
        if not (np.array_equal(frame['observation.state'].numpy(), raw_state[i]) and
                np.array_equal(frame['action'].numpy(), raw_action[i]) and
                int(frame['episode_index']) == 0 and int(frame['frame_index']) == i):
            save(folder/'loader_failure.json', dict(stage='converted_frame', frame=i))
            raise ValueError('Converted frame differs from raw at frame %d' % i)
        for key in IMAGES:
            loaded = _as_chw_u8(frame['observation.images.'+key]).numpy()
            if not np.array_equal(loaded, images[key][i].transpose(2, 0, 1)):
                save(folder/'loader_failure.json', dict(stage='converted_image', frame=i, image=key))
                raise ValueError('Converted image differs from raw at frame %d: %s' % (i, key))
    centers = np.asarray(seq.indices)
    stats = compute_stats(seq)
    input_counts = np.zeros(n, dtype=int)
    prediction_counts, executed_counts = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    prediction_by_offset, execution_by_offset = np.zeros(32, dtype=int), np.zeros(16, dtype=int)
    stop = acceptance['stop_step']
    stages = dict(pre_pass=0, conversion=0, hold=0)
    transitions, hold_read_centers = [], []
    image_map = dict(mask='wrist_handle_mask', masked_depth='wrist_masked_depth', front_mask='front_handle_mask', front_masked_depth='front_masked_depth')
    started = time.monotonic()
    for i, center in enumerate(centers):
        item = seq[i]  # Every window is actually read through the unchanged loader.
        history = np.maximum(np.arange(center-15, center+1), seq.episode_starts[center])
        loaded_action = item['action'].numpy()
        exact = np.array_equal(item['state'].numpy(), state[history]) and np.array_equal(loaded_action, action[center:center+32])
        for key, raw_key in image_map.items():
            exact &= np.array_equal(item[key].numpy(), images[raw_key][history].transpose(0, 3, 1, 2))
        normalized = normalize_batch({key: value.unsqueeze(0) for key, value in item.items()}, stats, 'cpu', FORWARD_V3)
        expected_state = (torch.from_numpy(state[history])-stats['state_mean'])/stats['state_std']
        expected_action = normalize_action(torch.from_numpy(action[center:center+32]), stats, FORWARD_V3)
        exact &= torch.equal(normalized[0][0], expected_state) and torch.equal(normalized[-1][0], expected_action)
        if not exact or seq.episodes[center] != seq.episodes[center+31]:
            save(folder/'loader_failure.json', dict(stage='actual_window', item=i, center=int(center)))
            raise ValueError('Actual loader window differs from raw/preprocessing at center %d' % center)
        # Coverage counts are derived only after validating the actually read tensors.
        target_frames = np.arange(center, center+32)
        stop_labels = (target_frames >= p) & np.all(loaded_action[:, :2] == 0, axis=1)
        prediction_by_offset += stop_labels
        execution_by_offset += stop_labels[:16]
        np.add.at(input_counts, history, 1)
        np.add.at(prediction_counts, target_frames[stop_labels], 1)
        np.add.at(executed_counts, target_frames[:16][stop_labels[:16]], 1)
        stages['pre_pass' if center < p-16 else 'conversion' if center <= stop else 'hold'] += 1
        if center > stop:
            hold_read_centers.append(int(center))
        if center in [p-31, p-16, p-15, p-1, p, p+1]:
            transitions.append(dict(center=int(center), history_frames=history.tolist(),
                full32_stop_labels=int(stop_labels.sum()), executed16_stop_labels=int(stop_labels[:16].sum()),
                first_action_vx=float(loaded_action[0, 0]), first_action_yaw=float(loaded_action[0, 1])))
        if i % 100 == 0 or i == len(centers)-1:
            verify_frozen(root, REPO)
            progress = dict(read_windows=i+1, total_windows=len(seq), elapsed_sec=time.monotonic()-started)
            save(folder/'window_progress.json', progress)
            print(json.dumps(progress), flush=True)
    transformed = normalize_action(torch.from_numpy(action), stats, FORWARD_V3)
    roundtrip = float(torch.max(torch.abs(denormalize_action(transformed, stats, FORWARD_V3)-torch.from_numpy(action))))
    span = (hold_read_centers[-1]-hold_read_centers[0])*dt if hold_read_centers else 0.
    checks = dict(loader_window_count=len(seq) == n-31, all_converted_frames_exact=True,
        all_windows_actually_read=True, all_loaded_windows_and_preprocessing_exact=True,
        conversion_is_input=bool(p in centers and p-1 in centers),
        consecutive_hold_current_inputs_two_seconds=span >= 2.,
        consecutive_hold_centers=hold_read_centers == list(range(stop+1, int(centers[-1])+1)),
        stop_labels_in_all_executed_offsets=bool(np.all(execution_by_offset > 0)),
        raw_stop_targets_zero=bool(np.all(action[p:, :2] == 0)),
        unchanged_27d_10d_contract=state.shape[1] == 27 and action.shape[1] == 10,
        action_minmax_roundtrip=roundtrip <= 5e-7)
    output = dict(seed=seed, passed=all(checks.values()), checks=checks, frames=n, windows=len(seq),
        stage_definition='pre_pass center<p-16; conversion p-16<=center<=measured_stop; hold center>measured_stop (50Hz)',
        stages={key: dict(windows=value, fraction=value/len(seq)) for key, value in stages.items()},
        first_stop_input=p, last_valid_center=int(centers[-1]), terminal_hold_current_inputs=len(hold_read_centers),
        terminal_hold_current_input_span_sec=span, converted_frames_actually_read=n,
        all_windows_actually_read=len(seq), actual_transition_windows=transitions,
        unique_terminal_frames_in_history=int(np.count_nonzero(input_counts[p:])),
        terminal_full32_label_positions=int(prediction_counts.sum()),
        terminal_full32_labels_by_offset=prediction_by_offset.tolist(),
        terminal_full32_target_fraction=float(prediction_counts.sum()/(len(seq)*32)),
        terminal_front16_label_positions=int(executed_counts[p:].sum()),
        terminal_front16_labels_by_offset=execution_by_offset.tolist(),
        terminal_front16_target_fraction=float(executed_counts.sum()/(len(seq)*16)),
        unique_terminal_front16_targets=int(np.count_nonzero(executed_counts)),
        dropped_tail_centers=31, dropped_tail_seconds=31*dt, action_roundtrip_max_error=roundtrip,
        contract=FORWARD_V3, horizons=[16, 32, 16], actual_loader=str(Path(sys.modules['train_door_dp'].__file__)))
    save(folder/'window_acceptance.json', output)
    with (folder/'window_coverage.csv').open('w') as stream:
        writer = csv.writer(stream); writer.writerow(['frame', 'valid_current_input', 'history_appearances', 'parking_full32_target_appearances', 'parking_front16_target_appearances', 'post_pass'])
        valid = set(centers)
        writer.writerows((i, i in valid, input_counts[i], prediction_counts[i], executed_counts[i], i >= p) for i in range(n))
    print(json.dumps(output), flush=True)
    if not output['passed']: raise RuntimeError('Window acceptance failed for seed %d' % seed)


def finalize(root, error=None):
    plan = json.loads((root/'plan.json').read_text()) if (root/'plan.json').exists() else {}
    preservation = verify_inputs(plan)
    save(root/'preservation.json', preservation)
    results = []
    for seed in SEEDS:
        folder = root/'episodes'/('seed_%d' % seed)
        entry = dict(seed=seed, attempted=folder.exists())
        for key, name in [('data', 'data_acceptance.json'), ('windows', 'window_acceptance.json'), ('video', 'video_verification.json')]:
            entry[key] = json.loads((folder/name).read_text()) if (folder/name).exists() else None
        results.append(entry)
    passed = error is None and preservation['passed'] and all(all(v.get(key) and v[key]['passed'] for key in ['data', 'windows', 'video']) for v in results)
    save(root/'acceptance.json', dict(passed=passed, error=error, scheduled=3, results=results,
        no_training=True, no_holdout=True, no_replay=True, final_acceptance_reserved_for_independent_review=True))
    state(root, 'data_acceptance_complete' if passed else 'failed_stopped', error=error, passed=passed)
    save(root/'delivery_sha256.json', {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('delivery_sha256.json', 'supervisor.log', 'diagnostics.lock')})


def execute(root):
    if any((root/name).exists() for name in ['plan.json', 'state.json', 'failure.json', 'sources']):
        raise FileExistsError('Existing run is preserved; choose a new independent run')
    root.mkdir(parents=True, exist_ok=True)
    try:
        initialize(root)
        execute_initialized(root)
    except BaseException as exc:
        save(root/'failure.json', dict(error=repr(exc), error_type=type(exc).__name__,
            initialization_complete=(root/'plan.json').exists(), auto_retry=False))
        try:
            finalize(root, repr(exc))
        except BaseException as delivery_error:
            state(root, 'failed_stopped', error=repr(exc), delivery_error=repr(delivery_error), passed=False)
            save(root/'acceptance.json', dict(passed=False, error=repr(exc), delivery_error=repr(delivery_error),
                no_training=True, no_holdout=True, no_replay=True))
        raise


def execute_initialized(root):
    with phase_lock(root, 'stop_tail'):
        try:
            state(root, 'tests')
            cpu(root, 'initialization_tests', ['-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_stop_tail_initialization.py', '-v'])
            cpu(root, 'tests', ['-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_expert_tail.py', '-v'])
            cpu(root, 'data_check_tests', ['-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_stop_data_checks.py', '-v'])
            cpu(root, 'forward_pass_tests', ['-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_forward_pass.py', '-v'])
            cpu(root, 'contract_tests', ['-m', 'unittest', 'discover', '-s', str(REPO/'high-level/dp'), '-p', 'test_action_contract.py', '-v'])
            for seed in SEEDS:
                verify_frozen(root, REPO)
                state(root, 'collecting', seed=seed)
                cfg = json.loads((root/'reference'/('seed_%d/config.json' % seed)).read_text())['args']
                folder = root/'episodes'/('seed_%d' % seed)
                command = ['python', '-u', str(HERE/'wholebody_float.py'), '--headless', '--no_show_seg',
                    '--camera_seg', '--camera_depth', '--float_video', '--float_output', str(folder),
                    '--float_seed', str(seed), '--steps', '2500', '--float_expert_post_pass_seconds', '3']
                for key in ['robot_x', 'robot_y', 'robot_z', 'robot_yaw', 'stop_distance', 'push_base_vx',
                            'lever_step_size', 'float_execution', 'float_pass_mode', 'float_push_target']:
                    command += ['--'+key, str(cfg[key])]
                code = launch(root, 'expert_%d' % seed, command, timeout=600, total=None, environment='gym')
                verify_frozen(root, REPO)
                if code: raise RuntimeError('Expert seed %d exited %d' % (seed, code))
                state(root, 'auditing_data', seed=seed)
                cpu(root, 'data_%d' % seed, [__file__, '--root', root, '--audit-seed', seed])
                state(root, 'converting', seed=seed)
                cpu(root, 'conversion_%d' % seed, [REPO/'high-level/dp/convert_door_raw_to_lerobot.py',
                    '--raw_root', folder/'raw', '--root', folder/'lerobot', '--repo_id', 'local/stop_tail'])
                state(root, 'enumerating_windows', seed=seed)
                cpu(root, 'windows_%d' % seed, [__file__, '--root', root, '--windows-seed', seed])
            verify_frozen(root, REPO)
            finalize(root)
        except BaseException as exc:
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--audit-seed', type=int, choices=SEEDS)
    parser.add_argument('--windows-seed', type=int, choices=SEEDS)
    args = parser.parse_args(); root = Path(args.root).resolve()
    if args.audit_seed:
        audit_episode(root, args.audit_seed)
    elif args.windows_seed:
        audit_windows(root, args.windows_seed)
    else:
        execute(root)
