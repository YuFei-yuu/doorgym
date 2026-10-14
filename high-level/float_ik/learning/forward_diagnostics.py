"""Isolated, evidence-preserving diagnostics; deliberately has no train phase."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from artifacts import finite_json, write_new_json
from diagnostic_contract import (action_contract, normalize_action, denormalize_action,
    legacy_contract, validate_checkpoint_contract, quaternion_angle, select_windows)
from diagnostic_integrity import phase_lock, assert_no_active, source_paths, verify_frozen

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VAL_SEEDS = list(range(16100, 16118, 2))
REPLAY_SEEDS = [16100, 16108, 16116]
SAMPLING_SEEDS = [42, 43, 44]
MODES = ['recorded', 'actions', 'clipped']
IMAGE_KEYS = ['wrist_handle_mask', 'wrist_masked_depth', 'front_handle_mask', 'front_masked_depth']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def csv_write(path, values):
    if not values:
        return
    with Path(path).open('w', newline='') as stream:
        fields = list(dict.fromkeys(key for value in values for key in value))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for value in values:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in value.items()})


def inventory(root):
    return {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in root.rglob('*') if p.is_file()}


def initialize(root, original):
    root.mkdir(parents=True, exist_ok=False)
    original_plan = json.loads((original/'plan.json').read_text())
    protected = [original/'pipeline_state.json', original/'stop_after_updates.json',
                 original/'user_stop_3000/completion.json', original/'manifest.json',
                 original/'recording_contract.json', original/'training/metrics.jsonl']
    protected += [original/'training/checkpoints'/('step_%06d.pt' % step) for step in [1000, 2000, 3000]]
    protected += [original/'resources'/(name+'.json') for name in
                  ['training__004000', 'training__005000', 'offline__5000']]
    hashes = {str(p.relative_to(original)): digest(p) for p in protected}
    if hashes['training/checkpoints/step_003000.pt'] != '0d7dbe50eb5468d9f4ac9877d78ead681ec49bb39a904c5efa43ac62b4e0a0a6':
        raise ValueError('Final checkpoint hash changed')
    if json.loads((original/'pipeline_state.json').read_text())['phase'] != 'stopped_by_user':
        raise ValueError('Original experiment is not stopped')
    files = source_paths(REPO, json.loads((original/'recording_contract.json').read_text())['source_hashes'])
    source_hashes = {}
    for path in files:
        name = str(path.relative_to(REPO))
        source_hashes[name] = digest(path)
        target = root/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    plan = dict(original=str(original), source_baseline=original_plan['source_baseline'],
        replay_seeds=REPLAY_SEEDS, replay_modes=MODES, validation_seeds=VAL_SEEDS,
        sampling_seeds=SAMPLING_SEEDS, checkpoints=[1000, 2000, 3000],
        windows=['start', 'approach_end', 'grasp', 'push', 'through'],
        diffusion_timesteps=[0, 25, 50, 75, 99], no_training=True, holdout_test_used=False,
        cpu_threads=2, nice=10, replay_timeout_seconds=600, total_time_limit=None,
        source_hashes=source_hashes, original_protected_hashes=hashes,
        authorization='User requested implementation of the no-training diagnostic plan',
        created=time.time(), wandb_mode='offline')
    write_new_json(root/'plan.json', plan)
    write_new_json(root/'original_inventory.json', inventory(original))
    write_new_json(root/'state.json', dict(phase='initialized', no_training=True, at=time.time()))
    print(json.dumps(dict(phase='initialized', root=str(root))), flush=True)


def verify_sources(root):
    return verify_frozen(root, REPO)


def replay(root):
    with phase_lock(root, 'replay'):
        return _replay(root)


def _replay(root):
    plan = verify_sources(root)
    original = Path(plan['original'])
    if (root/'replay/results.json').exists():
        print('Replay phase already complete', flush=True)
        return
    sys.path.insert(0, str(HERE/'run_tools'))
    from launch import launch
    from forward_results import episode_result
    result = []
    for mode in MODES:
        for seed in REPLAY_SEEDS:
            folder = root/'replay'/mode/('seed_%d' % seed)
            folder.parent.mkdir(parents=True, exist_ok=True)
            expert = original/'dataset'/('val_seed_%d' % seed)
            cfg = json.loads((expert/'config.json').read_text())['args']
            horizon = json.loads((expert/'summary.json').read_text())['frames']
            case = dict(seed=seed, robot_x=cfg['robot_x'], robot_y=cfg['robot_y'])
            stage = 'replay__%s_%d' % (mode, seed)
            attempt = root/'replay'/(stage+'.json')
            if attempt.exists():
                result.append(json.loads(attempt.read_text()))
                continue
            # An incomplete attempt stays failed; never overwrite its evidence.
            if folder.exists() or (root/'resources'/(stage+'.json')).exists():
                assert_no_active(root)
                ledger = rows(root/'budget.jsonl') if (root/'budget.jsonl').exists() else []
                terminal = [entry for entry in ledger if entry['stage'] == stage]
                code = terminal[-1].get('exit_code') if terminal else None
            else:
                command = ['python', '-u', str(HERE/'diagnostic_replay.py'), '--diagnostic-mode', mode,
                    '--headless', '--no_show_seg', '--camera_seg', '--camera_depth', '--float_video',
                    '--float_replay', str(expert/'trajectory.npz'), '--float_output', str(folder),
                    '--float_seed', str(seed), '--steps', str(horizon)]
                for key in ['robot_x', 'robot_y', 'robot_z', 'robot_yaw', 'stop_distance',
                            'push_base_vx', 'lever_step_size', 'float_execution', 'float_pass_mode', 'float_push_target']:
                    command += ['--'+key, str(cfg[key])]
                if mode == 'clipped':
                    command += ['--stats-checkpoint', str(original/'training/checkpoints/step_003000.pt')]
                print(json.dumps(dict(phase='replay', mode=mode, seed=seed, horizon=horizon)), flush=True)
                code = launch(root, stage, command, timeout=600, total=None, environment='gym')
            value = episode_result(folder, case, code, policy=False)
            value.update(replay_mode=mode, expected_frames=horizon, autonomous_policy=False,
                         replay_horizon_complete=value['frames'] == horizon)
            write_new_json(attempt, value)
            result.append(value)
            print(json.dumps(finite_json(value)), flush=True)
    write_new_json(root/'replay/results.json', result)
    csv_write(root/'replay/results.csv', result)


def cached_prediction(model, observation, action, timesteps):
    """Reuse only constant observation features, without changing noise RNG."""
    import torch
    cond = torch.cat([observation, model.timestep_encoder(timesteps)], -1)
    x = model.input_proj(action.transpose(1, 2))
    for block in model.blocks:
        x = block(x, cond)
    return model.output_proj(x).transpose(1, 2)


def offline(root):
    with phase_lock(root, 'offline'):
        return _offline(root)


def _offline(root):
    plan = verify_sources(root)
    original = Path(plan['original'])
    out = root/'offline'
    out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():
        print('Offline phase already complete', flush=True)
        return
    import torch
    from diffusers import DDPMScheduler
    sys.path.insert(0, str(REPO/'high-level/dp'))
    from models.door_diffusion_policy import DoorDiffusionPolicy
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU diagnostic must not initialize CUDA')
    ckpt = torch.load(original/'training/checkpoints/step_003000.pt', map_location='cpu', weights_only=False)
    validate_checkpoint_contract(ckpt['config'])
    train_actions = np.concatenate([np.load(p)['action'] for p in sorted((original/'data/raw_train').glob('episode_*.npz'))])
    contract = action_contract(train_actions)
    normalized = normalize_action(train_actions, contract)
    restored = denormalize_action(normalized, contract)
    legacy = legacy_contract(ckpt['stats'])
    stats_test = dict(training_frames=len(train_actions), contract=contract,
        coverage=float((abs(normalized) <= 1.+1e-12).all(1).mean()),
        roundtrip_max_error=float(abs(restored-train_actions).max()),
        invalid_quaternions=int((abs(np.linalg.norm(restored[:, 5:9], axis=1)-1.) > 1e-5).sum()),
        legacy_outside_per_channel=(abs(normalize_action(train_actions, legacy)) > 1.+1e-6).mean(0).tolist(),
        legacy_checkpoint_version=validate_checkpoint_contract(ckpt['config']),
        old_checkpoint_reinterpreted=False, candidate_fitted_split='train')
    write_new_json(out/'normalization_candidate.json', stats_test)
    write_new_json(out/'action_minmax_v2.json', contract)
    specifications, observations, truths, histories = [], [], [], []
    for seed in VAL_SEEDS:
        with np.load(original/'dataset'/('val_seed_%d' % seed)/'trajectory.npz') as z:
            windows = select_windows(z['subtask_index'])
            for phase, center in windows:
                ids = np.maximum(np.arange(center-15, center+1), 0)
                specifications.append(dict(seed=seed, phase=phase, center=center, start_padding_frames=int((np.arange(center-15, center+1) < 0).sum())))
                observations.append((z['state'][ids].copy(), [z[k][ids].copy() for k in IMAGE_KEYS]))
                truths.append(z['action'][center:center+32].copy())
            q = z['action'][:, 5:9]
            raw_dots = np.sum(q[1:]*q[:-1], axis=1)
            histories.append(dict(seed=seed, training_excludes_first_centers=15,
                negative_quaternion_dot_count=int((raw_dots < 0).sum()),
                sign_flip_frames=(np.flatnonzero(raw_dots < 0)+1).tolist(),
                sign_invariant_step_angle_p95_deg=float(np.rad2deg(np.nanpercentile(quaternion_angle(q[1:], q[:-1]), 95))),
                stationary_fraction=float((abs(z['action'][:, 0]) < .01).mean()),
                phase_counts={str(int(p)): int((z['subtask_index'] == p).sum()) for p in np.unique(z['subtask_index'])}))
    write_new_json(out/'windows.json', specifications)
    write_new_json(out/'history_quaternion_audit.json', histories)
    state_raw = torch.from_numpy(np.stack([v[0] for v in observations]))
    image_batch = [torch.from_numpy(np.stack([v[1][k] for v in observations])).permute(0, 1, 4, 2, 3) for k in range(4)]
    truth = np.stack(truths)
    details, noise_details = [], []
    predictions = []
    started = time.monotonic()
    import wandb
    run = wandb.init(project='door-dp-diagnostics', name=root.name, id='forward-diagnostics-v1',
                     mode='offline', dir=str(root), config=plan)
    write_new_json(out/'wandb.json', dict(mode='offline', id=run.id, directory=run.dir, uploads=False))
    equivalence = []
    with torch.inference_mode():
        for step in plan['checkpoints']:
            checkpoint = original/'training/checkpoints'/('step_%06d.pt' % step)
            c = torch.load(checkpoint, map_location='cpu', weights_only=False)
            validate_checkpoint_contract(c['config'])
            model = DoorDiffusionPolicy(27, 10, 16, 32)
            model.load_state_dict(c['model']); model.eval()
            state = (state_raw-c['stats']['state_mean'])/c['stats']['state_std']
            # Encoding one window at a time avoids changing convolution batch
            # arithmetic from the production controller (which has batch=1).
            encoded = [model.encode_obs(state[i:i+1], *[v[i:i+1] for v in image_batch]) for i in range(len(specifications))]
            for random_seed in SAMPLING_SEEDS:
                # Every window has a separate reproducible generator. The same
                # seed/noise is used across checkpoints and diffusion timesteps.
                for i, spec in enumerate(specifications):
                    torch.manual_seed(random_seed)
                    scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon')
                    scheduler.set_timesteps(100)
                    action = torch.randn(1, 32, 10)
                    if random_seed == SAMPLING_SEEDS[0]:
                        t = torch.full((1,), 99, dtype=torch.long)
                        error = float((cached_prediction(model, encoded[i], action, t)-model(action, t, state[i:i+1], *[v[i:i+1] for v in image_batch])).abs().max())
                        equivalence.append(dict(step=step, **spec, cached_forward_max_error=error))
                        if error > 1e-6:
                            raise ValueError('Cached encoder changed model output')
                    for timestep in scheduler.timesteps:
                        ts = torch.full((1,), int(timestep), dtype=torch.long)
                        pred_noise = cached_prediction(model, encoded[i], action, ts)
                        action = scheduler.step(pred_noise, timestep, action).prev_sample
                    predicted = (action[0]*c['stats']['action_std']+c['stats']['action_mean']).numpy()[:16]
                    target = truth[i, :16]
                    rotation = quaternion_angle(predicted[:, 5:9], target[:, 5:9])
                    normalized_pred = (predicted-c['stats']['action_mean'].numpy())/c['stats']['action_std'].numpy()
                    value = dict(step=step, sampling_seed=random_seed, **spec,
                        vx_rmse=float(np.sqrt(np.mean((predicted[:, 0]-target[:, 0])**2))),
                        predicted_vx_mean=float(predicted[:, 0].mean()), expert_vx_mean=float(target[:, 0].mean()),
                        position_rmse_m=float(np.sqrt(np.mean(np.sum((predicted[:, 2:5]-target[:, 2:5])**2, axis=1)))),
                        rotation_rmse_deg=float(np.rad2deg(np.sqrt(np.nanmean(rotation**2)))),
                        rotation_p95_deg=float(np.rad2deg(np.nanpercentile(rotation, 95))),
                        invalid_quaternion_count=int((~np.isfinite(rotation)).sum()),
                        clip_fraction_per_channel=(abs(normalized_pred) >= .9998).mean(0).tolist(),
                        rmse_per_channel=np.sqrt(np.mean((predicted-target)**2, axis=0)).tolist())
                    details.append(value); predictions.append(predicted)
                    # Fixed validation noise, independent from sampling RNG.
                    generator = torch.Generator().manual_seed(random_seed)
                    noise = torch.randn((1, 32, 10), generator=generator)
                    targets = (torch.from_numpy(truth[i:i+1])-c['stats']['action_mean'])/c['stats']['action_std']
                    for timestep in plan['diffusion_timesteps']:
                        ts = torch.full((1,), timestep, dtype=torch.long)
                        noisy = scheduler.add_noise(targets, noise, ts)
                        residual = cached_prediction(model, encoded[i], noisy, ts)-noise
                        noise_details.append(dict(step=step, sampling_seed=random_seed, **spec, timestep=timestep,
                            noise_mse=float((residual**2).mean()), noise_mse_per_channel=(residual**2).mean((0, 1)).tolist()))
                    if len(details) % 15 == 0:
                        print(json.dumps(dict(phase='offline', completed=len(details), scheduled=405,
                                             elapsed_sec=time.monotonic()-started, step=step, sampling_seed=random_seed)), flush=True)
                        progress = dict(completed=len(details), scheduled=405, elapsed_sec=time.monotonic()-started)
                        (out/'progress.json').write_text(json.dumps(progress))
                        run.log(dict(diagnostic_windows=len(details), elapsed_sec=progress['elapsed_sec']))
            for phase in plan['windows']:
                group = [v for v in details if v['step'] == step and v['phase'] == phase]
                run.log(dict(checkpoint=step, **{'%s/vx_rmse' % phase: float(np.sqrt(np.mean([v['vx_rmse']**2 for v in group]))),
                                               '%s/position_rmse_m' % phase: float(np.sqrt(np.mean([v['position_rmse_m']**2 for v in group])))}))
    if torch.cuda.is_initialized():
        raise RuntimeError('Unexpected CUDA initialization')
    write_new_json(out/'sampling_results.json', details)
    write_new_json(out/'noise_results.json', noise_details)
    write_new_json(out/'cached_forward_equivalence.json', equivalence)
    csv_write(out/'sampling_results.csv', details)
    csv_write(out/'noise_results.csv', noise_details)
    np.savez_compressed(out/'predictions.npz', prediction=np.stack(predictions), expert=np.stack([truth[i % 45, :16] for i in range(405)]),
                        checkpoint=np.array([v['step'] for v in details]), seed=np.array([v['seed'] for v in details]),
                        sampling_seed=np.array([v['sampling_seed'] for v in details]), center=np.array([v['center'] for v in details]))
    write_new_json(out/'summary.json', dict(scheduled_sampling_windows=405, completed_sampling_windows=len(details),
        scheduled_noise_windows=2025, completed_noise_windows=len(noise_details),
        invalid_quaternion_count=sum(v['invalid_quaternion_count'] for v in details),
        cached_forward_max_error=max(v['cached_forward_max_error'] for v in equivalence),
        cpu_cuda_initialized=torch.cuda.is_initialized(), elapsed_sec=time.monotonic()-started,
        no_training=True, holdout_test_used=False, sampling_device='cpu', diffusers_version=__import__('diffusers').__version__))
    run.finish()


def verify_original(root):
    plan = json.loads((root/'plan.json').read_text())
    original = Path(plan['original'])
    before = json.loads((root/'original_inventory.json').read_text())
    after = inventory(original)
    changed = [name for name, value in before.items() if after.get(name) != value]
    added = sorted(set(after)-set(before))
    hash_matches = {name: digest(original/name) == h for name, h in plan['original_protected_hashes'].items()}
    result = dict(original_files=len(before), changed_or_missing=changed, added=added,
                  protected_hash_matches=hash_matches, passed=not changed and not added and all(hash_matches.values()),
                  stopped_state=json.loads((original/'pipeline_state.json').read_text())['phase'])
    write_new_json(root/'preservation.json', result)
    if not result['passed'] or result['stopped_state'] != 'stopped_by_user':
        raise ValueError('Original experiment preservation check failed')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--original', default=str(REPO/'experiments/float-push/20261013-forward-dp-v1'))
    parser.add_argument('--phase', choices=['initialize', 'replay', 'offline', 'verify'], required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.phase == 'initialize':
        initialize(root, Path(args.original).resolve())
    elif args.phase == 'replay':
        replay(root)
    elif args.phase == 'offline':
        offline(root)
    else:
        verify_original(root)


if __name__ == '__main__':
    main()
