"""Authorized three-episode forward_v3 pilot, capped at 1000 updates."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from artifacts import write_new_json
from diagnostic_integrity import phase_lock, source_paths, sha256, verify_frozen, materialize_external_links
from forward_diagnostics import csv_write, inventory
from forward_results import episode_result, aggregate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ORIGINAL = REPO/'experiments/float-push/20261013-forward-dp-v1'
SEEDS = [16040, 16041, 16042]
CHECKPOINTS = [250, 500, 1000]
sys.path.insert(0, str(HERE/'run_tools'))
from launch import launch
from bounded import atomic_json, read_rows


def state(root, phase, **extra):
    value = dict(phase=phase, at=time.time(), pid=os.getpid(), **extra)
    atomic_json(root/'state.json', value)
    print(json.dumps(value), flush=True)


def cpu(root, stage, arguments):
    verify_frozen(root, REPO)
    prefix = REPO/'.repro/envs/door-dp-train'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        WANDB_MODE='offline', XDG_CACHE_HOME=str(REPO/'.repro/cache/train'), LD_LIBRARY_PATH=str(prefix/'lib'))
    env.pop('PYTHONPATH', None)
    (root/'logs').mkdir(exist_ok=True)
    with (root/'logs'/(stage+'.log')).open('x') as output:
        code = subprocess.call(['nice', '-n', str(max(0, 10-os.getpriority(os.PRIO_PROCESS, 0))),
            str(prefix/'bin/python'), '-u', *map(str, arguments)], cwd=REPO, env=env,
            stdout=output, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError('CPU stage failed: %s (exit %d)' % (stage, code))


def initialize(root):
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ORIGINAL/'manifest.json').read_text())
    selected = [next(v for v in manifest if v['seed'] == seed) for seed in SEEDS]
    originals = {}
    raw = root/'data/raw'; raw.mkdir(parents=True)
    reference = root/'reference'; reference.mkdir()
    for index, item in enumerate(selected):
        if not item['success'] or item['robot_x'] != 4.1 or item['robot_y'] != 0:
            raise ValueError('Expected an existing successful nominal training episode')
        folder = ORIGINAL/Path(item['episode']).parent
        if sha256(folder/'trajectory.npz') != item['sha256'] or sha256(folder/'raw/episode_000000.npz') != item['raw_sha256']:
            raise ValueError('Expert recording hash changed')
        target = reference/('seed_%d' % item['seed']); target.mkdir()
        for name in ['trajectory.npz', 'execution.jsonl', 'config.json', 'summary.json', 'rollout.mp4']:
            source = folder/name
            originals[str(source.relative_to(ORIGINAL))] = sha256(source)
            shutil.copy2(source, target/name)
        shutil.copy2(folder/'raw/episode_000000.npz', raw/('episode_%06d.npz' % index))
        shutil.copy2(folder/'raw/door_dp_feature_names.json', raw/'door_dp_feature_names.json')
    prior = REPO/'experiments/float-push/20261013-forward-dp-diagnostics-v1'
    protected = json.loads((prior/'plan.json').read_text())['original_protected_hashes']
    cfg = json.loads((reference/'seed_16040/config.json').read_text())
    paths = source_paths(REPO, list(cfg['source_hashes'])+[str(Path(cfg['args']['door_cfg']).relative_to(REPO))])
    hashes = {}
    for source in paths:
        name = str(source.relative_to(REPO)); hashes[name] = sha256(source)
        target = root/'sources'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    plan = dict(source_baseline='0acc2f050479fb54a0670ccb90bc72b98b52a54a',
        source_hashes=hashes, train_seeds=SEEDS, evaluation_seeds=SEEDS, checkpoints=CHECKPOINTS,
        max_updates=1000, batch_size=64, num_workers=0, random_initialization=True, training_seed=42,
        contract='door_action_minmax_continuous_quat_v3', train_frames=sum(v['frames'] for v in selected),
        train_sequences=sum(v['frames']-31 for v in selected), lr=1e-4, obs_horizon=16,
        pred_horizon=32, action_horizon=16, diffusion_iterations=100, total_time_limit=None,
        evaluation_steps=2500, simulation_timeout_seconds=600, evaluation_is_training_initial_states=True,
        holdout_test_used=False, auto_expand=False, original=str(ORIGINAL),
        original_expert_hashes=originals, protected_original_hashes=protected,
        authorization='User authorized three nominal episodes, forward_v3 random initialization, batch64 workers0, 1000 updates and 250/500/1000 video checks',
        logging='Native metrics every 10 updates; W&B offline import after optimization to keep staged RNG unchanged',
        created=time.time())
    write_new_json(root/'plan.json', plan)
    write_new_json(root/'recording_contract.json', dict(schema=cfg['schema'], execution_profile=cfg['execution_profile'],
        tensor_pipeline=cfg['tensor_pipeline'], fps=50, source_expert_seeds=SEEDS,
        expert_source_hashes=cfg['source_hashes'], preprocessing_version=plan['contract']))
    write_new_json(root/'manifest.json', selected)
    state(root, 'initialized', planned_updates=1000, train_frames=plan['train_frames'])


def audit_checkpoint(root, step):
    import torch
    from action_contract import FORWARD_V3, checkpoint_version, canonicalize_sequence
    checkpoint = root/'training/checkpoints'/('step_%06d.pt' % step)
    c = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = c['config']; plan = json.loads((root/'plan.json').read_text())
    if (checkpoint_version(cfg) != FORWARD_V3 or cfg['completed_updates'] != step or
        cfg['batch_size'] != 64 or cfg['num_workers'] != 0 or cfg['dataset_frames'] != plan['train_frames'] or
        cfg['dataset_sequences'] != plan['train_sequences'] or cfg['state_dim'] != 27 or cfg['action_dim'] != 10):
        raise ValueError('Checkpoint training contract mismatch')
    expected_resume = None if step == 250 else str(root/'training/checkpoints'/('step_%06d.pt' % (250 if step == 500 else 500)))
    if cfg.get('resume') != expected_resume:
        raise ValueError('Training did not start fresh or resume this run as planned')
    actions, states = [], []
    for path in sorted((root/'data/raw').glob('episode_*.npz')):
        with np.load(path) as z:
            actions.append(canonicalize_sequence(z['action'], slice(5, 9)))
            states.append(canonicalize_sequence(z['state'], slice(-5, -1))[:-31])
    action = torch.from_numpy(np.concatenate(actions)); values = torch.from_numpy(np.concatenate(states))
    exact = all([torch.equal(c['stats']['action_min'], action.amin(0)), torch.equal(c['stats']['action_max'], action.amax(0)),
                 torch.equal(c['stats']['state_mean'], values.mean(0)), torch.equal(c['stats']['state_std'], values.std(0).clamp_min(1e-6))])
    if not exact:
        raise ValueError('Checkpoint statistics are not fitted to the three episodes')
    result = dict(step=step, checkpoint=str(checkpoint), sha256=sha256(checkpoint), passed=True,
        statistics_three_episodes_only=True, configured_max_updates=1000, dataset_frames=cfg['dataset_frames'],
        dataset_sequences=cfg['dataset_sequences'], random_initialization=step == 250,
        action_min=c['stats']['action_min'].tolist(), action_max=c['stats']['action_max'].tolist())
    write_new_json(root/('checkpoint_%06d.json' % step), result)


def train_command(root, target, previous):
    if target not in CHECKPOINTS or target > 1000:
        raise ValueError('Unauthorized training target')
    if previous != {250: None, 500: 250, 1000: 500}[target]:
        raise ValueError('Unauthorized checkpoint/resume sequence')
    command = ['python', '-u', str(REPO/'high-level/dp/train_door_dp.py'),
        '--root', str(root/'data/lerobot/local/small_forward_v3'), '--repo_id', 'local/small_forward_v3',
        '--run_name', root.name, '--output_dir', str(root/'training'), '--steps', str(target),
        '--batch_size', '64', '--num_workers', '0', '--lr', '1e-4', '--action_contract', 'forward_v3',
        '--obs_horizon', '16', '--pred_horizon', '32', '--action_horizon', '16', '--num_diffusion_iters', '100',
        '--seed', '42', '--log_interval', '10', '--save_interval', str(target),
        '--recording_metadata', str(root/'recording_contract.json')]
    if previous:
        command += ['--resume', str(root/'training/checkpoints'/('step_%06d.pt' % previous))]
    return command


def evaluate(root, step):
    suite = root/'evaluation'/('step_%06d' % step); suite.mkdir(parents=True, exist_ok=True)
    attempts = []
    for seed in SEEDS:
        folder = suite/('seed_%d' % seed)
        cfg = json.loads((root/'reference'/('seed_%d/config.json' % seed)).read_text())['args']
        command = ['python', '-u', str(HERE/'wholebody_float.py'), '--headless', '--no_show_seg',
            '--camera_seg', '--camera_depth', '--float_video', '--float_output', str(folder),
            '--float_seed', str(seed), '--steps', '2500', '--dp_policy_checkpoint',
            str(root/'training/checkpoints'/('step_%06d.pt' % step)), '--dp_inference_steps', '100', '--dp_action_horizon', '16']
        for key in ['robot_x', 'robot_y', 'robot_z', 'robot_yaw', 'stop_distance', 'push_base_vx',
                    'lever_step_size', 'float_execution', 'float_pass_mode', 'float_push_target']:
            command += ['--'+key, str(cfg[key])]
        verify_frozen(root, REPO)
        code = launch(root, 'evaluation__%06d_%d' % (step, seed), command, timeout=600, total=None, environment='gym')
        case = dict(seed=seed, robot_x=4.1, robot_y=0., checkpoint_updates=step)
        result = episode_result(folder, case, code, policy=True)
        if (folder/'execution.jsonl').exists():
            from diagnostic_report import usable_rows, motion
            log, error = usable_rows(folder/'execution.jsonl')
            if log:
                result.update(motion(log)); result['motion_log_error'] = error
        attempts.append(result)
        atomic_json(suite/'attempts.json', attempts)
        print(json.dumps(dict(phase='evaluated', step=step, seed=seed, success=result['success'], max_angle=result['max_signed_door_deg'])), flush=True)
    write_new_json(suite/'results.json', attempts); csv_write(suite/'results.csv', attempts)
    write_new_json(suite/'aggregate.json', aggregate(attempts, 3))
    return attempts


def report(root, final=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from forward_report import rollout_plots
    analysis = root/'analysis'; analysis.mkdir(exist_ok=True)
    metric_path = root/'training/metrics.jsonl'
    metrics = [json.loads(v) for v in metric_path.read_text().splitlines()] if metric_path.exists() else []
    if metrics:
        csv_write(root/'training/metrics.csv', metrics)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot([v['train/step'] for v in metrics], [v['train/loss'] for v in metrics])
        axes[0].set(xlabel='Updates', ylabel='Epsilon MSE (logged batches)'); axes[0].grid(alpha=.2)
        axes[1].plot([v['train/step'] for v in metrics], [v['train/elapsed_sec'] for v in metrics])
        axes[1].set(xlabel='Updates', ylabel='Cumulative optimization elapsed (s)'); axes[1].grid(alpha=.2)
        fig.savefig(analysis/'training_curve.png', dpi=150); plt.close(fig)
    all_results, summary = [], []
    for step in CHECKPOINTS:
        path = root/'evaluation'/('step_%06d/results.json' % step)
        if not path.exists(): continue
        values = json.loads(path.read_text()); all_results.extend(values)
        summary.append(dict(step=step, **aggregate(values, 3)))
        for value in values:
            try:
                rollout_plots(Path(value['folder']))
            except (ValueError, OSError, IndexError) as exc:
                value['plot_error'] = repr(exc)
    csv_write(analysis/'checkpoint_success.csv', summary)
    csv_write(analysis/'all_results.csv', all_results)
    lines = ['# Three-episode forward_v3 pilot', '', 'Random initialization; train seeds16040–16042, 3871 frames, 3778 windows. Batch64, workers0, lr1e-4; maximum1000 updates.',
        'Closed-loop checks reuse the three training initial states; these results do not measure held-out generalization.', '',
        '| Updates | Opened >=80deg | Passed | Autonomous joint | Attempts |', '|---:|---:|---:|---:|---:|']
    for v in summary: lines.append('| %d | %d | %d | %d | %d |' % (v['step'], v['opened'], v['passed'], v['successes'], v['attempts']))
    lines += ['', '| Updates | Seed | Max signed angle | Net base displacement | Mean vx | Autonomous | End reason |', '|---:|---:|---:|---:|---:|---|---|']
    for v in all_results:
        lines.append('| %d | %d | %.3f deg | %.3f m | %.5f m/s | %s | %s |' % (v['checkpoint_updates'], v['seed'],
            v['max_signed_door_deg'] or 0., v.get('net_m', 0.), v.get('mean_vx', 0.), v['autonomous'], v['end_reason']))
    lines += ['', 'All attempts, process failures and timeouts remain in the denominator. Source snap_arm_to_zero is retained and separately logged.',
        'Success requires correct-direction >=80 degrees, pass_done within0.12m and DP-only valid-image control.',
        'Full videos are in evaluation/step_*/seed_*/rollout.mp4. Comparisons, when available, pair the same recorded initial observation.',
        'Check plan.json, checkpoint_*.json, source snapshots, resource logs and preservation.json. No automatic training beyond1000 updates.']
    (analysis/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if final:
        import wandb
        run = wandb.init(project='door-dp-small-forward-v3', name=root.name, mode='offline', dir=str(root),
                         config=json.loads((root/'plan.json').read_text()))
        for item in metrics: run.log(item, step=item['train/step'])
        run.finish()


def finish(root):
    import imageio_ffmpeg
    summaries = []
    for step in CHECKPOINTS:
        suite = root/'evaluation'/('step_%06d' % step)
        values = json.loads((suite/'results.json').read_text()); summaries.append(dict(step=step, **aggregate(values, 3)))
        for value in values:
            path = Path(value['folder'])/'rollout.mp4'
            code = subprocess.call([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-xerror', '-threads', '2', '-i', str(path), '-f', 'null', '-']) if path.exists() else None
            value['video_decoded'] = code == 0
        write_new_json(suite/'video_verification.json', [dict(seed=v['seed'], decoded=v['video_decoded']) for v in values])
    plan = json.loads((root/'plan.json').read_text())
    protected = all(sha256(ORIGINAL/name) == expected for name, expected in plan['protected_original_hashes'].items())
    experts = all(sha256(ORIGINAL/name) == expected for name, expected in plan['original_expert_hashes'].items())
    prior_inventory = json.loads((REPO/'experiments/float-push/20261013-forward-dp-diagnostics-v1/original_inventory.json').read_text())
    metadata = inventory(ORIGINAL) == prior_inventory
    write_new_json(root/'preservation.json', dict(original_protected_hashes_unchanged=protected,
        expert_recordings_unchanged=experts, original_inventory_unchanged=metadata))
    if not protected or not experts or not metadata: raise RuntimeError('Original evidence preservation failed')
    write_new_json(root/'materialized_links.json', materialize_external_links(root))
    completed = json.loads((root/'training/progress.json').read_text())['step']
    if completed != 1000 or len(list((root/'training/checkpoints').glob('step_*.pt'))) != 3:
        raise ValueError('Pilot update/checkpoint limit violated')
    completion = dict(phase='completed_small_data_pilot', completed_updates=1000, checkpoint_summaries=summaries,
        auto_expand=False, holdout_test_used=False, objective_complete=False,
        gpu_state=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader'], text=True))
    write_new_json(root/'completion.json', completion); state(root, 'completed_small_data_pilot', completed_updates=1000, auto_expand=False)
    write_new_json(root/'delivery_sha256.json', {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'delivery_sha256.json'})


def run(root):
    if not root.exists(): initialize(root)
    with phase_lock(root, 'small_forward_v3'):
        try:
            execute(root)
        except BaseException as exc:
            state(root, 'failed', error=repr(exc), auto_expand=False)
            raise


def execute(root):
        verify_frozen(root, REPO)
        if (root/'completion.json').exists(): raise RuntimeError('Pilot already complete; no automatic restart')
        if (root/'training/checkpoints/model_latest.pt').exists(): raise RuntimeError('Partial pilot exists; inspect before resuming')
        state(root, 'converting_three_episodes')
        cpu(root, 'conversion', [REPO/'high-level/dp/convert_door_raw_to_lerobot.py', '--raw_root', root/'data/raw',
            '--root', root/'data/lerobot', '--repo_id', 'local/small_forward_v3'])
        previous = None
        for target in CHECKPOINTS:
            verify_frozen(root, REPO); state(root, 'training', target_updates=target, maximum_updates=1000)
            code = launch(root, 'training__%06d' % target, train_command(root, target, previous), timeout=None, total=None, environment='train')
            if code: raise RuntimeError('Training stage failed with exit %s' % code)
            audit_checkpoint(root, target)
            state(root, 'evaluating', completed_updates=target, scheduled_episodes=3)
            evaluate(root, target); report(root); previous = target
        report(root, final=True); finish(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); args = parser.parse_args()
    sys.path.insert(0, str(REPO/'high-level/dp'))
    output = Path(args.root).resolve()
    run(output)
