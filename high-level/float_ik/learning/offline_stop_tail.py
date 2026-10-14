"""Offline acceptance of existing v2/16040. No simulation or training entry."""
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
sys.path.insert(0, str(REPO/'high-level/dp'))
from artifacts import finite_json
import collect_stop_tail as shared
from diagnostic_integrity import sha256, verify_frozen, source_paths, phase_lock

INPUT = REPO/'experiments/float-push/20261014-forward-stop-tail3s-v2'
FAILED_V1 = REPO/'experiments/float-push/20261014-forward-stop-tail3s-v1'
SEED = 16040


def save(path, value):
    shared.save(path, finite_json(value))


def initialize(root):
    source = INPUT/'episodes/seed_16040'
    inputs, directories = {}, {}
    for prior in [FAILED_V1, INPUT]:
        shared.register_input(prior, inputs, directories)
        manifest = json.loads((prior/'delivery_sha256.json').read_text())
        if not all(sha256(prior/name) == expected for name, expected in manifest.items()):
            raise ValueError('Prior delivery changed: '+str(prior))
        if json.loads((prior/'acceptance.json').read_text())['passed']:
            raise ValueError('Expected preserved failed-run verdict')
    # Only the existing offline audit helper is intentionally edited this turn.
    previous_sources = json.loads((INPUT/'plan.json').read_text())['source_hashes']
    protected_source_matches = {name: sha256(REPO/name) == expected for name, expected in previous_sources.items()
        if name != 'high-level/float_ik/learning/collect_stop_tail.py'}
    if not all(protected_source_matches.values()):
        raise ValueError('Unexpected drift outside the authorized offline audit changes')
    save(root/'unchanged_control_contract_sources.json', protected_source_matches)
    paths = set(source_paths(REPO, json.loads((source/'config.json').read_text())['source_hashes']))
    for name in shared.git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0'):
        path = REPO/name
        if name.startswith(('high-level/', 'low-level/')) and path.suffix in ('.py', '.sh', '.yaml', '.yml', '.json', '.urdf') and path.is_file():
            paths.add(path)
    hashes = {}
    for path in sorted(paths):
        name = str(path.relative_to(REPO)); hashes[name] = sha256(path)
        target = root/'sources'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    mirror = root/'inputs/seed_16040'
    shutil.copytree(source, mirror)
    mirror_hashes = {str(path.relative_to(source)): sha256(path) for path in source.rglob('*') if path.is_file()}
    if not all(sha256(mirror/name) == expected for name, expected in mirror_hashes.items()):
        raise ValueError('Read-only input mirror differs from v2')
    save(root/'plan.json', dict(source_hashes=hashes, input_hashes=inputs, input_directories=directories,
        mirror_hashes=mirror_hashes, input_run=str(INPUT), input_episode=str(source), seed=SEED,
        planned={'16040': dict(planned_frames=1446)}, action_contract='door_action_minmax_continuous_quat_v3',
        horizons=[16, 32, 16], no_training=True, no_simulation=True, no_replay=True, no_holdout=True,
        actual_loading='Every converted frame and every valid sequence through original loader',
        observation_cutoff='Current pre-action frame p is available; no next_state[p] or state[p+1:] used as an advance signal',
        server_head=shared.git('rev-parse', 'HEAD').decode().strip(), local_base='5dd64f2', created=time.time()))
    save(root/'preservation_initial.json', shared.verify_inputs(json.loads((root/'plan.json').read_text())))
    (root/'server_initial_status.txt').write_bytes(shared.git('status', '--short', '--branch'))
    (root/'server_worktree.diff').write_bytes(shared.git('diff', '--binary', 'HEAD'))
    (root/'server_untracked.txt').write_bytes(shared.git('ls-files', '--others', '--exclude-standard'))
    shared.state(root, 'offline_initialized')


def observations(root):
    from action_contract import FORWARD_V3, canonicalize_sequence
    from train_door_dp import DoorDPSequenceDataset
    source = root/'inputs/seed_16040'
    output = root/'episodes/seed_16040'
    acceptance = json.loads((output/'data_acceptance.json').read_text())
    p = acceptance['first_pass_step']
    cfg = json.loads((source/'config.json').read_text())
    rows = [json.loads(line) for line in (source/'execution.jsonl').read_text().splitlines()]
    seq = DoorDPSequenceDataset(output/'lerobot/local/stop_tail', 'local/stop_tail', 16, 32, FORWARD_V3)
    with np.load(source/'raw/episode_000000.npz', allow_pickle=True) as raw:
        state = canonicalize_sequence(raw['state'], slice(-5, -1))
        action = raw['action'].copy()
        images = {key: raw[key].copy() for key in shared.IMAGES}
        names = [str(value) for value in raw['state_feature_names']]
    # Raw observations are captured before step(action[p]); the prior-only version
    # also excludes the current pre-action frame, as a conservative sensitivity check.
    decision_ids = np.arange(p-15, p+1)
    previous_ids = np.arange(p-16, p)
    decision = seq[seq.indices.index(p)]['state'].numpy()
    previous = seq[seq.indices.index(p-1)]['state'].numpy()
    if not np.array_equal(decision, state[decision_ids]) or not np.array_equal(previous, state[previous_ids]):
        raise ValueError('Actual pre-action histories differ from raw/preprocessing')
    summaries = {}
    for label, ids, loaded in [('decision_pre_action', decision_ids, decision), ('strict_previous_only', previous_ids, previous)]:
        summaries[label] = dict(frames=ids.tolist(), latest_frame=int(ids[-1]), latest_time_sec=float(ids[-1]*cfg['dt']),
            images={key: dict(all_zero=bool(np.all(images[key][ids] == 0)),
                distinct_arrays=int(len(np.unique(images[key][ids].reshape(16, -1), axis=0)))) for key in shared.IMAGES},
            state_features={name: dict(first=float(loaded[0, i]), last=float(loaded[-1, i]),
                minimum=float(loaded[:, i].min()), maximum=float(loaded[:, i].max()),
                range=float(np.ptp(loaded[:, i])), total_change=float(loaded[-1, i]-loaded[0, i])) for i, name in enumerate(names)})
    # Compare only earlier moving histories. Labels are comparison annotations,
    # never input features; post-stop histories are excluded from this search.
    nearest = []
    for center in seq.indices:
        if center < 15 or center >= p or abs(action[center, 0]) <= .01:
            continue
        ids = np.arange(center-15, center+1)
        same_images = all(np.array_equal(images[key][ids], images[key][decision_ids]) for key in shared.IMAGES)
        if not same_images:
            continue
        delta = state[ids]-decision
        nearest.append(dict(center=center, max_abs_state_difference=float(np.abs(delta).max()),
            state_l2_difference=float(np.linalg.norm(delta)), label_vx=float(action[center, 0]),
            label_yaw=float(action[center, 1]), images_exactly_equal=True))
    nearest.sort(key=lambda value: (value['max_abs_state_difference'], value['state_l2_difference']))
    for candidate in nearest[:5]:
        center = candidate['center']; actual = seq[seq.indices.index(center)]['state'].numpy()
        if not np.array_equal(actual, state[center-15:center+1]):
            raise ValueError('Actual moving comparison history differs')
        candidate['actual_loader_history_rechecked'] = True
    result = dict(first_stop_action_frame=p, first_stop_action_vx=float(action[p, 0]),
        first_stop_action_yaw=float(action[p, 1]), preceding_action_vx=float(action[p-1, 0]),
        data_timing='state[p] and images[p] are pre-execution; next_state[p] and state[p+1:] excluded',
        no_snap_at_decision=rows[p]['pre_observation_event'] == 'none', histories=summaries,
        moving_histories_with_identical_images=len(nearest), nearest_earlier_moving_histories=nearest[:5],
        raw_state_units_mixed=True, no_distance_threshold_or_classifier_claim=True,
        excluded_from_signals=['pass_done', 'phase', 'through_target', 'world base position', 'command/action labels',
                               'freeze flags', 'next_state[p]', 'state after p'],
        candidate_signals=[
            dict(signal='Four policy image histories', limitation='If all-zero histories also occur while moving, image absence alone cannot specify the stop time.'),
            dict(signal='Pre-action base angular velocity, arm/gripper positions and velocities, relative end-effector pose',
                 limitation='Reported variation may correlate with approach/servo dynamics in this single expert trace; it does not establish a unique goal-distance or stopping cue.'),
            dict(signal='Finite 16-frame temporal trend', limitation='No world base position, target distance, completion flag or base linear velocity is in the 27D policy state; timing/generalization remains unproven.')],
        hidden_control=acceptance['hidden_control_audit'],
        hidden_control_dynamic_effect='Unverified: recorded IK candidate versus frozen servo target is a static comparison, not a replay or counterfactual rollout.',
        conclusion='No post-stop state change is used to claim advance identifiability. A single accepted expert episode does not prove DP autonomous stopping.')
    save(output/'pre_action_observations.json', result)
    with (output/'pre_action_history.csv').open('w') as stream:
        writer = csv.writer(stream); writer.writerow(['history', 'frame', *names])
        for label, ids in [('decision_pre_action', decision_ids), ('strict_previous_only', previous_ids)]:
            writer.writerows([label, int(i), *state[i].tolist()] for i in ids)
    print(json.dumps(finite_json(dict(first_stop_frame=p, history_frames=decision_ids.tolist(),
        identical_image_moving_histories=len(nearest), nearest=nearest[:2]))), flush=True)


def finish(root, error=None):
    import torch
    plan = json.loads((root/'plan.json').read_text()) if (root/'plan.json').exists() else {}
    preservation = shared.verify_inputs(plan)
    mirror = root/'inputs/seed_16040'
    preservation['mirror_unchanged'] = bool(plan) and all(sha256(mirror/name) == value for name, value in plan.get('mirror_hashes', {}).items())
    preservation['passed'] &= preservation['mirror_unchanged']
    save(root/'preservation.json', preservation)
    output = root/'episodes/seed_16040'
    checks = {}
    for key, filename in [('data', 'data_acceptance.json'), ('video', 'video_verification.json'), ('windows', 'window_acceptance.json')]:
        checks[key] = json.loads((output/filename).read_text()) if (output/filename).exists() else None
    observation_ready = (output/'pre_action_observations.json').exists()
    passed = error is None and preservation['passed'] and observation_ready and all(value and value['passed'] for value in checks.values())
    save(root/'acceptance.json', dict(audit_chain_passed=passed, single_episode_16040_data_passed=passed,
        three_episode_task_complete=False, three_episode_task_status='Only existing 16040 audited; 16041/16042 not collected',
        passed=passed, error=error, checks=checks, observation_analysis_complete=observation_ready,
        original_v1_v2_verdicts_preserved=preservation['passed'], no_training=True, no_simulation=True,
        no_replay=True, no_holdout=True, cuda_initialized=torch.cuda.is_initialized()))
    shared.state(root, 'offline_acceptance_complete' if passed else 'failed_stopped', error=error,
                 audit_chain_passed=passed, single_episode_16040_data_passed=passed, three_episode_task_complete=False)
    save(root/'delivery_sha256.json', {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob('*'))
        if path.is_file() and path.name not in ('delivery_sha256.json', 'supervisor.log', 'diagnostics.lock')})


def run(root):
    if any((root/name).exists() for name in ['plan.json', 'state.json', 'sources', 'failure.json']):
        raise FileExistsError('Existing offline run is preserved; use a new directory')
    root.mkdir(parents=True, exist_ok=True)
    try:
        initialize(root)
        with phase_lock(root, 'offline_stop_tail'):
            shared.state(root, 'offline_regressions')
            for stage, directory, pattern in [
                ('audit_regressions', HERE, 'test_stop_audit.py'),
                ('stop_threshold_regressions', HERE, 'test_stop_data_checks.py'),
                ('contract_regressions', REPO/'high-level/dp', 'test_action_contract.py')]:
                shared.cpu(root, stage, ['-m', 'unittest', 'discover', '-s', directory, '-p', pattern, '-v'])
            shared.state(root, 'offline_data_and_video')
            shared.cpu(root, 'data_video', [__file__, '--root', root, '--worker', 'data'])
            shared.state(root, 'offline_conversion')
            shared.cpu(root, 'conversion', [REPO/'high-level/dp/convert_door_raw_to_lerobot.py',
                '--raw_root', root/'inputs/seed_16040/raw', '--root', root/'episodes/seed_16040/lerobot', '--repo_id', 'local/stop_tail'])
            shared.state(root, 'offline_actual_loader_windows')
            shared.cpu(root, 'actual_windows', [__file__, '--root', root, '--worker', 'windows'])
            shared.state(root, 'offline_pre_action_observations')
            shared.cpu(root, 'pre_action_observations', [__file__, '--root', root, '--worker', 'observations'])
            verify_frozen(root, REPO)
            finish(root)
    except BaseException as exc:
        save(root/'failure.json', dict(error=repr(exc), error_type=type(exc).__name__, auto_retry=False))
        try:
            finish(root, repr(exc))
        except BaseException as delivery_error:
            shared.state(root, 'failed_stopped', error=repr(exc), delivery_error=repr(delivery_error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--worker', choices=['data', 'windows', 'observations'])
    args = parser.parse_args(); root = Path(args.root).resolve()
    if args.worker == 'data':
        shared.audit_episode(root, SEED, source_folder=root/'inputs/seed_16040', output_folder=root/'episodes/seed_16040')
    elif args.worker == 'windows':
        shared.audit_windows(root, SEED, source_folder=root/'inputs/seed_16040')
    elif args.worker == 'observations':
        observations(root)
    else:
        run(root)
