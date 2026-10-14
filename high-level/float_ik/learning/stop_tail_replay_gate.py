"""Exactly one A/B freeze replay pair; stop on failure and never train here."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO/'high-level/dp'))
import collect_stop_tail as shared
from artifacts import finite_json
from diagnostic_integrity import sha256, source_paths, verify_frozen, phase_lock
from stop_data_checks import stop_acceptance
from stop_audit import compare_actions, inspect_video

ACCEPTED = REPO/'experiments/float-push/20261014-stop-tail-offline-16040-v1'


def save(path, value):
    shared.save(path, finite_json(value))


def initialize(root):
    original = ACCEPTED/'inputs/seed_16040'
    verdict = json.loads((ACCEPTED/'acceptance.json').read_text())
    if not verdict['single_episode_16040_data_passed']:
        raise ValueError('16040 input was not accepted')
    manifest = json.loads((ACCEPTED/'delivery_sha256.json').read_text())
    if not all(sha256(ACCEPTED/name) == value for name, value in manifest.items()):
        raise ValueError('Accepted offline delivery changed')
    inputs, directories = {}, {}
    shared.register_input(ACCEPTED, inputs, directories)
    source_hashes = {}
    cfg = json.loads((original/'config.json').read_text())
    paths = set(source_paths(REPO, cfg['source_hashes']))
    for name in shared.git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0'):
        path = REPO/name
        if name.startswith(('high-level/', 'low-level/')) and path.suffix in ('.py', '.sh', '.yaml', '.yml', '.json', '.urdf') and path.is_file():
            paths.add(path)
    for path in sorted(paths):
        name = str(path.relative_to(REPO)); source_hashes[name] = sha256(path)
        target = root/'sources'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    shutil.copytree(original, root/'reference/seed_16040')
    save(root/'plan.json', dict(source_hashes=source_hashes, input_hashes=inputs, input_directories=directories,
        seed=16040, steps=1446, scheduled=['A', 'B'], attempts_per_group=1,
        A='Restore recorded expert freeze flags', B='Do not restore freeze flags',
        quaternion_mode='action in both; original apply_door_dp_action normalization; no executed quaternion restoration',
        preserves_natural_snap=True, no_training=True, source_commit='0acc2f050479fb54a0670ccb90bc72b98b52a54a',
        stop_thresholds=dict(settle_seconds=.5, speed_m_s=.01, angular_speed_rad_s=.02, hold_seconds=2., drift_m=.02, extra_path_m=.10),
        replay_recording='Exactly 1446 steps; stopping task thresholds require 2-second measured hold, not an extra recording extension',
        server_head=shared.git('rev-parse', 'HEAD').decode().strip(), created=time.time()))
    (root/'server_initial_status.txt').write_bytes(shared.git('status', '--short', '--branch'))
    (root/'server_worktree.diff').write_bytes(shared.git('diff', '--binary', 'HEAD'))


def command(root, group):
    cfg = json.loads((root/'reference/seed_16040/config.json').read_text())['args']
    result = ['python', '-u', str(HERE/'diagnostic_replay.py'), '--diagnostic-mode',
        'recorded' if group == 'A' else 'actions', '--diagnostic-quaternion-mode', 'action',
        '--float_replay', str(root/'reference/seed_16040/trajectory.npz'), '--headless', '--no_show_seg',
        '--camera_seg', '--camera_depth', '--float_video', '--float_seed', '16040', '--steps', '1446',
        '--float_output', str(root/'replay'/group)]
    for key in ['robot_x', 'robot_y', 'robot_z', 'robot_yaw', 'stop_distance', 'push_base_vx',
                'lever_step_size', 'float_execution', 'float_pass_mode', 'float_push_target']:
        result += ['--'+key, str(cfg[key])]
    return result


def metrics(root, group):
    output = root/'replay'/group
    cfg = json.loads((output/'config.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    rows = [json.loads(line) for line in (output/'execution.jsonl').read_text().splitlines()]
    reference = root/'reference/seed_16040'
    expert = [json.loads(line) for line in (reference/'execution.jsonl').read_text().splitlines()]
    options = json.loads((output/'diagnostic.json').read_text())
    result = stop_acceptance(rows, cfg['dt'], cfg['door_runtime']['door_motion_sign'],
        cfg['args']['pass_align_pos_tol'], expected_source='replay', required_post_pass_seconds=0.)
    checks = result['checks']
    checks.update(full_horizon=len(rows) == 1446 and summary['frames'] == 1446,
                  normal_exit=summary['error'] is None and summary.get('replay_exhausted_normally') is True,
                  same_quaternion_processing=options['quaternion_mode'] == 'action' and not options['restores_executed_quaternion'],
                  expected_restore_mode=options['restores_freeze_flags'] == (group == 'A'))
    with np.load(output/'trajectory.npz', allow_pickle=True) as actual, np.load(reference/'trajectory.npz', allow_pickle=True) as original:
        checks['same_requested_10d_actions'] = bool(np.array_equal(np.asarray([row['diagnostic']['requested_action'] for row in rows], np.float32), original['action']))
        checks['same_initial_state'] = bool(np.array_equal(actual['state'][0], original['state'][0]))
        checks['same_initial_images'] = all(np.array_equal(actual[key][0], original[key][0]) for key in shared.IMAGES)
        checks['finite_state_action'] = bool(np.isfinite(actual['state']).all() and np.isfinite(actual['action']).all() and np.isfinite(actual['next_state']).all())
        action_audit = compare_actions(actual['action'], actual['action'], rows)
        checks['action_application_log'] = action_audit['passed']
        if group == 'B':
            with np.load(root/'replay/A/trajectory.npz', allow_pickle=True) as a:
                checks['AB_applied_actions_exact'] = bool(np.array_equal(actual['action'], a['action']))
                checks['AB_executed_quaternions_exact'] = bool(np.array_equal(actual['executed_goal_quaternion'], a['executed_goal_quaternion']))
    actual_flags = [(row['freeze_arm_default'], row['freeze_arm_zero']) for row in rows]
    expected_flags = [(row['freeze_arm_default'], row['freeze_arm_zero']) for row in expert] if group == 'A' else [(False, False)]*1446
    checks['only_expected_freeze_restoration'] = actual_flags == expected_flags
    checks['initial_base_pose_exact'] = np.array_equal(rows[0]['root_before'][:7], expert[0]['root_before'][:7])
    q = np.asarray([row['q_after'] for row in rows]); eq = np.asarray([row['q_after'] for row in expert])
    dq = np.asarray([row['dq_after'] for row in rows])
    checks['finite_joint_motion'] = bool(np.isfinite(q).all() and np.isfinite(dq).all())
    ee_error = np.asarray([row['ee']['pos_error'] for row in rows])
    ee = np.asarray([row['ee']['actual_pos_world'] for row in rows]); eee = np.asarray([row['ee']['actual_pos_world'] for row in expert])
    limit_error = np.maximum(np.asarray(cfg['joint_lower'])-q, q-np.asarray(cfg['joint_upper']))
    result.update(group=group, joint_rmse_vs_expert=np.sqrt(np.mean((q-eq)**2, axis=0)).tolist(),
        max_joint_difference_vs_expert_rad=float(np.abs(q-eq).max()), max_joint_speed_rad_s=float(np.abs(dq).max()),
        max_joint_limit_violation_rad=float(max(0., limit_error.max())),
        ee_position_path_rmse_vs_expert_m=float(np.sqrt(np.mean((ee-eee)**2))),
        max_ee_target_position_error_m=float(np.linalg.norm(ee_error, axis=1).max()),
        post_pass_max_ee_target_position_error_m=float(np.linalg.norm(ee_error[result['first_pass_step']:], axis=1).max()) if result['first_pass_step'] is not None else None,
        finger_handle_contact_frames=sum(row['finger_handle_contact'] for row in rows),
        expert_finger_handle_contact_frames=sum(row['finger_handle_contact'] for row in expert),
        snap_event_steps=[row['step'] for row in rows if row['pre_observation_event'] != 'none'],
        no_framewise_arm_equality_requirement=True)
    video = inspect_video(output/'rollout.mp4', 723)
    save(output/'video_verification.json', video); checks['complete_video'] = video['passed']
    result['passed'] = all(checks.values())
    save(output/'gate.json', result)
    print(json.dumps(finite_json(result)), flush=True)
    if not result['passed']:
        raise RuntimeError('Replay %s gate failed' % group)


def finalize(root, error=None):
    plan = json.loads((root/'plan.json').read_text()) if (root/'plan.json').exists() else {}
    preserved = shared.verify_inputs(plan); save(root/'preservation.json', preserved)
    results = {group: json.loads((root/'replay'/group/'gate.json').read_text()) if (root/'replay'/group/'gate.json').exists() else None for group in ['A', 'B']}
    passed = error is None and preserved['passed'] and all(value and value['passed'] for value in results.values())
    save(root/'acceptance.json', dict(passed=passed, error=error, groups=results, scheduled=2,
        automatic_next_stage=False, data_collection_started=False, training_started=False))
    shared.state(root, 'replay_pair_complete' if passed else 'failed_stopped', error=error, passed=passed)
    save(root/'delivery_sha256.json', {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob('*')) if path.is_file() and path.name not in ['delivery_sha256.json', 'supervisor.log', 'diagnostics.lock']})


def run(root):
    if any((root/name).exists() for name in ['plan.json', 'state.json', 'sources']):
        raise FileExistsError('Existing replay run is preserved')
    root.mkdir(parents=True, exist_ok=True)
    try:
        initialize(root)
        with phase_lock(root, 'freeze_replay_pair'):
            shared.cpu(root, 'paired_contract_tests', ['-m', 'unittest', 'discover', '-s', HERE, '-p', 'test_stop_tail_pilot.py', '-v'])
            for group in ['A', 'B']:
                verify_frozen(root, REPO)
                shared.state(root, 'replaying_'+group)
                code = shared.launch(root, 'replay_'+group, command(root, group), timeout=600, total=None, environment='gym')
                verify_frozen(root, REPO)
                if code: raise RuntimeError('Replay %s exit %d' % (group, code))
                shared.cpu(root, 'gate_'+group, [__file__, '--root', root, '--metrics', group])
            finalize(root)
    except BaseException as exc:
        save(root/'failure.json', dict(error=repr(exc), auto_retry=False))
        finalize(root, repr(exc)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); parser.add_argument('--metrics', choices=['A', 'B'])
    args = parser.parse_args(); root = Path(args.root).resolve()
    if args.metrics: metrics(root, args.metrics)
    else: run(root)
