#!/usr/bin/env python3
"""Python3.8 parser and actual worker protocol checks; no simulator or model."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'high-level/float_ik'))
sys.path.insert(0, str(ROOT / 'high-level/dp'))
sys.path.insert(0, str(ROOT / 'high-level'))
from commands import build
from isaacgym import gymapi


def forbidden(*args, **kwargs):
    raise RuntimeError('Simulator acquisition forbidden in preparation checks')


gymapi.acquire_gym = forbidden
import numpy as np
import isaacgym_float_ik_a2w_basearn_push_door_parallel as expert
import isaacgym_a2w_ik_push_door_parallel as a2w
import door_policy_backend as backend


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_expert(cmd):
    sys.argv = cmd[1:]
    args = expert.parse_args()
    assert args.door_name == 'wc4' and args.arm_ik_solver == 'gym_jacobian'
    assert args.door_twin_skill_program is not None
    assert args.ikpush_robot_pitch_rand_min == 0
    assert abs(args.ikpush_robot_pitch_rand_max - np.deg2rad(5)) < 1e-12
    assert not args.no_ikpush_env_randomization
    assert args.depth_only and args.camera_fps == 25 and args.dp_fps == 25
    assert args.camera_intrinsics_mode == 'legacy'
    assert args.enable_depth_camera_randomization and not args.enable_depth_noise
    assert not args.enable_depth_gaussian_blur
    return args


def main():
    recorder = load('wc4_recorder', 'high-level/dp/record/record_door_dp_dataset_a2w_state10.py')
    sys.argv = build('record', 'preparation-command-check', gpu=1, max_rollouts=1)['argv'][1:]
    args = recorder.parse_args()
    captured = []
    recorder.run_logged_subprocess = lambda cmd, *a, **kw: captured.append(cmd)
    recorder.run_one('ikpush', 0, args)
    record_args = parse_expert(captured[0])
    assert record_args.dp_record_state_mode == 'a2w_last_command_joint_state9'
    evaluation = load('wc4_evaluation', 'high-level/dp/eval/eval_door_policy_success.py')
    sys.argv = build('evaluate', 'preparation-command-check', gpu=1, trials=64)['argv'][1:]
    eval_args = evaluation.parse_args()
    play_cmd = evaluation.build_play_command(eval_args, 16, 0, ROOT / '.repro/not-created.jsonl')
    play = load('wc4_play', 'high-level/dp/play/play_door_policy.py')
    sys.argv = play_cmd[1:]
    captured_play = []
    original_run = subprocess.run
    try:
        play.subprocess.run = lambda cmd, **kw: captured_play.append(cmd)
        play.main()
    finally:
        subprocess.run = original_run
    eval_expert = parse_expert(captured_play[0])
    fields = ['asset_root', 'asset_file', 'door_cfg', 'skill_program_json', 'camera_intrinsics_mode',
              'ikpush_robot_pitch_rand_min', 'ikpush_robot_pitch_rand_max', 'camera_fps', 'dp_fps',
              'camera_depth_clip_lower', 'camera_depth_clip_far', 'enable_depth_camera_randomization',
              'depth_camera_pos_rand_m', 'depth_camera_rot_rand_deg', 'enable_depth_noise',
              'enable_depth_gaussian_blur', 'arm_ik_solver', 'ee_pose_frame']
    for field in fields:
        assert getattr(record_args, field) == getattr(eval_expert, field), field
    with tempfile.TemporaryDirectory(prefix='wc4-asset-check-') as tmp:
        split, base, arm = a2w.build_a2wz1_split_asset_root(record_args.asset_root, record_args.asset_file, Path(tmp))
        assert (split / base).is_file() and (split / arm).is_file()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1')
    cmd = backend._worker_python_command()
    with tempfile.TemporaryFile(mode='w+b') as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, env=env)
        try:
            # Real author framing and worker. Unknown command also decodes a Py3.8/NumPy1 payload.
            backend._write_pickle_message(proc.stdin, {'cmd': 'preparation_probe', 'array': np.zeros((2, 9), dtype=np.float32)})
            response = backend._read_pickle_message(proc.stdout)
            assert not response['ok'] and 'Unknown worker command' in response['error'], response
            backend._write_pickle_message(proc.stdin, {'cmd': 'close'})
            closed = backend._read_pickle_message(proc.stdout)
            assert closed['ok'], closed
            proc.wait(timeout=30)
            assert proc.returncode == 0
        finally:
            if proc.poll() is None:
                proc.kill()
            err.seek(0)
            worker_stderr = err.read().decode(errors='replace')
    print(json.dumps({'python': sys.executable, 'gym_api': gymapi.__file__, 'worker_command': cmd,
        'record_effective_args': vars(record_args), 'evaluate_effective_args': vars(eval_expert),
        'matched_fields': fields, 'worker_probe_response': response, 'worker_close_response': closed,
        'worker_stderr': worker_stderr, 'simulator_acquisition_disabled': True,
        'limitation': 'Protocol/import/argument check only; no checkpoint loaded or policy inference performed.'}, indent=2, default=str))


if __name__ == '__main__':
    main()
