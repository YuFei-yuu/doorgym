#!/usr/bin/env python3
"""Print commands only. Execution and experiment budgets need a later task order."""
import argparse
import json
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads(Path(__file__).with_name('config.json').read_text())


def build(stage, run_id, gpu=None, seed=1000, max_rollouts=None, train_steps=None, trials=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Use a unique run id with letters, digits, underscore or hyphen')
    root = ROOT / '.repro' / 'runs' / run_id
    raw = root / 'raw'
    repo_id = 'local/wc4_joint9_skill_' + run_id
    dataset_root = root / 'lerobot'
    dataset = dataset_root / repo_id
    train = root / 'train'
    official = train / 'checkpoints' / 'last'
    wrapped = root / 'door-checkpoint'
    c = CFG
    common = ['--door_cfg', str(ROOT / c['door_cfg']), '--door_name', 'wc4',
              '--asset_root', str(ROOT / 'high-level/data/asset/a2wz1'), '--asset_file', 'urdf/a2wz1.urdf',
              '--a2w_float_ik_config', str(ROOT / 'high-level/float_ik/config/a2w_float_ik_push_door_parallel.yaml'),
              '--depth_aug_config', str(ROOT / 'high-level/dp/config/depth_camera_aug_default.yaml'),
              '--skill_program_json', str(ROOT / c['skill']), '--arm_ik_solver', 'gym_jacobian',
              '--ee_pose_frame', 'robot_base_full',
              '--ikpush_robot_pitch_rand_min', '0', '--ikpush_robot_pitch_rand_max', str(c['pitch_randomization_rad'][1]),
              '--camera_intrinsics_mode', c['camera_intrinsics_mode'],
              '--no_collision_geom_check', '--no_show_camera_images']
    depth = ['--no_enable_depth_noise', '--no_enable_depth_gaussian_blur',
             '--enable_depth_camera_randomization', '--depth_camera_pos_rand_m', '0.02',
             '--depth_camera_rot_rand_deg', '5.0']
    if stage == 'record':
        if not max_rollouts or max_rollouts < 1:
            raise ValueError('record requires --max-rollouts (future approved attempt budget)')
        kind = 'gym'
        cmd = ['python', 'high-level/dp/record/record_door_dp_dataset_a2w_state10.py',
               '--mode', 'ikpush', '--state_action_mode', 'joint_state9', '--num_episodes', '50',
               '--raw_root', str(raw), '--run_log_root', str(root / 'record-logs'),
               '--num_envs', '16', '--record_all_envs', '--steps', str(c['rollout_steps']),
               '--max_quota_rollouts', str(max_rollouts), '--seed', str(seed), '--fps', '25',
               '--camera_fps', '25', '--camera_depth_clip_lower', '0.2', '--camera_depth_clip_far', '1.5',
               '--depth_only', '--no_keyframe_loss_weights', '--headless', '--graphics_device_id', '0'] + depth + ['--'] + common
        guard = f'test ! -e {shlex.quote(str(raw))}'
    elif stage == 'convert':
        kind = 'act'
        cmd = ['python', 'high-level/dp/convert_door_raw_to_lerobot.py', '--raw_root', str(raw),
               '--root', str(dataset_root), '--repo_id', repo_id, '--state_action_mode', 'joint_state9',
               '--depth_only', '--fps', '25', '--image_storage', 'video', '--video_codec', 'h264', '--num_workers', '4']
        guard = f'python migration/wc4_act/validate_data.py raw {shlex.quote(str(raw))} && test ! -e {shlex.quote(str(dataset))}'
    elif stage == 'train':
        if not train_steps or train_steps < 1:
            raise ValueError('train requires --train-steps (future approved update budget)')
        kind = 'act'
        cmd = ['python', 'high-level/lerobot/src/lerobot/scripts/lerobot_train.py',
               '--dataset.repo_id=' + repo_id, '--dataset.root=' + str(dataset), '--dataset.use_imagenet_stats=false',
               '--policy.type=act', '--policy.device=cuda', '--policy.push_to_hub=false',
               '--policy.chunk_size=100', '--policy.n_action_steps=50', '--policy.vision_backbone=resnet18',
               '--policy.optimizer_lr=1e-5', '--policy.use_action_loss_weight=false',
               '--policy.plucker_conditioning=false', '--policy.interaction_state_conditioning=false',
               '--policy.camera_input_gating=false', '--policy.handle_latent_aux=false',
               '--policy.point_cloud_conditioning=false', '--policy.end_signal_prediction=false',
               '--keyframe_sampling_ratio=0', '--seed=' + str(seed), '--steps=' + str(train_steps),
               '--batch_size=16', '--num_workers=3', '--save_freq=10000', '--log_freq=100',
               '--output_dir=' + str(train), '--job_name=wc4-act-joint9-' + run_id,
               '--wandb.enable=true', '--wandb.mode=online', '--wandb.entity=' + c['wandb_entity'],
               '--wandb.project=' + c['wandb_project'], '--wandb.disable_artifact=true']
        guard = f'python migration/wc4_act/validate_data.py lerobot {shlex.quote(str(dataset))} && test ! -e {shlex.quote(str(train))}'
    elif stage == 'export':
        kind = 'act'
        cmd = ['python', 'high-level/dp/export_official_lerobot_act_to_door_checkpoint.py',
               '--official_checkpoint', str(official), '--root', str(dataset_root), '--repo_id', repo_id,
               '--out_dir', str(wrapped), '--depth_only', '--device', 'cpu', '--action_horizon', '25']
        guard = f'test -d {shlex.quote(str(official))} && test ! -e {shlex.quote(str(wrapped))}'
    elif stage == 'evaluate':
        if not trials or trials < 1:
            raise ValueError('evaluate requires --trials (future approved trial budget)')
        kind = 'gym'
        cmd = ['python', 'high-level/dp/eval/eval_door_policy_success.py',
               '--checkpoint', str(wrapped / 'model_latest.pt'), '--mode', 'ikpush', '--robot_body', 'a2wz1',
               '--door_cfg', str(ROOT / c['door_cfg']), '--num_envs', '16', '--total_trials', str(trials),
               '--parallel_batches', '1', '--steps', str(c['rollout_steps']), '--base_seed', str(seed),
               '--pass_open_angle_deg', '80', '--success_metric', 'abs', '--dp_fps', '25',
               '--dp_action_horizon', '25', '--dp_log_interval', '1', '--depth_only', '--headless',
               '--graphics_device_id', '0', '--run_root', str(root / 'evaluation')] + depth + ['--'] + common + [
               '--camera_depth_clip_lower', '0.2', '--camera_depth_clip_far', '1.5', '--camera_fps', '25']
        guard = f'test -f {shlex.quote(str(wrapped / "model_latest.pt"))} && test ! -e {shlex.quote(str(root / "evaluation"))}'
    else:
        raise ValueError(stage)
    if stage in ('record', 'train', 'evaluate') and gpu is None:
        raise ValueError('GPU stages require --gpu, selected after checking availability')
    env = 'VIDEO2DOOR_GPU=' + shlex.quote(str(gpu)) if gpu is not None else 'VIDEO2DOOR_GPU='
    shell = [env, 'source migration/wc4_act/activate.sh ' + kind, guard, 'exec nice -n 10 ' + shlex.join(cmd)]
    return {'stage': stage, 'cwd': str(ROOT), 'run_root': str(root), 'dataset': str(dataset),
            'argv': cmd, 'shell': ' && '.join(shell)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['record', 'convert', 'train', 'export', 'evaluate'])
    p.add_argument('--run-id', required=True)
    p.add_argument('--gpu', type=int)
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--max-rollouts', type=int)
    p.add_argument('--train-steps', type=int)
    p.add_argument('--trials', type=int)
    p.add_argument('--json', action='store_true')
    a = p.parse_args()
    result = build(a.stage, a.run_id, a.gpu, a.seed, a.max_rollouts, a.train_steps, a.trials)
    print(json.dumps(result, ensure_ascii=False, indent=2) if a.json else 'cd ' + shlex.quote(str(ROOT)) + ' && ' + result['shell'])


if __name__ == '__main__':
    main()
