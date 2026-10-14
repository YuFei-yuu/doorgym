"""Numerical evidence, replay divergence and plots for the no-training run."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from artifacts import write_new_json
from diagnostic_contract import quaternion_angle
from forward_diagnostics import csv_write, digest, rows, VAL_SEEDS, REPLAY_SEEDS, MODES
from diagnostic_integrity import frozen_phase


def motion(log):
    before = np.array([r['root_before'] for r in log])
    after = np.array([r['base_after'] for r in log])
    vx = np.array([r['robot_command']['vx'] for r in log])
    rate = np.array([r['robot_command']['yaw'] for r in log])
    d = after[:, :2]-before[:, :2]
    q = before[:, 3:7]
    yaw = np.arctan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]), 1-2*(q[:, 1]**2+q[:, 2]**2))
    pred = sum(vx[:, None]*.005*np.stack([np.cos(yaw+rate*j*.005), np.sin(yaw+rate*j*.005)], axis=1) for j in range(4))
    return dict(net_m=float(np.linalg.norm(after[-1, :2]-before[0, :2])),
        path_m=float(np.linalg.norm(d, axis=1).sum()), mean_vx=float(vx.mean()),
        low_speed_fraction=float((abs(vx) < .01).mean()),
        integration_max_m=float(np.linalg.norm(d-pred, axis=1).max()),
        max_signed_door_deg=float((-np.rad2deg([r['door_after'][0] for r in log])).max()))


@frozen_phase
def audit(root):
    import torch
    import pyarrow.parquet as pq
    from PIL import Image
    torch.set_num_threads(2)
    plan = json.loads((root/'plan.json').read_text())
    original = Path(plan['original'])
    out = root/'audit'
    out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():
        return
    original_motion = []
    for step in plan['checkpoints']:
        for seed in VAL_SEEDS:
            log = rows(original/('evaluation/step_%06d/seed_%d/execution.jsonl' % (step, seed)))
            original_motion.append(dict(step=step, seed=seed, **motion(log)))
    write_new_json(out/'original_dp_motion.json', original_motion)
    csv_write(out/'original_dp_motion.csv', original_motion)
    numeric = []
    all_actions = []
    all_states = []
    phases = []
    for split in ['train', 'val']:
        base = original/'data/lerobot'/split/'local'/('forward_'+split)/'data'
        for p in sorted((original/'data'/('raw_'+split)).glob('episode_*.npz')):
            with np.load(p) as z:
                action, state = z['action'], z['state']
                table = pq.read_table(next(base.rglob(p.stem+'.parquet')), columns=['action', 'observation.state']).to_pydict()
                numeric.append(dict(split=split, episode=p.name, frames=len(action),
                    action_max_error=float(abs(np.asarray(table['action'])-action).max()),
                    state_max_error=float(abs(np.asarray(table['observation.state'])-state).max())))
                if split == 'train':
                    all_actions.append(action.copy()); all_states.append(state.copy()); phases.append(z['subtask_index'].reshape(-1).copy())
    ckpt = torch.load(original/'training/checkpoints/step_003000.pt', map_location='cpu', weights_only=False)
    stats_check = {}
    for name, values in [('action', all_actions), ('state', all_states)]:
        v = torch.from_numpy(np.concatenate(values))
        stats_check[name+'_mean_max_error'] = float((v.mean(0)-ckpt['stats'][name+'_mean']).abs().max())
        stats_check[name+'_std_max_error'] = float((v.std(0).clamp_min(1e-6)-ckpt['stats'][name+'_std']).abs().max())
    images = []
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'dp'))
    from train_door_dp import _as_chw_u8
    image_keys = ['wrist_handle_mask', 'wrist_masked_depth', 'front_handle_mask', 'front_masked_depth']
    for episode in [0, 40, 80]:
        p = original/'data/raw_train'/('episode_%06d.npz' % episode)
        with np.load(p) as z:
            f = next((original/'data/lerobot/train/local/forward_train/data').rglob(p.stem+'.parquet'))
            table = pq.read_table(f)
            for key in image_keys:
                data = z[key]
                for frame in [0, 15, 50, 400, 1000, len(data)-1]:
                    value = table['observation.images.'+key][frame].as_py()
                    decoded = np.asarray(Image.open(io.BytesIO(value['bytes']))) if value['bytes'] else np.asarray(Image.open(f.parents[2]/value['path']))
                    result = _as_chw_u8(torch.from_numpy(decoded.copy()).permute(2, 0, 1).float()/255).permute(1, 2, 0).numpy()
                    images.append(dict(episode=episode, frame=frame, image=key,
                        max_error=int(abs(result.astype(int)-data[frame].astype(int)).max())))
    a, ph = np.concatenate(all_actions), np.concatenate(phases)
    mean, std = ckpt['stats']['action_mean'].numpy(), ckpt['stats']['action_std'].numpy()
    by_phase = []
    for phase in np.unique(ph):
        b = a[ph == phase]
        by_phase.append(dict(phase=int(phase), frames=len(b), fraction=len(b)/len(a),
            mean_vx=float(b[:, 0].mean()), legacy_outside_per_channel=(abs((b-mean)/std) > 1.+1e-6).mean(0).tolist()))
    write_new_json(out/'raw_lerobot_numeric.json', numeric)
    write_new_json(out/'image_checks.json', images)
    write_new_json(out/'training_phase_distribution.json', by_phase)
    processes = subprocess.check_output(['ps', '-axo', 'pid,ppid,nice,etime,command'], text=True)
    project_processes = [line for line in processes.splitlines() if '/data/zyf/video2door' in line and 'ps -axo' not in line]
    write_new_json(out/'resources.json', dict(gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader'], text=True),
        project_processes=project_processes, at=time.time()))
    write_new_json(out/'summary.json', dict(raw_lerobot_episodes=len(numeric),
        raw_lerobot_action_max_error=max(v['action_max_error'] for v in numeric),
        raw_lerobot_state_max_error=max(v['state_max_error'] for v in numeric),
        image_comparisons=len(images), image_max_pixel_error=max(v['max_error'] for v in images),
        checkpoint_stats=stats_check, stationary_fraction=float((abs(a[:, 0]) < .01).mean()),
        train_frames=len(a), original_motion_attempts=len(original_motion),
        cuda_initialized=torch.cuda.is_initialized(), no_training=True))
    print('AUDIT_COMPLETE', flush=True)


def comparison_failure(seed, mode, error):
    fields = ['compared_frames', 'net_m', 'path_m', 'mean_vx', 'low_speed_fraction', 'integration_max_m',
              'max_signed_door_deg', 'initial_root_exact', 'initial_state_exact', 'initial_images_exact',
              'q_max_error_vs_expert', 'base_max_error_vs_expert_m', 'ee_rmse_vs_expert_m',
              'ee_max_error_vs_expert_m', 'first_divergence_vs_recorded_frame', 'divergence_threshold',
              'first_divergence_detail', 'target_tracking_rmse_m', 'servo_at_joint_limit_fraction',
              'actual_joint_outside_limit_frames', 'freeze_arm_default_frames', 'freeze_arm_zero_frames',
              'snap_frames', 'horizon_complete', 'original_expert_frames']
    return dict(seed=seed, mode=mode, available=False, error=error, **{key: None for key in fields})


def usable_rows(path):
    """Keep a completed prefix if a process died while writing its last row."""
    result = []
    required = ['root_before', 'base_after', 'q_before', 'q_after', 'ee', 'door_after',
                'servo_target', 'robot_command', 'freeze_arm_default', 'freeze_arm_zero', 'pre_observation_event']
    if not Path(path).exists():
        return result, 'missing_execution_log'
    for line in Path(path).read_text().splitlines():
        try:
            value = json.loads(line)
            if not all(key in value for key in required):
                return result, 'incomplete_execution_row'
            result.append(value)
        except (ValueError, TypeError):
            return result, 'truncated_execution_log'
    return result, None if result else 'empty_execution_log'


def replay_details(root):
    plan = json.loads((root/'plan.json').read_text())
    original = Path(plan['original'])
    results = json.loads((root/'replay/results.json').read_text())
    comparisons = []
    for value in results:
        seed, mode = value['seed'], value['replay_mode']
        folder = Path(value['folder'])
        log, log_error = usable_rows(folder/'execution.jsonl')
        if not log:
            comparisons.append(comparison_failure(seed, mode, log_error))
            continue
        reference = rows(original/('dataset/val_seed_%d/execution.jsonl' % seed))
        baseline_path = root/'replay/recorded'/('seed_%d/execution.jsonl' % seed)
        baseline, _ = usable_rows(baseline_path)
        baseline = baseline or reference
        if not (folder/'config.json').exists() or not (folder/'trajectory.npz').exists():
            detail = comparison_failure(seed, mode, 'missing_config_or_trajectory')
            detail.update(compared_frames=len(log), **motion(log))
            comparisons.append(detail)
            continue
        n = min(len(log), len(reference))
        state = np.array([r['q_before'] for r in log[:n]])
        q_ref = np.array([r['q_before'] for r in reference[:n]])
        b = np.array([r['root_before'][:2] for r in log[:n]])
        b_ref = np.array([r['root_before'][:2] for r in reference[:n]])
        ee = np.array([r['ee']['actual_pos_world'] for r in log[:n]])
        ee_ref = np.array([r['ee']['actual_pos_world'] for r in reference[:n]])
        pose_error = np.linalg.norm(ee-ee_ref, axis=1)
        base_error = np.linalg.norm(b-b_ref, axis=1)
        q_error = abs(state-q_ref).max(1)
        m = min(len(log), len(baseline))
        baseline_ee = np.array([r['ee']['actual_pos_world'] for r in baseline[:m]])
        baseline_q = np.array([r['q_before'] for r in baseline[:m]])
        baseline_b = np.array([r['root_before'][:2] for r in baseline[:m]])
        divergence = (np.linalg.norm(ee[:m]-baseline_ee, axis=1) > .001) | (abs(state[:m]-baseline_q).max(1) > .001) | (np.linalg.norm(b[:m]-baseline_b, axis=1) > .001)
        first = int(np.flatnonzero(divergence)[0]) if divergence.any() else None
        cfg = json.loads((folder/'config.json').read_text())
        targets = np.array([r['servo_target'] for r in log])
        lower, upper = np.array(cfg['joint_lower']), np.array(cfg['joint_upper'])
        joints = np.array([r['q_after'] for r in log])
        first_before = log[0]['root_before']
        baseline_before = reference[0]['root_before']
        with np.load(folder/'trajectory.npz') as z, np.load(original/('dataset/val_seed_%d/trajectory.npz' % seed)) as expert:
            initial_images = all(np.array_equal(z[k][0], expert[k][0]) for k in ['wrist_handle_mask', 'wrist_masked_depth', 'front_handle_mask', 'front_masked_depth'])
            initial_state = np.array_equal(z['state'][0], expert['state'][0])
        detail = dict(seed=seed, mode=mode, available=True, error=log_error, compared_frames=n, **motion(log),
            initial_root_exact=first_before == baseline_before, initial_state_exact=bool(initial_state), initial_images_exact=initial_images,
            q_max_error_vs_expert=float(q_error.max()), base_max_error_vs_expert_m=float(base_error.max()),
            ee_rmse_vs_expert_m=float(np.sqrt(np.mean(pose_error**2))), ee_max_error_vs_expert_m=float(pose_error.max()),
            first_divergence_vs_recorded_frame=first, divergence_threshold=.001,
            first_divergence_detail=dict(expert_flags=[baseline[first]['freeze_arm_default'], baseline[first]['freeze_arm_zero']],
                current_flags=[log[first]['freeze_arm_default'], log[first]['freeze_arm_zero']],
                q_before=log[first]['q_before'], reference_q_before=baseline[first]['q_before']) if first is not None else None,
            target_tracking_rmse_m=float(np.sqrt(np.mean(np.sum(np.array([r['ee']['pos_error'] for r in log])**2, axis=1)))),
            servo_at_joint_limit_fraction=float(((abs(targets-lower) < 1e-5) | (abs(targets-upper) < 1e-5)).any(1).mean()),
            actual_joint_outside_limit_frames=int(((joints < lower-.01) | (joints > upper+.01)).any(1).sum()),
            freeze_arm_default_frames=sum(r['freeze_arm_default'] for r in log),
            freeze_arm_zero_frames=sum(r['freeze_arm_zero'] for r in log),
            snap_frames=[r['step'] for r in log if r['pre_observation_event'] != 'none'],
            horizon_complete=len(log) == len(reference), original_expert_frames=len(reference))
        comparisons.append(detail)
    write_new_json(root/'replay/comparisons.json', comparisons)
    csv_write(root/'replay/comparisons.csv', comparisons)
    return results, comparisons


@frozen_phase
def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = root/'analysis'; out.mkdir(exist_ok=True)
    results, comparisons = replay_details(root)
    samples = json.loads((root/'offline/sampling_results.json').read_text())
    noise = json.loads((root/'offline/noise_results.json').read_text())
    history = json.loads((root/'offline/history_quaternion_audit.json').read_text())
    summary = []
    for step in [1000, 2000, 3000]:
        for phase in ['start', 'approach_end', 'grasp', 'push', 'through']:
            group = [v for v in samples if v['step'] == step and v['phase'] == phase]
            summary.append(dict(step=step, phase=phase, windows=len(group),
                vx_rmse=float(np.sqrt(np.mean([v['vx_rmse']**2 for v in group]))),
                position_rmse_m=float(np.sqrt(np.mean([v['position_rmse_m']**2 for v in group]))),
                rotation_rmse_deg=float(np.sqrt(np.mean([v['rotation_rmse_deg']**2 for v in group]))),
                predicted_vx_mean=float(np.mean([v['predicted_vx_mean'] for v in group])),
                expert_vx_mean=float(np.mean([v['expert_vx_mean'] for v in group])),
                clip_fraction_per_channel=np.mean([v['clip_fraction_per_channel'] for v in group], axis=0).tolist()))
    write_new_json(out/'phase_metrics.json', summary); csv_write(out/'phase_metrics.csv', summary)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    phases = ['start', 'approach_end', 'grasp', 'push', 'through']
    for ax, key, label in zip(axes, ['vx_rmse', 'position_rmse_m', 'rotation_rmse_deg'], ['VX RMSE (m/s)', 'EE position RMSE (m)', 'Rotation RMSE (deg)']):
        for j, step in enumerate([1000, 2000, 3000]):
            ax.bar(np.arange(5)+(j-1)*.25, [v[key] for v in summary if v['step'] == step], width=.25, label=str(step))
        ax.set_xticks(np.arange(5)); ax.set_xticklabels(phases, rotation=25); ax.set_ylabel(label); ax.grid(axis='y', alpha=.2)
        ax.legend(title='Updates', fontsize=8)
    fig.savefig(out/'offline_phase_metrics.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    original = Path(json.loads((root/'plan.json').read_text())['original'])
    colors = dict(recorded='black', actions='royalblue', clipped='darkorange')
    for i, seed in enumerate(REPLAY_SEEDS):
        for mode in MODES:
            result = next((v for v in results if v['seed'] == seed and v['replay_mode'] == mode), None)
            log, error = usable_rows(Path(result['folder'])/'execution.jsonl') if result else ([], 'not_attempted')
            if not log:
                axes[i, 0].plot([], [], color=colors[mode], label=mode+' (unavailable)')
                axes[i, 1].plot([], [], color=colors[mode], label=mode+' (unavailable)')
                continue
            t = np.arange(len(log))*.02
            axes[i, 0].plot(t, -np.rad2deg([r['door_after'][0] for r in log]), label=mode, color=colors[mode])
            b = np.array([r['base_after'][:2] for r in log])
            axes[i, 1].plot(b[:, 0], b[:, 1], label=mode, color=colors[mode])
        axes[i, 0].axhline(80, color='red', linestyle='--'); axes[i, 0].set(xlabel='Simulation time (s)', ylabel='Signed door angle (deg)', title='Seed %d' % seed)
        axes[i, 1].set(xlabel='World X (m)', ylabel='World Y (m)', title='Seed %d base path' % seed); axes[i, 1].axis('equal')
        for ax in axes[i]:
            ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out/'replay_door_and_path.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), constrained_layout=True)
    for ax, phase in zip(axes, phases):
        for step in [1000, 2000, 3000]:
            ys = [np.mean([r['noise_mse'] for r in noise if r['step'] == step and r['phase'] == phase and r['timestep'] == t]) for t in [0, 25, 50, 75, 99]]
            ax.plot([0, 25, 50, 75, 99], ys, marker='o', label=str(step))
        ax.set(title=phase, xlabel='Diffusion timestep', ylabel='Fixed-noise epsilon MSE'); ax.set_yscale('log'); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.savefig(out/'fixed_noise_validation.png', dpi=150); plt.close(fig)
    lines = ['# 浮动底座DP无训练诊断报告', '',
        '本轮只运行专家动作回放、CPU离线推理和数值检查。所有回放均非DP自主成功。', '',
        '## 三种回放结果', '', '| seed | 模式 | 正向最大门角 | 通过 | 成功 | 完整帧数 | 净位移 |', '|---|---|---:|---|---|---:|---:|']
    for value in results:
        match = next(v for v in comparisons if v['seed'] == value['seed'] and v['mode'] == value['replay_mode'])
        displacement = '%.3f m' % match['net_m'] if match.get('net_m') is not None else 'unavailable'
        lines.append('| %d | %s | %.3f° | %s | %s | %d/%d | %s |' % (value['seed'], value['replay_mode'], value['max_signed_door_deg'] or 0, value['pass_done'], value['success'], value['frames'], value['expected_frames'], displacement))
    lines += ['', 'recorded恢复专家冻结标志及执行四元数；actions仅经DP动作接口应用10D动作；clipped还模拟原归一化采样范围限制。三组均保留自然snap逻辑。', '',
              '## 分阶段离线评估', '', '| 更新 | 阶段 | 窗口数 | vx RMSE | 末端位置RMSE | 姿态RMSE |', '|---:|---|---:|---:|---:|---:|']
    for value in summary:
        lines.append('| %d | %s | %d | %.4f m/s | %.4f m | %.2f° |' % (value['step'], value['phase'], value['windows'], value['vx_rmse'], value['position_rmse_m'], value['rotation_rmse_deg']))
    lines += ['', '45个固定专家观测窗口 × 3个采样随机种子 × 3个checkpoint；另外每个窗口检查5个固定扩散时刻。没有修改模型权重，未使用最终留出测试。', '',
        '## 起始历史与姿态', '', '训练不包含前15帧作为预测中心；推理起始窗口使用重复首帧填充。现有数据不能据此证明历史填充是主因。',
        '四元数相邻负内积次数：'+', '.join('%d: %d' % (v['seed'], v['negative_quaternion_dot_count']) for v in history)+'。姿态指标使用符号不变角误差；符号翻转不等同于物理姿态跳变。', '',
        '## 修复候选与训练边界', '', '候选action_minmax_v2仅从训练动作拟合，状态保持z-score。范围和往返检查见offline/normalization_candidate.json；此候选未接入训练器或旧模型控制器。',
        '原checkpoint继续按旧z-score契约读取。没有将旧模型改按min/max解释，没有运行无裁剪模型仿真。',
        'W&B使用offline模式，记录本轮诊断指标；本地CSV/JSON/NPZ为完整证据。', '',
        '## 证据入口', '', '- replay/results.csv、replay/comparisons.json：全部回放及首次分歧。',
        '- offline/sampling_results.csv、noise_results.csv、predictions.npz：全部预定离线窗口。',
        '- audit/summary.json：raw/LeRobot、归一化、图像抽查与原27回合位移。',
        '- preservation.json：原run文件清单与模型/停止记录哈希检查。',
        '- plan.json、sources/、resources/、budget.jsonl：冻结计划、代码与进程资源记录。',
        '- analysis/replay_door_and_path.png、offline_phase_metrics.png、fixed_noise_validation.png：诊断图。', '',
        '最终任务仍需模型自主正确方向开门至少80°并直接前穿。回放结果、离线误差和候选覆盖率均不代替最终任务验收。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    acceptance = dict(replay_attempts=len(results), replay_scheduled=9,
        all_replay_horizons_complete=all(v['replay_horizon_complete'] and v['exit_code'] == 0 for v in results),
        recorded_successes=sum(v['success'] for v in results if v['replay_mode'] == 'recorded'),
        pure_actions_successes=sum(v['success'] for v in results if v['replay_mode'] == 'actions'),
        clipped_successes=sum(v['success'] for v in results if v['replay_mode'] == 'clipped'),
        initial_observations_equal=bool(comparisons) and all(v.get('initial_root_exact') and v.get('initial_state_exact') and v.get('initial_images_exact') for v in comparisons),
        unavailable_comparisons=sum(not v['available'] for v in comparisons),
        sampling_windows=len(samples), noise_windows=len(noise), no_training=True,
        objective_complete=False, report_source_sha256=digest(Path(__file__)))
    write_new_json(out/'acceptance.json', acceptance)
    print(json.dumps(acceptance), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--phase', choices=['audit', 'report'], required=True)
    a = p.parse_args(); root = Path(a.root).resolve()
    if a.phase == 'audit': audit(root)
    else: report(root)


if __name__ == '__main__':
    main()
