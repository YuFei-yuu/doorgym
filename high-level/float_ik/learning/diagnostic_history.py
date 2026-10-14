"""Controlled observation-history and stationary-phase diagnostics, without training."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from diffusers import DDPMScheduler
from artifacts import write_new_json
from diagnostic_contract import quaternion_angle, validate_checkpoint_contract
from forward_diagnostics import cached_prediction, csv_write, VAL_SEEDS, IMAGE_KEYS, digest
from diagnostic_integrity import frozen_phase


@frozen_phase
def run(root):
    original = Path(json.loads((root/'plan.json').read_text())['original'])
    out = root/'offline'
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'dp'))
    from models.door_diffusion_policy import DoorDiffusionPolicy
    torch.set_num_threads(2)
    episodes = {}
    aliases = []
    final = torch.load(original/'training/checkpoints/step_003000.pt', map_location='cpu', weights_only=False)
    state_std = final['stats']['state_std'].numpy()
    for seed in VAL_SEEDS:
        with np.load(original/('dataset/val_seed_%d/trajectory.npz' % seed)) as z:
            phases = z['subtask_index'].reshape(-1)
            episodes[seed] = (z['state'][:16].copy(), [z[k][:16].copy() for k in IMAGE_KEYS], z['action'][15:47].copy())
            for phase in [1, 4, 6]:
                ids = np.flatnonzero(phases == phase)
                early, late = int(ids[len(ids)//2]), int(ids[-1])
                if early < 15 or late+32 > len(phases):
                    continue
                s1, s2 = z['state'][early-15:early+1], z['state'][late-15:late+1]
                image1, image2 = np.stack([z[k][early-15:early+1] for k in IMAGE_KEYS]), np.stack([z[k][late-15:late+1] for k in IMAGE_KEYS])
                a1, a2 = z['action'][early:early+32], z['action'][late:late+32]
                aliases.append(dict(seed=seed, phase=phase, early_frame=early, late_frame=late,
                    state_history_zscore_rms=float(np.sqrt(np.mean(((s1-s2)/state_std)**2))),
                    state_history_max_abs_difference=float(abs(s1-s2).max()),
                    images_exactly_equal=bool(np.array_equal(image1, image2)),
                    image_mean_abs_pixel_difference=float(abs(image1.astype(float)-image2).mean()),
                    future_vx_rmse=float(np.sqrt(np.mean((a1[:, 0]-a2[:, 0])**2))),
                    future_position_rmse_m=float(np.sqrt(np.mean(np.sum((a1[:, 2:5]-a2[:, 2:5])**2, axis=1)))),
                    future_gripper_rmse=float(np.sqrt(np.mean((a1[:, 9]-a2[:, 9])**2))),
                    future_rotation_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(quaternion_angle(a1[:, 5:9], a2[:, 5:9])**2)))),
                    proves_nonidentifiability=bool(np.array_equal(s1, s2) and np.array_equal(image1, image2) and not np.array_equal(a1, a2))))
    results = []
    scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon')
    with torch.inference_mode():
        for step in [1000, 2000, 3000]:
            ckpt = torch.load(original/'training/checkpoints'/('step_%06d.pt' % step), map_location='cpu', weights_only=False)
            validate_checkpoint_contract(ckpt['config'])
            model = DoorDiffusionPolicy(27, 10, 16, 32); model.load_state_dict(ckpt['model']); model.eval()
            for seed, (raw_state, raw_images, raw_target) in episodes.items():
                for mode in ['real_history', 'repeat_current']:
                    ids = np.arange(16) if mode == 'real_history' else np.full(16, 15)
                    state = (torch.from_numpy(raw_state[ids]).unsqueeze(0)-ckpt['stats']['state_mean'])/ckpt['stats']['state_std']
                    images = [torch.from_numpy(v[ids]).permute(0, 3, 1, 2).unsqueeze(0) for v in raw_images]
                    cond = model.encode_obs(state, *images)
                    target = (torch.from_numpy(raw_target).unsqueeze(0)-ckpt['stats']['action_mean'])/ckpt['stats']['action_std']
                    for random_seed in [42, 43, 44]:
                        generator = torch.Generator().manual_seed(random_seed)
                        noise = torch.randn((1, 32, 10), generator=generator)
                        for timestep in [0, 25, 50, 75, 99]:
                            t = torch.tensor([timestep], dtype=torch.long)
                            prediction = cached_prediction(model, cond, scheduler.add_noise(target, noise, t), t)
                            error = (prediction-noise)**2
                            results.append(dict(step=step, seed=seed, center=15, mode=mode,
                                sampling_seed=random_seed, timestep=timestep, noise_mse=float(error.mean()),
                                vx_noise_mse=float(error[:, :, 0].mean())))
            print('HISTORY_CHECKPOINT', step, flush=True)
    write_new_json(out/'stationary_history_pairs.json', aliases)
    write_new_json(out/'padding_ablation.json', results)
    csv_write(out/'padding_ablation.csv', results)
    write_new_json(out/'history_summary.json', dict(stationary_pairs=len(aliases),
        exact_observation_alias_pairs=sum(v['proves_nonidentifiability'] for v in aliases),
        controlled_ablation_windows=len(results), current_frame_and_future_identical=True,
        note='Repeat-current at center15 tests history sensitivity; it does not isolate the real center0 failure cause.',
        no_training=True, cuda_initialized=torch.cuda.is_initialized(), source_sha256=digest(Path(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    run(Path(parser.parse_args().root))
