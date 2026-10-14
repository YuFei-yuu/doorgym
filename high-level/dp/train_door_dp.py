import argparse
import json
import time
import random
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .door_dp_common import ACTION_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH, import_lerobot_or_raise
    from .models.door_diffusion_policy import DoorDiffusionPolicy
    from .training_state import ResumableBatchSampler, capture_rng, restore_rng
    from .action_contract import (LEGACY, FORWARD_V3, quaternion_sign, observation_indices,
                                  normalize_action, checkpoint_version)
except ImportError:
    from door_dp_common import ACTION_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH, import_lerobot_or_raise
    from models.door_diffusion_policy import DoorDiffusionPolicy
    from training_state import ResumableBatchSampler, capture_rng, restore_rng
    from action_contract import (LEGACY, FORWARD_V3, quaternion_sign, observation_indices,
                                 normalize_action, checkpoint_version)


DP_ROOT = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = DP_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train a wrist-mask/depth conditioned Door Diffusion Policy.")
    parser.add_argument("--root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_dp")
    parser.add_argument("--run_name", type=str, default="debug")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--obs_horizon", type=int, default=16)
    parser.add_argument("--pred_horizon", type=int, default=32)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--num_diffusion_iters", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="door-dp")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_save_checkpoints", action="store_true", help="Upload checkpoint files to wandb.")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--recording_metadata", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Full-state checkpoint to continue.")
    parser.add_argument('--action_contract', choices=['legacy', 'forward_v3'], default='legacy')
    parser.add_argument('--wandb_mode', choices=['offline', 'disabled'], default='offline')
    parser.add_argument('--stop_on_nonfinite_loss', action='store_true',
                        help='Abort before backward/optimizer when the computed loss is nonfinite')
    return parser.parse_args()


def _load_lerobot_dataset(root, repo_id):
    import_lerobot_or_raise()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = _resolve_lerobot_root(root, repo_id)
    try:
        return LeRobotDataset(repo_id=repo_id, root=str(root))
    except TypeError:
        return LeRobotDataset(repo_id, root=str(root))


def _resolve_lerobot_root(root, repo_id):
    root = Path(root).expanduser().resolve()
    if (root / "meta" / "info.json").is_file():
        return root
    nested = root / repo_id
    if (nested / "meta" / "info.json").is_file():
        return nested
    return root


def _field(frame, key):
    if key in frame:
        return frame[key]
    if hasattr(frame, "__getitem__"):
        return frame[key]
    raise KeyError(key)


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    try:
        from PIL import Image

        if isinstance(x, Image.Image):
            return np.asarray(x)
    except Exception:
        pass
    return np.asarray(x)


def _as_chw_u8(x):
    arr = _as_numpy(x)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        chw = arr
    else:
        chw = np.transpose(arr[..., :3], (2, 0, 1))
    if chw.shape[-2:] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        import cv2

        hwc = np.transpose(chw, (1, 2, 0))
        hwc = cv2.resize(hwc, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        chw = np.transpose(hwc, (2, 0, 1))
    # LeRobot 0.3.x decodes images as CHW floats in [0, 1]. Recover
    # original uint8 camera pixels before the source encoder divides by 255.
    if np.issubdtype(chw.dtype, np.floating) and np.isfinite(chw).all() and chw.min() >= 0 and chw.max() <= 1:
        chw = np.rint(chw * 255.0)
    return torch.from_numpy(chw.astype(np.uint8))


def _image_or_zeros(frame, key):
    try:
        return _as_chw_u8(_field(frame, key))
    except Exception:
        return torch.zeros(3, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.uint8)


def _scalar_int(x, default=0):
    if x is None:
        return int(default)
    arr = _as_numpy(x)
    return int(arr.reshape(-1)[0])


class DoorDPSequenceDataset(Dataset):
    def __init__(self, root, repo_id, obs_horizon, pred_horizon, contract_version=LEGACY):
        self.dataset = _load_lerobot_dataset(root, repo_id)
        self.contract_version = contract_version
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.length = len(self.dataset)
        minimum = self.pred_horizon if contract_version == FORWARD_V3 else self.obs_horizon+self.pred_horizon
        if self.length < minimum:
            raise ValueError("Dataset is too short for the requested horizons.")
        self.episodes = []
        self.state_signs, self.action_signs, self.episode_starts = [], [], []
        previous_episode = None
        previous_state_quat = previous_action_quat = None
        start = 0
        for i in range(self.length):
            frame = self.dataset[i]
            try:
                ep = _scalar_int(_field(frame, "episode_index"))
            except Exception:
                ep = 0
            self.episodes.append(ep)
            if ep != previous_episode:
                start = i
                previous_state_quat = previous_action_quat = None
            self.episode_starts.append(start)
            if contract_version == FORWARD_V3:
                state_q = _as_numpy(_field(frame, 'observation.state'))[-5:-1]
                action_q = _as_numpy(_field(frame, 'action'))[5:9]
                state_sign = quaternion_sign(state_q, previous_state_quat)
                action_sign = quaternion_sign(action_q, previous_action_quat)
                self.state_signs.append(state_sign); self.action_signs.append(action_sign)
                previous_state_quat = state_q*state_sign
                previous_action_quat = action_q*action_sign
            previous_episode = ep
        self.indices = self._build_indices()
        self.state_stats_mask = np.zeros(self.length, dtype=bool)
        self.action_stats_mask = np.zeros(self.length, dtype=bool)
        for center in self.indices:
            self.state_stats_mask[max(self.episode_starts[center], center-self.obs_horizon+1):center+1] = True
            self.action_stats_mask[center:center+self.pred_horizon] = True

    def _build_indices(self):
        indices = []
        start = 0
        while start < self.length:
            ep = self.episodes[start]
            end = start + 1
            while end < self.length and self.episodes[end] == ep:
                end += 1
            first = start if self.contract_version == FORWARD_V3 else start + self.obs_horizon - 1
            last = end - self.pred_horizon
            indices.extend(range(first, max(last + 1, first)))
            start = end
        if not indices:
            raise ValueError("No valid train sequences found. Record longer episodes or reduce horizons.")
        return indices

    def _frame(self, idx):
        return self.dataset[int(idx)]

    def state_value(self, frame, idx):
        result = torch.as_tensor(_as_numpy(_field(frame, 'observation.state')), dtype=torch.float32).clone()
        if self.contract_version == FORWARD_V3:
            result[-5:-1] *= self.state_signs[int(idx)]
        return result

    def action_value(self, frame, idx):
        result = torch.as_tensor(_as_numpy(_field(frame, 'action')), dtype=torch.float32).clone()
        if self.contract_version == FORWARD_V3:
            result[5:9] *= self.action_signs[int(idx)]
        return result

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        center = self.indices[item]
        obs_ids = (observation_indices(center, self.obs_horizon, self.episode_starts[center])
                   if self.contract_version == FORWARD_V3 else range(center-self.obs_horizon+1, center+1))
        action_ids = range(center, center + self.pred_horizon)
        states, masks, depths, front_masks, front_depths = [], [], [], [], []
        for idx in obs_ids:
            frame = self._frame(idx)
            states.append(self.state_value(frame, idx))
            masks.append(_image_or_zeros(frame, "observation.images.wrist_handle_mask"))
            depths.append(_image_or_zeros(frame, "observation.images.wrist_masked_depth"))
            front_masks.append(_image_or_zeros(frame, "observation.images.front_handle_mask"))
            front_depths.append(_image_or_zeros(frame, "observation.images.front_masked_depth"))
        actions = []
        for idx in action_ids:
            frame = self._frame(idx)
            actions.append(self.action_value(frame, idx))
        return {
            "state": torch.stack(states, dim=0),
            "mask": torch.stack(masks, dim=0),
            "masked_depth": torch.stack(depths, dim=0),
            "front_mask": torch.stack(front_masks, dim=0),
            "front_masked_depth": torch.stack(front_depths, dim=0),
            "action": torch.stack(actions, dim=0),
        }


def compute_stats(seq_dataset):
    base = seq_dataset.dataset
    states, actions = [], []
    for i in range(len(base)):
        frame = base[i]
        if seq_dataset.contract_version == LEGACY or seq_dataset.state_stats_mask[i]:
            states.append(seq_dataset.state_value(frame, i))
        if seq_dataset.contract_version == LEGACY or seq_dataset.action_stats_mask[i]:
            actions.append(seq_dataset.action_value(frame, i))
    states = torch.stack(states, dim=0)
    actions = torch.stack(actions, dim=0)
    result = {
        "state_mean": states.mean(dim=0),
        "state_std": states.std(dim=0).clamp_min(1e-6),
        "action_mean": actions.mean(dim=0),
        "action_std": actions.std(dim=0).clamp_min(1e-6),
    }
    if seq_dataset.contract_version == FORWARD_V3:
        result.update(action_min=actions.amin(0), action_max=actions.amax(0))
    return result


def normalize_batch(batch, stats, device, contract_version=LEGACY):
    state = batch["state"].to(device)
    action = batch["action"].to(device)
    state = (state - stats["state_mean"].to(device)) / stats["state_std"].to(device)
    action = normalize_action(action, stats, contract_version)
    return (
        state,
        batch["mask"].to(device),
        batch["masked_depth"].to(device),
        batch["front_mask"].to(device),
        batch["front_masked_depth"].to(device),
        action,
    )


def save_checkpoint(path, model, optimizer, stats, config, feature_names, sampler, elapsed_sec):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stats": {k: v.cpu() for k, v in stats.items()},
            "config": config,
            "feature_names": feature_names,
            "action_names": ACTION_NAMES,
            "training_state": dict(sampler=sampler.state_dict(), rng=capture_rng(),
                                   elapsed_sec=elapsed_sec),
        }
    # A step checkpoint is immutable. model_latest is only a convenience pointer.
    with path.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    latest = path.with_name('model_latest.pt')
    temporary = latest.with_suffix('.pt.tmp')
    shutil.copyfile(path, temporary)
    temporary.replace(latest)


def require_finite_loss(loss, next_step):
    if not bool(torch.isfinite(loss).all().item()):
        raise FloatingPointError('Nonfinite loss before optimizer update %d' % next_step)


def main():
    args = parse_args()
    if args.steps <= 0 or args.save_interval <= 0 or args.log_interval <= 0:
        raise ValueError('Update and logging intervals must be positive')
    if args.num_workers != 0:
        raise ValueError('Exact sample-position resume requires num_workers=0')
    torch.set_num_threads(2)
    # Fresh processes must choose reproducible kernels as well as restore RNG.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dataset_root = _resolve_lerobot_root(args.root, args.repo_id)
    print(f"Using LeRobot dataset root: {dataset_root}", flush=True)
    contract_version = FORWARD_V3 if args.action_contract == 'forward_v3' else LEGACY
    dataset = DoorDPSequenceDataset(dataset_root, args.repo_id, args.obs_horizon, args.pred_horizon, contract_version)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if checkpoint and checkpoint_version(checkpoint['config']) != contract_version:
        raise ValueError('Resume action contract differs; use random initialization in a new run')
    stats = checkpoint['stats'] if checkpoint else compute_stats(dataset)
    sidecar = dataset_root / "door_dp_feature_names.json"
    if sidecar.exists():
        with open(sidecar, "r", encoding="utf-8") as f:
            feature_names = json.load(f).get("state", [])
    else:
        feature_names = [f"state_{i}" for i in range(stats["state_mean"].numel())]

    sampler = ResumableBatchSampler(len(dataset), args.batch_size,
                                    (args.seed if args.seed is not None else torch.initial_seed()) + 1)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        # Iterator construction must not consume model/noise RNG on resume.
        generator=torch.Generator().manual_seed(0),
    )
    if len(loader) == 0:
        raise ValueError("No complete training batch; dataset must contain at least batch_size sequences")
    model = DoorDiffusionPolicy(
        state_dim=stats["state_mean"].numel(),
        action_dim=len(ACTION_NAMES),
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
    ).to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    if checkpoint and contract_version == FORWARD_V3:
        noise_scheduler = DDPMScheduler.from_config(checkpoint['config']['diffusion_scheduler'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    run_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "logs" / "door-dp" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    config = vars(args).copy()
    config.update(action_normalization_version=contract_version,
                  observation_padding='repeat_first' if contract_version == FORWARD_V3 else 'none',
                  quaternion_rule='largest_component_positive_then_continuous' if contract_version == FORWARD_V3 else 'recorded')
    if contract_version == FORWARD_V3:
        config['diffusion_scheduler'] = dict(noise_scheduler.config)
    if args.recording_metadata:
        config["recording_metadata"] = json.loads(Path(args.recording_metadata).read_text())
    config.update(
        {
            "state_dim": stats["state_mean"].numel(),
            "action_dim": len(ACTION_NAMES),
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
        }
    )
    config['dataset_sequences'] = len(dataset)
    config['dataset_frames'] = dataset.length
    config['deterministic_algorithms'] = True
    step, prior_elapsed = 0, 0.0
    if checkpoint:
        if 'training_state' not in checkpoint:
            raise ValueError('Checkpoint lacks exact resume state; refusing approximate resume')
        keys = ['root', 'repo_id', 'batch_size', 'lr', 'obs_horizon', 'pred_horizon',
                'action_horizon', 'num_diffusion_iters', 'seed', 'state_dim', 'action_dim',
                'recording_metadata', 'dataset_sequences', 'dataset_frames', 'deterministic_algorithms']
        if any(config.get(k) != checkpoint['config'].get(k) for k in keys):
            raise ValueError('Training contract changed during resume')
        if checkpoint['feature_names'] != feature_names:
            raise ValueError('Resume feature names changed')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        sampler.load_state_dict(checkpoint['training_state']['sampler'])
        step = int(checkpoint['config']['completed_updates'])
        prior_elapsed = float(checkpoint['training_state']['elapsed_sec'])
        if args.steps <= step:
            raise ValueError('Resume target must exceed completed updates')
        restore_rng(checkpoint['training_state']['rng'])
    elif (run_dir / 'metrics.jsonl').exists() or (ckpt_dir / 'model_latest.pt').exists():
        raise FileExistsError('Fresh training requires a new output directory')
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("Install wandb in the training env, e.g. `pip install wandb`.") from exc
        wandb_run = wandb.init(
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.run_name,
            config=config,
            dir=str(run_dir),
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

    t0 = time.time()
    while step < args.steps:
        for batch in loader:
            state, mask, masked_depth, front_mask, front_masked_depth, action = normalize_batch(batch, stats, device, contract_version)
            noise = torch.randn_like(action)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (action.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_action = noise_scheduler.add_noise(action, noise, timesteps)
            pred = model(noisy_action, timesteps, state, mask, masked_depth, front_mask, front_masked_depth)
            loss = F.mse_loss(pred, noise)
            if args.stop_on_nonfinite_loss:
                require_finite_loss(loss, step + 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            config["completed_updates"] = step
            if step % 10 == 0 or step == args.steps:
                progress = run_dir / "progress.json"
                temporary = progress.with_suffix(".tmp")
                temporary.write_text(json.dumps({"step": step, "target_steps": args.steps,
                                                  "elapsed_sec": prior_elapsed + time.time() - t0}))
                temporary.replace(progress)
            if step % args.log_interval == 0:
                elapsed = prior_elapsed + time.time() - t0
                metrics = {"train/step": step, "train/loss": float(loss.item()),
                           "train/lr": float(optimizer.param_groups[0]["lr"]),
                           "train/elapsed_sec": float(elapsed)}
                with (run_dir / "metrics.jsonl").open("a") as log:
                    log.write(json.dumps(metrics) + "\n")
                print(f"[step {step}] loss={loss.item():.6f} elapsed={elapsed:.1f}s", flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step": step,
                            "train/loss": float(loss.item()),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/elapsed_sec": float(elapsed),
                        }
                    )
            if step % args.save_interval == 0 or step == args.steps:
                save_checkpoint(ckpt_dir / ('step_%06d.pt' % step), model, optimizer,
                                stats, config, feature_names, sampler, prior_elapsed + time.time() - t0)
                if wandb_run is not None and args.wandb_save_checkpoints:
                    wandb_run.save(str(ckpt_dir / "model_latest.pt"))
            if step >= args.steps:
                break
    if wandb_run is not None and args.wandb_save_checkpoints:
        wandb_run.save(str(ckpt_dir / "model_latest.pt"))
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Saved {ckpt_dir / 'model_latest.pt'}")


if __name__ == "__main__":
    main()
