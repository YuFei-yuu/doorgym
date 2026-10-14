"""Real-data CPU acceptance of reviewed fixes. No optimizer or simulator is used."""
import argparse
from collections import deque
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from artifacts import write_new_json
from diagnostic_integrity import source_paths, sha256
from forward_diagnostics import inventory

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/'high-level/dp'))
import train_door_dp as training
from action_contract import (LEGACY, FORWARD_V3, normalize_action, denormalize_action,
                             canonicalize_sequence, checkpoint_version)
from door_dp_common import DoorDPPolicyController


class RawFrames:
    """Acceptance-only adapter; production LeRobot reading remains unchanged."""
    def __init__(self, episodes):
        self.episodes = episodes
        self.ends = np.cumsum([len(v['action']) for v in episodes])

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        episode = int(np.searchsorted(self.ends, index, side='right'))
        start = 0 if episode == 0 else int(self.ends[episode-1])
        data = self.episodes[episode]
        return {'episode_index': episode, 'observation.state': data['state'][index-start],
                'action': data['action'][index-start]}


def controller_for_preprocessing(stats):
    controller = DoorDPPolicyController.__new__(DoorDPPolicyController)
    controller.device = torch.device('cpu'); controller.stats = stats
    controller.contract_version = FORWARD_V3; controller.obs_horizon = 16
    controller.obs_buffer = deque(maxlen=16)
    return controller


def legacy_regression(original, previous):
    source = previous/'sources/high-level/dp/door_dp_common.py'
    spec = importlib.util.spec_from_file_location('frozen_legacy_door_dp_common', source)
    old_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(old_module)
    actual_load = torch.load
    def trusted_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return actual_load(*args, **kwargs)
    with patch('torch.load', trusted_load):
        old = old_module.DoorDPPolicyController(original/'training/checkpoints/step_003000.pt', device='cpu')
        current = DoorDPPolicyController(original/'training/checkpoints/step_003000.pt', device='cpu')
    results = []
    windows = [v for v in json.loads((previous/'offline/windows.json').read_text())
               if v['seed'] == 16108 and v['phase'] in ['start', 'grasp', 'through']]
    with np.load(original/'dataset/val_seed_16108/trajectory.npz') as z:
        images = [z[k] for k in ['wrist_handle_mask', 'wrist_masked_depth', 'front_handle_mask', 'front_masked_depth']]
        for value in windows:
            outputs = []
            for controller in [old, current]:
                controller.obs_buffer.clear(); controller.action_queue.clear()
                for frame in np.maximum(np.arange(value['center']-15, value['center']+1), 0):
                    controller.append_observation(z['state'][frame], *[v[frame] for v in images])
                torch.manual_seed(42); controller.sample_action_chunk()
                outputs.append(np.asarray(list(controller.action_queue)))
            results.append(dict(phase=value['phase'], center=value['center'],
                bitwise_equal=bool(np.array_equal(*outputs)), max_action_error=float(abs(outputs[0]-outputs[1]).max())))
    return results


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    p.add_argument('--original', default='experiments/float-push/20261013-forward-dp-v1')
    p.add_argument('--previous', default='experiments/float-push/20261013-forward-dp-diagnostics-v1')
    a = p.parse_args(); root = Path(a.output).resolve(); original = Path(a.original).resolve(); previous = Path(a.previous).resolve()
    torch.set_num_threads(2); started = time.monotonic()
    snapshots = {}
    for source in source_paths(REPO):
        relative = str(source.relative_to(REPO)); snapshots[relative] = sha256(source)
        target = root/'source_after'/relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    write_new_json(root/'source_after_sha256.json', snapshots)
    train = []
    for path in sorted((original/'data/raw_train').glob('episode_*.npz')):
        with np.load(path) as z:
            train.append({key: z[key].copy() for key in ['state', 'action']})
    adapter = RawFrames(train)
    with patch.object(training, '_load_lerobot_dataset', return_value=adapter):
        sequence = training.DoorDPSequenceDataset('acceptance_adapter', 'train', 16, 32, FORWARD_V3)
        legacy_sequence = training.DoorDPSequenceDataset('acceptance_adapter', 'train', 16, 32, LEGACY)
    stats = training.compute_stats(sequence)
    legacy_stats = training.compute_stats(legacy_sequence)
    checkpoint = torch.load(original/'training/checkpoints/step_003000.pt', map_location='cpu', weights_only=False)
    old_stats_exact = all(torch.equal(legacy_stats[key], checkpoint['stats'][key]) for key in legacy_stats)
    canonical = np.concatenate([canonicalize_sequence(v['action'], slice(5, 9)) for v in train])
    normalized = normalize_action(torch.from_numpy(canonical), stats, FORWARD_V3)
    restored = denormalize_action(normalized, stats, FORWARD_V3).numpy()
    raw = np.concatenate([v['action'] for v in train])
    sign_only = bool(np.array_equal(abs(canonical[:, 5:9]), abs(raw[:, 5:9])))
    validation_actions = []
    for path in sorted((original/'data/raw_val').glob('episode_*.npz')):
        with np.load(path) as z:
            validation_actions.append(canonicalize_sequence(z['action'], slice(5, 9)))
    validation = normalize_action(torch.from_numpy(np.concatenate(validation_actions)), stats, FORWARD_V3)
    write_new_json(root/'forward_v3_contract.json', dict(version=FORWARD_V3,
        observation_padding='repeat_first', quaternion_rule='largest_component_positive_then_continuous',
        fitted_split='train', stats={key: value.tolist() for key, value in stats.items()},
        training_frames=len(adapter), training_sequences=len(sequence),
        state_stat_frames=int(sequence.state_stats_mask.sum()), action_stat_frames=int(sequence.action_stats_mask.sum())))
    print('CONTRACT_REAL_DATA_COMPLETE', flush=True)
    # Exercise production __getitem__ against the real LeRobot image reader;
    # only metadata signs come from the already audited identical raw records.
    real = training._load_lerobot_dataset(original/'data/lerobot/train/local/forward_train', 'local/forward_train')
    if len(real) != len(adapter):
        raise ValueError('Raw and real LeRobot frame counts differ')
    sequence.dataset = real
    parity = []
    item_for_center = {center: i for i, center in enumerate(sequence.indices)}
    starts = [0, int(adapter.ends[39]), int(adapter.ends[88])]
    for start in starts:
        controller = controller_for_preprocessing(stats)
        for offset in range(16):
            frame = real[start+offset]
            images = [training._as_chw_u8(frame['observation.images.'+key]).permute(1, 2, 0).numpy()
                      for key in ['wrist_handle_mask', 'wrist_masked_depth', 'front_handle_mask', 'front_masked_depth']]
            controller.append_observation(training._as_numpy(frame['observation.state']), *images)
            if offset in [0, 1, 15]:
                batch = sequence[item_for_center[start+offset]]
                actual = torch.stack([v[0] for v in controller.obs_buffer])
                expected = (batch['state']-stats['state_mean'])/stats['state_std']
                equal_images = all(torch.equal(torch.stack([v[k+1] for v in controller.obs_buffer]), batch[name])
                                   for k, name in enumerate(['mask', 'masked_depth', 'front_mask', 'front_masked_depth']))
                parity.append(dict(center=start+offset, state_exact=torch.equal(actual, expected), images_exact=equal_images))
    write_new_json(root/'real_lerobot_parity.json', parity)
    print('LEROBOT_STREAMING_PARITY_COMPLETE', flush=True)
    regression = legacy_regression(original, previous)
    write_new_json(root/'legacy_policy_regression.json', regression)
    print('LEGACY_POLICY_REGRESSION_COMPLETE', flush=True)
    prior_plan = json.loads((previous/'plan.json').read_text())
    protected = {name: sha256(original/name) == value for name, value in prior_plan['original_protected_hashes'].items()}
    prior_inventory = json.loads((previous/'original_inventory.json').read_text())
    current_inventory = inventory(original)
    previous_hashes = json.loads((previous/'delivery_sha256.json').read_text())
    previous_intact = all(sha256(previous/name) == value for name, value in previous_hashes.items())
    preservation = dict(original_metadata_unchanged=prior_inventory == current_inventory,
                        protected_hashes=protected, previous_delivery_unchanged=previous_intact)
    write_new_json(root/'preservation.json', preservation)
    source_unchanged = all(sha256(REPO/name) == value for name, value in snapshots.items())
    result = dict(no_dp_training=True, no_simulation=True, cuda_initialized=torch.cuda.is_initialized(),
        normalization_version=FORWARD_V3, train_action_coverage=float((abs(normalized) <= 1.+1e-6).all(1).float().mean()),
        float32_roundtrip_max_error=float(abs(restored-canonical).max()), quaternion_changes_sign_only=sign_only,
        validation_outside_frames=int((abs(validation) > 1.+1e-6).any(1).sum()), validation_used_for_fit=False,
        legacy_stats_bitwise_equal=old_stats_exact, legacy_policy_outputs_bitwise_equal=all(v['bitwise_equal'] for v in regression),
        real_lerobot_windows=len(parity), preprocessing_exact=all(v['state_exact'] and v['images_exact'] for v in parity),
        source_snapshots_unchanged=source_unchanged, original_metadata_unchanged=preservation['original_metadata_unchanged'],
        original_protected_hashes_unchanged=all(protected.values()), previous_delivery_unchanged=previous_intact,
        elapsed_sec=time.monotonic()-started)
    result['passed'] = (result['train_action_coverage'] == 1. and result['float32_roundtrip_max_error'] <= 1e-5 and
        sign_only and old_stats_exact and result['legacy_policy_outputs_bitwise_equal'] and result['preprocessing_exact'] and
        source_unchanged and result['original_metadata_unchanged'] and result['original_protected_hashes_unchanged'] and
        previous_intact and not result['cuda_initialized'])
    write_new_json(root/'acceptance.json', result)
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
