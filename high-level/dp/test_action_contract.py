"""No optimizer steps: shared preprocessing and legacy compatibility tests."""
from collections import deque
import unittest
from unittest.mock import patch

import numpy as np
import torch
from action_contract import (LEGACY, FORWARD_V3, checkpoint_version, normalize_action,
                             denormalize_action, canonicalize_sequence, observation_indices)
from train_door_dp import DoorDPSequenceDataset, compute_stats
from door_dp_common import DoorDPPolicyController


def frames():
    result = []
    for episode in range(2):
        for i in range(48):
            state = np.zeros(27, np.float32); state[0] = episode*100+i
            state[24] = (1 if i == 0 else -1)*(-1 if episode else 1)
            action = np.zeros(10, np.float32); action[0] = i*.01; action[2] = i
            action[7] = state[24]
            value = {'episode_index': episode, 'observation.state': state, 'action': action}
            for key in ['wrist_handle_mask', 'wrist_masked_depth', 'front_handle_mask', 'front_masked_depth']:
                value['observation.images.'+key] = np.zeros((54, 96, 3), np.uint8)
            result.append(value)
    return result


class ActionContractTests(unittest.TestCase):
    def dataset(self, version):
        with patch('train_door_dp._load_lerobot_dataset', return_value=frames()):
            return DoorDPSequenceDataset('unused', 'unused', 16, 32, version)

    def test_legacy_stats_and_windows_unchanged(self):
        data = self.dataset(LEGACY)
        self.assertEqual(data.indices, [15, 16, 63, 64])
        stats = compute_stats(data)
        raw = frames()
        for key, field in [('state', 'observation.state'), ('action', 'action')]:
            x = torch.stack([torch.from_numpy(v[field]) for v in raw])
            self.assertTrue(torch.equal(stats[key+'_mean'], x.mean(0)))
            self.assertTrue(torch.equal(stats[key+'_std'], x.std(0).clamp_min(1e-6)))
        value = torch.randn(2, 32, 10)
        self.assertTrue(torch.equal(normalize_action(value, stats), (value-stats['action_mean'])/stats['action_std']))

    def test_new_windows_start_at_zero_and_never_cross_episode(self):
        data = self.dataset(FORWARD_V3)
        self.assertEqual(data.indices[0], 0); self.assertEqual(data.indices[17], 48)
        self.assertTrue(torch.all(data[0]['state'][:, 0] == 0))
        self.assertTrue(torch.all(data[17]['state'][:, 0] == 100))
        self.assertEqual(int(data.state_stats_mask.sum()), 34)
        self.assertEqual(int(data.action_stats_mask.sum()), 96)
        self.assertTrue(torch.all(data[0]['action'][:, 7] == 1))

    def test_numpy_torch_roundtrip_constant_and_geometry(self):
        data = self.dataset(FORWARD_V3); stats = compute_stats(data)
        action = torch.stack([data.action_value(frame, i) for i, frame in enumerate(data.dataset)])
        normalized = normalize_action(action, stats, FORWARD_V3)
        self.assertLessEqual(float(abs(normalized).max()), 1.000001)
        self.assertLess(float(abs(denormalize_action(normalized, stats, FORWARD_V3)-action).max()), 1e-5)
        np.testing.assert_allclose(normalize_action(action.numpy(), stats, FORWARD_V3), normalized.numpy(), atol=1e-6)
        self.assertTrue(torch.all(normalized[:, 1] == 0))
        for i, frame in enumerate(data.dataset):
            self.assertAlmostEqual(abs(float(np.dot(action[i, 5:9], frame['action'][5:9]))), 1.)

    def test_streaming_observations_match_training_padding_and_signs(self):
        data = self.dataset(FORWARD_V3); stats = compute_stats(data)
        controller = DoorDPPolicyController.__new__(DoorDPPolicyController)
        controller.device = torch.device('cpu'); controller.stats = stats; controller.contract_version = FORWARD_V3
        controller.obs_horizon = 16; controller.obs_buffer = deque(maxlen=16)
        image = np.zeros((54, 96, 3), np.uint8)
        for i in range(17):
            controller.append_observation(data.dataset[i]['observation.state'], image, image, image, image)
            actual = torch.stack([v[0] for v in controller.obs_buffer])
            expected = (data[i]['state']-stats['state_mean'])/stats['state_std']
            self.assertTrue(torch.equal(actual, expected))
        controller.obs_buffer.clear()
        controller.append_observation(data.dataset[48]['observation.state'], image, image, image, image)
        expected = (data[17]['state']-stats['state_mean'])/stats['state_std']
        self.assertTrue(torch.equal(torch.stack([v[0] for v in controller.obs_buffer]), expected))

    def test_unknown_or_incomplete_checkpoint_contract_rejected(self):
        self.assertEqual(checkpoint_version({}), LEGACY)
        with self.assertRaises(ValueError): checkpoint_version({'action_normalization_version': 'unknown'})
        with self.assertRaises(ValueError): checkpoint_version({'action_normalization_version': FORWARD_V3})

    def test_new_controller_requires_frozen_scheduler_and_rejects_wrong_clipping(self):
        from diffusers import DDPMScheduler
        class DummyPolicy(torch.nn.Module):
            def __init__(self, **kwargs): super().__init__()
        config = dict(action_normalization_version=FORWARD_V3, observation_padding='repeat_first',
            quaternion_rule='largest_component_positive_then_continuous', state_dim=27, action_dim=10)
        stats = dict(state_mean=torch.zeros(27), state_std=torch.ones(27), action_min=torch.zeros(10), action_max=torch.ones(10))
        payload = dict(config=config, stats=stats, model={})
        with patch('torch.load', return_value=payload), patch('models.door_diffusion_policy.DoorDiffusionPolicy', DummyPolicy):
            with self.assertRaisesRegex(ValueError, 'scheduler configuration'):
                DoorDPPolicyController('dummy', device='cpu')
            config['diffusion_scheduler'] = dict(DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', clip_sample=True).config)
            controller = DoorDPPolicyController('dummy', device='cpu')
            self.assertEqual(controller.noise_scheduler.config.num_train_timesteps, 100)
            config['diffusion_scheduler']['clip_sample'] = False
            with self.assertRaisesRegex(ValueError, 'normalized sample clipping'):
                DoorDPPolicyController('dummy', device='cpu')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
