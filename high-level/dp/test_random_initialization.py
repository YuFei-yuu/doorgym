"""Fresh random provenance checks must reject loading/matching the wrong state."""
import copy
import random
import unittest
import numpy as np
import torch
from train_door_dp import verify_random_initialization
from training_state import ResumableBatchSampler,capture_rng


class RandomAuditTests(unittest.TestCase):
    def setUp(self):
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        self.model=torch.nn.Linear(2,2);self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-4)
        self.stats={'state_mean':torch.zeros(2)};self.sampler=ResumableBatchSampler(1415,8,43)
        self.config=dict(root='/new/training/only',repo_id='local/single_B16040',seed=42,batch_size=8,lr=1e-4,
            obs_horizon=16,pred_horizon=32,action_horizon=16,num_diffusion_iters=100,dataset_frames=1446,
            dataset_sequences=1415,action_normalization_version='forward_v3',completed_updates=0)
        self.expected=dict(config=dict(self.config),model=copy.deepcopy(self.model.state_dict()),stats=copy.deepcopy(self.stats),
            optimizer=self.optimizer.state_dict(),training_state=dict(sampler=self.sampler.state_dict(),rng=capture_rng()))

    def check(self):verify_random_initialization(self.model,self.optimizer,self.stats,self.sampler,self.config,self.expected)

    def test_pure_comparison_preserves_fresh_state(self):
        state=copy.deepcopy(self.model.state_dict());rng=torch.get_rng_state().clone();self.check()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(self.optimizer.state)
        for k,v in state.items():self.assertTrue(torch.equal(v,self.model.state_dict()[k]))

    def test_wrong_weights_rejected(self):
        self.expected['model']['weight']+=1
        with self.assertRaises(ValueError):self.check()

    def test_old_optimized_state_rejected(self):
        self.expected['config']['completed_updates']=5000
        with self.assertRaises(ValueError):self.check()

    def test_wrong_stats_rejected(self):
        self.expected['stats']['state_mean']+=1
        with self.assertRaises(ValueError):self.check()

    def test_consumed_rng_rejected(self):
        torch.randn(2)
        with self.assertRaises(ValueError):self.check()

    def test_changed_contract_rejected(self):
        self.expected['config']['batch_size']=64
        with self.assertRaises(ValueError):self.check()


if __name__=='__main__':unittest.main()
