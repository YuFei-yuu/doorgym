"""Branch initialization must not silently become a partial full-state resume."""
import unittest
import torch
from train_door_dp import initialize_branch


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.model=torch.nn.Linear(2,2)
        self.cfg=dict(state_dim=27,action_dim=10,obs_horizon=16,pred_horizon=32,action_horizon=16,
            num_diffusion_iters=100,image_height=54,image_width=96,action_normalization_version='v3',
            observation_padding='repeat_first',quaternion_rule='continuous')
        self.cp=dict(config=dict(self.cfg),feature_names=['x'],model={k:v.clone()+1 for k,v in self.model.state_dict().items()},
            stats={'state_mean':torch.zeros(27)},optimizer={'unusable_old_state':True},training_state={'unusable_old_rng':True})

    def test_only_weights_consumed(self):
        rng=torch.get_rng_state().clone();optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-4)
        initialize_branch(self.model,self.cp,self.cfg,['x'])
        self.assertFalse(optimizer.state)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in self.model.state_dict().items():self.assertTrue(torch.equal(v,self.cp['model'][k]))

    def test_contract_rejected(self):
        self.cp['config']['pred_horizon']=16
        with self.assertRaises(ValueError):initialize_branch(self.model,self.cp,self.cfg,['x'])

    def test_features_rejected(self):
        with self.assertRaises(ValueError):initialize_branch(self.model,self.cp,self.cfg,['wrong'])

    def test_nonfinite_rejected(self):
        self.cp['stats']['state_mean'][0]=float('nan')
        with self.assertRaises(ValueError):initialize_branch(self.model,self.cp,self.cfg,['x'])


if __name__=='__main__':unittest.main()
