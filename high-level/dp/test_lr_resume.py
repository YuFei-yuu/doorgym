import copy,unittest
import torch
from train_door_dp import validate_resume_contract,verify_full_restoration,exact_state
from training_state import ResumableBatchSampler,capture_rng

class LRResumeTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(root='/accepted',repo_id='local/single_B16040',batch_size=8,lr=1e-4,obs_horizon=16,pred_horizon=32,
            action_horizon=16,num_diffusion_iters=100,seed=42,state_dim=27,action_dim=10,recording_metadata={'approved':True},
            dataset_sequences=1415,dataset_frames=1446,deterministic_algorithms=True)
        self.model=torch.nn.Linear(2,2);self.opt=torch.optim.AdamW(self.model.parameters(),lr=1e-4,weight_decay=1e-6)
        self.model(torch.ones(1,2)).sum().backward();self.opt.step()
        self.sampler=ResumableBatchSampler(1415,8,43);next(iter(self.sampler));self.stats={'x':torch.zeros(2)}
        self.cp=dict(config=copy.deepcopy(self.cfg),feature_names=['x'],model=copy.deepcopy(self.model.state_dict()),
            optimizer=copy.deepcopy(self.opt.state_dict()),stats=copy.deepcopy(self.stats),training_state=dict(sampler=self.sampler.state_dict(),rng=capture_rng()))
    def test_default_rejects_lr_change(self):
        self.cfg['lr']=1e-5
        with self.assertRaises(ValueError):validate_resume_contract(self.cfg,self.cp,['x'])
    def test_explicit_lower_lr_allowed(self):
        self.cfg['lr']=1e-5;validate_resume_contract(self.cfg,self.cp,['x'],1e-4)
    def test_wrong_source_lr_rejected(self):
        self.cfg['lr']=1e-5
        with self.assertRaises(ValueError):validate_resume_contract(self.cfg,self.cp,['x'],1e-3)
    def test_other_contract_still_rejected(self):
        self.cfg.update(lr=1e-5,batch_size=64)
        with self.assertRaises(ValueError):validate_resume_contract(self.cfg,self.cp,['x'],1e-4)
    def test_unchanged_lr_resume(self):validate_resume_contract(self.cfg,self.cp,['x'])
    def test_optimizer_config_disagreement_rejected(self):
        self.cp['optimizer']['param_groups'][0]['lr']=1e-5
        with self.assertRaises(ValueError):validate_resume_contract(self.cfg,self.cp,['x'])
    def test_full_restoration_pure_comparison(self):
        before=capture_rng();verify_full_restoration(self.model,self.opt,self.stats,self.sampler,self.cp)
        self.assertTrue(exact_state(before,capture_rng()))
    def test_changed_moments_rejected(self):
        next(iter(self.opt.state.values()))['exp_avg']+=1
        with self.assertRaises(ValueError):verify_full_restoration(self.model,self.opt,self.stats,self.sampler,self.cp)
    def test_rng_or_sampler_changed_rejected(self):
        torch.randn(1)
        with self.assertRaises(ValueError):verify_full_restoration(self.model,self.opt,self.stats,self.sampler,self.cp)
if __name__=='__main__':unittest.main()
