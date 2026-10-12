"""CPU checks for the new Cartesian contract; run without creating a simulator."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from schema import EE_SCHEMA, JOINT_SCHEMA, contract, model_config
from ee_actions import normalize_ee_action, expert_pose_reference
from dataset_contract import validate_episode, validate_replay
from kinematics import FK


class CartesianContractTests(unittest.TestCase):
    def test_expert_reference_limits_without_changing_base_or_gripper(self):
        previous=np.array([0,0,0,0,0,0,0,0,0,1,-1.],dtype=np.float32)
        desired=np.array([.2,.1,.01,1,0,0,0,0,1,0,-.5],dtype=np.float32)
        result=expert_pose_reference(desired,previous,np.zeros(3))
        self.assertAlmostEqual(float(result[3]),.006,places=7)
        angle=2*np.arctan2(np.linalg.norm(result[6:9]),abs(result[9]))
        self.assertAlmostEqual(float(angle),.035,places=6)
        np.testing.assert_array_equal(result[:3],desired[:3])
        self.assertEqual(result[-1],desired[-1])
        for _ in range(100):previous=expert_pose_reference(desired,previous,np.zeros(3))
        self.assertLessEqual(np.linalg.norm(previous[3:6]),.0600001)

    def test_action_normalization_and_rejection(self):
        raw = np.array([4., -2., 3., 2., .2, .7, 0., 0., 0., -2., 3.])
        applied = normalize_ee_action(raw, -1.57, 0., [0, 0, 0, 1])
        np.testing.assert_allclose(applied[:3], [.6, -.3, .4])
        np.testing.assert_array_equal(applied[3:6], raw[3:6].astype(np.float32))
        np.testing.assert_array_equal(applied[6:10], [0, 0, 0, 1])
        self.assertEqual(applied[-1], 0.)
        rng=np.random.RandomState(17)
        for _ in range(100):
            raw[6:10]=rng.normal(size=4)
            first=normalize_ee_action(raw,-1.57,0.)
            np.testing.assert_array_equal(normalize_ee_action(first,-1.57,0.,first[6:10]),first)
        for bad in [np.zeros(10), np.zeros(11), np.full(11, np.nan)]:
            with self.assertRaises(ValueError):
                normalize_ee_action(bad, -1.57, 0.)

    def test_dataset_contract_prevents_silent_old_label_reuse(self):
        actions, states = contract(EE_SCHEMA)
        a = np.zeros((3, 11), np.float32); a[:, 9] = 1
        s = np.zeros((3, 39), np.float32); s[1:, -11:] = a[:-1]
        z = dict(schema=np.array(EE_SCHEMA), action_names=np.array(actions), state_names=np.array(states), action=a, state=s, images=np.zeros((3,4,54,96),np.uint8))
        validate_episode(z, EE_SCHEMA)
        validate_replay({k:z[k] for k in ('schema','action_names','action')}, EE_SCHEMA)
        validate_replay(dict(action=np.zeros((3,10))), JOINT_SCHEMA)
        with self.assertRaises(ValueError): validate_episode(z, JOINT_SCHEMA)
        z['state'][1,-1] = 1
        with self.assertRaises(ValueError): validate_episode(z, EE_SCHEMA)
        del z['schema']
        with self.assertRaises(ValueError): validate_episode(z, EE_SCHEMA)

    def test_return_target_fk_includes_orientation_and_fixed_tip(self):
        urdf = '''<robot name="test"><link name="root"/><link name="arm"/><link name="tip"/>
        <joint name="joint1" type="revolute"><parent link="root"/><child link="arm"/><axis xyz="0 0 1"/></joint>
        <joint name="offset" type="fixed"><parent link="arm"/><child link="tip"/><origin xyz="1 0 0"/></joint></robot>'''
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.urdf';p.write_text(urdf)
            pos, quat = FK(p, 'tip', ['joint1']).pose([np.pi/2])
        np.testing.assert_allclose(pos, [0,1,0], atol=1e-8)
        np.testing.assert_allclose(abs(quat), [0,0,2**-.5,2**-.5], atol=1e-8)

    def test_network_gradient_sampling_and_checkpoint_schema(self):
        import torch
        from policy import DoorDiffusionPolicy, Policy, scheduler, images_to_torch
        torch.set_num_threads(2)
        config=model_config(EE_SCHEMA)
        self.assertEqual(config['state_dim'],39);self.assertEqual(config['action_dim'],11)
        torch.manual_seed(42)
        model=DoorDiffusionPolicy(**config)
        state=torch.zeros(1,4,39)
        images=images_to_torch(np.zeros((1,4,4,54,96),np.uint8),'cpu')
        action=torch.zeros(1,16,11);noise=torch.randn_like(action);t=torch.tensor([30])
        loss=(model(scheduler().add_noise(action,noise,t),t,state,*images)-noise).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        actions,states=contract(EE_SCHEMA)
        ck=dict(schema=EE_SCHEMA,model_config=config,action_names=actions,state_names=states,model=model.state_dict(),step=0,stats=dict(state_mean=torch.zeros(39),state_std=torch.ones(39),action_mean=torch.zeros(11),action_std=torch.ones(11)))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ck.pt';torch.save(ck,p)
            with self.assertRaises(ValueError): Policy(p,device='cpu',schema=JOINT_SCHEMA)
            policy=Policy(p,device='cpu',schema=EE_SCHEMA)
            obs=dict(state=np.zeros(39,np.float32),images=np.zeros((4,54,96),np.uint8))
            out,decision=policy.act(obs)
            self.assertTrue(decision);self.assertEqual(out.shape,(11,));self.assertTrue(np.isfinite(out).all())
            with self.assertRaises(ValueError): policy.act(dict(obs,state=np.zeros(38)))


if __name__ == '__main__':
    unittest.main()
