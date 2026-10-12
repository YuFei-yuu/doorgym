"""Recording isolation, original train statistics and portable analysis checks."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from artifacts import resolve_recorded_path,write_new_json
from checks import audit_run
from dataset_contract import validate_episode

NAMES=['vx','yaw','ee_x','ee_y','ee_z','ee_qx','ee_qy','ee_qz','ee_qw','gripper']


def episode(path,offset=0):
    state=np.full((4,27),offset,np.float32);action=np.zeros((4,10),np.float32);action[:,8]=1
    np.savez(path,schema='float_push_upstream_ee_v3',execution_profile='upstream_wholebody_float_v3',state_names=['state_'+str(i) for i in range(27)],action_names=NAMES,state=state,next_state=state.copy(),action=action,images=np.zeros((4,4,54,96),np.uint8),frame=np.arange(4),next_frame=np.arange(1,5))
    return state,action


class AnalysisTests(unittest.TestCase):
    def test_portable_paths_and_new_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir();(root/'episode.npz').touch()
            self.assertEqual(resolve_recorded_path(root,'/server/experiments/run/episode.npz'),root/'episode.npz')
            out=root/'diagnostic.json';write_new_json(out,dict(overflow=np.inf,finite=np.float64(2)))
            self.assertEqual(json.loads(out.read_text()),dict(overflow=None,finite=2))
            with self.assertRaises(FileExistsError):write_new_json(out,{})

    def test_split_and_train_only_statistics(self):
        import torch
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state,action=episode(root/'train.npz');episode(root/'val.npz',100)
            entries=[dict(episode='train.npz',seed=1,split='train',success=True),dict(episode='val.npz',seed=2,split='val',success=True)]
            (root/'manifest.json').write_text(json.dumps(entries));self.assertTrue(audit_run(root)['passed'])
            stats={}
            for key,values in [('state',state),('action',action)]:
                t=torch.from_numpy(values);stats[key+'_mean']=t.mean(0);stats[key+'_std']=t.std(0).clamp_min(1e-6)
            torch.save(dict(config=dict(state_dim=27,action_dim=10),stats=stats),root/'model.pt')
            self.assertTrue(audit_run(root,'model.pt')['passed'])
            stats['state_mean']+=50;torch.save(dict(config=dict(state_dim=27,action_dim=10),stats=stats),root/'model.pt')
            self.assertFalse(audit_run(root,'model.pt')['checks']['normalization_train_only'])
            entries[1]['seed']=1;(root/'manifest.json').write_text(json.dumps(entries));self.assertFalse(audit_run(root)['checks']['disjoint_seeds'])

    def test_invalid_quaternion_and_transition(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'episode.npz';episode(p)
            with np.load(p) as z:data={k:z[k].copy() for k in z.files}
            data['action'][:,8]=0
            with self.assertRaisesRegex(ValueError,'quaternion'):validate_episode(data)
            data['action'][:,8]=1;data['next_state'][0,0]=1
            with self.assertRaisesRegex(ValueError,'misaligned'):validate_episode(data)

    def test_pair_requires_same_executor(self):
        from analyze import compare_pair
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);e=root/'expert';p=root/'policy';e.mkdir();p.mkdir()
            (e/'summary.json').write_text(json.dumps(dict(mode='expert',execution_profile='a')))
            (p/'summary.json').write_text(json.dumps(dict(mode='policy',execution_profile='b')))
            with self.assertRaisesRegex(ValueError,'different execution profiles'):compare_pair(e,p,root,0)


if __name__=='__main__':unittest.main()
