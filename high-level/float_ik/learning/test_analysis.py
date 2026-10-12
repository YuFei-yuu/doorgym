"""Regression checks for current recording audit and portable artifact analysis."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from artifacts import resolve_recorded_path,write_new_json
from checks import audit_run
from schema import DEFAULT_SCHEMA,contract


class AnalysisTests(unittest.TestCase):
    def test_mirror_resolution_and_nonfinite_diagnostic_serialization(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';p=root/'data/ep.npz';p.parent.mkdir(parents=True);p.touch()
            self.assertEqual(resolve_recorded_path(root,'/server/experiments/run/data/ep.npz'),p)
            out=root/'diagnostic.json';write_new_json(out,dict(overflow=np.inf,finite=np.float64(2)))
            self.assertEqual(json.loads(out.read_text()),dict(overflow=None,finite=2))
            with self.assertRaises(FileExistsError):write_new_json(out,{})

    def test_current_recording_alignment_and_split_leak_detection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);names,states=contract(DEFAULT_SCHEMA)
            action=np.zeros((3,11),np.float32);action[:,9]=1
            state=np.zeros((3,39),np.float32);state[1:,-11:]=action[:-1]
            following=np.vstack([state[1:],state[-1:]])
            path=root/'episode.npz';np.savez(path,schema=DEFAULT_SCHEMA,action_names=names,state_names=states,action=action,state=state,next_state=following,images=np.zeros((3,4,54,96),np.uint8),frame=np.arange(3),next_frame=np.arange(1,4))
            item=dict(episode=str(path),seed=1,split='train',success=True)
            (root/'manifest.json').write_text(json.dumps([item]));self.assertTrue(audit_run(root)['passed'])
            (root/'manifest.json').write_text(json.dumps([item,dict(item,split='val')]))
            result=audit_run(root);self.assertFalse(result['passed']);self.assertFalse(result['checks']['disjoint_seeds'])

    def test_different_execution_profiles_cannot_be_paired(self):
        from analyze import compare_pair
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);e=root/'expert';p=root/'policy';e.mkdir();p.mkdir()
            (e/'summary.json').write_text(json.dumps(dict(mode='expert',execution_profile='default')))
            (p/'summary.json').write_text(json.dumps(dict(mode='policy',execution_profile='diagnostic')))
            with self.assertRaisesRegex(ValueError,'different execution profiles'):compare_pair(e,p,root,0)


if __name__=='__main__':unittest.main()
