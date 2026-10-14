import json
import tempfile
import unittest
from pathlib import Path

from forward_results import aggregate, episode_result


class ResultTest(unittest.TestCase):
    def test_failures_stay_in_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            case = dict(seed=16108,robot_x=4.1,robot_y=0.)
            row = episode_result(d,case,124)
            result = aggregate([row],1)
            self.assertEqual(result['attempts'],1)
            self.assertEqual(result['successes'],0)

    def test_script_takeover_is_not_autonomous_success(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            row=dict(step=0,root_before=[0,0,0,0,0,0,1],base_after=[0,0,0,0,0,0,1],
                     door_after=[-1.5,0],phase_name='pass_through',
                     extra=dict(command_source='script',dp_action_applied=False,image_valid=True,
                                pass_done=True,through_target=[0,0]))
            (folder/'execution.jsonl').write_text(json.dumps(row)+'\n')
            (folder/'summary.json').write_text(json.dumps(dict(frames=1,opened=True,pass_done=True,
                     success=True,dp_frames=0,script_frames=1,invalid_image_frames=0)))
            case=dict(seed=1,robot_x=4.1,robot_y=0.)
            result=episode_result(folder,case,0)
            self.assertTrue(result['opened']); self.assertTrue(result['pass_done'])
            self.assertFalse(result['success']); self.assertFalse(result['autonomous'])
            row['extra'].update(command_source='dp',dp_action_applied=True)
            (folder/'execution.jsonl').write_text(json.dumps(row)+'\n')
            summary=json.loads((folder/'summary.json').read_text())
            summary.update(dp_frames=1,script_frames=0)
            (folder/'summary.json').write_text(json.dumps(summary))
            self.assertTrue(episode_result(folder,case,0)['success'])
            # Missing/partial logs cannot inherit the summary's successful flag.
            summary['frames']=2
            (folder/'summary.json').write_text(json.dumps(summary))
            self.assertFalse(episode_result(folder,case,0)['success'])


if __name__ == '__main__': unittest.main()
