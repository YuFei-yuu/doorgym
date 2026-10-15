import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO=Path('/data/zyf/video2door')
sys.path.insert(0,str(REPO/'high-level/float_ik/learning/run_tools'))
from launch import launch

spec=importlib.util.spec_from_file_location('stop_control',Path(__file__).with_name('stop_after_3000.py'))
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)


class StopTest(unittest.TestCase):
    def test_guard_precedes_gpu_selection_and_process_launch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'resources').mkdir()
            (root/'resources/training__004000.json').write_text('{}')
            with patch('launch.subprocess.check_output',side_effect=AssertionError('GPU query must not run')):
                self.assertRaises(FileExistsError,launch,root,'training__004000',['never-run'])

    def fixture(self,root,expected=True):
        (root/'stop_after_updates.json').write_text(json.dumps(dict(supervisor_pid=99999999,supervisor_start_ticks='0')))
        error=("FileExistsError('Stage already attempted: %s')"%(root/'resources/training__004000.json')
               if expected else "RuntimeError('Unrelated genuine failure')")
        (root/'pipeline_state.json').write_text(json.dumps(dict(phase='error',error=error)))
        folder=root/'evaluation/step_003000';folder.mkdir(parents=True)
        (folder/'aggregate.json').write_text(json.dumps(dict(attempts=9,scheduled=9,all_scheduled_completed=True)))
        (root/'checkpoint_checks_003000.json').write_text('{"passed":true}')
        folder=root/'training/checkpoints';folder.mkdir(parents=True)
        (folder/'step_003000.pt').write_bytes(b'unit-test fixture')
        (folder/'step_003000.json').write_text(json.dumps(dict(completed_updates=3000,
                 sha256=hashlib.sha256(b'unit-test fixture').hexdigest())))

    def test_records_completed_requested_stop(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            control.watch(root)
            result=json.loads((root/'pipeline_state.json').read_text())
            self.assertEqual(result['phase'],'stopped_by_user')
            self.assertEqual(result['completed_updates'],3000)
            self.assertFalse(result['objective_complete'])
            self.assertTrue((root/'user_stop_3000/supervisor_terminal_state.json').exists())

    def test_does_not_mask_other_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,expected=False)
            self.assertRaises(AssertionError,control.watch,root)
            self.assertEqual(json.loads((root/'pipeline_state.json').read_text())['phase'],'error')


if __name__=='__main__':unittest.main()
