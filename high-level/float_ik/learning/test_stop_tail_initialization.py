"""Directory evidence, failed initialization and refusal to overwrite old runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collect_stop_tail as pipeline


class InitializationTests(unittest.TestCase):
    def test_directory_files_and_empty_directories_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'user_stop_3000'
            (source/'empty').mkdir(parents=True)
            (source/'completion.json').write_text('{"stopped": true}')
            hashes, directories = {}, {}
            pipeline.register_input(source, hashes, directories)
            plan = dict(input_hashes=hashes, input_directories=directories)
            self.assertTrue(pipeline.verify_inputs(plan)['passed'])
            (source/'completion.json').write_text('changed')
            self.assertFalse(pipeline.verify_inputs(plan)['passed'])

    def test_directory_additions_and_removals_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            hashes, directories = {}, {}
            pipeline.register_input(source, hashes, directories)
            plan = dict(input_hashes=hashes, input_directories=directories)
            (source/'new').write_text('new')
            self.assertFalse(pipeline.verify_inputs(plan)['passed'])
            (source/'new').unlink()
            self.assertTrue(pipeline.verify_inputs(plan)['passed'])
            source.rmdir()
            self.assertFalse(pipeline.verify_inputs(plan)['passed'])

    def test_initialization_failure_records_status_without_launching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'new_run'
            with patch.object(pipeline, 'initialize', side_effect=IsADirectoryError('injected directory failure')), \
                 patch.object(pipeline, 'launch') as launch:
                with self.assertRaises(IsADirectoryError):
                    pipeline.execute(root)
                launch.assert_not_called()
            self.assertEqual(json.loads((root/'state.json').read_text())['phase'], 'failed_stopped')
            self.assertFalse(json.loads((root/'acceptance.json').read_text())['passed'])
            self.assertFalse(json.loads((root/'failure.json').read_text())['initialization_complete'])

    def test_existing_failure_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'failure.json').write_text('original failure')
            with self.assertRaises(FileExistsError):
                pipeline.execute(root)
            self.assertEqual((root/'failure.json').read_text(), 'original failure')


if __name__ == '__main__':
    unittest.main()
