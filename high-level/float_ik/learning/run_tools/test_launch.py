"""GPU selection must avoid another user's process and preserve inherited nice10."""
import tempfile
import unittest
from unittest.mock import patch

from launch import launch


class LaunchTests(unittest.TestCase):
    def test_occupied_gpu_and_nice_inheritance(self):
        def output(command, **kwargs):
            if '--query-compute-apps=gpu_uuid' in command:
                return 'GPU-test-occupied\n'
            return ('0, GPU-test-occupied, 72000, 0, 0\n'
                    '1, GPU-test-idle, 72000, 0, 0\n')
        with tempfile.TemporaryDirectory() as root, \
                patch('launch.subprocess.check_output', side_effect=output), \
                patch('launch.time.sleep'), patch('launch.os.getpriority',return_value=10), \
                patch('launch.run',return_value=0) as child:
            self.assertEqual(launch(root,'fake',['python','-c','pass']),0)
            self.assertEqual(child.call_args.kwargs['gpu'],1)
            self.assertIsNone(child.call_args.kwargs['total'])
            self.assertIsNone(child.call_args.kwargs['timeout'])
            self.assertIn('exec nice -n 0 ',child.call_args.args[2][2])


if __name__=='__main__': unittest.main()
