"""Negative regressions use changed full-precision actions and real damaged video."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'dp'))
from door_dp_common import _to_list
from stop_audit import compare_actions, inspect_video


class ActionAuditTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.linspace(-.8, .9, 40, dtype=np.float32).reshape(4, 10)
        self.rows = [dict(applied_action=row) for row in _to_list(self.raw)]

    def test_original_serialization_and_exact_full_precision(self):
        result = compare_actions(self.raw, self.raw.copy(), self.rows)
        self.assertTrue(result['passed'])
        self.assertGreater(result['max_raw_log_difference'], 0.)

    def test_real_execution_log_difference_rejected(self):
        self.rows[1]['applied_action'][0] += .001
        self.assertFalse(compare_actions(self.raw, self.raw.copy(), self.rows)['passed'])

    def test_sub_rounding_raw_corruption_still_rejected(self):
        changed = self.raw.copy(); changed[1, 0] = np.nextafter(changed[1, 0], np.float32(1.))
        self.assertEqual(_to_list(changed), _to_list(self.raw))
        self.assertFalse(compare_actions(changed, self.raw, self.rows)['passed'])

    def test_both_full_precision_copies_corrupted_log_detects_it(self):
        changed = self.raw.copy(); changed[2, 3] += .001
        self.assertFalse(compare_actions(changed, changed.copy(), self.rows)['passed'])

    def test_nonfinite_and_truncated_actions_rejected(self):
        changed = self.raw.copy(); changed[0, 0] = np.nan
        self.assertFalse(compare_actions(changed, changed.copy(), self.rows)['passed'])
        self.assertFalse(compare_actions(self.raw[:-1], self.raw, self.rows)['passed'])


class VideoAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import imageio_ffmpeg
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.valid = cls.folder/'valid.mp4'
        cls.wrong_fps = cls.folder/'wrong_fps.mp4'
        cls.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        for path, fps in [(cls.valid, 25), (cls.wrong_fps, 24)]:
            subprocess.run([cls.ffmpeg, '-v', 'error', '-f', 'lavfi', '-i',
                'testsrc=size=64x48:rate=%d' % fps, '-frames:v', '8', '-c:v', 'libx264',
                '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_video_passes(self):
        self.assertTrue(inspect_video(self.valid, 8)['passed'])

    def test_real_truncated_video_fails(self):
        data = self.valid.read_bytes()
        damaged = self.folder/'truncated.mp4'; damaged.write_bytes(data[:len(data)//2])
        self.assertFalse(inspect_video(damaged, 8)['passed'])

    def test_wrong_frame_count_fails(self):
        result = inspect_video(self.valid, 9)
        self.assertTrue(result['decoded'])
        self.assertEqual(result['decoded_frames'], 8)
        self.assertFalse(result['passed'])

    def test_wrong_fps_and_missing_file_fail(self):
        self.assertFalse(inspect_video(self.wrong_fps, 8)['passed'])
        self.assertFalse(inspect_video(self.folder/'missing.mp4', 8)['passed'])


if __name__ == '__main__':
    unittest.main()
