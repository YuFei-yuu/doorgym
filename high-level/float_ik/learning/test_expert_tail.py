"""Recording boundaries, incomplete horizons and compatibility without a simulator."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from expert_tail import ExpertRecordingTail


class ExpertTailTests(unittest.TestCase):
    def test_three_seconds_includes_real_observation_at_endpoint(self):
        tail = ExpertRecordingTail(3., .005 * 4)
        for step in range(300):
            tail.observe(step, step >= 149)
            self.assertEqual(tail.complete, step == 299)
        self.assertEqual(tail.first_pass_step, 149)
        self.assertEqual(tail.steps, 150)
        self.assertEqual(tail.recorded_seconds, 3.)
        tail.require_complete()

    def test_float32_dt_does_not_add_spurious_frame(self):
        tail = ExpertRecordingTail(3., .019999999552965164)
        self.assertEqual(tail.steps, 150)

    def test_max_steps_before_or_after_pass_is_not_completion(self):
        for first_pass in [None, 5]:
            tail = ExpertRecordingTail(3., .02)
            for step in range(100):
                tail.observe(step, first_pass is not None and step >= first_pass)
            self.assertFalse(tail.complete)
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                tail.require_complete()

    def test_legacy_ends_at_first_pass(self):
        tail = ExpertRecordingTail(0., .02)
        tail.observe(0, False)
        self.assertFalse(tail.complete)
        tail.observe(1, True)
        self.assertTrue(tail.complete)
        self.assertEqual(tail.recorded_seconds, 0.)

    def test_rejects_gaps_and_bad_duration(self):
        tail = ExpertRecordingTail(3., .02)
        tail.observe(0, False)
        with self.assertRaises(ValueError):
            tail.observe(2, True)
        for seconds, dt in [(2., .02), (3., 0.), (3., float('nan'))]:
            with self.assertRaises(ValueError):
                ExpertRecordingTail(seconds, dt)

    def test_policy_and_replay_cannot_enable_tail(self):
        path = Path(__file__).with_name('wholebody_float.py')
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'configure')
        method.decorator_list = []
        namespace = {'PROFILE': 'test'}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        for checkpoint, replay in [('model.pt', None), (None, 'trajectory.npz')]:
            args = SimpleNamespace(num_envs=1, float_output='new', record_dp_dataset=False,
                float_expert_post_pass_seconds=3., dp_policy_checkpoint=checkpoint, float_replay=replay,
                float_pass_mode='forward')
            with self.assertRaisesRegex(ValueError, 'restricted'):
                namespace['configure'](SimpleNamespace(), args)


if __name__ == '__main__':
    unittest.main()
