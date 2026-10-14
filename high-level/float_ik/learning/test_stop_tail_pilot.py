"""One-variable replay options and unchanged physical stopping thresholds."""
import ast
from pathlib import Path
import unittest

from stop_data_checks import stop_acceptance
from test_stop_data_checks import recording


class PilotContractTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name('diagnostic_replay.py')
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'replay_configuration')
        ns = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        self.variant = ns['replay_configuration']

    def test_paired_replay_changes_only_freeze(self):
        a, b = self.variant('recorded', 'action'), self.variant('actions', 'action')
        self.assertTrue(a['restore_freeze']); self.assertFalse(b['restore_freeze'])
        self.assertEqual(a['restore_quaternion'], b['restore_quaternion'])
        self.assertFalse(a['restore_quaternion'])

    def test_old_diagnostic_defaults_preserved(self):
        self.assertEqual(self.variant('recorded'), dict(restore_freeze=True, restore_quaternion=True))
        self.assertEqual(self.variant('actions'), dict(restore_freeze=False, restore_quaternion=False))

    def test_replay_and_policy_sources_are_explicit(self):
        rows = recording([0.]*151)
        for row in rows:
            row['extra']['command_source'] = 'replay'
        self.assertTrue(stop_acceptance(rows, .02, expected_source='replay')['passed'])
        self.assertFalse(stop_acceptance(rows, .02)['passed'])

    def test_policy_hold_requires_two_real_seconds_but_not_expert_three_second_tail(self):
        rows = recording([0.]*100)
        for row in rows:
            row['extra'].update(command_source='dp', dp_action_applied=True)
        self.assertTrue(stop_acceptance(rows, .02, expected_source='dp', required_post_pass_seconds=0.)['passed'])
        self.assertFalse(stop_acceptance(rows[:-1], .02, expected_source='dp', required_post_pass_seconds=0.)['passed'])


if __name__ == '__main__':
    unittest.main()
