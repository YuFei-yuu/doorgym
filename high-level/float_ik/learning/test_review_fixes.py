"""Regression tests for reviewed orchestration, failure reporting and delivery."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from diagnostic_integrity import (phase_lock, process_identity, materialize_external_links,
                                  source_paths, verify_frozen, sha256)
from diagnostic_report import replay_details, usable_rows
from finalize_diagnostics import inspect_video
from forward_results import episode_result
import forward_diagnostics as diagnostics


class IntegrityTests(unittest.TestCase):
    def test_live_child_cannot_be_finalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'active_command.json').write_text(json.dumps(dict(pid=os.getpid(), proc_start_ticks=process_identity(os.getpid()))))
            with self.assertRaisesRegex(RuntimeError, 'still active'):
                diagnostics.replay(root)
            self.assertFalse((root/'replay/results.json').exists())

    def test_run_lock_excludes_second_supervisor(self):
        with tempfile.TemporaryDirectory() as directory:
            with phase_lock(directory, 'first'):
                with self.assertRaisesRegex(RuntimeError, 'owns this run'):
                    with phase_lock(directory, 'second'):
                        self.fail('Concurrent phase acquired the lock')

    def test_dead_child_restores_real_ledger_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); original = root/'original'
            expert = original/'dataset/val_seed_16100'; expert.mkdir(parents=True)
            (expert/'config.json').write_text(json.dumps(dict(args=dict(robot_x=3.9, robot_y=-.06))))
            (expert/'summary.json').write_text(json.dumps(dict(frames=100)))
            (root/'resources').mkdir()
            (root/'resources/replay__actions_16100.json').write_text('{}')
            (root/'budget.jsonl').write_text(json.dumps(dict(stage='replay__actions_16100', exit_code=124))+'\n')
            with patch.object(diagnostics, 'verify_sources', return_value={'original': str(original)}), \
                 patch.object(diagnostics, 'MODES', ['actions']), patch.object(diagnostics, 'REPLAY_SEEDS', [16100]):
                diagnostics.replay(root)
            result = json.loads((root/'replay/results.json').read_text())[0]
            self.assertEqual(result['exit_code'], 124)
            self.assertFalse(result['success'])

    def test_source_helpers_frozen_and_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)/'repo'; output = Path(directory)/'run'; output.mkdir()
            helper = repo/'high-level/float_ik/learning/forward_results.py'
            helper.parent.mkdir(parents=True); helper.write_text('original')
            artifact = helper.with_name('artifacts.py'); artifact.write_text('serialization')
            self.assertIn(helper, source_paths(repo)); self.assertIn(artifact, source_paths(repo))
            (output/'plan.json').write_text(json.dumps(dict(source_hashes={str(helper.relative_to(repo)): sha256(helper)})))
            verify_frozen(output, repo)
            helper.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'dependency changed'):
                verify_frozen(output, repo)

    def test_external_log_is_portable_after_move(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); root = base/'delivery'; root.mkdir()
            target = base/'external.log'; target.write_bytes(b'private run log\n')
            link = root/'debug-core.log'; link.symlink_to(target)
            expected = sha256(target)
            provenance = materialize_external_links(root)
            self.assertEqual(provenance[0]['sha256'], expected)
            self.assertFalse(link.is_symlink())
            target.unlink(); root.rename(base/'moved')
            self.assertEqual(sha256(base/'moved/debug-core.log'), expected)


class FailureReportTests(unittest.TestCase):
    def valid_case(self, root, seed=16108):
        folder = root/'valid'; folder.mkdir()
        reference = root/'original/dataset'/('val_seed_%d' % seed); reference.mkdir(parents=True)
        row = dict(root_before=[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            base_after=[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            q_before=[0.]*7, q_after=[0.]*7, ee=dict(actual_pos_world=[0.]*3, pos_error=[0.]*3),
            door_after=[0., 0.], robot_command=dict(vx=0., yaw=0.), servo_target=[0.]*7,
            freeze_arm_default=False, freeze_arm_zero=False, pre_observation_event='none')
        for where in [folder, reference]:
            (where/'execution.jsonl').write_text(json.dumps(row)+'\n')
            payload = dict(state=np.zeros((1, 27), np.float32))
            for key in diagnostics.IMAGE_KEYS:
                payload[key] = np.zeros((1, 54, 96, 3), np.uint8)
            np.savez(where/'trajectory.npz', **payload)
        (folder/'config.json').write_text(json.dumps(dict(joint_lower=[-1.]*7, joint_upper=[1.]*7)))
        return folder

    def test_missing_first_log_does_not_break_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); valid = self.valid_case(root)
            (root/'replay').mkdir()
            (root/'plan.json').write_text(json.dumps(dict(original=str(root/'original'))))
            values = [dict(seed=16100, replay_mode='actions', folder=str(root/'missing')),
                      dict(seed=16108, replay_mode='recorded', folder=str(valid))]
            (root/'replay/results.json').write_text(json.dumps(values))
            _, comparisons = replay_details(root)
            self.assertEqual(len(comparisons), 2)
            self.assertFalse(comparisons[0]['available']); self.assertTrue(comparisons[1]['available'])
            self.assertTrue((root/'replay/comparisons.csv').exists())

    def test_empty_and_truncated_logs_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder = self.valid_case(root)
            path = folder/'execution.jsonl'
            with path.open('a') as stream: stream.write('{"unfinished":')
            log, error = usable_rows(path)
            self.assertEqual(len(log), 1); self.assertEqual(error, 'truncated_execution_log')
            path.write_text('')
            self.assertEqual(usable_rows(path), ([], 'empty_execution_log'))

    def test_missing_video_is_reported_without_aborting_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            result = inspect_video(Path(directory)/'missing.mp4', {'frames': 100}, 'unused', None)
            self.assertFalse(result['decoded'])
            self.assertIn('missing_video', result['error'])

    def test_incomplete_execution_stays_in_result_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'execution.jsonl').write_text('{"unfinished":')
            value = episode_result(root, dict(seed=1, robot_x=4.1, robot_y=0.), 124)
            self.assertFalse(value['success']); self.assertFalse(value['complete_log'])
            self.assertEqual(value['log_error'], 'truncated_or_incomplete_execution_log')


if __name__ == '__main__':
    unittest.main()
