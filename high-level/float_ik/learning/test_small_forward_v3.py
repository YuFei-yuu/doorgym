"""The pilot must stay within the explicitly authorized optimization schedule."""
from pathlib import Path
import unittest
from small_forward_v3 import train_command, CHECKPOINTS


class SmallPilotTests(unittest.TestCase):
    def test_only_three_targets_and_random_first_stage(self):
        self.assertEqual(CHECKPOINTS, [250, 500, 1000])
        command = train_command(Path('/tmp/pilot'), 250, None)
        self.assertNotIn('--resume', command)
        for key, value in [('--action_contract', 'forward_v3'), ('--batch_size', '64'),
                           ('--num_workers', '0'), ('--save_interval', '250')]:
            self.assertEqual(command[command.index(key)+1], value)

    def test_resume_is_only_from_this_runs_preceding_checkpoint(self):
        command = train_command(Path('/tmp/pilot'), 1000, 500)
        self.assertEqual(command[command.index('--resume')+1], '/tmp/pilot/training/checkpoints/step_000500.pt')
        with self.assertRaises(ValueError): train_command(Path('/tmp/pilot'), 1000, 250)

    def test_no_automatic_extra_target(self):
        for target in [0, 750, 1001, 2000]:
            with self.assertRaises(ValueError): train_command(Path('/tmp/pilot'), target, None)


if __name__ == '__main__': unittest.main()
