import unittest
from stop_data_checks import stop_acceptance


def recording(displacements):
    rows, x = [], 1.1
    for i, dx in enumerate(displacements):
        root = [x, -.1, .5, 0., 0., 0., 1.] + [0.]*6
        x += dx
        after = [x, -.1, .5, 0., 0., 0., 1.] + [0.]*6
        rows.append(dict(step=i, completed_physical_steps=(i+1)*4, root_before=root,
            base_after=after, door_after=[-1.5], phase_id=12,
            extra=dict(pass_done=True, through_target=[1.1, -.1], image_valid=True,
                       command_source='script', dp_action_applied=False)))
    return rows


class StopDataChecksTests(unittest.TestCase):
    def test_real_hold_passes_and_short_recording_fails(self):
        self.assertTrue(stop_acceptance(recording([0.]*151), .02)['passed'])
        self.assertFalse(stop_acceptance(recording([0.]*150), .02)['passed'])

    def test_zero_commands_cannot_hide_motion_or_restart(self):
        rows = recording([0.]*140+[.005]+[0.]*10)
        for row in rows:
            row['robot_command'] = dict(vx=0., yaw=0.)
        result = stop_acceptance(rows, .02)
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['settles_within_half_second'])

    def test_uses_maximum_drift_not_final_displacement(self):
        rows = recording([.00019]*110+[-.00019]*41)
        result = stop_acceptance(rows, .02)
        self.assertTrue(result['checks']['maximum_two_second_drift'])
        rows = recording([.001]*25+[0.]*126)
        result = stop_acceptance(rows, .02)
        self.assertEqual(result['stop_step'], 25)
        self.assertAlmostEqual(result['extra_path_until_stopped_m'], .025)

    def test_original_pass_and_camera_integrity_required(self):
        rows = recording([0.]*151)
        for row in rows:
            row['extra']['through_target'] = [2.5, -.1]
        self.assertFalse(stop_acceptance(rows, .02)['passed'])
        rows = recording([0.]*151)
        rows[5]['extra']['image_valid'] = False
        self.assertFalse(stop_acceptance(rows, .02)['passed'])


if __name__ == '__main__':
    unittest.main()
