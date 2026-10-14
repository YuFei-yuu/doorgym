import unittest
import numpy as np
from diagnostic_contract import (action_contract, normalize_action, denormalize_action,
    clipped_legacy_action, validate_checkpoint_contract, quaternion_angle, select_windows, LEGACY, MINMAX)


class DiagnosticContractTests(unittest.TestCase):
    def test_minmax_coverage_roundtrip_and_constant(self):
        x = np.arange(60, dtype=float).reshape(6, 10)/7
        x[:, 1] = .125
        c = action_contract(x)
        y = normalize_action(x, c)
        self.assertLessEqual(abs(y).max(), 1.)
        np.testing.assert_array_equal(y[:, 1], 0.)
        np.testing.assert_allclose(denormalize_action(y, c), x, atol=1e-12)
        self.assertEqual(c['source_split'], 'train')

    def test_legacy_clipping_and_checkpoint_guard(self):
        c = dict(version=LEGACY, mean=[1.]*10, std=[2.]*10)
        np.testing.assert_array_equal(clipped_legacy_action([10.]*10, c), [3.]*10)
        self.assertEqual(validate_checkpoint_contract({}), LEGACY)
        with self.assertRaises(ValueError):
            validate_checkpoint_contract({}, MINMAX)
        with self.assertRaises(ValueError):
            validate_checkpoint_contract({'action_normalization_version': MINMAX})

    def test_quaternion_sign_and_invalid(self):
        self.assertAlmostEqual(float(quaternion_angle([0, 0, 0, 1], [0, 0, 0, -1])), 0.)
        self.assertTrue(np.isnan(quaternion_angle([0, 0, 0, 0], [0, 0, 0, 1])))

    def test_windows_have_full_future(self):
        phases = np.repeat([0, 3, 7, 11], 50)
        windows = select_windows(phases)
        self.assertEqual([k for k, _ in windows], ['start', 'approach_end', 'grasp', 'push', 'through'])
        self.assertTrue(all(i+32 <= len(phases) for _, i in windows))
        with self.assertRaises(ValueError):
            select_windows(np.zeros(100))


if __name__ == '__main__':
    unittest.main()
