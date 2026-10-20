"""No simulation: measured-time regression for the R5 early-latch defect."""
import unittest
import numpy as np
from fullcycle_contract import ObservedHold


class HoldChecks(unittest.TestCase):
    def test_r5_25_samples_are_only_24_intervals(self):
        dt=float(np.float32(.02));hold=ObservedHold()
        for step in range(3471,3496):
            hold.observe((step+1)*dt,True)
        self.assertAlmostEqual(hold.elapsed,.48,places=6)
        self.assertFalse(hold.reached(.5))
        # Querying at the next pre-physics boundary gives no unobserved credit.
        self.assertFalse(hold.reached(.5))
        for step in [3496,3497]:
            hold.observe((step+1)*dt,True)
        self.assertTrue(hold.reached(.5))

    def test_100_samples_do_not_prove_two_seconds(self):
        hold=ObservedHold()
        for step in range(100):
            hold.observe((step+1)*.02,True)
        self.assertFalse(hold.reached(2.))
        hold.observe(2.04,True)
        self.assertTrue(hold.reached(2.))

    def test_failure_resets_streak(self):
        hold=ObservedHold()
        hold.observe(10.,True);hold.observe(10.6,False);hold.observe(10.7,True)
        self.assertEqual(hold.elapsed,0.)
        self.assertFalse(hold.reached(.5))
        hold.observe(11.3,True)
        self.assertTrue(hold.reached(.5))

    def test_single_sample_and_bad_clocks(self):
        hold=ObservedHold();hold.observe(50.,True)
        self.assertFalse(hold.reached(.5))
        for time in [49.,50.,float("nan")]:
            with self.assertRaises(ValueError):hold.observe(time,True)


if __name__=="__main__":
    unittest.main()
