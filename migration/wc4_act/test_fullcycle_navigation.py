"""CPU-only regression checks for forward/reverse Joint9 navigation."""
import copy
import math
import unittest

import numpy as np

from fullcycle_navigation import compact_nominal_plan, route_directions, waypoint_command


class NavigationChecks(unittest.TestCase):
    def test_reverse_translation_keeps_robot_heading(self):
        v, w, done = waypoint_command([0, 0], math.pi, [1, 0, math.pi], -1)
        self.assertLess(v, 0)
        self.assertEqual(w, 0)
        self.assertFalse(done)
        displacement = v * .02 * np.array([math.cos(math.pi), math.sin(math.pi)])
        self.assertGreater(displacement[0], 0)
        self.assertLess(abs(displacement[1]), 1e-12)

    def test_forward_only_default_still_turns_before_driving(self):
        v, w, done = waypoint_command([0, 0], math.pi, [1, 0, 0])
        self.assertEqual(v, 0)
        self.assertEqual(abs(w), .6)
        self.assertFalse(done)
        self.assertEqual(waypoint_command([0, 0], 0, [1, 0, 0]), (.25, 0., False))

    def test_position_arrival_still_requires_heading(self):
        self.assertEqual(waypoint_command([1, 0], 0, [1, 0, math.pi]), (0., -.6, False))
        self.assertEqual(waypoint_command([1, 0], -math.pi, [1, 0, math.pi]), (0., 0., True))

    def test_invalid_direction_and_nonfinite_state_rejected(self):
        for direction in [0, 2, float("nan")]:
            with self.assertRaises(ValueError):
                waypoint_command([0, 0], 0, [1, 0, 0], direction)
        with self.assertRaises(ValueError):
            waypoint_command([float("nan"), 0], 0, [1, 0, 0])
        with self.assertRaises(ValueError):
            route_directions({"route": [[0, 0, 0]], "route_directions": [1, -1]})

    def test_plan_change_preserves_close_pose_and_input(self):
        source = {
            "status": "candidate_found",
            "selected": {"radius": .65, "distance": .75, "z": .95, "q": [1, 2, 3, 4, 5, 6]},
            "nominal_robot": [4.1, .18, .5, math.pi],
            "route": [[.8, .18, math.pi], [1.817, -1.213, math.atan2(.65, -.75)]],
            "route_audit": [{"old": True}],
        }
        before = copy.deepcopy(source)
        revised = compact_nominal_plan(source)
        self.assertEqual(source, before)
        self.assertEqual(revised["selected"], before["selected"])
        self.assertEqual(revised["route"][-1], before["route"][-1])
        self.assertEqual(route_directions(revised), [1, 1, -1])
        self.assertNotIn("route_audit", revised)
        self.assertEqual(revised["navigation_validation_status"], "requires_independent_validation")


if __name__ == "__main__":
    unittest.main()
