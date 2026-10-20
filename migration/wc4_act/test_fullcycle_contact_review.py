"""CPU regression checks: observe contact without inventing failure criteria."""
import ast
from pathlib import Path
import unittest

from fullcycle_contract import door_contact_review_flags


class ContactReviewChecks(unittest.TestCase):
    def test_r4_wrist_contact_does_not_invalidate_opening(self):
        for value in [4.955710411071777, .15870770812034607]:
            flags = door_contact_review_flags("old_expert", value, True, False)
            self.assertTrue(flags["nonzero_door_contact"])
            self.assertFalse(flags["nonzero_end_panel_contact"])
            self.assertFalse(flags["off_target_closing_contact"])

    def test_same_wrist_record_in_close_requires_review_not_end_credit(self):
        flags = door_contact_review_flags("close", 4.955710411071777, True, False)
        self.assertTrue(flags["off_target_closing_contact"])
        self.assertFalse(flags["nonzero_end_panel_contact"])

    def test_zero_record_is_not_loaded_hold_contact_or_pushing_evidence(self):
        for phase in ["close", "release", "final_hold"]:
            flags = door_contact_review_flags(phase, 0., True, True)
            self.assertFalse(any(flags.values()))

    def test_actual_end_panel_nonzero_record_is_retained(self):
        flags = door_contact_review_flags("close", 1., True, True)
        self.assertTrue(flags["nonzero_end_panel_contact"])
        self.assertFalse(flags["off_target_closing_contact"])

    def test_nonfinite_record_remains_invalid(self):
        with self.assertRaises(ValueError):
            door_contact_review_flags("old_expert", float("nan"), True, False)

    def test_runtime_retains_task_guards_but_no_contact_heuristic_abort(self):
        tree = ast.parse(Path(__file__).with_name("fullcycle_expert.py").read_text())
        reasons = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                if isinstance(node.exc.func, ast.Name) and node.exc.func.id == "StopAttempt":
                    reasons.update(n.value for n in ast.walk(node.exc) if isinstance(n, ast.Constant) and isinstance(n.value, str))
        self.assertFalse(any("unexpected_robot_environment_contact" in x for x in reasons))
        self.assertFalse(any("unexpected_loaded_self_contact_with_blocked_arm" in x for x in reasons))
        for reason in ["nonfinite_measured_state", "return_crossing_after_whole_body_pass",
                       "close_sequence_before_full_body_pass", "forbidden_unlocked_hinge_write",
                       "nonzero_auxiliary_hinge_effort", "missing_end_effector_physical_closing_evidence"]:
            self.assertIn(reason, reasons)


if __name__ == "__main__":
    unittest.main()
