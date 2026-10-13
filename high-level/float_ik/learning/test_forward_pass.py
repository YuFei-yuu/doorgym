"""Check source navigation parity and the near/crossed-door regression."""
import ast
import copy
import math
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[3]
ENTRY = 'high-level/play_b1z1_push_with_door_asset_camera.py'
UPSTREAM = '0acc2f050479fb54a0670ccb90bc72b98b52a54a'


def evaluate(text, x, initial_x, mode):
    tree = ast.parse(text)
    node = copy.deepcopy(next(n for n in ast.walk(tree)
                              if isinstance(n, ast.FunctionDef) and n.name == 'update_pass_state'))
    for i, statement in enumerate(node.body):
        if isinstance(statement, ast.Nonlocal):
            node.body[i] = ast.copy_location(ast.Global(names=statement.names), statement)
    args = SimpleNamespace(float_pass_mode=mode, pass_open_angle_deg=80., pass_front_distance=.5,
                           robot_front_offset=.55, pass_through_distance=1.4, pass_left_offset=.1,
                           pass_center_y_offset=0., pass_backoff_distance=.5,
                           pass_align_pos_tol=.12, gripper_open=-.02)
    env = SimpleNamespace(root_states=torch.zeros(1, 13), door_root_state=torch.zeros(1, 13),
                          _door_dof_pos=torch.tensor([[-math.radians(85), 0.]]),
                          freeze_arm_default=torch.ones(1, dtype=torch.bool),
                          freeze_arm_zero=torch.zeros(1, dtype=torch.bool),
                          external_gripper_target=torch.zeros(1, 1), ee_pos=torch.zeros(1, 3),
                          ee_orn=torch.tensor([[0., 0., 0., 1.]]),
                          curr_ee_goal_cart_world=torch.zeros(1, 3),
                          ee_goal_orn_quat=torch.zeros(1, 4), ee_goal_orn_delta_rpy=torch.zeros(1, 3))
    env.root_states[0, 0] = x
    env.door_root_state[0, 0] = 2.5
    initial = torch.tensor([[initial_x, 0.]])
    ns = dict(torch=torch, math=math, args=args, env=env, backend=object(), start_xy=initial,
              DOOR_RUNTIME={'door_motion_sign': -1.}, phase_id=torch.zeros(1, dtype=torch.long),
              pass_side=torch.ones(1), snap_calls=[])
    for key in ['pass_started', 'pass_backoff_done', 'pass_arm_zero_snapped', 'pass_align_done', 'pass_done']:
        ns[key] = torch.zeros(1, dtype=torch.bool)
    for key in ['pass_backoff_start_xy', 'pass_pre_xy', 'pass_through_xy']:
        ns[key] = env.root_states[:, :2].clone()
    ns['door_center_xy'] = lambda mask: env.door_root_state[mask, :2].clone()
    ns['snap_arm_to_zero'] = lambda mask: ns['snap_calls'].append(mask.tolist())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), ENTRY, 'exec'), ns)
    ns['update_pass_state']()
    return ns


class ForwardPassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = (ROOT / ENTRY).read_text()
        cls.source = subprocess.check_output(['git', 'show', UPSTREAM + ':' + ENTRY], cwd=ROOT, text=True)

    def test_backend_specific_defaults(self):
        node = next(n for n in ast.parse(self.current).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'resolve_float_pass_mode')
        ns = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), ENTRY, 'exec'), ns)
        resolve = ns['resolve_float_pass_mode']
        self.assertEqual(resolve(None, True), 'forward')
        self.assertEqual(resolve(None, False), 'source')
        self.assertEqual(resolve('source', True), 'source')
        with self.assertRaises(ValueError):
            resolve('forward', False)

    def test_default_source_behavior_unchanged(self):
        for x, initial in [(2.504, 4.1), (2.4, 4.1), (2.6, 1.), (2.5, 4.1)]:
            a, b = evaluate(self.source, x, initial, 'source'), evaluate(self.current, x, initial, 'source')
            for key in ['pass_side', 'pass_pre_xy', 'pass_through_xy', 'pass_align_done', 'phase_id']:
                self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertEqual(a['snap_calls'], b['snap_calls'])

    def test_forward_near_and_crossed_plane_keep_far_side_goal(self):
        for x in [2.504, 2.5, 2.4]:
            result = evaluate(self.current, x, 4.1, 'forward')
            self.assertAlmostEqual(result['pass_through_xy'][0, 0].item(), 1.1, places=5)
            self.assertTrue(result['pass_align_done'].item())
            self.assertEqual(result['phase_id'].item(), 11)
            self.assertEqual(len(result['snap_calls']), 1)

    def test_opposite_approach_uses_initial_side(self):
        result = evaluate(self.current, 2.6, 1., 'forward')
        self.assertAlmostEqual(result['pass_through_xy'][0, 0].item(), 3.9, places=5)
        self.assertEqual(result['phase_id'].item(), 11)


if __name__ == '__main__':
    unittest.main()
