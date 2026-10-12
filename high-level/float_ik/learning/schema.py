"""Versioned contract shared by simulator, trainer, and checkpoint loader."""
EE_SCHEMA = 'float_push_ee_world_v1'
DEFAULT_SCHEMA = EE_SCHEMA
SCHEMAS = (EE_SCHEMA,)
BASE_ACTION_NAMES = ['base_vx_body_mps', 'base_vy_body_mps', 'base_wz_radps']


def contract(schema):
    if schema != EE_SCHEMA:
        raise ValueError('Unsupported schema: %s; only Cartesian EE actions are supported' % schema)
    arm = ['ee_target_%s_world_m' % x for x in 'xyz'] + ['ee_target_q%s_world' % x for x in 'xyzw']
    actions = BASE_ACTION_NAMES + arm + ['gripper_target_rad']
    states = ['base_x_initial_m', 'base_y_initial_m', 'base_yaw_sin', 'base_yaw_cos'] + BASE_ACTION_NAMES + ['q%d_rad' % i for i in range(7)] + ['dq%d_radps' % i for i in range(7)] + ['ee_x_base_m', 'ee_y_base_m', 'ee_z_base_m', 'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw'] + ['previous_' + x for x in actions]
    return actions, states


def model_config(schema):
    actions, states = contract(schema)
    return dict(state_dim=len(states), action_dim=len(actions), obs_horizon=4, pred_horizon=16)


def task_success(signed_angle_deg, unlocked, rear_clearance_m):
    """Correct direction and final rear clearance; lock is a physical task state."""
    return bool(unlocked and signed_angle_deg >= 80.0 and rear_clearance_m >= 0.52)
