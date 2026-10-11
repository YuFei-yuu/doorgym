"""Versioned contract shared by simulator, trainer, and checkpoint loader."""
SCHEMA = 'float_push_joint_v1'
ACTION_NAMES = ['base_vx_body_mps', 'base_vy_body_mps', 'base_wz_radps'] + ['joint%d_target_rad' % i for i in range(1, 7)] + ['gripper_target_rad']
STATE_NAMES = ['base_x_initial_m', 'base_y_initial_m', 'base_yaw_sin', 'base_yaw_cos'] + ACTION_NAMES[:3] + ['q%d_rad' % i for i in range(7)] + ['dq%d_radps' % i for i in range(7)] + ['ee_x_base_m', 'ee_y_base_m', 'ee_z_base_m', 'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw'] + ['previous_' + x for x in ACTION_NAMES]



def task_success(signed_angle_deg, unlocked, rear_clearance_m):
    """Correct direction and final rear clearance; lock is a physical task state."""
    return bool(unlocked and signed_angle_deg >= 80.0 and rear_clearance_m >= 0.52)
