"""Original actuator calls only; host retains physics, cameras, timing and logging.

Call after CompleteExpert.tick(), before the existing physical step. A latch
request must be authorized by the unchanged TaskMonitor and logged by the host.
No leg/wheel resets, auxiliary hinge effort, or observation schema changes.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from author_rewrite.geometry import angle_delta


def apply_target(m, gym, sim, st, target, dof_names, previous_base, step, dt):
    if target.failed:
        raise RuntimeError(target.failed)
    B=target.base
    m.set_robot_base_pose(gym,st.env,st.actor_handles,B[:2],st.args.robot_z,B[2],st.args.robot_pitch)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    if target.joints is not None:
        st.dof_positions[:]=target.joints
        m.reset_tracik_controller_to_targets(gym,st,step,dt)
    else:
        quat=None if target.rotation is None else Rotation.from_matrix(target.rotation).as_quat()
        m.set_ik_target(st.ik_state,target.position,quat)
        if st.args.arm_ik_solver=='tracik':
            m.update_tracik_arm_targets_for_env(gym,st,step,dt)
        else:
            m.update_arm_ik_targets_for_env(gym,st.env,st.arm_actor,st.index,st.dof_positions,st.ik_state,st.args,len(dof_names))
    gi=dof_names.index('jointGripper')
    if target.gripper is not None:
        st.dof_positions[gi]=np.clip(target.gripper,st.ik_state.lower[gi].item(),st.ik_state.upper[gi].item())
    gym.set_actor_dof_position_targets(st.env,st.arm_actor,st.dof_positions)
    dx=B[:2]-np.asarray(previous_base)[:2]
    yaw=float(previous_base[2])
    v=float(dx@np.array([math.cos(yaw),math.sin(yaw)]))/dt
    w=angle_delta(yaw,B[2])/dt
    ids=[dof_names.index('joint%d'%i) for i in range(1,7)]+[gi]
    # Same [forward velocity, yaw velocity, arm6, gripper] contract.
    return np.asarray(np.r_[v,w,st.dof_positions[ids]],dtype=np.float32)


def authorize_latch(target, measured, task_monitor):
    """Return a measured-angle request, not an angle write or a door command."""
    if target.latch_angle is None:return None
    if not task_monitor.may_latch():raise RuntimeError('R11 task monitor did not authorize latch')
    if target.latch_angle!=measured.door_angle:raise RuntimeError('Latch must use preceding measured sample')
    return float(measured.door_angle)
