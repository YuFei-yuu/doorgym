"""Pure action validation shared by the Cartesian recorder and executor."""
import numpy as np


def normalize_ee_action(action, gripper_lower, gripper_upper, previous_quat=None):
    raw = np.asarray(action, dtype=np.float32)
    if raw.shape != (11,) or not np.isfinite(raw).all():
        raise ValueError('Expected finite 11D world-pose action')
    out = raw.copy()
    out[:3] = np.clip(out[:3], [-.6, -.3, -.4], [.6, .3, .4])
    quat = out[6:10].astype(np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        raise ValueError('Invalid near-zero target quaternion')
    # Float32 unit quaternions have rounding error. Do not repeatedly perturb an
    # already-normalized recorded command: replay must preserve its exact bytes.
    if abs(norm - 1.) > 1e-6:
        quat /= norm
    # q and -q are the same rotation. Keep action history/training labels continuous.
    if previous_quat is not None and np.dot(quat, previous_quat) < 0:
        quat = -quat
    out[6:10] = quat
    out[10] = np.clip(out[10], gripper_lower, gripper_upper)
    return out


def expert_pose_reference(action, previous, measured_position, max_step=.006,
                          max_tracking_distance=.06, max_angle_step=.035):
    """Bound a scripted teacher reference before IK; never invoked by policy step.

    The recorded label is this pre-IK reference, not FK of solved joint targets.
    Position moves at most 6 mm/control tick and stays near the measured arm.
    """
    out = np.asarray(action, dtype=np.float32).copy()
    desired = out[3:6].astype(float)
    offset = desired - measured_position
    desired = np.asarray(measured_position) + offset * min(1., max_tracking_distance / max(np.linalg.norm(offset), 1e-12))
    delta = desired - previous[3:6]
    out[3:6] = previous[3:6] + delta * min(1., max_step / max(np.linalg.norm(delta), 1e-12))
    q0 = np.asarray(previous[6:10], dtype=float); q0 /= np.linalg.norm(q0)
    q1 = out[6:10].astype(float); q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0: q1 = -q1; dot = -dot
    half_angle = np.arccos(np.clip(dot, 0., 1.))
    if 2 * half_angle > max_angle_step:
        t = max_angle_step / (2 * half_angle)
        q1 = (np.sin((1-t)*half_angle)*q0 + np.sin(t*half_angle)*q1) / np.sin(half_angle)
    out[6:10] = q1
    return out
