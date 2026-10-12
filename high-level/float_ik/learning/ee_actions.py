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
