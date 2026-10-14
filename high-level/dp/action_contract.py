"""Versioned action transforms shared by training and policy inference."""
import numpy as np
import torch

LEGACY = 'door_action_zscore_v1'
FORWARD_V3 = 'door_action_minmax_continuous_quat_v3'


def checkpoint_version(config):
    version = config.get('action_normalization_version', LEGACY)
    if version not in (LEGACY, FORWARD_V3):
        raise ValueError('Unsupported checkpoint action contract: '+str(version))
    if version == FORWARD_V3:
        if config.get('observation_padding') != 'repeat_first' or config.get('quaternion_rule') != 'largest_component_positive_then_continuous':
            raise ValueError('Incomplete forward_v3 sequence/quaternion contract')
    return version


def quaternion_sign(quaternion, previous=None):
    q = np.asarray(quaternion)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('Expected a finite nonzero xyzw quaternion')
    if previous is None:
        return -1 if q[np.argmax(np.abs(q))] < 0 else 1
    return -1 if np.dot(q, previous) < 0 else 1


def canonicalize_sequence(values, quaternion_slice):
    result = np.array(values, copy=True)
    previous = None
    for row in result:
        row[quaternion_slice] *= quaternion_sign(row[quaternion_slice], previous)
        previous = row[quaternion_slice].copy()
    return result


def observation_indices(center, horizon, episode_start=0):
    return np.maximum(np.arange(center-horizon+1, center+1), episode_start)


def _like(value, reference):
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.asarray(reference).dtype)


def normalize_action(action, stats, version=LEGACY):
    if version == LEGACY:
        return (action-_like(stats['action_mean'], action))/_like(stats['action_std'], action)
    if version != FORWARD_V3:
        raise ValueError('Unsupported action contract: '+str(version))
    lo, hi = _like(stats['action_min'], action), _like(stats['action_max'], action)
    span = hi-lo
    if isinstance(action, torch.Tensor):
        return torch.where(span == 0, torch.zeros_like(action), 2*(action-lo)/torch.where(span == 0, torch.ones_like(span), span)-1)
    return np.where(span == 0, 0., 2*(action-lo)/np.where(span == 0, 1., span)-1).astype(action.dtype)


def denormalize_action(action, stats, version=LEGACY):
    if version == LEGACY:
        return action*_like(stats['action_std'], action)+_like(stats['action_mean'], action)
    if version != FORWARD_V3:
        raise ValueError('Unsupported action contract: '+str(version))
    lo, hi = _like(stats['action_min'], action), _like(stats['action_max'], action)
    return (action+1)*.5*(hi-lo)+lo


def validate_stats(stats, version):
    names = ['state_mean', 'state_std', 'action_mean', 'action_std'] if version == LEGACY else ['state_mean', 'state_std', 'action_min', 'action_max']
    for name in names:
        value = torch.as_tensor(stats[name])
        if value.ndim != 1 or not torch.isfinite(value).all():
            raise ValueError('Invalid checkpoint statistic: '+name)
    if not torch.all(torch.as_tensor(stats['state_std']) > 0):
        raise ValueError('State standard deviation must be positive')
    if version == LEGACY and not torch.all(torch.as_tensor(stats['action_std']) > 0):
        raise ValueError('Action standard deviation must be positive')
    if version == FORWARD_V3:
        lo, hi = torch.as_tensor(stats['action_min']), torch.as_tensor(stats['action_max'])
        if lo.shape != (10,) or hi.shape != (10,) or torch.any(hi < lo):
            raise ValueError('Invalid 10D min/max action bounds')
