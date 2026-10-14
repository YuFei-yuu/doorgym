"""Explicit action contracts and metrics used only by no-training diagnostics."""
import numpy as np


LEGACY = 'door_action_zscore_v1'
MINMAX = 'door_action_minmax_v2'


def action_contract(actions):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 10 or not np.isfinite(actions).all():
        raise ValueError('Expected finite [frames, 10] training actions')
    return dict(version=MINMAX, source_split='train', minimum=actions.min(0).tolist(),
                maximum=actions.max(0).tolist(), constant=(np.ptp(actions, axis=0) == 0).tolist(),
                state_normalization='zscore', clip_sample=True, clip_sample_range=1.0,
                checkpoint_compatible_with_legacy=False)


def normalize_action(actions, contract):
    x = np.asarray(actions, dtype=np.float64)
    if contract['version'] == LEGACY:
        return (x - np.asarray(contract['mean'])) / np.asarray(contract['std'])
    if contract['version'] != MINMAX:
        raise ValueError('Unknown action normalization version')
    lo, hi = np.asarray(contract['minimum']), np.asarray(contract['maximum'])
    width = hi-lo
    return np.where(width == 0, 0., 2.*(x-lo)/np.where(width == 0, 1., width)-1.)


def denormalize_action(actions, contract):
    x = np.asarray(actions, dtype=np.float64)
    if contract['version'] == LEGACY:
        return x*np.asarray(contract['std'])+np.asarray(contract['mean'])
    if contract['version'] != MINMAX:
        raise ValueError('Unknown action normalization version')
    lo, hi = np.asarray(contract['minimum']), np.asarray(contract['maximum'])
    return (x+1.)*.5*(hi-lo)+lo


def legacy_contract(stats):
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    return dict(version=LEGACY, mean=array(stats['action_mean']).tolist(),
                std=array(stats['action_std']).tolist())


def validate_checkpoint_contract(config, expected=LEGACY):
    actual = config.get('action_normalization_version', LEGACY)
    if actual != expected:
        raise ValueError('Checkpoint action contract %s cannot be interpreted as %s' % (actual, expected))
    return actual


def clipped_legacy_action(action, contract):
    return denormalize_action(np.clip(normalize_action(action, contract), -1., 1.), contract).astype(np.float32)


def quaternion_angle(prediction, target):
    p, q = np.asarray(prediction, float), np.asarray(target, float)
    pn, qn = np.linalg.norm(p, axis=-1), np.linalg.norm(q, axis=-1)
    dot = np.sum(p*q, axis=-1)/np.maximum(pn*qn, 1e-15)
    result = 2*np.arccos(np.clip(np.abs(dot), 0., 1.))
    return np.where((pn > 1e-8) & (qn > 1e-8), result, np.nan)


def select_windows(phases, pred_horizon=32):
    """Predeclared windows, with full future targets and no test-set access."""
    phases = np.asarray(phases).reshape(-1)
    last = len(phases)-pred_horizon
    result = [('start', 0)]
    for name, phase, end in [('approach_end', 0, True), ('grasp', 3, False),
                             ('push', 7, False), ('through', 11, False)]:
        ids = np.flatnonzero((phases == phase) & (np.arange(len(phases)) <= last))
        if not len(ids):
            raise ValueError('Missing complete diagnostic phase: '+name)
        center = int(ids[-1] if end else ids[len(ids)//2])
        result.append((name, center))
    return result
