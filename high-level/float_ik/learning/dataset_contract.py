"""Validate the supported Cartesian recording before normalization or training."""
import numpy as np
from schema import EE_SCHEMA, contract


def validate_replay(z, schema):
    actions, states = contract(schema)
    if 'schema' not in z:
        raise ValueError('Cartesian data requires embedded schema; recollect expert demonstrations')
    if str(z['schema'].item()) != schema:
        raise ValueError('Recording schema mismatch')
    if z['action_names'].tolist() != actions:
        raise ValueError('Recording feature names mismatch')
    n = len(z['action'])
    if z['action'].shape != (n, len(actions)) or not np.isfinite(z['action']).all():
        raise ValueError('Recording action dimensions/values mismatch')
    if schema == EE_SCHEMA and not np.allclose(np.linalg.norm(z['action'][:, 6:10], axis=1), 1., atol=1e-5):
        raise ValueError('Recorded target quaternions must be unit xyzw')


def validate_episode(z, schema):
    validate_replay(z, schema)
    actions, states = contract(schema)
    n = len(z['action'])
    if 'schema' in z and z['state_names'].tolist() != states:
        raise ValueError('Recording state names mismatch')
    if z['state'].shape != (n, len(states)):
        raise ValueError('Recording dimensions mismatch')
    if z['images'].shape != (n, 4, 54, 96):
        raise ValueError('Recording image dimensions mismatch')
    if not np.isfinite(z['action']).all() or not np.isfinite(z['state']).all():
        raise ValueError('Nonfinite training data')
    if schema == EE_SCHEMA:
        if not np.allclose(z['state'][1:, -len(actions):], z['action'][:-1], atol=1e-6):
            raise ValueError('Previous action is not the executed Cartesian command')
