"""Validate embedded recording metadata, without legacy execution schemas."""
import numpy as np


def layout(action_names):
    names=list(action_names)
    for xyz,quat in [(['ee_x','ee_y','ee_z'],['ee_qx','ee_qy','ee_qz','ee_qw']),
                     (['ee_target_'+x+'_world_m' for x in 'xyz'],['ee_target_q'+x+'_world' for x in 'xyzw'])]:
        if all(x in names for x in xyz+quat):
            return dict(position=[names.index(x) for x in xyz],quaternion=[names.index(x) for x in quat])
    raise ValueError('Recording lacks named Cartesian target coordinates')


def feature_names(z):
    key='state_names' if 'state_names' in z else 'state_feature_names'
    return [str(s) for s in z[key].tolist()],[str(s) for s in z['action_names'].tolist()]


def validate_episode(z):
    states,actions=feature_names(z);n=len(z['action']);indices=layout(actions)
    if len(set(states))!=len(states) or len(set(actions))!=len(actions):raise ValueError('Duplicate feature names')
    if n==0 or z['action'].shape!=(n,len(actions)) or z['state'].shape!=(n,len(states)):raise ValueError('Recording dimensions mismatch')
    if not np.isfinite(z['state']).all() or not np.isfinite(z['action']).all():raise ValueError('Nonfinite recording')
    if not np.allclose(np.linalg.norm(z['action'][:,indices['quaternion']],axis=1),1.,atol=1e-5):raise ValueError('Target quaternion must be unit xyzw')
    if 'next_state' in z:
        if z['next_state'].shape!=z['state'].shape or not np.isfinite(z['next_state']).all():raise ValueError('Invalid next state')
        if not np.array_equal(z['state'][1:],z['next_state'][:-1]):raise ValueError('Observation transitions are misaligned')
    for field,expected in [('frame',np.arange(n)),('next_frame',np.arange(1,n+1))]:
        if field in z and not np.array_equal(z[field],expected):raise ValueError('Frame sequence mismatch')
    return dict(state_names=states,action_names=actions,frames=n,schema=str(z['schema'].item()) if 'schema' in z else None,execution_profile=str(z['execution_profile'].item()) if 'execution_profile' in z else None)
