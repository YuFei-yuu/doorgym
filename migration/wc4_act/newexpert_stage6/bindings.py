"""Spawn identity checks at float32 storage precision; never changes simulation."""
import math
import numpy as np

EPS32=float(np.finfo(np.float32).eps)
ROTATION_ATOL_RAD=16*EPS32
UNIT_NORM_ATOL=32*EPS32

def quaternion_rotation_error(actual,expected):
    """Normalize near-unit inputs and compare SO(3), including q == -q.

    Four float32 components plus unit normalization have O(eps32) rounding.
    16 eps32 radians bounds several such operations (1.90735e-6 rad), while
    32 eps32 norm error permits float32 normalization but rejects malformed
    scale. atan2 of the relative quaternion avoids acos near-one cancellation.
    """
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(expected,dtype=np.float64)
    if a.shape!=(4,) or b.shape!=(4,) or not np.isfinite(np.r_[a,b]).all():
        return {'valid':False,'reason':'nonfinite_or_wrong_shape_quaternion'}
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    if abs(na-1)>UNIT_NORM_ATOL or abs(nb-1)>UNIT_NORM_ATOL:
        return {'valid':False,'reason':'invalid_quaternion_norm','actual_norm':na,'expected_norm':nb}
    a=a/na;b=b/nb
    v=b[3]*a[:3]-a[3]*b[:3]-np.cross(a[:3],b[:3])
    angle=2*math.atan2(float(np.linalg.norm(v)),abs(float(np.dot(a,b))))
    return {'valid':True,'rotation_error_rad':angle,'actual_norm':na,'expected_norm':nb,
        'rotation_atol_rad':ROTATION_ATOL_RAD,'unit_norm_atol':UNIT_NORM_ATOL,
        'matches':angle<=ROTATION_ATOL_RAD}

def position_check(actual,expected):
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(expected,dtype=np.float64)
    if a.shape!=b.shape or not np.isfinite(np.r_[a.ravel(),b.ravel()]).all():
        return {'matches':False,'reason':'nonfinite_or_wrong_shape_position'}
    # Four float32 rounding units at coordinate scale. This is a storage check,
    # not a navigation/physics tolerance. Also check actual root xy, not metadata alone.
    tolerance=4*EPS32*np.maximum(1,abs(b))
    return {'matches':bool(np.all(abs(a-b)<=tolerance)),
        'difference':(a-b).tolist(),'atol_m':tolerance.tolist()}

def compare_spawn(current,expected):
    pose=current['root']['pose'];p=pose['p'];q=pose['r']
    root=position_check([p[k] for k in ['x','y','z']],[*expected['base_start'],expected['robot_z']])
    base=position_check(current['base_start'],expected['base_start'])
    rotation=quaternion_rotation_error([q[k] for k in ['x','y','z','w']],expected['root_quat'])
    yaw_a=float(current['yaw']);yaw_b=float(expected['yaw'])
    yaw_finite=math.isfinite(yaw_a) and math.isfinite(yaw_b)
    yaw_error=abs((yaw_a-yaw_b+math.pi)%(2*math.pi)-math.pi) if yaw_finite else None
    yaw_match=yaw_finite and yaw_error<=ROTATION_ATOL_RAD
    return {'matches_frozen':bool(root['matches'] and base['matches'] and rotation.get('matches',False) and yaw_match),
        'root_position':root,'base_start':base,'rotation':rotation,
        'yaw':{'finite':yaw_finite,'wrapped_error_rad':yaw_error,'matches':yaw_match},
        'semantics':'Storage identity only; no expert, physics, R11 or trajectory tolerance changed.'}

def recording_contract(steps,state_rows,contact_rows,clock_steps,rgb_frames,rgb_fps,depth_receipt,depth_video_exists):
    expected=(steps+1)//2;fail=[]
    if state_rows!=list(range(steps)):fail.append('state_step_continuity')
    if contact_rows!=list(range(steps)):fail.append('contact_step_continuity')
    if clock_steps!=list(range(0,steps,2)):fail.append('recording_clock_continuity')
    if rgb_frames!=expected or rgb_fps!=25:fail.append('rgb_count_or_rate')
    if depth_video_exists:fail.append('unexpected_depth_video_encoding')
    if depth_receipt.get('encoded') is not False or depth_receipt.get('raw_depth_observations_retained') is not True:
        fail.append('depth_encoding_receipt')
    if depth_receipt.get('buffers_observed')!={'front_depth.mp4':expected,'wrist_depth.mp4':expected}:
        fail.append('depth_buffer_count')
    return {'passed':not fail,'failures':fail,'frames':expected}
