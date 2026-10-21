"""R11 offline task scoring. No simulator, actions, expert phases or model inputs.

Physical push still needs a separate original-video review; a contact flag or
automatic terminal result never authorizes admission by itself.
"""
import math
import numpy as np


def wrap(x):
    return (x+math.pi)%(2*math.pi)-math.pi


def terminal(s, ref):
    return bool(s['whole_geometry_max_x']<s['pass_plane_x']
        and abs(s['door_q'][0])<=math.radians(1) and abs(s['door_qd'][0])<=math.radians(1)
        and s['latched'] is not None and not s['unlocked']
        and not s['loaded_door_contact']
        and abs(wrap(s['actual_base'][2]-ref['old_expert_final_yaw']))<=math.radians(2)
        and s['base_measured_fd_speed']<=.01 and s['base_measured_fd_yaw_rate']<=.01
        and np.all(abs(np.asarray(s['arm_q'][:6])-ref['home'][:6])<=.03)
        and np.all(abs(np.asarray(s['arm_qd'][:6]))<=.05))


def score(states, locks, ref):
    n=len(states)
    failures=[]
    t=np.array([s['t'] for s in states],dtype=np.float64)
    finite=np.concatenate([np.asarray([s[k] for s in states],float).ravel()
                           for k in ['t','action','actual_base','arm_q','arm_qd','ee_actual','door_q','door_qd']])
    if not np.isfinite(finite).all():failures.append('nonfinite_state_or_action')
    if [s['step'] for s in states]!=list(range(n)) or np.any(np.diff(t)<=0):failures.append('noncontinuous_state_clock')
    far=[s['whole_geometry_max_x']<s['pass_plane_x'] for s in states]
    passed=next((i for i,v in enumerate(far) if v),None)
    returned=passed is not None and not all(far[passed:])
    if passed is None:failures.append('whole_body_not_passed')
    if returned:failures.append('returned_to_original_side')
    opened=max(abs(s['door_q'][0]) for s in states)>=math.radians(80)
    if not opened:failures.append('opening_below_80deg')
    unlock=next((i for i,s in enumerate(states) if s['unlocked']),None)
    rel=[x for x in locks if x['kind']=='relocked']
    latch=rel[0] if rel else None
    latch_step=latch['step'] if latch else n
    if latch is None:failures.append('no_relock')
    if any(x['kind'] not in ['initial','relocked'] for x in locks):failures.append('unknown_lock_write')
    if unlock is None:failures.append('no_unlock')
    else:
        if any(unlock<x['step']<latch_step for x in locks):failures.append('hinge_write_during_unlocked_interval')
        if any(s['door_effort'][0]!=0 for s in states[unlock:latch_step]):failures.append('auxiliary_hinge_effort')
    hold=None
    if latch:
        k=latch['step'];b=k-1
        closed=lambda s:abs(s['door_q'][0])<=math.radians(1) and abs(s['door_qd'][0])<=math.radians(1) and s['latched'] is None
        if k<=0 or k>=n or not closed(states[k-1]):failures.append('not_physically_closed_before_latch')
        else:
            while b>0 and closed(states[b-1]):b-=1
            hold=dict(first_measured_time_s=states[b]['t'],latch_pre_time_s=latch['time_s'],
                      span_s=latch['time_s']-states[b]['t'],measured_angle=states[k-1]['door_q'][0],latched=latch['latched'])
            if hold['span_s']<.5:failures.append('close_hold_below_0p5s')
            if latch['latched']!=states[k-1]['door_q'][0] or latch['delta']!=0:failures.append('latch_did_not_store_measured_angle')
    # Check logged pre/post state continuity outside declared lock writes.
    writes={x['step'] for x in locks}
    unexplained=[]
    for i in range(1,n):
        if i not in writes and states[i]['pre_physics']['door_q'][0]!=states[i-1]['door_q'][0]:unexplained.append(i)
    if unexplained:failures.append('unlogged_pre_physics_hinge_change')
    separated=next((i for i in range(passed or n,n) if not states[i]['any_door_contact']),None)
    push=[]
    if separated is not None:
        for i in range(separated+1,min(n,latch_step)):
            s=states[i]
            if s['end_panel_raw_contact'] and s['door_q'][0]<s['pre_physics']['door_q'][0] and np.linalg.norm(np.asarray(s['ee_actual'])-states[i-1]['ee_actual'])>0:push.append(i)
    if not push:failures.append('no_aligned_measured_end_push_evidence')
    b=n-1
    while b>0 and terminal(states[b-1],ref):b-=1
    final_span=t[-1]-t[b] if terminal(states[-1],ref) else 0.
    if final_span<2:failures.append('final_joint_hold_below_2s')
    negative=[s for s in states if s['action'][0]<0]
    cmd=np.array([s['action'] for s in states]);changed=np.any(np.diff(cmd,axis=0)!=0,axis=1)
    return dict(measured_task_gates_pass=not failures,failures=failures,opening_max_abs_deg=math.degrees(max(abs(s['door_q'][0]) for s in states)),
        whole_body_pass_step=passed,returned=returned,old_contact_separation_step=separated,
        first_aligned_end_push_step=push[0] if push else None,aligned_end_push_steps=push,
        push_door_angle_drop_deg=math.degrees(sum(states[i]['pre_physics']['door_q'][0]-states[i]['door_q'][0] for i in push)),
        close_hold=hold,final_common_first_time_s=float(t[b]) if final_span else None,final_common_span_s=float(final_span),
        unlogged_hinge_changes=unexplained,diagnostics=dict(negative_command_steps=len(negative),
        minimum_raw_vx=min((s['action'][0] for s in states),default=0),
        physics_dt_min_s=float(np.min(np.diff(t))),physics_dt_max_s=float(np.max(np.diff(t))),
        command_changed_intervals=int(changed.sum()),command_held_intervals=int((~changed).sum()),
        mean_command_change_rate_hz=float(changed.sum()/(t[-1]-t[0])),
        zero_lambda_door_record_steps=sum(s['zero_lambda_door_records']>0 for s in states),
        nonend_door_record_steps=[s['step'] for s in states if s['nonend_door_records'] and passed is not None and passed<=s['step']<latch_step]),
        requires_original_video_and_code_review=True,phase_labels_used_for_score=False,
        release_evidence='No nonzero door records in final measured hold, EE/home withdrawal and original-video review. Zero-lambda rows remain diagnostic, not contact proof.')
