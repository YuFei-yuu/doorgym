"""Strict offline acceptance of real expert motion, independent of commands."""
import math
import numpy as np


def measured_motion(rows, dt):
    before = np.asarray([r['root_before'] for r in rows], dtype=np.float64)
    after = np.asarray([r['base_after'] for r in rows], dtype=np.float64)
    distance = np.linalg.norm(after[:, :3]-before[:, :3], axis=1)
    qa, qb = before[:, 3:7], after[:, 3:7]
    qa = qa/np.linalg.norm(qa, axis=1, keepdims=True)
    qb = qb/np.linalg.norm(qb, axis=1, keepdims=True)
    angular_speed = 2*np.arccos(np.clip(np.abs(np.sum(qa*qb, axis=1)), 0, 1))/dt
    return before, after, distance, distance/dt, angular_speed


def stop_acceptance(rows, dt, motion_sign=-1., target_tolerance=.12,
                    expected_source='script', required_post_pass_seconds=3.):
    before, after, distance, speed, omega = measured_motion(rows, dt)
    passes = [i for i, r in enumerate(rows) if r['extra']['pass_done']]
    p = passes[0] if passes else None
    signed_angles = motion_sign*np.rad2deg([r['door_after'][0] for r in rows])
    checks = dict(contiguous_steps=[r['step'] for r in rows] == list(range(len(rows))),
        physical_steps=all(r['completed_physical_steps'] == (i+1)*4 for i, r in enumerate(rows)),
        root_continuity=bool(np.allclose(after[:-1, :7], before[1:, :7], rtol=0, atol=1e-6)),
        opened_80=bool(np.max(signed_angles) >= 80.), pass_done=p is not None,
        forward_no_alignment=all(r['phase_id'] != 10 for r in rows),
        image_valid=all(r['extra']['image_valid'] for r in rows),
        command_source_matches=all(r['extra']['command_source'] == expected_source and
            r['extra']['dp_action_applied'] == (expected_source == 'dp') for r in rows),
        finite_motion=bool(np.isfinite(before).all() and np.isfinite(after).all()))
    result = dict(first_pass_step=p, frames=len(rows), max_signed_door_deg=float(np.max(signed_angles)), checks=checks)
    if p is None:
        checks['post_pass_complete'] = False
        result['passed'] = False
        return result
    event_distance = float(np.linalg.norm(before[p, :2]-rows[p]['extra']['through_target']))
    good = (speed < .01) & (omega < .02)
    persistent = np.logical_and.accumulate(good[p:][::-1])[::-1]
    candidates = np.flatnonzero(persistent)
    stop = p+int(candidates[0]) if len(candidates) else None
    checks.update(original_target_distance=event_distance <= target_tolerance,
        pass_latched=all(r['extra']['pass_done'] for r in rows[p:]),
        post_pass_complete=(len(rows)-1-p)*dt >= required_post_pass_seconds-1e-6,
        settles_within_half_second=stop is not None and (stop-p)*dt <= .5+1e-6)
    result.update(event_target_distance_m=event_distance, post_pass_observation_span_sec=(len(rows)-1-p)*dt,
        post_pass_control_intervals=len(rows)-p, stop_step=stop,
        max_post_pass_measured_speed_m_s=float(speed[p:].max()),
        max_post_pass_measured_angular_speed_rad_s=float(omega[p:].max()),
        post_pass_total_path_m=float(distance[p:].sum()))
    if stop is not None:
        hold_steps = int(round(2./dt))
        hold = after[stop:stop+hold_steps, :3]
        drift = float(np.linalg.norm(hold-before[stop, :3], axis=1).max()) if len(hold) else math.inf
        path = float(distance[p:stop+1].sum())
        checks.update(two_second_hold_complete=len(hold) == hold_steps,
            maximum_two_second_drift=drift <= .02, extra_path_until_stopped=path <= .10,
            no_later_movement=bool(good[stop:].all()))
        result.update(settle_latency_sec=(stop-p)*dt, max_two_second_drift_m=drift,
            extra_path_until_stopped_m=path, max_remaining_speed_m_s=float(speed[stop:].max()),
            max_remaining_angular_speed_rad_s=float(omega[stop:].max()),
            max_remaining_position_drift_m=float(np.linalg.norm(after[stop:, :3]-before[stop, :3], axis=1).max()))
    result['passed'] = all(checks.values())
    return result
