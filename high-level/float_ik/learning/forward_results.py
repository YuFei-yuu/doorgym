"""Full-attempt task results, autonomy audits, and forward-path diagnostics."""
import json
from pathlib import Path

import numpy as np


def episode_result(folder, case, exit_code, policy=True):
    folder = Path(folder)
    summary_path, log_path = folder/'summary.json', folder/'execution.jsonl'
    log_error = None
    try:
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    except (ValueError, OSError):
        summary = {}
        log_error = 'invalid_summary'
    rows = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                row = json.loads(line)
                if not all(k in row for k in ['step', 'extra', 'door_after', 'root_before', 'base_after']):
                    raise ValueError('Incomplete execution row')
                rows.append(row)
            except (ValueError, TypeError):
                log_error = 'truncated_or_incomplete_execution_log'
                break
    n = len(rows)
    sources = [r['extra'].get('command_source') for r in rows]
    complete = log_error is None and n > 0 and n == summary.get('frames') and all(r.get('step') == i for i, r in enumerate(rows))
    autonomous = bool(complete and sources.count('dp') == n and
                      all(r['extra'].get('dp_action_applied') for r in rows) and
                      summary.get('dp_frames') == n and summary.get('script_frames') == 0 and
                      summary.get('invalid_image_frames') == 0 and
                      all(r['extra'].get('image_valid') for r in rows))
    angles = -np.rad2deg([r['door_after'][0] for r in rows])
    opened = bool(exit_code == 0 and complete and summary.get('opened') and np.any(angles >= 80))
    pass_rows = [r for r in rows if r['extra'].get('pass_done')]
    # pass_done is latched by the source before applying this frame's policy.
    distance = (float(np.linalg.norm(np.asarray(pass_rows[0]['root_before'][:2]) -
                                    pass_rows[0]['extra']['through_target'])) if pass_rows else None)
    passed = bool(exit_code == 0 and complete and summary.get('pass_done') and
                  distance is not None and distance <= .12001)
    success = bool(opened and passed and summary.get('success') and (autonomous or not policy))
    latency = [r['extra']['inference_sec']*1000 for r in rows
               if r['extra'].get('dp_replanned') and r['extra'].get('inference_sec') is not None]
    result = dict(case, folder=str(folder), exit_code=exit_code, frames=n,
                  complete_log=complete, autonomous=autonomous, opened=opened, pass_done=passed,
                  success=success, max_signed_door_deg=float(angles.max()) if n else None,
                  final_signed_door_deg=float(angles[-1]) if n else None,
                  first_open_frame=summary.get('first_open_frame'), first_pass_frame=summary.get('first_pass_frame'),
                  through_distance=summary.get('through_distance'), pass_event_distance_m=distance,
                  dp_frames=sources.count('dp'), script_frames=sources.count('script'),
                  snap_events=sum(r.get('pre_observation_event') == 'upstream_snap_arm_to_zero' for r in rows),
                  invalid_image_frames=summary.get('invalid_image_frames'),
                  pass_align_frames=sum(r.get('phase_name') == 'pass_align' for r in rows),
                  backward_travel_after80_m=None, yaw_travel_after80_deg=None,
                  replan_latency_p50_ms=float(np.percentile(latency,50)) if latency else None,
                  replan_latency_p95_ms=float(np.percentile(latency,95)) if latency else None,
                  end_reason=summary.get('end_reason','process_failure'), error=summary.get('error'), log_error=log_error)
    if n and np.any(angles >= 80):
        first = int(np.flatnonzero(angles >= 80)[0])
        roots = np.array([r['base_after'] for r in rows])
        q = roots[:,3:7]
        yaw = np.unwrap(np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),
                                  1-2*(q[:,1]**2+q[:,2]**2)))
        initial = rows[0]['root_before'][3:7]
        theta = np.arctan2(2*(initial[3]*initial[2]+initial[0]*initial[1]),
                           1-2*(initial[1]**2+initial[2]**2))
        forward = np.diff(roots[first:,:2],axis=0) @ [np.cos(theta),np.sin(theta)]
        result['backward_travel_after80_m'] = float(np.maximum(0,-forward).sum())
        result['yaw_travel_after80_deg'] = float(np.rad2deg(np.abs(np.diff(yaw[first:])).sum()))
    return result


def aggregate(rows, scheduled=None):
    n = len(rows)
    nominal = [r for r in rows if r['robot_x'] == 4.1 and r['robot_y'] == 0]
    return dict(attempts=n, scheduled=n if scheduled is None else scheduled,
                opened=sum(r['opened'] for r in rows), passed=sum(r['pass_done'] for r in rows),
                successes=sum(r['success'] for r in rows),
                opening_rate=sum(r['opened'] for r in rows)/n if n else 0.,
                passing_rate=sum(r['pass_done'] for r in rows)/n if n else 0.,
                success_rate=sum(r['success'] for r in rows)/n if n else 0.,
                nominal_attempts=len(nominal), nominal_successes=sum(r['success'] for r in nominal),
                all_scheduled_completed=scheduled is None or n == scheduled)
