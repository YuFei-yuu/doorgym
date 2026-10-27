"""Audit actual latch and terminal evidence; no simulation or state writes."""
import json,sys,math
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);a=p/'attempt_000';out=p/'review';rows=[json.loads(l) for l in (out/'compact-actual-trace.jsonl').open()];result=json.loads((a/'result.json').read_text());events=json.loads((out/'event-index.json').read_text());locks=[json.loads(l) for l in (a/'lock-writes.jsonl').open()]
first=next((r for r in rows if r['latched'] is not None),None);prev=rows[first['step']-1] if first else None
relocked=[x for x in locks if x['kind']=='relocked'];invalid_initial=[x['step'] for x in locks if x['kind']=='initial' and x['step']>0 and rows[x['step']-1]['unlocked']]
expected=first['latched'] if first else None
r=dict(latch_observed=first is not None,first_latch_post_time_s=first['t'] if first else None,previous_actual_door_q=prev['door_q'] if prev else None,previous_actual_door_qd=prev['door_qd'] if prev else None,previous_task_monitor=prev['task_monitor'] if prev else None,first_relock_write=relocked[0] if relocked else None,invalid_initial_writes_after_unlock=invalid_initial,all_relock_targets_identical=all(x['latched']==expected for x in relocked),relock_target_equals_previous_actual_angle=bool(prev and prev['door_q'][0]==expected),
    reference_before_latch=bool(first and (first.get('compact_expert') or {}).get('reference_complete')),latch_hold_s=result['latch_observed_hold_s'],final_hold_s=result['final_observed_hold_s'],final_monitor=result['independent_task_monitor'],lock_write_count=len(locks),relock_write_count=len(relocked))
assert first and prev and abs(prev['door_q'][0])<=math.radians(1) and abs(prev['door_qd'][0])<=math.radians(1) and r['latch_hold_s']>=.5
assert r['relock_target_equals_previous_actual_angle'] and r['all_relock_targets_identical'] and not invalid_initial and r['reference_before_latch']
last=rows[-1];r['last_actual_state']=dict(base=last['actual_base'],q=last['arm_q'],qd=last['arm_qd'],door=last['door_q'],door_velocity=last['door_qd'],max_x=last['whole_geometry_max_x'])
closing=[x for x in rows if (x.get('compact_expert') or {}).get('phase')=='close'];r['closing_reference_tracking']=dict(max_position_error_m=max(x['compact_expert']['measured_position_error'] for x in closing),max_orientation_error_rad=max(x['compact_expert']['measured_orientation_error'] for x in closing),last_reference=closing[-1]['compact_expert']['reference'])
r['milestones']={name:next((x for x in events if (x.get('kind')==name if name not in ['release','retract','restore_heading','final_hold'] else x.get('phase')==name)),None) for name in ['dynamic_switch','base_follow_start','reference_complete','endpoint_stopped','latch','release','retract','restore_heading','final_hold']}
r['automatic_and_recorded_latch_checks_passed']=True
(out/'lock-and-terminal-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
