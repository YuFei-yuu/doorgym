"""Aggregate completed trials only, on pro5000 CPU; no simulation or scoring changes."""
import argparse,collections,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path('/data/zyf/video2door-wc4-act')
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
assert (r/'lease-released.txt').exists(),'Only aggregate after scheduler release'
if (r/'continue-remaining.sh').exists():assert (r/'continuation-lease-released.txt').exists()
m=read(r/'manifest.json');cases=[]
for trial in m['trials']:
    i=trial['index'];run=r/('trial_%02d'%i);out=run/'attempt_000';audit=run/'audit'
    item={'index':i,'seed':trial['seed'],'scene_claimed':(run/'scene-claim.json').exists(),
          'status':'not_started','r11_measured_gates_pass':None,'review_decision':'not_executed',
          'expected_spawn':trial['expected_spawn']}
    if not (out/'result.json').exists():
        if item['scene_claimed']:item.update(status='infrastructure_interruption',review_decision='no_complete_task_evidence')
        cases.append(item);continue
    result=read(out/'result.json');record=read(audit/'record-validation.json') if (audit/'record-validation.json').exists() else None
    summary=read(audit/'summary.json') if (audit/'summary.json').exists() else None
    scoring=read(audit/'r11-score.json') if (audit/'r11-score.json').exists() else None
    phases=read(audit/'phase-intervals.json') if (audit/'phase-intervals.json').exists() else []
    first={s['phase']:s['first_time_s'] for s in phases}
    loaded=collections.defaultdict(list);close_points=[];close_nonend=[]
    events=audit/'post-pass-door-contacts.jsonl'
    if events.exists():
        for line in events.open():
            e=json.loads(line)
            if e['lambda']==0:continue
            loaded[e['phase']+' | '+e['pair']].append(e)
            if e['phase']=='close':
                if not e['end_panel']:close_nonend.append(e)
                elif 'gripperStator' in e['pair']:
                    c=e['raw_contact'];j=next(j for j,x in enumerate(c['names']) if x['body']=='gripperStator')
                    close_points.append([c['localPos'+str(j)][k] for k in ['x','y','z']])
    contact={'loaded_groups':{k:{'records':len(v),'first_step':v[0]['step'],'last_step':v[-1]['step'],
        'first_time_s':v[0]['time_s'],'last_time_s':v[-1]['time_s']} for k,v in loaded.items()},
        'close_loaded_nonend_records':len(close_nonend),
        'point_limit':'Raw local contact positions support source-face identification; no cooked subshape IDs or new force criterion.'}
    if close_points:
        v=np.array(close_points);contact['close_stator_source_face']={'records':len(v),'min_xyz_m':v.min(0).tolist(),
            'max_xyz_m':v.max(0).tolist(),'source_face_x_m':.15,'max_x_offset_m':float(abs(v[:,0]-.15).max())}
    # Preserve non-door/self diagnostics too; these are not new R11 criteria.
    exceptional={};phase_by_step=['unknown']*result['steps']
    for interval in phases:
        for step in range(interval['first_step'],interval['last_step']+1):phase_by_step[step]=interval['phase']
    for line in (out/'contacts.jsonl').open():
        row=json.loads(line)
        for c in row['contacts']:
            classification=c.get('classification','unclassified')
            if c.get('lambda',0)==0 or classification not in ['unexpected_external_loaded','other_self_contact_loaded']:continue
            key=phase_by_step[row['step']]+' | '+classification+' | '+' / '.join(x.get('body','unknown') for x in c['names'])
            if key not in exceptional:exceptional[key]={'records':0,'first_step':row['step'],'last_step':row['step']}
            exceptional[key]['records']+=1;exceptional[key]['last_step']=row['step']
    contact['other_loaded_contact_diagnostics']=exceptional
    if audit.exists():write(audit/'contact-review-details.json',contact)
    item.update(status='finished_physics',automatic_terminal=result['automatic_final_conditions_met'],
        sim_seconds=result['sim_seconds'],physics_steps=result['steps'],stop_reason=result['reason'],
        last_phase=phases[-1]['phase'] if phases else None,recording_checks_pass=record['recording_checks_pass'] if record else False,
        r11_measured_gates_pass=scoring['measured_task_gates_pass'] if scoring else None,
        r11_failures=scoring['failures'] if scoring else ['missing_score'],
        parts={'opening':bool(scoring and scoring['opening_max_abs_deg']>=80),
               'whole_body_pass':bool(scoring and scoring['whole_body_pass_step'] is not None and not scoring['returned']),
               'bypass_entered':'compact_bypass' in first,'deployment_arrived':'compact_seek' in first,
               'aligned_end_push':bool(scoring and scoring['first_aligned_end_push_step'] is not None),
               'latch':bool(scoring and scoring['close_hold'] is not None),
               'release_entered':'release' in first,'retract_entered':'retract' in first,
               'final_common_hold':bool(scoring and scoring['final_common_span_s']>=2)},
        phase_start_times=first,recorded_rgb_frames=record['frames'] if record else None,
        deployment_gate=summary['deployment_arrival_gate'] if summary else None,
        deployment_duration_s=summary['deployment_duration_s'] if summary else None,
        deployment_max_reference_tracking_m=summary['deployment_max_pad_reference_tracking_m'] if summary else None,
        constant_root_z_range_m=summary['root_z_range_m'] if summary else None,
        final_home_error_rad=summary['final_joint_home_error_rad'] if summary else None,
        final_heading_error_rad=summary['final_heading_error_rad'] if summary else None,
        contact_details=contact,event_index=str(audit/'event-index.json'),
        original_movie=str(out/'overview/env_0000/rollout.mp4'),
        review_decision='pending_original_rgb_contact_code_review')
    cases.append(item)
report={'trials':cases,'authorized_trials':8,'scene_claims':sum(c['scene_claimed'] for c in cases),
    'physical_trials_with_result':sum(c['status']=='finished_physics' for c in cases),
    'physics_steps':sum(c.get('physics_steps',0) for c in cases),'sim_seconds':sum(c.get('sim_seconds',0) for c in cases),
    'automatic_terminal_count':sum(c.get('automatic_terminal',False) for c in cases),
    'r11_measured_gate_pass_count':sum(c.get('r11_measured_gates_pass') is True for c in cases),
    'unstarted_indices':[c['index'] for c in cases if c['status']=='not_started'],
    'first_scheduler_diagnostic_stop':(r/'SYSTEMIC_STOP.txt').read_text().strip() if (r/'SYSTEMIC_STOP.txt').exists() else None,
    'systemic_stop':(r/'CONTINUATION_SYSTEMIC_STOP.txt').read_text().strip() if (r/'CONTINUATION_SYSTEMIC_STOP.txt').exists() else None,
    'stage5_infrastructure_attempt_separate':{'scene_creations':1,'physics_steps':0,'expert_failures':0},
    'training_updates':0,'act_evaluations':0,'gpu_locks_held':False,'monitors_active':False}
write(r/'aggregate.json',report)
lines=['# 第六轮逐项实测结果（与原片联合审查分开记录）','',
    '|Trial|Seed|时长s|自动终态|R11实测门槛|停止位置|失败项|','|---|---:|---:|---|---|---|---|']
for c in cases:
    lines.append('|%02d|%d|%s|%s|%s|%s|%s|'%(c['index'],c['seed'],str(round(c.get('sim_seconds',0),2)),
        str(c.get('automatic_terminal')),str(c.get('r11_measured_gates_pass')),c.get('last_phase',c['status']),','.join(c.get('r11_failures',[]))))
(r/'PER_TRIAL.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2))
