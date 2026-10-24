"""Build immutable-source catalog after scheduling; never transforms raw data."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--preliminary',action='store_true');a=p.parse_args();r=a.run
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
if not a.preliminary:assert (r/'scheduler-exit.txt').exists()
m=read(r/'manifest.json');reuse=read(r/'reuse-index.json');rows=[]
decisions=read(r/'source-review-decisions.json') if (r/'source-review-decisions.json').exists() else None
if not a.preliminary and decisions is not None:assert sorted(d['source_index'] for d in decisions['sources'])==list(range(50))
for old in reuse['sources']:
    files=old['files']
    rows.append({'source_index':old['source_index'],'source_id':old['source_id'],'kind':old['kind'],'seed':old['seed'],
        'status':'reused_original_bytes','source_dir':old['source_dir'],'execution_commit':old['execution_commit'],
        'automatic_terminal':old['automatic_terminal'],'complete_task_review':old['complete_task_review'],
        'r11_measured_gates_pass':True,
        'raw_quality_pass':old['recording_checks_pass'],'frames':old['frames'],'files':files,
        'raw_npz':files['attempt_000/raw/episode_000000.npz'],'reexecuted':False,'video_scope':'Original stage-six RGB retained at original source',
        'provisional_eligible':True,'exclusion_or_hold_reasons':[],'full_raw_episode':True,'trimmed':False,'spliced':False})
for t in m['trials']:
    i=t['index'];run=r/('trial_%02d'%i);out=run/'attempt_000';audit=run/'audit'
    row={'source_index':t['source_index'],'source_id':'stage7-seed%d'%t['seed'],'kind':'new_stage7','seed':t['seed'],
        'new_trial_index':i,'source_dir':str(run),'execution_commit':m['start_commit'],'status':'not_started',
        'automatic_terminal':None,'complete_task_review':'not_executed','raw_quality_pass':None,'frames':None,
        'files':{},'raw_npz':None,'reexecuted':False,'video_scope':'No video, explicitly authorized for new stage-seven trials',
        'provisional_eligible':False,'exclusion_or_hold_reasons':['not_started'],'full_raw_episode':None,'trimmed':False,'spliced':False}
    if (run/'scene-claim.json').exists():row['status']='scene_created_without_completed_raw_audit';row['exclusion_or_hold_reasons']=['raw_or_task_audit_not_complete']
    if (out/'result.json').exists():
        result=read(out/'result.json');row.update(status='physics_finished',automatic_terminal=result['automatic_final_conditions_met'],
            physics_steps=result['steps'],sim_seconds=result['sim_seconds'],stop_reason=result['reason'])
    if (audit/'record-validation.json').exists():
        q=read(audit/'record-validation.json');row.update(raw_quality_pass=q['recording_checks_pass'],frames=q['frames'],raw_quality_reasons=q['systemic_errors'])
    if (audit/'task-review.json').exists():
        v=read(audit/'task-review.json');row.update(status='reviewed',complete_task_review=v['complete_task_review'],
            r11_measured_gates_pass=v['r11_measured_gates_pass'],provisional_eligible=v['eligible_raw_episode'],
            exclusion_or_hold_reasons=v['exclusion_or_hold_reasons'],failure_or_stop_phase=v['failure_or_stop_phase'],
            intended_contact_face=v['contact_face_diagnostic'],nonend_close_records=len(v['ambiguous_nonend_close_records']),
            initial_binding_pass=v['initial_binding_pass'],physical_binding_pass=v['physical_binding_pass'],full_raw_episode=True)
    required=['attempt_000/raw/episode_000000.npz','attempt_000/states.jsonl','attempt_000/contacts.jsonl',
        'attempt_000/lock-writes.jsonl','attempt_000/recording-clock.jsonl','attempt_000/result.json','attempt_000/command.json',
        'attempt_000/initial-state.json','attempt_000/frozen-trial.json','attempt_000/physical-camera-binding.json',
        'attempt_000/trajectory.npz','attempt_000/scene-frame.json','attempt_000/body-map.json','attempt_000/reference.json','attempt_000/stop-state.json',
        'audit/record-validation.json','audit/r11-score.json','audit/task-review.json']
    if not a.preliminary:
        for name in required:
            path=run/name
            if path.is_file():row['files'][name]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
    if 'attempt_000/raw/episode_000000.npz' in row['files']:row['raw_npz']=row['files']['attempt_000/raw/episode_000000.npz']
    rows.append(row)
assert [x['source_index'] for x in rows]==list(range(50))
for row in rows:
    d=next((d for d in decisions['sources'] if d['source_index']==row['source_index']),None) if decisions else None
    row['final_complete_task_qualified']=bool(d and d.get('complete_task_qualified'))
    row['final_admitted']=bool(d and d['admitted'] and row['raw_quality_pass'] and row.get('r11_measured_gates_pass') and row['final_complete_task_qualified'])
    row['final_review_decision']=d if d else {'status':'pending_final_source_review'}
    if d and d['admitted']:assert row['raw_quality_pass'] and row.get('r11_measured_gates_pass') and row['final_complete_task_qualified'] and row['raw_npz']
    if not a.preliminary and row['kind']=='reused_stage6_original':
        for name,f in row['files'].items():assert sha(Path(f['path']))==f['sha256'],'Reused source changed: '+f['path']
report={'source_slots':50,'reused_sources':8,'planned_new_trials':42,'rows':rows,
    'new_scene_attempts':sum((r/('trial_%02d'%i)/'scene-claim.json').exists() for i in range(42)),
    'new_physics_results':sum('physics_steps' in x for x in rows if x['kind']=='new_stage7'),
    'new_physics_nonzero_trials':sum(x.get('physics_steps',0)>0 for x in rows if x['kind']=='new_stage7'),
    'new_raw_npz_present':sum(x['raw_npz'] is not None for x in rows if x['kind']=='new_stage7'),
    'new_physics_steps':sum(x.get('physics_steps',0) for x in rows if x['kind']=='new_stage7'),
    'new_sim_seconds':sum(x.get('sim_seconds',0) for x in rows if x['kind']=='new_stage7'),
    'provisional_eligible_count':sum(x['provisional_eligible'] for x in rows),'final_admitted_count':sum(x['final_admitted'] for x in rows),
    'not_admitted_indices':[x['source_index'] for x in rows if not x['final_admitted']],
    'new_unstarted_indices':[x['new_trial_index'] for x in rows if x['kind']=='new_stage7' and x['status']=='not_started'],
    'source_frame_total_reported':sum(x.get('frames') or 0 for x in rows),
    'source_frame_total_quality_pass':sum(x.get('frames') or 0 for x in rows if x['raw_quality_pass']),
    'selected_frame_total':sum(x['frames'] for x in rows if x['final_admitted']),
    'conversion':0,'training_updates':0,'raw_stage_only':True,'preliminary':a.preliminary,
    'systemic_stop':read(r/'SYSTEMIC_STOP.json') if (r/'SYSTEMIC_STOP.json').exists() else None}
target=r/('unified-preliminary.json' if a.preliminary else 'unified-source-index.json')
target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
if not a.preliminary:
    assert decisions is not None,'Final source-review decisions required for admission'
    candidates={'raw_stage_only':True,'source_index_sha256':sha(target),'episodes':[{
        'source_index':x['source_index'],'source_id':x['source_id'],'seed':x['seed'],'frames':x['frames'],'raw_npz':x['raw_npz'],
        'source_dir':x['source_dir'],'full_episode':True,'frame_slice':None,'transformed':False} for x in rows if x['final_admitted']]}
    (r/'candidate-index.json').write_text(json.dumps(candidates,indent=2,ensure_ascii=False)+'\n')
    lines=['# 50个来源回合统一索引','','|来源序号|Seed|来源|状态|自动终态|完整任务审核|原始质量|帧数|入选|未入选/限制|',
        '|---|---:|---|---|---|---|---|---:|---|---|']
    for x in rows:lines.append('|%02d|%d|%s|%s|%s|%s|%s|%s|%s|%s|'%(x['source_index'],x['seed'],x['kind'],x['status'],str(x['automatic_terminal']),x['complete_task_review'],str(x['raw_quality_pass']),str(x['frames']),str(x['final_admitted']),';'.join(x['exclusion_or_hold_reasons']+[x['final_review_decision'].get('reason','')])))
    (r/'SOURCE_TABLE.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False))
