"""Read-only checks of actual emitted/applied/completed records. No policy calls."""
import json,math,hashlib
from pathlib import Path
import numpy as np


class CriticalEvidenceError(RuntimeError):pass


def stream(path):
    with Path(path).open() as f:
        for line in f:yield json.loads(line)


def verify_journal(path):
    sequence=0;chunk=0;index=-1;rows=None;pending=None;substeps=0;commands=0;completed=0;new_chunks=0;epoch=None
    for row in stream(path):
        event=row['event']
        if event=='emitted':
            if pending is not None and substeps!=2:raise CriticalEvidenceError('Prior action not physically acknowledged twice')
            e=row['receipt'];raw=np.asarray(row['policy_output'],np.float32)
            if e['emission_seq']!=sequence+1 or e['executed_horizon']!=25:raise CriticalEvidenceError('Worker sequence/horizon')
            if epoch is not None and e['epoch']!=epoch:raise CriticalEvidenceError('Unexpected worker reset')
            if e['new_chunk']:
                if chunk and index!=24:raise CriticalEvidenceError('Early replan')
                if e['chunk_id']!=chunk+1 or e['action_index']!=0 or e['queue_before']!=0:raise CriticalEvidenceError('Chunk boundary')
                rows=np.asarray(e['selected_chunk_actions'],np.float32)
                if rows.shape!=(25,9) or hashlib.sha256(rows.tobytes()).hexdigest()!=e['chunk_sha256']:raise CriticalEvidenceError('Actual sampled rows/hash')
                new_chunks+=1
            elif e['chunk_id']!=chunk or e['action_index']!=index+1:raise CriticalEvidenceError('Discontinuous chunk')
            index=e['action_index'];chunk=e['chunk_id'];epoch=e['epoch'];sequence+=1
            if not np.array_equal(raw,rows[index]) or not np.array_equal(raw,np.asarray(e['action'],np.float32)):raise CriticalEvidenceError('Worker action differs')
            if e['queue_after']!=24-index:raise CriticalEvidenceError('Worker actual queue depth')
            if not e['new_chunk'] and e['queue_before']!=25-index:raise CriticalEvidenceError('Pre-dequeue depth')
            pending=np.asarray(row['applied_action'],np.float32);substeps=0
        elif event=='command_applied':
            if row['emission_seq']!=sequence or row['substep']!=substeps or commands!=completed:raise CriticalEvidenceError('Command sequencing')
            if not np.array_equal(np.asarray(row['action'],np.float32),pending):raise CriticalEvidenceError('Actual target differs')
            commands+=1
        elif event=='physics_completed':
            if row['emission_seq']!=sequence or row['substep']!=substeps or commands!=completed+1:raise CriticalEvidenceError('Physics acknowledgement')
            substeps+=1;completed+=1
        elif event=='end':
            if row['emitted']!=sequence or row['last_action_completed_substeps']!=substeps:raise CriticalEvidenceError('End receipt')
        else:raise CriticalEvidenceError('Unknown journal event')
    return dict(emitted=sequence,new_chunks=new_chunks,commands=commands,completed_physics_steps=completed,
                last_substeps=substeps,last_chunk_index=index,full_chunks=sequence//25)


def audit_trial(directory):
    directory=Path(directory);result=json.loads((directory/'result.json').read_text())
    if result['error']:raise CriticalEvidenceError('Runtime/recording error: '+str(result['error']))
    proof=verify_journal(directory/'worker-applied-evidence.jsonl')
    if proof['completed_physics_steps']!=result['steps']:raise CriticalEvidenceError('Physics ledger vs result')
    if proof['emitted']!=(result['steps']+1)//2:raise CriticalEvidenceError('25Hz vs 50Hz record count')
    illegal={'nonfinite_state_or_action','noncontinuous_state_clock','hinge_write_during_unlocked_interval',
        'auxiliary_hinge_effort','not_physically_closed_before_latch','close_hold_below_0p5s',
        'latch_did_not_store_measured_angle','unlogged_pre_physics_hinge_change','unknown_lock_write'}
    score=json.loads((directory/'r11-score.json').read_text())
    if illegal.intersection(score['failures']):raise CriticalEvidenceError('Illegal runtime state: '+str(illegal.intersection(score['failures'])))
    step_count=0;previous=None;first_t=None;clipped=0;held_action=None
    inputs=iter(stream(directory/'policy-inputs-actions.jsonl'));current=None;last_base=np.zeros(2,np.float32);frame=0
    for row in stream(directory/'states.jsonl'):
        if row['step']!=step_count:raise CriticalEvidenceError('State step discontinuity')
        if step_count%2==0:
            current=next(inputs);s=np.asarray(current['state'],np.float32)
            if current['frame']!=frame or current['source_physics_step']!=step_count-1:raise CriticalEvidenceError('Observation clock')
            if not np.array_equal(s[:2],last_base):raise CriticalEvidenceError('Previous 25Hz actual base command')
            if abs(current['source_time_s']-row['pre_physics']['clock']['command_time_s'])>1e-8:raise CriticalEvidenceError('Image/action source clock')
            held_action=np.asarray(current['applied_action'],np.float32);last_base=held_action[:2].copy();frame+=1
            clipped+=int(np.any(np.asarray(current['clipping_delta'])!=0))
        if not np.array_equal(np.asarray(row['action'],np.float32),held_action):raise CriticalEvidenceError('Two-substep hold changed')
        if row['door_effort'][0]!=0:raise CriticalEvidenceError('Auxiliary hinge effort')
        if previous is not None and row['t']<=previous:raise CriticalEvidenceError('Non-increasing physics time')
        if first_t is None:first_t=row['t']
        previous=row['t'];step_count+=1
    if next(inputs,None) is not None or step_count!=result['steps']:raise CriticalEvidenceError('Input/state counts')
    proof.update(state_steps=step_count,input_frames=frame,clipped_action_frames=clipped,
        physical_dt_s=first_t,control_hz=25,physics_hz=50,r11_measured_pass=score['measured_task_gates_pass'],
        task_failures=score['failures'],runtime_evidence_passed=True,physical_push_release_review='separate-original-evidence-review')
    (directory/'r14-runtime-audit.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof
