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
            if e['emission_seq']!=sequence+1 or e['executed_horizon']!=15:raise CriticalEvidenceError('Worker sequence/horizon')
            if epoch is not None and e['epoch']!=epoch:raise CriticalEvidenceError('Unexpected worker reset')
            if e['new_chunk']:
                if chunk and index!=14:raise CriticalEvidenceError('Early replan')
                if e['chunk_id']!=chunk+1 or e['action_index']!=0 or e['queue_before']!=0:raise CriticalEvidenceError('Chunk boundary')
                rows=np.asarray(e['selected_chunk_actions'],np.float32)
                if rows.shape!=(15,9) or hashlib.sha256(rows.tobytes()).hexdigest()!=e['chunk_sha256']:raise CriticalEvidenceError('Actual sampled rows/hash')
                new_chunks+=1
            elif e['chunk_id']!=chunk or e['action_index']!=index+1:raise CriticalEvidenceError('Discontinuous chunk')
            index=e['action_index'];chunk=e['chunk_id'];epoch=e['epoch'];sequence+=1
            if not np.array_equal(raw,rows[index]) or not np.array_equal(raw,np.asarray(e['action'],np.float32)):raise CriticalEvidenceError('Worker action differs')
            if e['queue_after']!=14-index:raise CriticalEvidenceError('Worker actual queue depth')
            if not e['new_chunk'] and e['queue_before']!=15-index:raise CriticalEvidenceError('Pre-dequeue depth')
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
                last_substeps=substeps,last_chunk_index=index,full_chunks=sequence//15)

