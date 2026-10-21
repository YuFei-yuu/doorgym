"""Join worker queue receipts to actual 50Hz command/physics acknowledgements."""
import json,hashlib,os
import numpy as np


class ActionJournal:
    def __init__(self,path):
        self.file=open(path,'x',buffering=1)
        self.sequence=0;self.epoch=None;self.chunk=0;self.rows=None;self.index=-1
        self.pending=None;self.commands=0;self.completed=0

    def record(self,event,**data):
        self.file.write(json.dumps(dict(event=event,**data))+'\n')

    def begin(self,receipt,raw,applied,source):
        if self.pending is not None and self.completed!=2:raise RuntimeError('Previous action not physically completed twice')
        if not receipt or receipt['schema']!='door_worker_action_v1':raise RuntimeError('Missing worker evidence')
        if receipt['executed_horizon']!=25:raise RuntimeError('Execution horizon changed')
        if self.epoch is not None and receipt['epoch']!=self.epoch:raise RuntimeError('Unexpected mid-episode reset')
        if receipt['emission_seq']!=self.sequence+1:raise RuntimeError('Skipped/repeated worker emission')
        raw=np.asarray(raw,dtype=np.float32);applied=np.asarray(applied,dtype=np.float32)
        if not np.array_equal(raw,np.asarray(receipt['action'],dtype=np.float32)):raise RuntimeError('Parent/worker action mismatch')
        if receipt['new_chunk']:
            if self.chunk and self.index!=24:raise RuntimeError('Replanned before 25 observed emitted actions')
            if receipt['chunk_id']!=self.chunk+1 or receipt['action_index']!=0 or receipt['queue_before']!=0:raise RuntimeError('Bad sample boundary')
            self.rows=np.asarray(receipt['selected_chunk_actions'],dtype=np.float32)
            if self.rows.shape!=(25,9) or hashlib.sha256(self.rows.tobytes()).hexdigest()!=receipt['chunk_sha256']:raise RuntimeError('Selected chunk evidence corrupt')
        elif receipt['chunk_id']!=self.chunk or receipt['action_index']!=self.index+1:raise RuntimeError('Non-contiguous queue row')
        index=receipt['action_index']
        if not np.array_equal(raw,self.rows[index]) or receipt['queue_after']!=24-index:raise RuntimeError('Queue row/depth inconsistent')
        if not receipt['new_chunk'] and receipt['queue_before']!=25-index:raise RuntimeError('Wrong actual pre-dequeue depth')
        self.sequence+=1;self.epoch=receipt['epoch'];self.chunk=receipt['chunk_id'];self.index=index
        self.pending=applied.copy();self.commands=self.completed=0
        self.record('emitted',receipt=receipt,source=source,policy_output=raw.tolist(),applied_action=applied.tolist(),clipping_delta=(applied-raw).tolist())

    def command_applied(self,step,time_s,actual_target_action):
        if self.pending is None or self.commands!=self.completed or self.commands>=2:raise RuntimeError('Unexpected command application')
        actual=np.asarray(actual_target_action,dtype=np.float32)
        if not np.array_equal(actual,self.pending):raise RuntimeError('Actual target differs from declared applied action')
        self.commands+=1
        self.record('command_applied',emission_seq=self.sequence,substep=self.commands-1,step=step,time_s=time_s,action=actual.tolist())

    def physics_completed(self,step,time_s):
        if self.commands!=self.completed+1:raise RuntimeError('Physics acknowledgement lacks command')
        self.completed+=1
        self.record('physics_completed',emission_seq=self.sequence,substep=self.completed-1,step=step,time_s=time_s)

    def close(self):
        self.record('end',emitted=self.sequence,last_chunk=self.chunk,last_action_index=self.index,
            last_action_completed_substeps=self.completed,partial_final_action=self.pending is not None and self.completed!=2,
            partial_final_chunk=self.chunk!=0 and self.index!=24)
        self.file.flush();os.fsync(self.file.fileno());self.file.close()
