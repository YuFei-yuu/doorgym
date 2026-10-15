"""Stop this already-running frozen supervisor after the 3000-update suite.

The loaded supervisor has no live control interface. Reserve its later launch
stages using the launcher's existing pre-launch guard; never signal a trainer.
The watcher records the requested terminal state only after the guarded
supervisor exits and checkpoint plus all nine evaluation attempts are saved.
"""
import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def process(pid):
    proc=Path('/proc')/str(pid)
    try:
        return dict(start_ticks=proc.joinpath('stat').read_text().split()[21],
                    command=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:
        return None


def arm(root):
    with (root/'stop_arm.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        control=root/'stop_after_updates.json'
        if control.exists():
            assert read(control)['stop_after_updates']==3000
            return
        active=read(root/'active_command.json')
        assert active['stage']=='training__003000',active
        child=process(active['pid']);supervisor=process(active['supervisor_pid'])
        assert child and child['start_ticks']==active['proc_start_ticks']
        assert '--steps 3000' in child['command'] and 'train_door_dp.py' in child['command']
        assert supervisor and 'forward_pipeline.py' in supervisor['command'] and str(root) in supervisor['command'] or (
            supervisor and 'forward_pipeline.py' in supervisor['command'] and root.name in supervisor['command'])
        assert not list((root/'training/checkpoints').glob('step_004*.pt'))
        assert not list((root/'training/checkpoints').glob('step_005*.pt'))
        stages=['training__004000','training__005000','offline__5000']
        for stage in stages:
            assert not (root/'resources'/(stage+'.json')).exists(),stage
        value=dict(stop_after_updates=3000,after_checkpoint_validation=True,evaluation_attempts=9,
                   authorization='User: 记录完3000次更新后终止本次训练',requested_at=time.time(),
                   supervisor_pid=active['supervisor_pid'],supervisor_start_ticks=supervisor['start_ticks'],
                   active_training_pid=active['pid'],active_training_start_ticks=child['start_ticks'],
                   blocked_stages=stages,mechanism='Existing stage-file pre-launch refusal; no process signals')
        write(control,value)
        for stage in stages:
            with (root/'resources'/(stage+'.json')).open('x') as stream:
                json.dump(dict(stage=stage,status='blocked_by_user',not_attempted=True,
                               stop_after_updates=3000,control=str(control),at=time.time()),stream,indent=2)
        print(json.dumps(dict(armed=True,stop_after_updates=3000,blocked_stages=stages)),flush=True)


def watch(root):
    with (root/'stop_watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        control=read(root/'stop_after_updates.json')
        while True:
            state=read(root/'pipeline_state.json')
            if state.get('phase')=='stopped_by_user':return
            supervisor=process(control['supervisor_pid'])
            if supervisor and supervisor['start_ticks']==control['supervisor_start_ticks']:
                time.sleep(15);continue
            break
        expected=str(root/'resources/training__004000.json')
        assert state.get('phase')=='error' and expected in state.get('error',''),state
        assert 'FileExistsError' in state['error'],state
        assert not (root/'active_command.json').exists()
        result=read(root/'evaluation/step_003000/aggregate.json')
        assert result['attempts']==result['scheduled']==9 and result['all_scheduled_completed'],result
        assert read(root/'checkpoint_checks_003000.json')['passed']
        checkpoint=root/'training/checkpoints/step_003000.pt'
        metadata=read(checkpoint.with_suffix('.json'))
        h=hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        assert metadata['completed_updates']==3000 and metadata['sha256']==h.hexdigest()
        history=root/'user_stop_3000';history.mkdir(exist_ok=True)
        with (history/'supervisor_terminal_state.json').open('x') as stream:json.dump(state,stream,indent=2)
        terminal=dict(phase='stopped_by_user',completed_updates=3000,checkpoint=str(checkpoint),
                      checkpoint_sha256=h.hexdigest(),objective_complete=False,
                      stop_requested=True,evaluation=result,at=time.time(),pid=os.getpid(),
                      next_action='Training ended at user-requested 3000 updates; no automatic restart')
        write(root/'pipeline_state.json',terminal)
        write(history/'completion.json',terminal)
        with (root/'supervisor.log').open('a') as stream:
            stream.write('\nUSER_REQUESTED_STOP '+json.dumps(terminal)+'\n')
        print(json.dumps(terminal),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',action='store_true')
    a=p.parse_args();root=Path(a.root).resolve()
    try:
        if a.arm:arm(root)
        else:watch(root)
    except BaseException as exc:
        write(root/'stop_control_error.json',dict(error=repr(exc),at=time.time()))
        raise
