"""Sequential process accounting with an explicit wall-time cap and exit reserve."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def read_rows(root):
    path = Path(root)/'budget.jsonl'
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def used(root):
    return sum(x['elapsed_sec'] for x in read_rows(root))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def reconcile(root):
    """Only account a dead prior command; never signal an unidentified process."""
    root = Path(root)
    active = root/'active_command.json'
    if not active.exists():
        return
    row = json.loads(active.read_text())
    if any(x['command_id'] == row['command_id'] for x in read_rows(root)):
        active.unlink()
        return
    if row.get('pid') and Path('/proc/%s' % row['pid']).exists():
        raise RuntimeError('Prior process may still be alive; inspect active_command.json')
    # A lost supervisor cannot prove the exact death time. Charge the entire
    # assigned hard timeout, avoiding silent under-accounting after a disconnect.
    hard = row.get('hard_timeout_sec')
    row.update(elapsed_sec=hard if hard is not None else max(0, time.time()-row['started']),
               exit_code=None, reconciled=True,
               accounting=('conservative full assigned timeout after supervisor loss' if hard is not None
                           else 'upper bound from start to reconciliation; exact runtime unknown'))
    with (root/'budget.jsonl').open('a') as log:
        log.write(json.dumps(row)+'\n')
        log.flush()
        os.fsync(log.fileno())
    active.unlink()


def run(root, stage, command, timeout, total=21600, reserve=0, gpu=None,
        gpu_uuid=None, log_name=None, grace=30):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reconcile(root)
        hard = float(timeout) if timeout is not None else None
        if total is not None:
            available = total-used(root)-reserve
            hard = min(hard, available) if hard is not None else available
        limits_path = root/'budget_limits.json'
        if limits_path.exists():
            limits = json.loads(limits_path.read_text())
            category = stage.split('__', 1)[0]
            if category not in limits:
                raise ValueError('Unbudgeted stage category: '+category)
            consumed = sum(r['elapsed_sec'] for r in read_rows(root)
                           if r['stage'].split('__', 1)[0] == category)
            available = float(limits[category])-consumed
            hard = min(hard, available) if hard is not None else available
        if hard is not None and hard <= grace+2:
            raise RuntimeError('Insufficient budget, including shutdown reserve')
        row = dict(command_id='%d-%d' % (time.time_ns(), os.getpid()), stage=stage,
                   command=list(map(str, command)), started=time.time(),
                   hard_timeout_sec=hard, term_after_sec=hard-grace if hard is not None else None,
                   physical_gpu=gpu, gpu_uuid=gpu_uuid, supervisor_pid=os.getpid())
        active = root/'active_command.json'
        atomic_json(active, row)
        output = root/'logs'/(log_name or (row['command_id']+'.log'))
        output.parent.mkdir(exist_ok=True)
        started = time.monotonic()
        child = None
        code = None
        interrupted = None
        try:
            with output.open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                row['pid'] = child.pid
                row['proc_start_ticks'] = Path('/proc/%s/stat' % child.pid).read_text().split()[21]
                atomic_json(active, row)
                try:
                    code = child.wait(timeout=(max(.01, hard-grace-(time.monotonic()-started))
                                               if hard is not None else None))
                except subprocess.TimeoutExpired:
                    row['timed_out'] = True
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=max(.01, hard-(time.monotonic()-started)-1))
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    code = 124
        except BaseException as exc:
            interrupted = exc
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=(max(.01, hard-(time.monotonic()-started)-1)
                                        if hard is not None else grace))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            row['supervisor_error'] = repr(exc)
        finally:
            row.update(elapsed_sec=time.monotonic()-started, ended=time.time(), exit_code=code)
            with (root/'budget.jsonl').open('a') as ledger:
                ledger.write(json.dumps(row)+'\n')
                ledger.flush()
                os.fsync(ledger.fileno())
            active.unlink()
        if interrupted is not None:
            raise interrupted
        print('BUDGET', json.dumps({k: row[k] for k in ('stage','elapsed_sec','exit_code','physical_gpu')}), flush=True)
        return code
