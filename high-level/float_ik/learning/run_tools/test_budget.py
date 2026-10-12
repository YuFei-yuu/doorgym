"""Exercise actual process timeout, accounting, recovery, and mutual exclusion."""
import fcntl
import json
from pathlib import Path
import sys
import tempfile
from bounded import run, used, reconcile, atomic_json

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    assert run(root, 'normal', [sys.executable,'-c','pass'], 4, total=20, grace=1) == 0
    before = used(root)
    assert run(root, 'timeout', [sys.executable,'-c','import time; time.sleep(20)'], 4, total=20, grace=1) == 124
    assert 2.8 < used(root)-before < 4.5
    with (root/'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(root, 'locked', [sys.executable,'-c','pass'], 4, total=20, grace=1)
            raise AssertionError('Concurrent process accepted')
        except BlockingIOError:
            pass
    try:
        run(root, 'reserve', [sys.executable,'-c','pass'], 4, total=used(root)+3, reserve=2, grace=1)
        raise AssertionError('Reserve exceeded')
    except RuntimeError:
        pass
    atomic_json(root/'active_command.json', dict(command_id='dead',pid=99999999,hard_timeout_sec=2))
    reconcile(root)
    assert json.loads((root/'budget.jsonl').read_text().splitlines()[-1])['reconciled']
    count = len((root/'budget.jsonl').read_text().splitlines())
    reconcile(root)
    assert len((root/'budget.jsonl').read_text().splitlines()) == count
print(json.dumps(dict(passed=True,checks=['normal_exit','TERM_timeout_accounting','lock','reserve','dead_process_reconciliation','no_double_accounting'])))
