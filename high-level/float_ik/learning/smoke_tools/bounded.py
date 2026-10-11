#!/usr/bin/env python3
"""Run one project command with a shared, conservative GPU wall-time ledger."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--stage', required=True)
p.add_argument('--limit', type=float, required=True)
p.add_argument('--timeout', type=float, required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
root = Path(a.root)
root.mkdir(parents=True, exist_ok=True)
lock = (root / 'budget.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
ledger = root / 'budget.jsonl'
rows = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
used = sum(x['elapsed_sec'] for x in rows)
stage_used = sum(x['elapsed_sec'] for x in rows if x['stage'] == a.stage)
seconds = min(a.timeout, a.limit - stage_used, 7200 - used)
if seconds <= 1:
    raise SystemExit('Budget exhausted')
cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
record = dict(stage=a.stage, command=cmd, started=time.time(), timeout_sec=seconds,
              physical_gpu=os.environ.get('VIDEO2DOOR_GPU'))
active = root / 'active_command.json'
if active.exists():
    raise SystemExit('Unreconciled previous command: inspect active_command.json')
active.write_text(json.dumps(record, indent=2))
child = subprocess.Popen(cmd, start_new_session=True)
try:
    code = child.wait(timeout=seconds)
except subprocess.TimeoutExpired:
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    code = 124
record.update(elapsed_sec=time.time() - record['started'], exit_code=code)
with ledger.open('a') as f:
    f.write(json.dumps(record) + '\n')
active.unlink()
print('BUDGET ' + json.dumps(record), flush=True)
raise SystemExit(code)
