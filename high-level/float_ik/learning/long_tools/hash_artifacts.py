"""Seal hashes only after both the GPU driver and CPU finalizer have exited."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time
from bounded import atomic_json
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root)
locks=[]
for name in ('driver','finalize','budget','resume_finish'):
    lock=(r/(name+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(lock)
assert not (r/'active_command.json').exists()
result={}
for path in sorted(r.rglob('*')):
    if not path.is_file() or path.name=='artifact_sha256.json' or path.suffix in ('.lock','.tmp'):continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    result[str(path.relative_to(r))]=dict(bytes=path.stat().st_size,sha256=h.hexdigest())
atomic_json(r/'artifact_sha256.json',result)
print(json.dumps(dict(sealed_files=len(result),time=time.time())),flush=True)
