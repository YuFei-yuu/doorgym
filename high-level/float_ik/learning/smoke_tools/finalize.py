#!/usr/bin/env python3
"""CPU-only final checks, charts, provenance and artifact inventory."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

r=Path(sys.argv[1]).resolve()
repo=Path('/data/zyf/video2door')
assert not (r/'active_command.json').exists(),'GPU command still active'
stamp=str(time.time_ns())
for label,script in [('checks','high-level/float_ik/learning/checks.py'),('analysis','high-level/float_ik/learning/analyze.py'),('provenance',str(r/'tools/provenance.py'))]:
    args=[sys.executable,script]+([str(r)] if label=='provenance' else ['--root',str(r)])
    with (r/'logs'/('final-'+label+'-'+stamp+'.log')).open('x') as log:
        subprocess.run(args,cwd=str(repo),stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',label,flush=True)
inventory={}
for p in sorted(r.rglob('*')):
    if not p.is_file() or p.name.startswith('._') or p.name=='artifact_sha256.json':continue
    digest=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
    inventory[str(p.relative_to(r))]=dict(bytes=p.stat().st_size,sha256=digest.hexdigest())
(r/'artifact_sha256.json').write_text(json.dumps(inventory,indent=2))
print('INVENTORY',len(inventory),'files',flush=True)
