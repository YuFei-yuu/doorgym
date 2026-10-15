"""Fast-forward the idle server checkout, preserving its untracked learning tree."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

repo=Path('/data/zyf/video2door')
backup=repo/'.repro/repository-sync/20261012-184254'
run=repo/'experiments/float-push/20261012-005832-controlled'
target='1f0f112ace6a94083273e3ff2dc59d8421542f89'
def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()
locks=[]
for name in ('driver','budget','finalize','resume_finish'):
    lock=(run/(name+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
assert not (run/'active_command.json').exists()
assert not git('status','--porcelain=v1','--untracked-files=no'),'Tracked changes need review'
assert git('rev-parse','origin/main')==target
subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor','HEAD',target],check=True)
processes=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True).splitlines()
for line in processes[1:]:
    parts=line.split(None,2)
    if len(parts)!=3 or not Path(parts[1]).name.startswith('python'):continue
    assert not any(path in parts[2] for path in ('learning/train.py','learning/rollout.py','learning/long_tools/driver.py','learning/long_tools/finalize.py')),line
ledger_before=digest(run/'budget.jsonl')
head_before=git('rev-parse','HEAD')
learning=repo/'high-level/float_ik/learning'
assert not git('ls-files','high-level/float_ik/learning'),'Expected the old server checkout with an untracked adapter'
preserved=backup/'learning.before-fast-forward';assert not preserved.exists()
shutil.move(str(learning),str(preserved))
try:
    result=subprocess.run(['git','-C',str(repo),'merge','--ff-only','origin/main'],text=True,capture_output=True)
    (backup/'fast-forward.log').write_text(result.stdout+result.stderr)
    result.check_returncode()
except BaseException:
    if not learning.exists():shutil.move(str(preserved),str(learning))
    raise
assert git('rev-parse','HEAD')==target
assert not git('status','--porcelain=v1'),'Server checkout has unexpected changes'
assert digest(run/'budget.jsonl')==ledger_before
seal=json.loads((run/'artifact_sha256.json').read_text())
for name,item in seal.items():
    path=run/name;assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256'],name
manifest={}
for directory in (repo/'docs',learning):
    for path in directory.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and not path.name.startswith('._'):
            manifest[str(path.relative_to(repo))]=dict(bytes=path.stat().st_size,sha256=digest(path))
proof=dict(verified_at=time.time(),head_before=head_before,head=target,server_status=git('status','--porcelain=v1'),
    remote_main=git('rev-parse','origin/main'),sealed_experiment_files_unchanged=len(seal),budget_sha256=ledger_before,
    preserved_server_tree=str(preserved),files=manifest)
(backup/'sync_verified.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof))
