"""Advance Git records only after proving published content matches every file."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

repo = Path('/data/zyf/video2door')
audit = repo/'.repro/code-sync/20261015-stop-tail-publish'
target = sys.argv[1]

def git(*args):
    return subprocess.check_output(['git', *args], cwd=repo)

old = (audit/'server-before/head.txt').read_text().strip()
assert git('symbolic-ref', 'HEAD').decode().strip() == 'refs/heads/main'
assert git('rev-parse', 'HEAD').decode().strip() == old, 'Server HEAD changed after snapshot'
assert git('rev-parse', 'origin/main').decode().strip() == target, 'Remote target mismatch'
subprocess.run(['git', 'merge-base', '--is-ancestor', old, target], cwd=repo, check=True)
assert not git('diff', '--cached', '--name-only'), 'Unexpected staged server changes'

expected = json.loads((audit/'server-before/manifest.json').read_text())
def verify_manifest():
    names = set(n for n in git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0') if n and (repo/n).is_file())
    assert names == set(expected), 'Working file inventory changed'
    changed = [n for n, digest in expected.items() if hashlib.sha256((repo/n).read_bytes()).hexdigest() != digest]
    assert not changed, 'File content changed: '+repr(changed)

verify_manifest()
tree = git('ls-tree', '-r', '-z', target).decode().rstrip('\0').split('\0')
tracked = set()
for record in tree:
    meta, name = record.split('\t', 1)
    mode, kind, object_id = meta.split()
    assert kind == 'blob', 'Non-file tree entry requires separate review: '+name
    tracked.add(name)
    path = repo/name
    data = os.readlink(path).encode() if mode == '120000' else path.read_bytes()
    actual = hashlib.sha1(('blob %d\0' % len(data)).encode()+data).hexdigest()
    assert actual == object_id, 'Published Git blob differs from server file: '+name
assert tracked == set(expected), 'Published tracked files differ from preserved inventory'

index = Path(git('rev-parse', '--git-path', 'index').decode().strip())
if not index.is_absolute():
    index = repo/index
shutil.copy2(index, audit/'index.before-align')
# read-tree without -u updates only the index. update-ref uses compare-and-swap;
# neither command writes, removes, or checks out any working-tree file.
subprocess.run(['git', 'read-tree', target], cwd=repo, check=True)
subprocess.run(['git', 'update-ref', '-m', 'Align published parking-tail code; preserve worktree',
                'refs/heads/main', target, old], cwd=repo, check=True)
verify_manifest()
status = git('status', '--porcelain').decode()
assert not status, 'Unexpected post-alignment status: '+status
result = dict(passed=True, previous_head=old, head=target, files=len(expected),
              working_files_unchanged=True, clean=True, method='verified read-tree without -u and compare-and-swap fast-forward update-ref',
              at=time.time())
(audit/'publish-result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
