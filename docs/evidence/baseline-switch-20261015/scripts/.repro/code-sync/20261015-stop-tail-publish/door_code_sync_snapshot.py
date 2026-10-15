"""Read repository contents; archive dirty files without changing source or Git."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

repo = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=False)

def git(*args):
    return subprocess.check_output(['git', *args], cwd=repo)

names = sorted(set(n for n in git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0') if n))
manifest = {}
for name in names:
    path = repo/name
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        manifest[name] = digest.hexdigest()
(out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
(out/'head.txt').write_bytes(git('rev-parse', 'HEAD'))
(out/'status.txt').write_bytes(git('status', '--short', '--branch'))
(out/'worktree.diff').write_bytes(git('diff', '--binary', 'HEAD'))
(out/'index.diff').write_bytes(git('diff', '--cached', '--binary'))
dirty = sorted(set(n for n in (git('diff', '--name-only', '-z', 'HEAD')+git('ls-files', '-z', '--others', '--exclude-standard')).decode().split('\0') if n))
with tarfile.open(out/'dirty_files.tar.gz', 'w:gz') as archive:
    for name in dirty:
        if (repo/name).exists():
            archive.add(repo/name, arcname=name, recursive=False)
print(json.dumps(dict(files=len(manifest), dirty_files=len(dirty), snapshot=str(out))))
