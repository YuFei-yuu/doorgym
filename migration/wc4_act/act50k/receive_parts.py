"""Verify independent transport chunks and exact completed-stage files."""
from pathlib import Path
import sys,json,hashlib,tarfile
ROOT=Path('/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act50k-20261029-r2')
DEST=ROOT/'INCOMING/evidence';stage=int(sys.argv[1]);tag=f'{stage:06d}'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
if stage!=10000:
 manifest=json.loads((ROOT/'transfers'/f'{tag}-manifest.json').read_text())
 for row in manifest['archives']:
  part=row['part'];archive=ROOT/'transfers'/f'{tag}-part-{part:02d}.tar.gz'
  assert archive.stat().st_size==row['bytes'] and sha(archive)==row['sha256'],archive
  with tarfile.open(archive) as t:t.extractall(DEST)
  archive.unlink();print('extracted',stage,part,flush=True)
proof=json.loads((DEST/f'stage-audit-{tag}.json').read_text());assert proof['passed']
for name,row in proof['files'].items():
 p=DEST/name
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
(ROOT/'transfers'/f'{tag}-verified.json').write_text(json.dumps(dict(stage=stage,all_source_files_verified=len(proof['files']),all_bytes_equal=True,transport_archives_removed=True),indent=2)+'\n')
print(json.dumps(dict(stage=stage,verified_files=len(proof['files']),all_bytes_equal=True)),flush=True)
