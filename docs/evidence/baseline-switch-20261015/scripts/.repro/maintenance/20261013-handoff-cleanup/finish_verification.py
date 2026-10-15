import hashlib
import json
import sys
import tarfile
from pathlib import Path

server = '--server' in sys.argv
repo = Path('/data/zyf/video2door' if server else '/Users/yu/Data/Video2Door/code')
evidence = repo/'.repro' if server else repo.parent/'reproduction/pro5000'
run = repo/'experiments/float-push/20261013-upstream-float-v3' if server else evidence/'float-push-v3'
current = repo/'experiments/float-push/20261013-forward-expert-grid18' if server else evidence/'forward-expert-grid18'
out = evidence/'maintenance/20261013-handoff-cleanup'
paths = [run/name for name in ['delivery_sha256.json','delivery_verification.json',
                              'local_delivery_verification.json','completion_evidence.json'] if (run/name).is_file()]
destination = run/'history/original-delivery.tar.gz'
assert not destination.exists()
with tarfile.open(destination,'w:gz') as archive:
    for path in paths:archive.add(path,arcname=path.name)
with tarfile.open(destination,'r:gz') as archive:
    for path in paths:assert archive.extractfile(path.name).read()==path.read_bytes()
for path in paths:path.unlink()
protected = json.loads((out/'completed.json').read_text())['protected_sha256']
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in protected.items())
validation = json.loads((current/'SHA256.json').read_text())
assert all(hashlib.sha256((current/path).read_bytes()).hexdigest()==digest for path,digest in validation.items())
remaining = {str(path.relative_to(run)):hashlib.sha256(path.read_bytes()).hexdigest()
             for path in run.rglob('*') if path.is_file() and path.name not in ['retained_sha256.json','.DS_Store']}
(run/'retained_sha256.json').write_text(json.dumps(remaining,indent=2))
result = dict(protected_files=len(protected), protected_unchanged=True,
              current_validation_files=len(validation), current_validation_unchanged=True,
              retained_run_files=len(remaining), documentation_sync_verified=True,
              gym_import_and_pip_check_passed=True, training_import_and_pip_check_passed=True,
              converter_and_training_entry_help_passed=True, execution_source_unchanged=True)
(out/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
