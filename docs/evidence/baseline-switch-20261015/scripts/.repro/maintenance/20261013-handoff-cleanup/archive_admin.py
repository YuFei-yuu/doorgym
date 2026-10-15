"""Put superseded run orchestration records into one history archive."""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

server = '--server' in sys.argv
run = Path('/data/zyf/video2door/experiments/float-push/20261013-upstream-float-v3' if server else
           '/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-v3')
history = run/'history'
history.mkdir(exist_ok=True)
names = ['budget.jsonl','budget.lock','evaluation-resume.lock','evaluation_resume.json',
         'pipeline-driver.log','pipeline-resume.log','collection-driver.log','pipeline_state.json.before_eval_resume',
         'failure_evidence.json','active_command.retained_training_snapshot.json','prepare_eval_resume.py',
         'verify_delivery.py','live_curves.png']
paths = [run/name for name in names]
paths += list(run.glob('budget_authorization*.json')) + list(run.glob('budget_limits*'))
paths = sorted({p for p in paths if p.is_file()})
destination = history/'administrative.tar.gz'
assert not destination.exists()
with tarfile.open(destination,'w:gz') as archive:
    for path in paths:archive.add(path,arcname=path.name)
with tarfile.open(destination,'r:gz') as archive:
    assert set(archive.getnames())=={p.name for p in paths}
    for path in paths:
        assert archive.extractfile(path.name).read()==path.read_bytes()
for path in paths:path.unlink()
(history/'README.txt').write_text('Historical orchestration and runtime records; current state is in ../pipeline_state.json.\n')
hashes = {str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in run.rglob('*') if p.is_file() and p.name!='retained_sha256.json'}
(run/'retained_sha256.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps(dict(archived=[p.name for p in paths],retained_files=len(hashes))))
