"""Read-only hashes of reused data/models/logs against the historical seal."""
import argparse
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--old',required=True);a=p.parse_args()
r,old=Path(a.root),Path(a.old)
manifest=json.loads((old/'manifest.json').read_text())
seal=json.loads((old/'artifact_sha256.json').read_text())
files=[Path(x['episode']) for x in manifest]
files += [old/'manifest.json',old/'training/train.jsonl',old/'budget.jsonl',old/'training/checkpoints/step_000000.pt',old/'training/checkpoints/step_005000.pt']
result=[]
for path in files:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    relative=str(path.relative_to(old));expected=seal.get(relative)
    result.append(dict(old_relative_path=relative,bytes=path.stat().st_size,sha256=h.hexdigest(),
                       historical_sha256=expected['sha256'] if expected else None,
                       matches_historical=expected['sha256']==h.hexdigest() if expected else None))
payload=dict(old_run=str(old),files=result,all_historically_sealed_files_match=all(x['matches_historical'] for x in result if x['historical_sha256']),
             all_files_historically_sealed=all(x['historical_sha256'] is not None for x in result))
(r/'reused_data_provenance.json').write_text(json.dumps(payload,indent=2))
print(json.dumps(dict(checked_files=len(result),all_historically_sealed_files_match=payload['all_historically_sealed_files_match'],all_files_historically_sealed=payload['all_files_historically_sealed'])),flush=True)
assert payload['all_historically_sealed_files_match']
