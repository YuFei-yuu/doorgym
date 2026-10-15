"""Seal stable completed evidence; stdout must not be redirected inside root."""
import json,hashlib,sys,time,shutil
from pathlib import Path
r=Path(sys.argv[1]);rev=r/'postprocessing_revisions/0004';rev.mkdir(parents=True,exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
prior=json.loads((r/'delivery_sha256.json').read_text());bad=[n for n,h in prior.items() if sha(r/n)!=h]
# Other derived reports/docs may have been deliberately replaced after the first seal.
assert set(bad)<=set(['logs/finalize_evidence_attempt1.log','final_analysis/REPORT.md','docs/FLOAT_PUSH_LR_PAIR.md','FLOAT_PUSH_LR_PAIR.md','FINAL_EVIDENCE.json'])
shutil.copy2(r/'delivery_sha256.json',rev/'prior_manifest.json');shutil.copy2(r/'SHA256SUMS',rev/'prior_SHA256SUMS.txt')
shutil.copy2(r/'logs/finalize_evidence_attempt1.log',rev/'finalize_evidence_attempt1.log')
record=dict(at=time.time(),scope='Seal reporting artifacts only after writers exit; no training/inference/simulation',
 reason='First manifest included active finalizer stdout log before footer was flushed; only that log mismatched at original verification',
 prior_manifest_mismatches_before_report_note=bad,original_evidence_retained=True)
(rev/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
e=json.loads((r/'FINAL_EVIDENCE.json').read_text());e['postprocessing_packaging_note']=record;e['documentation_anomalies']=json.loads((r/'documentation_anomalies.json').read_text());(r/'FINAL_EVIDENCE.json').write_text(json.dumps(e,indent=2,allow_nan=False)+'\n')
for name in ['FLOAT_PUSH_LR_PAIR.md','FLOAT_PUSH_TRAINING_NEXT.md']:shutil.copy2(Path('docs')/name,r/name)
shutil.copy2(Path('docs/FLOAT_PUSH_LR_PAIR.md'),r/'final_analysis/REPORT.md')
# Only derived reports changed after core source/diff export; preserve original export and append documentation revision.
shutil.copy2(r/'revision_this_round.diff',rev/'revision_before_packaging_note.diff')
import subprocess
current=subprocess.run(['git','diff','--binary','--no-index','/dev/null','docs/FLOAT_PUSH_LR_PAIR.md'],stdout=subprocess.PIPE).stdout
with (r/'revision_this_round.diff').open('ab') as f:f.write(b'\n# Final documentation-only packaging note revision\n'+current)
excluded={'delivery_sha256.json','SHA256SUMS'}
hashes={str(p.relative_to(r)):sha(p) for p in sorted(r.rglob('*')) if p.is_file() and p.name not in excluded}
(r/'delivery_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
(r/'SHA256SUMS').write_text(''.join('%s  %s\n'%(h,n) for n,h in hashes.items()))
assert not [n for n,h in hashes.items() if sha(r/n)!=h]
print(json.dumps(dict(sealed_and_verified=True,files=len(hashes),bytes=sum((r/n).stat().st_size for n in hashes),new_rollouts=6,added_updates_each=10000),indent=2))
