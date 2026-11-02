from common import *
import shutil,time,csv
verify();a=read(RUN/'stage-audit-120000.json');assert a['passed'] and len(a['trials'])==16
s=read(RUN/'evaluation/120000/summary.json');old=read(RUN/'baseline-h25-summary.json');assert [r['seed'] for r in s['results']]==[r['seed'] for r in old['results']]
rows=[]
for before,after in zip(old['results'],s['results']):rows.append(dict(seed=after['seed'],h25_success=before['success'],h15_success=after['success'],h25_max_open_deg=before['max_open_deg'],h15_max_open_deg=after['max_open_deg'],h25_events=before['events'],h15_events=after['events']))
write(RUN/'paired-comparison.json',dict(checkpoint_step=120000,prediction=100,h25_successes=old['successes'],h15_successes=s['successes'],denominator=16,rows=rows,training_updates=0,causal_scope='Same checkpoint and fixed actual initial states; execution horizon25 vs15; finite16 paired episodes'))
with (RUN/'paired-comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['seed','h25_success','h15_success','h25_max_open_deg','h15_max_open_deg']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
write(RUN/'complete.json',dict(official_trials=16,optimizer_updates=0,horizon=15,stage=120000,successes=s['successes'],denominator=16,audit_passed=True,at=time.time()))
(RUN/'RESULT.txt').write_text(f"120k checkpoint独立horizon15评测：{s['successes']}/16；原horizon25为{old['successes']}/16。预测100/执行15，25Hz，每回合120s。完整固定16及审核通过，零训练更新；详见paired-comparison.json，事件单列。原冻结成果未改。\n")
final=ROOT/'.repro/reviews/wc4-act100-120k-h15-eval16-20261102-FINAL';final.mkdir(exist_ok=False)
def links(src,dst):
 dst.mkdir(exist_ok=True)
 for p in src.iterdir():
  if p.is_symlink():continue
  q=dst/p.name
  if p.is_dir():links(p,q)
  else:os.link(p,q)
links(RUN,final/'evidence');shutil.copytree(Path(__file__).parent,final/'code',ignore=shutil.ignore_patterns('__pycache__'))
write(final/'INDEX.json',dict(run=str(RUN),checkpoint_source=read(RUN/'checkpoint-source.json'),full_checkpoint_remains_in_parent_freeze=True,official_trials=16,optimizer_updates=0,source_commit=read(RUN/'manifest.json')['start_commit']))
write(final/'SHA256.json',[dict(path=str(p.relative_to(final)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(final.rglob('*')) if p.is_file()])
for p in final.rglob('*'):
 if p.is_file():p.chmod(0o444)
for p in sorted([p for p in final.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
final.chmod(0o555);print(json.dumps(dict(final=str(final),successes=s['successes'],baseline=old['successes'],manifest_sha256=sha(final/'SHA256.json'))))
