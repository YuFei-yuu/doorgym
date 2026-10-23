"""Read-only stage results paired with original fixed50k/h25; never selects model."""
import json,math
from pathlib import Path
from r15.paired import record_diagnostics

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def summarize(run,step):
 base=Path(run)/'evaluation'/f'{step:06d}';prior={r['seed']:r for r in read(Path(run)/'baseline-050000.json')['records']};rows=[];counts={};before={}
 for d in sorted(base.glob('trial_*')):
  if not d.is_dir():continue
  r=read(d/'paired-diagnostics.json');old=prior[r['seed']]['gates']
  r.update(baseline_050000_gates=old,improved_items=[k for k,v in r['gates'].items() if v and not old[k]],regressed_items=[k for k,v in r['gates'].items() if not v and old[k]])
  for k,v in r['gates'].items():counts[k]=counts.get(k,0)+int(v);before[k]=before.get(k,0)+int(old[k])
  rows.append(r)
 assert len(rows)==16
 write(base/'paired-summary.json',dict(checkpoint_step=step,execution_horizon=25,denominator=16,counts=counts,baseline_050000_counts=before,automatic_candidates=counts['automatic_candidates'],reviewed_successes=0 if not counts['automatic_candidates'] else None,records=rows,independent_generalization=False,no_new_50000_evaluation=True,visual_and_physical_review_separate=True))
