"""Fixed16 paired with R20-75k; at100k add R18-uniform100k, later branch trend only."""
import json
from pathlib import Path
from r15.paired import record_diagnostics
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def summarize(run,step):
 run=Path(run);base=run/'evaluation'/f'{step:06d}';controls={'R20_075000':read(run/'baseline-r20-075000.json')['records']}
 if step==100000:controls['R18_uniform100000']=read(run/'baseline-r18-100000.json')['records']
 if step in [150000,200000]:controls[f'R21_{100000 if step==150000 else 150000}']=read(run/'evaluation'/f'{100000 if step==150000 else 150000:06d}'/'paired-summary.json')['records']
 maps={k:{r['seed']:r for r in rows} for k,rows in controls.items()};rows=[];counts={};control_counts={k:{} for k in controls}
 for d in sorted(base.glob('trial_*')):
  if not d.is_dir():continue
  r=read(d/'paired-diagnostics.json');r['paired_controls']={}
  for label,prior in maps.items():
   old=prior[r['seed']]['gates'];r['paired_controls'][label]=dict(gates=old,improved=[k for k,v in r['gates'].items() if v and not old[k]],regressed=[k for k,v in r['gates'].items() if not v and old[k]])
   for k,v in old.items():control_counts[label][k]=control_counts[label].get(k,0)+int(v)
  for k,v in r['gates'].items():counts[k]=counts.get(k,0)+int(v)
  rows.append(r)
 assert len(rows)==16
 write(base/'paired-summary.json',dict(checkpoint_step=step,execution_horizon=25,denominator=16,counts=counts,control_counts=control_counts,automatic_candidates=counts['automatic_candidates'],reviewed_successes=0 if not counts['automatic_candidates'] else None,records=rows,independent_generalization=False,controls_rerun=0,equal_update_uniform_control_available=(step==100000),later_stages_branch_trend_only=step!=100000,visual_and_physical_review_separate=True))
