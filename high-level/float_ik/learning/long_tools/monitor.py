"""Read-only compact experiment progress; does not import any GPU libraries."""
import argparse
import json
from pathlib import Path
import time

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--compact',action='store_true');a=p.parse_args();r=Path(a.root)
def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default
status=read(r/'status.json',{})
active=read(r/'active_command.json',{})
ledger=[]
if (r/'budget.jsonl').exists():
    for line in (r/'budget.jsonl').read_text().splitlines():
        try:ledger.append(json.loads(line))
        except json.JSONDecodeError:pass
result=dict(phase=status.get('phase'),reason=status.get('reason'),
            accounted_hours=sum(x['elapsed_sec'] for x in ledger)/3600,
            active_stage=active.get('stage'),active_elapsed_sec=time.time()-active['started'] if active else None,
            active_command=active.get('command',[''])[-1][-240:],
            checkpoints=sorted(x.name for x in (r/'training/checkpoints').glob('*.pt')),
            validation_points=[read(x) for x in sorted(r.glob('validation_point_*.json'))],
            selection=read(r/'selection.json'),groups={})
for category in ('diagnostic','validation','test'):
    groups={}
    for path in sorted((r/category).glob('*/summary.json')):
        item=read(path)
        if category=='test':key=path.parent.name.split('_seed_')[0]
        elif category=='validation':key=str(item.get('checkpoint_step'))
        else:key='%s_d%s'%(item['mode'],item.get('inference_steps'))
        group=groups.setdefault(key,dict(started=0,completed=0,successes=0,errors=0))
        group['started']+=1;group['completed']+=item['status']=='completed';group['successes']+=bool(item.get('success'));group['errors']+=item['status'] not in ('started','completed')
    result['groups'][category]=groups
if a.compact:
    result=dict(phase=result['phase'],reason=result['reason'],accounted_hours=round(result['accounted_hours'],3),
                active_elapsed_sec=round(result['active_elapsed_sec'] or 0,1),
                validation_points=[dict(step=x['step'],successes=x['successes'],angle=x['mean_final_angle_deg'],loss=x['val_loss']) for x in result['validation_points']],
                selection=result['selection'],
                groups={category:{k:[v['completed'],v['successes'],v['errors']] for k,v in gg.items() if k!='None'} for category,gg in result['groups'].items()},
                delivery=read(r/'delivery_status.json'))
print(json.dumps(result),flush=True)
