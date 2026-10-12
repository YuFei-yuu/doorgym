"""CPU-only selection from complete validation after a resource-limited stop."""
import argparse
import json
from pathlib import Path
import time
from bounded import atomic_json
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root)
status=json.loads((r/'status.json').read_text())
assert status['phase']=='stopped' and 'No eligible GPU' in status['reason']
assert not (r/'active_command.json').exists() and not list((r/'test').glob('*/summary.json'))
points=[json.loads(x.read_text()) for x in r.glob('validation_point_*.json')]
assert points and all(x['attempts']==10 and x['seeds']==list(range(4000,4010)) for x in points)
best=max(points,key=lambda x:(x['successes'],x['mean_final_angle_deg'],-x['val_loss'],-x['step']))
last=max(int(x.stem.split('_')[1]) for x in (r/'training/checkpoints').glob('step_*.pt'))
selected=dict(best_step=best['step'],last_step=last,rule='success_then_final_angle_then_loss_then_earlier',
              validation_seeds=list(range(4000,4010)),test_seeds=list(range(5000,5030)),
              stop_reason=status['reason'],frozen_at=time.time(),best_equals_last=best['step']==last,
              status='validation_only_resource_stopped',independent_tests_started=0,
              last_has_complete_closed_loop_validation=any(x['step']==last for x in points))
assert not (r/'selection.json').exists()
atomic_json(r/'selection.json',selected)
methods=[('expert','expert',None),('untrained','policy',0),('best','policy',best['step'])]
if best['step']!=last:methods.append(('last','policy',last))
atomic_json(r/'test_plan.json',dict(attempts=[dict(method=name,mode=mode,step=step,seed=seed,status='not_started_resource_stop') for seed in range(5000,5030) for name,mode,step in methods],alias={'last':'best'} if best['step']==last else {},status='not_started_resource_stop'))
print(json.dumps(selected),flush=True)
