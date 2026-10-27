import json,sys
from pathlib import Path
import numpy as np
root=Path('/data/zyf/video2door-wc4-act');run=Path(sys.argv[1]);prior=root/'.repro/runs/wc4-close-inplace-stage5-20261027-retry2/attempt_000/states.jsonl'
fields=['actual_base','action','arm_q','door_q','door_qd'];maxerr={k:0. for k in fields};count=0
with prior.open() as a,(run/'attempt_000/states.jsonl').open() as b:
    for la,lb in zip(a,b):
        x=json.loads(la);y=json.loads(lb)
        if y['step']>=3365:break
        assert x['step']==y['step'];count+=1
        for k in fields:maxerr[k]=max(maxerr[k],float(np.max(abs(np.array(x[k])-np.array(y[k])))))
report=dict(compared_steps=count,through_step=3364,source=str(prior),max_absolute_difference=maxerr,all_exact=all(v==0 for v in maxerr.values()),meaning='Actual prefix before dynamic switch, not a simulated replay')
(run/'review/protected-prefix-comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
