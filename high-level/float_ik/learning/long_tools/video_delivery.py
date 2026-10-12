"""At most two illustrative reruns, after frozen independent tests only.

Repeated test seeds are video reproductions, not new independent samples. All
started reproductions, including mismatches/errors, have their own denominator.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import time
from bounded import atomic_json, run

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root).resolve()
learning=Path(__file__).resolve().parents[1];repo=learning.parents[2]
with (r/'driver.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected=json.loads((r/'selection.json').read_text())
    assert json.loads((r/'status.json').read_text())['phase']=='gpu_work_complete'
    methods=[('best',selected['best_step'])]
    if not selected['best_equals_last']:methods.append(('last',selected['last_step']))
    plan=[]
    for role,step in methods:
        attempts=[(path,json.loads(path.read_text())) for path in sorted((r/'test').glob(role+'_seed_*/summary.json'))]
        assert len(attempts)==30 and all(s['status']=='completed' for _,s in attempts)
        for outcome in (True,False):
            eligible=[(path,s) for path,s in attempts if s['success']==outcome]
            if not eligible or any((path.parent/'rollout.mp4').exists() for path,_ in eligible):continue
            path,s=eligible[0]
            plan.append(dict(role=role,step=step,seed=s['seed'],intended_success=outcome,
                             original_test=str(path.relative_to(r)),
                             output='videos/%s_%s_seed_%d'%(role,'success' if outcome else 'failure',s['seed'])))
    assert len(plan)<=2
    atomic_json(r/'video_reproduction_plan.json',dict(attempts=plan,meaning='Repeated frozen test seeds for illustration; not additional independent test samples'))
    if plan:
        samples=[]
        for i in range(3):
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            samples.append([[v.strip() for v in line.split(',')] for line in raw.strip().splitlines()])
            if i<2:time.sleep(2)
        choices=[]
        for index,item in enumerate(samples[0]):
            load=statistics.mean(float(s[index][3]) for s in samples)
            if min(float(s[index][2]) for s in samples)>=16384 and load<70:choices.append((load,int(item[0]),item[1]))
        atomic_json(r/'resource_video_reproduction.json',dict(samples=samples,candidates=choices))
        if not choices:raise SystemExit('No eligible GPU for supplemental video; preserve existing final-test videos')
        _,gpu,uuid=min(choices)
        for i,item in enumerate(plan):
            output=r/item['output']
            if (output/'summary.json').exists():continue
            command=['python',str(learning/'rollout.py'),'--schema','float_push_joint_v1','--mode','policy','--checkpoint',str(r/'training/checkpoints'/('step_%06d.pt'%item['step'])),
                     '--seed',str(item['seed']),'--output',str(output),'--video','--diagnostic-steps','60','--split','video_reproduction']
            output.parent.mkdir(exist_ok=True)
            shell='export VIDEO2DOOR_GPU=%d; source %s || exit; exec nice -n 10 %s'%(gpu,shlex.quote(str(repo/'.repro/blackwell/activate.sh')),shlex.join(command))
            rc=run(r,'video_reproduction',['bash','-c',shell],timeout=240,total=21600,
                   gpu=gpu,gpu_uuid=uuid,log_name='video_reproduction_%s_%d.log'%(item['role'],item['seed']))
            summary=output/'summary.json'
            if not summary.exists():
                output.mkdir(exist_ok=True)
                atomic_json(summary,dict(status='launch_error',success=False,seed=item['seed'],mode='policy'))
            value=json.loads(summary.read_text())
            value.update(original_test=item['original_test'],intended_success=item['intended_success'],purpose='video_reproduction_not_independent_test')
            if rc or value['status']=='started':value.update(status='timeout' if rc==124 else 'error',success=False)
            atomic_json(summary,value)
            if rc:break
print('VIDEO_REPRODUCTIONS_RECORDED',len(plan),flush=True)
