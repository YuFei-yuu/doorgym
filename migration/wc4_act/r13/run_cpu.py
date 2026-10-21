"""Record commands, exit codes and raw CPU regression logs; never run ACT/Gym."""
import subprocess,os,sys,time
from runtime import *
run=ROOT/'.repro/runs/wc4-engineering-r13-20261022'
number=len(list(run.glob('regression_*')))+1
attempt=run/('regression_%03d'%number);attempt.mkdir()
commands=[
 ('before',[sys.executable,'-B',str(SUPPORT/'r13/test_before.py'),str(attempt/'before')]),
 ('ipc-act',[sys.executable,'-B',str(SUPPORT/'r13/test_ipc.py'),str(attempt/'ipc-act')]),
 ('ipc-gym',['bash','-c','source migration/wc4_act/activate.sh gym >/dev/null && export CUDA_VISIBLE_DEVICES="" && exec python -B "$@"','r13',str(SUPPORT/'r13/test_ipc.py'),str(attempt/'ipc-gym')]),
 ('stopping',[sys.executable,'-B',str(SUPPORT/'r13/test_stopping.py'),str(attempt/'stopping')]),
 ('contacts',[sys.executable,'-B',str(SUPPORT/'r13/test_contacts.py'),str(attempt/'contacts')]),
 ('link',[sys.executable,'-B',str(SUPPORT/'r13/test_link.py'),str(attempt/'link')]),
 ('r11-regression',[sys.executable,'-B',str(SUPPORT/'r11/test_criteria.py')]),
]
records=[]
for name,argv in commands:
 started=time.time()
 with (attempt/(name+'.log')).open('x') as log:
  code=subprocess.run(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',VIDEO2DOOR_GPU='0',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT).returncode
 record=dict(name=name,argv=argv,returncode=code,started=started,ended=time.time());records.append(record)
 write(attempt/(name+'-command-exit.json'),record)
 print(name,code,flush=True)
write(attempt/'result.json',dict(passed=all(r['returncode']==0 for r in records),commands=records,
    formal_ACT_training=0,real_policy_inference=0,simulation=0))
if any(r['returncode']!=0 for r in records):raise SystemExit(1)
