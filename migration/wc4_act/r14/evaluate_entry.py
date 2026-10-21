"""R13 evaluator with passive between-trial audits and cooperative stop propagation."""
import os,sys,subprocess,runpy,traceback,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];R13=ROOT/'migration/wc4_act/r13'
sys.path[:0]=[str(Path(__file__).parent),str(R13)]
from audit import audit_trial
from stopping import request_stop
run=Path(os.environ['WC4_EXECUTION_RUN']);step=int(sys.argv[1]);base=run/'evaluation'/f'{step:06d}'
original=subprocess.run

def audit_available():
    for directory in sorted(base.glob('trial_*')):
        if directory.is_dir() and (directory/'r11-score.json').exists() and not (directory/'r14-runtime-audit.json').exists():audit_trial(directory)

def observed(command,*a,**k):
    if isinstance(command,(list,tuple)) and str(R13/'rollout.py') in command:audit_available()
    return original(command,*a,**k)
subprocess.run=observed
try:
    sys.path.insert(0,str(R13))
    runpy.run_path(str(R13/'evaluate.py'),run_name='__main__')
    audit_available()
except BaseException as exc:
    if isinstance(exc,SystemExit) and exc.code in [None,0]:raise
    request_stop(run,'evaluation/export/runtime evidence failed',checkpoint_step=step,error=repr(exc),traceback=traceback.format_exc())
    raise SystemExit(75)
