"""A failed monitor requests stop; actual child exits with a committed boundary."""
import os,sys,json,subprocess
from pathlib import Path
from supervision import supervise
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
worker=out/'cooperative_child.py'
worker.write_text('''import sys,time
from pathlib import Path
sys.path.insert(0,'''+repr(str(Path(__file__).parent))+''')
from stopping import StopBoundary,BoundaryStop,commit_checkpoint
r=Path(sys.argv[1])
def save(step):
 def writer(p,s):(p/'fixture-state.txt').write_text(str(s))
 return commit_checkpoint(r/'checkpoints',step,writer)
b=StopBoundary(r,save);b.install_signals()
try:
 while True:b.update(lambda:time.sleep(.02))
except BoundaryStop:sys.exit(75)
finally:b.close()
''')
with (out/'child.log').open('x') as log:
 child=subprocess.Popen([sys.executable,'-B',str(worker),str(out)],stdout=log,stderr=subprocess.STDOUT)
 monitor=subprocess.Popen([sys.executable,'-c','raise SystemExit(4)'])
 code=supervise(child,monitor,out,poll_s=.01,notice_after_s=2.)
assert code==75 and (out/'STOP_REQUEST.json').exists() and (out/'STOPPED.json').exists()
stopped=json.loads((out/'STOPPED.json').read_text());assert stopped['uncertain_updates']==0
(out/'result.json').write_text(json.dumps(dict(status='passed',monitor_exit=4,child_exit=75,stop_request_written=True,
 completed=stopped['completed_updates'],forced_kill=False,committed_state=True,ACT_training_updates=0),indent=2)+'\n')
print('PASS: monitor failure -> persisted stop request -> child boundary checkpoint -> exit75; no force kill')
