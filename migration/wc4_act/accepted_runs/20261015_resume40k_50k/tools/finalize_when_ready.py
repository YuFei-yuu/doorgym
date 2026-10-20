import subprocess,time,sys
from common import *
assert not (RUN/'finalizer-started.json').exists();save(RUN/'finalizer-started.json',{'pid':os.getpid(),'at':time.time(),'scope':'Analysis only, after final64; no further training or simulation'})
while not (RUN/'evaluation-complete.json').exists():
 assert not list(RUN.glob('stop*.json'))
 if (RUN/'train-result.json').exists():assert json.loads((RUN/'train-result.json').read_text())['complete']
 time.sleep(20)
while not (RUN/'analysis-existing/manifest.json').exists():time.sleep(3)
with (RUN/'accept-results.log').open('w') as log:result=subprocess.run([sys.executable,'-B',str(RUN/'tools/accept_results.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
save(RUN/'accept-results-exit.json',{'returncode':result.returncode,'at':time.time()});result.check_returncode()
