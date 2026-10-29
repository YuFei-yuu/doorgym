"""One bounded continuation and audit/freeze, no budget extension or silent restart."""
from common import *
import subprocess,time
verify()
command=['bash','migration/wc4_act/act300k/launch.sh',sys.argv[1],sys.argv[2],*sys.argv[3:]]
write(RUN/'launch-binding.json',dict(argv=command,at=time.time(),gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True),parent_checkpoint=str(PARENT),source_commit=read(RUN/'manifest.json')['start_commit']))
with (RUN/('training-process-'+str(time.time_ns())+'.log')).open('x') as log:
 code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
write(RUN/'training-process-exit.json',dict(code=code,at=time.time()))
if code:raise SystemExit(code)
assert read(RUN/'complete.json')['new_updates']==100000
cmd=['bash','-c','source migration/wc4_act/activate.sh act && export CUDA_VISIBLE_DEVICES="" && exec python -B migration/wc4_act/act300k/freeze.py']
write(RUN/'freeze-command.json',cmd)
with (RUN/'freeze-process.log').open('x') as log:
 code=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
# Frozen tree is never modified after its SHA manifest and permissions publish.
receipt=ROOT/'.repro/reviews/wc4-act300k-20261030-FINALIZATION.json'
write(receipt,dict(code=code,at=time.time(),final=str(ROOT/'.repro/reviews/wc4-act300k-20261030-FINAL'),manifest_sha256=sha(ROOT/'.repro/reviews/wc4-act300k-20261030-FINAL/SHA256.json') if code==0 else None))
raise SystemExit(code)
