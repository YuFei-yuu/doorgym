"""Freeze R13 engineering scope and unchanged initial conditions, CPU only."""
import sys,subprocess,hashlib,shutil
from runtime import *
RUN=ROOT/'.repro/runs/wc4-engineering-r13-20261022'
if RUN.exists():raise RuntimeError('New engineering evidence directory required')
if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise RuntimeError('Commit repairs before regression')
RUN.mkdir(parents=True)
protected={}
for name,digest in read(R12/'source-binding.json').items():
 if name not in ['high-level/dp/door_policy_backend.py','high-level/dp/door_policy_worker.py']:
  if sha(ROOT/name)!=digest:raise RuntimeError('Unexpected protected source change: '+name)
  protected[str(ROOT/name)]=digest
for name,digest in read(R12/'dataset-sha256.json').items():
 if sha(DATA/name)!=digest:raise RuntimeError('Accepted dataset changed')
 protected[str(DATA/name)]=digest
for name,item in read(R12/'delivery-sha256.json')['files'].items():
 if sha(R12/name)!=item['sha256']:raise RuntimeError('R12 evidence changed: '+name)
 protected[str(R12/name)]=item['sha256']
files=list((SUPPORT/'r13').glob('*.py'))+[ROOT/'high-level/dp/door_action_evidence.py',ROOT/'high-level/dp/door_policy_backend.py',ROOT/'high-level/dp/door_policy_worker.py']
source=read(R12/'source-binding.json');source.update({str(p.relative_to(ROOT)):sha(p) for p in files})
master=read(R12/'manifest.json');original_trials=master['trials'];master['source_sha256']=source
write(RUN/'manifest.json',master)
for trial in original_trials:
 folder=RUN/'frozen'/str(trial['seed']);folder.mkdir(parents=True)
 one=dict(master,trials=[trial],seeds=[trial['seed']]);write(folder/'manifest.json',one)
assert read(RUN/'manifest.json')['trials']==read(R12/'manifest.json')['trials']
write(RUN/'source-binding.json',source)
write(RUN/'protected-before.json',protected)
write(RUN/'scope.json',dict(formal_ACT_training=0,real_policy_inference=0,simulation=0,closed_loop=0,
    CPU_fixture_regression_only=True,training_authorization=False,
    r12_completed_updates=1122,r12_consumption_upper=1123,original_frozen_trials_unchanged=True))
print('Frozen R13 CPU-only engineering scope and unchanged 72 initial/camera tuples; no training order created')
