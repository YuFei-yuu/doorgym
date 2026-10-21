"""Seal implementation before first update or scene; never resample parameters."""
from common import *
import subprocess,shutil
assert not (RUN/'train-command.json').exists() and not (RUN/'evaluation-claims').exists()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
# Keep the original prepared binding for provenance. Parameters are compared
# before changing only implementation digests inside per-trial manifests.
if not (RUN/'prepared-source-binding.json').exists():shutil.copyfile(RUN/'source-binding.json',RUN/'prepared-source-binding.json')
old=read(RUN/'source-binding.json');source=dict(old)
for p in (SUPPORT/'r12').iterdir():
 if p.is_file():source[str(p.relative_to(ROOT))]=sha(p)
for name in old:
 if not name.startswith('migration/wc4_act/r12/'):assert sha(ROOT/name)==old[name],name
before={p:read(p)['trials'] for p in [RUN/'manifest.json',*sorted((RUN/'frozen').glob('*/manifest.json'))]}
for p,trials in before.items():
 manifest=read(p);manifest['source_sha256']=source;write(p,manifest)
 assert read(p)['trials']==trials
write(RUN/'source-binding.json',source)
frozen=read(RUN/'frozen-files.json')
for name,digest in frozen.items():
 if Path(name).name!='manifest.json':assert sha(RUN/name)==digest,name
for p in before:frozen[str(p.relative_to(RUN))]=sha(p)
write(RUN/'frozen-files.json',frozen)
write(RUN/'execution-version.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),worktree_clean=True,parameters_unchanged=True,scene_count=0,updates=0))
print('Implementation sealed; all 72 frozen initial/camera tuples unchanged; zero updates/scenes.')
