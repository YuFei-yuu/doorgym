"""Offline author sampler only; no simulator, model, optimizer or filtering."""
from common import *
sys.path.insert(0,str(SUPPORT))
from expert50.sample import Sampler
import shutil,subprocess
assert not RUN.exists()
RUN.mkdir(parents=True)
old=ROOT/'.repro/reviews/wc4-expert50-20261029-FINAL/evidence'
# Accepted collector manifest retains exact original-author arguments and physics.
source=read(old/'manifest.json')
shutil.copyfile(old/'frozen-physics.yaml',RUN/'frozen-physics.yaml')
source.update(start_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),yaml=str(RUN/'frozen-physics.yaml'),yaml_sha256=sha(RUN/'frozen-physics.yaml'))
files=subprocess.check_output(['git','ls-files','migration/wc4_act','high-level'],text=True).splitlines()
binding={n:sha(ROOT/n) for n in files if (ROOT/n).is_file()}
source['source_sha256']=binding
write(RUN/'source-binding.json',binding)
write(RUN/'manifest.json',source)
(RUN/'attempts').mkdir()
# Preserve the prior fixed16 numeric states/cameras exactly; never redraw.
prior_run=ROOT/'.repro/runs/wc4-act50k-20261029'
prior=read(prior_run/'fixed16.json');rows=[]
for row in prior['trials']:
 seed=row['seed'];target=RUN/'attempts'/('seed%d'%seed);target.mkdir()
 old_doc=read(row['manifest']);trial=old_doc['trials'][0]
 original=dict(seed=seed,sampling=trial['sampling_before_asset_overrides'],cameras=trial['camera_local_poses'])
 argv=trial['argv'];argv[argv.index('--a2w_float_ik_config')+1]=source['yaml']
 assert argv[argv.index('--steps')+1]=='6000'
 doc=dict(start_commit=source['start_commit'],trials=[trial],yaml=source['yaml'],yaml_sha256=source['yaml_sha256'],source_sha256=binding,r7_physics_after_asset_overrides=source['r7_physics_after_asset_overrides'])
 write(target/'manifest.json',doc)
 assert original==dict(seed=seed,sampling=trial['sampling_before_asset_overrides'],cameras=trial['camera_local_poses'])
 rows.append(dict(seed=seed,manifest=str(target/'manifest.json'),sha256=sha(target/'manifest.json'),trial=trial))
write(RUN/'fixed16.json',dict(seeds=prior['seeds'],training_source_seeds=list(range(62100,62150)),overlap=[],sampled_once=True,filtered=False,trials=rows,reused_numeric_initials_without_redraw=True,prior_manifest_sha256=sha(prior_run/'fixed16.json')))
write(RUN/'order.json',dict(dataset_commit='197a169133c847e23bbf0f4ea5c389bf1234fae4',steps=50000,stages=STAGES,eval_per_stage=16,max_seconds=120,seed=1000,batch=16,lr=1e-5,chunk=100,training_n_action_steps=50,eval_action_horizon=25,video=False))
print('Reused all16 fixed numeric states/cameras exactly; no resampling or simulation.')
