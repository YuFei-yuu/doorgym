"""Verify frozen parent/data, bind existing fixed16; no training or simulation."""
from common import *
import shutil,subprocess,time
import torch
from safetensors.torch import load_file
assert not RUN.exists();RUN.mkdir(parents=True)
assert sha(PARENT_FINAL/'SHA256.json')=='365415bfec8a126ab41fffa03749f052c180dd071c92c03828081c0a1c72b5b4'
manifest={row['path']:row for row in read(PARENT_FINAL/'SHA256.json')}
parent_files=[]
for p in sorted(PARENT.rglob('*')):
 if not p.is_file():continue
 rel=str(p.relative_to(PARENT_FINAL));expected=manifest[rel]
 assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256']
 parent_files.append(dict(path=str(p.relative_to(PARENT)),bytes=p.stat().st_size,sha256=expected['sha256']))
assert read(PARENT/'training_state/training_step.json')['step']==200000
opt=load_file(str(PARENT/'training_state/optimizer_state.safetensors'))
steps=[float(v) for k,v in opt.items() if k.endswith('/step')];assert len(steps)>0 and set(steps)=={200000.}
assert all(torch.isfinite(v).all() for v in opt.values())
model=load_file(str(PARENT/'pretrained_model/model.safetensors'));assert all(torch.isfinite(v).all() for v in model.values())
rng=load_file(str(PARENT/'training_state/rng_state.safetensors'));assert 'torch_cuda_rng_state' in rng
cfg=read(PARENT/'pretrained_model/train_config.json');assert cfg['scheduler'] is None
assert all(g['lr']==1e-5 for g in read(PARENT/'training_state/optimizer_param_groups.json'))
write(RUN/'parent-checkpoint-binding.json',dict(freeze_commit='b80a9540d66718a1ce12bfbb62cec1ef9999ec36',parent_checkpoint=str(PARENT),parent_manifest_sha256=sha(PARENT_FINAL/'SHA256.json'),step=200000,model_tensors=len(model),optimizer_step_tensors=len(steps),optimizer_steps=200000,rng_keys=list(rng),scheduler=None,lr=1e-5,files=parent_files,dataloader_cursor_stored=False))
del model,opt,rng
old=PARENT_FINAL/'evidence'
assert read(PARENT_FINAL/'INDEX.json')['execution_commit']=='b80a9540d66718a1ce12bfbb62cec1ef9999ec36'
for name,expected in read(old/'source-binding.json').items():assert sha(ROOT/name)==expected,('Parent source changed',name)
free=shutil.disk_usage(ROOT).free
estimated_new_bytes=sum(row['bytes'] for row in manifest.values())*5//6
assert free>estimated_new_bytes+28*1024**3, 'Insufficient room for five stages plus25GiB reserve and3GiB temporary headroom'
write(RUN/'storage-preflight.json',dict(free_bytes=free,estimated_new_bytes=estimated_new_bytes,reserve_bytes=28*1024**3,parent_data_bytes_unchanged=True))
databinding=read(old/'input-dataset-binding.json')
for row in databinding['files']:
 p=DATA/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
write(RUN/'input-dataset-binding.json',databinding)
shutil.copyfile(old/'frozen-physics.yaml',RUN/'frozen-physics.yaml')
source=read(old/'manifest.json');source.update(start_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),yaml=str(RUN/'frozen-physics.yaml'),yaml_sha256=sha(RUN/'frozen-physics.yaml'))
files=subprocess.check_output(['git','ls-files','migration/wc4_act','high-level'],text=True).splitlines()
binding={n:sha(ROOT/n) for n in files if (ROOT/n).is_file()}
source['source_sha256']=binding;write(RUN/'source-binding.json',binding);write(RUN/'manifest.json',source)
prior=read(old/'fixed16.json');rows=[]
for row in prior['trials']:
 seed=row['seed'];target=RUN/'attempts'/('seed%d'%seed);target.mkdir(parents=True)
 trial=json.loads(json.dumps(row['trial']));original=json.loads(json.dumps(trial));argv=trial['argv'];argv[argv.index('--a2w_float_ik_config')+1]=source['yaml']
 assert trial['sampling_before_asset_overrides']==original['sampling_before_asset_overrides'] and trial['camera_local_poses']==original['camera_local_poses']
 doc=dict(start_commit=source['start_commit'],trials=[trial],yaml=source['yaml'],yaml_sha256=source['yaml_sha256'],source_sha256=binding,r7_physics_after_asset_overrides=source['r7_physics_after_asset_overrides'])
 write(target/'manifest.json',doc);rows.append(dict(seed=seed,manifest=str(target/'manifest.json'),sha256=sha(target/'manifest.json'),trial=trial))
assert [r['seed'] for r in rows]==list(range(62300,62316))
write(RUN/'fixed16.json',dict(seeds=prior['seeds'],training_source_seeds=prior['training_source_seeds'],overlap=[],sampled_once=True,filtered=False,trials=rows,reused_numeric_initials_without_redraw=True,prior_manifest_sha256=sha(old/'fixed16.json')))
shutil.copytree(old/'actual-initial',RUN/'actual-initial')
write(RUN/'order.json',dict(parent_freeze='b80a954',parent_step=200000,new_updates=100000,total_updates=300000,stages=STAGES,eval_per_stage=16,total_evaluation_trials=80,max_seconds=120,seed=1000,batch=16,lr=1e-5,scheduler=None,chunk=100,training_n_action_steps=50,eval_action_horizon=25,video=False,checkpoint_schedule='220000,240000,260000,280000,300000'))
write(RUN/'preflight.json',dict(passed=True,parent_hashes_verified=True,dataset_hashes_verified=True,fixed_initials_no_redraw=True,at=time.time()))
verify();print(json.dumps(read(RUN/'parent-checkpoint-binding.json'),indent=2))
