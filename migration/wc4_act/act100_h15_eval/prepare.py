from common import *
import shutil,time
assert not RUN.exists();RUN.mkdir(parents=True)
parent=ROOT/'.repro/reviews/wc4-act100-fresh200k-20261031-FINAL';e=parent/'evidence'
index={x['path']:x for x in read(parent/'SHA256.json')}
for base in ['train/checkpoints/120000','checkpoints/120000']:
 for p in (e/base).rglob('*'):
  if p.is_file():assert sha(p)==index[str(p.relative_to(parent))]['sha256']
 dest=RUN/base;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(e/base,target_is_directory=True)
shutil.copytree(e/'actual-initial',RUN/'actual-initial')
fixed=read(e/'fixed16.json')
for row in fixed['trials']:
 old=Path(row['manifest']);rel=old.relative_to(ROOT/'.repro/runs/wc4-act100-fresh200k-20261031');row['manifest']=str(e/rel)
 assert Path(row['manifest']).is_file()
write(RUN/'fixed16.json',fixed)
shutil.copyfile(e/'evaluation/120000/summary.json',RUN/'baseline-h25-summary.json')
write(RUN/'checkpoint-source.json',dict(parent_final=str(parent),stage=120000,official_model_sha256=sha(e/'train/checkpoints/120000/pretrained_model/model.safetensors'),original_export_consistency=read(e/'checkpoints/120000/export-consistency.json'),weights_stats_unchanged=True,training_updates=0,evaluation_only_override=dict(prediction=100,execution=15,control_hz=25)))
bind=read(e/'source-binding.json')
for p in Path(__file__).parent.glob('*.py'):bind[str(p.relative_to(ROOT))]=sha(p)
bind[str(Path(__file__).with_name('SCOPE.txt').relative_to(ROOT))]=sha(Path(__file__).with_name('SCOPE.txt'))
write(RUN/'source-binding.json',bind);write(RUN/'manifest.json',dict(start_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),at=time.time(),user_authorized_official_trials=16,optimizer_updates=0,horizon=15,parent_manifest_sha256=sha(parent/'SHA256.json')))
verify();print(json.dumps(dict(prepared=True,run=str(RUN),official_trials=16,training_updates=0)))
