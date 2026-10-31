"""Bind426feb7 complete dataset and old actual fixed16. No model or sampling."""
from common import *
import shutil,time,copy
assert not RUN.exists();RUN.mkdir(parents=True)
assert sha(INPUT_FINAL/'SHA256SUMS')=='4ddfd0043f90587b23a238d854ebc3430e71541199b670a95ac864fb1b80723b'
meta=read(INPUT_FINAL/'manifest.json');assert meta['final_commit']=='426feb7473e2bc08bbe2d3754a0f75f09dd9bf45'
assert (meta['episodes'],meta['frames'])==(100,154667)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
databinding=read(INPUT_FINAL/'dataset-SHA256.json')
for row in databinding['files']:
 p=DATA/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
write(RUN/'input-dataset-binding.json',dict(input_freeze=meta['final_commit'],repo_id=REPO,frames=FRAMES,episodes=100,files=databinding['files'],statistics_sha256=meta['statistics_sha256'],input_source_index_sha256=meta['source_index_sha256'],input_sha256sums_sha256=sha(INPUT_FINAL/'SHA256SUMS')))
provenance=RUN/'input-provenance';provenance.mkdir()
for name in ['manifest.json','acceptance-summary.json','SOURCE_TABLE.csv','candidate-index.json','dataset-SHA256.json','SHA256SUMS']:
 shutil.copyfile(INPUT_FINAL/name,provenance/name)
shutil.copyfile(DATA/'meta/stats.json',RUN/'dataset-statistics.json')
old=OLD50_FINAL/'evidence';assert old.is_dir()
assert sha(OLD50_FINAL/'SHA256.json')=='365415bfec8a126ab41fffa03749f052c180dd071c92c03828081c0a1c72b5b4'
old_hashes={row['path']:row['sha256'] for row in read(OLD50_FINAL/'SHA256.json')}
for p in [old/'fixed16.json',old/'manifest.json',old/'frozen-physics.yaml',old/'eleven-stage-comparison.json',*sorted((old/'actual-initial').rglob('*.json'))]:
 assert sha(p)==old_hashes[str(p.relative_to(OLD50_FINAL))]
shutil.copyfile(old/'frozen-physics.yaml',RUN/'frozen-physics.yaml')
source=read(old/'manifest.json');source.update(start_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),yaml=str(RUN/'frozen-physics.yaml'),yaml_sha256=sha(RUN/'frozen-physics.yaml'))
files=subprocess.check_output(['git','ls-files','migration/wc4_act','high-level'],cwd=ROOT,text=True).splitlines()
binding={n:sha(ROOT/n) for n in files if (ROOT/n).is_file()};source['source_sha256']=binding
source.update(training_initialization='fresh ACT, no old task weights; ImageNet ResNet18 V1',target_updates=200000,evaluation_budget=160,training_data_freeze=meta['final_commit'])
write(RUN/'source-binding.json',binding);write(RUN/'manifest.json',source)
prior=read(old/'fixed16.json');rows=[]
for row in prior['trials']:
 seed=row['seed'];target=RUN/'attempts'/('seed%d'%seed);target.mkdir(parents=True)
 trial=copy.deepcopy(row['trial']);original=copy.deepcopy(trial);argv=trial['argv'];j=argv.index('--a2w_float_ik_config')+1
 assert sha(Path(argv[j]))==sha(RUN/'frozen-physics.yaml');argv[j]=source['yaml']
 assert argv[argv.index('--steps')+1]=='6000'
 assert trial['sampling_before_asset_overrides']==original['sampling_before_asset_overrides'] and trial['camera_local_poses']==original['camera_local_poses']
 doc=dict(start_commit=source['start_commit'],trials=[trial],yaml=source['yaml'],yaml_sha256=source['yaml_sha256'],source_sha256=binding,r7_physics_after_asset_overrides=source['r7_physics_after_asset_overrides'])
 write(target/'manifest.json',doc);rows.append(dict(seed=seed,manifest=str(target/'manifest.json'),sha256=sha(target/'manifest.json'),trial=trial))
assert [r['seed'] for r in rows]==list(range(62300,62316))
training_seeds=[e['seed'] for e in read(INPUT_FINAL/'candidate-index.json')['episodes']];assert not set(training_seeds)&set(prior['seeds'])
write(RUN/'fixed16.json',dict(seeds=prior['seeds'],training_source_seeds=training_seeds,overlap=[],sampled_once=True,filtered=False,trials=rows,reused_numeric_initials_without_redraw=True,prior_manifest=str(old/'fixed16.json'),prior_manifest_sha256=sha(old/'fixed16.json')))
shutil.copytree(old/'actual-initial',RUN/'actual-initial')
shutil.copyfile(old/'eleven-stage-comparison.json',RUN/'old50-stage-comparison.json')
write(RUN/'fixed16-provenance.json',dict(reference_final=str(OLD50_FINAL),fixed16_sha256=sha(old/'fixed16.json'),actual_initials=[dict(path=str(p.relative_to(old/'actual-initial')),sha256=sha(p)) for p in sorted((old/'actual-initial').rglob('*')) if p.is_file()],no_random_draws=True))
write(RUN/'order.json',dict(dataset_commit=meta['final_commit'],steps=200000,start_step=0,stages=STAGES,eval_per_stage=16,total_evaluation_trials=160,max_seconds=120,seed=1000,batch=16,lr=1e-5,scheduler=None,chunk=100,training_n_action_steps=50,eval_action_horizon=25,video=False,initialization='fresh; ImageNet backbone only'))
free=shutil.disk_usage(ROOT).free;assert free>65*1024**3
write(RUN/'storage-preflight.json',dict(free_bytes=free,checkpoint_export_and_compressed_evidence_estimate_bytes=25*1024**3,reserve_bytes=40*1024**3,raw_logs_lossless_compression_each_completed_trial=True,no_historical_cleanup=True))
write(RUN/'preflight.json',dict(passed=True,dataset_hashes_verified=True,fixed_initials_no_redraw=True,from_scratch=True,at=time.time()))
verify();print('Bound426feb7 all100/154667 and old actual fixed16 without redraw; no ACT model constructed')
