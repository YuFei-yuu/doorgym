"""R17 preflight: freeze exactly original16 and fixed50k, no policy or scene."""
import sys,subprocess,shutil,ast,time
from common import *
assert not RUN.exists() and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
assert shutil.disk_usage(ROOT).free>100*1024**3
RUN.mkdir();(RUN/'evaluation').mkdir();(RUN/'evaluation-claims').mkdir()
source=read(BASE/'source-binding.json')
for rel,h in source.items():assert sha(ROOT/rel)==h,rel
for p in Path(__file__).parent.iterdir():
 if p.is_file():
  if p.suffix=='.py':ast.parse(p.read_text())
  source[str(p.relative_to(ROOT))]=sha(p)
write(RUN/'source-binding.json',source)
pretrained=BASE/'train/checkpoints/050000/pretrained_model'
assert sha(pretrained/'model.safetensors')==MODEL_SHA
paths=[p for p in pretrained.iterdir() if p.is_file()]+[CHECKPOINT,CHECKPOINT.parent/'model_latest/door_policy_meta.json']
write(RUN/'model-binding.json',dict(step=50000,model_sha256=MODEL_SHA,checkpoint=str(CHECKPOINT),files={str(p):sha(p) for p in paths},export_reused=True,new_export_or_forward=False))
base_records=[]
master=read(BASE/'manifest.json');trials=[]
for i,seed in enumerate(SEEDS):
 d=RUN/'frozen'/str(seed);d.mkdir(parents=True)
 old=BASE/'frozen'/str(seed)/'manifest.json';frozen=read(old);trial=frozen['trials'][0];assert trial['seed']==seed
 # Same actual scene parameters, argv, image randomization and physics, only source binding differs.
 frozen['source_sha256']=source;write(d/'manifest.json',frozen);trials.append(trial)
 original=BASE/'evaluation/050000'/f'trial_{i:02d}'
 assert read(original/'frozen-trial.json')==trial
 record=read(original/'paired-diagnostics.json');record.update(original_directory=str(original),full_task_success=read(original/'evaluation-result.json')['full_task_success'],frozen_files={name:sha(original/name) for name in ['initial-state.json','physical-camera-binding.json','frozen-trial.json','r11-score.json','evaluation-result.json','paired-diagnostics.json']})
 base_records.append(record)
master.update(source_sha256=source,trials=trials,seeds=SEEDS);write(RUN/'manifest.json',master)
write(RUN/'baseline-h25.json',dict(model_step=50000,model_sha256=MODEL_SHA,horizon=25,records=base_records,no_baseline_rerun=True))
for n in ['frozen-r7-physics.yaml','geometry.json','dataset-sha256.json','collection-disjoint.json']:shutil.copy2(BASE/n,RUN/n)
write(RUN/'contract.json',dict(task='R17 fixed50k horizon5 paired16',model_step=50000,model_sha256=MODEL_SHA,execution_horizon=5,baseline_horizon=25,chunk100=True,action_hz=25,physics_hz=50,action_hold_substeps=2,steps_per_episode=9000,max_sim_seconds=180,seeds=SEEDS,evaluation_budget=16,training_updates=0,new_collection=0,extra_smoke_scenes=0,new_export=False,new_inputs=False,minimum_success_threshold=None,ordinary_rotation_or_camera_escape_stops_batch=False,automatic_restart=False,original_RGB_keep_all=True,only_research_variable='executed action horizon 25 to5'))
write(RUN/'startup-binding.json',dict(at=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.executable,free_bytes=shutil.disk_usage(ROOT).free,no_model_forward_or_scene=True))
write(RUN/'frozen-files.json',{str(p.relative_to(RUN)):sha(p) for p in [RUN/'contract.json',RUN/'manifest.json',RUN/'baseline-h25.json',RUN/'frozen-r7-physics.yaml',RUN/'geometry.json',*sorted((RUN/'frozen').glob('*/manifest.json'))]})
print('Prepared fixed50k h5, exactly16; zero model calls/scenes',flush=True)
