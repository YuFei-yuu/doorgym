"""Accept final50k and compare fixed64 to36k/40k; never run a model/simulator."""
import csv,gzip,math,shutil,time
from collections import defaultdict
from common import *
import numpy as np,torch,wandb
from safetensors import safe_open
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
source_check();assert not list(RUN.glob('stop*.json'))
assert json.loads((RUN/'training-complete.json').read_text())['updates']==50000
assert json.loads((RUN/'train-result.json').read_text())['returncode']==0
assert json.loads((RUN/'evaluation-complete.json').read_text())['total_policy_trials']==64
updates=[json.loads(s) for s in (RUN/'updates.jsonl').read_text().splitlines()]
assert [v['update'] for v in updates]==list(range(40001,50001))
assert all(all(math.isfinite(v[k]) for k in ['loss','grad_norm','lr']) and v['lr']==1e-5 for v in updates)
previous=json.loads((PARENT/'acceptance.json').read_text())
parent_cp=next(v for v in previous['checkpoints'] if v['step']==40000)
for row in parent_cp['checkpoint_files']:assert sha(row['path'])==row['sha256']
save(RUN/'final-dataset-check.json',{'unchanged':True,'files':data_check()})
assert json.loads((RUN/'restored-rng-check.json').read_text())['all_rng_tensors_equal_after_author_restore']
assert all(v['all_equal'] for v in json.loads((RUN/'restored-statistics.json').read_text())['processors'])
receipt=json.loads((RUN/'wandb-run.json').read_text());run=wandb.Api(timeout=60).run(f"{receipt['entity']}/{receipt['project']}/{receipt['id']}")
for attempt in range(3):
 history=list(run.scan_history(keys=['train/steps','train/loss'],page_size=1000));unique={}
 for row in history:
  step=int(row['train/steps'])
  if step in unique:assert unique[step]==row
  unique[step]=row
 if sorted(unique)==list(range(40100,50001,100)):break
 time.sleep(10)
assert sorted(unique)==list(range(40100,50001,100))
means=np.array([r['loss'] for r in updates]).reshape(100,100).mean(1)
error=max(abs(unique[s]['train/loss']-means[i]) for i,s in enumerate(sorted(unique)));assert error<1e-6,error
save(RUN/'wandb-training-history.json',history)
checkpoints=[]
for step in [42000,44000,46000,48000,50000]:
 directory=RUN/'train/checkpoints'/f'{step:06d}'
 assert json.loads((directory/'training_state/training_step.json').read_text())['step']==step
 with safe_open(directory/'training_state/optimizer_state.safetensors',framework='pt',device='cpu') as f:steps=[float(f.get_tensor(k)) for k in f.keys() if k.endswith('/step')]
 assert steps and set(steps)=={float(step)}
 for name in ['policy_preprocessor_step_3_normalizer_processor.safetensors','policy_postprocessor_step_0_unnormalizer_processor.safetensors']:
  with safe_open(SOURCE/'pretrained_model'/name,framework='pt',device='cpu') as old,safe_open(directory/'pretrained_model'/name,framework='pt',device='cpu') as new:
   assert old.keys()==new.keys()
   for key in old.keys():assert torch.equal(old.get_tensor(key),new.get_tensor(key))
 files=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(directory.rglob('*')) if p.is_file()]
 checkpoints.append({'step':step,'files':files,'optimizer_step_tensors':len(steps),'model_sha256':sha(directory/'pretrained_model/model.safetensors'),'statistics_equal_40k':True,'closed_loop_evaluated':step==50000})
CP=RUN/'checkpoints/050000';evaluation=json.loads((CP/'evaluation-results.json').read_text());assert evaluation['trials']==64 and evaluation['status']=='passed'
trials=evaluation['results'];assert [r['trial_index'] for r in trials]==list(range(64))
old_by={s:{v['trial_index']:v for v in previous['all_trials'] if v['checkpoint_step']==s} for s in [36000,40000]}
new_by={v['trial_index']:v for v in trials};pair_counts={};per_trial=[]
for baseline in [36000,40000]:
 a={i for i,v in old_by[baseline].items() if v['success']};b={i for i,v in new_by.items() if v['success']};u=set(range(64))
 groups={'common_success':sorted(a&b),'common_failure':sorted(u-(a|b)),'improvement':sorted(b-a),'regression':sorted(a-b)}
 assert sum(map(len,groups.values()))==64
 pair_counts[str(baseline)]={'baseline_successes':len(a),'final_successes':len(b),'counts':{k:len(v) for k,v in groups.items()},'trial_ids':groups,'net_success_change':len(b)-len(a)}
 for i in range(64):
  x=old_by[baseline][i];y=new_by[i];assert (x['seed'],x['env_id'],x['env_seed'])==(y['seed'],y['env_id'],y['env_seed'])
  category=next(k for k,v in groups.items() if i in v)
  per_trial.append({'baseline':baseline,'trial_index':i,'seed':y['seed'],'env_id':y['env_id'],'baseline_success':x['success'],'final_success':y['success'],'baseline_max_abs_door_deg':x['max_abs_open_deg'],'final_max_abs_door_deg':y['max_abs_open_deg'],'category':category})
assert len({m['path'] for c in checkpoints for m in c['files']})==55
archive=[];initial_checks=[];metrics=[];curves={}
for batch in range(4):
 d=CP/'evaluation'/f'batch_{batch:03d}';summary=json.loads((d/'summary.json').read_text());independent=json.loads((d/'independent-check.json').read_text());assert summary['successes']==independent['successes']
 source_detail={v['env_id']:v for v in summary['trials_detail']}
 raw=d/'batch_0000.jsonl';records=defaultdict(list)
 with raw.open() as f:
  for line in f:
   row=json.loads(line);records[row['controlled_env_id']].append(row)
 for env,rows in sorted(records.items()):
  trial=batch*16+env;current=new_by[trial];assert source_detail[env]['success']==current['success'] and abs(source_detail[env]['max_open_deg']-current['max_abs_open_deg'])<1e-8
  first=rows[0]
  for baseline in [36000,40000]:
   oldlog=Path(old_by[baseline][trial]['log'])
   with oldlog.open() as f:oldfirst=[json.loads(next(f)) for _ in range(16)][env]
   assert oldfirst['controlled_env_id']==env and oldfirst['step']==0
   assert first['ikpush_randomization']==oldfirst['ikpush_randomization']
   errors={'state':float(np.abs(np.array(first['state'])-oldfirst['state']).max())}
   for k in ['replay_dof_pos','replay_dof_vel','replay_door_root_state','replay_door_dof_pos','replay_door_dof_vel','replay_ee_pos','replay_ee_quat','front_camera_pose_base','wrist_camera_pose_base']:
    errors[k]=float(np.abs(np.array(first['sim_snapshot'][k])-oldfirst['sim_snapshot'][k]).max())
   assert max(errors.values())<=1e-5+1e-12,(baseline,trial,errors)
   initial_checks.append({'baseline':baseline,'trial_index':trial,'same_randomization':True,'max_errors':errors})
  door=np.abs(np.rad2deg([v['door']['dof'][0] for v in rows]));handle=np.abs(np.rad2deg([v['door']['dof'][1] for v in rows]));unlock=np.array([v['sim_snapshot']['replay_door_open_stage'][0]>.5 for v in rows])
  contact=[v.get('extra',{}).get('gripper_handle_contact_any',[None])[0] for v in rows]
  stage='success_80deg' if current['success'] else 'unlocked_but_below80' if unlock.any() else 'unlock_not_observed'
  metrics.append({'step':50000,'trial_index':trial,'seed':current['seed'],'env_id':env,'success':current['success'],'max_abs_door_deg':float(door.max()),'final_abs_door_deg':float(door[-1]),'max_abs_handle_deg':float(handle.max()),'first_observed_unlock_step':int(np.flatnonzero(unlock)[0]) if unlock.any() else None,'observed_failure_stage':stage,'positive_contact_any_rows':sum(v is not None and v>.5 for v in contact),'contact_field_rows':sum(v is not None for v in contact)})
  curves[trial]={'door':door,'handle':handle,'unlock':unlock}
 gz=raw.with_suffix('.jsonl.gz');assert not gz.exists()
 with raw.open('rb') as src,gzip.open(gz,'wb',compresslevel=6) as dest:shutil.copyfileobj(src,dest)
 archive.append({'raw':str(raw),'raw_sha256':sha(raw),'gzip':str(gz),'gzip_sha256':sha(gz),'raw_preserved':True})
focus=json.loads((RUN/'analysis-existing/focus.json').read_text());old_metrics=json.loads((RUN/'analysis-existing/focus-metrics.json').read_text());focus_ids=set().union(*[set(v) for v in focus['groups'].values()])
focus_comparison=[]
for name,ids in focus['groups'].items():
 for i in ids:
  a=next(v for v in old_metrics if v['step']==36000 and v['trial_index']==i);b=next(v for v in old_metrics if v['step']==40000 and v['trial_index']==i);c=next(v for v in metrics if v['trial_index']==i)
  focus_comparison.append({'group':name,'trial_index':i,'seed':c['seed'],'env_id':c['env_id'],'36k':a,'40k':b,'50k':c})
assert len(focus_comparison)==17
save(RUN/'failure-stage-50k.json',{'metrics':metrics,'stage_census':{k:sum(v['observed_failure_stage']==k for v in metrics) for k in ['success_80deg','unlock_not_observed','unlocked_but_below80']},'limitations':json.loads((RUN/'analysis-existing/limitations.json').read_text()),'focus_defined_before50k':focus})
save(RUN/'focus-36k-40k-50k.json',focus_comparison);save(RUN/'paired-comparison.json',pair_counts);save(RUN/'paired-initial-state-check.json',{'status':'passed','comparisons':initial_checks,'count':len(initial_checks)})
for name,ids in focus['groups'].items():
 fig,axes=plt.subplots(math.ceil(len(ids)/2),2,figsize=(14,3*math.ceil(len(ids)/2)),squeeze=False)
 for ax,i in zip(axes.flat,ids):
  v=curves[i];ax.plot(np.arange(1100)*.02,v['door'],label='50k absolute hinge');ax.plot(np.arange(1100)*.02,v['handle'],ls='--',label='50k absolute handle');ax.axhline(80,color='grey',ls=':',lw=.7);ax.set_title(f'trial {i}, seed {615455575+i//16}, env {i%16}');ax.set_ylim(-2,95);ax.grid(alpha=.2)
 for ax in list(axes.flat)[len(ids):]:ax.set_visible(False)
 axes.flat[0].legend(fontsize=7);fig.supxlabel('Simulation seconds');fig.supylabel('Degrees');fig.suptitle('50k outcomes on predeclared '+name);fig.tight_layout();fig.savefig(RUN/(name+'-50k.png'),dpi=140);plt.close(fig)
with (RUN/'per-trial-results.csv').open('w') as f:
 keys=['checkpoint_step','trial_index','batch_index','seed','env_id','env_seed','success','max_abs_open_deg','model_sha256'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:v[k] for k in keys} for v in trials)
with (RUN/'paired-comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(per_trial[0]));w.writeheader();w.writerows(per_trial)
fig,ax=plt.subplots(figsize=(10,4));ax.plot(np.arange(40100,50001,100),means);ax.set_xlabel('Optimizer update');ax.set_ylabel('Train loss (100-step mean)');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(RUN/'training-40k-50k.png',dpi=160);plt.close(fig)
for v in evaluation['selected_videos'].values():assert sha(v['video'])==v['video_sha256']
result={'status':'passed','source_commit':HEAD,'start_step':40000,'end_step':50000,'new_updates':10000,'all_training_values_finite':True,'wandb_url':run.url,'wandb_unique_training_points':len(unique),'wandb_raw_training_points':len(history),'wandb_local_max_error':float(error),'first1000_loss':float(np.mean([v['loss'] for v in updates[:1000]])),'last1000_loss':float(np.mean([v['loss'] for v in updates[-1000:]])),'recovery_checkpoints':checkpoints,'evaluated_checkpoint_steps':[50000],'policy_trials':64,'successes':evaluation['successes'],'success_rate':evaluation['success_rate'],'paired_comparisons':pair_counts,'selected_videos':evaluation['selected_videos'],'missing_video_classes':evaluation['missing_video_classes'],'raw_log_archives':archive,'parent40k_unchanged':True,'dataset_unchanged':True,'no_old_model_reruns':True,'fixed_evaluation_set_only':True}
save(RUN/'acceptance.json',result)
online_eval=json.loads((RUN/'wandb-evaluation-run.json').read_text())
with wandb.init(id=online_eval['id'],resume='must',entity='fi-yu-individual',project='door-dp-stop-tail',mode='online',group=ID,name=ID+'-evaluation',job_type='final50k-evaluation',dir=str(RUN),config={'source_commit':HEAD,'checkpoint':50000,'trials':64,'seeds':[615455575+i for i in range(4)],'fixed_evaluation_set_only':True,'no_extra_rollouts_for_analysis':True}) as er:
 er.log({'successes':evaluation['successes'],'trials':64,'success_rate':evaluation['success_rate'],'per_trial':wandb.Table(columns=['trial','seed','env','success','max_abs_deg'],data=[[v['trial_index'],v['seed'],v['env_id'],v['success'],v['max_abs_open_deg']] for v in trials]),'training_curve':wandb.Image(str(RUN/'training-40k-50k.png'))})
 for base,v in pair_counts.items():er.log({f'vs_{base}/{k}':n for k,n in v['counts'].items()})
 save(RUN/'wandb-evaluation-run.json',{'id':er.id,'url':er.url,'mode':'online'})
print(json.dumps({k:result[k] for k in ['status','new_updates','end_step','policy_trials','successes','success_rate','paired_comparisons','first1000_loss','last1000_loss','wandb_local_max_error']},indent=2))

for attempt in range(3):
 try:
  remote=wandb.Api(timeout=60).run('fi-yu-individual/door-dp-stop-tail/'+online_eval['id'])
  assert int(remote.summary['evaluation/completed_trials'])==64
  assert int(remote.summary['evaluation/cumulative_successes'])==evaluation['successes']
  assert int(remote.summary['successes'])==evaluation['successes']
  save(RUN/'wandb-evaluation-final-check.json',{'online':True,'completed_trials':64,'successes':evaluation['successes'],'url':remote.url,'at':time.time()});break
 except Exception as exc:
  if attempt==2:
   save(RUN/'stop-final-online-check.json',{'error':repr(exc),'at':time.time()});raise
  time.sleep(10)
