"""Analyze immutable 32k-40k logs only; no model or simulator imports."""
import csv,math
from collections import defaultdict
from common import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=RUN/'analysis-existing';assert not O.exists();O.mkdir()
accepted=json.loads((PARENT/'acceptance.json').read_text())
by={s:{v['trial_index']:v for v in accepted['all_trials'] if v['checkpoint_step']==s} for s in [32000,34000,36000,38000,40000]}
failed={s:{i for i,v in rows.items() if not v['success']} for s,rows in by.items()}
groups={'persistent_failure_32k_40k':sorted(set.intersection(*failed.values())),'regression_36k_40k':sorted(failed[40000]-failed[36000]),'improvement_36k_40k':sorted(failed[36000]-failed[40000])}
assert [len(v) for v in groups.values()]==[7,8,2]
focus=set().union(*[set(v) for v in groups.values()]);assert len(focus)==17
save(O/'focus.json',{'groups':groups,'defined_before_50k_evaluation':True,'source_acceptance':str(PARENT/'acceptance.json'),'source_acceptance_sha256':sha(PARENT/'acceptance.json'),'no_new_simulation_or_inference':True,'definition':'persistent = failure intersection across32/34/36/38/40k; regression/improvement=success status change36k->40k'})
metrics=[];curves={};sources=[]
for entry in accepted['raw_log_archives']:
 p=Path(entry['raw']);assert sha(p)==entry['raw_sha256'];sources.append({'path':str(p),'sha256':entry['raw_sha256']})
 step=int(p.parents[2].name);batch=int(p.parent.name.split('_')[1]);records=defaultdict(list)
 with p.open() as f:
  for line in f:
   row=json.loads(line);records[row['controlled_env_id']].append(row)
 for env,rows in sorted(records.items()):
  trial=batch*16+env;assert [v['step'] for v in rows]==list(range(1100))
  assert all(v['phase_name']=='dp_policy' and v['action_mode']=='a2w_joint_action9' for v in rows)
  door=np.abs(np.rad2deg([v['door']['dof'][0] for v in rows]));handle=np.abs(np.rad2deg([v['door']['dof'][1] for v in rows]))
  unlock=np.array([bool(v['sim_snapshot']['replay_door_open_stage'][0]>.5) for v in rows])
  assert np.isfinite(door).all() and np.isfinite(handle).all()
  contact_any=[];contact_both=[]
  for v in rows:
   e=v.get('extra',{});contact_any.append(e.get('gripper_handle_contact_any',[None])[0]);contact_both.append(e.get('gripper_handle_contact_both',[None])[0])
  current=np.array([v['state'][2:9] for v in rows[::2]],dtype=float);target=np.array([v['joint_targets'] for v in rows[::2]],dtype=float)
  assert current.shape==target.shape==(550,7)
  errors=target-current;root=np.array([v['base']['xy'] for v in rows]);maximum=float(door.max());success=maximum>=80
  assert success==by[step][trial]['success'] and abs(maximum-by[step][trial]['max_abs_open_deg'])<1e-8
  observed='success_80deg' if success else 'unlocked_but_below80' if unlock.any() else 'unlock_not_observed'
  first=lambda mask:int(np.flatnonzero(mask)[0]) if np.any(mask) else None
  item={'step':step,'trial_index':trial,'seed':615455575+batch,'env_id':env,'success':success,'max_abs_door_deg':maximum,'final_abs_door_deg':float(door[-1]),'max_abs_handle_deg':float(handle.max()),'first_observed_unlock_step':first(unlock),'first_80deg_step':first(door>=80),'unlock_flag_frames':int(unlock.sum()),'observed_failure_stage':observed,'contact_field_rows':sum(v is not None for v in contact_any),'positive_contact_any_rows':sum(v is not None and v>.5 for v in contact_any),'positive_contact_both_rows':sum(v is not None and v>.5 for v in contact_both),'commanded_base_displacement_m':float(np.linalg.norm(root[-1]-root[0])),'last_vx_command':float(rows[-1]['dp_action_raw'][0]),'target_vs_measured_joint_rmse_rad':np.sqrt(np.mean(errors**2,axis=0)).tolist(),'last2s_joint_rmse_rad':np.sqrt(np.mean(errors[-50:]**2,axis=0)).tolist(),'initial_randomization':rows[0]['ikpush_randomization'],'source_log':str(p)}
  metrics.append(item)
  if trial in focus:curves[f'{step}:{trial}']={'door':door.tolist(),'handle':handle.tolist(),'unlock':unlock.astype(int).tolist()}
assert len(metrics)==320
census={str(s):{label:sum(v['step']==s and v['observed_failure_stage']==label for v in metrics) for label in ['success_80deg','unlock_not_observed','unlocked_but_below80']} for s in by}
selected=[v for v in metrics if v['trial_index'] in focus]
save(O/'metrics-all320.json',metrics);save(O/'focus-metrics.json',selected);save(O/'stage-census.json',census)
cols=['step','trial_index','seed','env_id','success','max_abs_door_deg','final_abs_door_deg','max_abs_handle_deg','first_observed_unlock_step','first_80deg_step','unlock_flag_frames','observed_failure_stage','contact_field_rows','positive_contact_any_rows','positive_contact_both_rows','commanded_base_displacement_m','last_vx_command']
with (O/'focus-metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:v[k] for k in cols} for v in selected)
for name,ids in groups.items():
 ncols=2;nrows=math.ceil(len(ids)/2);fig,axes=plt.subplots(nrows,ncols,figsize=(14,3*nrows),squeeze=False)
 for ax,trial in zip(axes.flat,ids):
  for step,color in [(36000,'tab:blue'),(40000,'tab:orange')]:
   v=curves[f'{step}:{trial}'];t=np.arange(1100)*.02
   ax.plot(t,v['door'],color=color,label=f'{step//1000}k door')
   ax.plot(t,v['handle'],color=color,ls='--',alpha=.7,label=f'{step//1000}k handle')
   u=np.flatnonzero(v['unlock'])
   if len(u):ax.axvline(u[0]*.02,color=color,ls=':',alpha=.6)
  ax.axhline(80,color='grey',ls='--',lw=.6);ax.set_title(f'trial {trial}, seed {615455575+trial//16}, env {trial%16}');ax.set_ylim(-2,95);ax.grid(alpha=.2)
 for ax in list(axes.flat)[len(ids):]:ax.set_visible(False)
 axes.flat[0].legend(fontsize=7);fig.supxlabel('Simulation seconds; dotted vertical line = first observed unlock flag');fig.supylabel('Absolute joint angle (degrees)');fig.suptitle(name+' (existing logs only)');fig.tight_layout();fig.savefig(O/(name+'.png'),dpi=140);plt.close(fig)
notes={'stage_definition':'Failure stage is based on recorded replay_door_open_stage: unlock not observed, or unlock observed but max hinge<80deg. Not inferred from expert phase names; all actual control phases are dp_policy.','contact_limitation':'Positive contact flags are logged observations. Zero does not prove no contact: author getter returns zeros when body handles unavailable or contact API fails, and records no validity flag. Score may be a count proxy, not a calibrated force.','ee_error_limitation':'Joint9 logger sets EE target position from current EE observation; logged ee.pos_error is not an independent handle-goal distance and is not used to diagnose grasp.','tracking_limitation':'Joint target-vs-state RMSE on fresh 25Hz samples is actuator tracking observation, not expert-action imitation error; base displacement is commanded path, not collision-free traversal.','causality':'No new rollout, image reconstruction, causal intervention or algorithm change. Lower handle angle/absent unlock can localize observed failure stage, but cannot distinguish perception, approach, grasp, rotation or contact-maintenance root causes.'}
save(O/'limitations.json',notes)
files=[{'path':str(p.relative_to(O)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(O.iterdir()) if p.is_file()]
save(O/'manifest.json',{'files':files,'source_logs':sources,'defined_before_50k_evaluation':True,'no_new_simulation_or_inference':True})
print(json.dumps({'groups':groups,'stage_census':census,'focus_episode_rows':len(selected),'all_episode_rows':len(metrics)},indent=2))
