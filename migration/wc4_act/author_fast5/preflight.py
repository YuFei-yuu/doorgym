"""Necessary CPU reference/contract checks, no simulator imports or parameter search."""
import argparse,ast,hashlib,json,math,os,sys,subprocess
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from author_fast5.geometry import DoorFrame,transform,closing_reference,progress
from author_fast5.trajectory import CompleteExpert,Measured

def main():
 p=argparse.ArgumentParser();p.add_argument('--check',type=Path,required=True);p.add_argument('--design',default='design03');p.add_argument('--baseline-ref',default='bbcbe44');a=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 here=Path(__file__).parent;cfg=json.loads((here/'config.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 for p in here.glob('*.py'):compile(p.read_text(),str(p),'exec')
 bind=json.loads((a.check/a.design/'binding.json').read_text())
 for name in ['config.json','geometry.py','check_design.py']:assert bind['sources'][name]==sha(here/name)
 frame=DoorFrame(transform([2.61,0,.01],[0,0,math.pi/2]))
 duration=cfg['close_s'];ramp=cfg['progress_ramp_s']
 tt=np.arange(0,duration+.01,.02);samples=[closing_reference(cfg,frame,t) for t in tt]
 B=np.array([r[0] for r in samples]);P=np.array([r[1] for r in samples]);v=np.diff(B[:,:2],axis=0)/.02;w=np.diff(B[:,2])/.02
 lateral=np.sum(v*np.column_stack((-np.sin(B[:-1,2]),np.cos(B[:-1,2]))),axis=1)
 assert max(abs(lateral))<1e-8
 u=np.array([progress(t,duration,ramp) for t in tt]);du=np.diff(u)/.02
 plateau=(tt[:-1]>=ramp)&(tt[1:]<=duration-ramp)
 assert np.all(du>0) and max(du[plateau])-min(du[plateau])<1e-10
 assert progress(0,duration,ramp)==0 and progress(duration,duration,ramp)==1
 args=NS(door_x=2.5,door_y=0.,door_z_offset=.01,robot_z=.5,robot_pitch=0.,gripper_open=-math.pi/2)
 st=NS(args=args,door=NS(actor_position_offset=[.11,0,0],bounding={'min':[0,0,0]},actor_scale=1.,actor_yaw=math.pi/2),
  home_positions=np.r_[np.zeros(6),-1.1775],base_traverse=np.array([1.26,.18]),yaw_traverse=math.pi)
 e=CompleteExpert(NS(),st,['joint%d'%i for i in range(1,7)]+['jointGripper'],cfg)
 assert np.linalg.norm(e.Bclose[:2]-st.base_traverse)<1e-12
 o=Measured(0.,e.Bclose.copy(),st.home_positions.copy(),np.zeros(7),np.array([1.55,-.49,.92]),np.eye(3),1.57,0.,True,True,0.,0.,False)
 e.enter('close',o)
 x=e.tick(0,None,replace(o,time=2.,door_angle=1.4,released=False));y=e.tick(0,None,replace(o,time=2.,door_angle=.2,released=True))
 assert np.array_equal(x.base,y.base) and np.array_equal(x.position,y.position)
 source=(here/'trajectory.py').read_text()
 assert 'move_to_start' not in source and 'turn_to_start' not in source
 effective=json.loads((ROOT/'.repro/runs/wc4-newexpert-stage4-20261025/attempt_000/effective-config.json').read_text())
 open_s=sum(effective[n] for n in ['walk_steps','initial_hold_steps','grasp_steps','grasp_hold_steps','gripper_close_steps','handle_rotate_steps','door_push_steps'])*.02
 traverse_s=effective['traverse_steps']*.02
 t=open_s+max(traverse_s,cfg['parallel_withdraw_s']+cfg['fold_s'])
 timeline=[dict(phase='author_open',start=0,end=open_s),dict(phase='author_traverse',start=open_s,end=open_s+traverse_s),
  dict(phase='parallel_withdraw_and_fold',start=open_s,end=open_s+cfg['parallel_withdraw_s']+cfg['fold_s'])]
 for phase,duration in [('turn_for_close',cfg['turn_s']),('deploy',cfg['deploy_s']),('approach',cfg['approach_s']),('close',cfg['close_s']),('lock_confirmation',.5),('release',cfg['release_s']),('fold',cfg['fold_s']),('restore_heading',cfg['restore_s']),('final_hold',2.)]:
  timeline.append(dict(phase=phase,start=t,end=t+duration));t+=duration
 oldcfg=json.loads(subprocess.check_output(['git','show',a.baseline_ref+':migration/wc4_act/author_fast5/config.json'],cwd=ROOT,text=True))
 scale=cfg['close_s']/oldcfg['close_s']
 oldsamples=[closing_reference(oldcfg,frame,t/scale) for t in tt]
 path_difference=max(float(np.max(abs(samples[i][j]-oldsamples[i][j]))) for i in range(len(tt)) for j in [1,2])
 base_xy_difference=max(float(np.max(abs(samples[i][0][:2]-oldsamples[i][0][:2]))) for i in range(len(tt)))
 base_yaw_difference=max(float(abs(samples[i][0][2]-oldsamples[i][0][2])) for i in range(len(tt)))
 inset_residual=0.
 for candidate,old_sample in zip(samples,oldsamples):
  theta=candidate[3];delta=0.
  if cfg.get('contact_inset_m',0.):
   start=math.radians(cfg['contact_inset_start_deg']);end=math.radians(cfg['contact_inset_end_deg'])
   from author_fast5.geometry import smooth
   delta=cfg['contact_inset_m']*smooth((start-theta)/(start-end))
  expected=-delta*(frame.R@np.array([-math.sin(theta),math.cos(theta),0.]))
  inset_residual=max(inset_residual,float(np.max(abs(candidate[1]-old_sample[1]-expected))))
 assert inset_residual<1e-12
 assert max(float(np.max(abs(x[2]-y[2]))) for x,y in zip(samples,oldsamples))<1e-12
 assert base_xy_difference<1e-12
 changed=[k for k in cfg if cfg[k]!=oldcfg.get(k)]
 result=dict(physics_steps=0,passed=True,author_traverse_endpoint_reused=True,no_micro_relocation=True,closing_uses_measured_door=False,
  reference_seconds=cfg['close_s'],progress_plateau_rate_s=float(np.mean(du[plateau])),max_reference_ee_speed_m_s=float(np.linalg.norm(np.diff(P,axis=0)/.02,axis=1).max()),
  max_base_speed_m_s=float(np.linalg.norm(v,axis=1).max()),max_base_yaw_rate_rad_s=float(max(abs(w))),max_lateral_speed_m_s=float(max(abs(lateral))),
  reference_total_nominal_s=t,reference_timeline=timeline,source_sha256={p.name:sha(p) for p in here.iterdir() if p.is_file()},
  comparison_baseline_ref=a.baseline_ref,arm_time_scale=scale,ee_reference_max_abs_difference=path_difference,inset_formula_max_residual_m=inset_residual,base_xy_reference_max_abs_difference=base_xy_difference,base_yaw_schedule_max_difference_rad=base_yaw_difference,changed_config_fields=changed,
  limitation='ideal DLS check; no physical or robustness success implied; actual contact-to-reference-end timing is measured in the one rollout')
 (a.check/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
