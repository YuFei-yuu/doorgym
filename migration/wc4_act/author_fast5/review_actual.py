"""Read-only actual-run review, original R11, plots/video audit; no simulator."""
import argparse,json,math,sys,csv,subprocess,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/data/zyf/video2door-wc4-act');sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from r11.criteria import score

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run;out=run/'attempt_000';dest=run/'review_final';dest.mkdir(exist_ok=False)
 rows=[json.loads(l) for l in (out/'states.jsonl').open()];contact=[json.loads(l) for l in (out/'contacts.jsonl').open()];locks=[json.loads(l) for l in (out/'lock-writes.jsonl').open()]
 ref=json.loads((out/'reference.json').read_text());result=json.loads((out/'result.json').read_text());n=len(rows)
 bodymap=json.loads((out/'body-map.json').read_text());door_ids={x['actor_handle'] for x in bodymap.values() if x['body']=='door_panel'};robot_ids={x['actor_handle'] for x in bodymap.values() if x['actor'].startswith(('b1_base','z1_arm'))}
 for r,c in zip(rows,contact):
  r['loaded_door_contact']=r['nonzero_door_contact'];r['end_panel_raw_contact']=r['end_panel_contact']
  r['zero_lambda_door_records']=sum(x.get('lambda',0)==0 and any(z.get('actor_handle') in door_ids for z in x['names']) for x in c['contacts'])
  r['nonend_door_records']=sum(any(z.get('actor_handle') in door_ids for z in x['names']) and any(z.get('actor_handle') in robot_ids and z.get('body') not in ['gripperStator','gripperMover'] for z in x['names']) for x in c['contacts'])
 gates=score(rows,locks,ref) if n>1 else {'measured_task_gates_pass':False,'failures':['insufficient_physics_steps']}
 write(dest/'r11.json',gates)
 phases=[]
 for i,r in enumerate(rows):
  if not phases or phases[-1]['phase']!=r['phase']:phases.append(dict(phase=r['phase'],first_step=i,start_pre_s=r['timing']['command_time_s'],first_post_s=r['t'],last_step=i,end_post_s=r['t']))
  else:phases[-1].update(last_step=i,end_post_s=r['t'])
 for x in phases:x['observed_duration_s']=x['end_post_s']-x['start_pre_s']
 write(dest/'actual-stage-times.json',phases)
 plan=json.loads((run/'preflight-binding.json').read_text())['preflight']['reference_timeline']
 with (dest/'complete-stage-times.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['phase','planned_start_s','planned_end_s','actual_first_post_s','actual_end_post_s','status'])
  for x in plan:
   matching=[y for y in phases if y['phase']==x['phase']]
   w.writerow([x['phase'],x['start'],x['end'],matching[0]['first_post_s'] if matching else '',matching[-1]['end_post_s'] if matching else '', 'observed' if matching else 'see author subphases / not reached'])
  for y in phases:w.writerow([y['phase'],'','',y['first_post_s'],y['end_post_s'],'actual phase'])
 first=next((r for r in rows if r['phase'] in ['deploy','approach','close'] and r['nonzero_end_panel_contact']),None)
 complete=next((r for r in rows if r['expert_target']['reference_complete']),None)
 latch=next((x for x in locks if x['kind']=='relocked'),None)
 close_rows=[r for r in rows if r['phase']=='close'];moving_close=[r for r in close_rows if not r['expert_target']['reference_complete']]
 foot={}
 for r in rows:
  for name,value in r['foot_guard']['deviations'].items():
   if name not in foot or value>foot[name]['max_deviation_rad']:foot[name]={'max_deviation_rad':value,'time_s':r['t'],'phase':r['phase']}
 bad=next((r for r in rows if r['foot_guard']['bad']),None)
 first_loaded_any=next((r for r in rows if r['nonzero_end_panel_contact']),None)
 metrics=dict(result=result,physics_steps=n,reference_duration_s=json.loads((out/'effective-expert.json').read_text())['close_s'],first_postdeployment_loaded_gripper_contact=None if first is None else {'time_s':first['t'],'phase':first['phase'],'door_deg':math.degrees(first['door_q'][0])},
  first_loaded_any_time_s=None if first_loaded_any is None else first_loaded_any['t'],
  reference_end_post_s=None if complete is None else complete['t'],
  actual_contact_to_reference_end_s=None if first is None or complete is None else complete['t']-first['t'],
  actual_contact_to_latch_s=None if first is None or latch is None else latch['time_s']-first['t'],
  first_foot_threshold_crossing=None if bad is None else {'time_s':bad['t'],'phase':bad['phase'],'joints':bad['foot_guard']['bad']},
  foot_maxima=foot,reference_completion_is_not_task_success=True)
 write(dest/'metrics.json',metrics)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 t=np.array([r['t'] for r in rows]);dt=np.diff(t);ee=np.array([r['ee_actual'] for r in rows]);target=np.array([r['ee_target'] for r in rows]);base=np.array([r['actual_base'] for r in rows])
 # Joint-mode target overrides the author's Cartesian target during parallel fold.
 # Keep raw logged targets untouched, reconstruct the reference actually applied.
 from fullcycle_geometry import URDF,transform
 robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf')
 effective=json.loads((out/'effective-config.json').read_text());cfg=json.loads((out/'effective-expert.json').read_text())
 names=ref['dof_names'];first_traverse=next((r['timing']['command_time_s'] for r in rows if r['phase']=='traverse_door'),None)
 reconstructed=[]
 for i,r in enumerate(rows):
  direct=r['phase'] in ['walk','traverse_fold_finish','turn_for_close','deploy','fold','restore_heading','final_hold'] or (r['phase']=='traverse_door' and r['timing']['command_time_s']-first_traverse>=cfg['parallel_withdraw_s'])
  if direct:
   B=r['command_base'];F=robot.fk(dict(zip(names,r['arm_command'])),transform([*B[:2],effective['robot_z']],[0,effective['robot_pitch'],B[2]]))['ee_gripper_link'];target[i]=F[:3,3];reconstructed.append(i)
 write(dest/'reference-target-audit.json',dict(reconstructed_joint_mode_steps=reconstructed,raw_log_unchanged=True,
  limitation='Raw ee_target/scripted_target fields during direct-joint prefix overrides may retain the superseded author Cartesian target. Curves use commanded joint FK for direct-joint phases and raw IK target for Cartesian phases; no actual EE pose is substituted as its own reference.'))
 actual_ee=np.r_[np.nan,np.linalg.norm(np.diff(ee,axis=0),axis=1)/dt];ref_ee=np.r_[np.nan,np.linalg.norm(np.diff(target,axis=0),axis=1)/dt]
 base_v=np.array([r['base_measured_fd_speed'] for r in rows]);cmd_v=np.array([r['action'][0] for r in rows]);cmd_w=np.array([r['action'][1] for r in rows]);actual_w=np.r_[np.nan,np.abs((np.diff(base[:,2])+math.pi)%(2*math.pi)-math.pi)/dt]
 fig,axs=plt.subplots(4,1,figsize=(12,10),sharex=True)
 axs[0].plot(t,ref_ee,label='Applied reference FD (IK target / command FK)',alpha=.7);axs[0].plot(t,actual_ee,label='Actual EE FD',lw=1);axs[0].set_ylabel('EE m/s')
 axs[1].plot(t,cmd_v,label='Joint9 forward command');axs[1].plot(t,base_v,label='Actual base speed',lw=1);axs[1].set_ylabel('Base m/s')
 axs[2].plot(t,cmd_w,label='Joint9 yaw command');axs[2].plot(t,actual_w,label='Actual |yaw rate|',lw=1);axs[2].set_ylabel('rad/s')
 axs[3].plot(t,np.degrees([r['door_q'][0] for r in rows]),label='Measured door angle');axs[3].set_ylabel('Door deg');axs[3].set_xlabel('Actual post-physics time (s)')
 for ax in axs:
  for ph in phases:
   ax.axvline(ph['first_post_s'],color='gray',alpha=.12)
  if first:ax.axvline(first['t'],color='purple',ls=':',label='First loaded closing contact' if ax==axs[0] else None)
  ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=8)
 fig.suptitle('Original physical run: reference / actual speed (no smoothing)');fig.tight_layout();fig.savefig(dest/'speeds-full.png',dpi=160)
 if first:
  for ax in axs:ax.set_xlim(first['t']-1,min(t[-1],first['t']+10))
  fig.savefig(dest/'speeds-closing.png',dpi=160)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,4.5))
 for name in foot:ax.plot(t,[r['foot_guard']['deviations'][name] for r in rows],label=name,lw=1)
 ax.axhline(.2,color='red',ls=':',label='hip stop deviation');ax.axhline(.3,color='black',ls=':',label='thigh/calf stop deviation');ax.set_xlabel('Actual post-physics time (s)');ax.set_ylabel('Deviation from nominal leg target (rad)');ax.legend(fontsize=7,ncol=4);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(dest/'feet.png',dpi=160);plt.close(fig)
 np.savez_compressed(dest/'speed-series.npz',t=t,reference_ee_fd=ref_ee,actual_ee_fd=actual_ee,command_base_v=cmd_v,actual_base_speed=base_v,command_base_yaw_rate=cmd_w,actual_abs_yaw_rate=actual_w)
 # Reference comparison and actual executed base path; never mix ideal traces with physical rows.
 from author_rewrite.geometry import DoorFrame,transform,closing_reference as old_reference
 from author_fast5.geometry import closing_reference as new_reference
 frame=DoorFrame(transform([2.61,0,.01],[0,0,math.pi/2]));old=json.loads((ROOT/'migration/wc4_act/author_rewrite/config.json').read_text());new=json.loads((out/'effective-expert.json').read_text())
 oldb=np.array([old_reference(old,frame,x)[0] for x in np.linspace(0,42,501)]);newb=np.array([new_reference(new,frame,x)[0] for x in np.linspace(0,new['close_s'],501)])
 oldee=np.array([old_reference(old,frame,x)[1] for x in np.linspace(0,42,501)]);newee=np.array([new_reference(new,frame,x)[1] for x in np.linspace(0,new['close_s'],501)])
 fig,ax=plt.subplots(figsize=(9,7));ax.plot(oldb[:,0],oldb[:,1],'--',color='gray',label='Previous closing base (42 s)');ax.plot([1.26,1.18],[.18,.10],color='gray',label='Removed micro-relocation')
 ax.plot(newb[:,0],newb[:,1],'--',color='#087d9b',lw=3,label='New closing base (%g s)'%new['close_s']);ax.plot(newee[:,0],newee[:,1],color='#bb405d',label='New EE reference');ax.plot(oldee[:,0],oldee[:,1],color='gray',alpha=.4,label='Previous EE reference')
 ax.plot(base[:,0],base[:,1],color='black',lw=1,label='Executed base path (observed only)')
 for deg in [0,45,90]:
  th=math.radians(deg);h=frame.hinge[:2];tip=h+.93*np.array([-math.sin(th),math.cos(th)]);ax.plot(*np.array([h,tip]).T,color='#b17a44');ax.text(*tip,f'{deg}°')
 ax.axvline(2.5,color='gray',ls=':');ax.set_xlim(.65,2.7);ax.set_ylim(-.8,1.0);ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_title('Old / new planned paths + actual executed base\nDoor drawings are reference poses, not measured states');fig.tight_layout();fig.savefig(dest/'old-new-top-view.png',dpi=180)
 # Reuse the completed full-decode audit from the first read-only review.
 import shutil
 previous=run/'review'
 if (previous/'video-check.json').exists():
  for name in ['video-check.json','live-keyframes.jpg']:shutil.copy2(previous/name,dest/name)
  video=out/'overview/env_0000/rollout.mp4'
  check=json.loads((dest/'video-check.json').read_text());check['sha256']=hashlib.sha256(video.read_bytes()).hexdigest();write(dest/'video-check.json',check)
  print(json.dumps({'metrics':metrics,'r11':gates['failures'],'video_frames':check['frames']},indent=2));return
 # Full decoding of original RGB, plus unmodified selected frames in a contact sheet.
 import imageio
 from PIL import Image,ImageDraw
 video=out/'overview/env_0000/rollout.mp4';reader=imageio.get_reader(str(video));meta=reader.get_meta_data();wanted=set(np.linspace(0,max(0,math.ceil(n/2)-1),12).astype(int).tolist());tiles=[];count=0
 for i,im in enumerate(reader):
  count+=1
  if i in wanted:
   tile=Image.fromarray(im).resize((768,256));ImageDraw.Draw(tile).text((8,239),f'original frame {i} / video t={i/25:.2f}s',fill='yellow');tiles.append(tile)
 reader.close();sheet=Image.new('RGB',(1536,256*math.ceil(len(tiles)/2)),(245,245,245))
 for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*768,(i//2)*256))
 sheet.save(dest/'live-keyframes.jpg',quality=92)
 write(dest/'video-check.json',dict(frames=count,fps=meta['fps'],size=meta['size'],duration_s=count/25,expected_frames=math.ceil(n/2),full_decode=True,original_execution=True))
 assert count==math.ceil(n/2) and abs(meta['fps']-25)<1e-6
 print(json.dumps({'metrics':metrics,'r11':gates['failures'],'video_frames':count},indent=2))
if __name__=='__main__':main()
