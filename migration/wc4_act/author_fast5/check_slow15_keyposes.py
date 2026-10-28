import json,math,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
root=Path('/data/zyf/video2door-wc4-act');sys.path.insert(0,str(root/'migration/wc4_act'))
from fullcycle_geometry import URDF,transform
from fullcycle_collision import collision_pieces,signed_gap
from author_fast5.geometry import DoorFrame
run=root/'.repro/runs/wc4-fast5-repair-20261028/attempt_000';dest=root/'.repro/reviews/wc4-slow15-20261028-check'
rows=[json.loads(l) for l in (run/'states.jsonl').open()];slow=[json.loads(l) for l in (dest/'timing-only/trace.jsonl').open() if json.loads(l)['phase']=='close']
robot=URDF(root/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');frame=DoorFrame(transform([2.61,0,.01],[0,0,math.pi/2]))
R=collision_pieces(robot.path);D=collision_pieces(frame.door.path);slab=next(p for p in D if p.link=='door_panel' and p.name=='door_slab')
pieces=[p for p in R if p.link in ['wrist_down_mount','wrist_up_mount'] or (p.link=='gripperStator' and p.name=='2')]
mp=json.loads((run/'body-map.json').read_text());arm_names=[v['body'] for k,v in sorted(mp.items(),key=lambda x:int(x[0])) if v['actor_handle']==1]
def actual_pose(v):
 T=np.eye(4);T[:3,3]=v[0][0];T[:3,:3]=Rotation.from_quat(v[0][1]).as_matrix();return T
def geometry(poses,theta):
 dp=frame.door.fk({'door_hinge_joint':theta},frame.actor);door=slab.world(dp['door_panel']);n=np.array([math.cos(theta),math.sin(theta),0]);t=np.array([-math.sin(theta),math.cos(theta),0]);out={}
 for p in pieces:
  a=p.world(poses[p.link]);r=(a[0]-frame.hinge)@t;out[p.link+'/'+p.name]=dict(slab_gap_m=signed_gap(a,door),normal_max_m=float(((a[0]-frame.hinge)@n).max()),radial_range_m=[float(min(r)),float(max(r))])
 return out
results=[]
for t in [35.68,36.50,37.46,38.06]:
 old=min(rows,key=lambda r:abs(r['t']-t));now=min(slow,key=lambda r:abs(r['t']-3*old['expert_target']['reference_elapsed_s']))
 poses={name:actual_pose(v) for name,v in zip(arm_names,old['arm_body_states'])}
 q=dict(zip(['joint%d'%i for i in range(1,7)],now['q']));q['jointGripper']=-1.1775;ideal=robot.fk(q,frame.base_transform(now['base']))
 pe=float(np.linalg.norm(np.array(old['ee_actual'])-old['ee_target']));oe=float(Rotation.from_matrix(Rotation.from_quat(old['ee_target_quat']).as_matrix()@Rotation.from_quat(old['ee_quat']).as_matrix().T).magnitude())
 results.append(dict(prior_time_s=old['t'],prior_ref_elapsed_s=old['expert_target']['reference_elapsed_s'],prior_actual_door_deg=math.degrees(old['door_q'][0]),reference_angle_deg=math.degrees(now['theta']),
  old_actual_position_error_m=pe,old_actual_orientation_error_rad=oe,old_measured_geometry=geometry(poses,old['door_q'][0]),
  new_ideal_elapsed_s=now['t'],new_ideal_position_error_m=now['position_error_m'],new_ideal_orientation_error_rad=now['orientation_error_rad'],new_ideal_joint_margin_rad=now['joint_margin_rad'],
  new_ideal_reference_door_geometry=geometry(ideal,now['theta']),new_ideal_prior_door_stress_geometry=geometry(ideal,old['door_q'][0])))
text=dict(keyposes=results,scope='Four measured wrist-contact-neighborhood poses only. New DLS is ideal tracking, not predicted physical response. Prior-angle stress geometry is not a prediction of the 15s door motion. Convex slab/wrist bounds; no contact-force threshold.')
(dest/'keypose-check.json').write_text(json.dumps(text,indent=2)+'\n');print(json.dumps(text,indent=2))
