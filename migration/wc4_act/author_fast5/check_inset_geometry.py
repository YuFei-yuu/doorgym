"""A small source-asset/IK feasibility check for the explicitly requested inset."""
import argparse,json,math,sys,os,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from fullcycle_geometry import URDF,transform
from fullcycle_collision import collision_pieces,signed_gap
from author_fast5.geometry import DoorFrame,closing_reference

def main():
 p=argparse.ArgumentParser();p.add_argument('--check',type=Path,required=True);a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 cfg=json.loads(Path(__file__).with_name('config.json').read_text());old=json.loads(subprocess.check_output(['git','show','d79e086:migration/wc4_act/author_fast5/config.json'],cwd=ROOT,text=True))
 frame=DoorFrame(transform([2.61,0,.01],[0,0,math.pi/2]));robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf',include_visual=True)
 rows=[json.loads(l) for l in (a.check/'inset/trace.jsonl').open()];close=[r for r in rows if r['phase']=='close']
 R=collision_pieces(robot.path);D=collision_pieces(frame.door.path)
 legs={s+j:v for s in ['FL','FR','RL','RR'] for j,v in [('_hip_joint',0),('_thigh_joint',.8),('_calf_joint',-1.5)]}
 # Ray intersection with the actual slab mesh at the intended radial/height point.
 mesh=frame.door.path.parent/'assets/meshes/door_slab.obj';vertices=[];faces=[]
 for line in mesh.read_text().splitlines():
  fields=line.split()
  if fields and fields[0]=='v':vertices.append(list(map(float,fields[1:4])))
  if fields and fields[0]=='f':
   ids=[int(x.split('/')[0])-1 for x in fields[1:]]
   faces.extend([(ids[0],ids[i],ids[i+1]) for i in range(1,len(ids)-1)])
 V=np.array(vertices)
 def slab_hits(radius,z):
  hits=[]
  for face in faces:
   tri=V[list(face)];M=np.column_stack((tri[1,[0,2]]-tri[0,[0,2]],tri[2,[0,2]]-tri[0,[0,2]]))
   if abs(np.linalg.det(M))<1e-12:continue
   w=np.linalg.solve(M,np.array([radius,z])-tri[0,[0,2]])
   if min(w)>=-1e-8 and sum(w)<=1+1e-8:hits.append(float(tri[0,1]+w@(tri[1:,1]-tri[0,1])))
  return sorted(set(round(x,9) for x in hits))
 result=[]
 for deg in [90,75,70,65,60,55,50,35,0]:
  r=min(close,key=lambda r:abs(math.degrees(r['theta'])-deg));theta=r['theta']
  q=dict(legs,**dict(zip(['joint%d'%i for i in range(1,7)],r['q'])));q['jointGripper']=-1.1775
  poses=robot.fk(q,frame.base_transform(r['base']));dp=frame.door.fk({'door_hinge_joint':theta,'handle_spindle_joint':0},frame.actor)
  minimum={};nonend=[p for p in R if p.link not in ['gripperStator','gripperMover']]
  for group,pieces in [('wrist',[p for p in R if p.link in ['wrist_down_mount','wrist_up_mount']]),('other_robot',nonend),('fixed_finger',[p for p in R if p.link=='gripperStator' and p.name=='2'])]:
   best=(1e9,None)
   for rp in pieces:
    aa=rp.world(poses[rp.link])
    for dd in D:
     bb=dd.world(dp[dd.link]);broad=float(np.maximum(bb[0].min(0)-aa[0].max(0),aa[0].min(0)-bb[0].max(0)).max())
     if broad>best[0]:continue
     gap=signed_gap(aa,bb) if broad<.05 else broad
     if gap<best[0]:best=(gap,[rp.link,rp.name,dd.link,dd.name])
   minimum[group]={'gap_m':best[0],'pair':best[1]}
  _,_,pad,_,_=frame.contact(cfg,theta);tangent=frame.R@np.array([-math.sin(theta),math.cos(theta),0.]);radius=float((pad-frame.hinge)@tangent)
  hits=slab_hits(radius,cfg['contact_height_above_hinge_m'])
  result.append(dict(reference_angle_deg=math.degrees(theta),t=r['t'],radius_m=radius,slab_local_y_hits=hits,
   contact_is_on_existing_slab_face=bool(hits and abs(max(hits)-frame.slab_hi[1])<1e-6),minimums=minimum,
   ik_position_error_m=r['position_error_m'],ik_orientation_error_rad=r['orientation_error_rad'],joint_margin_rad=r['joint_margin_rad']))
 data=dict(poses=result,spatial_change='Reference-angle 70→50 degrees: fixed-finger contact radius .905→.805 m; same base, height, orientation and 15s clock.',
  scope='Nine fixed reference poses. Source mesh ray checks establish intended point on the slab. Discrete convex external checks and ideal DLS do not prove physical tracking, continuous collision freedom, or foot quality.')
 (a.check/'inset-geometry.json').write_text(json.dumps(data,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(12,6));times=np.linspace(0,cfg['close_s'],601)
 for c,label,col in [(old,'Previous edge path','#999999'),(cfg,'Inset path','#087c9b')]:
  ref=[closing_reference(c,frame,t) for t in times];P=np.array([x[4] for x in ref]);B=np.array([x[0] for x in ref]);rad=[np.linalg.norm((x[4]-frame.hinge)[:2]) for x in ref]
  axs[0].plot(P[:,0],P[:,1],color=col,lw=2,label=label);axs[1].plot(times,rad,color=col,label=label)
 axs[0].plot(B[:,0],B[:,1],'--',color='#41754a',label='Base XY unchanged')
 H=frame.hinge[:2]
 for deg in [0,50,70,90]:
  th=math.radians(deg);tip=H+frame.slab_hi[0]*np.array([-math.sin(th),math.cos(th)]);axs[0].plot(*np.array([H,tip]).T,color='#b17a44',alpha=.5);axs[0].text(*tip,f'{deg}°')
 axs[0].set_aspect('equal');axs[0].set_xlabel('World x (m)');axs[0].set_ylabel('World y (m)');axs[1].set_xlabel('Closing reference time (s)');axs[1].set_ylabel('Intended contact distance from hinge (m)')
 for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Planned fixed-finger contact moves inward by 0.10 m — NOT a rollout');fig.tight_layout();fig.savefig(a.check/'inset-before-after.png',dpi=180)
 print(json.dumps(data,indent=2))
if __name__=='__main__':main()
