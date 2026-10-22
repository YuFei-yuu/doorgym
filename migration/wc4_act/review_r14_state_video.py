"""Render stored R14 body states. No policy call or physics integration is allowed."""
import argparse,hashlib,json,math,os,sys,time,copy
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def flatten(v):
 if isinstance(v,(list,tuple)):
  return [x for item in v for x in flatten(item)]
 return [float(v)]

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 source=a.source;command=json.loads((source/'command.json').read_text())
 frozen=json.loads((source/'frozen-trial.json').read_text());seed=frozen['seed']
 provenance={n:sha(source/n) for n in ['states.jsonl','r11-score.json','frozen-trial.json','command.json']}
 sys.path[:0]=[str(ROOT/'migration/wc4_act'),str(ROOT/'high-level/float_ik'),str(ROOT/'high-level/dp')]
 from overview import load,Overview
 import numpy as np
 m=load('r14_review_author','high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
 def forbidden(*args,**kwargs):raise RuntimeError('State visualization cannot run physics, expert motion, or policy')
 acquire=m.gymapi.acquire_gym
 class RenderOnlyGym:
  def __init__(self,g):self.g=g;self.asset_sources={}
  def __getattr__(self,k):return getattr(self.g,k)
  def simulate(self,*a,**k):forbidden()
  def load_asset(self,sim,root,file,options):
   asset=self.g.load_asset(sim,root,file,options)
   self.asset_sources[asset]=(Path(root)/file,options)
   return asset
 m.gymapi.acquire_gym=lambda:RenderOnlyGym(acquire())
 m.trajectory_targets=forbidden;m.update_arm_ik_targets_for_env=forbidden
 m.enforce_locked_door_hinge=forbidden;m.dc.enforce_locked_door_hinge=forbidden
 original_make=m.make_env_args
 def make(args,index):
  env=original_make(args,index)
  assert json.loads(env.ikpush_randomization_json)==frozen['sampling_before_asset_overrides']
  env._camera_axis_local_poses={k:{n:np.array(v,dtype=np.float32) for n,v in pose.items()} for k,pose in frozen['camera_local_poses'].items()}
  return env
 m.make_env_args=make
 def create_visual_scene(gym,sim,*unused,**kwargs):
  from fullcycle_geometry import origin
  body_map=json.loads((source/'body-map.json').read_text())
  groups={}
  for v in body_map.values():groups.setdefault(v['actor_handle'],[]).append(v['body'])
  env=gym.create_env(sim,m.gymapi.Vec3(-10,-10,-1),m.gymapi.Vec3(10,10,5),1)
  assets=a.output/'visual-assets';assets.mkdir()
  clones=[];geometry=[]
  for actor_id,names in groups.items():
   if actor_id not in [0,1,2]:continue  # Static side walls are rebuilt below.
   matches=[(asset,path,opt) for asset,(path,opt) in list(gym.asset_sources.items()) if set(gym.get_asset_rigid_body_names(asset))==set(names)]
   assert len(matches)==1,('Ambiguous source visual asset',actor_id,names)
   _,path,opt=matches[0];xml=ET.parse(path).getroot();links={l.get('name'):l for l in xml.findall('link')}
   parents={j.find('child').get('link'):j for j in xml.findall('joint')}
   visuals={n:[] for n in names}
   for link_name,link in links.items():
    for visual in link.findall('visual'):
     owner=link_name;T=np.eye(4)
     while owner not in visuals:
      joint=parents[owner];assert joint.get('type')=='fixed',('Unmapped moving visual',owner)
      T=origin(joint)@T;owner=joint.find('parent').get('link')
     v=copy.deepcopy(visual);T=T@origin(v)
     old=v.find('origin')
     if old is not None:v.remove(old)
     R=T[:3,:3];pitch=math.asin(max(-1.,min(1.,-R[2,0])))
     if abs(math.cos(pitch))>1e-8:roll=math.atan2(R[2,1],R[2,2]);yaw=math.atan2(R[1,0],R[0,0])
     else:roll=0.;yaw=math.atan2(-R[0,1],R[1,1])
     ET.SubElement(v,'origin',xyz=' '.join(map(str,T[:3,3])),rpy=' '.join(map(str,[roll,pitch,yaw])))
     for node in list(v.iter('mesh'))+list(v.iter('texture')):
      filename=node.get('filename')
      if filename and not Path(filename).is_absolute():node.set('filename',str((path.parent/filename).resolve()))
     visuals[owner].append(v)
   key={0:'base_body_states',1:'arm_body_states',2:'door_body_states'}[actor_id]
   for index,name in enumerate(names):
    robot=ET.Element('robot',name='recorded_visual')
    for material in xml.findall('material'):robot.append(copy.deepcopy(material))
    link=ET.SubElement(robot,'link',name=name)
    for visual in visuals[name]:link.append(visual)
    # Each stored body is an independent visual actor, with no articulated FK.
    inertial=ET.SubElement(link,'inertial');ET.SubElement(inertial,'mass',value='1')
    ET.SubElement(inertial,'inertia',ixx='1',iyy='1',izz='1',ixy='0',ixz='0',iyz='0')
    file=assets/('%d-%02d.urdf'%(actor_id,index));ET.ElementTree(robot).write(file)
    options=m.gymapi.AssetOptions();options.disable_gravity=True;options.collapse_fixed_joints=False
    options.flip_visual_attachments=opt.flip_visual_attachments
    asset=gym.load_asset(sim,str(assets),file.name,options)
    handle=gym.create_actor(env,asset,m.gymapi.Transform(),'stored-body-%d-%d'%(actor_id,index),0,0)
    clones.append((handle,key,index))
   geometry.append(dict(original_asset_sha256=sha(path),body_names=names,visuals=sum(map(len,visuals.values()))))
  env_args=m.make_env_args(unused[-1],0)
  door=m.clone_door_runtime(unused[2][0])
  walls=m.dc.create_door_side_walls(gym,sim,env,door,env_args,0)
  assert len(walls)==2
  scene=json.loads((source/'scene-frame.json').read_text())
  assert abs(env_args.door_x+env_args.door_wall_x_offset-scene['wall_plane_x'])<1e-6
  gym.prepare_sim(sim)
  (a.output/'geometry-provenance.json').write_text(json.dumps(geometry,indent=2)+'\n')
  return [SimpleNamespace(env=env,clones=clones)],'render-only'
 m.create_parallel_env_states=create_visual_scene
 def render(gym,sim,states,viewer,args,dt,dof_names):
  assert len(states)==1
  st=states[0];ga=m.gymapi
  result=json.loads((source/'result.json').read_text())
  cam=Overview(gym,sim,st.env,ga,a.output/'overview','RECORDED STATE REPLAY | seed '+str(seed)+' | NOT NEW EXECUTION')
  cam.expected=(result['steps']+1)//2
  errors=[];start=time.time()
  try:
   with (source/'states.jsonl').open() as f:
    for line in f:
     row=json.loads(line)
     if row['step']%2:continue
     frame_error=0.
     for actor,key,index in st.clones:
      target=np.asarray(flatten(row[key][index]),dtype=np.float32)[:7]
      body=gym.get_actor_rigid_body_states(st.env,actor,ga.STATE_ALL)
      body.view(np.float32).reshape(1,13)[0,:7]=target
      gym.set_actor_rigid_body_states(st.env,actor,body,ga.STATE_POS)
      actual=gym.get_actor_rigid_body_states(st.env,actor,ga.STATE_ALL).view(np.float32).reshape(1,13)[0,:7]
      frame_error=max(frame_error,float(np.max(np.abs(actual-target))))
     assert frame_error<1e-5,('Stored body restore mismatch',row['step'],frame_error)
     angle=math.degrees(row['door_q'][0]);locked=row['latched'] is not None and not row['unlocked']
     cam.capture(row['step'],f'R14 030000 | door={angle:.3f}deg | measured_latched={locked}',render=True,time_s=row['t'])
     errors.append(frame_error)
     if cam.count%500==0:print(json.dumps(dict(frames=cam.count,target=cam.expected,elapsed_s=time.time()-start)),flush=True)
  finally:cam.close()
  assert cam.count==cam.expected
  assert all(sha(source/n)==digest for n,digest in provenance.items())
  receipt=dict(source=str(source),seed=seed,checkpoint='030000',meaning='Stored-body-pose RGB visualization, not original live RGB, new policy evaluation or proof of contact forces',source_sha256=provenance,frames=cam.count,fps=25,source_time_range_s=[cam.times[0],cam.times[-1]],max_body_pose_restore_error=max(errors),restored_fields='Recorded world positions/quaternions for all bodies; velocities are not replayed or integrated',physics_integration_steps=0,policy_forward_calls=0,optimizer_updates=0,additional_evaluation_episodes=0,render_only_scenes=1,source_scoring_unchanged=True,video=str(a.output/'overview/rollout.mp4'))
  (a.output/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
 m.run_parallel_demo=render
 argv=command['argv'].copy();sys.argv=argv
 (a.output/'render-command.json').write_text(json.dumps(dict(argv=argv,renderer=str(Path(__file__).resolve()),environment={k:os.environ.get(k) for k in ['CONDA_PREFIX','CUDA_VISIBLE_DEVICES','VIDEO2DOOR_GPU','VIDEO2DOOR_VK_DEVICE_INDEX']}),indent=2)+'\n')
 m.main()

if __name__=='__main__':main()
