"""Render stored R14 body states. No policy call or physics integration is allowed."""
import argparse,hashlib,json,math,os,sys,time
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
  def __init__(self,g):self.g=g
  def __getattr__(self,k):return getattr(self.g,k)
  def simulate(self,*a,**k):forbidden()
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
     # The floating base and arm share an articulation. Restore the entire
     # environment in one native call rather than incomplete actor segments.
     body=gym.get_env_rigid_body_states(st.env,ga.STATE_ALL)
     assert body.dtype.itemsize==52
     target_all=body.view(np.float32).reshape(len(body),13).copy()
     for actor,key in [(st.actor_handles[0],'base_body_states'),(st.arm_actor,'arm_body_states'),(st.door_actor,'door_body_states')]:
      target=np.asarray([flatten(v) for v in row[key]],dtype=np.float32)
      ids=[gym.get_actor_rigid_body_index(st.env,actor,i,ga.DOMAIN_ENV) for i in range(len(target))]
      target_all[ids]=target
     body.view(np.float32).reshape(len(body),13)[:]=target_all
     gym.set_env_rigid_body_states(st.env,body,ga.STATE_POS)
     actual=gym.get_env_rigid_body_states(st.env,ga.STATE_ALL).view(np.float32).reshape(len(body),13)
     frame_error=float(np.max(np.abs(actual[:,:7]-target_all[:,:7])))
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
