"""Render recorded expert states using the author's replay/asset implementation."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from overview import ROOT,Overview,load

def main():
    output=Path(sys.argv[1]);assert sys.argv[2]=='--'
    sys.argv=[str(ROOT/'high-level/dp/record/replay_door_dp_raw_in_isaacgym.py'),*sys.argv[3:]]
    replay=load('wc4_author_replay','high-level/dp/record/replay_door_dp_raw_in_isaacgym.py')
    original_load=replay.load_float_ik_module;original_apply=replay.apply_float_ik_state_frame
    cameras=[];errors=[]
    def load_float(mode,data):
        module=original_load(mode,data);original_create=module.create_env_actors
        def create(*args,**kwargs):
            result=original_create(*args,**kwargs);gym,sim=args[:2]
            cameras.append(Overview(gym,sim,result[0],module.gymapi,output,'RECORDED STATE REPLAY | expert episode 000000'))
            return result
        module.create_env_actors=create
        return module
    def apply(module,gym,sim,env,arm,actors,door,dof_names,data,index):
        original_apply(module,gym,sim,env,arm,actors,door,dof_names,data,index)
        # The author advances one physics tick to update the scene. Restore the
        # saved state again before rendering, so no new dynamics become evidence.
        root=np.asarray(data['replay_root_state'][index],dtype=np.float32)
        module.set_robot_base_pose(gym,env,actors,root[:2],root[2],replay.yaw_from_quat_xyzw(root[3:7]),pitch=replay.pitch_from_quat_xyzw(root[3:7]))
        replay.set_float_ik_arm_dofs(module,gym,env,arm,dof_names,data['replay_dof_pos'][index],data['replay_dof_vel'][index])
        replay.set_float_actor_root_pose(module,gym,env,door,data['replay_door_root_state'][index])
        replay.set_float_ik_door_dofs(module,gym,env,door,data['replay_door_dof_pos'][index],data['replay_door_dof_vel'][index])
        measured=gym.get_actor_dof_states(env,arm,module.gymapi.STATE_ALL)['pos']
        indices=[module.FLOAT_ARM_TO_DP_DOF[n] for n in dof_names]
        expected=np.asarray(data['replay_dof_pos'][index])[indices]
        measured_door=gym.get_actor_dof_states(env,door,module.gymapi.STATE_ALL)['pos']
        errors.append({'frame':int(index),'arm_q_error':float(np.max(np.abs(measured-expected))),
            'door_q_error':float(np.max(np.abs(measured_door-data['replay_door_dof_pos'][index])))})
    def render(module,gym,sim,env,arm,actors,viewer,args,data,index,action,vision):
        degree=float(np.rad2deg(data['replay_door_dof_pos'][index,0]))
        phases=list(data['phase_names']);phase=phases[int(np.asarray(data['subtask_index'][index]).reshape(-1)[0])]
        cameras[0].capture(index,f'{phase} | hinge={degree:.1f}deg',render=True)
        return True
    replay.load_float_ik_module=load_float;replay.apply_float_ik_state_frame=apply;replay.render_float_ik_state_frame=render
    try:replay.main()
    finally:
        for camera in cameras:camera.close()
    episode=Path(sys.argv[sys.argv.index('--raw_episode')+1])
    receipt={'source_npz':str(episode),'source_sha256':hashlib.sha256(episode.read_bytes()).hexdigest(),
        'frames':len(errors),'restored_state_errors':errors,'new_expert_collection':False,'new_policy_evaluation':False,
        'meaning':'Recorded-state visualization; no new success statistic. Original RGB was not recorded during collection.'}
    (output/'replay-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert len(errors)==550
    assert max(v['arm_q_error'] for v in errors)<1e-6 and max(v['door_q_error'] for v in errors)<1e-6
    print(json.dumps({'frames':len(errors),'max_arm_q_error':max(v['arm_q_error'] for v in errors),'max_door_q_error':max(v['door_q_error'] for v in errors),'video':str(output/'rollout.mp4')}))

if __name__=='__main__':main()
