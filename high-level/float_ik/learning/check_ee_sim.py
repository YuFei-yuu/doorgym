"""Bounded simulator contract probe, not training or a task-success evaluation."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from float_env import FloatEnv, demo  # Isaac Gym must precede torch.
import torch
from policy import DoorDiffusionPolicy, Policy
from schema import EE_SCHEMA, contract, model_config

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output)
if out.exists():raise FileExistsError(out)
torch.set_num_threads(2)
checks={}
env=FloatEnv(expert=False, schema=EE_SCHEMA)
original=demo.trajectory_targets
def forbidden_planner(*args, **kwargs):
    raise AssertionError('Policy invoked scripted expert planning')
demo.trajectory_targets=forbidden_planner
try:
    obs=env.reset(7001)
    initial={k:v.copy() for k,v in obs.items() if isinstance(v,np.ndarray)}
    try:env.expert_action()
    except RuntimeError:checks['expert_action_guard']=True
    else:raise AssertionError('Expert action allowed')
    ep,eq=demo.get_body_pose(env.gym,env.env,env.arm,env.ee_index)
    target=ep+np.array([-.02,0,.02])
    initial_error=float(np.linalg.norm(target-ep))
    deltas=[]
    for i in range(60):
        action=np.r_[0.,0.,0.,target,eq,env.args.gripper_open]
        obs,info,_,applied=env.step(action)
        np.testing.assert_allclose(obs['state'][-11:],applied,atol=1e-7)
        deltas.append(info['ik_joint_delta_max_rad'])
    checks['pose_feedback_reduces_error']=info['ee_command_pos_error_m'] < initial_error*.5
    checks['ik_increment_bound']=max(deltas)<=env.args.ik_max_step+1e-5
    checks['no_scripted_phase']=env.phase is None
    again=env.reset(7001)
    checks['same_seed_reset_state']=bool(np.allclose(again['state'],initial['state'],atol=1e-6))
    checks['same_seed_reset_images']=bool(np.array_equal(again['images'],initial['images']))
    names,states=contract(EE_SCHEMA);config=model_config(EE_SCHEMA)
    torch.manual_seed(123)
    ck=dict(schema=EE_SCHEMA,model_config=config,action_names=names,state_names=states,model=DoorDiffusionPolicy(**config).state_dict(),step=0,stats=dict(state_mean=torch.zeros(39),state_std=torch.ones(39),action_mean=torch.zeros(11),action_std=torch.ones(11)))
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'untrained.pt';torch.save(ck,path)
        try:Policy(path,device='cpu',schema='unsupported_schema')
        except ValueError:checks['checkpoint_mismatch_rejected']=True
        controller=Policy(path,schema=EE_SCHEMA,seed=7001)
        for i in range(12):
            action,_=controller.act(again)
            again,info,_,applied=env.step(action)
            np.testing.assert_allclose(again['state'][-11:],applied)
            assert info['ik_joint_delta_max_rad']<=env.args.ik_max_step+1e-5
        checks['network_to_ik_to_servo_finite']=bool(np.isfinite(again['state']).all())
        checks['world_target_preserved']=bool(np.array_equal(applied[3:6],action[3:6]))
        checks['unit_target_quaternion']=bool(abs(np.linalg.norm(applied[6:10])-1)<1e-6)
    result=dict(checks=checks,passed=all(checks.values()),initial_error_m=initial_error,max_pose_probe_ik_increment_rad=max(deltas),meaning='Untrained model used only for interface verification; no optimizer updates or success-rate claim')
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    assert result['passed'],result
finally:
    demo.trajectory_targets=original
    env.close()
