"""Actual Gym Python -> ACT GPU worker using fixed recorded inputs; no simulator."""
import json
import os
from pathlib import Path
import sys
import isaacgym
import numpy as np
import torch

from common import ROOT,RUN
R=RUN/'checkpoints'/f'{int(sys.argv[1]):06d}'
sys.path.insert(0,str(ROOT/'high-level/dp'))
from door_policy_backend import DoorPolicySubprocessController
from door_dp_common import sanitize_door_dp_action
reference=torch.load(R/'export-reference.pt',map_location='cpu')
batch=reference['batch'];expected=reference['expected_actions'].numpy()
names=['vx','yaw','joint1','joint2','joint3','joint4','joint5','joint6','jointGripper']
expected=sanitize_door_dp_action(expected,action_names=names,eps=1e-5)
controller=DoorPolicySubprocessController(R/'door-checkpoint/model_latest.pt',device='cuda:0',action_horizon=25)
try:
    assert controller.action_names==names and controller.action_dim==9 and controller.pred_horizon==100 and controller.action_horizon==25
    for i in range(len(expected)):
        wrist=np.rint(batch['observation.images.wrist_masked_depth'][i].permute(1,2,0).numpy()*255).astype(np.uint8)
        front=np.rint(batch['observation.images.front_masked_depth'][i].permute(1,2,0).numpy()*255).astype(np.uint8)
        mask=np.zeros_like(wrist)
        controller.append_observation_for_env(i,batch['observation.state'][i].numpy(),mask,wrist,mask,front)
    got=controller.predict_action_chunks_for_envs(list(range(len(expected))))
    assert got.shape==expected.shape==(5,100,9) and np.isfinite(got).all()
    error=float(np.max(np.abs(got-expected)))
    assert np.allclose(got,expected,atol=1e-3,rtol=1e-3),error
    result={'status':'passed','parent_python':sys.executable,'worker_python':os.environ['DOOR_DP_LEROBOT_PYTHON'],'device':str(controller.device),'action_names':controller.action_names,'action_dim':controller.action_dim,'pred_horizon':controller.pred_horizon,'action_horizon':controller.action_horizon,'physical_action_max_error_cpu_vs_gpu':error,'tolerance':{'atol':1e-3,'rtol':1e-3},'simulation':False}
    (R/'gpu-bridge-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
finally:controller.close()
