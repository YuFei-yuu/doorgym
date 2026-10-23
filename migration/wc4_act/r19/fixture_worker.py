"""CPU engineering replacement for backend only; production controller/worker stay real.

No ACT/DP model is instantiated, no checkpoint is read, no policy inference runs.
"""
import sys,os,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'high-level/dp'))
import torch
import door_policy_backend as backend


class FixtureBackend:
    device=torch.device('cpu');vision_mode='depth_only';action_frame='world'
    image_keys=['observation.images.wrist_masked_depth','observation.images.front_masked_depth']
    obs_horizon=1;horizon=100;action_horizon=25;action_dim=9;uses_observation_sequence=False
    sidecar_config={'action':['vx','yaw']+['joint%d'%i for i in range(1,7)]+['jointGripper']}
    def __init__(self):self.calls=0
    def metadata(self):return {'policy_config':{}}
    def predict_action_chunks_from_batch(self,batch,noise=None):
        self.calls+=1
        # Unique, exactly representable identifiers for every chunk and row.
        rows=torch.zeros(1,100,9,dtype=torch.float32)
        rows[0,:,0]=torch.arange(100,dtype=torch.float32)+self.calls*1000
        rows[0,:,1]=batch['observation.state'][0,0]
        with open(os.environ['R13_FIXTURE_BACKEND_EVENTS'],'a') as f:
            f.write(json.dumps(dict(event='fixture_backend_call',call=self.calls,pid=os.getpid(),
                observation=float(batch['observation.state'][0,0]),real_policy_inference=False))+'\n')
        return rows


backend.load_door_policy_backend=lambda *a,**k:FixtureBackend()
# Import production worker after replacing ONLY the backend factory. Its real
# framing, command dispatch, controller preprocessing, sampler and queue execute.
sys.path.insert(0,str(Path(__file__).parent))
import worker as timed_worker
import door_policy_worker
if __name__=='__main__':door_policy_worker.main()
