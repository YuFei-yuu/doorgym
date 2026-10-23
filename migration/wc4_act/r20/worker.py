"""Original RPC worker with passive wall-time evidence; no action/queue changes."""
import sys,time,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'high-level/dp'))
import door_action_evidence
Original=door_action_evidence.WorkerActionEvidence
class TimedEvidence(Original):
 def __init__(self,controller):
  if hasattr(controller.backend,'policy'):
   import torch
   from safetensors.torch import load_file
   from door_policy_backend import load_lerobot_policy_normalizer_stats
   stage=int(os.environ['WC4_R20_STAGE']);assert stage in [75000]
   official=Path(os.environ['WC4_EXECUTION_RUN'])/'train/checkpoints'/f'{stage:06d}'/'pretrained_model'
   weights=load_file(str(official/'model.safetensors'),device='cpu')
   actual=controller.backend.policy.state_dict();assert set(actual)==set(weights) and len(weights)==234
   for k,v in actual.items():assert torch.equal(v.cpu(),weights[k]),k
   stats=load_lerobot_policy_normalizer_stats(official);compared=0
   for key,values in stats.items():
    for name,value in values.items():
     assert key in controller.backend.stats and name in controller.backend.stats[key]
     assert torch.equal(value.cpu(),controller.backend.stats[key][name].cpu()),(key,name)
     compared+=1
   assert controller.action_horizon==25 and controller.pred_horizon==100 and controller.obs_horizon==1 and controller.action_dim==9
   output=Path(os.environ['WC4_R20_TRIAL_OUTPUT'])
   (output/'worker-model-binding.json').write_text(json.dumps(dict(model_weights_equal=234,normalizer_tensors_equal=compared,official=str(official),execution_horizon=25,chunk=100,extra_forward_calls=0,policy_device=str(controller.backend.device),worker_python=sys.executable,torch=torch.__version__),indent=2)+'\n')
   del weights
  original=controller.sample_action_chunk
  self.sample_wall_s=None
  def timed(*a,**k):
   started=time.perf_counter();result=original(*a,**k);self.sample_wall_s=time.perf_counter()-started
   return result
  controller.sample_action_chunk=timed
  super().__init__(controller)
 def act(self,*a,**k):
  started=time.perf_counter();result=super().act(*a,**k)
  self.last['worker_act_wall_s']=time.perf_counter()-started
  self.last['worker_sample_wall_s']=self.sample_wall_s if self.last['new_chunk'] else 0.
  return result
door_action_evidence.WorkerActionEvidence=TimedEvidence
if __name__=='__main__':
 import door_policy_worker
 door_policy_worker.main()
