"""Original RPC worker with passive wall-time evidence; no action/queue changes."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'high-level/dp'))
import door_action_evidence
Original=door_action_evidence.WorkerActionEvidence
class TimedEvidence(Original):
 def __init__(self,controller):
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
