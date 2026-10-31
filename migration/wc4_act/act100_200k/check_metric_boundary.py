"""Real author checkpoint after a stop requested inside a toy update; ACT0."""
import sys,tempfile,json
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act/r13'))
import torch
from training_guard import TrainingGuard
from stopping import BoundaryStop,request_stop
from test_stopping import Toy,Config,Processor
from lerobot.utils.train_utils import save_checkpoint
from metric_boundary import MetricBoundary
with tempfile.TemporaryDirectory() as d:
 p=Path(d);model=Toy();opt=torch.optim.AdamW(model.parameters(),lr=1e-3);rows=[]
 def update(metrics,policy,batch,optimizer,**kwargs):
  optimizer.zero_grad();loss=(policy.w**2).sum();loss.backward();optimizer.step()
  request_stop(p,'stop_after_real_optimizer_step_before_metrics')
  return float(loss)
 m=SimpleNamespace(update_policy=update,save_checkpoint=save_checkpoint,make_pre_post_processors=lambda:(Processor('pre'),Processor('post')))
 guard=TrainingGuard(m,p,lambda:Config());boundary=MetricBoundary(guard.boundary);m.make_pre_post_processors()
 result=m.update_policy(None,model,{},opt,accelerator=SimpleNamespace(unwrap_model=lambda x:x))
 assert guard.boundary.completed==1 and not guard.boundary.inflight and not (p/'STOPPED.json').exists()
 rows.append(dict(update=1,loss=result))
 try:boundary.after_metrics(1)
 except BoundaryStop:pass
 else:raise AssertionError('Safe stop was not honored after original metrics')
 stopped=json.loads((p/'STOPPED.json').read_text());assert stopped['completed_updates']==1 and rows[0]['update']==1
 guard.boundary.close()
 print(json.dumps(dict(passed=True,metrics_logged_before_stop=True,full_checkpoint_at_exact_completed_boundary=True,ACT_training_updates=0,simulation=False)))
