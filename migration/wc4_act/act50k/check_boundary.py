"""No ACT/optimizer/simulation: regression for a post-update evidence exception."""
import sys,tempfile,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act/r13'))
from stopping import StopBoundary,BoundaryStop,request_stop
with tempfile.TemporaryDirectory() as d:
 root=Path(d);saves=[]
 def save(n):saves.append(n);return dict(step=n,test_fixture=True)
 boundary=StopBoundary(root,save)
 # Author operation returns, then the completed boundary publishes; a later
 # evidence exception does not invalidate that already-completed state.
 result=boundary.update(lambda:dict(loss=1))
 assert boundary.completed==1 and not boundary.inflight and result==dict(loss=1)
 try:raise AssertionError('synthetic post-update source audit failure')
 except AssertionError:
  request_stop(root,'synthetic audit failure')
  try:boundary.checkpoint_if_requested()
  except BoundaryStop:pass
 assert saves==[1]
 receipt=json.loads((root/'STOPPED.json').read_text())
 assert receipt['completed_updates']==1 and receipt['uncertain_updates']==0
 boundary.close()
 print(json.dumps(dict(passed=True,post_update_audit_exception_saves_completed_boundary=True,ACT_constructed=False,optimizer_constructed=False,simulation=False)))
