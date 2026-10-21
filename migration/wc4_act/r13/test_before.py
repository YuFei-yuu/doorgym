"""Reproduce R12 proxy-vs-worker discrepancy with historical code and CPU backend."""
import os,sys,json,subprocess,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];DP=ROOT/'high-level/dp'
sys.path.insert(0,str(DP))
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
for name in ['door_policy_backend.py','door_policy_worker.py']:
 (out/name).write_bytes(subprocess.check_output(['git','show','3c713de:high-level/dp/'+name],cwd=ROOT))
harness=out/'before_worker.py'
harness.write_text('''import sys,runpy,json,os
sys.path[:0]=['''+repr(str(out))+','+repr(str(DP))+''']
import door_policy_backend as backend
import door_policy_worker
original=backend.DoorPolicyController.act
def observed(self,*a,**k):
 action=original(self,*a,**k)
 with open(os.environ['R13_BEFORE_QUEUE'],'a') as f:f.write(json.dumps(dict(real_worker_queue_after=len(self.action_queue)))+'\\n')
 return action
backend.DoorPolicyController.act=observed
runpy.run_path('''+repr(str(ROOT/'migration/wc4_act/r13/fixture_worker.py'))+''',run_name='__main__')
''')
spec=importlib.util.spec_from_file_location('before_backend',out/'door_policy_backend.py');backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
os.environ['R13_BEFORE_QUEUE']=str(out/'actual-worker-queue.jsonl');os.environ['R13_FIXTURE_BACKEND_EVENTS']=str(out/'backend-events.jsonl')
backend._worker_python_command=lambda cp:[os.environ['DOOR_DP_LEROBOT_PYTHON'],'-B',str(harness)]
c=backend.DoorPolicySubprocessController('fixture-no-checkpoint',device='cpu',action_horizon=25)
try:
 image=np.zeros((4,5,3),np.uint8);c.act(np.zeros(9,np.float32),None,image,front_masked_depth_rgb=image)
 parent=len(c.action_queue);worker=json.loads((out/'actual-worker-queue.jsonl').read_text())['real_worker_queue_after']
 assert parent==0 and worker==24
 (out/'result.json').write_text(json.dumps(dict(historical_head='3c713de',parent_proxy_queue_after=parent,
   actual_worker_queue_after=worker,discrepancy_reproduced=True,real_policy_inference=0),indent=2)+'\n')
 print('PASS: R12 defect reproduced through historical subprocess: parent length 0, actual worker deque 24')
finally:c.close()
