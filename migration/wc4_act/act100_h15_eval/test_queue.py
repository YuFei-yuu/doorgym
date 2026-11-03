"""Actual h15 journal -> independent auditor; synthetic callbacks, no physics/model."""
import json,tempfile,hashlib,ast
from pathlib import Path
import numpy as np
from runtime_audit import verify_journal,CriticalEvidenceError
from action_journal import ActionJournal
imports=[n for n in ast.walk(ast.parse(Path(__file__).with_name('rollout.py').read_text())) if isinstance(n,ast.ImportFrom) and any(a.name=='ActionJournal' for a in n.names)]
assert len(imports)==1 and imports[0].module=='action_journal'
def receipt(seq):
 index=(seq-1)%15;chunk=(seq-1)//15+1;actions=np.arange(135,dtype=np.float32).reshape(15,9)+chunk
 return dict(schema='door_worker_action_v1',emission_seq=seq,executed_horizon=15,epoch=1,new_chunk=index==0,chunk_id=chunk,action_index=index,queue_before=0 if index==0 else 15-index,queue_after=14-index,action=actions[index].tolist(),selected_chunk_actions=actions.tolist(),chunk_sha256=hashlib.sha256(actions.tobytes()).hexdigest())
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'journal.jsonl';journal=ActionJournal(p)
 for seq in range(1,32):
  e=receipt(seq);journal.begin(e,e['action'],e['action'],dict(frame=seq-1))
  for sub in range(2):
   journal.command_applied((seq-1)*2+sub,((seq-1)*2+sub)*.02,e['action']);journal.physics_completed((seq-1)*2+sub,((seq-1)*2+sub+1)*.02)
 journal.close();proof=verify_journal(p);assert proof['new_chunks']==3 and proof['emitted']==31 and proof['completed_physics_steps']==62
 for field,value in [('executed_horizon',25),('queue_after',13)]:
  e=receipt(1);e[field]=value;j=ActionJournal(Path(tmp)/(field+'.jsonl'))
  try:j.begin(e,e['action'],e['action'],{})
  except RuntimeError:pass
  else:raise AssertionError('Invalid horizon/queue accepted by live journal')
  finally:j.close()
 print(json.dumps(dict(passed=True,actual_rollout_journal_import_verified=True,live_h15_journal_roundtrip=True,wrong_horizon_queue_rejected=True,synthetic_emissions=31,synthetic_acknowledgements=62,actual_physics_steps=0,model_calls=0,optimizer_updates=0)))
