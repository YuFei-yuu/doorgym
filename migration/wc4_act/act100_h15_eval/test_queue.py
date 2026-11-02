"""Synthetic queue evidence acceptance/rejection; no model, physics or policy."""
import json,tempfile,hashlib
from pathlib import Path
import numpy as np
from runtime_audit import verify_journal,CriticalEvidenceError
rows=[]
for seq in range(1,32):
 index=(seq-1)%15;chunk=(seq-1)//15+1;actions=np.arange(135,dtype=np.float32).reshape(15,9)+chunk
 e=dict(emission_seq=seq,executed_horizon=15,epoch=1,new_chunk=index==0,chunk_id=chunk,action_index=index,queue_before=0 if index==0 else 15-index,queue_after=14-index,action=actions[index].tolist(),selected_chunk_actions=actions.tolist(),chunk_sha256=hashlib.sha256(actions.tobytes()).hexdigest())
 rows.append(dict(event='emitted',receipt=e,policy_output=e['action'],applied_action=e['action']))
 for sub in range(2):rows.extend([dict(event='command_applied',emission_seq=seq,substep=sub,action=e['action']),dict(event='physics_completed',emission_seq=seq,substep=sub)])
rows.append(dict(event='end',emitted=31,last_action_completed_substeps=2))
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'journal.jsonl'
 def save():p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
 save();proof=verify_journal(p);assert proof['new_chunks']==3 and proof['emitted']==31 and proof['completed_physics_steps']==62
 for field,value in [('executed_horizon',25),('queue_after',13)]:
  e=rows[0]['receipt'];old=e[field];e[field]=value;save()
  try:verify_journal(p)
  except CriticalEvidenceError:pass
  else:raise AssertionError('Invalid horizon/queue accepted')
  e[field]=old
print(json.dumps(dict(passed=True,horizon15_partial_chunk_verified=True,wrong_horizon_and_queue_rejected=True,model_calls=0,physics_steps=0,optimizer_updates=0)))
