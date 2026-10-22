"""Cross-interpreter production proxy/worker tests with a non-policy backend."""
import os,sys,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'high-level/dp'),str(Path(__file__).parent)]
import door_policy_backend as backend
from action_journal import ActionJournal

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['R13_FIXTURE_BACKEND_EVENTS']=str(out/'backend-events.jsonl')
worker=ROOT/'migration/wc4_act/r17/fixture_worker.py'
backend._worker_python_command=lambda checkpoint:[os.environ['DOOR_DP_LEROBOT_PYTHON'],'-B',str(worker)]
controller=backend.DoorPolicySubprocessController('CPU-FIXTURE-NO-CHECKPOINT',device='cpu',action_horizon=5)
controller.reset()
journal=ActionJournal(out/'applied.jsonl');receipts=[];image=np.zeros((4,5,3),np.uint8)
try:
 assert len(controller.action_queue)==0
 for i in range(15):
    state=np.zeros(9,np.float32);state[0]=i
    # Real parent RPC, worker/controller append->sample->pop, response receipt.
    raw=controller.act(state,None,image,front_masked_depth_rgb=image)
    receipt=controller.last_action_evidence
    receipts.append(receipt)
    assert receipt['worker_pid']!=os.getpid()
    assert len(controller.action_queue)==receipt['queue_after']
    assert len(controller.obs_buffer)==1
    journal.begin(receipt,raw,raw,dict(fixture_frame=i,no_simulation=True))
    for substep in range(2):
        journal.command_applied(2*i+substep,.02*(2*i+substep),raw.copy())
        journal.physics_completed(2*i+substep,.02*(2*i+substep+1))
 journal.close()
 events=[json.loads(l) for l in (out/'backend-events.jsonl').open()]
 assert [e['observation'] for e in events]==[0,5,10]
 assert len(events)==3
 for chunk in range(3):
    group=receipts[chunk*5:(chunk+1)*5]
    assert [r['action_index'] for r in group]==list(range(5))
    assert [r['queue_after'] for r in group]==list(range(4,-1,-1))
    assert [r['new_chunk'] for r in group]==[True]+[False]*4
    assert [r['action'][0] for r in group]==list(range((chunk+1)*1000,(chunk+1)*1000+5))
 # Actual proxy now observes nonempty queue; the R12 constant-zero behavior
 # would fail the first act and all receipt depth comparisons above.
 controller.reset();controller.act(np.zeros(9,np.float32),None,image,front_masked_depth_rgb=image)
 assert len(controller.action_queue)==4
 controller.action_queue.clear();assert len(controller.action_queue)==0
 controller.reset();assert len(controller.action_queue)==len(controller.obs_buffer)==0
 # Mutated transport receipt must fail parent join before another application.
 mutated=dict(receipts[0]);mutated['action_index']=1
 bad=ActionJournal(out/'rejected.jsonl')
 try:bad.begin(mutated,np.array(receipts[0]['action']),np.array(receipts[0]['action']),{})
 except RuntimeError:pass
 else:raise AssertionError('Corrupt boundary accepted')
 finally:bad.close()
 partial=ActionJournal(out/'partial-final.jsonl')
 partial.begin(receipts[0],np.array(receipts[0]['action']),np.array(receipts[0]['action']),{})
 partial.command_applied(0,0.,np.array(receipts[0]['action']));partial.physics_completed(0,.02);partial.close()
 end=json.loads((out/'partial-final.jsonl').read_text().splitlines()[-1]);assert end['partial_final_action'] and end['partial_final_chunk']
 from audit import verify_journal
 proof=verify_journal(out/'applied.jsonl');assert proof['emitted']==15 and proof['new_chunks']==3
 assert all('worker_act_wall_s' in r for r in receipts)
 (out/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
 result=dict(status='passed',parent_python=sys.executable,parent_version=sys.version,
     worker_python=os.environ['DOOR_DP_LEROBOT_PYTHON'],parent_pid=os.getpid(),worker_pid=receipts[0]['worker_pid'],
     fixture_chunks_verified=3,fixture_actions_verified=15,fixture_application_acknowledgements=30,
     additional_reset_clear_backend_calls=1,real_policy_inference=0,ACT_training_updates=0,simulation_scenes=0,
     evidence='actual worker sampling events, real deque transitions, emitted queue row bytes and parent acknowledgements')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
finally:controller.close()
