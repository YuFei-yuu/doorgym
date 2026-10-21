import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r13.contacts import diagnostic
from fullcycle_contract import door_contact_review_flags
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
args=dict(body_map={},robot_ids={0,1,2},end_ids={2},door_ids={8,9},panel_id=8,angle_before=.5,angle_after=.49)
cases={
 'nonend_nonzero_closing':diagnostic(1,8,1.,**args),
 'nonend_zero_closing':diagnostic(1,8,0.,**args),
 'end_nonzero_closing':diagnostic(2,8,1.,**args),
 'wall_nonzero_closing':diagnostic(1,10,1.,**args),
 'self_nonzero_closing':diagnostic(0,1,1.,**args),
 'nonend_nonzero_opening':diagnostic(1,8,1.,**dict(args,angle_after=.51)),
 'nonend_nonzero_hold':diagnostic(1,8,1.,**dict(args,angle_after=.5)),
 'reversed_body_order':diagnostic(8,1,1.,**args),
}
assert cases['nonend_nonzero_closing']['nonend_nonzero_during_measured_closing']
assert cases['reversed_body_order']==cases['nonend_nonzero_closing']
assert all(not cases[k]['nonend_nonzero_during_measured_closing'] for k in cases if k not in ['nonend_nonzero_closing','reversed_body_order'])
assert cases['nonend_zero_closing']['nonend_door_record'] and cases['nonend_zero_closing']['zero_lambda_door_record']
assert not cases['wall_nonzero_closing']['nonend_door_record']
assert all(not x['automatic_failure'] and not x['phase_used'] for x in cases.values())
before=door_contact_review_flags('act',1.,True,False)
assert not before['off_target_closing_contact']
(out/'before-after.json').write_text(json.dumps(dict(r12_phase_act_diagnostic=before,r13_measured_diagnostics=cases,
 unchanged_scoring=True,no_causal_contact_claim=True),indent=2)+'\n')
print('PASS: eight phase-free measured contact diagnostic cases; raw records do not assert force, causality or failure')
