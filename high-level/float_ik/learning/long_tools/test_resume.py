"""CPU checks of selection retirement, immutable checkpoints and patience."""
import json
from pathlib import Path
import tempfile
from resume_state import digest, patience_state, prepare_resume

def check():
    points=[dict(step=5000,successes=0,mean_final_angle_deg=0),dict(step=15000,successes=5,mean_final_angle_deg=50),
            dict(step=17000,successes=1,mean_final_angle_deg=44),dict(step=19000,successes=0,mean_final_angle_deg=46)]
    assert patience_state(points)==(2,5,50)
    assert patience_state(points+[dict(step=21000,successes=0,mean_final_angle_deg=50.0009)])[0]==3
    assert patience_state(points+[dict(step=21000,successes=0,mean_final_angle_deg=50.002)])[0]==0
    assert patience_state(points+[dict(step=21000,successes=6,mean_final_angle_deg=0)])[0]==0
    with tempfile.TemporaryDirectory() as directory:
        r=Path(directory);(r/'training/checkpoints').mkdir(parents=True)
        for s in (15000,21000):(r/'training/checkpoints'/('step_%06d.pt'%s)).write_bytes(str(s).encode())
        (r/'budget.jsonl').write_text(json.dumps(dict(command_id='original',elapsed_sec=5256))+'\n')
        original=(r/'budget.jsonl').read_bytes()
        (r/'training/train.jsonl').write_text(''.join(json.dumps(dict(step=s))+'\n' for s in range(5001,21001)))
        (r/'audit_early.json').write_text('{"passed":true}')
        (r/'config.json').write_text('{}')
        selection=dict(status='validation_only_resource_stopped',best_step=15000,last_step=21000)
        (r/'selection.json').write_text(json.dumps(selection))
        (r/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(r)):dict(sha256=digest(p)) for p in (r/'training/checkpoints').glob('*.pt')}))
        for point in points:(r/('validation_point_%06d.json'%point['step'])).write_text(json.dumps(point))
        value=prepare_resume(r,21000)
        assert value['consecutive_no_improvement_before']==2 and value['new_optimizer_steps_before']==16000
        assert (r/'budget.jsonl').read_bytes()==original and not (r/'selection.json').exists()
        assert json.loads((r/value['archive']/'selection.json').read_text())==selection
        assert prepare_resume(r,21000)==value
        # A later explicit resource-stop recovery keeps the first audit, ledger
        # prefix and completed validation, rather than treating it as test use.
        (r/'selection.json').write_text(json.dumps(selection))
        again=prepare_resume(r,21000)
        assert again['prior_resume_events']==[value['archive']]
        assert json.loads((r/again['archive']/'resume_state.json').read_text())==value
        assert (r/'budget.jsonl').read_bytes()==original
        assert all(digest(r/name)==h for name,h in again['validation_completed_before'].items())
        (r/'test/best_seed_5000').mkdir(parents=True);(r/'test/best_seed_5000/summary.json').write_text('{}')
        try:prepare_resume(r,21000)
        except AssertionError as exc:assert 'Test already accessed' in str(exc)
        else:raise AssertionError('Failed to prohibit test leakage')
    print(json.dumps(dict(passed=True,checks=['historical_patience','angle_tolerance','success_improvement','immutable_budget_prefix','checkpoint_hashes','archived_provisional_selection','idempotent_before_progress','test_leakage_guard'])))

if __name__=='__main__':check()
