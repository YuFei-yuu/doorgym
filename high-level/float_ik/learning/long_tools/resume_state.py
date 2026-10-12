"""Audit an authorized continuation without resetting the original experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import time
from bounded import atomic_json, read_rows


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def patience_state(points, tolerance=.001):
    assert points and [p['step'] for p in points]==sorted({p['step'] for p in points})
    best_success=points[0]['successes'];best_angle=points[0]['mean_final_angle_deg'];stale=0
    for point in points[1:]:
        improved=point['successes']>best_success or point['mean_final_angle_deg']>best_angle+tolerance
        stale=0 if improved else stale+1
        best_success=max(best_success,point['successes']);best_angle=max(best_angle,point['mean_final_angle_deg'])
    return stale,best_success,best_angle


def prepare_resume(root, step):
    r=Path(root)
    assert not (r/'active_command.json').exists()
    assert not list((r/'test').glob('*/summary.json')), 'Test already accessed: cannot resume model selection'
    ledger=read_rows(r)
    assert len({x['command_id'] for x in ledger})==len(ledger)
    assert sum(x['elapsed_sec'] for x in ledger)<21600
    rows=[json.loads(x) for x in (r/'training/train.jsonl').read_text().splitlines()]
    latest=max(int(p.stem.split('_')[1]) for p in (r/'training/checkpoints').glob('step_*.pt'))
    assert [x['step'] for x in rows]==list(range(5001,latest+1)), 'Training history must be contiguous; never repeat updates'
    assert step<=latest<=25000
    assert json.loads((r/'audit_early.json').read_text())['passed']
    record=r/'resume_state.json'
    previous=None
    if record.exists():
        previous=json.loads(record.read_text())
        for name,h in previous['protected_checkpoints'].items():assert digest(r/name)==h
        assert digest(r/'config.json')==previous['config_sha256']
        assert hashlib.sha256((r/'budget.jsonl').read_bytes()[:previous['budget_prefix_bytes']]).hexdigest()==previous['budget_prefix_sha256']
        if not (r/'selection.json').exists():
            assert previous['resume_step']==step
            return previous
    assert latest==step
    selection=json.loads((r/'selection.json').read_text())
    assert selection['status']=='validation_only_resource_stopped'
    assert selection['last_step']==step
    seal=json.loads((r/'artifact_sha256.json').read_text())
    protected=dict(previous['protected_checkpoints']) if previous else {}
    for n in (selection['best_step'],step):
        name='training/checkpoints/step_%06d.pt'%n
        protected[name]=digest(r/name)
        assert protected[name]==seal[name]['sha256'], 'Previously sealed checkpoint changed'
    archive=r/'resume_events'/str(time.time_ns());archive.mkdir(parents=True)
    for name in ('selection.json','test_plan.json','status.json','stop_error.json','REPORT.md','delivery_status.json','delivery_checks.json','artifact_sha256.json','resume_state.json','resume_source_audit.json','resume_finish_status.json'):
        if (r/name).exists():shutil.copy2(r/name,archive/name)
    source=r/'provenance/final-learning-source.tar.gz'
    if source.exists():shutil.copy2(source,archive/source.name)
    points=[json.loads(p.read_text()) for p in sorted(r.glob('validation_point_*.json'))]
    stale,success,angle=patience_state(points)
    value=dict(resume_step=step,authorized_at=time.time(),authorization='User explicitly authorized resource recheck, missing validation, bounded continuation and independent tests under original cumulative six-hour ledger',
        prior_selection=selection,archive=str(archive.relative_to(r)),protected_checkpoints=protected,
        budget_prefix_bytes=(r/'budget.jsonl').stat().st_size,budget_prefix_sha256=digest(r/'budget.jsonl'),
        used_seconds_before=sum(x['elapsed_sec'] for x in ledger),new_optimizer_steps_before=len(rows),
        consecutive_no_improvement_before=stale,best_success_count_before=success,best_mean_angle_before=angle,
        test_attempts_before=0,config_sha256=digest(r/'config.json'),
        prior_resume_events=(previous.get('prior_resume_events',[])+[previous['archive']]) if previous else [],
        validation_completed_before={p.name:digest(p) for p in sorted(r.glob('validation_point_*.json'))})
    atomic_json(record,value)
    # Only retire the provisional selection after its complete audit is durable.
    (r/'selection.json').unlink()
    return value
