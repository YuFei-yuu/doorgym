"""Independent 100-source batch; unchanged accepted expert and raw contract."""
import subprocess
from pathlib import Path
from success_random5.common import ROOT,sha,read,write,head
from expert50.common import hardlink_tree,raw_path
BASE='405356ad56d08650b5a6cf80c4764c65095004dd'
OLD_FREEZE='b2dfd5ec685865e3b5d1676bc76fc1938237301d'
def verify(run):
    m=read(run/'manifest.json')
    assert (m['target_successes'],m['reused_successes'],m['new_target_successes'])==(100,50,50)
    assert m['maximum_sim_seconds']==120 and head()==m['start_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
    assert sha(m['yaml'])==m['yaml_sha256']
    assert sha(run/'geometry.json')==m['geometry_sha256']
    assert sha(run/'seed-exclusions.json')==m['seed_exclusions_sha256']
    return m

def qualification(out,origin_kind,run):
    from expert100.quality import audit
    q=audit(out)
    q['leg_diagnostic']=read(out/'result.json')['leg_posture_diagnostic']
    q['origin_kind']=origin_kind;q['evidence']=str(out.relative_to(run));q['raw_path']=str(raw_path(out).relative_to(run))
    q['bindings']={n:sha(out/n) for n in ['command.json','initial-state.json','frozen-trial.json','physical-camera-binding.json','effective-config.json','effective-expert.json','result.json','states.jsonl','contacts.jsonl','lock-writes.jsonl','recording-clock.jsonl','raw-render-receipt.json']}
    # Independently reconstruct the physical <=1deg/<=1deg/s interval used
    # by the unchanged latch. Events remain diagnostics, not extra scoring.
    import json,math
    previous=None;close_start=None;first=None;last=None
    for line in (out/'states.jsonl').open():
        row=json.loads(line);last=row
        assert row['timing']['measured_post_time_s']==row['t']
        if row['latched'] is None:
            closed=abs(row['door_q'][0])<=math.radians(1) and abs(row['door_qd'][0])<=math.radians(1)
            if closed:close_start=row['t'] if close_start is None else close_start
            else:close_start=None
        elif first is None:
            first=row
            assert previous is not None and close_start is not None
            assert previous['t']-close_start>=.5-1e-9
            assert abs(row['latched']-previous['door_q'][0])<1e-7
        previous=row
    q['independent_physical_close_span_verified']=first is not None
    q['all_events_separate_from_success']=True
    return q
