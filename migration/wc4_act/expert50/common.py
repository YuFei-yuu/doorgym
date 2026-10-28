import json,subprocess
from pathlib import Path
from success_random5.common import sha,read,write,head,ROOT
BASE='405356ad56d08650b5a6cf80c4764c65095004dd'
def verify(run):
    m=read(run/'manifest.json')
    assert m['target_successes']==50 and m['reused_successes']==5 and m['new_target_successes']==45
    assert m['maximum_sim_seconds']==120 and head()==m['start_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
    assert sha(m['yaml'])==m['yaml_sha256']
    assert sha(run/'geometry.json')==m['geometry_sha256']
    return m

def hardlink_tree(source,target):
    import os
    target.mkdir(parents=True,exist_ok=False)
    for p in sorted(source.rglob('*')):
        assert not p.is_symlink()
        q=target/p.relative_to(source)
        if p.is_dir():q.mkdir()
        else:os.link(p,q)

def raw_path(out):
    paths=list((out/'raw').glob('*.npz'));assert len(paths)==1
    return paths[0]

def qualification(out,origin_kind,run):
    from expert50.quality import audit
    from collections import Counter
    row=audit(out)
    result=read(out/'result.json');row['leg_diagnostic']=result['leg_posture_diagnostic']
    row['origin_kind']=origin_kind
    row['evidence']=str(out.relative_to(run))
    row['raw_path']=str(raw_path(out).relative_to(run))
    row['bindings']={n:sha(out/n) for n in ['command.json','initial-state.json','frozen-trial.json','physical-camera-binding.json','effective-config.json','effective-expert.json','result.json','states.jsonl','contacts.jsonl','lock-writes.jsonl','recording-clock.jsonl','raw-render-receipt.json']}
    return row
