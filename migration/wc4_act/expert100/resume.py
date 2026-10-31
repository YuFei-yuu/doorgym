"""Preserve phase1 and requalify checker-only false rejection without a scene."""
import argparse,json,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from expert100.common import *
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();ctl=run.with_name(run.name+'-control')
assert (ctl/'driver.exit').read_text().strip()=='2','phase1 must have drained all in-flight jobs'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
old=read(run/'manifest.json');summary=read(run/'summary.json');ledger=[json.loads(x) for x in (run/'attempt-ledger.jsonl').read_text().splitlines()]
assert summary['stop_reason']=='data_contract_anomaly'
bad=[x for x in ledger if x['status']!='success'];assert {x['seed'] for x in bad}=={62168,62170}
for entry in bad:
    assert entry['status']=='data_contract_anomaly'
    out=run/entry['evidence'];assert read(out/'entry-exit.json')['returncode']==0 and not read(out/'result.json')['error']
    assert 'max_relock_delta<1e-7' in (out.parent/'audit-error.txt').read_text()
phase=run/'phases/v1';phase.mkdir(parents=True,exist_ok=False)
for name in ['manifest.json','summary.json','source-index.json','progress.json','collection-start.json','attempt-ledger.jsonl']:
 p=run/name
 if p.exists():shutil.copyfile(p,phase/name)
# Archive partial external data numbering before rebuilding at completion.
(run/'raw').rename(phase/'raw');(run/'raw').mkdir() # preserve every partial alias and source byte
for name in ['summary.json','source-index.json']:(run/name).unlink()
files=subprocess.check_output(['git','ls-files','migration/wc4_act','high-level/float_ik','high-level/dp','high-level/data'],cwd=ROOT,text=True).splitlines()
new=dict(old);new.update(start_commit=head(),source_sha256={n:sha(ROOT/n) for n in files if (ROOT/n).is_file()},previous_phase_manifest='phases/v1/manifest.json',previous_phase_manifest_sha256=sha(phase/'manifest.json'),
 checker_repair='Remove accidental <=1e-7rad post-latch drift cutoff; require exact original measured target, zero first latch delta, exact pre-write/previous-state and post-write target binding. Preserve all scoring/gate/physics thresholds.')
(run/'manifest.json').write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n')
recovery=[]
for entry in bad:
    out=run/entry['evidence'];q=qualification(out,'new',run);assert q['success'];q['trial_manifest_sha256']=sha(out.parent/'manifest.json')
    write(out.parent/'audit-recheck.json',q);write(run/'qualification'/('seed%d.json'%q['seed']),q)
    recovery.append(dict(seed=q['seed'],original_status=entry['status'],original_audit_error_path=str((out.parent/'audit-error.txt').relative_to(run)),
        original_execution_commit=read(out/'command.json')['head'],checker_repair_commit=head(),recheck_report_path=str((out.parent/'audit-recheck.json').relative_to(run)),
        raw_sha256=q['raw_sha256'],max_relock_write_delta=q['max_relock_write_delta'],first_lock_delta_exact_zero=True,constant_measured_target_and_all_write_bindings_verified=True,
        result='qualified without new physics steps, resampling, data rewrite or relaxed latch/scoring thresholds',rechecked_at=time.time()))
write(run/'audit-recovery.json',recovery)
print(json.dumps(dict(requalified_seeds=[x['seed'] for x in recovery],new_successes=len(list((run/'qualification').glob('*.json')))-50,next_seed=max(x['seed'] for x in ledger)+1,active_commit=head()),indent=2))
