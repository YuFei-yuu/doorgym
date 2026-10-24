"""Record final source-by-source joint review after operator evidence inspection."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reviewed-new-ranges',required=True);p.add_argument('--joint-review-note',required=True);p.add_argument('--contact-overrides',type=Path);a=p.parse_args();r=a.run
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (r/'scheduler-exit.txt').exists(),'Do not finalize decisions while scheduling'
reviewed=set()
for part in a.reviewed_new_ranges.split(','):
    lo,hi=map(int,part.split(':'));reviewed.update(range(lo,hi+1))
overrides=read(a.contact_overrides) if a.contact_overrides else {}
decisions=[];reuse=read(r/'reuse-index.json')
for old in reuse['sources']:
    decisions.append({'source_index':old['source_index'],'source_id':old['source_id'],'seed':old['seed'],
        'complete_task_qualified':True,'admitted':True,'reason':'Previously accepted stage-six complete task and raw quality; original path and SHA reused without execution.',
        'evidence_scope':'Original stage-six RGB/state/contact/source review, now reused by explicit task order',
        'raw_npz_sha256':old['files']['attempt_000/raw/episode_000000.npz']['sha256']})
for i in range(42):
    run=r/('trial_%02d'%i);q=run/'audit/task-review.json';index=8+i
    if not q.exists():
        decisions.append({'source_index':index,'source_id':'stage7-seed%d'%(62700+i),'seed':62700+i,
            'complete_task_qualified':False,'admitted':False,'reason':'Not executed or no trustworthy completed raw/task audit; see attempt ledger.',
            'evidence_scope':'No fabricated task result for missing execution or systemic interruption'})
        continue
    assert i in reviewed,'Uninspected new source: '+str(i)
    v=read(q);scored=read(run/'audit/r11-score.json')
    qualified=v['complete_task_review']=='supported_by_measured_contact_motion_and_frozen_source'
    explanation='Full R11 measured gates, intended fixed-finger loaded contact locations with coupled actual EE/door motion, unchanged hinge/lock source and release/home/final window jointly support complete task; full original Joint9/depth/clock/source checks pass.'
    if str(index) in overrides:
        override=overrides[str(index)];assert override['reason'] and override['evidence']
        qualified=bool(override['complete_task_qualified']);explanation=override['reason']
    qualified=bool(qualified and scored['measured_task_gates_pass'])
    admitted=bool(qualified and v['raw_recording_quality_pass'])
    if not admitted:explanation='; '.join(v['exclusion_or_hold_reasons']) or 'Complete-task attribution is not qualified; held out without a new R11 veto or a physical retry.'
    decisions.append({'source_index':index,'source_id':'stage7-seed%d'%v['seed'],'seed':v['seed'],
        'complete_task_qualified':qualified,'admitted':admitted,'reason':explanation,
        'evidence_scope':'Stage-seven explicit no-video authorization: full measured state/contact/source joint review; no visual-review claim',
        'task_review_sha256':sha(q),'r11_score_sha256':sha(run/'audit/r11-score.json'),
        'raw_quality_review_sha256':sha(run/'audit/record-validation.json'),'raw_npz_sha256':v['raw_npz_sha256'],
        'contact_override':overrides.get(str(index))})
assert [x['source_index'] for x in decisions]==list(range(50))
value={'reviewer':'execution chat','review_method':a.joint_review_note,'sources':decisions,
    'new_source_dossier_indices_inspected':sorted(reviewed),'video_waiver_explicit':True,
    'independent_scheme_acceptance':'pending','raw_stage_only':True,'conversion':0,'training_updates':0,
    'note':'No force, contact-coverage or path-error success gate added. Incidental non-end contact may be resolved by explicit joint review; unresolved attribution is held out.'}
with (r/'source-review-decisions.json').open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({'source_decisions':len(decisions),'admitted':sum(x['admitted'] for x in decisions),'new_dossiers_inspected':len(reviewed)}))
