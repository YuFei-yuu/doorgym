"""Evidence bookkeeping and review admission; no motion or simulator calls."""
import argparse, json, math, shutil, subprocess, sys, time
from pathlib import Path
from collect import ROOT, OLD, read, write, sha


def ledger(run):
    manifest=read(run/'manifest.json');rows=[]
    for claim in sorted((run/'claims').glob('*.json')):
        i=read(claim)['index'];p=run/('trial_%02d'%i)
        row=dict(trial=i,seed=manifest['seeds'][i],scene_started=(p/'scene-creation-receipt.json').exists(),parameters=manifest['trials'][i])
        if (p/'result.json').exists():
            r=read(p/'result.json');row.update(automatic_terminal_pass=r.get('automatic_final_conditions_met',False),reason=r['reason'],sim_seconds=r['sim_seconds'],steps=r['steps'])
        if (p/'disposition.json').exists():row.update(read(p/'disposition.json'))
        rows.append(row)
    (run/'attempts.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    write(run/'progress.json',dict(invocations=len(rows),scenes=sum(x['scene_started'] for x in rows),automatic_terminal_passes=sum(x.get('automatic_terminal_pass',False) for x in rows),
        candidates=len(read(run/'candidate-index.json')['episodes']),new_candidates=sum(x.get('status')=='candidate' for x in rows),
        withheld=sum(x.get('status','').startswith('withheld') for x in rows),task_failures=sum(x.get('status')=='task_failed' for x in rows),
        pending_visual=[x['trial'] for x in rows if x.get('status')=='pending_visual_review'],budget_max=64,target=50))


def screen(run,i):
    p=run/('trial_%02d'%i);r=read(p/'result.json');v=read(p/'record-validation.json');review=read(p/'review/review.json')
    states=[json.loads(x) for x in (p/'states.jsonl').open()]
    locks=[json.loads(x) for x in (p/'lock-writes.jsonl').open()]
    violations=[]
    if v['systemic_errors']:violations+=v['systemic_errors']
    if any(x['door_effort'][0]!=0 for x in states if x['unlocked']):violations.append('nonzero_unlocked_hinge_effort')
    for lock in locks:
        k=lock['step']
        if lock['kind']=='initial' and k>0 and states[k-1]['unlocked']:violations.append('initial_hinge_write_after_unlock')
        if lock['kind'] not in ['initial','relocked']:violations.append('unknown_hinge_write_kind')
    rel=[x for x in locks if x['kind']=='relocked']
    if rel:
        hold=review['hold']
        if not hold or hold['span_s']<.5 or not hold['no_snap']:violations.append('relock_before_physical_hold_or_snap')
    if any(x in r['reason'] for x in ['nonfinite','forbidden_unlocked_hinge_write','nonzero_auxiliary_hinge_effort','missing_depth_observation','evidence_missing','infrastructure_or_recording_error']):violations.append(r['reason'])
    if violations:
        write(run/'SYSTEMIC_STOP.json',dict(index=i,violations=violations));status='systemic_stop'
    elif review['negative_command_diagnostic']['count']:
        status='withheld_negative_command'
    elif review['execution_review_supports_full_success']:
        status='pending_visual_review'
    elif not r.get('automatic_final_conditions_met'):
        status='task_failed'
    else:
        status='withheld_physical_evidence_review'
    write(p/'disposition.json',dict(status=status,candidate_training_eligible=False,frames=v['frames'],recording_checks_pass=v['recording_checks_pass'],
        negative_command_diagnostic=review['negative_command_diagnostic'],review=str(p/'review/review.json'),systemic_violations=violations))
    ledger(run);print(json.dumps(read(run/'progress.json')),flush=True)
    if violations:raise SystemExit(5)


def decide(run,i,decision,note):
    p=run/('trial_%02d'%i);d=read(p/'disposition.json')
    with (p/'visual-review-decision.json').open('x') as f:
        json.dump(dict(decision=decision,note=note,time=time.time(),basis='original execution RGB frames plus aligned states/contacts/lock logs; not state replay'),f,ensure_ascii=False,indent=2)
    if decision=='accept':
        assert d['status']=='pending_visual_review'
        index=read(run/'candidate-index.json');e=len(index['episodes']);assert e<50
        src=Path(read(p/'record-validation.json')['raw_npz']);dest=run/'candidates'/('episode_%06d.npz'%e)
        assert not dest.exists();shutil.copyfile(src,dest);assert sha(src)==sha(dest)
        trial=read(p/'frozen-trial.json')
        index['episodes'].append(dict(episode_index=e,trial=i,seed=trial['seed'],frames=d['frames'],source_run=str(run),source_npz=str(src),candidate_npz=str(dest),sha256=sha(dest),parameters=trial,provenance='R9 complete-task physical evidence and visual review'))
        write(run/'candidate-index.json',index);d.update(status='candidate',candidate_training_eligible=True,episode_index=e)
    else:
        d.update(status='withheld_visual_review',candidate_training_eligible=False,visual_note=note)
    write(p/'disposition.json',d);ledger(run);print(json.dumps(read(run/'progress.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--index',type=int);p.add_argument('--mode',choices=['screen','decision','ledger'],required=True);p.add_argument('--decision',choices=['accept','withhold']);p.add_argument('--note')
    a=p.parse_args()
    if a.mode=='screen':screen(a.run,a.index)
    elif a.mode=='decision':decide(a.run,a.index,a.decision,a.note)
    else:ledger(a.run)
