"""Admission requires measured, visual/code and recording decisions separately."""
import collections,hashlib,json,pathlib,shutil
from review import ROOT,RUN,read,write


def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    ledger=read(RUN/'measured-ledger.json')['trials']
    decisions=read(RUN/'visual-decisions.json')['trials']
    decisions={(d['source_run'],d['trial']):d for d in decisions}
    assert len(ledger)==len(decisions)==72
    original=read(ROOT/'.repro/runs/wc4-fulltask-expand-r9-20261021/candidate-index.json')['episodes']
    assert len(original)==46
    assert shutil.disk_usage(ROOT).free>20*1024**3, 'Reserve capacity; never delete history'
    dest=RUN/'candidates';dest.mkdir(exist_ok=False)
    rows=[];mapping=[];episodes=[]
    for x in sorted(ledger,key=lambda x:(x['old_candidate_episode'] is None,x['old_candidate_episode'] if x['old_candidate_episode'] is not None else x['seed'])):
        key=(x['source_run'],x['trial']);d=decisions[key]
        label='r%d-trial_%02d'%(8 if 'pilot-r8' in x['source_run'] else 9,x['trial'])
        record=read(RUN/label/'record-validation.json')
        task_ok=bool(x['measured_task_gates_pass'] and d['accept'] and d['code_audit_pass'])
        data_ok=bool(record['recording_checks_pass'])
        eligible=task_ok and data_ok
        src=pathlib.Path(x['raw_npz']);digest=sha(src)
        e=None
        if eligible:
            e=len(episodes);target=dest/('episode_%06d.npz'%e)
            shutil.copyfile(src,target);assert sha(target)==digest
            frozen=read(pathlib.Path(x['source_dir'])/'frozen-trial.json')
            episodes.append(dict(episode_index=e,trial=x['trial'],seed=x['seed'],frames=x['frames'],
                source_run=x['source_run'],source_npz=str(src),candidate_npz=str(target),sha256=digest,
                parameters=frozen,old_candidate_episode=x['old_candidate_episode'],review=str(RUN/label/'decision.json')))
        if x['old_candidate_episode'] is not None:
            assert digest==original[x['old_candidate_episode']]['sha256'], 'Old candidate bytes changed'
        entry=dict(source_run=x['source_run'],trial=x['trial'],seed=x['seed'],
            automatic_terminal_pass=x['automatic_terminal_pass'],full_task_review_pass=task_ok,
            raw_data_qualified=data_ok,candidate_eligible=eligible,new_episode=e,
            old_candidate_episode=x['old_candidate_episode'],was_isolated=x['was_isolated'],raw_sha256=digest,
            measured_failures=x['failures'],visual_decision=d,
            converted_data_acceptance='pending_independent_new_dataset_checks' if eligible else 'not_admitted')
        write(RUN/label/'decision.json',entry);rows.append(entry)
        mapping.append({k:entry[k] for k in ['source_run','trial','seed','old_candidate_episode','was_isolated','new_episode','raw_sha256','candidate_eligible']})
    assert len(episodes)<=72
    shutil.copyfile(ROOT/'.repro/runs/wc4-fulltask-expand-r9-20261021/candidates/door_dp_feature_names.json',dest/'door_dp_feature_names.json')
    write(RUN/'candidate-index.json',dict(episodes=episodes,criterion='R11 full-task review',maximum=72))
    write(RUN/'candidate-mapping.json',dict(trials=mapping))
    write(RUN/'review-ledger.json',dict(trials=rows,counts=dict(automatic_terminal=sum(x['automatic_terminal_pass'] for x in rows),
        full_task=sum(x['full_task_review_pass'] for x in rows),raw_data_qualified=sum(x['raw_data_qualified'] for x in rows),
        candidates=len(episodes),frames=sum(x['frames'] for x in episodes),
        restored=sum(x['was_isolated'] and x['candidate_eligible'] for x in rows),
        retained=sum(not x['was_isolated'] and x['candidate_eligible'] for x in rows)),
        scenes=0,collection=0,training=0,policy_inference=0))
    write(RUN/'candidate-seal.json',dict(sha256=sha(RUN/'candidate-index.json'),
        episodes=len(episodes),frames=sum(x['frames'] for x in episodes),all_npz_byte_copies=True))
    print(read(RUN/'review-ledger.json')['counts'],flush=True)


if __name__=='__main__':main()
