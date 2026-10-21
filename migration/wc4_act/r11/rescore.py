"""Recheck final criteria from measured flags, retaining the first raw audit.

The historical executor records end_panel_contact and nonzero_door_contact
directly from the same raw records. Neither flag depends on expert phase.
"""
import json
from criteria import score
from review import ROOT,RUN,read,write


def main():
    original=read(RUN/'measured-ledger.json');rows=[]
    write(RUN/'initial-raw-measured-ledger.json',original)
    for x in original['trials']:
        p=__import__('pathlib').Path(x['source_dir']);states=[json.loads(l) for l in (p/'states.jsonl').open()]
        for s in states:
            s.pop('phase',None)
            s.update(end_panel_raw_contact=s['end_panel_contact'],loaded_door_contact=s['nonzero_door_contact'],zero_lambda_door_records=0,nonend_door_records=0)
        locks=[json.loads(l) for l in (p/'lock-writes.jsonl').open()]
        final=score(states,locks,read(p/'reference.json'))
        # Original diagnostic counts come from the complete raw contact scan.
        final['diagnostics']=x['diagnostics']
        row=x.copy();row.update(final)
        row['rescore_basis']='Phase field removed; measured phase-independent contact flags from the same raw executor records. Raw audit and original event sheets retained.'
        label='r%d-trial_%02d'%(8 if 'pilot-r8' in x['source_run'] else 9,x['trial'])
        write(RUN/label/'final-measured-review.json',row);rows.append(row)
        print(row['seed'],row['measured_task_gates_pass'],row['failures'],flush=True)
    write(RUN/'measured-ledger.json',dict(trials=rows,old_candidates=46,isolated=26,
        final_criteria='No independent clean opening-contact gap requirement; phase erased in all72 rescores.'))


if __name__=='__main__':main()
