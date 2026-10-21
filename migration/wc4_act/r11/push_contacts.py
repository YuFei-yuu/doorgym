"""Diagnose raw body pairs in measured push intervals, without phase labels."""
import json,collections,re
from review import RUN,read,write


def main():
    ledger=read(RUN/'measured-ledger.json')['trials'];reports=[]
    for x in ledger:
        a=x['aligned_end_push_steps'];counts=collections.Counter();nonend=[]
        if a:
            with open(x['source_dir']+'/contacts.jsonl') as f:
                for line in f:
                    k=int(re.match(r'\{"step":\s*(\d+)',line).group(1))
                    if not a[0]<=k<=a[-1]:continue
                    c=json.loads(line)
                    for row in c['contacts']:
                        door=[n for n in row['names'] if n.get('actor_handle')==2]
                        robot=[n for n in row['names'] if n.get('actor_handle') in [0,1]]
                        if not door or not robot:continue
                        end=robot[0]['body'] in ['gripperStator','gripperMover']
                        pair=robot[0]['body']+' / '+door[0]['body']
                        counts[pair+(' | zero_lambda' if row['lambda']==0 else ' | nonzero_lambda')]+=1
                        if not end and row['lambda']!=0:nonend.append(dict(step=k,pair=pair,lambda_value=row['lambda']))
        reports.append(dict(seed=x['seed'],source_run=x['source_run'],trial=x['trial'],
            first_aligned_push_step=a[0] if a else None,last_aligned_push_step=a[-1] if a else None,
            counts=dict(counts),nonend_nonzero_records=nonend,
            interpretation='Raw manifold diagnostics only. Nonzero is not a calibrated force threshold. No contact coverage, force or work criterion. Full task needs original video plus measured motion and code.'))
        print(x['seed'],len(nonend),flush=True)
    write(RUN/'measured-push-contact-diagnostics.json',dict(trials=reports,phase_labels_used=False))


if __name__=='__main__':main()
