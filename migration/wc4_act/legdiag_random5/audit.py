"""Reuse the previous full raw audit, then compare diagnostic-only executions."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from success_random5.audit import audit
from success_random5.common import read,write,sha

def compare(old,new):
    keys=['state','action','actual_base','arm_q','arm_qd','door_q','door_qd','base_dof_actual']
    counts=0;maxerr={k:0. for k in keys}
    # Structured base DOFs are nested numeric lists in JSON.
    with old.open() as fa,new.open() as fb:
        for aa,bb in zip(fa,fb):
            a=json.loads(aa);b=json.loads(bb);assert a['step']==b['step']
            for k in keys:
                maxerr[k]=max(maxerr[k],float(np.max(abs(np.asarray(a[k],float)-np.asarray(b[k],float)))))
            counts+=1
    return dict(compared_prefix_steps=counts,max_abs_difference=maxerr,all_compared_values_exact=all(x==0 for x in maxerr.values()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run.resolve()
    prior=Path(read(run/'manifest.json')['previous_run']);rows=[]
    for i in range(5):
        out=run/('trial_%02d'%i)/'attempt_000';old=prior/('trial_%02d'%i)/'attempt_000'
        assert read(out/'actual-initial-identity.json')['exact_match']
        assert read(out/'initial-state.json')==read(old/'initial-state.json')
        row=audit(out);result=read(out/'result.json');old_result=read(old/'result.json')
        row['diagnostic']=result['leg_posture_diagnostic']
        row['previous']=dict(success=old_result['success'],reason=old_result['reason'],sim_seconds=old_result['sim_seconds'],
            max_abs_open_deg=old_result['simplified_score']['max_abs_open_deg'],legal_relock_time_s=old_result['simplified_score']['legal_relock_time_s'])
        row['comparison']=compare(old/'states.jsonl',out/'states.jsonl')
        row['exact_initial_match']=True
        # Re-derive the first former-abort condition directly from the original per-step records.
        first=None;qualified_count=0
        for line in (out/'states.jsonl').open():
            r=json.loads(line);g=r['foot_guard']
            if g['since_s'] is not None and r['t']-g['since_s']>=.10-1e-9:
                qualified_count+=1
                if first is None:first=dict(step=r['step'],time_s=r['t'],bad=g['bad'])
        assert bool(first)==row['diagnostic']['triggered']
        assert qualified_count==row['diagnostic']['qualified_steps']
        if first:
            assert first['step']==row['diagnostic']['first_event']['diagnostic_step']
            assert first['time_s']==row['diagnostic']['first_event']['physical_post_time_s']
        rows.append(row)
    selected=next((t for t in rows if t['diagnostic']['triggered']),None)
    write(run/'audit.json',dict(successes=sum(x['success'] for x in rows),denominator=5,trials=rows,
        representative_replay_index=None if selected is None else selected['index']))
    with (run/'comparison.csv').open('x') as f:
        keys=['seed','previous_success','success','max_abs_open_deg','legal_relock_time_s','whole_body_pass_time_s','sim_seconds','first_leg_event_s','stop_reason']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:
            event=r['diagnostic']['first_event'];d={k:r[k] for k in ['seed','success','max_abs_open_deg','legal_relock_time_s','whole_body_pass_time_s','sim_seconds','stop_reason']}
            d.update(previous_success=r['previous']['success'],first_leg_event_s=None if event is None else event['physical_post_time_s']);w.writerow(d)
    print((run/'comparison.csv').read_text())
if __name__=='__main__':main()
