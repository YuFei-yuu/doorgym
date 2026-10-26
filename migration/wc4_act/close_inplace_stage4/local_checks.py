"""Ten explicitly enumerated groups; no continuous trajectory or hidden restarts."""
import argparse,os,json,math,subprocess,sys
from pathlib import Path
import numpy as np
from common import *
def main(run):
    cfg=read(Path(__file__).with_name('config.json'));run.mkdir(parents=True,exist_ok=True)
    ledger=run/'local-ledger.json'
    if ledger.exists():raise RuntimeError('Existing ledger: no automatic re-run')
    G=Geometry();S=FullPose(G.robot);start,source=reference(cfg);B=start['base'];theta=start['theta']
    old=[json.loads(l) for l in (ROOT/cfg['source_c2']/'trace.jsonl').open()]
    records=[];write(run/'start.json',dict(source=str(ROOT/cfg['source_stage3']/'single/trace.jsonl'),source_sha256=sha(ROOT/cfg['source_stage3']/'single/trace.jsonl'),row=start,meaning='Synthetic third-round end-of-turn HOME at the fixed traversal endpoint; not a measured arrival. Third-round parking and 0.45m later plan unchanged.'))
    write(run/'config.json',cfg)
    def claim(kind,description):
        if len(records)>=cfg['max_local_groups']:raise RuntimeError('local budget exhausted')
        records.append(dict(id=len(records),kind=kind,description=description,status='started'))
        write(ledger,dict(limit=cfg['max_local_groups'],used=len(records),groups=records))
        return records[-1]
    for phase in ['deploy_arc','edge']:
        r=[r for r in old if r['phase']==phase][-1];entry=claim('existing_pose_check','Frozen c2 '+phase+' endpoint, single FK/geometry check')
        q=np.array(r['q']);result=evaluate(G,S,B,q,theta,cfg=cfg)
        entry.update(status='completed',source_time=r['t'],q=q.tolist(),result=result)
        write(ledger,dict(limit=cfg['max_local_groups'],used=len(records),groups=records));print(json.dumps(entry),flush=True)
    for tag,a,offset in [('hover',theta,cfg['hover_offset']),('contact_start',theta,0),('contact_80',math.radians(80),0),('contact_70',math.radians(70),0)]:
        oldq=min([r for r in source if r['phase']=='close'],key=lambda r:abs(r['theta']-a))['q']
        for roll in cfg['roll_candidates_deg']:
            entry=claim('single_seed_full_pose_ik',tag+' roll='+str(roll));P=point(cfg,a,offset);R=pose_rotation(a,roll)
            q,ik=S.solve(B,P,R,oldq,cfg['local_ik_nfev']);q=np.r_[q,HOME[-1]]
            result=evaluate(G,S,B,q,a,P,R,contact=offset==0,cfg=cfg)
            entry.update(status='completed',tag=tag,theta=a,offset=offset,roll_deg=roll,seed=oldq,target=P.tolist(),target_rotation=R.tolist(),q=q.tolist(),ik=ik,result=result)
            write(ledger,dict(limit=cfg['max_local_groups'],used=len(records),groups=records));print(json.dumps(dict(id=entry['id'],tag=tag,roll=roll,ik=ik,passed=result['strict_local_pass'],gaps={k:v['gap_m'] for k,v in result['geometry']['minimums'].items()})),flush=True)
    write(run/'local-receipt.json',dict(argv=sys.argv,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),local_groups=10,trajectories=0,complete_candidates=0,physics=0,source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('common.py').resolve(),Path(__file__).with_name('config.json').resolve()]}))
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('hide CUDA')
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
