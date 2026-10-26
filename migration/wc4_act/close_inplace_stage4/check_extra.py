"""Exactly one further local check: group 10, hover roll 7 degrees."""
import argparse,os,json,subprocess
from pathlib import Path
import numpy as np
from common import *
def main(run):
    cfg=read(run/'config.json');ledger=read(run/'local-ledger.json');assert ledger['used']==10
    entry=dict(id=10,kind='single_seed_full_pose_ik',description='hover roll=7 deg, same radius/height/1cm offset',status='started',tag='hover',roll_deg=7.)
    ledger['groups'].append(entry);ledger['used']=11;write(run/'local-ledger.json',ledger)
    G=Geometry();S=FullPose(G.robot);start,rows=reference(cfg);B=start['base'];a=start['theta'];P=point(cfg,a,cfg['hover_offset']);R=pose_rotation(a,7.)
    seed=ledger['groups'][2]['q'];q,ik=S.solve(B,P,R,seed,cfg['local_ik_nfev']);q=np.r_[q,HOME[-1]]
    result=evaluate(G,S,B,q,a,P,R,contact=False,cfg=cfg)
    entry.update(status='completed',theta=a,offset=cfg['hover_offset'],seed=seed,target=P.tolist(),target_rotation=R.tolist(),q=q.tolist(),ik=ik,result=result)
    write(run/'local-ledger.json',ledger);write(run/'group10.json',entry);print(json.dumps(entry),flush=True)
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('hide CUDA')
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
