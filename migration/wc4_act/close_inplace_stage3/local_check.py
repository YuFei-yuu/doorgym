"""Only two previously failed c2 snapshots; no routes or time integration."""
import argparse,json,math,os,sys
from pathlib import Path
import numpy as np
from geometry import Geometry,Solver,ROOT,baseT,apply,rot,write,read

def main(out):
    out.mkdir(parents=True,exist_ok=False);G=Geometry();S=Solver(G.robot,'corner')
    src=ROOT/'.repro/reviews/wc4-close-inplace-stage2-20261026/c2'
    rows=[json.loads(l) for l in (src/'trace.jsonl').open()]
    failures=[min(rows,key=lambda r:abs(r['t']-t)) for t in [64.19999960616697,76.69999960616448]]
    results=[]
    for k,r in enumerate(failures):
        B=r['base'];theta=r['theta'];q=np.array(r['q']);normal=np.array([math.cos(theta),math.sin(theta),0.])
        S.axis=np.array(r['contact_local_axis']);F=baseT(B)@S.fk(q)
        for radius in [.88,.895,.905]:
            p=np.array([2.467-radius*math.sin(theta)-.002*math.cos(theta),-.463+radius*math.cos(theta)-.002*math.sin(theta),.85])
            for roll_deg in [-60,-30,0,30,60]:
                targetR=rot(normal,math.radians(roll_deg))@F[:3,:3]
                solved,e=S.solve_point(B,p,normal,q,120,full_rotation=targetR)
                actual=baseT(B)@S.fk(solved);axis=actual[:3,:3]@S.axis;pad=actual[:3,3]+actual[:3,:3]@S.pad
                angle=float(np.arccos(np.clip(axis@normal,-1,1)))
                g=G.check(B,np.r_[solved,q[-1]],theta)
                item=dict(snapshot=k,source_time=r['t'],radius=radius,height=.85,roll_deg=roll_deg,base=B,theta=theta,q=np.r_[solved,q[-1]].tolist(),target=p.tolist(),target_rotation=targetR.tolist(),position_error_m=float(np.linalg.norm(pad-p)),axis_error_rad=angle,solver=e,geometry=g)
                results.append(item);print(json.dumps({x:item[x] for x in ['snapshot','radius','roll_deg','position_error_m','axis_error_rad']}|{}) if False else json.dumps(dict(snapshot=k,radius=radius,roll=roll_deg,pos=item['position_error_m'],axis=angle,gaps={x:g['minimums'][x]['gap_m'] for x in g['minimums']})),flush=True)
    write(out/'results.json',dict(scope='30 static IK/geometry combinations at TWO existing failed c2 poses; no complete candidate',source=str(src),cases=results,self_exclusions=G.excluded,finite_walls=G.wallmeta))
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
