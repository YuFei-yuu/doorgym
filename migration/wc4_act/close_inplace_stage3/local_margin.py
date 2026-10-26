"""Local height/yaw correction near failed seek; select one deployment, not a route."""
import argparse,json,os,math
from pathlib import Path
import numpy as np
from geometry import Geometry,Solver,ROOT,HOME,write,read
from close_inplace_stage2.candidate import smooth
def main(run):
    out=run/'local-03';out.mkdir(exist_ok=False);G=Geometry();S=Solver(G.robot,'corner')
    old=read(run/'local-02/results.json');base=np.array(read(run/'local-01/results.json')['cases'][0]['base']);theta=read(run/'local-01/results.json')['cases'][0]['theta'];n=np.array([math.cos(theta),math.sin(theta),0]);S.axis=n.copy()
    seed=np.array(old['offsets'][-1]['q']);cases=[]
    for yaw in [270,285]:
        B=base.copy();B[2]=math.radians(yaw)
        for height in [.90,.95]:
            endpoints=[]
            for off in [.04,0]:
                p=np.array([2.467-.905*math.sin(theta)-(.002+off)*math.cos(theta),-.463+.905*math.cos(theta)-(.002+off)*math.sin(theta),height]);q,e=S.solve_point(B,p,n,seed,160);g=G.check(B,np.r_[q,HOME[-1]],theta)
                endpoints.append(dict(offset=off,q=np.r_[q,HOME[-1]].tolist(),target=p.tolist(),ik=e,geometry=g,joint_slack=float(np.min(np.r_[q-S.lo,S.hi-q]))))
            good=all(x['ik']['position_error_m']<.008 and x['ik']['axis_error_rad']<.03 and x['geometry']['minimums']['external']['gap_m']>0 and x['geometry']['minimums']['self']['gap_m']>0 for x in endpoints)
            cases.append(dict(yaw=yaw,height=height,base=B.tolist(),theta=theta,endpoints=endpoints,eligible=good));print(json.dumps(cases[-1]),flush=True)
    eligible=[x for x in cases if x['eligible']];chosen=max(eligible,key=lambda c:min(x['joint_slack'] for x in c['endpoints'])) if eligible else None
    arc=[];seek=[]
    if chosen:
        qh=np.array(chosen['endpoints'][0]['q']);B=chosen['base']
        for u in np.linspace(0,1,41):
            q=(1-smooth(u))*HOME+smooth(u)*qh;arc.append(dict(fraction=float(u),q=q.tolist(),geometry=G.check(B,q,theta)))
        q=qh.copy();p0=np.array(chosen['endpoints'][0]['target']);p1=np.array(chosen['endpoints'][1]['target'])
        for u in np.linspace(0,1,41):
            p=(1-smooth(u))*p0+smooth(u)*p1;sol,e=S.solve_point(B,p,n,q,100);q=np.r_[sol,HOME[-1]];seek.append(dict(fraction=float(u),q=q.tolist(),target=p.tolist(),ik=e,geometry=G.check(B,q,theta)))
    write(out/'results.json',dict(scope='8 static IK endpoints at failed-seek neighborhood, one chosen local deployment arc and normal approach; NOT full routes',cases=cases,selected=chosen,arc=arc,seek=seek))
    print(json.dumps(dict(selected=None if chosen is None else dict(yaw=chosen['yaw'],height=chosen['height']),arc_min_external=min([r['geometry']['minimums']['external']['gap_m'] for r in arc]or[0]),arc_min_self=min([r['geometry']['minimums']['self']['gap_m'] for r in arc]or[0]),seek_min_external=min([r['geometry']['minimums']['external']['gap_m'] for r in seek]or[0]))),flush=True)
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden')
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
