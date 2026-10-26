"""Small interface/limit/suffix endpoint check; no simulation or route search."""
import sys,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from close_inplace_stage4.common import FullPose,Geometry,point,pose_rotation
from close_inplace_stage5.controller import WAYPOINTS
from close_inplace_stage2.candidate import smooth

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    G=Geometry();S=FullPose(G.robot);q=WAYPOINTS[-1].copy();rows=[]
    for deg in [90,80,70,60,45,35,20,0]:
        theta=math.radians(deg);B=np.array([1.259999752+.45*smooth((35-deg)/35),.180000007,1.5*math.pi+math.pi/2*smooth((70-deg)/35)])
        roll=13.5+2*smooth((90-deg)/10);p=point({'radius':.905,'height':.85},theta);R=pose_rotation(theta,roll)
        q,e=S.solve(B,p,R,q,140);g=G.check(B,np.r_[q,-1.1775000095367432],theta)
        rows.append(dict(deg=deg,base=B.tolist(),q=q.tolist(),ik=e,minimums=g['minimums'],far=g['far']))
    (out/'endpoint-check.json').write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:print(json.dumps(r),flush=True)
if __name__=='__main__':main()
