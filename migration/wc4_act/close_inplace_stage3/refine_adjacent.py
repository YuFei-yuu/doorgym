"""Localize the negative adjacent-pair witnesses, without any new trajectory."""
import argparse,os,json
from pathlib import Path
import numpy as np
from geometry import Geometry,HOME,NAMES,baseT,apply,write,read
from fullcycle_geometry import origin
from review import collision,samples,contained
def main(run):
    out=run/'single';G=Geometry();s=read(out/'adjacent-self-summary.json');rows=[json.loads(l) for l in (out/'trace.jsonl').open()];states=[dict(t=G.anchor['t'],phase='measured_anchor',q=G.anchor['arm_q'])]+rows;result=[]
    for key,x in s['minima'].items():
        if x['gap_m']>=0:continue
        j=NAMES.index(x['joint']);i=next(i for i,r in enumerate(states) if abs(r['q'][j]-x['value'])<1e-12)
        q=HOME.copy();q[j]=x['value'];poses=G.robot.fk(dict(G.legs,**dict(zip(NAMES,q))),baseT([0,0,0]))
        a,ag,ar=collision(G,*x['a'][:2]);b,bg,br=collision(G,*x['b'][:2]);pa=samples(ag,ar);world=apply(poses[x['a'][0]]@origin(a),pa);local=apply(np.linalg.inv(poses[x['b'][0]]@origin(b)),world);mask,method=contained(local,bg,br)
        result.append(dict(**x,first_source_index=i,time_s=states[i]['t'],phase=states[i]['phase'],original_points=len(pa),inside_count=int(mask.sum()),method=method,canonical_q=q.tolist(),examples_canonical_world=world[np.flatnonzero(mask)[:12]].tolist(),
            interpretation='Existing directly connected joint-component geometry; source overlap/uncertainty is recorded, not automatically a new forbidden physical collision.'))
    write(out/'adjacent-self-localization.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden')
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
