"""Supplementary adjacent-component check on the SAME frozen q sequence.

Adjacent rigid components have relative geometry determined only by their
connecting joint. Canonical evaluation removes irrelevant upstream transforms.
Same rigid-assembly pieces are still listed separately, not motion collisions.
"""
import argparse,os,json,math
from pathlib import Path
import numpy as np
from geometry import Geometry,ROOT,HOME,NAMES,baseT,write,sha,separating_gap
def main(run):
    out=run/'single';G=Geometry();rows=[json.loads(l) for l in (out/'trace.jsonl').open()];b=G.anchor
    qvalues=np.array([b['arm_q']]+[r['q'] for r in rows]);groups={};rigid=[]
    connects={frozenset((G.components[j[2]],G.components[j[3]])):j[0] for j in G.robot.joints if j[1] in ('revolute','continuous')}
    for a,an,b,bn,_ in G.excluded:
        ca,cb=G.components[a],G.components[b]
        if ca==cb:rigid.append([a,an,b,bn]);continue
        joint=connects[frozenset((ca,cb))]
        if joint not in NAMES:continue
        groups.setdefault(joint,[]).append((next(p for p in G.R if p.link==a and p.name==an),next(p for p in G.R if p.link==b and p.name==bn)))
    minima={};per_joint={};stream=(out/'adjacent-self.jsonl').open('x')
    for joint,pairs in groups.items():
        vals=np.unique(qvalues[:,NAMES.index(joint)]);samples=[]
        for v in vals:
            q=HOME.copy();q[NAMES.index(joint)]=v;poses=G.robot.fk(dict(G.legs,**dict(zip(NAMES,q))),baseT([0,0,0]));items=[]
            for a,b in pairs:
                gap=separating_gap(a.world(poses[a.link]),b.world(poses[b.link]));key=joint+'/'+a.link+':'+a.name+'/'+b.link+':'+b.name
                item=dict(joint=joint,value=float(v),a=[a.link,a.name,a.kind],b=[b.link,b.name,b.kind],gap_m=gap);items.append(item)
                if key not in minima or gap<minima[key]['gap_m']:minima[key]=item
            sample=dict(joint=joint,value=float(v),minimum_gap_m=min(x['gap_m'] for x in items),pairs=items);samples.append(sample);stream.write(json.dumps(sample)+'\n')
        intervals=[dict(a=x['value'],b=y['value'],separated=bool(x['minimum_gap_m']>2*abs(y['value']-x['value']))) for x,y in zip(samples[:-1],samples[1:])]
        per_joint[joint]=dict(source_values=len(vals),pairs=len(pairs),value_range=[float(vals.min()),float(vals.max())],unresolved_intervals=sum(not x['separated'] for x in intervals))
        print(json.dumps(dict(joint=joint,**per_joint[joint])),flush=True)
    stream.close();write(out/'adjacent-self-summary.json',dict(trace_sha256=sha(out/'trace.jsonl'),scope='all directly joint-adjacent moving component pairs omitted from main nonadjacent audit, evaluated at all unique stored joint values; canonical relative frame, not new motion candidate',groups=per_joint,minima=minima,same_rigid_assembly_pairs=rigid,interpretation='Joint-neighbor overlap may be the existing designed joint assembly/filter baseline. Listed explicitly, not declared physically forbidden or globally clear. Unresolved intervals retained.'))
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden')
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
