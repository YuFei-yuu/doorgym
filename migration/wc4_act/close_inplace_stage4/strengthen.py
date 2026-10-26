"""Stronger SAT bounds on already stored uncertain intervals; no new IK/targets."""
import argparse,os,json,math
from pathlib import Path
import numpy as np
from common import *
from close_inplace_stage2.audit import fast_gap
from close_inplace_stage3.geometry import transform
def main(run):
    out=run/'trajectory';rows=[json.loads(l) for l in (out/'trace.jsonl').open()];start=read(run/'start.json')['row'];states=[dict(t=0.,q=start['q'],theta=start['theta'])]+rows;times=np.array([r['t'] for r in states]);B=start['base'];G=Geometry();intervals=[json.loads(l) for l in (out/'intervals.jsonl').open()]
    def at(t):
        i=int(np.searchsorted(times,t,side='right')-1);i=max(0,min(i,len(states)-1))
        if i==len(states)-1 or abs(times[i]-t)<1e-10:return np.array(states[i]['q']),states[i]['theta']
        f=(t-times[i])/(times[i+1]-times[i]);return (1-f)*np.array(states[i]['q'])+f*np.array(states[i+1]['q']),(1-f)*states[i]['theta']+f*states[i+1]['theta']
    cache={};results=[]
    for k,x in enumerate(intervals):
        if x['external_certified']:continue
        q,a=at(x['start']);q1,a1=at(x['end']);bound=2*float(abs(q1-q).sum())+1.2*abs(a1-a)
        if x['start'] not in cache:
            rp=G.robot.fk(dict(G.legs,**dict(zip(NAMES,q))),baseT(B));dp=G.door.fk({'door_hinge_joint':a,'handle_spindle_joint':0},transform([2.61,0,.01],[0,0,math.pi/2]));A=[p.world(rp[p.link]) for p in G.R];D=G.D+[p for p,_ in G.W];BB=[p.world(dp[p.link]) for p in G.D]+[b for _,b in G.W];best=dict(gap_m=100.)
            for i,p in enumerate(G.R):
                for j,d in enumerate(D):
                    if p.link=='gripperStator' and p.name=='2' and d.link=='door_panel':continue
                    broad=float(np.maximum(BB[j][0].min(0)-A[i][0].max(0),A[i][0].min(0)-BB[j][0].max(0)).max())
                    if broad>.05 and broad>=best['gap_m']:continue
                    gap=fast_gap(A[i],BB[j])
                    if gap<best['gap_m']:best=dict(gap_m=gap,a=[p.link,p.name],b=[d.link,d.name])
            cache[x['start']]=best
        best=cache[x['start']];results.append(dict(start=x['start'],end=x['end'],phase=x['phase'],bound_m=float(bound),minimum=best,certified=bool(best['gap_m']>bound)))
    write(out/'external-interval-strengthening.json',dict(trace_sha256=sha(out/'trace.jsonl'),scope='Only stronger complete-SAT projected bounds at existing interval endpoints; no new robot target, seed, IK, trajectory or policy',intervals_rechecked=len(results),certified=sum(x['certified'] for x in results),still_unproved=sum(not x['certified'] for x in results),results=results,
        no_gate_rewrite=True,meaning='Original failed prefix and diagnostic labels remain; this cannot turn the reference into ordered execution. Selected contact body and self-baseline proofs are separate.'))
    print(json.dumps(dict(rechecked=len(results),certified=sum(x['certified'] for x in results),still_unproved=sum(not x['certified'] for x in results))))
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('hide CUDA')
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
