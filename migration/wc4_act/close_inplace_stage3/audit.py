"""Read one frozen candidate; adaptive source geometry and self-collision audit."""
import argparse,os,json,math,sys,time
from pathlib import Path
import numpy as np
from geometry import Geometry,ROOT,HOME,NAMES,read,write,sha

def audit(run,resume=False):
    out=run/'single';rows=[json.loads(l) for l in (out/'trace.jsonl').open()];G=Geometry();cfg=read(out/'config.json')
    b=read(out/'measured-anchor.json')['row'];anchor=dict(t=b['t'],phase='measured_anchor',base=[*b['actual_base'][:2],math.pi],q=b['arm_q'],theta=b['door_q'][0]);states=[anchor]+rows
    baseline=G.check([1.2599997520446777,.18000000715255737,math.pi],HOME,anchor['theta'])
    write(out/'self-baseline.json',dict(home=baseline,excluded_pairs=G.excluded,included_pairs=len(G.selfpairs),exclusion_rule='same rigid assembly or directly joint-adjacent components only; inherited home overlaps NOT excluded'))
    parent={j[3]:j for j in G.robot.joints};ancestors={};radius={}
    for name,points in G.robot.clouds.items():
        axes=[];current=name;r=float(np.linalg.norm(points,axis=1).max())
        while current in parent:
            j=parent[current];r+=float(np.linalg.norm(j[4][:3,3]));current=j[2]
            if j[0] in NAMES:axes.append(NAMES.index(j[0]))
        ancestors[name]=axes;radius[name]=r
    assert max(radius.values())<2
    prior_method=read(out/'geometry-method.json') if (out/'geometry-method.json').exists() else None
    if resume:
        assert prior_method['trace_sha256']==sha(out/'trace.jsonl')
        write(out/'geometry-method-before-resume.json',prior_method)
    write(out/'geometry-method.json',dict(trace_sha256=sha(out/'trace.jsonl'),leg_binding=G.far_binding,finite_walls=G.wallmeta,resumed_from_saved_checks=resume,
        all_visual_and_collision_triangle_radius_bound_m=max(radius.values()),robot_radius_used_m=2,door_radius_used_m=1.2,
        source_method='Separate source collision pieces, convex meshes, circumscribed cylinders and sphere boxes. SAT gap is projected lower separation bound; negative is not physical penetration.',
        interpolation='Piecewise linear interpolation of this one stored q/base/door sequence, no IK or new candidate. Coarse 0.1s plus boundaries; uncertain positive gaps recursively split to <=2.5ms. Negative envelopes are kept unresolved, not refined away.',
        self_exclusion='same fixed component or directly revolute-adjacent only; full list in self-baseline.json'))
    cache={};intervals=[]
    if resume:
        with (out/'geometry.jsonl').open() as f:
            for line in f:
                r=json.loads(line);cache[r['index']]=r
    stream=(out/'geometry.jsonl').open('a' if resume else 'x')
    def state(x):
        i=int(math.floor(x));f=x-i
        if f<1e-12:return states[i]
        a=states[i];b=states[i+1]
        return dict(t=(1-f)*a['t']+f*b['t'],phase=b['phase'],base=((1-f)*np.array(a['base'])+f*np.array(b['base'])).tolist(),q=((1-f)*np.array(a['q'])+f*np.array(b['q'])).tolist(),theta=(1-f)*a['theta']+f*b['theta'])
    def evaluate(x):
        x=float(x)
        if x not in cache:
            r=state(x);g=G.check(r['base'],r['q'],r['theta']);item=dict(index=x,t=r['t'],phase=r['phase'],base=r['base'],q=r['q'],theta=r['theta'],interpolated=not x.is_integer(),**g)
            cache[x]=item;stream.write(json.dumps(item)+'\n');stream.flush()
        return cache[x]
    def bounds(l,h):
        points=[l]+[float(i) for i in range(math.floor(l)+1,math.ceil(h))]+[h];ss=[state(x) for x in points]
        B=np.array([s['base'] for s in ss]);Q=np.array([s['q'] for s in ss]);th=np.array([s['theta'] for s in ss])
        dx=np.diff(B[:,:2],axis=0);yaw=float(abs(np.diff(B[:,2])).sum());joints=abs(np.diff(Q,axis=0)).sum(0)
        transl=float(np.linalg.norm(dx,axis=1).sum());door=float(abs(np.diff(th)).sum())
        return transl+2*yaw+2*float(joints.sum())+1.2*door,4*float(joints.sum()),float(np.maximum(dx[:,0],0).sum()),yaw,joints
    def effective_ext(r):
        e=r['minimums']['external']['gap_m']
        if r['phase'] not in ['seek','close','settle','release']:e=min(e,r['minimums']['intended']['gap_m'])
        return e
    def interval(l,h):
        left=evaluate(l);right=evaluate(h);extb,selfb,positive_x,yaw,joints=bounds(l,h)
        eg=effective_ext(left);sg=left['minimums']['self']['gap_m']
        far_ok=all(2.5-left['link_max_x'][n]>positive_x+2*yaw+2*float(joints[ids].sum()) for n,ids in ancestors.items())
        ext_ok=eg>extb;self_ok=sg>selfb
        need=(0<=eg and not ext_ok) or (0<=sg and not self_ok) or not far_ok
        if need and h-l>.125+1e-10:
            if h-l>1:mid=float((math.floor(l)+math.ceil(h))//2)
            else:mid=(l+h)/2
            if l<mid<h:interval(l,mid);interval(mid,h);return
        intervals.append(dict(first=l,last=h,duration_s=(h-l)*.02,external_certified=bool(ext_ok),self_certified=bool(self_ok),far_certified=bool(far_ok),external_gap=eg,self_gap=sg,external_motion_bound=extb,self_motion_bound=selfb))
    indices=set(range(0,len(states),5));indices.add(len(states)-1)
    for i in range(1,len(states)):
        if states[i]['phase']!=states[i-1]['phase']:indices.update([i-1,i])
    indices=sorted(indices)
    for n,(l,h) in enumerate(zip(indices[:-1],indices[1:])):
        interval(float(l),float(h))
        if n%100==0:print(json.dumps(dict(coarse_intervals=n,total=len(indices)-1,geometry_samples=len(cache))),flush=True)
    stream.close();minima={};negative=[]
    for r in cache.values():
        for cat,x in r['minimums'].items():
            key=r['phase']+'/'+cat
            if key not in minima or x['gap_m']<minima[key]['gap_m']:minima[key]=dict(index=r['index'],t=r['t'],**x)
        for x in r['overlaps']:negative.append(dict(index=r['index'],t=r['t'],phase=r['phase'],**x))
    with (out/'overlaps.jsonl').open('x') as f:
        for x in negative:f.write(json.dumps(x)+'\n')
    write(out/'intervals.json',intervals)
    result=dict(source_and_interpolated_samples=len(cache),source_sample_count=sum(not r['interpolated'] for r in cache.values()),minimums=minima,
        unresolved_external_intervals=sum(not x['external_certified'] for x in intervals),unresolved_self_intervals=sum(not x['self_certified'] for x in intervals),unresolved_far_intervals=sum(not x['far_certified'] for x in intervals),
        sampled_far_return_count=sum(not r['far'] for r in cache.values()),minimum_far_margin_m=2.5-max(r['whole_max_x'] for r in cache.values()),
        statement='Certificates apply only to declared kinematic interpolation and source envelopes, not physical tracking or cooked shapes. Retain inherited and new self overlaps separately.')
    write(out/'geometry-summary.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden')
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();audit(a.run,a.resume)
