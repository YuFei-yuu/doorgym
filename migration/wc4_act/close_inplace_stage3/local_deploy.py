"""One local deployment arc and three offset endpoints near c2 failed seek pose."""
import argparse,json,os,math
from pathlib import Path
import numpy as np
from geometry import Geometry,Solver,ROOT,HOME,baseT,write,read
from close_inplace_stage2.candidate import smooth
from close_inplace_stage2.refine_overlap import slab_mesh

def main(run,name,offsets):
    out=run/name;out.mkdir(exist_ok=False);G=Geometry();S=Solver(G.robot,'corner')
    cases=read(run/'local-01/results.json')['cases'];case=next(x for x in cases if x['snapshot']==0 and x['radius']==.905 and x['roll_deg']==0)
    B=case['base'];theta=case['theta'];n=np.array([math.cos(theta),math.sin(theta),0]);S.axis=n.copy();seed=np.array(case['q'])
    sols=[]
    for offset in offsets:
        point=np.array([2.467-.905*math.sin(theta)-(.002+offset)*math.cos(theta),-.463+.905*math.cos(theta)-(.002+offset)*math.sin(theta),.85])
        q,e=S.solve_point(B,point,n,seed,160);g=G.check(B,np.r_[q,HOME[-1]],theta)
        sols.append(dict(offset=offset,target=point.tolist(),q=np.r_[q,HOME[-1]].tolist(),ik=e,geometry=g));print(json.dumps(sols[-1]),flush=True)
    good=[s for s in sols if s['ik']['position_error_m']<.008 and s['ik']['axis_error_rad']<.03 and s['geometry']['minimums']['external']['gap_m']>0 and s['geometry']['minimums']['self']['gap_m']>0]
    selected=next((s for s in good if s['offset']>0),None);arc=[]
    if selected:
        for u in np.linspace(0,1,41):
            q=(1-smooth(u))*HOME+smooth(u)*np.array(selected['q']);g=G.check(B,q,theta);arc.append(dict(fraction=float(u),q=q.tolist(),geometry=g))
    # Actual source surface: point [.905,.002,.84] must lie on a slab triangle.
    tri,meta=slab_mesh(G.door.path.parent/'assets/meshes/door_slab.obj');p=np.array([.905,.002,.84]);on=[]
    for i,T in enumerate(tri):
        a,b,c=T;v=b-a;w=c-a;normal=np.cross(v,w);norm=np.linalg.norm(normal)
        if norm<1e-12:continue
        if abs(np.dot(normal/norm,p-a))>1e-7:continue
        uv=np.linalg.lstsq(np.column_stack([v,w]),p-a,rcond=None)[0]
        if min(uv)>=-1e-7 and sum(uv)<=1+1e-7:on.append(dict(triangle=i,normal=(normal/norm).tolist(),barycentric=[float(1-sum(uv)),*uv.tolist()]))
    write(out/'results.json',dict(scope='three static offsets plus ONE local joint arc from home to selected hover; no complete route',radius=.905,contact_height=.85,contact_surface_local=p.tolist(),source_triangles_containing_point=on,slab_mesh=meta,offsets=sols,selected=selected,arc=arc))
    print(json.dumps(dict(surface_triangles=on,selected_offset=None if selected is None else selected['offset'],arc_samples=len(arc),min_external=min([x['geometry']['minimums']['external']['gap_m'] for x in arc] or [0]),min_self=min([x['geometry']['minimums']['self']['gap_m'] for x in arc] or [0]))),flush=True)
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden')
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--name',default='local-02');p.add_argument('--offsets',type=float,nargs='+',default=[.04,.02,0.]);a=p.parse_args();main(a.run,a.name,a.offsets)
