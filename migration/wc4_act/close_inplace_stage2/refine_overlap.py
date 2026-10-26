"""Localize existing source-envelope overlaps using original slab triangles.

Only existing trajectory rows are read. No IK, path changes, or physics.
Two non-axis ray parities on a welded closed mesh localize source points inside
the actual slab; absence of such sample points is not a separation proof.
"""
import argparse,json,sys,itertools
from pathlib import Path
from collections import Counter
import numpy as np
from candidate import ROOT,read,write,URDF,transform,apply,baseT,LEGS,NAMES
from fullcycle_geometry import origin,vec
from fullcycle_collision import mesh_points

def slab_mesh(path):
    vertices=[];faces=[]
    for line in path.read_text().splitlines():
        if line.startswith('v '):vertices.append(list(map(float,line.split()[1:4])))
        if line.startswith('f '):
            ids=[int(x.split('/')[0])-1 for x in line.split()[1:]]
            faces.extend([[ids[0],ids[i],ids[i+1]] for i in range(1,len(ids)-1)])
    v=np.array(vertices);f=np.array(faces);_,inv=np.unique(np.round(v,8),axis=0,return_inverse=True)
    welded=inv[f];edges=Counter(tuple(sorted((int(a),int(b)))) for tri in welded for a,b in zip(tri,np.roll(tri,-1)))
    return v[f],dict(vertices=len(v),triangles=len(f),welded_edges=len(edges),all_edges_have_two_incident_triangles=all(n==2 for n in edges.values()))
def inside(points,tri,direction):
    d=np.array(direction);d=d/np.linalg.norm(d);e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    h=np.cross(d,e2);det=np.einsum('ij,ij->i',e1,h);ok=abs(det)>1e-11;inv=np.zeros(len(det));inv[ok]=1/det[ok]
    result=[];dist=[]
    for p in points:
        s=p-tri[:,0];u=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,e1);w=(q@d)*inv;t=np.einsum('ij,ij->i',e2,q)*inv
        hits=t[ok&(u>=-1e-9)&(w>=-1e-9)&(u+w<=1+1e-9)&(t>1e-7)]
        # Merge shared-edge crossings, so triangulation diagonals count once.
        hits=np.unique(np.round(hits,8));result.append(len(hits)%2==1);dist.append(float(hits.min()) if len(hits) else None)
    return np.array(result),dist
def main(run):
    robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');door=URDF(ROOT/'high-level/data/asset/door_set/wc4/model.urdf')
    tri,meta=slab_mesh(door.path.parent/'assets/meshes/door_slab.obj');assert meta['all_edges_have_two_incident_triangles']
    reports=[];views=[]
    for cid in ['c0','c1','c2']:
        out=run/cid;rows=[json.loads(l) for l in (out/'trace.jsonl').open()];g=read(out/'clearance-summary.json')
        for key,x in g['minimums'].items():
            if not key.endswith('/arm_panel') or x['gap_m']>=-.0001:continue
            row=rows[x['index']];link=robot.links[x['robot_link']];collisions=link.findall('collision')
            c=next(c for i,c in enumerate(collisions) if c.get('name',str(i))==x['robot_piece']);geom=list(c.find('geometry'))[0]
            if geom.tag=='mesh':p=mesh_points(robot.path.parent/geom.get('filename'))*vec(geom.get('scale'),'1 1 1');method='original collision mesh vertices'
            elif geom.tag=='box':
                size=vec(geom.get('size'))/2;p=np.array(list(itertools.product(*[np.linspace(-s,s,5) for s in size])));method='5x5x5 samples of exact source box volume'
            else:continue
            p=np.unique(np.round(p,9),axis=0);p=apply(origin(c),p)
            rp=robot.fk(dict(LEGS,**dict(zip(NAMES,row['q']))),baseT(row['base']));dp=door.fk({'door_hinge_joint':row['theta'],'handle_spindle_joint':0},transform([2.61,0,.01],[0,0,np.pi/2]))
            world=apply(rp[x['robot_link']],p);local=apply(np.linalg.inv(dp['door_panel']),world)
            a,da=inside(local,tri,[1,.317,.193]);b,db=inside(local,tri,[-.283,1,.467]);both=a&b
            ids=np.flatnonzero(both);examples=[dict(robot_point_world=world[i].tolist(),door_local_point=local[i].tolist(),ray_exit_distances_m=[da[i],db[i]]) for i in ids[:12]]
            reports.append(dict(candidate=cid,phase=key.split('/')[0],row_index=x['index'],time_s=row['t'],source_pair=x,
                sampled_points=len(p),method=method,both_ray_inside_count=int(both.sum()),ray_disagreement_count=int((a!=b).sum()),examples=examples,
                conclusion='Original robot geometry sample points lie inside the closed source door mesh at this SYNTHETIC pose; not merely convex-hull overlap' if len(ids) else 'No sampled point established inside original slab; conservative overlap remains unresolved, not cleared',
                limitations='Not physical contact/force/depth, not cooked mesh verification; sampled-point method cannot prove no intersection.'))
            if cid=='c2' and key.split('/')[0] in ['seek','close']:views.append((key,x,local,both))
    write(run/'source-mesh-localization.json',dict(slab_mesh=meta,results=reports));print(json.dumps(reports,indent=2))
    if views:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        fig,axes=plt.subplots(len(views),2,figsize=(12,5*len(views)),squeeze=False)
        for row,(key,x,local,both) in zip(axes,views):
            for ax,dims in zip(row,[(0,1),(1,2)]):
                ax.add_collection(PolyCollection(tri[:,:,dims],facecolor='#c89c6a',edgecolor='#8f643b',alpha=.15,lw=.4))
                ax.scatter(local[:,dims[0]],local[:,dims[1]],s=7,color='#287eaa',alpha=.6,label='Original robot geometry samples')
                ax.scatter(local[both,dims[0]],local[both,dims[1]],s=18,color='#c92e3c',label='Inside original closed slab mesh')
                ax.set(xlabel=['Door-local radial x (m)','Door-local thickness y (m)'][dims[0]==1],ylabel='Door-local y (m)' if dims[1]==1 else 'Door-local height z (m)',title=f"c2 {key} | {x['robot_link']} / {x['robot_piece']} | t={x['t']:.2f}s")
                lo=local[:,dims].min(0);hi=local[:,dims].max(0);ax.set_xlim(lo[0]-.04,hi[0]+.04);ax.set_ylim(lo[1]-.04,hi[1]+.04);ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Source-mesh localization at synthetic poses | neither physical contact nor cooked-shape penetration depth');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(run/'source-overlap-detail.png',dpi=160)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
