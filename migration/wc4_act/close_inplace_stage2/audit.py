"""Source-piece external geometry audit with bounded interpolation diagnostics."""
import argparse,json,sys,os,math,time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from candidate import ROOT,read,write,sha,emit,baseT,NAMES,LEGS,HOME
from fullcycle_geometry import URDF,transform,apply
from fullcycle_collision import collision_pieces,signed_gap
from newexpert_stage2.finite_walls import walls

def fast_gap(A,B):
    """Same SAT axes, vectorized normalization; duplicate axes need not be removed."""
    av,an,ae=A;bv,bn,be=B
    broad=np.maximum(bv.min(0)-av.max(0),av.min(0)-bv.max(0))
    if broad.max()>.05:return float(broad.max())
    cross=np.cross(ae[:,None,:],be[None,:,:]).reshape(-1,3)
    axes=np.concatenate([an,bn,cross]);norm=np.linalg.norm(axes,axis=1);axes=axes[norm>1e-9]/norm[norm>1e-9,None]
    aa=av@axes.T;bb=bv@axes.T
    return float(np.maximum(bb.min(0)-aa.max(0),aa.min(0)-bb.max(0)).max())

def audit(out):
    rows=[json.loads(l) for l in (out/'trace.jsonl').open()];cfg=read(out/'config.json')['global_config'];c=read(out/'config.json')['candidate']
    source=ROOT/cfg['source_run'];effective=read(source/'effective-config.json')
    rp=ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf';dp=ROOT/'high-level/data/asset/door_set/wc4/model.urdf'
    robot=URDF(rp);door=URDF(dp);R=collision_pieces(rp);D=collision_pieces(dp)
    W,wallmeta=walls(effective,read(dp.parent/'bounding_box.json'),0.)
    write(out/'finite-walls.json',wallmeta)
    inventory=[dict(link=p.link,piece=p.name,kind=p.kind,vertices=len(p.vertices)) for p in R]
    write(out/'geometry-method.json',dict(robot_inventory=inventory,door_inventory=[dict(link=p.link,piece=p.name,kind=p.kind) for p in D],
        method='URDF collision elements separately; mesh convex hull, circumscribed cylinder 32-gon, sphere outer box; exact SAT sign for these polytopes. Positive gap is a projected distance lower bound. Negative gap is source-envelope overlap, not physical penetration.',
        padding='No PhysX contact offset/cooked-shape inference. Finite wall geometry from source construction.',
        sphere_mesh_overlap='Unresolved unless supported by exact box-box overlap or actual mesh surface detail. All pairs retained for localization.',
        trace_sha256=sha(out/'trace.jsonl'),source_sha256={str(p):sha(p) for p in [rp,dp,ROOT/'migration/wc4_act/fullcycle_collision.py',ROOT/'migration/wc4_act/newexpert_stage2/finite_walls.py']}))
    categories=['pad_panel','arm_panel','base_panel','handle','frame','wall'];cache={}
    allpieces=D+[p for p,_ in W]
    labels=np.empty((len(R),len(allpieces)),dtype=int)
    for ia,p in enumerate(R):
        isarm=p.link.startswith('link') or p.link.startswith('gripper') or p.link.startswith('wrist')
        for ib,d in enumerate(allpieces):
            cat='wall' if ib>=len(D) else ('pad_panel' if p.link=='gripperStator' and p.name=='2' else 'arm_panel' if isarm else 'base_panel') if d.link=='door_panel' else 'handle' if 'handle' in d.link or 'lever' in d.link else 'frame'
            labels[ia,ib]=categories.index(cat)
    masks=[np.flatnonzero(labels.ravel()==k) for k in range(len(categories))]
    progress=(out/'geometry-progress.jsonl').open('x')
    def evaluate(i):
        if i in cache:return cache[i]
        r=rows[i];poses=robot.fk(dict(LEGS,**dict(zip(NAMES,r['q']))),baseT(r['base']))
        doors=door.fk({'door_hinge_joint':r['theta'],'handle_spindle_joint':0},transform([2.61,0,.01],[0,0,math.pi/2]))
        AW=[p.world(poses[p.link]) for p in R];BW=[p.world(doors[p.link]) for p in D]+[b for _,b in W];pieces=D+[p for p,_ in W]
        amin=np.array([a[0].min(0) for a in AW]);amax=np.array([a[0].max(0) for a in AW]);bmin=np.array([b[0].min(0) for b in BW]);bmax=np.array([b[0].max(0) for b in BW])
        broad=np.maximum(bmin[None]-amax[:,None],amin[:,None]-bmax[None]).max(2)
        gaps=broad.copy()
        for ia,ib in np.argwhere(broad<=.05):gaps[ia,ib]=fast_gap(AW[ia],BW[ib])
        def entry(ia,ib):
            p=R[ia];d=pieces[ib]
            return dict(gap_m=float(gaps[ia,ib]),robot_link=p.link,robot_piece=p.name,robot_kind=p.kind,obstacle_link=d.link,obstacle_piece=d.name,obstacle_kind=d.kind)
        best={}
        for cat,mask in zip(categories,masks):
            ia,ib=np.unravel_index(mask[np.argmin(gaps.ravel()[mask])],gaps.shape);best[cat]=entry(ia,ib)
        overlaps=[dict(category=categories[labels[ia,ib]],**entry(ia,ib)) for ia,ib in np.argwhere(gaps<0)]
        item=dict(index=i,t=r['t'],phase=r['phase'],minimums=best,overlaps=overlaps)
        cache[i]=item;progress.write(json.dumps(item)+'\n');progress.flush();return item
    period=cfg['geometry_period_steps'];indices=set(range(0,len(rows),period));indices.add(len(rows)-1)
    for i in range(1,len(rows)):
        if rows[i]['phase']!=rows[i-1]['phase']:indices.update([i-1,i])
    for k,i in enumerate(sorted(indices)):
        evaluate(i)
        if k%100==0:emit(dict(stage='geometry',candidate=c['id'],checked=k,total=len(indices)))
    # Every near/uncertain coarse interval is refined to the actual 50Hz grid.
    # Radius 2m bounds source robot points about each ancestor axis for this asset;
    # 1.2m bounds door/handle radius about vertical hinge. Loose, not optimized.
    b=np.array([r['base'] for r in rows]);q=np.array([r['q'] for r in rows]);theta=np.array([r['theta'] for r in rows])
    motion=np.linalg.norm(np.diff(b[:,:2],axis=0),axis=1)+2*abs(np.diff(b[:,2]))+2*np.sum(abs(np.diff(q,axis=0)),axis=1)+1.2*abs(np.diff(theta))
    intervals=[];ordered=sorted(indices)
    for l,h in zip(ordered[:-1],ordered[1:]):
        bound=float(motion[l:h].sum());clear=min(cache[l]['minimums'][k]['gap_m'] for k in categories if k!='pad_panel')
        if clear<=bound+.001:
            for i in range(l+1,h):evaluate(i)
    ordered=sorted(cache)
    for l,h in zip(ordered[:-1],ordered[1:]):
        bound=float(motion[l:h].sum());gap=min(cache[l]['minimums'][k]['gap_m'] for k in categories if k!='pad_panel')
        intervals.append(dict(first=l,last=h,motion_bound_m=bound,external_lower_gap_m=gap,certified_separated_under_linear_interpolation=bool(gap>bound)))
    progress.close()
    minima={};negative=[]
    for i in ordered:
        item=cache[i]
        for k,v in item['minimums'].items():
            key=item['phase']+'/'+k
            if key not in minima or v['gap_m']<minima[key]['gap_m']:minima[key]=dict(index=i,t=item['t'],**v)
        negative.extend(dict(index=i,t=item['t'],phase=item['phase'],**x) for x in item['overlaps'])
    with (out/'clearance.jsonl').open('x') as f:
        for i in ordered:f.write(json.dumps(cache[i])+'\n')
    with (out/'overlaps.jsonl').open('x') as f:
        for x in negative:f.write(json.dumps(x)+'\n')
    write(out/'intervals.json',intervals)
    summary=dict(samples=len(cache),total_50hz_rows=len(rows),minimums=minima,overlap_records=len(negative),
        unresolved_intervals=sum(not x['certified_separated_under_linear_interpolation'] for x in intervals),
        maximum_checked_interval_s=max((x['last']-x['first'])*cfg['dt'] for x in intervals),
        statement='No physical collision claim. Pad contact excluded only from interpolation separation certification, not from raw logs. Negative source hulls and unresolved intervals require review. No self-collision or cooked-mesh certification.')
    write(out/'clearance-summary.json',summary);emit(dict(candidate=c['id'],samples=summary['samples'],overlap_records=len(negative),unresolved_intervals=summary['unresolved_intervals']))
    plot(out,rows,cache,robot,door,R,D,W)

def plot(out,rows,cache,robot,door,R,D,W):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c=read(out/'config.json')['candidate'];time=np.array([r['t'] for r in rows]);b=np.array([r['base'] for r in rows])
    fig,axes=plt.subplots(4,1,figsize=(12,12),sharex=True)
    axes[0].plot(time,[r['tracking_m'] if r['tracking_m'] is not None else np.nan for r in rows]);axes[0].axhline(.008,color='r',ls='--');axes[0].set_ylabel('Target error (m)')
    axes[1].plot(time,[r['joint_slack'] for r in rows],label='joint slack rad');axes[1].plot(time,[r['joint_rate_max'] for r in rows],label='max joint rate rad/s');axes[1].axhline(.7,color='r',ls='--');axes[1].legend()
    times=[cache[i]['t'] for i in sorted(cache)]
    for cat in ['pad_panel','arm_panel','base_panel','handle','frame','wall']:axes[2].plot(times,[cache[i]['minimums'][cat]['gap_m'] for i in sorted(cache)],label=cat)
    axes[2].set_ylim(-.15,.6);axes[2].axhline(0,color='k',lw=.7);axes[2].set_ylabel('Source gap (m)');axes[2].legend(ncol=3,fontsize=8)
    axes[3].plot(time,[r['action'][0] for r in rows],label='v m/s');axes[3].plot(time,[r['action'][1] for r in rows],label='w rad/s');axes[3].plot(time,b[:,0]-1.259999752,label='x - endpoint m');axes[3].legend();axes[3].set_xlabel('Source-aligned synthetic time (s)')
    for ax in axes:
        ax.grid(alpha=.2)
        for i in range(1,len(rows)):
            if rows[i]['phase']!=rows[i-1]['phase']:ax.axvline(time[i],color='grey',alpha=.2)
    fig.suptitle(c['id']+' | attempted kinematics / source-envelope gaps | NOT physical execution');fig.tight_layout();fig.savefig(out/'curves.png',dpi=150);plt.close(fig)
    phases=['traverse_continuation','fold','turn','deploy_arc','edge','seek','close','release','retract','restore']
    selected=[]
    for phase in phases:
        rr=[i for i,r in enumerate(rows) if r['phase']==phase]
        selected.append(rr[-1])
    rr=[i for i,r in enumerate(rows) if r['phase']=='close'];selected.insert(7,rr[len(rr)//2]);selected.insert(8,rr[3*len(rr)//4])
    fig,axes=plt.subplots(3,4,figsize=(18,13))
    def draw(ax,pts,color,alpha):
        xy=np.unique(np.round(pts[:,:2],7),axis=0)
        if len(xy)>=3:
            try:xy=xy[ConvexHull(xy).vertices];ax.fill(xy[:,0],xy[:,1],color=color,alpha=alpha)
            except Exception:pass
    for ax,i in zip(axes.flat,selected):
        r=rows[i];poses=robot.fk(dict(LEGS,**dict(zip(NAMES,r['q']))),baseT(r['base']));doors=door.fk({'door_hinge_joint':r['theta'],'handle_spindle_joint':0},transform([2.61,0,.01],[0,0,math.pi/2]))
        for p in R:draw(ax,p.world(poses[p.link])[0],'#2075ac' if p.link.startswith('link') or p.link.startswith('gripper') else '#78848d',.35)
        for p in D:draw(ax,p.world(doors[p.link])[0],'#ab6d31',.25)
        for _,w in W:draw(ax,w[0],'#777',.2)
        ax.plot(b[:i+1,0],b[:i+1,1],color='#005d88',lw=1);ax.scatter(*r['pad'][:2],color='#bd2447',s=15)
        if r['target'] is not None:ax.scatter(*r['target'][:2],marker='x',color='k',s=25)
        ax.set(title=f"{r['phase']}  t={r['t']:.2f}s",xlim=(.35,2.85),ylim=(-1.05,1.25));ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle(c['id']+' | source collision pieces projected from 3D | projection overlap is NOT penetration\nSynthetic q after failed IK is retained; red actual-FK pad, black requested point',fontsize=14);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'key-poses.png',dpi=150);plt.close(fig)
    write(out/'key-poses.json',[dict(index=i,**rows[i]) for i in selected])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate-dir',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden')
    audit(Path(a.candidate_dir))
