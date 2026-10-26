"""Postprocess the one stored attempt; localize every distinct overlap pair."""
import argparse,json,math,os,itertools,struct,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
from geometry import Geometry,Solver,ROOT,HOME,NAMES,baseT,apply,write,read,sha
from fullcycle_geometry import origin,vec
from fullcycle_collision import mesh_points
from close_inplace_stage2.refine_overlap import inside

def collision(G,link,name,obstacle=False):
    u=G.door if obstacle else G.robot
    c=next(c for i,c in enumerate(u.links[link].findall('collision')) if c.get('name',str(i))==name)
    return c,list(c.find('geometry'))[0],u.path.parent
def samples(g,root):
    if g.tag=='mesh':return np.unique(np.round(mesh_points(root/g.get('filename'))*vec(g.get('scale'),'1 1 1'),9),axis=0)
    if g.tag=='box':
        d=vec(g.get('size'))/2;return np.array(list(itertools.product(*[np.linspace(-x,x,5) for x in d])))
    if g.tag=='cylinder':
        r=float(g.get('radius'));h=float(g.get('length'))/2
        return np.array([[rho*np.cos(a),rho*np.sin(a),z] for rho in [0,.5*r,.95*r] for a in np.linspace(0,2*np.pi,24,endpoint=False) for z in np.linspace(-.95*h,.95*h,5)])
    if g.tag=='sphere':
        r=float(g.get('radius'));return np.array([[rho*np.sin(v)*np.cos(a),rho*np.sin(v)*np.sin(a),rho*np.cos(v)] for rho in [0,.5*r,.95*r] for a in np.linspace(0,2*np.pi,16,endpoint=False) for v in np.linspace(0,np.pi,9)])
    raise ValueError(g.tag)
def triangles(path):
    if path.suffix.lower()=='.obj':
        v=[];f=[]
        for line in path.read_text().splitlines():
            if line.startswith('v '):v.append(list(map(float,line.split()[1:4])))
            if line.startswith('f '):
                ids=[int(x.split('/')[0]) for x in line.split()[1:]];ids=[x-1 if x>0 else len(v)+x for x in ids]
                f.extend([[ids[0],ids[i],ids[i+1]] for i in range(1,len(ids)-1)])
        return np.array(v)[np.array(f)]
    b=path.read_bytes();n=struct.unpack('<I',b[80:84])[0]
    if len(b)==84+n*50:
        dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);return np.frombuffer(b,dtype=dtype,count=n,offset=84)['v'].astype(float)
    return np.array([list(map(float,l.split()[1:4])) for l in b.decode().splitlines() if l.strip().startswith('vertex ')]).reshape(-1,3,3)
def contained(p,g,root):
    eps=1e-7
    if g.tag=='box':return np.all(abs(p)<vec(g.get('size'))/2-eps,axis=1),dict(method='exact source box volume')
    if g.tag=='cylinder':return (np.linalg.norm(p[:,:2],axis=1)<float(g.get('radius'))-eps)&(abs(p[:,2])<float(g.get('length'))/2-eps),dict(method='exact source cylinder volume')
    if g.tag=='sphere':return np.linalg.norm(p,axis=1)<float(g.get('radius'))-eps,dict(method='exact source sphere volume')
    tr=triangles(root/g.get('filename'))*vec(g.get('scale'),'1 1 1')
    _,inv=np.unique(np.round(tr.reshape(-1,3),8),axis=0,return_inverse=True);f=inv.reshape(-1,3)
    counts=Counter(tuple(sorted((int(a),int(b)))) for t in f for a,b in zip(t,np.roll(t,-1)))
    closed=all(n==2 for n in counts.values());result=np.zeros(len(p),bool)
    if not closed:return result,dict(method='source triangle mesh is not closed after welding; no inside certification',closed=False)
    ids=np.flatnonzero(np.all((p>tr.reshape(-1,3).min(0)-eps)&(p<tr.reshape(-1,3).max(0)+eps),axis=1))
    if not len(ids):return result,dict(method='original closed triangles; no sampled points inside mesh AABB',closed=True,triangles=len(tr),ray_disagreements=0)
    a,_=inside(p[ids],tr,[1,.317,.193]);b,_=inside(p[ids],tr,[-.283,1,.467]);result[ids]=a&b
    return result,dict(method='two-ray parity on original closed triangles',closed=True,triangles=len(tr),ray_disagreements=int((a!=b).sum()))

def review(run):
    out=run/'single';G=Geometry();rows=[json.loads(l) for l in (out/'trace.jsonl').open()];checks=[json.loads(l) for l in (out/'geometry.jsonl').open()];cache={x['index']:x for x in checks}
    overlaps=[json.loads(l) for l in (out/'overlaps.jsonl').open()];groups={}
    home_pairs={frozenset((tuple(x['a'][:2]),tuple(x['b'][:2]))):x['gap_m'] for x in read(out/'self-baseline.json')['home']['overlaps'] if x['category']=='self'}
    for x in overlaps:
        key=(x['category'],x['phase'],tuple(x['a']),tuple(x['b']))
        if key not in groups:groups[key]=dict(worst=x,phases=set(),count=0)
        groups[key]['phases'].add(x['phase']);groups[key]['count']+=1
        if x['gap_m']<groups[key]['worst']['gap_m']:groups[key]['worst']=x
    localized=[]
    for key,item in sorted(groups.items(),key=lambda kv:kv[1]['worst']['gap_m']):
        x=item['worst'];r=cache[x['index']];rp=G.robot.fk(dict(G.legs,**dict(zip(NAMES,r['q']))),baseT(r['base']));dp=G.door.fk({'door_hinge_joint':r['theta'],'handle_spindle_joint':0},__import__('geometry').transform([2.61,0,.01],[0,0,math.pi/2]))
        a,ag,ar=collision(G,*x['a'][:2]);pa=samples(ag,ar)
        if len(pa)>8000:pa=pa[np.linspace(0,len(pa)-1,8000,dtype=int)]
        world=apply(rp[x['a'][0]]@origin(a),pa)
        try:
            b,bg,br=collision(G,*x['b'][:2],obstacle=x['category']!='self')
            bt=(rp if x['category']=='self' else dp)[x['b'][0]]@origin(b)
            local=apply(np.linalg.inv(bt),world);mask,method=contained(local,bg,br)
        except (KeyError,StopIteration):
            wall=next((w for p,w in G.W if p.link==x['b'][0]),None)
            if wall is not None:
                mask=np.all((world>wall[0].min(0)+1e-7)&(world<wall[0].max(0)-1e-7),axis=1);method=dict(method='exact axis-aligned finite wall source box')
            else:mask=np.zeros(len(world),bool);method=dict(method='unresolved source lookup; retained for review')
        inside_count=int(mask.sum());pair_key=frozenset((tuple(x['a'][:2]),tuple(x['b'][:2])));baseline_pair=x['category']=='self' and pair_key in home_pairs
        localized.append(dict(pair=dict(category=x['category'],a=x['a'],b=x['b']),worst_gap_m=x['gap_m'],t=x['t'],index=x['index'],phases=sorted(item['phases']),records=item['count'],sampled_original_points=len(world),inside_count=inside_count,method=method,
            inherited_home_pair=baseline_pair,baseline_home_gap_m=home_pairs.get(pair_key) if baseline_pair else None,deeper_than_home_baseline_m=max(0,home_pairs[pair_key]-x['gap_m']) if baseline_pair else None,examples_world=world[np.flatnonzero(mask)[:12]].tolist(),
            conclusion='source-geometry intersection supported at this synthetic pose' if inside_count else 'source envelope overlap remains unresolved; absence of contained samples is not separation proof'))
    write(out/'overlap-localization.json',localized)
    anchor=read(out/'measured-anchor.json');summary=read(out/'summary.json');geo=read(out/'geometry-summary.json');cfg=read(out/'config.json');parking=read(out/'parking.json')
    suffix=[r for r in rows if r['phase']!='retimed_traverse'];phase={}
    for name in dict.fromkeys(r['phase'] for r in rows):
        rr=[r for r in rows if r['phase']==name]
        phase[name]=dict(first=rr[0]['t'],last=rr[-1]['t'],max_tracking_m=max([r['tracking_m'] for r in rr if r['tracking_m'] is not None]or[0]),max_axis_rad=max([r['tracking_axis_rad'] for r in rr if r['tracking_axis_rad'] is not None]or[0]),min_joint_slack=min(r['joint_slack'] for r in rr),max_joint_rate=max(float(max(abs(np.array(r['commanded_joint_rate'])[:6]))) for r in rr))
    n=len(parking['samples']);v=np.r_[anchor['row']['action'][0],[r['action'][0] for r in rows[:n+1]]]
    parking_checks=dict(start_velocity_sample_difference=float(v[1]-v[0]),last_moving_interval_velocity=float(v[-2]),first_stopped_velocity=float(v[-1]),max_speed_increment=float(abs(np.diff(v)).max()),
        no_position_overshoot=all(r['base'][0]>=parking['endpoint'][0]-1e-12 for r in rows[:n]),endpoint_position_error_m=float(np.linalg.norm(np.array(rows[n-1]['base'][:2])-parking['endpoint'][:2])),
        prior_observed_command_joint_rate=((np.array(anchor['row']['action'][2:])-anchor['previous_row']['action'][2:])/.02).tolist(),first_new_command_joint_rate=rows[0]['commanded_joint_rate'],
        final_parking_joint_rate=rows[n-1]['commanded_joint_rate'],first_hold_joint_rate=rows[n]['commanded_joint_rate'])
    coef=parking['polynomial'];T=parking['duration'];parking_checks['analytic_start_acceleration']=2*coef[2]/T**2;parking_checks['analytic_end_acceleration']=(2*coef[2]+6*coef[3]+12*coef[4]+20*coef[5])/T**2
    command=np.array([r['action'][2:8] for r in suffix]);joint_acc=float(abs(np.diff(command,n=2,axis=0)/.02**2).max())
    confirmed_external=[x for x in localized if x['pair']['category']=='external' and x['inside_count']>0]
    confirmed_self=[x for x in localized if x['pair']['category']=='self' and x['inside_count']>0]
    premature=[x for x in localized if x['pair']['category']=='intended' and any(p in ['deploy','turn','fold','withdraw'] for p in x['phases'])]
    post_stop_confirmed=[x for x in confirmed_external if any(p not in ['measured_anchor','retimed_traverse'] for p in x['phases'])]
    decision=dict(geometry='not certified globally; inspect localized inherited/new source intersections and unresolved intervals',continuity='same-endpoint base stop analytically continuous; sampled joint command rates bounded; joint-rate transitions reported, not claimed globally C1',
        synthetic_tracking='approach gate passed' if summary['approach_gate_passed'] else 'approach gate blocked close',physical_admission=False,
        full_candidates=1,trajectory_sha256=sha(out/'trace.jsonl'),phase_metrics=phase,parking=parking_checks,max_suffix_joint_acceleration_rad_s2=joint_acc,
        confirmed_external_phase_pair_groups=len(confirmed_external),confirmed_self_phase_pair_groups=len(confirmed_self),distinct_overlap_phase_pair_groups=len(localized),
        confirmed_post_stop_external_groups=post_stop_confirmed,
        premature_selected_piece_confirmed_groups=[x for x in premature if x['inside_count']>0],
        overall='NEEDS_CORRECTION_NO_PHYSICAL_ADMISSION',
        selected_piece_premature_overlap_groups=premature,
        total_reference_time_s=summary['total_reference_time_s'],reference_c2_time_s=180.39999960618107,
        source_reference='measured step880 anchor; synthetic frozen leg pose and retimed virtual DLS handoff; prescribed door angle; no real push, latch or data quality conclusion')
    write(out/'decision.json',decision)
    from scipy.spatial.transform import Rotation
    local=next(x for x in read(run/'local-01/results.json')['cases'] if x['snapshot']==1 and x['radius']==.905 and x['roll_deg']==0)
    nearby=min([r for r in rows if r['phase']=='close'],key=lambda r:abs(r['theta']-local['theta'])) if summary['close_evaluated'] else None
    if nearby:
        solver=Solver(G.robot,'corner');L=baseT(local['base'])@solver.fk(local['q']);A=baseT(nearby['base'])@solver.fk(nearby['q']);rv=Rotation.from_matrix(A[:3,:3]@L[:3,:3].T).as_rotvec();normal=np.array([math.cos(local['theta']),math.sin(local['theta']),0])
        compare=dict(local_source_time=local['source_time'],candidate_time=nearby['t'],local_theta=local['theta'],candidate_theta=nearby['theta'],door_angle_difference_rad=nearby['theta']-local['theta'],
            local_geometry=local['geometry']['minimums'],candidate_geometry=G.check(nearby['base'],nearby['q'],nearby['theta'])['minimums'],
            orientation_difference_rad=float(np.linalg.norm(rv)),rotation_component_about_contact_normal_rad=float(rv@normal),local_q=local['q'],candidate_q=nearby['q'],
            meaning='Read-only comparison at nearly matched door angle; local study used a full-orientation reference, candidate IK constrains point plus normal and leaves rotation about that normal free. No extra IK or candidate.')
        write(out/'local-vs-continuous.json',compare)
    plot(run,rows,checks,G,localized)
    make_report(run,decision,summary,geo,localized)
    print(json.dumps(decision,indent=2),flush=True)

def plot(run,rows,checks,G,localized):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from geometry import transform
    out=run/'single';p=read(out/'parking.json');old=[json.loads(l) for l in (ROOT/'.repro/reviews/wc4-close-inplace-stage2-20261026/c2/trace.jsonl').open()];n=len(p['samples']);rr=rows[:n+25]
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    old=[r for r in old if r['phase'] in ['traverse_continuation','endpoint_hold']]
    for seq,label,color in [(old,'c2 reference','#888888'),(rr,'single corrected','#0078ae')]:
        tt=[r['t'] for r in seq];axes[0].plot(tt,[r['base'][0] for r in seq],label=label,color=color);axes[1].plot(tt,[r['action'][0] for r in seq],label=label,color=color)
    for j in range(6):axes[2].plot([r['t'] for r in rr],[r['commanded_joint_rate'][j] for r in rr],label='joint%d'%(j+1))
    axes[0].axhline(p['endpoint'][0],color='k',ls='--',label='same endpoint');axes[0].set_ylabel('Base x (m)');axes[1].set_ylabel('Joint9 vx (m/s)');axes[2].set_ylabel('Joint command rate (rad/s)');axes[2].set_xlabel('Source-aligned reference time (s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
    fig.suptitle('Parking handoff | measured anchor, retimed synthetic continuation | no physics');fig.tight_layout();fig.savefig(out/'parking-curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(12,12),sharex=True);t=[r['t'] for r in rows]
    axes[0].plot(t,[r['tracking_m'] if r['tracking_m'] is not None else np.nan for r in rows]);axes[0].axhline(.008,color='r',ls='--');axes[0].set_ylabel('Tracking error m')
    axes[1].plot(t,[r['joint_slack'] for r in rows],label='joint slack rad');axes[1].plot(t,[max(abs(np.array(r['commanded_joint_rate'])[:6])) for r in rows],label='joint rate rad/s');axes[1].legend()
    ordered=sorted(checks,key=lambda r:r['t'])
    for c in ['external','self','intended']:axes[2].plot([r['t'] for r in ordered],[r['minimums'][c]['gap_m'] for r in ordered],label=c)
    axes[2].axhline(0,color='k');axes[2].set_ylim(-.17,.15);axes[2].set_ylabel('Source gap m');axes[2].legend()
    axes[3].plot(t,[r['action'][0] for r in rows],label='v');axes[3].plot(t,[r['action'][1] for r in rows],label='yaw rate');axes[3].plot(t,[2.5-r['whole_max_x'] for r in rows],label='full-body far margin');axes[3].legend();axes[3].set_xlabel('Reference time (s)')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('ONE corrected CPU candidate | source geometry and synthetic tracking, not execution');fig.tight_layout();fig.savefig(out/'error-clearance-curves.png',dpi=150);plt.close(fig)
    selected=[]
    for phase in ['retimed_traverse','withdraw','fold','turn','deploy','seek','close','release','retract','restore']:
        ss=[r for r in rows if r['phase']==phase]
        if ss:selected.append(ss[-1])
        if phase=='close' and ss:selected.extend([ss[len(ss)//4],ss[len(ss)//2]])
    selected.sort(key=lambda r:r['t']);fig,axes=plt.subplots(3,4,figsize=(18,13))
    def hull(ax,p,color,alpha):
        p=np.unique(np.round(p[:,:2],7),axis=0)
        try:h=p[ConvexHull(p).vertices];ax.fill(h[:,0],h[:,1],color=color,alpha=alpha)
        except Exception:pass
    for ax,r in zip(axes.flat,selected):
        rp=G.robot.fk(dict(G.legs,**dict(zip(NAMES,r['q']))),baseT(r['base']));dp=G.door.fk({'door_hinge_joint':r['theta'],'handle_spindle_joint':0},transform([2.61,0,.01],[0,0,math.pi/2]))
        for piece in G.R:hull(ax,piece.world(rp[piece.link])[0],'#2178aa',.35)
        for piece in G.D:hull(ax,piece.world(dp[piece.link])[0],'#ac753a',.25)
        for _,w in G.W:hull(ax,w[0],'#777',.2)
        ax.scatter(*r['pad'][:2],color='red',s=12);ax.set(title=f"{r['phase']}  {r['t']:.2f}s",xlim=(.35,2.8),ylim=(-1.05,1.2));ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle('Key synthetic poses | source collision pieces projected from 3D; projection overlap is not penetration');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'key-poses.png',dpi=150);plt.close(fig)
    write(out/'key-poses.json',selected)
    fig,ax=plt.subplots(figsize=(8,6))
    # Actual chosen box face/edge in stator local coordinates, distinct from door reference.
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    plt.close(fig);fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d')
    verts=np.array(list(itertools.product([.12,.15],[-.03,.03],[-.03,-.01])))
    for axis in range(3):
        for value in np.unique(verts[:,axis]):
            ids=np.flatnonzero(abs(verts[:,axis]-value)<1e-9);face=verts[ids];center=face.mean(0);others=[j for j in range(3) if j!=axis];order=np.argsort(np.arctan2(face[:,others[1]]-center[others[1]],face[:,others[0]]-center[others[0]]));ax.add_collection3d(Poly3DCollection([face[order]],facecolor='#62a9cf',edgecolor='#266080',alpha=.2))
    ax.plot([.15,.15],[.03,.03],[-.03,-.01],color='red',lw=4,label='selected +x/+y edge');ax.scatter(.15,.03,-.02,color='red',s=45)
    for a in [0,math.pi/4,math.pi/2]:ax.quiver(.15,.03,-.02,.025*math.cos(a),.025*math.sin(a),0,color='#a53728')
    ax.set(xlim=(.11,.185),ylim=(-.04,.07),zlim=(-.04,.005),xlabel='Stator x m',ylabel='Stator y m',zlabel='Stator z m',title='Actual collision box 2 edge and outward normal cone\nDoor contact r=0.905 m on source slab triangle 40');ax.legend();fig.tight_layout();fig.savefig(out/'contact-part.png',dpi=160);plt.close(fig)
    by_index={r['index']:r for r in checks}
    def priority(x):
        return (0 if x['pair']['category']=='external' and 'close' in x['phases'] else 1 if 'deploy' in x['phases'] else 2 if x['inherited_home_pair'] else 3,x['worst_gap_m'])
    chosen=sorted(localized,key=priority)[:8]
    if chosen:
        fig,axes=plt.subplots(math.ceil(len(chosen)/2),2,figsize=(14,5*math.ceil(len(chosen)/2)),squeeze=False)
        for ax,x in zip(axes.flat,chosen):
            r=by_index[x['index']];rp=G.robot.fk(dict(G.legs,**dict(zip(NAMES,r['q']))),baseT(r['base']));dp=G.door.fk({'door_hinge_joint':r['theta'],'handle_spindle_joint':0},transform([2.61,0,.01],[0,0,math.pi/2]))
            a=x['pair']['a'];b=x['pair']['b'];pa=next(p for p in G.R if p.link==a[0] and p.name==a[1]);A=pa.world(rp[pa.link])[0]
            if x['pair']['category']=='self':pb=next(p for p in G.R if p.link==b[0] and p.name==b[1]);B=pb.world(rp[pb.link])[0]
            else:
                pb=next((p for p in G.D if p.link==b[0] and p.name==b[1]),None);B=pb.world(dp[pb.link])[0] if pb else next(w[0] for p,w in G.W if p.link==b[0])
            # x/z projection helps distinguish the inherited folded-arm overlap.
            for pts,col,label in [(A,'#246f9b','robot source hull'),(B,'#ba7e38','other source hull')]:
                p=pts[:,[0,2]]
                try:h=p[ConvexHull(p).vertices];ax.fill(h[:,0],h[:,1],color=col,alpha=.25,label=label)
                except Exception:ax.scatter(p[:,0],p[:,1],s=4,color=col,label=label)
            e=np.array(x['examples_world'])
            if len(e):ax.scatter(e[:,0],e[:,2],color='red',s=18,label='contained original samples')
            ax.set(title=f"{x['pair']['category']} | {a[0]}:{a[1]} vs {b[0]}:{b[1]}\nt={x['t']:.2f}s, contained={x['inside_count']}",xlabel='World x m',ylabel='World z m');ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(chosen):]:ax.axis('off')
        fig.suptitle('Overlap localization | projected source envelopes and original contained samples; not physical penetration depth');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/'overlap-localization.png',dpi=150);plt.close(fig)

def make_report(run,d,s,g,loc):
    p=d['parking'];lines=['# 第三轮：单条候选的停车、部署与干涉修正','',
        '执行侧完成唯一一条完整候选及只读检查，冻结待独立审查。停车和接近衔接得到改善，但全程几何仍需修正／存在未证部分，不进入物理试运行。没有第二条完整路线。几何、连续性与合成跟踪分别报告，不把参考轨迹完成当作完整任务成功。','',
        '## 相对 c2 的修正','',
        '- 全身通过前保留旧实测前缀；之后沿原直线到同一配置终点(1.259999752,0.180000007)m，仅重排时间。用端点速度/加速度约束的五次位置函数，在7.16s内从原0.2000033855m/s减至0，无越点。',
        '- 既有c2虚拟DLS关节序列按空间进度作单调PCHIP时间重参数化，再保留原3rad/s指令界；实际起点、前一真实指令和合成状态分开记录。该模型没有重新运行真实开门或物理跟踪。',
        '- 接触半径0.88→0.905m，高度仍0.85m，初始转向仍270°；接触为原固定指box 2的+x/+y棱，法向在其实体外法向锥内变化。原门板三角形40包含该接触参考点。',
        '- 4cm预接触退距局部求解失败后，保持原8mm/0.03rad误差门槛，选用1cm退距；以一条平滑关节弧线从home部署到已求解近接触姿态，随后短程法向接近，取消旧部署过程中同时大幅扭转的方式。',
        '- 在推关前显式检查位置/法向误差、关节/底盘速度、远侧以及非接触部位/自碰撞源几何间隔。失败则跳过close/settle/release并标记，不按时间强行推关。本次该门通过。',
        '- 底盘在停车终点保持至推关；仍在预设门角35°以下、朝向已回正后平移0.45m。本轮不减量。回合底盘z始终0.5m；接触高度不等于底盘高度。','',
        '## 分项结果','',
        f"- **停车连续性**：终点位置误差 {p['endpoint_position_error_m']:.3g}m；无越点={p['no_position_overshoot']}；首个离散速度与原速度之差 {p['start_velocity_sample_difference']:.6g}m/s，最后移动区间速度 {p['last_moving_interval_velocity']:.6g}m/s，随后0；解析起/止加速度分别 {p['analytic_start_acceleration']:.3g}/{p['analytic_end_acceleration']:.3g}m/s²。离散区间平均速度不等于瞬时端点速度。",
        f"- **关节与底盘**：新后段指令最大关节速度 {s['max_command_joint_rate_suffix']:.6g}rad/s，继承段 {s['max_command_joint_rate_inherited']:.6g}rad/s；侧向速度残差 {s['maximum_lateral_mps']:.3g}m/s。新后段最大指令二阶差分 {d['max_suffix_joint_acceleration_rad_s2']:.6g}rad/s²，未声称全程关节速度C1或真实驱动平滑性。",
        f"- **合成跟踪**：接近门通过={s['approach_gate_passed']}；实际计算到推关={s['close_evaluated']}。参考总时长 {s['total_reference_time_s']:.3f}s，相比c2的180.400s缩短；90s推关段没有增加。累计推关底盘平移 {s['after_endpoint_translation_m']:.6g}m，最大位移 {s['maximum_displacement_after_endpoint_m']:.6g}m。",
        f"- **几何**：检查 {g['source_and_interpolated_samples']} 个原始/插值姿态；外部/自碰撞/远侧未证区间分别 {g['unresolved_external_intervals']}/{g['unresolved_self_intervals']}/{g['unresolved_far_intervals']}。采样远侧返回数 {g['sampled_far_return_count']}；包含实测锚点的最小远侧边界余量 {g['minimum_far_margin_m']:.6g}m。证明仅限所声明的源包络与分段线性运动，不代表物理轨迹。",'',
        '|阶段|最大位置跟踪误差 m|最大法向误差 rad|最小关节命令范围余量 rad|最大指令关节速度 rad/s|','|---|---|---|---|---|']
    for name,x in d['phase_metrics'].items():lines.append(f"|{name}|{x['max_tracking_m']:.6g}|{x['max_axis_rad']:.6g}|{x['min_joint_slack']:.6g}|{x['max_joint_rate']:.6g}|")
    lines+=['','关节弧线阶段没有笛卡尔跟踪目标，表中0表示不适用，不代表端点残差自动为零；撤离端点解见withdraw-endpoint.json。原始测量锚点中跟踪误差与指令范围差异保留，不裁剪旧实测。','',
        '## 口径和几何疑点','',
        '远侧检查与运行时同样纳入全部collision及visual包围几何。初始预检查发现名义腿目标会使边界相差约2.829mm，因此在完整候选生成前，从旧实测刚体父子旋转恢复腿/轮关节姿态；起点复算与运行时max_x相差约7.7e-8m。此后腿姿态冻结仍属合成假设，未来需用实际腿关节状态检查。',
        '自碰撞只排除同一固定组件及直接关节相邻组件；其余相关运动部件全部纳入，link02/link06的既有home重叠没有排除。外部源凸包、球外包盒和圆柱外包多边形重叠按实际部件逐项定位；封闭原网格用两种射线，源box/cylinder/sphere用原始实体体积核对。无内部采样点不证明无交叠。','',
        '`intended`仅标识预选接触基元，并不自动表示该时刻接触合法。部署/转向/收臂/撤离中的预选基元重叠单独保留；按阶段分别定位，不能让更早的继承接触掩盖新后段问题。','',
        '|类别|部件对|最坏源包络间隔 m|时间 s|原几何内部采样点|说明|','|---|---|---|---|---|---|']
    for x in loc:
        pair=x['pair'];lines.append(f"|{pair['category']}|{pair['a'][0]}:{pair['a'][1]} / {pair['b'][0]}:{pair['b'][1]}|{x['worst_gap_m']:.6g}|{x['t']:.3f}|{x['inside_count']}/{x['sampled_original_points']}|{'既有home部件对；未豁免' if x['inherited_home_pair'] else x['conclusion']}|")
    lines+=['','## 边界与停止','',
        '这轮没有物理关门、接触力、真实锁存或最终共同保持证据。门角在停车/部署时固定于旧首过点实测角；接近门通过后按预设曲线关闭，不能当作推关动力学反馈。实际q/qd、门角、接触、制动和释放仍待未来实测；本轮不进入物理试运行。',
        '1条完整候选额度已使用，不换参数重跑。生成前仅在两个既有失败姿态及其接近邻域作43个局部端点求解，并检查1条局部部署弧线；没有用局部名义筛选多条完整路线。全部局部失败与最初导入错误保留。',
        '物理、资产、R11、ACT与观测及运行专家源码保持不变；只新增独立CPU工具。未增加推关前底盘移位、未改穿越空间路径/终点、未恢复绕行。无仿真、录像、采集、转换、训练或策略推理，无GPU锁或后台监控。',
        '正式结论以decision.json、geometry-summary.json、overlap-localization.json及全部未证区间为准。确定的非接触部位干涉或关键未解部分不得被低末端误差掩盖；本次冻结交付仅供方案对话独立审查。']
    if (run/'single/local-vs-continuous.json').exists():
        c=read(run/'single/local-vs-continuous.json');lines+=['','## 局部解与连续解的差异','',
            f"在门角相差{c['door_angle_difference_rad']:.6g}rad的两姿态中，完整候选与局部全姿态解的工具旋转差为{c['orientation_difference_rad']:.6g}rad，其中沿接触法向分量{c['rotation_component_about_contact_normal_rad']:.6g}rad。局部测试约束全姿态；连续求解只约束接触点和法向，剩余旋转自由度没有被净空约束控制。因此不能把局部静态净空直接延伸为整段安全证明。精确位姿、关节及净空见local-vs-continuous.json。"]
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden')
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();review(a.run)
