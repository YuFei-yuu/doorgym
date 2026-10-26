"""Read-only report and source-mesh localization of the single recorded local trace."""
import argparse,os,json,math,itertools,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from common import *
from close_inplace_stage3.review import collision,samples,contained
from fullcycle_geometry import origin

def analyze(run):
    out=run/'trajectory';cfg=read(run/'config.json');ledger=read(run/'local-ledger.json');summary=read(out/'summary.json');G=Geometry();S=FullPose(G.robot)
    rows=[json.loads(l) for l in (out/'trace.jsonl').open()];geos=[json.loads(l) for l in (out/'geometry.jsonl').open()];intervals=[json.loads(l) for l in (out/'intervals.jsonl').open()]
    assert ledger['used']==11 and read(run/'trajectory-ledger.json')['used']==1
    assert sha(out/'trace.jsonl')==summary['trace_sha256']
    source_start=read(run/'start.json')['row'];bases=np.array([r['base'] for r in rows]);assert np.array_equal(bases,np.repeat(np.array(source_start['base'])[None],len(rows),axis=0))
    phases={}
    for phase in dict.fromkeys(r['phase'] for r in rows):
        rr=[r for r in rows if r['phase']==phase];gg=[g for g in geos if g['phase']==phase];ii=[i for i in intervals if i['phase']==phase]
        phases[phase]=dict(samples=len(rr),max_position_m=max(r['position_error_m'] for r in rr),max_orientation_rad=max(r['orientation_error_rad'] for r in rr),max_normal_rad=max(r['normal_error_rad'] for r in rr),
            max_roll_error_rad=max(abs((r['twist_rad']-r['target_twist_rad']+np.pi)%(2*np.pi)-np.pi) for r in rr),min_joint_slack_rad=min(r['joint_slack'] for r in rr),max_joint_rate_rad_s=max(float(max(abs(np.array(r['joint_rate'])[:6]))) for r in rr),
            min_external_source_gap_m=min(g['minimums']['external']['gap_m'] for g in gg),min_selected_piece_gap_m=min(g['minimums']['intended']['gap_m'] for g in gg),min_self_source_gap_m=min(g['minimums']['self']['gap_m'] for g in gg),
            unresolved_selected_intervals=sum(not i['selected_certified'] for i in ii),unresolved_external_intervals=sum(not i['external_certified'] for i in ii),unresolved_self_intervals=sum(not i['self_certified'] for i in ii),unresolved_far_intervals=sum(not i['far_certified'] for i in ii),diagnostic_samples=sum(r['status'].startswith('diagnostic') for r in rr))
    groups={}
    for r,g in zip(rows,geos):
        for x in g['overlaps']:
            key=(r['phase'],x['category'],tuple(x['a']),tuple(x['b']))
            if key not in groups or x['gap_m']<groups[key]['entry']['gap_m']:groups[key]=dict(row=r,entry=x)
    localized=[]
    for item in sorted(groups.values(),key=lambda x:x['entry']['gap_m']):
        r,x=item['row'],item['entry'];rp=G.robot.fk(dict(G.legs,**dict(zip(NAMES,r['q']))),baseT(r['base']));dp=G.door.fk({'door_hinge_joint':r['theta'],'handle_spindle_joint':0},__import__('close_inplace_stage3.geometry',fromlist=['transform']).transform([2.61,0,.01],[0,0,np.pi/2]))
        a,ag,ar=collision(G,*x['a'][:2]);p=samples(ag,ar)
        if len(p)>8000:p=p[np.linspace(0,len(p)-1,8000,dtype=int)]
        world=apply(rp[x['a'][0]]@origin(a),p)
        try:
            b,bg,br=collision(G,*x['b'][:2],obstacle=x['category']!='self');T=(rp if x['category']=='self' else dp)[x['b'][0]]@origin(b);local=apply(np.linalg.inv(T),world);mask,method=contained(local,bg,br)
        except (KeyError,StopIteration):mask=np.zeros(len(p),bool);method=dict(method='source lookup unresolved; not cleared')
        localized.append(dict(phase=r['phase'],t=r['t'],status=r['status'],category=x['category'],a=x['a'],b=x['b'],gap_m=x['gap_m'],inside_count=int(mask.sum()),sample_points=len(p),method=method,examples_world=world[np.flatnonzero(mask)[:12]].tolist(),q=r['q'],theta=r['theta'],base=r['base']))
    write(out/'localization.json',localized)
    endpoint=next(g for r,g in zip(rows[::-1],geos[::-1]) if r['phase']=='align_approach')
    decision=dict(overall='NOT_PASSED_NO_PHYSICS',local_groups_used=11,local_groups_unused=1,local_trajectories=1,complete_candidates=0,
        ordered_completion=False,gate=summary['gate'],endpoint_geometry=endpoint['minimums'],first_issue=summary['first_issue'],phase_metrics=phases,
        baseline_note='Existing HOME exit is conditional and recorded, not a blanket exemption. New pairs/deepening/recurrence fail. Remaining proofs use declared source geometry and interpolation only.',
        base_translation=0,base_yaw_change=0,stop_door_deg=70,later_045m_plan_untouched=True,
        normal_roll_control='Identical full 3x3 rotation residual and gates in local/continuous IK; explicit R(theta,roll) and no unconstrained normal-axis spin. Tracking failures are not suppressed.',
        trace_sha256=sha(out/'trace.jsonl'),confirmed_nonbaseline_groups=[x for x in localized if x['category']!='self' and x['inside_count']>0 and x['gap_m']< -cfg['geometry_epsilon_m']])
    write(out/'decision.json',decision);plots(run,rows,geos,G,localized);report(run,decision,localized,ledger,summary)
    print(json.dumps(decision,indent=2))

def plots(run,rows,geos,G,localized):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from close_inplace_stage3.geometry import transform
    out=run/'trajectory';t=np.array([r['t'] for r in rows]);R=np.array([r['actual_rotation'] for r in rows]);T=np.array([r['target_rotation'] for r in rows])
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    actual=np.unwrap(Rotation.from_matrix(R).as_euler('xyz'),axis=0)*180/np.pi;target=np.unwrap(Rotation.from_matrix(T).as_euler('xyz'),axis=0)*180/np.pi
    for j,name in enumerate(['x','y','z']):axes[0].plot(t,actual[:,j],label='FK Euler '+name);axes[0].plot(t,target[:,j],'--',alpha=.55)
    axes[0].legend(ncol=3);axes[0].set_ylabel('Euler angles deg (display only)')
    axes[1].plot(t,np.unwrap([r['twist_rad'] for r in rows])*180/np.pi,label='FK twist about door normal');axes[1].plot(t,np.unwrap([r['target_twist_rad'] for r in rows])*180/np.pi,'--',label='prescribed twist');axes[1].legend();axes[1].set_ylabel('Twist representation deg')
    axes[2].plot(t,[r['orientation_error_rad'] for r in rows],label='full SO(3) error');axes[2].plot(t,[r['normal_error_rad'] for r in rows],label='normal difference (gate at contact)');axes[2].axhline(.01,color='r',ls='--',label='full orientation gate');axes[2].legend();axes[2].set_ylabel('rad');axes[2].set_xlabel('Local reference time s')
    for ax in axes:ax.grid(alpha=.2);ax.axvline(22,color='k',alpha=.3)
    fig.suptitle('Full tool pose | synthetic; gate FAILED, push is DIAGNOSTIC\nTwist during misaligned deployment is a representation, not a valid contact-roll angle');fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'orientation-roll.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    axes[0].plot(t,[r['position_error_m']*1000 for r in rows]);axes[0].axhline(8,color='r',ls='--');axes[0].set_ylabel('Position error mm')
    for cat in ['external','intended','self']:axes[1].plot(t,[g['minimums'][cat]['gap_m'] for g in geos],label=cat)
    axes[1].axhline(0,color='k');axes[1].legend();axes[1].set_ylabel('Source gap m')
    axes[2].plot(t,[r['joint_slack'] for r in rows],label='joint slack rad');axes[2].plot(t,[max(abs(np.array(r['joint_rate'])[:6])) for r in rows],label='joint rate rad/s');axes[2].axhline(.7,color='r',ls='--');axes[2].legend();axes[2].set_xlabel('Local reference time s')
    first=read(out/'summary.json')['first_issue']
    for ax in axes:
        ax.grid(alpha=.2)
        if first:ax.axvspan(first['t'],t[-1],color='#ffb3a6',alpha=.15)
    fig.suptitle('Tracking and clearance | shaded suffix follows failed/unproved prefix\nSelected contact box is NOT exempted during deployment');fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'errors-clearance.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,8));pad=np.array([r['pad'] for r in rows]);goal=np.array([r['target'] for r in rows])
    ax.plot(goal[:,0],goal[:,1],'--',label='prescribed pad reference');ax.plot(pad[:,0],pad[:,1],label='synthetic FK pad');ax.scatter(*rows[0]['base'][:2],color='k',label='fixed base XY')
    for deg in [90,80,70]:
        a=np.deg2rad(deg);end=np.array([2.467,-.463])+np.array([-.913*np.sin(a),.913*np.cos(a)]);ax.plot([2.467,end[0]],[-.463,end[1]],color='#a97541',alpha=.6);ax.annotate(str(deg)+'°',end,xytext=(-22,-15),textcoords='offset points')
    for phase in ['deploy_a','deploy_b','align_approach','push_to_70']:
        r=[r for r in rows if r['phase']==phase][-1];ax.scatter(*r['pad'][:2],s=30);ax.annotate(phase,r['pad'][:2],xytext=(12,-18) if phase=='deploy_b' else (12,10),textcoords='offset points',fontsize=9)
    ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend();ax.set(xlabel='World x m',ylabel='World y m',title='Fixed-base local reference | gate FAILED, push segment is diagnostic');fig.tight_layout();fig.savefig(out/'deployment-path.png',dpi=160);plt.close(fig)
    selected=[rows[0]]+[next(r for r in rows if r['t']>=u) for u in [4,8,11,14,18,22,28,33]]+[rows[-1]]
    fig,axes=plt.subplots(2,5,figsize=(20,9))
    for ax,r in zip(axes.flat,selected):
        rp=G.robot.fk(dict(G.legs,**dict(zip(NAMES,r['q']))),baseT(r['base']));dp=G.door.fk({'door_hinge_joint':r['theta'],'handle_spindle_joint':0},transform([2.61,0,.01],[0,0,np.pi/2]))
        for pieces,poses,color in [(G.R,rp,'#287fa9'),(G.D,dp,'#b68247')]:
            for p in pieces:
                xy=np.unique(np.round(p.world(poses[p.link])[0][:,:2],8),axis=0)
                try:xy=xy[ConvexHull(xy).vertices];ax.fill(xy[:,0],xy[:,1],color=color,alpha=.27)
                except Exception:pass
        ax.scatter(*r['pad'][:2],color='red',s=12);ax.set(xlim=(.55,2.75),ylim=(-.95,.9),title=f"{r['phase']} {r['t']:.2f}s\n{'DIAGNOSTIC' if r['status'].startswith('diagnostic') else 'reference'}");ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle('Key synthetic poses | projection overlap is not physical penetration');fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'key-poses.png',dpi=140);plt.close(fig)
    chosen=sorted(localized,key=lambda x:(x['category']=='self',x['gap_m']))[:6]
    if chosen:
        fig,axes=plt.subplots(math.ceil(len(chosen)/2),2,figsize=(13,5*math.ceil(len(chosen)/2)),squeeze=False)
        for ax,x in zip(axes.flat,chosen):
            rp=G.robot.fk(dict(G.legs,**dict(zip(NAMES,x['q']))),baseT(x['base']));dp=G.door.fk({'door_hinge_joint':x['theta'],'handle_spindle_joint':0},transform([2.61,0,.01],[0,0,np.pi/2]))
            a=next(p for p in G.R if [p.link,p.name]==x['a'][:2]);A=a.world(rp[a.link])[0]
            parts=G.R if x['category']=='self' else G.D;b=next(p for p in parts if [p.link,p.name]==x['b'][:2]);B=b.world((rp if x['category']=='self' else dp)[b.link])[0]
            for v,c in [(A,'#287fa9'),(B,'#b68247')]:
                xy=v[:,[0,2]]
                try:xy=xy[ConvexHull(xy).vertices];ax.fill(xy[:,0],xy[:,1],color=c,alpha=.3)
                except Exception:pass
            e=np.array(x['examples_world'])
            if len(e):ax.scatter(e[:,0],e[:,2],color='red',s=20)
            ax.set_xlim(A[:,0].min()-.03,A[:,0].max()+.03);ax.set_ylim(A[:,2].min()-.03,A[:,2].max()+.03);ax.set_aspect('equal');ax.grid(alpha=.2);ax.set(title=f"{x['phase']} t={x['t']:.2f}s | inside={x['inside_count']}\n{x['a'][0]}:{x['a'][1]} vs {x['b'][0]}:{x['b'][1]}",xlabel='World x m',ylabel='World z m')
        for ax in list(axes.flat)[len(chosen):]:ax.axis('off')
        fig.suptitle('Source overlap localization | original contained samples red | no physical penetration claim');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'interference.png',dpi=150);plt.close(fig)

def report(run,d,loc,ledger,s):
    lines=['# 第四轮：部署至70°的局部几何修正','',
        '**未通过，不进入物理。** 11组局部检查、1条局部连续参考已完成；剩余1组静态额度不再使用。完整候选、物理仿真均为0。','',
        '## 起点、范围与方法','',
        '起点是第三轮44.28s的合成转向结束HOME姿态，固定底盘(1.259999752,0.180000007)m、yaw=270°、z=0.5m；未声称实际到达。第三轮停车结果和后续0.45m计划只保留原字节，没有重算、延伸或检验后段接续。',
        '用完整3×3工具姿态残差取代只约束接触点/法向的自由旋转。局部与连续IK使用同一FullPose求解器、单一已记录初值及同一数值/几何要求；全姿态误差门槛0.01rad，原位置8mm、法向0.03rad不放宽。几何1e-6m仅为数值带，不是允许毫米级穿透。',
        '局部组0/1检查两个旧c2外侧部署路标。组2—9检查1cm悬停／近90°接触／80°／70°的15.5°、16.5°绕法向角；组10检查7°悬停。悬停位姿未通过，不把失败q当成合格路标。唯一参考取消固定悬停点，以两个安全外侧路标后的一条完整姿态插值曲线接近门面。接触起始绕法向角13.5°是该唯一轨迹中未预验证的参考，不冒称额外局部检查通过；到80°平滑变为经局部检查的15.5°，保持至70°。',
        '局部预设R(theta)=Rot(n(theta),roll)。部署关节弧线指定全部六轴，接近时完整姿态Slerp，推关用同一完整R目标；每个连续样本只以实际合成前态作一次求解，不做隐藏多初值重试。','',
        '## 局部账本','',
        '|组|类型/目标|位置误差 m|全姿态误差 rad|外部间隔 m|预选基元间隔 m|判定|','|---|---|---|---|---|---|---|']
    for x in ledger['groups']:
        r=x['result'];m=r['geometry']['minimums'];lines.append(f"|{x['id']}|{x.get('tag',x['description'])} {x.get('roll_deg','')}|{r['position_error_m']}|{r['orientation_error_rad']}|{m['external']['gap_m']:.6g}|{m['intended']['gap_m']:.6g}|{r['strict_local_pass']}|")
    lines+=['','## 唯一连续参考与联合门槛','',f"局部参考38s、1900个50Hz样本，结束门角70°，底盘平移/旋转均为0。首个失败/未证点：{s['first_issue']}。随后记录的状态全部带diagnostic标签；不得称为从起点按序完成。",'',
        f"接近端点位置误差{s['gate']['endpoint_position']:.6g}m、全姿态误差{s['gate']['endpoint_orientation']:.6g}rad、法向误差{s['gate']['endpoint_normal']:.6g}rad、关节速度{s['gate']['endpoint_joint_speed']:.6g}rad/s；但此前路径未联合通过，因此门槛false，推关至70°仅为诊断后缀。",'',
        '|阶段|位置最大 m|姿态最大 rad|绕法向表示误差最大 rad|关节最小余量 rad|最大关节速度 rad/s|外部最小源间隔 m|预选基元最小间隔 m|','|---|---|---|---|---|---|---|---|']
    for name,x in d['phase_metrics'].items():lines.append(f"|{name}|{x['max_position_m']:.6g}|{x['max_orientation_rad']:.6g}|{x['max_roll_error_rad']:.6g}|{x['min_joint_slack_rad']:.6g}|{x['max_joint_rate_rad_s']:.6g}|{x['min_external_source_gap_m']:.6g}|{x['min_selected_piece_gap_m']:.6g}|")
    lines+=['','部署未对预选box 2豁免交叠。所有样本检查完整夹爪、腕部、连杆、门板/门框/把手及有限墙；远侧包含collision和visual。采样间采用同一存储q/门角的线性插值，至2.5ms最小子区间。选定实体box另用全顶点背面半空间及导数界检查，并单查其他门板装饰基元；未证区间不通过。它们不是物理跟踪或cooked形状证明。',
        '已有HOME交叠只在初始、连续且不加深的离开区间单列；新对子、加深或再出现均不豁免。该基线条件仍不构成无条件自碰撞认证。继承的固定装配/邻接方法及排除清单保留在baseline.json。','',
        '## 原始实体定位','', '|阶段|源部件对|源间隔 m|原实体内部采样点|状态|','|---|---|---|---|---|']
    for x in loc:lines.append(f"|{x['phase']}|{x['a'][0]}:{x['a'][1]} / {x['b'][0]}:{x['b'][1]}|{x['gap_m']:.6g}|{x['inside_count']}/{x['sample_points']}|{x['status']}|")
    lines+=['','源凸包负间隔不直接叫真实穿透；原网格双射线或原box/cylinder/sphere内部采样为进一步定位。没有内部采样点也不能清除疑点。所有源重叠、未知区间和失败保留。','',
        '## 相对第三轮及停止','',
        '已实现：完整姿态统一约束、显式绕法向参考、外侧部署路标、预选基元部署期严格检查，以及路径与接近端点联合门槛。没有再次用单一末端误差宣布通过。',
        '仍未证明：整条部署→接近→初段推关能按序满足全部可达、限速和连续净空要求。后缀的局部低误差或某些姿态分离，不能回填成前缀通过。第三轮停车不在本轮重验范围；70°后跟进、释放或闭门也没有生成。',
        '本轮额度按全局local-ledger.json和trajectory-ledger.json记账；没有换名或分段增加候选。唯一轨迹完成后停止，不继续使用剩余静态额度。无GPU锁、监控、仿真、录像、采集、转换、训练或策略推理。等待独立审查。']
    if (run/'trajectory/external-interval-strengthening.json').exists():
        strong=read(run/'trajectory/external-interval-strengthening.json')
        lines+=['','## 同一轨迹的后验分离证据','',
            f"原检查中非预选部件的{strong['intervals_rechecked']}个外部未证区间，仅在原有端点上重算完整SAT轴的更强投影下界；{strong['certified']}个得到证明，仍未证{strong['still_unproved']}个。没有新IK、初值、目标姿态或轨迹。原快速分离值仍是保守下界，不应把图上的很小投影值直接等同实际欧氏间距。",
            '因此，除预选接触基元外，本诊断参考中的机器人部件（含腕部、活动指）与外部障碍，已有逐区间源包络分离证据。该结论不包括预选基元的已确认交叠、自碰撞基线/未证区段，也不是物理跟踪证明。接近末端预选box 2约0.133mm源负间隔及7/125原实体内部采样点仍然存在。',
            '没有回写原门槛或轨迹状态；推关后缀仍是诊断，不得改称按序完成。']
        write(run/'trajectory/final-decision.json',dict(overall='NOT_PASSED',ordered_completion=False,original_gate_rewritten=False,local_groups=11,local_trajectories=1,complete_candidates=0,physics=0,
            nonselected_external_interval_evidence='certified in declared source-envelope interpolation' if strong['still_unproved']==0 else 'still unresolved',selected_contact_geometry='confirmed overlap at approach end and initial diagnostic push',
            self_and_baseline='original conditional baseline and unresolved intervals retained',original_decision_sha256=sha(run/'trajectory/decision.json'),strengthening_sha256=sha(run/'trajectory/external-interval-strengthening.json'),trace_sha256=sha(run/'trajectory/trace.jsonl')))
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('hide CUDA')
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();analyze(a.run)
