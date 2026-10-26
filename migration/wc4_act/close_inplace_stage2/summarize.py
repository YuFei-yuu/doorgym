"""Postprocess existing attempts only. Does not invoke candidate generation or IK."""
import argparse,json,math,sys,subprocess,hashlib
from pathlib import Path
import numpy as np
from candidate import ROOT,read,write,sha,NAMES,LEGS,baseT
from fullcycle_geometry import URDF,transform
from fullcycle_collision import collision_pieces

def main(run):
    result=[];robotpath=ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf';doorpath=ROOT/'high-level/data/asset/door_set/wc4/model.urdf'
    robot=URDF(robotpath);door=URDF(doorpath);R=collision_pieces(robotpath);D=collision_pieces(doorpath)
    from newexpert_stage2.finite_walls import walls
    effective=read(ROOT/'.repro/runs/wc4-newexpert-stage4-20261025/attempt_000/effective-config.json')
    W,_=walls(effective,read(doorpath.parent/'bounding_box.json'),0)
    protected=['migration/wc4_act/fullcycle_expert.py','migration/wc4_act/fullcycle_compact.py','migration/wc4_act/fullcycle_recording.py','migration/wc4_act/r11/criteria.py','migration/wc4_act/newexpert_stage2/push_geometry.py','high-level/data/asset/a2wz1/urdf/a2wz1.urdf','high-level/data/asset/door_set/wc4/model.urdf','high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py','high-level/float_ik/isaacgym_a2w_ik_push_door_parallel.py','high-level/float_ik/door_common.py']
    bindings={}
    for name in protected:
        before=subprocess.check_output(['git','show','1638f9b:'+name],cwd=ROOT)
        bindings[name]=dict(current_sha256=sha(ROOT/name),baseline_sha256=hashlib.sha256(before).hexdigest(),unchanged=before==(ROOT/name).read_bytes())
    write(run/'protected-source-check.json',bindings)
    # Hash all referenced collision/visual asset bytes, not only the two URDFs.
    import xml.etree.ElementTree as ET
    files={robotpath,doorpath}
    for u in [robotpath,doorpath]:
        for m in ET.parse(u).findall('.//mesh'):files.add((u.parent/m.get('filename')).resolve())
    write(run/'asset-SHA256.json',{str(p):sha(p) for p in sorted(files)})
    for cid in ['c0','c1','c2']:
        out=run/cid;s=read(out/'summary.json');g=read(out/'clearance-summary.json');h=read(out/'handoff.json')
        rows=[json.loads(l) for l in (out/'trace.jsonl').open()];solves=[json.loads(l) for l in (out/'ik-solves.jsonl').open()]
        overlap=[json.loads(l) for l in (out/'overlaps.jsonl').open()]
        bypair={}
        for x in overlap:
            if x['category']=='pad_panel':continue
            key=(x['phase'],x['robot_link'],x['robot_piece'],x['obstacle_link'],x['obstacle_piece'])
            if key not in bypair or x['gap_m']<bypair[key]['gap_m']:bypair[key]=x
        localization=[]
        for x in sorted(bypair.values(),key=lambda x:x['gap_m'])[:30]:
            r=rows[x['index']];poses=robot.fk(dict(LEGS,**dict(zip(NAMES,r['q']))),baseT(r['base']));doors=door.fk({'door_hinge_joint':r['theta'],'handle_spindle_joint':0},transform([2.61,0,.01],[0,0,math.pi/2]))
            p=next(p for p in R if p.link==x['robot_link'] and p.name==x['robot_piece'])
            if x['category']=='wall':d,B=next((p,B) for p,B in W if p.link==x['obstacle_link'])
            else:d=next(d for d in D if d.link==x['obstacle_link'] and d.name==x['obstacle_piece']);B=d.world(doors[d.link])
            A=p.world(poses[p.link]);localization.append(dict(**x,robot_world_vertices=A[0].tolist(),obstacle_world_vertices=B[0].tolist(),
                interpretation='Declared source box intersection (not a cooked-physics contact proof)' if p.kind==d.kind=='box' else 'Conservative source envelope overlap; actual mesh/sphere detail unresolved',
                row=r))
        write(out/'overlap-localization.json',localization)
        suffix=[r for r in rows if r['phase']!='traverse_continuation']
        bases=np.array([r['base'] for r in rows]);qs=np.array([r['q'] for r in rows]);dt=.02
        sweep=np.linalg.norm(np.diff(bases[:,:2],axis=0),axis=1)+2*abs(np.diff(bases[:,2]))+2*np.sum(abs(np.diff(qs,axis=0)),axis=1)
        far_unresolved=int(sum(2.5-rows[i]['whole_max_x']<=sweep[i] for i in range(len(sweep))))
        close=[r for r in suffix if r['phase']=='close'];seek=[r for r in suffix if r['phase']=='seek']
        rawjumps=[float(max(abs(np.array(b['q'])-a['q']))) for a,b in zip(solves[:-1],solves[1:]) if a['phase']==b['phase']]
        m=g['minimums'];external={k:v for k,v in m.items() if not k.endswith('/pad_panel')}
        worst=min(external,key=lambda k:external[k]['gap_m']);substantial=[x for x in overlap if x['category']!='pad_panel' and x['gap_m']<-.0001]
        suffix_gaps={k:min(x['gap_m'] for name,x in m.items() if name.endswith('/'+k) and not name.startswith('traverse_continuation/')) for k in ['arm_panel','base_panel','handle','frame','wall']}
        data=dict(id=cid,status='NEEDS_CORRECTION_NO_PHYSICAL_ADMISSION',execution_commit=read(out/'receipt.json')['head'],
            translation_m=s['suffix_translation_m'],max_displacement_m=s['suffix_max_displacement_m'],
            preceding_continuation_translation_m=s['continuation_translation_m'],follow_condition='prescribed synthetic door angle below 35 deg, yaw already 360 deg; no contact gate in this offline attempt',
            hover_ik=read(out/'deployment-endpoint-solve.json')['diagnostics'],
            max_raw_ik_position_m=max(x['position_error_m'] for x in solves),
            max_close_tracking_m=max(r['tracking_m'] for r in close),max_seek_tracking_m=max(r['tracking_m'] for r in seek),
            min_close_joint_slack_rad=min(r['joint_slack'] for r in close),min_close_slack_per_joint=(np.minimum(np.array([r['q'][:6] for r in close])-np.array([float(robot.root.find("joint[@name='%s']/limit"%n).get('lower'))*.75 for n in NAMES[:6]]),np.array([float(robot.root.find("joint[@name='%s']/limit"%n).get('upper'))*.75 for n in NAMES[:6]])-np.array([r['q'][:6] for r in close])).min(0)).tolist(),
            raw_solver_consecutive_max_jump_rad=max(rawjumps),actual_suffix_max_joint_rate=s['max_joint_rate_suffix'],
            lateral_velocity_residual=s['maximum_lateral_mps'],endpoint_velocity_jump=s['max_v_jump'],
            maximum_suffix_whole_x=max(r['whole_max_x'] for r in suffix),minimum_far_side_margin=2.5-max(r['whole_max_x'] for r in rows),
            z=.5,total_trace_end_s=rows[-1]['t'],geometry_samples=g['samples'],unresolved_geometry_intervals=g['unresolved_intervals'],
            unresolved_far_side_interpolation_intervals=far_unresolved,
            max_suffix_joint_acceleration=float(abs(np.diff(np.array([r['q'][:6] for r in suffix]),n=2,axis=0)/dt**2).max()),
            worst_external_category=worst,worst_external_pair=external[worst],substantial_external_envelope_overlap_records=len(substantial),
            suffix_minimum_gaps_by_category_m=suffix_gaps,
            inherited_arrival_joint_state=h['synthetic_arrival']['q'],inherited_endpoint_ee_error_m=h['synthetic_arrival']['tracking_m'],
            final_home_error=s['final_home_error'],physical_rollouts=0,task_success=None,dataset_quality=None)
        write(out/'decision.json',data);result.append(data)
    write(run/'comparison.json',dict(candidates=result,recommendation='c2 is the best kinematic refinement basis only; no full candidate accepted. Do not reduce translation or start physics from this delivery.',
        scope='3 full nominal attempts completed; 12 static contact IK seeds for c2 diagnosis; no perturbations or physics. No fourth path.',
        common_issue='Exact original linear traverse has an abrupt endpoint stop and its held EE target does not track in the synthetic continuation. Arrival q/theta are not measured.'))
    table=['candidate,translation_m,max_displacement_m,close_tracking_m,seek_tracking_m,min_close_slack_rad,raw_branch_jump_rad,geometry_samples,unresolved_intervals,status']
    for x in result:table.append(','.join(str(x[k]) for k in ['id','translation_m','max_displacement_m','max_close_tracking_m','max_seek_tracking_m','min_close_joint_slack_rad','raw_solver_consecutive_max_jump_rad','geometry_samples','unresolved_geometry_intervals','status']))
    (run/'comparison.csv').write_text('\n'.join(table)+'\n')
    lines=['# 第二轮：名义路线连续可达性与几何验证','',
        '冻结结论：三条完整候选均需修正，未授予物理试运行准入。c2仅推荐作为下一轮连续性修正的基础；本轮不继续求解或缩小平移。','',
        '|候选|后段累计平移/最大位移 m|部署端点IK误差 m|seek最大跟踪误差 m|close最大跟踪误差 m|close最小关节余量 rad|后段外部源包络最小间隔 m|未证净空间隔数|',
        '|---|---|---|---|---|---|---|---|']
    for x in result:lines.append(f"|{x['id']}|{x['translation_m']:.3f}/{x['max_displacement_m']:.3f}|{x['hover_ik']['position_error_m']:.6g}|{x['max_seek_tracking_m']:.6g}|{x['max_close_tracking_m']:.6g}|{x['min_close_joint_slack_rad']:.6g}|{min(x['suffix_minimum_gaps_by_category_m'].values()):.6g}|{x['unresolved_geometry_intervals']}|")
    lines+=['','所有方案从同一配置终点(1.259999752,0.180000007)m、yaw=π起始后段；底盘z=0.5m恒定。原穿越继续段平移0.715999961m另列，不计入推关后段0.45m。后段平移仅在合成门角<35°、底盘已转到360°时开始；不把预设门角当成接触反馈。',
        '','实测参考为stage4/seed62000至step880的前缀。原路径继续到step1059终点的关节状态与门角均为合成假设：门角固定为首个全身通过时的实测值，使用原目标与DLS参数、源解析Jacobian和理想跟踪。并非真实仿真或实际到达记录。',
        '','共同交接问题：原线性穿越参考到终点后由0.2m/s直接变0，没有平滑减速段。没有通过更换终点、提前停车或私自改写原逻辑消除它。合成末端到原冻结世界目标的误差在终点为0.582231m；原实测首过点已经有约0.289738m目标误差，不能把这项合成增长解释成新的物理任务失败。终点合成关节并非home，详见handoff.json。',
        '','c0：固定指+x面、300°、r=0.84m、z_contact=0.94m；部署和推关连续IK失败。c1：固定指+y侧面、300°、r=0.82m、z_contact=0.90m，推关放慢至75s；仍有贴限/分支切换。c2：固定指原box的+x/+y棱边，270°、r=0.88m、z_contact=0.85m；部署不强制法向，接近时逐渐对齐，推关接触局部法向从+y转到+x，推关90s。接触高度只是各候选的设计值，不是新的任务约束；没有底盘升降。',
        '','c2的close段位置/法向跟踪很好，但不能据此跳过进入接触阶段的约17mm短时残差、近零关节余量、交接停车跳变及几何未证区段。末端棱边接触的实际稳定性、摩擦与非末端接触都未验证。三条未通过可行性检查，因此没有进行低于0.45m的路线扩展。',
        '','几何：使用完整机器人源URDF collision pieces与真实默认腿姿态，门板、把手、门框及作者构造的两块有限墙；不是规划矩形。每0.1s及阶段边界检查；可能不安全的区间加密至50Hz，并给出线性插值运动包络证明/未证标记。源凸包、球外包盒、圆柱外包32边形有保守性；负间隔不直接认定真实穿透。overlap-localization.json定位至连杆、子碰撞基元、障碍物、时刻和世界顶点。无cooked PhysX形状或完整自碰撞认证。',
        '','Joint9动作顺序与vx/yaw-rate语义不变，平面侧向速度数值残差接近零。新后段关节命令限速0.7rad/s；原继续段保留原单步0.06rad上限（50Hz对应3rad/s），分别报告。限速不等于速度/加速度全程光滑，原始曲线与二阶差分诊断保留。所有50Hz源包络样本保持远侧；采样间远侧不能证明的区间另列。',
        '','每条trace.jsonl是完整尝试，不在求解失败处伪造通过或删段；失败之后的后缀只是定位诊断，不证明按序完成。measured-prefix.jsonl与trace.jsonl分开，严禁把二者拼接后称为完整真实执行或训练数据。自动任务结果、完整任务审核、数据质量均不适用；仿真、采集、录像、转换、训练、策略推理均为0。',
        '','来源、执行提交、工具异常与命令详见COMMANDS.txt、ledger.json、receipt.json、sources/asset SHA、执行源码快照及Git bundle。本轮只新增独立CPU工具，受保护源码和资产一致性见protected-source-check.json。',
        '','未来待实测：继续穿越的门角/角速度、终点q/qd与末端实际位置、接触继承/脱离、终点停车行为；后段接触作用、锁前真实保持、合法锁存、释放/收臂和最终共同保持。需要更改原穿越时间参数等边界事项应由方案侧在下一轮明确，不在本轮扩大范围。']
    if (run/'source-mesh-localization.json').exists():
        mesh=read(run/'source-mesh-localization.json');lines+=['','## 原始门板网格定位','',
            '门板原OBJ焊接后168条边均有两个相邻三角形（112个三角形）；用两种非轴向射线核对源几何采样点是否在其内部。以下是合成姿态的源几何交叠证据，不是实际物理接触、力值或cooked网格穿透深度。','',
            '|候选/阶段|时间 s|具体源几何|落入门板内部的原始顶点/源box采样点|','|---|---|---|---|']
        for x in mesh['results']:
            pair=x['source_pair'];lines.append(f"|{x['candidate']}/{x['phase']}|{x['time_s']:.2f}|{pair['robot_link']} / {pair['robot_piece']}|{x['both_ray_inside_count']}/{x['sampled_points']}|")
        lines+=['','尤其c2在约76.70s有441个腕部安装件原始顶点落入门板内部；seek结束也有9个固定指源box采样点落入门板。不能以低IK残差接受该路线。共同穿越继续段的最坏凸包重叠未由本次顶点采样确认，仍为未解风险；零内部采样点不证明分离。详见source-mesh-localization.json和source-overlap-detail.png。',
            '','本轮没有可执行的整条推荐方案。c2只保留为后续IK设计参考；应先解决交接停车与非末端几何干涉，再讨论平移缩短。不得据本轮结果启动物理试验或增加第四条候选。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    axes[0].bar([x['id'] for x in result],[x['max_close_tracking_m']*1000 for x in result],color=['#9a5c5c','#c18e49','#3b8f9b']);axes[0].set(ylabel='Max closing tracking error (mm)',title='Closing segment only')
    axes[1].bar([x['id'] for x in result],[x['max_seek_tracking_m']*1000 for x in result],color=['#9a5c5c','#c18e49','#3b8f9b']);axes[1].axhline(8,color='r',ls='--');axes[1].set(ylabel='Max seek tracking error (mm)',title='Entry remains unresolved')
    axes[2].bar([x['id'] for x in result],[x['translation_m'] for x in result],color='#3b8f9b');axes[2].set(ylabel='Post-endpoint base translation (m)',title='No reduction before feasibility')
    fig.suptitle('Three nominal CPU attempts | no complete candidate accepted | no physical execution');fig.tight_layout();fig.savefig(run/'comparison.png',dpi=160)
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
