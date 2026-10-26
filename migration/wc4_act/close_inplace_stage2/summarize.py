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
        close=[r for r in suffix if r['phase']=='close'];seek=[r for r in suffix if r['phase']=='seek']
        rawjumps=[float(max(abs(np.array(b['q'])-a['q']))) for a,b in zip(solves[:-1],solves[1:]) if a['phase']==b['phase']]
        m=g['minimums'];external={k:v for k,v in m.items() if not k.endswith('/pad_panel')}
        worst=min(external,key=lambda k:external[k]['gap_m']);substantial=[x for x in overlap if x['category']!='pad_panel' and x['gap_m']<-.0001]
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
            worst_external_category=worst,worst_external_pair=external[worst],substantial_external_envelope_overlap_records=len(substantial),
            inherited_arrival_joint_state=h['synthetic_arrival']['q'],inherited_endpoint_ee_error_m=h['synthetic_arrival']['tracking_m'],
            final_home_error=s['final_home_error'],physical_rollouts=0,task_success=None,dataset_quality=None)
        write(out/'decision.json',data);result.append(data)
    write(run/'comparison.json',dict(candidates=result,recommendation='c2 is the best kinematic refinement basis only; no full candidate accepted. Do not reduce translation or start physics from this delivery.',
        scope='3 full nominal attempts completed; 12 static contact IK seeds for c2 diagnosis; no perturbations or physics. No fourth path.',
        common_issue='Exact original linear traverse has an abrupt endpoint stop and its held EE target does not track in the synthetic continuation. Arrival q/theta are not measured.'))
    table=['candidate,translation_m,max_displacement_m,close_tracking_m,seek_tracking_m,min_close_slack_rad,raw_branch_jump_rad,geometry_samples,unresolved_intervals,status']
    for x in result:table.append(','.join(str(x[k]) for k in ['id','translation_m','max_displacement_m','max_close_tracking_m','max_seek_tracking_m','min_close_joint_slack_rad','raw_solver_consecutive_max_jump_rad','geometry_samples','unresolved_geometry_intervals','status']))
    (run/'comparison.csv').write_text('\n'.join(table)+'\n')
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
