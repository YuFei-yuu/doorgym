"""CPU-only interpretation of the ONE saved property scene and old evidence.

Never imports Isaac Gym, reconstructs a live scene, or applies a saved state.
Keeps source geometry, unstepped SDK readback and recorded measured q separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from fullcycle_geometry import URDF, origin, apply
from fullcycle_collision import Piece, signed_gap

ROOT=Path(__file__).resolve().parents[2]


def clean(x):
    # Pybind vectors expose a recursive numpy dtype descriptor. Retain the raw
    # snapshot, and remove only this nonphysical descriptor in the derived view.
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k!='dtype'}
    if isinstance(x,list):return [clean(v) for v in x]
    return x


def vector(v):return np.array([v[k] for k in 'xyz'],float)


def pose(x):
    T=np.eye(4);T[:3,3]=vector(x['p']);T[:3,:3]=Rotation.from_quat([x['r'][k] for k in 'xyzw']).as_matrix();return T


def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')


def cylinder_pair(robot,outer=False):
    result=[]
    for name in ['link02','link06']:
        c=robot.links[name].findall('collision')[0];g=c.find('geometry/cylinder')
        r=float(g.get('radius'));half=float(g.get('length'))/2
        if outer:r/=np.cos(np.pi/128)
        v=np.array([[r*np.cos(t),r*np.sin(t),z] for t in np.arange(128)*2*np.pi/128 for z in [-half,half]])
        result.append(Piece(name,'collision_0',apply(origin(c),v),'cylinder_subset' if not outer else 'cylinder_outer'))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--r1',required=True);p.add_argument('--expert',required=True);a=p.parse_args()
    scene=Path(a.scene);out=scene/'analysis';out.mkdir(exist_ok=False)
    actors=clean(json.loads((scene/'actor-properties.json').read_text()));write(out/'actors-normalized.json',actors)
    imports=clean(json.loads((scene/'asset-imports.json').read_text()));write(out/'imports-normalized.json',imports)
    params=clean(json.loads((scene/'sim-parameters.json').read_text()))['parameters']
    write(out/'sim-parameters-normalized.json',params)
    # This is a reference input copy, not a changed asset or new state.
    robot_path=ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf';robot=URDF(robot_path)
    shutil.copy2(robot_path,out/'reference-full-robot.urdf')
    arm=next(x for x in actors if x['name']=='z1_arm_articulated')
    base=next(x for x in actors if x['name']=='b1_base_visual_0')
    door=next(x for x in actors if x['name']=='door_0')
    by_env={b['environment_index']:(actor,i) for actor in actors for i,b in enumerate(actor['body_indices'])}
    arm_poses={n:pose(s['pose']) for n,s in zip(arm['body_names'],arm['unstepped_body_states'])}
    joint_q={n:d['pos'] for actor in [base,arm] for n,d in zip(actor['dof_names'],actor['unstepped_dof_states'])}
    FK=robot.fk(joint_q,arm_poses['base'])
    comparisons=[]
    for actor in [base,arm]:
        for i,n in enumerate(actor['body_names']):
            S=pose(actor['unstepped_body_states'][i]['pose']);R=pose(actor['body_indices'][i]['get_rigid_transform'])
            com=vector(actor['body_properties'][i]['com']);cw=S[:3,3]+S[:3,:3]@com
            comparisons.append(dict(actor=actor['name'],body=n,body_state_vs_get_rigid_transform_position_error_m=float(np.linalg.norm(S[:3,3]-R[:3,3])),
                body_state_vs_get_rigid_transform_angle_error_rad=float(np.linalg.norm(Rotation.from_matrix(S[:3,:3]@R[:3,:3].T).as_rotvec())),
                source_FK_vs_unstepped_SDK_position_error_m=float(np.linalg.norm(S[:3,3]-FK[n][:3,3])),
                source_FK_vs_unstepped_SDK_angle_error_rad=float(np.linalg.norm(Rotation.from_matrix(S[:3,:3]@FK[n][:3,:3].T).as_rotvec())),
                exported_COM_in_body_coordinates=com.tolist(),derived_world_COM_position=cw.tolist()))
    write(out/'body-frame-checks.json',dict(comparisons=comparisons,
        caveat='SDK unstepped body readback and the stored DOF buffer need not be a synchronized FK state. Do not label this a measured equilibrium or repair it by writing state. COM world positions are derived from the exported body-space COM translation; no COM-frame quaternion was exported.'))
    # Exact association available: actor/body IDs -> shape property slots.
    # Not available: a documented slot -> cooked primitive/vertex transform map.
    shape_rows=[]
    for actor in actors:
        for i,n in enumerate(actor['body_names']):
            rg=actor['body_shape_ranges'][i]
            shape_rows.append(dict(actor=actor['name'],body=n,environment_body_index=actor['body_indices'][i]['environment_index'],
                actor_shape_start=rg['start'],actor_shape_count=rg['count'],
                filters=[x['filter'] for x in actor['shape_properties'][rg['start']:rg['start']+rg['count']]))
    write(out/'body-shape-associations.json',shape_rows)
    pairs=[]
    for n0,n1 in [('link02','link06'),('gripperStator','fixed_frame'),('gripperStator','door_panel')]:
        u=next(x for x in shape_rows if x['body']==n0);v=next(x for x in shape_rows if x['body']==n1)
        pairs.append(dict(bodies=[n0,n1],shape_masks=[u['filters'],v['filters']],bitmask_rule_allows_all_slot_pairs=all((x&y)==0 for x in u['filters'] for y in v['filters']),
            limitation='This tests exported shape bitmasks only, not any unexported native articulation pair exclusions.'))
    write(out/'filter-checks.json',pairs)
    inner=cylinder_pair(robot);outer=cylinder_pair(robot,True)
    def gaps(poses):
        return dict(inner_gap_m=signed_gap(inner[0].world(poses['link02']),inner[1].world(poses['link06'])),outer_gap_m=signed_gap(outer[0].world(poses['link02']),outer[1].world(poses['link06'])))
    cases=[dict(case='zero_target_source_FK',six_joints=[0]*6,**gaps(robot.fk({}))),
           dict(case='unstepped_SDK_body_pose_readback_not_equilibrium',six_joints_buffer=[joint_q['joint%d'%i] for i in range(1,7)],**gaps(arm_poses))]
    with np.load(a.expert,allow_pickle=True) as data:
        for label,index in [('accepted_expert_first_measured',0),('accepted_expert_final_measured',-1)]:
            q=data['state'][index,2:8];cmd=data['action'][index,2:8]
            P=robot.fk(dict(zip(['joint%d'%i for i in range(1,7)],q)))
            cases.append(dict(case=label,frame_index=index if index>=0 else len(data['state'])-1,six_joints=q.tolist(),six_joint_command=cmd.tolist(),
                max_target_error_rad=float(np.max(np.abs(q-cmd))),**gaps(P)))
    write(out/'home-target-versus-readback-versus-measured.json',dict(cases=cases,expert_source=str(Path(a.expert).resolve()),
        expert_sha256=hashlib.sha256(Path(a.expert).read_bytes()).hexdigest(),
        meaning='All gaps use original analytic source-cylinder inner/outer subsets. They are NOT directly exported cooked-shape distances or solver force measurements. Root rigid transformation does not change this pair gap.'))
    # Reproject the two R1 frame-contact rows under explicit frame hypotheses.
    r1=Path(a.r1);states=[json.loads(l) for l in (r1/'attempt_000/states.jsonl').open()]
    old_map=json.loads((r1/'attempt_000/body-map.json').read_text())
    actor_ids={x['actor_index']:x for x in actors};events=[]
    self_contact=dict(source=str((r1/'attempt_000/contacts.jsonl').resolve()),records=0,nonzero_lambda_records=0,nonzero_by_phase={},examples=[])
    for line in (r1/'attempt_000/contacts.jsonl').open():
        row=json.loads(line)
        for c in row['contacts']:
            body_names=sorted(n.get('body','') for n in c['names'])
            if body_names==['link02','link06']:
                self_contact['records']+=1
                if c['lambda']!=0:
                    self_contact['nonzero_lambda_records']+=1
                    s=states[row['step']];phase=s['phase']
                    self_contact['nonzero_by_phase'][phase]=self_contact['nonzero_by_phase'].get(phase,0)+1
                    if len(self_contact['examples'])<8:
                        P=robot.fk(dict(zip(arm['dof_names'],s['arm_q'])))
                        self_contact['examples'].append(dict(step=row['step'],phase=phase,lambda_value=c['lambda'],
                            raw_minDist=c['minDist'],raw_initialOverlap=c['initialOverlap'],six_joint_target=s['arm_command'][:6],
                            six_joints_measured=s['arm_q'][:6],**gaps(P)))
            if sorted(n.get('body','') for n in c['names'])!=['fixed_frame','gripperStator']:continue
            s=states[row['step']];body_T=[];com=[]
            for j in [0,1]:
                old_id=c['body%d'%j];actor,idx=by_env[old_id];expected=old_map[str(old_id)]
                assert actor['body_names'][idx]==expected['body']
                assert actor['actor_index']==expected['actor_handle']
                if actor['name']=='z1_arm_articulated':
                    raw=s['arm_body_states'][idx];T=np.eye(4);T[:3,3]=raw[0][0];T[:3,:3]=Rotation.from_quat(raw[0][1]).as_matrix()
                else:
                    # R1 did not retain door body states. Its fixed frame pose
                    # is reconstructed from the unchanged nominal actor init.
                    T=np.eye(4);T[:3,3]=[2.61,0,.01];T[:3,:3]=Rotation.from_euler('z',np.pi/2).as_matrix()
                body_T.append(T);com.append(vector(actor['body_properties'][idx]['com']))
            p0=vector(c['localPos0']);p1=vector(c['localPos1']);h={}
            for bits in [(0,0),(1,0),(0,1),(1,1)]:
                w0=apply(body_T[0],p0+bits[0]*com[0]);w1=apply(body_T[1],p1+bits[1]*com[1])
                h[str(bits)]=dict(point0=w0.tolist(),point1=w1.tolist(),distance_m=float(np.linalg.norm(w1-w0)))
            events.append(dict(step=row['step'],force=c['lambda'],raw_initialOverlap=c['initialOverlap'],raw_minDist=c['minDist'],
                frame_hypotheses=h,body_ids=[c['body0'],c['body1']],raw_record_has_shape_ids=any('shape' in k.lower() for k in c),
                raw_record_has_substep_or_time=any('time' in k.lower() or 'step' in k.lower() for k in c),
                interval_s=[row['step']*params['dt'],(row['step']+1)*params['dt']],
                physics_substeps=params['substeps'],contact_collection=params['physx']['contact_collection'].get('name'),
                limitation='Old arm states are post-step; fixed frame is nominal reconstruction; no old per-substep poses or actor shape properties exist. These hypotheses are not a calibrated force-point certification.'))
    write(out/'r1-contact-coordinate-hypotheses.json',events)
    self_contact['meaning']='Existing first-round execution only. Records across all substeps must not be counted as independent collisions or summed as step forces. Source post-state separation does not mean absence of solver contact. This is not a new run or a self-force measurement of the accepted dataset episode.'
    write(out/'r1-existing-self-contact-records.json',self_contact)
    # API field availability is obtained from actual saved objects, not docs.
    fields=sorted(set(k for x in actors for prop in x['shape_properties'] for k in prop))
    stat=clean(json.loads((scene/'inspection-result.json').read_text()))
    summary=dict(scene_status=stat,actors=[dict(name=x['name'],body_count=len(x['body_names']),shape_count=x['shape_count']) for x in actors],
        shape_property_fields=fields,per_shape_contact_offset_exported='contact_offset' in fields,per_shape_rest_offset_exported='rest_offset' in fields,
        global_contact_offset=params['physx']['contact_offset'],global_rest_offset=params['physx']['rest_offset'],dt=params['dt'],substeps=params['substeps'],
        contact_collection=params['physx']['contact_collection'].get('name'),
        max_source_FK_vs_unstepped_SDK_position_error_m=max(x['source_FK_vs_unstepped_SDK_position_error_m'] for x in comparisons),
        max_source_FK_vs_unstepped_SDK_angle_error_rad=max(x['source_FK_vs_unstepped_SDK_angle_error_rad'] for x in comparisons),
        max_state_vs_get_rigid_position_error_m=max(x['body_state_vs_get_rigid_transform_position_error_m'] for x in comparisons),
        home_cases=cases,r1_self_contact_records=self_contact['records'],r1_self_contact_nonzero_records=self_contact['nonzero_lambda_records'],full_task_launch_certified=False,
        blockers=['No documented per-shape cooked geometry/type/local-pose export or actual shape contact/rest offset readback from this binding',
                  'Contact rows have no shape IDs or per-substep timestamps while collection includes two substeps',
                  'Unstepped DOF buffer and body pose readback are not a synchronized FK state',
                  'Contact point/force-unit/normal convention, physical closing work and runtime ClosingEvidence are not certified'],
        no_home_asset_physics_or_filter_change=True)
    write(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
