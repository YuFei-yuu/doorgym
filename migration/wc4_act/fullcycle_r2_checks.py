"""R2 CPU gates: original evidence interpretation and full candidate geometry.

No Isaac Gym import, no dynamics execution, no model loading, no scene reset.
"""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from fullcycle_geometry import URDF, transform, apply, origin
from fullcycle_collision import collision_pieces, minimum_gap, signed_gap, Piece
from fullcycle_contract import classify_contact, smooth_gripper, timing, ClosingEvidence
from fullcycle_r2_path import stress_trace

ROOT=Path(__file__).resolve().parents[2]


def tests():
    checks={}
    box=Piece('a','box',np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]]),'box')
    I=np.eye(4);sep=transform([1.001,0,0]);penetrate=transform([.999,0,0])
    checks['zero_force_proven_separation']=classify_contact(0,signed_gap(box.world(I),box.world(sep)),reliable=True)=='proximity_separated'
    checks['zero_force_penetration_stops']=classify_contact(0,signed_gap(box.world(I),box.world(penetrate)),reliable=True)=='penetration_stop'
    checks['zero_force_boundary_stops']=classify_contact(0,0,reliable=True)=='uncertain_boundary_stop'
    checks['uncertified_separation_not_allowed']=classify_contact(0,.05)=='uncertain_stop'
    checks['loaded_separated_unexpected_stops']=classify_contact(.001,.002)=='unexpected_loaded_stop'
    checks['tiny_nonzero_unexpected_force_stops']=classify_contact(1e-9,.002)=='unexpected_loaded_stop'
    checks['unreliable_geometry_stops']=classify_contact(0,.1,reliable=False)=='uncertain_stop'
    stamps=[timing(i,.02) for i in range(50)]
    checks['50_control_25_observations']=len(stamps)==50 and sum(x['record_25hz'] for x in stamps)==25
    checks['post_observation_times']=[round(x['measured_post_time_s'],9) for x in stamps if x['record_25hz']]==[round(.02+.04*i,9) for i in range(25)]
    initial=-1.5707963705062866;home=-1.1775000095367432
    ramp=np.array([smooth_gripper(initial,home,t) for t in np.arange(51)*.02])
    checks['gripper_start_matches_measurement']=ramp[0]==initial
    checks['gripper_no_initial_jump']=np.max(np.abs(np.diff(ramp)))<.015 and abs(ramp[-1]-home)<1e-12
    # Exercise the REAL opening-command seam with an instrumented API double.
    # This is a call-order/control-cadence test, not a simulated robot rollout.
    from types import SimpleNamespace
    from fullcycle_expert import opening_command
    events=[]
    class API:
        def refresh_rigid_body_state_tensor(self,s):events.append('rigid')
        def refresh_dof_state_tensor(self,s):events.append('dof')
        def refresh_jacobian_tensors(self,s):events.append('jacobian')
    names=['joint%d'%i for i in range(1,7)]+['jointGripper']
    st=SimpleNamespace(_fullcycle_gym=API(),_fullcycle_sim=None,env=None,actor_handles=[],arm_actor=None,
        args=SimpleNamespace(robot_z=.5,robot_pitch=0.),traj={},home_positions=np.r_[np.zeros(6),home],
        dof_positions=np.zeros(7),ik_state=SimpleNamespace(lower=np.array([-2]*6+[-1.1775]),upper=np.ones(7)))
    def fake_ik(*args):events.append('ik');st.dof_positions[:6]+=.001
    m=SimpleNamespace(set_robot_base_pose=lambda *a:events.append('base'),set_ik_target=lambda *a:events.append('target'),update_arm_ik_targets_for_env=fake_ik)
    actions=[]
    for i in range(50):
        actions.append(opening_command(m,st,i,.02,names,'initial_hold',np.zeros(2),0.,np.zeros(3),None,home,np.zeros(2),0.,initial,(-1.57,0)))
    checks['actual_opening_50_ik_calls']=events.count('ik')==50
    checks['author_call_order_preserved']=events==['base','rigid','dof','jacobian','target','ik']*50
    checks['physical_gripper_initial_clip_small']=abs(actions[0][-1]-initial)<.001
    checks['no_25hz_hold_in_opening']=np.allclose(np.diff(np.array(actions)[:,2:8],axis=0),.001)
    # Regression: incidental contact once plus a later freely closing door
    # must not latch. Wrong torque, receding EE or absent load also cannot latch.
    checks['no_incidental_contact_latch']=True
    for kind in ['none','wrong_torque','receding_ee','zero_load']:
        e=ClosingEvidence(.02);a=1.4
        for i in range(120):
            b=max(0,a-.02)
            c=dict(end_panel=True,geometry_reliable=True,force=1.,hinge_torque=-1.,closing_velocity=.1,surface_gap=.001)
            if kind=='wrong_torque':c['hinge_torque']=1
            if kind=='receding_ee':c['closing_velocity']=-.1
            if kind=='zero_load':c['force']=0
            e.update(a,b,0,[] if kind=='none' else [c],0,.002,True,True);a=b
        checks['no_incidental_contact_latch'] &= not e.can_latch(1.4,True,True)
    e=ClosingEvidence(.02);a=1.4
    for i in range(120):
        b=max(0,a-.02)
        c=dict(end_panel=True,geometry_reliable=True,force=1.,hinge_torque=-1.,closing_velocity=.1,surface_gap=.001)
        e.update(a,b,0,[c],0,.002,True,True);a=b
    checks['aligned_support_then_settle_can_latch']=e.can_latch(1.4,True,True)
    if not all(checks.values()):raise AssertionError(checks)
    return checks


def home_overlap_proof(robot):
    # A convex intersection of INSCRIBED cylinder polygons is an intersection
    # of subsets of the actual analytic cylinders. This resolves the concern
    # that the 32-sided OUTER approximation invented the home intersection.
    pieces=[];details=[]
    for name in ['link02','link06']:
        c=robot.links[name].findall('collision')[0];g=c.find('geometry/cylinder')
        radius=float(g.get('radius'));length=float(g.get('length'))
        a=np.arange(128)*2*np.pi/128
        p=np.array([[radius*np.cos(t),radius*np.sin(t),z] for t in a for z in [-length/2,length/2]])
        pieces.append(Piece(name,'collision_0_inner',apply(origin(c),p),'inscribed_cylinder'))
        details.append(dict(link=name,radius_m=radius,length_m=length,origin=origin(c).tolist()))
    poses=robot.fk({})
    gap=signed_gap(pieces[0].world(poses[pieces[0].link]),pieces[1].world(poses[pieces[1].link]))
    return dict(home_six_joints_rad=[0]*6,method='SAT on 128-gon INSCRIBED finite-cylinder subsets, using original URDF origins and FK',signed_subset_gap_m=gap,
        subset_intersection=gap<-1e-5,meaning='If true, analytic source cylinders intersect; this does not prove a measured solver force or validate cooked shape metadata',geometry=details,
        same_rigid_cluster=False,adjacent_links=False,original_assets_modified=False)


def r1_contacts(source,robot,door,robot_pieces,door_pieces):
    states=[json.loads(l) for l in (source/'attempt_000/states.jsonl').open()]
    mapping=json.loads((source/'attempt_000/body-map.json').read_text())
    names=json.loads((source/'attempt_000/reference.json').read_text())['dof_names']
    DP=door.fk({},transform([2.61,0,.01],[0,0,math.pi/2]))
    frames=[x for x in door_pieces if x.link=='fixed_frame']
    stator=[x for x in robot_pieces if x.link=='gripperStator']
    raw=[json.loads(l) for l in (source/'attempt_000/contacts.jsonl').open()]
    selected=[]
    for record in raw:
        for c in record['contacts']:
            if sorted(n.get('body','') for n in c['names'])!=['fixed_frame','gripperStator']:continue
            k=record['step'];s=states[k];x,y,yaw=s['actual_base']
            RP=robot.fk(dict(zip(names,s['arm_q'])),transform([x,y,.5],[0,0,yaw]))
            gap,pair=minimum_gap(stator,RP,frames,DP)
            # Source R1 lacks action-time pre-solve rigid transforms; do not
            # pretend preceding post-solve state equals the missing pre state.
            world=[]
            for n in [0,1]:
                v=mapping[str(c['body%d'%n])]
                T=DP['fixed_frame'] if v['actor_handle']==2 else RP[v['body']]
                world.append(apply(T,np.array([c['localPos%d'%n][axis] for axis in 'xyz'])))
            selected.append(dict(step=k,interval_s=[k*.02,(k+1)*.02],previous_video_source_step=k-1 if k%2 else k,video_frame=(k-1)//2 if k%2 else k//2,raw_force=c['lambda'],raw_minDist=c['minDist'],raw_initialOverlap=c['initialOverlap'],post_geometry_gap_m=gap,geometry_pair=pair,post_feature_points=[v.tolist() for v in world],post_feature_distance_m=float(np.linalg.norm(world[1]-world[0])),classification=classify_contact(c['lambda'],gap),pre_action_geometry_available=False))
    return selected


def full_path_checks(source,robot,door,rpieces,dpieces,out,plan_path=None):
    samples,budget=stress_trace(source,robot,door,plan_path=plan_path)
    names=['joint%d'%i for i in range(1,7)]+['jointGripper']
    legs={}
    for side in ['FL','FR','RL','RR']:
        legs.update({side+'_hip_joint':0,side+'_thigh_joint':.8,side+'_calf_joint':-1.5})
    # Explicit mechanical-joint exclusions, not a blanket self-collision mask.
    parent={n:n for n in robot.links}
    def find(n):
        while parent[n]!=n:n=parent[n]
        return n
    for name,kind,p,c,T,axis in robot.joints:
        if kind=='fixed':parent[find(c)]=find(p)
    adjacent={frozenset([find(p),find(c)]) for name,kind,p,c,T,axis in robot.joints if kind!='fixed'}
    self_pairs=[]
    for i,a in enumerate(rpieces):
        for j,b in enumerate(rpieces[i+1:],i+1):
            if find(a.link)==find(b.link) or frozenset([find(a.link),find(b.link)]) in adjacent:continue
            self_pairs.append((i,j))
    # Side wall GEOMETRIC keepouts use the frozen author dimensions. The
    # author's generated wall URDF is visual-only: do not claim force contact.
    # wc4 bbox yawed by pi/2 gives opening y=[-.525,.525].
    wall_vertices=np.array([[x,y,z] for x in [-.04,.04] for y in [-.5,.5] for z in [-1.1,1.1]])
    walls=[Piece('wall_'+str(i),'wall',wall_vertices,'box') for i in range(2)]
    WP={'wall_0':transform([2.5,-1.025,1.11]),'wall_1':transform([2.5,1.025,1.11])}
    envpieces=dpieces+walls;records=[];counts={};clearances={};previous_cloud=None;max_chord=0.
    exclusions=dict(fixed_clusters={n:find(n) for n in robot.links},adjacent_clusters=[sorted(x) for x in adjacent])
    (out/'mechanical-adjacency.json').write_text(json.dumps(exclusions,indent=2)+'\n')
    for k,s in enumerate(samples):
        # Evaluate every 40ms plus all phase boundaries. Report the sampling
        # displacement bound; sampled separation is not continuous proof.
        if k%2 and k>0 and s['phase']==samples[k-1]['phase'] and k<len(samples)-1:continue
        RP=robot.fk(dict(legs,**dict(zip(names,s['q']))),transform([*s['xy'],.5],[0,0,s['yaw']]))
        DP=door.fk({'door_hinge_joint':s['door'],'handle_spindle_joint':s['handle']},transform([2.61,0,.01],[0,0,math.pi/2]));DP.update(WP)
        RW=[p.world(RP[p.link]) for p in rpieces];EW=[p.world(DP[p.link]) for p in envpieces]
        low=np.array([x[0].min(0) for x in RW]);high=np.array([x[0].max(0) for x in RW]);el=np.array([x[0].min(0) for x in EW]);eh=np.array([x[0].max(0) for x in EW])
        all_points=np.concatenate([x[0] for x in RW])
        if previous_cloud is not None:max_chord=max(max_chord,float(np.linalg.norm(all_points-previous_cloud,axis=1).max()))
        previous_cloud=all_points
        checks=[]
        for i,j in self_pairs:
            broad=float(np.maximum(low[j]-high[i],low[i]-high[j]).max())
            if broad<.01:checks.append((signed_gap(RW[i],RW[j]),'self',rpieces[i],rpieces[j]))
        broad=np.maximum(el[None,:,:]-high[:,None,:],low[:,None,:]-eh[None,:,:]).max(2)
        for i,j in np.argwhere(broad<.01):
            checks.append((signed_gap(RW[i],EW[j]),'environment',rpieces[i],envpieces[j]))
        for gap,kind,a,b in checks:
            label=s['phase']+':'+kind
            clearances[label]=min(clearances.get(label,float('inf')),gap)
            if gap<-1e-5:
                counts[label]=counts.get(label,0)+1
                if len(records)<100:records.append(dict(trace_step=k,time_s=s['t'],phase=s['phase'],kind=kind,signed_convex_gap_m=gap,pair=[a.link,a.name,b.link,b.name],kinds=[a.kind,b.kind],interpretation='sampled convex intersection; mesh hulls can be conservative, requires resolution before execution'))
    q=np.array([s['q'] for s in samples]);increments=np.abs(np.diff(q,axis=0))
    np.savez_compressed(out/'kinematic-stress-trace.npz',time=np.array([s['t'] for s in samples]),base=np.array([[*s['xy'],s['yaw']] for s in samples]),joint_targets=q,door_angle_scenario=np.array([s['door'] for s in samples]),phase=np.array([s['phase'] for s in samples]))
    summary=dict(budget=budget,phases=sorted(set(s['phase'] for s in samples)),samples=len(samples),sample_every_sim_steps=2,max_between_checked_vertex_displacement_m=max_chord,
        opening_target_error_max_m=max(s['ee_error'] for s in samples if s['target_pos'] is not None),joint_command_increment_max_rad=increments.max(0).tolist(),
        convex_intersection_counts=counts,first_intersections=records,minimum_checked_signed_gaps_m=clearances,
        coverage=['source opening target function at 50Hz with ideal door motion','far-side navigation','deploy','closing circle','release','retract','heading restore','final hold','frame','walls','wrists','nonadjacent self collision'],
        limitations=['prescribed door scenario is not execution evidence','finite-difference FK Jacobian is not the Gym Jacobian or physical tracking','convex mesh envelope overlap is not always exact mesh penetration','sampled clearance does not certify unsampled intervals'])
    (out/'full-path-analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--tests-only',action='store_true');p.add_argument('--geometry');a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    for path in [*Path(__file__).parent.glob('fullcycle*.py'),Path(__file__).parent/'overview.py']:
        shutil.copy2(path,out/'source'/path.name)
    rp=ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf';dp=ROOT/'high-level/data/asset/door_set/wc4/model.urdf'
    robot=URDF(rp);door=URDF(dp);rpieces=collision_pieces(rp);dpieces=collision_pieces(dp)
    regression={k:bool(v) for k,v in tests().items()};(out/'targeted-tests.json').write_text(json.dumps(regression,indent=2)+'\n')
    home=home_overlap_proof(robot)
    (out/'home-cylinder-proof.json').write_text(json.dumps(home,indent=2)+'\n')
    contacts=r1_contacts(source,robot,door,rpieces,dpieces)
    (out/'r1-contact-analysis.json').write_text(json.dumps(contacts,indent=2)+'\n')
    if a.tests_only:
        print(json.dumps(dict(tests=regression,home=home,r1_contacts=contacts)),flush=True);return
    plan_path=Path(a.geometry) if a.geometry else source/'offline-v2/geometry.json'
    path=full_path_checks(source,robot,door,rpieces,dpieces,out,plan_path=plan_path)
    # A source-geometry separation result is not enough to resolve the local
    # feature-coordinate mismatch of the first run. Keep that gate closed.
    reasons=['R1 contact local-feature coordinates do not reconcile with source collision features; action-time transforms/shape properties missing']
    if not path['budget']['within_90s']:reasons.append('selected route and control timing exceed 90 second envelope in ideal tracking')
    if path['convex_intersection_counts']:reasons.append('full-path convex intersection candidates require resolution')
    gate=dict(status='offline_gates_failed',reasons=reasons,new_rollouts=0,geometry_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('fullcycle*.py')},runtime_contact_classification_certified=False,runtime_closing_evidence_certified=False)
    (out/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps(dict(targeted_tests=regression,r1_contacts=contacts,budget=path['budget'],intersection_counts=path['convex_intersection_counts'],gate=gate)),flush=True)


if __name__=='__main__':main()
