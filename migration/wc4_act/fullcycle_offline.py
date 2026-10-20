"""Offline candidate comparison, run only on pro5000's Gym CPU environment."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fullcycle_geometry import URDF, transform, apply, close_pose

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(exist_ok=False)
    robot = URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf', include_visual=True)
    names = ['joint%d'%i for i in range(1,7)]
    lower, upper = [], []
    for n in names:
        j = robot.root.find("joint[@name='%s']"%n)
        lo,hi = float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))
        home = np.clip(0,lo,hi)
        lower.append(home-(home-lo)*.75); upper.append(home+(hi-home)*.75)
    lower,upper = np.array(lower),np.array(upper)
    # Precompute chain: the simulator splits the arm at root 'base'.
    chain = []
    child_j = {j[3]:j for j in robot.joints}
    current = 'ee_gripper_link'
    while current in child_j:
        j = child_j[current]; chain.insert(0,j); current=j[2]
    def fk(q):
        d = dict(zip(names,q)); T=np.eye(4)
        from fullcycle_geometry import rot
        for name,kind,parent,child,O,axis in chain:
            T=T@O
            if name in d:
                M=np.eye(4);M[:3,:3]=rot(axis,d[name]);T=T@M
        return T
    def solve(target, seed):
        def residual(q):
            F=fk(q)
            return np.r_[(F[:3,3]-target[:3,3])*4,Rotation.from_matrix(target[:3,:3]@F[:3,:3].T).as_rotvec()]
        best=None
        for s in [seed, np.clip([0,1.2,-1.2,-.5,0,0],lower,upper), np.clip([-.8,1.5,-1.,-.5,.8,0],lower,upper)]:
            r=least_squares(residual,np.clip(s,lower+1e-6,upper-1e-6),bounds=(lower,upper),max_nfev=120)
            if best is None or np.linalg.norm(r.fun)<np.linalg.norm(best.fun): best=r
            if np.linalg.norm(r.fun)<1e-5: break
        F=fk(best.x)
        return best.x,float(np.linalg.norm(F[:3,3]-target[:3,3])),float(np.linalg.norm(Rotation.from_matrix(target[:3,:3]@F[:3,:3].T).as_rotvec()))
    legs={}
    for side in ['FL','FR','RL','RR']:
        legs.update({side+'_hip_joint':0,side+'_thigh_joint':.8,side+'_calf_joint':-1.5})
    candidates=[]
    for radius in [.55,.65,.75]:
        for distance in [.55,.65,.75,.85]:
            for z in [.75,.85,.95]:
                B,E=close_pose(math.pi/4,radius,distance,z)
                # Offset the target 2 cm behind the panel; robot creates contact
                # by advancing at execution, never by changing door coordinates.
                E[:3,3]-=.02*E[:3,0]
                q,pe,re=solve(np.linalg.inv(B)@E,np.zeros(6))
                max_x=-1e9; min_panel_clear=1e9; min_ground=1e9
                if pe<.005 and re<.035:
                    for angle in np.linspace(0,math.pi/2,31):
                        B,E=close_pose(angle,radius,distance,z)
                        poses=robot.fk(dict(legs,**dict(zip(names,q)),jointGripper=0),B)
                        cloud=robot.cloud(poses)
                        max_x=max(max_x,float(cloud[:,0].max()))
                        min_ground=min(min_ground,float(cloud[:,2].min()))
                        # All geometry except distal contact links stays on the
                        # closing side of the infinite panel plane.
                        n=E[:2,0];panel=E[:2,3]
                        body=robot.cloud(poses,exclude=['gripperStator','gripperMover','link06','wrist_up_mount','wrist_down_mount'])
                        min_panel_clear=min(min_panel_clear,float(np.min((panel-body[:,:2])@n)))
                candidates.append(dict(radius=radius,distance=distance,z=z,q=q.tolist(),position_error_m=pe,orientation_error_rad=re,max_robot_x=max_x,min_ground=min_ground,non_distal_plane_clearance=min_panel_clear,accepted=bool(pe<.005 and re<.035 and max_x<2.495 and min_panel_clear>.015)))
    viable=[x for x in candidates if x['accepted']]
    exact=[x for x in viable if x['position_error_m']<1e-4 and x['orientation_error_rad']<1e-3]
    selected=next((x for x in exact if x['radius']==.65 and x['distance']==.75 and x['z']==.95),None)
    if selected is None and exact: selected=exact[0]
    fixed=[]
    if selected:
        # Same start/middle/end contacts with a fixed base at the middle pose.
        B,_=close_pose(math.pi/4,selected['radius'],selected['distance'],selected['z'])
        q=np.array(selected['q'])
        for angle in np.linspace(math.pi/2,0,19):
            _,E=close_pose(angle,selected['radius'],selected['distance'],selected['z']);E[:3,3]-=.02*E[:3,0]
            q,pe,re=solve(np.linalg.inv(B)@E,q)
            fixed.append(dict(angle_rad=float(angle),position_error_m=pe,orientation_error_rad=re,q=q.tolist()))
    home=dict(legs,**dict(zip(names,np.clip(np.zeros(6),lower,upper))),jointGripper=-1.57)
    home_cloud=robot.cloud(robot.fk(home))
    route=[]
    route_audit=[]
    if selected:
        B,_=close_pose(math.pi/2,selected['radius'],selected['distance'],selected['z'])
        route=[[.8,.18,math.pi],[.8,.18,-math.pi/2],[.8,-1.6,-math.pi/2],
               [.8,-1.6,0],[float(B[0,3]),-1.6,0],
               [float(B[0,3]),-1.6,math.pi/2],
               [float(B[0,3]),float(B[1,3]),math.pi/2],
               [float(B[0,3]),float(B[1,3]),float(math.atan2(B[1,0],B[0,0]))]]
        poses_to_check=[]
        prior=[1.26,.18,math.pi]
        for goal in route:
            delta=(goal[2]-prior[2]+math.pi)%(2*math.pi)-math.pi
            for t in np.linspace(0,1,41):
                pose=[(1-t)*prior[0]+t*goal[0],(1-t)*prior[1]+t*goal[1],prior[2]+t*delta]
                poses_to_check.append((pose,np.zeros(6),'navigate_home'))
            prior=goal
        for t in np.linspace(0,1,101):
            poses_to_check.append((route[-1],np.array(selected['q'])*t,'deploy'))
        # Door slab occupies local x=[0,.917], y=[-.043,.002].
        D=transform([2.467,-.463,.01],[0,0,math.pi])
        invD=np.linalg.inv(D)
        for pose,q,stage in poses_to_check:
            P=robot.fk(dict(legs,**dict(zip(names,q)),jointGripper=0 if stage=='deploy' else -1.57),transform([pose[0],pose[1],.5],[0,0,pose[2]]))
            cloud=robot.cloud(P)
            overlap=[]
            for name,pts in robot.clouds.items():
                x=apply(invD,apply(P[name],pts));lo=x.min(0);hi=x.max(0)
                if np.all(hi>np.array([0,-.043,.05])) and np.all(lo<np.array([.917,.002,2.2])):
                    overlap.append(name)
            route_audit.append(dict(stage=stage,pose=pose,max_x=float(cloud[:,0].max()),possible_panel_intersections=overlap))
        if any(x['max_x']>=2.5 or x['possible_panel_intersections'] for x in route_audit):
            selected=None
    report=dict(status='candidate_found' if selected else 'no_feasible_candidate',method='URDF mesh/primitive conservative AABBs; full six-axis pose IK; 31 sweep samples per viable candidate; fixed midpoint base comparison is not a global impossibility proof',door_plane_x=2.5,hinge_xy=[2.467,-.463],door_closed_rad=0,nominal_robot=[4.1,.18,.5,math.pi],joint_lower=lower.tolist(),joint_upper=upper.tolist(),home=home,home_geometry_bounds=[home_cloud.min(0).tolist(),home_cloud.max(0).tolist()],candidates=candidates,selected=selected,fixed_midpoint_comparison=fixed,route=route,route_audit=route_audit,limitations=['kinematic evidence, not collision/contact dynamics validation','Runtime geometry/contact guard still required','Nominal no-randomization; no learning inputs added'])
    (out/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','selected','home_geometry_bounds']}))


if __name__=='__main__': main()
