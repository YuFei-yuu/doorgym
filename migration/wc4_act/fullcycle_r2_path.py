"""Kinematic stress trace, NOT physics, state replay, or execution evidence.

Opening targets are the original author function at 50 Hz. Door motion is an
explicit idealized scenario for checking reach/collisions; it is never applied
to any simulator. New stages keep the R1 coordinated-circle candidate.
"""
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from fullcycle_geometry import transform, close_pose
from fullcycle_expert import angle_error
from fullcycle_contract import smooth_gripper
from fullcycle_navigation import route_directions, waypoint_command

ROOT=Path(__file__).resolve().parents[2]


def stress_trace(source,robot,door,plan_path=None):
    sys.path.insert(0,str(ROOT/'high-level/float_ik'))
    import isaacgym_float_ik_a2w_basearn_push_door_parallel as author
    args=SimpleNamespace(**json.loads((source/'attempt_000/effective-config.json').read_text()))
    args.door_twin_skill_program=author.load_skill_program(ROOT/'high-level/float_ik/door_twin/examples/wc4_smoke_skill.json')
    args.sim_dt=.02
    plan_path=Path(plan_path) if plan_path is not None else source/'offline-v2/geometry.json'
    plan=json.loads(plan_path.read_text());chosen=plan['selected']
    directions=route_directions(plan)
    qclose=np.r_[chosen['q'],0.];q=np.r_[np.zeros(6),-1.5707963705062866]
    home=np.r_[np.zeros(6),-1.1775000095367432]
    lo=np.r_[plan['joint_lower'],-1.1775];hi=np.r_[plan['joint_upper'],0.]
    names=['joint%d'%i for i in range(1,7)]+['jointGripper']
    xy=np.array([4.1,.18]);yaw=math.pi
    P=robot.fk(dict(zip(names,q)),transform([*xy,.5],[0,0,yaw]))
    T=P['ee_gripper_link'];ik=SimpleNamespace(current_pos_np=T[:3,3],current_quat_np=Rotation.from_matrix(T[:3,:3]).as_quat(),target_quat_np=Rotation.from_matrix(T[:3,:3]).as_quat())
    runtime=SimpleNamespace(spec={'name':'wc4','path':'wc4/model.urdf'},handle_body_index=2,handle_goal_offset=np.array([-.05,0,.02]),open_stage=False,dof_lower=np.array([0,0]),dof_upper=np.array([math.pi/2,math.pi/4]))
    scene={'angle':0.,'handle':0.}
    rootD=transform([2.61,0,.01],[0,0,math.pi/2])
    def get_handle(*args):
        F=door.fk({'door_hinge_joint':scene['angle'],'handle_spindle_joint':scene['handle']},rootD)['lever_handle']
        return F[:3,3].astype(np.float32),Rotation.from_matrix(F[:3,:3]).as_quat().astype(np.float32)
    author.get_body_pose=get_handle
    author.get_actor_dof_state=lambda *a:(np.array([scene['angle'],scene['handle']]),np.zeros(2))
    traj={'base_xy':xy.copy()};samples=[];return_start=None
    end=args.walk_steps+args.initial_hold_steps+args.grasp_steps+args.grasp_hold_steps+args.gripper_close_steps+args.handle_rotate_steps+args.door_push_steps+args.traverse_steps+args.return_home_steps
    rotate_end=args.walk_steps+args.initial_hold_steps+args.grasp_steps+args.grasp_hold_steps+args.gripper_close_steps+args.handle_rotate_steps
    push_end=rotate_end+args.door_push_steps
    def fk_arm(q_,xy_,yaw_):
        return robot.arm_fk(q_[:6]) if xy_ is None else robot.fk(dict(zip(names,q_)),transform([*xy_,.5],[0,0,yaw_]))['ee_gripper_link']
    for step in range(end+1):
        alpha=np.clip((step-rotate_end)/max(1,args.door_push_steps),0,1)
        scene['angle']=math.pi/2*(alpha*alpha*(3-2*alpha))
        scene['handle']=math.pi/4*float(np.clip((step-(rotate_end-args.handle_rotate_steps))/max(1,args.handle_rotate_steps),0,1)) if step<push_end else 0.
        runtime.open_stage=step>=rotate_end
        F=fk_arm(q,xy,yaw);ik.current_pos_np=F[:3,3].astype(np.float32);ik.current_quat_np=Rotation.from_matrix(F[:3,:3]).as_quat().astype(np.float32)
        phase,xy,yaw,pos,quat,gripper,_=author.trajectory_targets(step,args,runtime,None,None,None,ik,np.array([4.1,.18]),np.array([3.2,.18]),np.array([2.,.18]),np.array([1.26,.18]),math.pi,math.pi,math.pi,traj)
        if phase in ('walk','hold_home'):q=home.copy()
        elif phase=='return_home':
            if return_start is None:return_start=q.copy()
            q=(1-traj['return_home_alpha'])*return_start+traj['return_home_alpha']*home
        else:
            F=fk_arm(q,xy,yaw);J=np.zeros((6,6));eps=1e-5
            for j in range(6):
                plus=q.copy();plus[j]+=eps;G=fk_arm(plus,xy,yaw)
                J[:3,j]=(G[:3,3]-F[:3,3])/eps
                J[3:,j]=Rotation.from_matrix(G[:3,:3]@F[:3,:3].T).as_rotvec()/eps
            if quat is None:
                A=J[:3];err=np.asarray(pos)-F[:3,3]
            else:
                target=Rotation.from_quat(quat).as_matrix();rv=Rotation.from_matrix(target@F[:3,:3].T).as_rotvec();theta=np.linalg.norm(rv)
                orn=rv*np.sin(theta/2)/theta if theta>1e-9 else rv/2
                weights=np.r_[np.ones(3),np.full(3,args.ik_rot_weight)]
                A=J*weights[:,None];err=np.r_[args.ik_pos_gain*(pos-F[:3,3]),args.ik_rot_gain*orn]*weights
            delta=A.T@np.linalg.solve(A@A.T+np.eye(len(err))*args.ik_damping**2,err)
            q[:6]=np.clip(q[:6]+np.clip(delta,-args.ik_max_step,args.ik_max_step),lo[:6],hi[:6])
            q[-1]=np.clip(gripper,lo[-1],hi[-1]);ik.target_quat_np=quat
        if step*.02<1:q[-1]=np.clip(smooth_gripper(-1.5707963705062866,home[-1],step*.02),-1.57,0)
        F=fk_arm(q,xy,yaw)
        samples.append(dict(t=(step+1)*.02,phase=phase,xy=np.asarray(xy).copy(),yaw=float(yaw),q=q.copy(),door=scene['angle'],handle=scene['handle'],ee_error=float(np.linalg.norm(F[:3,3]-pos)),target_pos=np.asarray(pos).tolist(),target_quat=None if quat is None else np.asarray(quat).tolist()))
    # Integrate the existing post-traversal navigation controller at its 25 Hz
    # command cadence with the existing 50 Hz nonholonomic base integrator.
    nav=0;phase='navigate';started=samples[-1]['t'];t=started;start_q=q.copy();action=np.r_[0.,0.,home]
    close_yaw_start=None;theta=math.pi/2;restore_start=None
    for i in range(12000):
        elapsed=t-started
        if i%2==0:
            v=w=0.;qtarget=home.copy()
            if phase=='navigate':
                if nav>=len(plan['route']):phase='deploy';started=t;start_q=q.copy()
                else:
                    v,w,reached=waypoint_command(xy,yaw,plan['route'][nav],directions[nav])
                    if reached:nav+=1
            elif phase=='deploy':
                z=float(np.clip(elapsed/4,0,1));s=z**3*(10-15*z+6*z*z);qtarget=(1-s)*start_q+s*qclose
                if elapsed>=5:phase='close';started=t;close_yaw_start=yaw
            elif phase=='close':
                qtarget=qclose;rate=.075 if theta>=math.radians(4) else .02
                if theta>0:
                    v=rate*math.hypot(chosen['radius'],chosen['distance']);w=-rate
                elif elapsed>0:phase='settle';started=t
            elif phase=='settle':
                qtarget=qclose
                if elapsed>=.5:phase='release';started=t
            elif phase=='release':
                qtarget=qclose
                if elapsed<1.5:v=-.07;w=.07/math.hypot(chosen['radius'],chosen['distance'])
                else:phase='retract';started=t;start_q=q.copy()
            elif phase=='retract':
                z=float(np.clip(elapsed/4,0,1));s=z**3*(10-15*z+6*z*z);qtarget=(1-s)*start_q+s*home
                if elapsed>=5:phase='restore_heading';started=t
            elif phase=='restore_heading':
                e=angle_error(math.pi,yaw);w=float(np.clip(2*e,-.6,.6)) if abs(e)>.003 else 0
                if abs(e)<.003:phase='final_hold';started=t
            elif phase=='final_hold' and elapsed>=2:break
            action=np.r_[v,w,qtarget]
        xy=np.asarray(xy)+action[0]*.02*np.array([math.cos(yaw),math.sin(yaw)]);yaw+=action[1]*.02;q=action[2:].copy();t+=.02
        if phase=='close':theta=max(0,math.pi/2+yaw-close_yaw_start)
        if phase not in ['navigate','deploy','close']:theta=0.
        samples.append(dict(t=t,phase=phase,xy=np.asarray(xy).copy(),yaw=float(yaw),q=q.copy(),door=theta if phase=='close' else math.pi/2 if phase in ['navigate','deploy'] else 0.,handle=0.,ee_error=0.,target_pos=None,target_quat=None))
    return samples,dict(method='No dynamics. Prescribed ideal opening angle and perfect joint tracking; finite-difference FK DLS, not Isaac Jacobian validation.',planned_duration_s=t,within_90s=t<=90,old_expert_program_end_s=(end+1)*.02,post_traverse_controller_unchanged='navigation_revision' not in plan,navigation_plan_path=str(plan_path),navigation_revision=plan.get('navigation_revision','r1_forward_only'),closing_circle_candidate=chosen)
