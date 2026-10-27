"""One nominal live expert attempt; no learned policy, quota or success filter.

The frozen opening expert is reused. New controls are solely Joint9 commands.
Optional compact feedback control replaces the legacy post-traversal circle.
The lock captures measured angle only after physical closing has settled.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from fullcycle_geometry import URDF, transform, apply, close_pose
from fullcycle_contract import smooth_gripper, timing, door_contact_review_flags, ObservedHold
from fullcycle_navigation import route_directions, waypoint_command

ROOT=Path(__file__).resolve().parents[3]


def angle_error(a,b):
    return (float(a)-float(b)+math.pi)%(2*math.pi)-math.pi


def serial(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic):
        if x.dtype.names: return {n:serial(x[n]) for n in x.dtype.names}
        return x.item()
    if isinstance(x,dict): return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [serial(v) for v in x]
    return x


class StopAttempt(Exception): pass


def opening_command(m,st,step,dt,dof_names,phase,desired_xy,desired_yaw,
                    target_pos,target_quat,gripper,xy,yaw,initial_gripper,
                    physical_gripper_bounds):
    """Author's per-step base -> refresh -> IK sequence, independent of video."""
    dv=np.asarray(desired_xy)-xy
    lateral=float(np.dot(dv,[-math.sin(yaw),math.cos(yaw)]))
    if abs(lateral)>.001:raise StopAttempt('opening_base_requires_lateral_motion')
    v=float(np.dot(dv,[math.cos(yaw),math.sin(yaw)]))/dt
    w=angle_error(desired_yaw,yaw)/dt
    gym=st._fullcycle_gym;sim=st._fullcycle_sim
    m.set_robot_base_pose(gym,st.env,st.actor_handles,desired_xy,st.args.robot_z,desired_yaw,st.args.robot_pitch)
    gym.refresh_rigid_body_state_tensor(sim);gym.refresh_dof_state_tensor(sim);gym.refresh_jacobian_tensors(sim)
    if phase=='return_home':
        if 'return_home_start_dofs' not in st.traj:st.traj['return_home_start_dofs']=st.dof_positions.copy()
        st.dof_positions[:]=m.lerp(st.traj['return_home_start_dofs'],st.home_positions,float(st.traj.get('return_home_alpha',0)))
    elif phase in ('walk','hold_home'):
        st.dof_positions[:]=st.home_positions
    else:
        m.set_ik_target(st.ik_state,target_pos,target_quat)
        m.update_arm_ik_targets_for_env(gym,st.env,st.arm_actor,0,st.dof_positions,st.ik_state,st.args,len(dof_names))
        i=dof_names.index('jointGripper')
        st.dof_positions[i]=np.clip(gripper,st.ik_state.lower[i].item(),st.ik_state.upper[i].item())
    indices=[dof_names.index('joint%d'%i) for i in range(1,7)]+[dof_names.index('jointGripper')]
    action=np.r_[v,w,st.dof_positions[indices]]
    if step*dt<1.:
        action[-1]=np.clip(smooth_gripper(initial_gripper,st.home_positions[indices[-1]],step*dt),*physical_gripper_bounds)
    return action


def run(m, cams, output, plan, gym, sim, states, viewer, args, dt, dof_names):
    import imageio
    if len(states)!=1 or int(args.steps)*dt>120.000001: raise ValueError('Single env, <=120 seconds only')
    st=states[0];dc=m.dc;ga=m.gymapi
    directions=[] if plan.get('compact_expert') else route_directions(plan)
    robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf', include_visual=True)
    adjacent={frozenset((j[2],j[3])) for j in robot.joints}
    # Contact labels are diagnostic; they do not disable physical collisions
    # or turn a single contact record into an automatic task failure.
    inherited_home_pair=frozenset(('link02','link06'))
    names=['joint%d'%i for i in range(1,7)]+['jointGripper']
    indices=[dof_names.index(n) for n in names]
    home=st.home_positions.copy()
    from fullcycle_scene import SceneFrame
    scene=SceneFrame.from_runtime(m,st)
    from fullcycle_pilot import audit_effective_trial
    audit_effective_trial(gym,st,output)
    (output/'scene-frame.json').write_text(json.dumps(scene.metadata(),indent=2)+'\n')
    from fullcycle_task_monitor import TaskMonitor
    task_monitor=TaskMonitor(home,st.yaw_traverse)
    recorder=None
    if plan.get('live_joint9_recording'):
        from close_inplace_stage5.recording import LiveRecording
        recorder=LiveRecording(m,st,output,dof_names,dt)
    initial_arm=gym.get_actor_dof_states(st.env,st.arm_actor,ga.STATE_ALL)
    initial_gripper=float(initial_arm['pos'][indices[-1]])
    physical_properties=gym.get_actor_dof_properties(st.env,st.arm_actor)
    physical_gripper_lower=float(physical_properties['lower'][indices[-1]])
    physical_gripper_upper=float(physical_properties['upper'][indices[-1]])
    st._fullcycle_gym=gym;st._fullcycle_sim=sim
    selected=plan['selected']
    if plan.get('compact_expert'):
        qclose=home[indices].copy()
        radius=distance=height=None  # Legacy controller is not entered.
    else:
        qclose=np.r_[selected['q'],0.]
        radius,distance,height=[selected[k] for k in ['radius','distance','z']]
    body_map={};robot_ids=set();end_ids=set();door_ids=set();panel_id=None;actor_names={}
    for i in range(gym.get_actor_count(st.env)):
        actor=gym.get_actor_handle(st.env,i)
        actor_name=gym.get_actor_name(st.env,actor)
        actor_names[actor]=actor_name
        for j,b in enumerate(gym.get_actor_rigid_body_names(st.env,actor)):
            idx=gym.get_actor_rigid_body_index(st.env,actor,j,ga.DOMAIN_ENV)
            body_map[idx]=dict(actor=actor_name,actor_handle=int(actor),body=b)
            if actor in st.actor_handles:
                robot_ids.add(idx)
                if b in ['gripperStator','gripperMover']: end_ids.add(idx)
            if actor==st.door_actor:
                door_ids.add(idx)
                if b=='door_panel': panel_id=idx
    if panel_id is None or not end_ids: raise RuntimeError('Missing physical contact body mapping')
    configuration={k:v for k,v in vars(st.args).items() if isinstance(v,(int,float,str,bool,list,tuple,type(None)))}
    (output/'effective-config.json').write_text(json.dumps(serial(configuration),indent=2)+'\n')
    (output/'body-map.json').write_text(json.dumps(body_map,indent=2)+'\n')
    (output/'reference.json').write_text(json.dumps(dict(old_expert_final_yaw=st.yaw_traverse,old_expert_final_base=st.base_traverse.tolist(),home=home.tolist(),dof_names=dof_names,nominal_randomization=json.loads(st.args.ikpush_randomization_json),new_policy_inputs=[],expert_truth=['door angle','robot geometry','contact state'],lock='measured angle latch after 0.5s physical close; no zero snap',hinge_torque_sources={k:getattr(st.args,k) for k in ['door_open_resistance','door_open_damping','door_auto_open_force','door_lock_force','door_joint_friction','door_joint_damping','handle_spring_stiffness','handle_spring_damping']}),indent=2)+'\n')
    log=(output/'states.jsonl').open('x');contact_log=(output/'contacts.jsonl').open('x');lock_log=(output/'lock-writes.jsonl').open('x')
    depth={}  # Raw depth still captured and recorded; no display depth videos.

    phase='old_expert';passed=False;pass_step=None;nav_index=0;stage_start=0
    latched=None;ever_unlocked=False;settled=0;final_count=0;panel_contact_count=0;close_start=None
    panel_nonzero_contact_count=0;off_target_closing_contact_seen=False
    close_hold=ObservedHold();terminal_hold=ObservedHold();latch_observed_hold_s=None;simulated_steps=0
    prev_xy=None;prev_yaw=None;last_action=np.zeros(9);action=np.r_[0.,0.,home[indices]]
    previous_action=action.copy();approach=None;approach_index=0
    compact=None;compact_diagnostics=None;early=None;foot_bad_since=None
    nonzero_door_contact=False;nonzero_end_panel_contact=False
    samples=[];success=False;reason='budget_exhausted';step=-1;initial_record=None;pre_physics=None
    old_phase=''; transition_q=None;route=None;stop_error=None;blocked_self_steps=0
    def event(kind,**data):
        print(json.dumps(dict(event=kind,step=step,time_s=simulated_steps*dt,**serial(data))),flush=True)
    def save_stop_snapshot():
        # Render the CURRENT stopped state; no simulate/reset/replay call.
        target=np.asarray(st.ik_state.current_pos_np,float)
        camera=st._fullcycle_stop_camera
        gym.set_camera_location(camera,st.env,ga.Vec3(*(target+np.array([.8,.65,.35]))),ga.Vec3(*target))
        gym.step_graphics(sim);gym.render_all_camera_sensors(sim)
        im=np.asarray(gym.get_camera_image(sim,st.env,camera,ga.IMAGE_COLOR),np.uint8).reshape(720,1280,4)[:,:,:3]
        imageio.imwrite(str(output/'stop-near-rgb.png'),im)
        snapshot=dict(step=step,last_completed_step=simulated_steps-1,
            measured_post_time_s=simulated_steps*dt,reason=reason,
            pre_physics=pre_physics,door_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.door_actor,ga.STATE_ALL)),
            arm_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.arm_actor,ga.STATE_ALL)),
            base_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.actor_handles[0],ga.STATE_ALL)))
        (output/'stop-state.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    def change(new):
        nonlocal phase,stage_start,transition_q
        phase=new;stage_start=step;transition_q=measured_q.copy();event('phase',phase=phase)
    def locked_hold():
        # Initial lock retains its initial measured angle. The post-close lock
        # retains only the measured sample that passed the physical gate.
        if ever_unlocked and latched is None: raise StopAttempt('forbidden_unlocked_hinge_write')
        d=gym.get_actor_dof_states(st.env,st.door_actor,ga.STATE_ALL)
        before=float(d['pos'][0]);target=initial_angle if latched is None else latched
        lock_log.write(json.dumps(dict(step=step,time_s=step*dt,timing='pre_physics',kind='initial' if latched is None else 'relocked',before=before,latched=target,delta=target-before))+'\n')
        d['pos'][0]=target;d['vel'][0]=0
        gym.set_actor_dof_states(st.env,st.door_actor,d,ga.STATE_ALL)
    initial_d=gym.get_actor_dof_states(st.env,st.door_actor,ga.STATE_ALL)
    initial_angle=float(initial_d['pos'][0])
    initial_rb=gym.get_actor_rigid_body_states(st.env,st.arm_actor,ga.STATE_ALL)
    base_names=list(gym.get_actor_dof_names(st.env,st.actor_handles[0]))
    (output/'base-dof-binding.json').write_text(json.dumps(dict(names=base_names,properties=serial(gym.get_actor_dof_properties(st.env,st.actor_handles[0])),initial=serial(gym.get_actor_dof_states(st.env,st.actor_handles[0],ga.STATE_ALL))),indent=2)+'\n')
    initial_record=dict(door=serial(initial_d),arm=serial(gym.get_actor_dof_states(st.env,st.arm_actor,ga.STATE_ALL)),root=serial(initial_rb[0]),base_start=st.base_start.tolist(),yaw=st.yaw_start)
    (output/'initial-state.json').write_text(json.dumps(initial_record,indent=2)+'\n')
    xy=np.array(st.base_start,dtype=float);yaw=float(st.yaw_start)
    last_cmd_xy=xy.copy();last_cmd_yaw=yaw
    try:
        for step in range(int(args.steps)):
            gym.refresh_rigid_body_state_tensor(sim);gym.refresh_dof_state_tensor(sim);gym.refresh_jacobian_tensors(sim)
            m.current_ee_pose_from_refreshed_tensors(st.ik_state)
            arm=gym.get_actor_dof_states(st.env,st.arm_actor,ga.STATE_ALL)
            measured_q=np.asarray(arm['pos'],float);measured_qd=np.asarray(arm['vel'],float)
            door_pos,door_vel=m.get_actor_dof_state(gym,st.env,st.door_actor)
            observation_pre=dict(time_s=step*dt,state=np.r_[last_action[:2],measured_q[indices]].tolist(),
                arm_qd=measured_qd.tolist(),door_q=np.asarray(door_pos).tolist(),door_qd=np.asarray(door_vel).tolist())
            if phase=='old_expert' and passed and early is not None and plan.get('compact_expert'):
                from random_initial_observe.controller import LiveExpert
                def parking_command(desired_xy,B,q):
                    desired_xy=scene.to_world_base([*desired_xy,B[2]])[:2];B=scene.to_world_base(B)
                    dv=np.asarray(desired_xy)-np.asarray(B[:2])
                    v=float(np.dot(dv,[math.cos(B[2]),math.sin(B[2])]))/dt
                    w=angle_error(st.yaw_traverse,B[2])/dt
                    m.set_robot_base_pose(gym,st.env,st.actor_handles,desired_xy,st.args.robot_z,st.yaw_traverse,st.args.robot_pitch)
                    return np.r_[v,w,early.command(step*dt,q,nonzero_door_contact)]
                compact=LiveExpert(robot,plan['compact_expert'],[*xy,yaw],
                    measured_q[indices],last_action,float(door_pos[0]),step*dt,
                    home[indices],st.yaw_traverse,dt,scene,st.base_traverse,parking_command,early)
                change(compact.phase)
            if phase=='old_expert' and passed and plan.get('smooth_approach'):
                from fullcycle_smooth_approach import build_approach
                recipe=plan['smooth_approach']
                approach=build_approach([*xy,yaw],last_action,previous_action,recipe,qclose,dt,
                                        robot,measured_q[indices],start_step=step)
                np.savez_compressed(output/'approach-plan.npz',
                    **{k:v for k,v in approach.items() if isinstance(v,np.ndarray)})
                (output/'approach-plan.json').write_text(json.dumps(serial(dict(
                    start_step=step,start_time_s=step*dt,start_pose=[*xy,yaw],
                    recipe=recipe,diagnostics={k:v for k,v in approach.items() if not isinstance(v,np.ndarray)},
                    meaning='Generated from the live traversal handoff; Joint9 commands, not state replay',
                )),indent=2)+'\n')
                approach_index=0;change('approach')
            if phase=='old_expert':
                old_phase,desired_xy,desired_yaw,target_pos,target_quat,gripper,handle_goal=m.trajectory_targets(step,st.args,st.door,gym,st.env,st.door_actor,st.ik_state,st.base_start,st.base_stop,st.base_push,st.base_traverse,st.yaw_start,st.yaw_push,st.yaw_traverse,st.traj)
                if old_phase=='traverse_door':
                    if early is None:
                        from random_initial_observe.early_arm import EarlyArm
                        early=EarlyArm(robot,np.r_[xy,yaw],measured_q[indices],last_action[2:],previous_action[2:],step*dt,dt,home[indices],scene)
                        (output/'early-arm-plan.json').write_text(json.dumps(early.plan,indent=2)+'\n')
                        event('parallel_arm_start',plan=early.plan)
                    dv=np.asarray(desired_xy)-xy
                    v=float(np.dot(dv,[math.cos(yaw),math.sin(yaw)]))/dt;w=angle_error(desired_yaw,yaw)/dt
                    m.set_robot_base_pose(gym,st.env,st.actor_handles,desired_xy,st.args.robot_z,desired_yaw,st.args.robot_pitch)
                    action=np.r_[v,w,early.command(step*dt,measured_q[indices],nonzero_door_contact)]
                    target_pos=robot.fk(dict(zip(names,action[2:])),transform([*desired_xy,st.args.robot_z],[0,st.args.robot_pitch,desired_yaw]))['ee_gripper_link'][:3,3]
                    target_quat=None;gripper=action[-1]
                else:
                    action=opening_command(m,st,step,dt,dof_names,old_phase,desired_xy,desired_yaw,
                        target_pos,target_quat,gripper,xy,yaw,initial_gripper,
                        (physical_gripper_lower,physical_gripper_upper))
                st.last_handle_goal=handle_goal;st.last_target_pos=target_pos;st.last_target_quat=target_quat;st.last_gripper=gripper
                if old_phase=='hold_home':
                    if not passed: raise StopAttempt('old_expert_finished_without_full_body_traversal')
                    if np.max(np.abs(measured_q[:6]-home[:6]))<.03:
                        route=plan['route'];change('navigate')
            if compact is None and phase=='approach':
                if approach_index<len(approach['actions']):
                    action=approach['actions'][approach_index].copy();approach_index+=1
                else:
                    # Resume the unchanged R5 deploy formula at its original
                    # contact-entry command, then keep the existing suffix.
                    action=np.r_[0.,0.,approach['actions'][-1,2:]]
                    change('deploy')
                    stage_start=step-int(plan['smooth_approach']['deploy_entry_offset_steps'])
                    transition_q=measured_q.copy()
                    transition_q[indices]=np.asarray(plan['smooth_approach']['deploy_origin_q'],float)
                    event('deploy_tail_resume',original_deploy_elapsed_s=(step-stage_start)*dt)
            if compact is not None:
                feedback_root=gym.get_actor_rigid_body_states(st.env,st.arm_actor,ga.STATE_ALL)[0]['pose']
                feedback_B=np.array([feedback_root['p']['x'],feedback_root['p']['y'],m.yaw_from_quat_xyzw(np.array([feedback_root['r'][k] for k in ['x','y','z','w']]))])
                action,compact_diagnostics=compact.command(step*dt,feedback_B,
                    measured_q[indices],float(door_pos[0]),float(door_vel[0]),
                    nonzero_end_panel_contact,nonzero_door_contact,latched is not None)
                if phase!=compact.phase:
                    change(compact.phase)
                    if compact.route is not None and not (output/'compact-route.npy').exists():
                        np.save(output/'compact-route.npy',np.array([scene.to_world_base(p) for p in compact.route]))
                if latched is None and compact_diagnostics.get('reference_complete',False) and task_monitor.may_latch():
                    if panel_nonzero_contact_count==0:
                        raise StopAttempt('missing_end_effector_physical_closing_evidence')
                    latch_observed_hold_s=task_monitor.close_hold.elapsed
                    latched=float(door_pos[0]);st.door.open_stage=False
                    event('latch',measured_angle=latched,settled_steps=settled,
                          observed_hold_s=latch_observed_hold_s)
            if compact is None and phase not in ('old_expert','approach') and step%2==0:
                v=w=0.;qtarget=home[indices].copy()
                elapsed=(step-stage_start)*dt
                if phase=='navigate':
                    if nav_index>=len(route): change('deploy')
                    else:
                        v,w,reached=waypoint_command(xy,yaw,route[nav_index],directions[nav_index])
                        if reached:
                            nav_index+=1;event('navigation_waypoint',index=nav_index)
                elif phase=='deploy':
                    alpha=float(np.clip(elapsed/4,0,1));s=alpha**3*(10-15*alpha+6*alpha**2)
                    qtarget=(1-s)*transition_q[indices]+s*qclose
                    if elapsed>=5 and np.max(np.abs(measured_q[indices]-qclose))<.04:
                        close_start=float(door_pos[0]);change('close')
                elif phase=='close':
                    qtarget=qclose
                    # Forward speed and yaw-rate share circle parameter.
                    rate=.075
                    if door_pos[0]<math.radians(4): rate=.02
                    B,E=close_pose(max(0,float(door_pos[0])),radius,distance,height)
                    planned_angle=float(st.traj.get('close_angle',math.pi/2))
                    if planned_angle-float(door_pos[0])<-.12:
                        rate=0 # retain contact while actual door catches up
                    if planned_angle>-.018:
                        v=rate*math.hypot(radius,distance);w=-rate
                        st.traj['close_angle']=planned_angle-rate*.04
                    if abs(float(door_pos[0]))<=math.radians(1) and abs(float(door_vel[0]))<=math.radians(1):
                        v=w=0
                    if close_hold.reached(.5):
                        if panel_nonzero_contact_count==0:
                            raise StopAttempt('missing_end_effector_physical_closing_evidence')
                        latch_observed_hold_s=close_hold.elapsed
                        latched=float(door_pos[0]);st.door.open_stage=False;event('latch',measured_angle=latched,settled_steps=settled,observed_hold_s=latch_observed_hold_s)
                        change('release')
                elif phase=='release':
                    # Reverse along same circle for 0.1m to break contact,
                    # keeping the closing posture fixed before joint retraction.
                    qtarget=qclose
                    if elapsed<1.5: v=-.07;w=.07/math.hypot(radius,distance)
                    else: change('retract')
                elif phase=='retract':
                    alpha=float(np.clip(elapsed/4,0,1));s=alpha**3*(10-15*alpha+6*alpha**2)
                    qtarget=(1-s)*transition_q[indices]+s*home[indices]
                    if elapsed>=5: change('restore_heading')
                elif phase=='restore_heading':
                    err=angle_error(st.yaw_traverse,yaw)
                    w=float(np.clip(2*err,-.6,.6)) if abs(err)>.003 else 0
                    if abs(err)<.003: change('final_hold')
                elif phase=='final_hold': pass
                action=np.r_[v,w,qtarget]
            if phase!='old_expert':
                # New stages command joints; do not mislabel measured EE as
                # its own target. Record the FK of the commanded joints.
                command_fk=robot.fk(dict(zip(names,action[2:])),transform([*xy,st.args.robot_z],[0,st.args.robot_pitch,yaw]))['ee_gripper_link']
                st.last_target_pos=command_fk[:3,3].copy()
                st.last_target_quat=None;st.last_gripper=float(action[-1])
            if not np.isfinite(action).all(): raise StopAttempt('nonfinite_command')
            # Exactly the existing Joint9 base integrator at each physics step.
            xy,yaw,jtargets=m.apply_float_dp_joint_action9(action,xy,yaw,dt)
            m.set_a2w_joint_targets_from_action(st.dof_positions,dof_names,jtargets,st.ik_state.lower,st.ik_state.upper)
            if phase=='old_expert' and step*dt<1.:
                st.dof_positions[indices[-1]]=action[-1]
            if phase!='old_expert':
                m.set_robot_base_pose(gym,st.env,st.actor_handles,xy,st.args.robot_z,yaw,st.args.robot_pitch)
            gym.set_actor_dof_position_targets(st.env,st.arm_actor,st.dof_positions)
            st.traj['base_xy']=xy.copy();st.traj['yaw']=float(yaw)
            st.last_phase=old_phase if phase=='old_expert' else phase
            if (not ever_unlocked and not st.door.open_stage) or latched is not None: locked_hold()
            door_pos,door_vel=m.get_actor_dof_state(gym,st.env,st.door_actor)
            efforts=m.compute_door_efforts(st.door,door_pos,door_vel,st.args)
            if latched is not None:
                st.door.open_stage=False;efforts[0]=0
            if st.door.open_stage: ever_unlocked=True
            if ever_unlocked and latched is None and abs(float(efforts[0]))>0:
                raise StopAttempt('nonzero_auxiliary_hinge_effort')
            gym.apply_actor_dof_efforts(st.env,st.door_actor,efforts)
            pre_physics=dict(clock=timing(step,dt),door_q=np.asarray(door_pos).tolist(),door_qd=np.asarray(door_vel).tolist(),
                arm_dof=serial(gym.get_actor_dof_states(st.env,st.arm_actor,ga.STATE_ALL)),
                arm_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.arm_actor,ga.STATE_ALL)),
                base_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.actor_handles[0],ga.STATE_ALL)),
                door_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.door_actor,ga.STATE_ALL)),
                command=action.tolist())
            gym.simulate(sim);gym.fetch_results(sim,True)
            simulated_steps=step+1
            gym.refresh_rigid_body_state_tensor(sim);gym.refresh_dof_state_tensor(sim);gym.refresh_jacobian_tensors(sim)
            m.current_ee_pose_from_refreshed_tensors(st.ik_state)
            arm=gym.get_actor_dof_states(st.env,st.arm_actor,ga.STATE_ALL)
            measured_q=np.asarray(arm['pos'],float);measured_qd=np.asarray(arm['vel'],float)
            door_pos,door_vel=m.get_actor_dof_state(gym,st.env,st.door_actor)
            rb=gym.get_actor_rigid_body_states(st.env,st.arm_actor,ga.STATE_ALL)
            root=rb[0]['pose'];actual_xy=np.array([root['p']['x'],root['p']['y']]);quat=np.array([root['r'][k] for k in ['x','y','z','w']]);actual_yaw=m.yaw_from_quat_xyzw(quat)
            # FK uses measured joints and the actual root; all link collision
            # AND visible mesh bounds, not just chassis center or EE target.
            legs={}
            base_actor=st.actor_handles[0]
            base_joints=gym.get_actor_dof_states(st.env,base_actor,ga.STATE_ALL)
            legs.update(dict(zip(gym.get_actor_dof_names(st.env,base_actor),base_joints['pos'])))
            from scipy.spatial.transform import Rotation
            actual_root=np.eye(4);actual_root[:3,:3]=Rotation.from_quat(quat).as_matrix()
            actual_root[:3,3]=[root['p'][k] for k in ['x','y','z']]
            poses=robot.fk(dict(legs,**dict(zip(dof_names,measured_q))),actual_root)
            cloud=robot.cloud(poses);max_x=float(cloud[:,0].max());far=bool(max_x<scene.pass_plane_x)
            if not passed and far:
                passed=True;pass_step=step;event('whole_body_pass',max_x=max_x)
            if not np.isfinite(np.r_[measured_q,measured_qd,door_pos,door_vel,actual_xy,actual_yaw]).all():
                raise StopAttempt('nonfinite_measured_state')
            contacts=[];end_panel=False;any_door_contact=False;unexpected_loaded_self=False
            nonzero_door_contact=False;nonzero_end_panel_contact=False;off_target_closing_contact=False
            for c in gym.get_env_rigid_contacts(st.env):
                fields=c.dtype.names if hasattr(c,'dtype') and c.dtype.names else ['body0','body1','lambda','normal','localPos0','localPos1','initial_overlap','min_dist']
                entry={k:serial(dc.contact_field(c,k)) for k in fields}
                b0=int(dc.contact_field(c,'body0'));b1=int(dc.contact_field(c,'body1'))
                entry['names']=[body_map.get(b0,{'body':'ground_or_unknown'}),body_map.get(b1,{'body':'ground_or_unknown'})]
                contacts.append(entry)
                lam=float(entry.get('lambda',0))
                if not np.isfinite(lam): raise StopAttempt('nonfinite_contact_record')
                entry['classification']='unexplained_record_only'
                robot_body=b0 if b0 in robot_ids else b1 if b1 in robot_ids else None
                other=b1 if robot_body==b0 else b0
                if b0 in robot_ids and b1 in robot_ids:
                    pair=frozenset((body_map[b0]['body'],body_map[b1]['body']))
                    if pair==inherited_home_pair: entry['classification']='inherited_home_self_contact'
                    elif pair in adjacent: entry['classification']='kinematic_neighbor_self_contact'
                    else:
                        entry['classification']='other_self_contact_loaded' if lam!=0 else 'other_self_contact_zero_lambda'
                        unexpected_loaded_self|=lam!=0
                if robot_body is not None and other>=0 and other not in robot_ids:
                    any_door_contact=any_door_contact or other in door_ids
                    if robot_body in end_ids and other==panel_id: end_panel=True
                    expected=(robot_body in end_ids and other in door_ids and body_map[other]['body'] in ['door_panel','lever_handle'])
                    entry['classification']='expected_end_door_contact' if expected else ('unexpected_external_loaded' if lam!=0 else 'external_zero_lambda_record_only')
                    flags=door_contact_review_flags('close' if phase=='approach' else phase,lam,other in door_ids,robot_body in end_ids and other==panel_id)
                    entry['review_flags']=flags
                    nonzero_door_contact|=flags['nonzero_door_contact']
                    nonzero_end_panel_contact|=flags['nonzero_end_panel_contact']
                    off_target_closing_contact|=flags['off_target_closing_contact']
            off_target_closing_contact_seen|=off_target_closing_contact
            # Retain the historical stall heuristic as a trace only. It is
            # neither an author-baseline abort rule nor a success criterion.
            blocked_self=bool(unexpected_loaded_self and np.max(np.abs(measured_q[:6]-st.dof_positions[:6]))>.3 and np.max(np.abs(measured_qd[:6]))<.01)
            blocked_self_steps=blocked_self_steps+1 if blocked_self else 0
            if phase in ('approach','close') and end_panel: panel_contact_count+=1
            if phase in ('approach','close') and nonzero_end_panel_contact: panel_nonzero_contact_count+=1
            closed=abs(float(door_pos[0]))<=math.radians(1) and abs(float(door_vel[0]))<=math.radians(1)
            settled=settled+1 if phase=='close' and closed else 0
            close_hold.observe((step+1)*dt,phase=='close' and closed)
            velocity=float(np.linalg.norm(actual_xy-prev_xy)/dt) if prev_xy is not None else 0
            omega=abs(angle_error(actual_yaw,prev_yaw))/dt if prev_yaw is not None else 0
            task_monitor.observe((step+1)*dt,far,float(door_pos[0]),float(door_vel[0]),
                bool(st.door.open_stage),latched,any_door_contact,nonzero_end_panel_contact,
                measured_q,measured_qd,velocity,omega,actual_yaw)
            # Zero-lambda proximity reports alone do not reset the hold
            # window. Actual release/pushing still requires video/state review.
            final_ok=bool(phase=='final_hold' and far and not nonzero_door_contact and closed and latched is not None and not st.door.open_stage and abs(angle_error(actual_yaw,st.yaw_traverse))<=math.radians(2) and velocity<=.01 and omega<=.01 and np.all(np.abs(measured_q[:6]-home[:6])<=.03) and np.all(np.abs(measured_qd[:6])<=.05))
            final_count=final_count+1 if final_ok else 0
            terminal_hold.observe((step+1)*dt,final_ok)
            row=dict(step=step,t=(step+1)*dt,phase=st.last_phase,state=np.r_[last_action[:2],measured_q[indices]],action=action.copy(),command_base=[*xy,yaw],actual_base=[*actual_xy,actual_yaw],base_measured_fd_speed=velocity,base_measured_fd_yaw_rate=omega,arm_q=measured_q.copy(),arm_qd=measured_qd.copy(),arm_command=st.dof_positions.copy(),ee_actual=st.ik_state.current_pos_np.copy(),ee_quat=st.ik_state.current_quat_np.copy(),ee_target=st.last_target_pos,ee_target_quat=st.last_target_quat,door_q=door_pos,door_qd=door_vel,door_effort=efforts,unlocked=bool(st.door.open_stage),latched=latched,whole_geometry_max_x=max_x,whole_geometry_bounds=[cloud.min(0),cloud.max(0)],passed=passed,end_panel_contact=end_panel,any_door_contact=any_door_contact,physical_close_steps=settled,final_hold_steps=final_count,root_rb=serial(rb[0]),arm_body_states=serial(rb),base_body_states=serial(gym.get_actor_rigid_body_states(st.env,base_actor,ga.STATE_ALL)))
            row['base_dof_names']=base_names;row['base_dof_actual']=serial(base_joints)
            deviations={n:abs(float(base_joints['pos'][i])-({'hip':0.,'thigh':.8,'calf':-1.5}[next(k for k in ['hip','thigh','calf'] if k in n)])) for i,n in enumerate(base_names) if any(k in n for k in ['hip','thigh','calf'])}
            bad=[n for n,v in deviations.items() if v>(.2 if 'hip' in n else .3)]
            foot_bad_since=((step+1)*dt if foot_bad_since is None else foot_bad_since) if bad else None
            row['foot_guard']=dict(deviations=deviations,bad=bad,since_s=foot_bad_since,hold_s=.10)
            row['timing']=timing(step,dt);row['pre_physics']=pre_physics
            row['door_body_states']=serial(gym.get_actor_rigid_body_states(st.env,st.door_actor,ga.STATE_ALL))
            row['observation_pre']=observation_pre
            row['blocked_unexpected_self_steps']=blocked_self_steps
            row['nonzero_door_contact']=nonzero_door_contact
            row['nonzero_end_panel_contact']=nonzero_end_panel_contact
            row['off_target_closing_contact']=off_target_closing_contact
            row['physical_close_observed_span_s']=close_hold.elapsed
            row['final_hold_observed_span_s']=terminal_hold.elapsed
            row['approach_command_index']=approach_index-1 if phase=='approach' else None
            row['task_monitor']=task_monitor.metadata();row['pass_plane_x']=scene.pass_plane_x
            if early is not None:row['early_arm']=early.latest.copy()
            if compact is not None:
                row['compact_expert']=compact_diagnostics
            log.write(json.dumps(serial(row))+'\n');contact_log.write(json.dumps(dict(step=step,interval_s=[step*dt,(step+1)*dt],contacts=contacts))+'\n')
            samples.append([step,(step+1)*dt,*np.r_[last_action[:2],measured_q[indices]],*action, *actual_xy,actual_yaw,*door_pos,*door_vel])
            images=None
            if step%2==0:
                # install_live hooks provide synchronized original RGB.
                m.update_door_twin_tracker(st,step,door_pos)
                images=dc.capture_dp_camera_images_from_rendered(gym,sim,st.env,st.camera_handles,st.args)
                for view in depth:
                    im=images.get(view+'_masked_depth')
                    if im is None: raise StopAttempt('missing_depth_observation')
                    depth[view].append_data(im)
            if recorder is not None:
                recorder.observe(gym,sim,step,dof_names,indices[-1],door_pos,door_vel,images)
            prev_xy=actual_xy.copy();prev_yaw=actual_yaw;previous_action=last_action.copy();last_action=action.copy()
            if step%50==0: event('progress',phase=st.last_phase,door_deg=float(np.rad2deg(door_pos[0])),base=[*actual_xy,actual_yaw],geometry_max_x=max_x)
            if foot_bad_since is not None and (step+1)*dt-foot_bad_since>=.10-1e-9:raise StopAttempt('sustained_leg_posture_deviation:'+','.join(bad))
            if passed and not far: raise StopAttempt('return_crossing_after_whole_body_pass')
            if phase!='old_expert' and not passed: raise StopAttempt('close_sequence_before_full_body_pass')
            if terminal_hold.reached(2.) and task_monitor.accepted():
                success=True;reason='final_conditions_met_pending_physical_review';break
    except StopAttempt as e:
        reason=str(e);event('stop',reason=reason)
        try:save_stop_snapshot()
        except Exception as snapshot_error:
            stop_error='stop snapshot missing: '+repr(snapshot_error);reason+=';evidence_missing';event('error',reason=stop_error)
    except BaseException as e:
        reason='infrastructure_or_recording_error:'+repr(e);stop_error=traceback.format_exc();event('error',reason=reason);traceback.print_exc()
    finally:
        if (not success or recorder is not None) and step>=0 and not (output/'stop-state.json').exists():
            try:save_stop_snapshot()
            except Exception as snapshot_error:
                stop_error='stop snapshot missing: '+repr(snapshot_error);reason+=';evidence_missing'
        log.close();contact_log.close();lock_log.close()
        for writer in depth.values(): writer.close()
        if recorder is not None:recorder.close(success)
        for camera in cams.values():
            camera.expected=int(math.ceil(simulated_steps/2));camera.close()
        np.savez_compressed(output/'trajectory.npz',rows=np.asarray(samples),columns=np.array(['step','time']+['state_'+n for n in names[:0]]+['state_last_vx','state_last_yaw']+['state_'+n for n in names]+['action_vx','action_yaw']+['action_'+n for n in names]+['actual_x','actual_y','actual_yaw','door_angle','handle_angle','door_velocity','handle_velocity']))
        result=dict(success=success,reason=reason,steps=simulated_steps,attempted_step=step,
            sim_seconds=simulated_steps*dt,whole_body_pass_step=pass_step,end_panel_contact_steps=panel_contact_count,latched_angle=latched,final_hold_steps=final_count,attempts=1,error=stop_error,nominal_only=False,physical_push_review='requires aligned raw contact/state, live video and code audit; no coverage/work thresholds',video_frames={k:v.count for k,v in cams.items()})
        result.update(
            contact_counter_scope='postdeployment approach and close; loaded actual contact, not phase completion',
            automatic_final_conditions_met=success,
            independent_task_acceptance='pending' if success else 'not_reached',
            end_panel_nonzero_contact_steps=panel_nonzero_contact_count,
            off_target_closing_contact_seen=off_target_closing_contact_seen,
            success_field_meaning='Automatic sequence/terminal checks only; physical pushing and release require independent review.',
            latch_observed_hold_s=latch_observed_hold_s,
            final_observed_hold_s=terminal_hold.elapsed,
            independent_task_monitor=task_monitor.metadata(),
        )
        (output/'result.json').write_text(json.dumps(serial(result),indent=2)+'\n');print(json.dumps(result),flush=True)

