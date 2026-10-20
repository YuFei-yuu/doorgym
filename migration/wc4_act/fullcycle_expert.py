"""One nominal live expert attempt; no learned policy, quota or success filter.

The frozen opening expert is reused. New controls are solely Joint9 commands.
Post-traversal arc follows a nonholonomic circle with a fixed six-axis arm pose.
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

ROOT=Path(__file__).resolve().parents[2]


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


def run(m, cams, output, plan, gym, sim, states, viewer, args, dt, dof_names):
    import imageio
    if len(states)!=1 or int(args.steps)>4500: raise ValueError('Single env, <=90 seconds only')
    st=states[0];dc=m.dc;ga=m.gymapi
    robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf', include_visual=True)
    names=['joint%d'%i for i in range(1,7)]+['jointGripper']
    indices=[dof_names.index(n) for n in names]
    home=st.home_positions.copy()
    selected=plan['selected'];qclose=np.r_[selected['q'],0.]
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
    depth={v:imageio.get_writer(str(output/(v+'_depth.mp4')),fps=25,codec='libx264',quality=7,macro_block_size=1,ffmpeg_params=['-threads','2']) for v in ['front','wrist']}
    phase='old_expert';passed=False;pass_step=None;nav_index=0;stage_start=0
    latched=None;ever_unlocked=False;settled=0;final_count=0;panel_contact_count=0;close_start=None
    prev_xy=None;prev_yaw=None;last_action=np.zeros(9);action=np.r_[0.,0.,home[indices]]
    samples=[];success=False;reason='budget_exhausted';step=-1;initial_record=None
    old_phase=''; transition_q=None;route=None;stop_error=None
    def event(kind,**data):
        print(json.dumps(dict(event=kind,step=step,time_s=(step+1)*dt,**serial(data))),flush=True)
    def change(new):
        nonlocal phase,stage_start,transition_q
        phase=new;stage_start=step;transition_q=measured_q.copy();event('phase',phase=phase)
    def locked_hold():
        # Initial lock retains its initial measured angle. The post-close lock
        # retains only the measured sample that passed the physical gate.
        d=gym.get_actor_dof_states(st.env,st.door_actor,ga.STATE_ALL)
        before=float(d['pos'][0]);target=initial_angle if latched is None else latched
        lock_log.write(json.dumps(dict(step=step,kind='initial' if latched is None else 'relocked',before=before,latched=target,delta=target-before))+'\n')
        d['pos'][0]=target;d['vel'][0]=0
        gym.set_actor_dof_states(st.env,st.door_actor,d,ga.STATE_ALL)
    initial_d=gym.get_actor_dof_states(st.env,st.door_actor,ga.STATE_ALL)
    initial_angle=float(initial_d['pos'][0])
    initial_rb=gym.get_actor_rigid_body_states(st.env,st.arm_actor,ga.STATE_ALL)
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
            if phase=='old_expert':
                old_phase,desired_xy,desired_yaw,target_pos,target_quat,gripper,handle_goal=m.trajectory_targets(step,st.args,st.door,gym,st.env,st.door_actor,st.ik_state,st.base_start,st.base_stop,st.base_push,st.base_traverse,st.yaw_start,st.yaw_push,st.yaw_traverse,st.traj)
                # The original opening has a straight, constant-heading base
                # trajectory. Sample at 25 Hz then hold each Joint9 action.
                if step%2==0:
                    ahead=min(step+1,int(args.steps)-1)
                    # At straight-motion boundaries use current desired base,
                    # avoiding any nonholonomic lateral interpolation.
                    dv=np.asarray(desired_xy)-xy
                    lateral=float(np.dot(dv,[-math.sin(yaw),math.cos(yaw)]))
                    if abs(lateral)>.001: raise StopAttempt('opening_base_requires_lateral_motion')
                    v=float(np.dot(dv,[math.cos(yaw),math.sin(yaw)]))/dt
                    w=angle_error(desired_yaw,yaw)/dt
                    # Existing straight expert v is piecewise 0/.2. Holding v
                    # for 2 steps follows its original 50 Hz base exactly.
                    m.set_robot_base_pose(gym,st.env,st.actor_handles,desired_xy,st.args.robot_z,desired_yaw,st.args.robot_pitch)
                    gym.refresh_rigid_body_state_tensor(sim);gym.refresh_dof_state_tensor(sim);gym.refresh_jacobian_tensors(sim)
                    if old_phase=='return_home':
                        if 'return_home_start_dofs' not in st.traj: st.traj['return_home_start_dofs']=st.dof_positions.copy()
                        st.dof_positions[:]=m.lerp(st.traj['return_home_start_dofs'],home,float(st.traj.get('return_home_alpha',0)))
                    elif old_phase in ('walk','hold_home'): st.dof_positions[:]=home
                    else:
                        m.set_ik_target(st.ik_state,target_pos,target_quat)
                        m.update_arm_ik_targets_for_env(gym,st.env,st.arm_actor,0,st.dof_positions,st.ik_state,st.args,len(dof_names))
                        st.dof_positions[dof_names.index('jointGripper')]=np.clip(gripper,st.ik_state.lower[dof_names.index('jointGripper')].item(),st.ik_state.upper[dof_names.index('jointGripper')].item())
                    action=np.r_[v,w,st.dof_positions[indices]]
                st.last_handle_goal=handle_goal;st.last_target_pos=target_pos;st.last_target_quat=target_quat;st.last_gripper=gripper
                if old_phase=='hold_home':
                    if not passed: raise StopAttempt('old_expert_finished_without_full_body_traversal')
                    if np.max(np.abs(measured_q[:6]-home[:6]))<.03:
                        route=plan['route'];change('navigate')
            if phase!='old_expert' and step%2==0:
                v=w=0.;qtarget=home[indices].copy()
                elapsed=(step-stage_start)*dt
                if phase=='navigate':
                    if nav_index>=len(route): change('deploy')
                    else:
                        goal=np.array(route[nav_index]);error=goal[:2]-xy;dist=float(np.linalg.norm(error));yawerr=angle_error(goal[2],yaw)
                        if dist<.003 and abs(yawerr)<.004:
                            nav_index+=1;event('navigation_waypoint',index=nav_index)
                        elif dist>=.003:
                            desired_heading=math.atan2(error[1],error[0]);headingerr=angle_error(desired_heading,yaw)
                            # Each route leg is rotate-then-drive; no sideslip.
                            if abs(headingerr)>.015: w=float(np.clip(2*headingerr,-.6,.6))
                            else: v=min(.25,dist/.04);w=float(np.clip(headingerr,-.1,.1))
                        else: w=float(np.clip(2*yawerr,-.6,.6))
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
                    if settled>=25:
                        if panel_contact_count==0 or close_start is None or close_start-float(door_pos[0])<.3:
                            raise StopAttempt('missing_end_effector_physical_closing_evidence')
                        latched=float(door_pos[0]);st.door.open_stage=False;event('latch',measured_angle=latched,settled_steps=settled)
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
                st.last_target_pos=st.ik_state.current_pos_np.copy();st.last_target_quat=st.ik_state.current_quat_np.copy();st.last_gripper=float(qtarget[-1])
            if not np.isfinite(action).all(): raise StopAttempt('nonfinite_command')
            # Exactly the existing Joint9 base integrator at each physics step.
            xy,yaw,jtargets=m.apply_float_dp_joint_action9(action,xy,yaw,dt)
            m.set_a2w_joint_targets_from_action(st.dof_positions,dof_names,jtargets,st.ik_state.lower,st.ik_state.upper)
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
            gym.apply_actor_dof_efforts(st.env,st.door_actor,efforts)
            gym.simulate(sim);gym.fetch_results(sim,True)
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
            poses=robot.fk(dict(legs,**dict(zip(dof_names,measured_q))),transform([*actual_xy,st.args.robot_z],[0,st.args.robot_pitch,actual_yaw]))
            cloud=robot.cloud(poses);max_x=float(cloud[:,0].max());far=bool(max_x<2.5)
            if not passed and far:
                passed=True;pass_step=step;event('whole_body_pass',max_x=max_x)
            violations=[];contacts=[];end_panel=False;any_door_contact=False
            for c in gym.get_env_rigid_contacts(st.env):
                fields=c.dtype.names if hasattr(c,'dtype') and c.dtype.names else ['body0','body1','lambda','normal','localPos0','localPos1','initial_overlap','min_dist']
                entry={k:serial(dc.contact_field(c,k)) for k in fields}
                b0=int(dc.contact_field(c,'body0'));b1=int(dc.contact_field(c,'body1'))
                entry['names']=[body_map.get(b0,{'body':'ground_or_unknown'}),body_map.get(b1,{'body':'ground_or_unknown'})]
                contacts.append(entry)
                robot_body=b0 if b0 in robot_ids else b1 if b1 in robot_ids else None
                other=b1 if robot_body==b0 else b0
                if robot_body is not None and other>=0 and other not in robot_ids:
                    any_door_contact=any_door_contact or other in door_ids
                    if robot_body in end_ids and other==panel_id: end_panel=True
                    # Only gripper contact with the panel/lever is expected.
                    if not(robot_body in end_ids and other in door_ids and body_map[other]['body'] in ['door_panel','lever_handle']):
                        violations.append(entry)
            if phase=='close' and end_panel: panel_contact_count+=1
            closed=abs(float(door_pos[0]))<=math.radians(1) and abs(float(door_vel[0]))<=math.radians(1)
            settled=settled+1 if phase=='close' and closed else 0
            velocity=float(np.linalg.norm(actual_xy-prev_xy)/dt) if prev_xy is not None else 0
            omega=abs(angle_error(actual_yaw,prev_yaw))/dt if prev_yaw is not None else 0
            final_ok=bool(phase=='final_hold' and far and not any_door_contact and closed and latched is not None and not st.door.open_stage and abs(angle_error(actual_yaw,st.yaw_traverse))<=math.radians(2) and velocity<=.01 and omega<=.01 and np.all(np.abs(measured_q[:6]-home[:6])<=.03) and np.all(np.abs(measured_qd[:6])<=.05))
            final_count=final_count+1 if final_ok else 0
            row=dict(step=step,t=(step+1)*dt,phase=st.last_phase,state=np.r_[last_action[:2],measured_q[indices]],action=action.copy(),command_base=[*xy,yaw],actual_base=[*actual_xy,actual_yaw],base_measured_fd_speed=velocity,base_measured_fd_yaw_rate=omega,arm_q=measured_q.copy(),arm_qd=measured_qd.copy(),arm_command=st.dof_positions.copy(),ee_actual=st.ik_state.current_pos_np.copy(),ee_quat=st.ik_state.current_quat_np.copy(),ee_target=st.last_target_pos,ee_target_quat=st.last_target_quat,door_q=door_pos,door_qd=door_vel,door_effort=efforts,unlocked=bool(st.door.open_stage),latched=latched,whole_geometry_max_x=max_x,whole_geometry_bounds=[cloud.min(0),cloud.max(0)],passed=passed,end_panel_contact=end_panel,any_door_contact=any_door_contact,physical_close_steps=settled,final_hold_steps=final_count,root_rb=serial(rb[0]),arm_body_states=serial(rb),base_body_states=serial(gym.get_actor_rigid_body_states(st.env,base_actor,ga.STATE_ALL)))
            log.write(json.dumps(serial(row))+'\n');contact_log.write(json.dumps(dict(step=step,contacts=contacts))+'\n')
            samples.append([step,(step+1)*dt,*np.r_[last_action[:2],measured_q[indices]],*action, *actual_xy,actual_yaw,*door_pos,*door_vel])
            if step%2==0:
                # install_live hooks provide synchronized original RGB.
                m.update_door_twin_tracker(st,step,door_pos)
                images=dc.capture_dp_camera_images_from_rendered(gym,sim,st.env,st.camera_handles,st.args)
                for view in depth:
                    im=images.get(view+'_masked_depth')
                    if im is None: raise StopAttempt('missing_depth_observation')
                    depth[view].append_data(im)
            prev_xy=actual_xy.copy();prev_yaw=actual_yaw;last_action=action.copy()
            if step%50==0: event('progress',phase=st.last_phase,door_deg=float(np.rad2deg(door_pos[0])),base=[*actual_xy,actual_yaw],geometry_max_x=max_x)
            if passed and not far: raise StopAttempt('return_crossing_after_whole_body_pass')
            if violations: raise StopAttempt('unexpected_robot_environment_contact:'+str(violations[0]['names']))
            if phase!='old_expert' and not passed: raise StopAttempt('close_sequence_before_full_body_pass')
            if final_count>=100:
                success=True;reason='complete_nominal_task';break
    except StopAttempt as e:
        reason=str(e);event('stop',reason=reason)
    except BaseException as e:
        reason='infrastructure_or_recording_error:'+repr(e);stop_error=traceback.format_exc();event('error',reason=reason);traceback.print_exc()
    finally:
        log.close();contact_log.close();lock_log.close()
        for writer in depth.values(): writer.close()
        for camera in cams.values():
            camera.expected=int(math.ceil((step+1)/2));camera.close()
        np.savez_compressed(output/'trajectory.npz',rows=np.asarray(samples),columns=np.array(['step','time']+['state_'+n for n in names[:0]]+['state_last_vx','state_last_yaw']+['state_'+n for n in names]+['action_vx','action_yaw']+['action_'+n for n in names]+['actual_x','actual_y','actual_yaw','door_angle','handle_angle','door_velocity','handle_velocity']))
        result=dict(success=success,reason=reason,steps=step+1,sim_seconds=(step+1)*dt,whole_body_pass_step=pass_step,end_panel_contact_steps=panel_contact_count,latched_angle=latched,final_hold_steps=final_count,attempts=1,error=stop_error,nominal_only=True,video_frames={k:v.count for k,v in cams.items()})
        (output/'result.json').write_text(json.dumps(serial(result),indent=2)+'\n');print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--geometry',required=True);p.add_argument('--preflight',action='store_true')
    a,remaining=p.parse_known_args()
    output=Path(a.output);output.mkdir(exist_ok=False)
    plan=json.loads(Path(a.geometry).read_text())
    if plan['status']!='candidate_found' or plan['selected'] is None: raise RuntimeError('Offline geometry gate not passed')
    sys.path.insert(0,str(ROOT/'high-level/float_ik'))
    from overview import load,install_live
    m=load('fullcycle_author','high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
    cams=install_live(m,output/'overview')
    original_parse=m.parse_args
    def parse():
        args=original_parse()
        # Camera construction only; custom runner never calls OpenCV.
        args.show_camera_images=True
        if args.dp_policy_checkpoint or args.record_dp_dataset: raise ValueError('No policy or success-filtered recorder allowed')
        return args
    m.parse_args=parse
    m.run_parallel_demo=lambda *args:run(m,cams,output,plan,*args)
    sys.argv=[str(ROOT/'high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py'),*remaining]
    (output/'command.json').write_text(json.dumps(dict(argv=sys.argv,outer_argv=[a.output,a.geometry],cwd=str(ROOT),gpu={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','VIDEO2DOOR_GPU','VK_INSTANCE_LAYERS','VK_LAYER_PATH']},start_time=time.time()),indent=2)+'\n')
    if a.preflight:
        args=m.parse_args()
        configured={k:v for k,v in vars(args).items() if isinstance(v,(int,float,str,bool,list,tuple,type(None)))}
        (output/'parsed-config.json').write_text(json.dumps(serial(configured),indent=2)+'\n')
        print('Argument preflight passed; no simulator created.')
        return
    m.main()


if __name__=='__main__': main()
