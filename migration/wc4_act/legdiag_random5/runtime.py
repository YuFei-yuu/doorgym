"""Accepted expert/physics loop; independent simplified success scoring.
Action, latch authorization and execution guards remain from cbcf6b9.
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
from author_fast5.trajectory import CompleteExpert, Measured
from author_fast5.adapter import apply_target, authorize_latch
from scipy.spatial.transform import Rotation
from success_random5.criterion import SuccessMonitor
from legdiag_random5.diagnostic import LegDiagnostic

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



def run(m, cams, output, plan, gym, sim, states, viewer, args, dt, dof_names):
    import imageio
    if len(states)!=1 or int(args.steps)*dt>120.000001: raise ValueError('Single env, <=120 seconds only')
    st=states[0];dc=m.dc;ga=m.gymapi
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
    score=SuccessMonitor();sequence_complete=False
    recorder=None
    if plan.get('live_joint9_recording'):
        from author_fast5.recording import LiveRecording
        recorder=LiveRecording(m,st,output,dof_names,dt)
    initial_arm=gym.get_actor_dof_states(st.env,st.arm_actor,ga.STATE_ALL)
    initial_gripper=float(initial_arm['pos'][indices[-1]])
    physical_properties=gym.get_actor_dof_properties(st.env,st.arm_actor)
    physical_gripper_lower=float(physical_properties['lower'][indices[-1]])
    physical_gripper_upper=float(physical_properties['upper'][indices[-1]])
    st._fullcycle_gym=gym;st._fullcycle_sim=sim
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

    phase='author';passed=False;pass_step=None
    latched=None;ever_unlocked=False;settled=0;final_count=0;panel_contact_count=0;close_start=None
    panel_nonzero_contact_count=0;off_target_closing_contact_seen=False
    close_hold=ObservedHold();terminal_hold=ObservedHold();latch_observed_hold_s=None;simulated_steps=0
    prev_xy=None;prev_yaw=None;last_action=np.zeros(9);action=np.r_[0.,0.,home[indices]]
    previous_action=action.copy()
    foot_bad_since=None;leg_diagnostic=LegDiagnostic()
    nonzero_door_contact=False;nonzero_end_panel_contact=False
    samples=[];success=False;reason='budget_exhausted';step=-1;initial_record=None;pre_physics=None
    old_phase=''; stop_error=None;blocked_self_steps=0
    expert=CompleteExpert(m,st,dof_names,plan['expert'])
    velocity=omega=0.;far=False
    phase_before=None
    (output/'effective-expert.json').write_text(json.dumps(expert.cfg,indent=2)+'\n')
    def event(kind,**data):
        print(json.dumps(dict(event=kind,step=step,time_s=simulated_steps*dt,**serial(data))),flush=True)
    def save_stop_snapshot():
        # Render the CURRENT stopped state; no simulate/reset/replay call.
        # This task requests raw depth/state evidence, with no extra RGB capture.
        snapshot=dict(step=step,last_completed_step=simulated_steps-1,
            measured_post_time_s=simulated_steps*dt,reason=reason,
            pre_physics=pre_physics,door_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.door_actor,ga.STATE_ALL)),
            arm_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.arm_actor,ga.STATE_ALL)),
            base_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.actor_handles[0],ga.STATE_ALL)))
        (output/'stop-state.json').write_text(json.dumps(snapshot,indent=2)+'\n')
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
            feedback_root=gym.get_actor_rigid_body_states(st.env,st.arm_actor,ga.STATE_ALL)[0]['pose']
            quat=np.array([feedback_root['r'][k] for k in ['x','y','z','w']])
            feedback_B=np.array([feedback_root['p']['x'],feedback_root['p']['y'],m.yaw_from_quat_xyzw(quat)])
            observed=Measured(step*dt,feedback_B,measured_q.copy(),measured_qd.copy(),
                st.ik_state.current_pos_np.copy(),Rotation.from_quat(st.ik_state.current_quat_np).as_matrix(),
                float(door_pos[0]),float(door_vel[0]),far,not nonzero_door_contact,
                velocity,omega,latched is not None,task_monitor.may_latch())
            target=expert.tick(step,gym,observed)
            if target.failed:raise StopAttempt(target.failed)
            phase=target.phase
            if phase!=phase_before:event('phase',phase=phase);phase_before=phase
            latch_request=authorize_latch(target,observed,task_monitor)
            if latch_request is not None:
                latched=latch_request;st.door.open_stage=False
                latch_observed_hold_s=task_monitor.close_hold.elapsed
                event('latch',measured_angle=latched,observed_hold_s=latch_observed_hold_s)
                score.authorize_relock(step*dt,latched,task_monitor.metadata())
            action=apply_target(m,gym,sim,st,target,dof_names,np.r_[xy,yaw],step,dt)
            if step*dt<1.:
                action[-1]=np.clip(smooth_gripper(initial_gripper,home[indices[-1]],step*dt),physical_gripper_lower,physical_gripper_upper)
                st.dof_positions[indices[-1]]=action[-1]
            if target.joints is None and target.position is not None:
                st.last_target_pos=target.position.copy()
                st.last_target_quat=None if target.rotation is None else Rotation.from_matrix(target.rotation).as_quat()
            else:
                F=robot.fk(dict(zip(names,action[2:])),expert.frame.base_transform(target.base))['ee_gripper_link']
                st.last_target_pos=F[:3,3].copy();st.last_target_quat=Rotation.from_matrix(F[:3,:3]).as_quat()
            st.last_gripper=float(action[-1])
            if not np.isfinite(action).all(): raise StopAttempt('nonfinite_command')
            # Same Joint9 integrator for the executed base command; verify the
            # source pose-target interface has not introduced hidden sideways motion.
            integrated_xy,integrated_yaw,jtargets=m.apply_float_dp_joint_action9(action,xy,yaw,dt)
            if np.linalg.norm(integrated_xy-target.base[:2])>1e-5:raise StopAttempt('joint9_lateral_mismatch')
            xy=target.base[:2].copy();yaw=float(target.base[2])
            gym.set_actor_dof_position_targets(st.env,st.arm_actor,st.dof_positions)
            st.traj['base_xy']=xy.copy();st.traj['yaw']=float(yaw)
            st.last_phase=phase
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
            score.observe((step+1)*dt,float(door_pos[0]),latched is not None and not st.door.open_stage)
            rb=gym.get_actor_rigid_body_states(st.env,st.arm_actor,ga.STATE_ALL)
            root=rb[0]['pose'];actual_xy=np.array([root['p']['x'],root['p']['y']]);quat=np.array([root['r'][k] for k in ['x','y','z','w']]);actual_yaw=m.yaw_from_quat_xyzw(quat)
            # FK uses measured joints and the actual root; all link collision
            # AND visible mesh bounds, not just chassis center or EE target.
            legs={}
            base_actor=st.actor_handles[0]
            base_joints=gym.get_actor_dof_states(st.env,base_actor,ga.STATE_ALL)
            legs.update(dict(zip(gym.get_actor_dof_names(st.env,base_actor),base_joints['pos'])))
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
            row['expert_target']=dict(phase=target.phase,author_phase=target.author_phase,next_phase=expert.phase,base=target.base.tolist(),reference_elapsed_s=target.reference_elapsed_s,reference_complete=target.reference_complete,mode='joint' if target.joints is not None else 'cartesian')
            row['task_monitor']=task_monitor.metadata();row['pass_plane_x']=scene.pass_plane_x
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
            diagnostic=leg_diagnostic.observe(step,(step+1)*dt,foot_bad_since,bad,deviations)
            if diagnostic is not None:event('sustained_leg_posture_deviation',**diagnostic)
            if passed and not far: raise StopAttempt('return_crossing_after_whole_body_pass')
            if target.author_phase is None and not passed: raise StopAttempt('close_sequence_before_full_body_pass')
            if terminal_hold.reached(2.) and task_monitor.accepted():
                sequence_complete=True;reason='expert_sequence_completed';break
    except StopAttempt as e:
        reason=str(e);event('stop',reason=reason)
        try:save_stop_snapshot()
        except Exception as snapshot_error:
            stop_error='stop snapshot missing: '+repr(snapshot_error);reason+=';evidence_missing';event('error',reason=stop_error)
    except BaseException as e:
        reason='infrastructure_or_recording_error:'+repr(e);stop_error=traceback.format_exc();event('error',reason=reason);traceback.print_exc()
    finally:
        success=score.result()['success']
        if (not success or recorder is not None) and step>=0 and not (output/'stop-state.json').exists():
            try:save_stop_snapshot()
            except Exception as snapshot_error:
                stop_error='stop snapshot missing: '+repr(snapshot_error);reason+=';evidence_missing'
        log.close();contact_log.close();lock_log.close()
        for writer in depth.values(): writer.close()
        if recorder is not None:
            recorder.st.dp_recorder.metadata['success_criterion']=score.result()['criterion']
            recorder.close(success)
        for camera in cams.values():
            camera.expected=int(math.ceil(simulated_steps/2));camera.close()
        np.savez_compressed(output/'trajectory.npz',rows=np.asarray(samples),columns=np.array(['step','time']+['state_'+n for n in names[:0]]+['state_last_vx','state_last_yaw']+['state_'+n for n in names]+['action_vx','action_yaw']+['action_'+n for n in names]+['actual_x','actual_y','actual_yaw','door_angle','handle_angle','door_velocity','handle_velocity']))
        result=dict(success=success,reason=reason,steps=simulated_steps,attempted_step=step,
            sim_seconds=simulated_steps*dt,whole_body_pass_step=pass_step,end_panel_contact_steps=panel_contact_count,latched_angle=latched,final_hold_steps=final_count,attempts=1,error=stop_error,nominal_only=False,physical_push_review='raw measured logs retained; no video required or visual certification claimed',video_frames={k:v.count for k,v in cams.items()})
        result.update(
            contact_counter_scope='postdeployment approach and close; loaded actual contact, not phase completion',
            automatic_final_conditions_met=sequence_complete,
            expert_sequence_complete=sequence_complete,
            independent_task_acceptance='not_performed_no_video_requested',
            end_panel_nonzero_contact_steps=panel_nonzero_contact_count,
            off_target_closing_contact_seen=off_target_closing_contact_seen,
            success_field_meaning=score.result()['criterion'],
            simplified_score=score.result(),
            leg_posture_diagnostic=leg_diagnostic.metadata(),
            latch_observed_hold_s=latch_observed_hold_s,
            final_observed_hold_s=terminal_hold.elapsed,
            independent_task_monitor=task_monitor.metadata(),
        )
        (output/'result.json').write_text(json.dumps(serial(result),indent=2)+'\n');print(json.dumps(result),flush=True)
