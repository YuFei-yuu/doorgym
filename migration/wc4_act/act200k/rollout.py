"""ACT50k horizon25 all-ACT closed loop; measured evidence adapted from the frozen R7 runner.
No expert motion, IK tracking, phase controller or gripper latch is called.
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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fullcycle_geometry import URDF
from fullcycle_contract import timing, door_contact_review_flags, ObservedHold
import hashlib
from r11.criteria import terminal
from fullcycle_task_monitor import TaskMonitor
from success_random5.criterion import SuccessMonitor
from legdiag_random5.diagnostic import LegDiagnostic
from r13.action_journal import ActionJournal
from r13.contacts import diagnostic

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
    if len(states)!=1 or int(args.steps)*dt>120.00001: raise ValueError('Single env, <=120 seconds only')
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
    lock=TaskMonitor(home,st.yaw_traverse);score=SuccessMonitor()
    leg_diagnostic=LegDiagnostic();foot_bad_since=None
    reference=dict(home=home.tolist(),old_expert_final_yaw=st.yaw_traverse)
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
    (output/'reference.json').write_text(json.dumps(dict(old_expert_final_yaw=st.yaw_traverse,old_expert_final_base=st.base_traverse.tolist(),home=home.tolist(),dof_names=dof_names,nominal_randomization=json.loads(st.args.ikpush_randomization_json),new_policy_inputs=[],scoring_truth_only=['door angle','robot geometry','contact state'],lock='measured angle latch after 0.5s physical close; no zero snap',hinge_torque_sources={k:getattr(st.args,k) for k in ['door_open_resistance','door_open_damping','door_auto_open_force','door_lock_force','door_joint_friction','door_joint_damping','handle_spring_stiffness','handle_spring_damping']}),indent=2)+'\n')
    log=(output/'states.jsonl').open('x');contact_log=(output/'contacts.jsonl').open('x');lock_log=(output/'lock-writes.jsonl').open('x')
    import door_policy_backend as backend
    from door_policy_backend import DoorPolicySubprocessController
    backend._worker_python_command=lambda checkpoint:[os.environ['DOOR_DP_LEROBOT_PYTHON'],'-B',str(ROOT/'migration/wc4_act/act200k/worker.py')]
    os.environ['WC4_ACT200K_TRIAL_OUTPUT']=str(output)
    controller=DoorPolicySubprocessController(plan['checkpoint'],device='cuda',action_horizon=25)
    assert controller.action_dim==9 and controller.action_horizon==25 and controller.pred_horizon==100
    assert controller.obs_horizon==1
    controller.reset()
    (output/'policy-binding.json').write_text(json.dumps(dict(config=controller.config,image_keys=controller.image_keys,obs_horizon=controller.obs_horizon,pred_horizon=controller.pred_horizon,executed_horizon=controller.action_horizon,checkpoint=plan['checkpoint']),indent=2)+'\n')
    physical_lower=np.asarray(physical_properties['lower'],dtype=np.float32)
    physical_upper=np.asarray(physical_properties['upper'],dtype=np.float32)
    (output/'joint-limits.json').write_text(json.dumps(dict(names=dof_names,
        physical_lower=physical_lower.tolist(),physical_upper=physical_upper.tolist(),
        ik_soft_lower=st.ik_state.lower.cpu().numpy().tolist(),
        ik_soft_upper=st.ik_state.upper.cpu().numpy().tolist(),
        applied_limits='physical; no IK soft clamp or gripper latch'),indent=2)+'\n')
    depth={} # No additional videos; original observation buffers retained below.
    phase='act';passed=False;pass_step=None;latched=None;ever_unlocked=False
    panel_contact_count=0;panel_nonzero_contact_count=0;off_target_closing_contact_seen=False
    terminal_hold=ObservedHold();simulated_steps=0;latch_observed_hold_s=None
    prev_xy=None;prev_yaw=None;last_action=np.zeros(9,dtype=np.float32)
    action=None;raw_action=None;previous_record_base=np.zeros(2,dtype=np.float32)
    samples=[];success=False;reason='budget_exhausted';step=-1;pre_physics=None
    stop_error=None;blocked_self_steps=0;final_count=0;settled=0
    policy_log=(output/'policy-inputs-actions.jsonl').open('x')
    applied_journal=ActionJournal(output/'worker-applied-evidence.jsonl')
    input_frames=[];input_first=0;input_directory=output/'policy-depth';input_directory.mkdir()
    def flush_inputs():
        nonlocal input_first
        if input_frames:
            np.savez_compressed(input_directory/('%06d.npz'%input_first),
                front=np.stack([x[0] for x in input_frames]),wrist=np.stack([x[1] for x in input_frames]))
            input_first+=len(input_frames);input_frames.clear()
    def event(kind,**data):
        print(json.dumps(dict(event=kind,step=step,time_s=simulated_steps*dt,**serial(data))),flush=True)
    def save_stop_snapshot():
        # Render the CURRENT stopped state; no simulate/reset/replay call.
        snapshot=dict(step=step,last_completed_step=simulated_steps-1,
            measured_post_time_s=simulated_steps*dt,reason=reason,
            pre_physics=pre_physics,door_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.door_actor,ga.STATE_ALL)),
            arm_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.arm_actor,ga.STATE_ALL)),
            base_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.actor_handles[0],ga.STATE_ALL)))
        (output/'stop-state.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    initial_base_dof=gym.get_actor_dof_states(st.env,st.actor_handles[0],ga.STATE_ALL).copy()
    initial_d=gym.get_actor_dof_states(st.env,st.door_actor,ga.STATE_ALL)
    initial_angle=float(initial_d['pos'][0])
    initial_rb=gym.get_actor_rigid_body_states(st.env,st.arm_actor,ga.STATE_ALL)
    initial_record=dict(door=serial(initial_d),arm=serial(gym.get_actor_dof_states(st.env,st.arm_actor,ga.STATE_ALL)),root=serial(initial_rb[0]),base_start=st.base_start.tolist(),yaw=st.yaw_start)
    (output/'initial-state.json').write_text(json.dumps(initial_record,indent=2)+'\n')
    paired_root=Path(os.environ['WC4_ACT200K_RUN'])/'actual-initial'/str(json.loads((output/'frozen-trial.json').read_text())['seed'])
    paired_root.mkdir(parents=True,exist_ok=True)
    for name in ['initial-state.json','physical-camera-binding.json','frozen-trial.json']:
        actual=json.loads((output/name).read_text());prior=paired_root/name
        if prior.exists():
            if json.loads(prior.read_text())!=actual:raise RuntimeError('Paired initial mismatch: '+name)
        else:prior.write_text(json.dumps(actual,indent=2)+'\n')
    (output/'paired-initial-binding.json').write_text(json.dumps(dict(reference=str(paired_root),all_actual_initial_physics_camera_equal=True),indent=2)+'\n')
    xy=np.array(st.base_start,dtype=float);yaw=float(st.yaw_start)
    last_cmd_xy=xy.copy();last_cmd_yaw=yaw
    try:
        for step in range(int(args.steps)):
            execution_run=os.environ.get('WC4_EXECUTION_RUN')
            if execution_run and (Path(execution_run)/'STOP_REQUEST.json').exists():raise StopAttempt('controlled_stop_request')
            gym.refresh_rigid_body_state_tensor(sim);gym.refresh_dof_state_tensor(sim);gym.refresh_jacobian_tensors(sim)
            m.current_ee_pose_from_refreshed_tensors(st.ik_state)
            arm=gym.get_actor_dof_states(st.env,st.arm_actor,ga.STATE_ALL)
            measured_q=np.asarray(arm['pos'],float);measured_qd=np.asarray(arm['vel'],float)
            door_pos,door_vel=m.get_actor_dof_state(gym,st.env,st.door_actor)
            observation_pre=dict(time_s=step*dt,state=np.r_[last_action[:2],measured_q[indices]].tolist(),
                arm_qd=measured_qd.tolist(),door_q=np.asarray(door_pos).tolist(),door_qd=np.asarray(door_vel).tolist())
            if step%2==0:
                # First observation is the physical initial state at t=0. Thereafter,
                # render measured state at 0.04s intervals before applying the next action.
                gym.step_graphics(sim);gym.render_all_camera_sensors(sim)
                policy_images=dc.capture_dp_camera_images_from_rendered(gym,sim,st.env,st.camera_handles,st.args)
                policy_state=np.asarray(np.r_[previous_record_base,measured_q[indices]],dtype=np.float32)
                input_frames.append(tuple(policy_images[v+'_masked_depth'].copy() for v in ['front','wrist']))
                if len(input_frames)>=100:flush_inputs()
                inference_started=time.perf_counter()
                raw_action=np.asarray(controller.act(policy_state,None,policy_images['wrist_masked_depth'],
                    front_masked_depth_rgb=policy_images['front_masked_depth']),dtype=np.float32)
                parent_act_wall_s=time.perf_counter()-inference_started
                if raw_action.shape!=(9,) or not np.isfinite(raw_action).all():raise StopAttempt('nonfinite_or_invalid_policy_action')
                action=raw_action.copy()
                action[2:]=np.clip(action[2:],physical_lower[indices],physical_upper[indices])
                policy_log.write(json.dumps(dict(frame=step//2,source_physics_step=step-1,
                    source_time_s=step*dt,applied_steps=[step,step+1],state=policy_state.tolist(),
                    input_dtype='float32 state; author uint8 depth preprocessing to float32',
                    images_sha256={v:hashlib.sha256(policy_images[v+'_masked_depth'].tobytes()).hexdigest() for v in ['front','wrist']},
                    worker_evidence=controller.last_action_evidence,
                    parent_act_wall_s=parent_act_wall_s,policy_output=raw_action.tolist(),applied_action=action.tolist(),
                    clipping_delta=(action-raw_action).tolist()))+'\n')
                applied_journal.begin(controller.last_action_evidence,raw_action,action,
                    dict(frame=step//2,physics_time_s=step*dt,state=policy_state.tolist()))
                previous_record_base=action[:2].copy()
            st.last_target_pos=None;st.last_target_quat=None;st.last_gripper=float(action[-1])
            if not np.isfinite(action).all(): raise StopAttempt('nonfinite_command')
            # Exactly the existing Joint9 base integrator at each physics step.
            xy,yaw,jtargets=m.apply_float_dp_joint_action9(action,xy,yaw,dt)
            m.set_a2w_joint_targets_from_action(st.dof_positions,dof_names,jtargets,physical_lower,physical_upper)
            m.set_robot_base_pose(gym,st.env,st.actor_handles,xy,st.args.robot_z,yaw,st.args.robot_pitch)
            gym.set_actor_dof_position_targets(st.env,st.arm_actor,st.dof_positions)
            applied_journal.command_applied(step,step*dt,np.r_[action[:2],st.dof_positions[indices]])
            st.traj['base_xy']=xy.copy();st.traj['yaw']=float(yaw)
            st.last_phase='act'
            # Unlock is driven by the actual handle; re-lock only observes measured
            # closure. This abstraction has no access to, or output of, robot motion.
            door_pos,door_vel=m.get_actor_dof_state(gym,st.env,st.door_actor)
            efforts=m.compute_door_efforts(st.door,door_pos,door_vel,st.args)
            ever_unlocked=ever_unlocked or bool(st.door.open_stage)
            if lock.may_latch() and latched is None:
                latched=float(door_pos[0]);latch_observed_hold_s=lock.close_hold.elapsed
                score.authorize_relock(step*dt,latched,lock.metadata());event("legal_relock",measured_angle=latched,gate=lock.metadata())
            if latched is not None:st.door.open_stage=False
            if (not ever_unlocked) or latched is not None:
                d=gym.get_actor_dof_states(st.env,st.door_actor,ga.STATE_ALL)
                target=initial_angle if latched is None else latched
                lock_log.write(json.dumps(dict(step=step,time_s=step*dt,timing='pre_physics',
                    kind='initial' if latched is None else 'relocked',before=float(d['pos'][0]),
                    latched=target,delta=target-float(d['pos'][0])))+'\n')
                d['pos'][0]=target;d['vel'][0]=0
                gym.set_actor_dof_states(st.env,st.door_actor,d,ga.STATE_ALL)
                door_pos,door_vel=m.get_actor_dof_state(gym,st.env,st.door_actor)
            if ever_unlocked and latched is None and float(efforts[0])!=0:raise StopAttempt('nonzero_auxiliary_hinge_effort')
            gym.apply_actor_dof_efforts(st.env,st.door_actor,efforts)
            pre_physics=dict(clock=timing(step,dt),door_q=np.asarray(door_pos).tolist(),door_qd=np.asarray(door_vel).tolist(),
                arm_dof=serial(gym.get_actor_dof_states(st.env,st.arm_actor,ga.STATE_ALL)),
                arm_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.arm_actor,ga.STATE_ALL)),
                base_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.actor_handles[0],ga.STATE_ALL)),
                door_bodies=serial(gym.get_actor_rigid_body_states(st.env,st.door_actor,ga.STATE_ALL)),
                command=action.tolist())
            gym.simulate(sim);gym.fetch_results(sim,True)
            simulated_steps=step+1
            applied_journal.physics_completed(step,(step+1)*dt)
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
                entry['diagnostic']=diagnostic(b0,b1,lam,body_map,robot_ids,end_ids,door_ids,panel_id,
                    pre_physics['door_q'][0],float(door_pos[0]))
                entry['classification']='unexplained_record_only'
                robot_body=b0 if b0 in robot_ids else b1 if b1 in robot_ids else None
                other=b1 if robot_body==b0 else b0
                if b0 in robot_ids and b1 in robot_ids:
                    pair=frozenset((body_map[b0]['body'],body_map[b1]['body']))
                    if pair==inherited_home_pair: entry['classification']='inherited_home_self_contact'
                    elif pair in adjacent: entry['classification']='kinematic_neighbor_self_contact'
                    else:
                        entry['classification']='other_self_contact_nonzero_lambda_record' if lam!=0 else 'other_self_contact_zero_lambda'
                        unexpected_loaded_self|=lam!=0
                if robot_body is not None and other>=0 and other not in robot_ids:
                    any_door_contact=any_door_contact or other in door_ids
                    if robot_body in end_ids and other==panel_id: end_panel=True
                    expected=(robot_body in end_ids and other in door_ids and body_map[other]['body'] in ['door_panel','lever_handle'])
                    entry['classification']='expected_end_door_contact' if expected else ('external_nonzero_lambda_record' if lam!=0 else 'external_zero_lambda_record_only')
                    flags=door_contact_review_flags(phase,lam,other in door_ids,robot_body in end_ids and other==panel_id)
                    entry['review_flags']=flags
                    nonzero_door_contact|=flags['nonzero_door_contact']
                    nonzero_end_panel_contact|=flags['nonzero_end_panel_contact']
                    off_target_closing_contact|=entry['diagnostic']['nonend_nonzero_during_measured_closing']
            off_target_closing_contact_seen|=off_target_closing_contact
            # Retain the historical stall heuristic as a trace only. It is
            # neither an author-baseline abort rule nor a success criterion.
            blocked_self=bool(unexpected_loaded_self and np.max(np.abs(measured_q[:6]-st.dof_positions[:6]))>.3 and np.max(np.abs(measured_qd[:6]))<.01)
            blocked_self_steps=blocked_self_steps+1 if blocked_self else 0
            panel_contact_count+=int(end_panel);panel_nonzero_contact_count+=int(nonzero_end_panel_contact)
            velocity=float(np.linalg.norm(actual_xy-prev_xy)/dt) if prev_xy is not None else 0
            omega=abs(angle_error(actual_yaw,prev_yaw))/dt if prev_yaw is not None else 0
            lock.observe((step+1)*dt,far,float(door_pos[0]),float(door_vel[0]),bool(st.door.open_stage),latched,any_door_contact,nonzero_end_panel_contact,measured_q,measured_qd,velocity,omega,actual_yaw)
            score.observe((step+1)*dt,float(door_pos[0]),latched is not None and not st.door.open_stage)
            base_names=list(gym.get_actor_dof_names(st.env,base_actor))
            deviations={n:abs(float(base_joints['pos'][i])-({'hip':0.,'thigh':.8,'calf':-1.5}[next(k for k in ['hip','thigh','calf'] if k in n)])) for i,n in enumerate(base_names) if any(k in n for k in ['hip','thigh','calf'])}
            bad=[n for n,v in deviations.items() if v>(.2 if 'hip' in n else .3)]
            foot_bad_since=((step+1)*dt if foot_bad_since is None else foot_bad_since) if bad else None
            leg_event=leg_diagnostic.observe(step,(step+1)*dt,foot_bad_since,bad,deviations)
            if leg_event is not None:event('sustained_leg_posture_deviation',**leg_event)
            closed_sample=abs(float(door_pos[0]))<=math.radians(1) and abs(float(door_vel[0]))<=math.radians(1)
            settled=settled+1 if latched is None and closed_sample else 0
            row=dict(step=step,t=(step+1)*dt,phase=st.last_phase,state=np.r_[policy_state[:2],measured_q[indices]],action=action.copy(),command_base=[*xy,yaw],actual_base=[*actual_xy,actual_yaw],base_measured_fd_speed=velocity,base_measured_fd_yaw_rate=omega,arm_q=measured_q.copy(),arm_qd=measured_qd.copy(),arm_command=st.dof_positions.copy(),ee_actual=st.ik_state.current_pos_np.copy(),ee_quat=st.ik_state.current_quat_np.copy(),ee_target=st.last_target_pos,ee_target_quat=st.last_target_quat,door_q=door_pos,door_qd=door_vel,door_effort=efforts,unlocked=bool(st.door.open_stage),latched=latched,whole_geometry_max_x=max_x,whole_geometry_bounds=[cloud.min(0),cloud.max(0)],passed=passed,end_panel_contact=end_panel,any_door_contact=any_door_contact,consecutive_unlatched_closed_samples_diagnostic=settled,final_hold_steps=final_count,root_rb=serial(rb[0]),arm_body_states=serial(rb),base_body_states=serial(gym.get_actor_rigid_body_states(st.env,base_actor,ga.STATE_ALL)))
            row['timing']=timing(step,dt);row['pre_physics']=pre_physics
            row['door_body_states']=serial(gym.get_actor_rigid_body_states(st.env,st.door_actor,ga.STATE_ALL))
            row['observation_pre']=observation_pre
            row['self_record_and_joint_tracking_gap_steps']=blocked_self_steps
            row['nonzero_door_contact']=nonzero_door_contact
            row['nonzero_end_panel_contact']=nonzero_end_panel_contact
            row['nonend_nonzero_record_during_measured_closing']=off_target_closing_contact
            row['contact_interpretation']='Raw SDK lambda is not calibrated force; these flags are diagnostics and do not score success/failure'
            row['r11_loaded_door_contact_alias']='nonzero lambda robot-door record; unchanged R11 input semantics'
            row.update(pass_plane_x=scene.pass_plane_x,end_panel_raw_contact=end_panel,
                loaded_door_contact=nonzero_door_contact,
                zero_lambda_door_records=sum(c.get('lambda',0)==0 and any(n.get('actor_handle')==st.door_actor for n in c['names']) for c in contacts),
                nonend_door_records=sum(c['diagnostic']['nonend_door_record'] for c in contacts))
            terminal_hold.observe((step+1)*dt,terminal(row,reference))
            final_count=final_count+1 if terminal(row,reference) else 0
            row['final_hold_steps']=final_count
            row['final_hold_observed_span_s']=terminal_hold.elapsed
            row['physical_close_observed_span_s']=lock.close_hold.elapsed
            row['task_monitor']=lock.metadata();row['simplified_score']=score.result()
            row['foot_guard']=dict(deviations=deviations,bad=bad,since_s=foot_bad_since,hold_s=.10)
            log.write(json.dumps(serial(row))+'\n');contact_log.write(json.dumps(dict(step=step,interval_s=[step*dt,(step+1)*dt],contacts=contacts))+'\n')
            samples.append([step,(step+1)*dt,*np.r_[policy_state[:2],measured_q[indices]],*action, *actual_xy,actual_yaw,*door_pos,*door_vel])
            images=None
            if step%2==0:
                # install_live hooks provide synchronized original RGB.
                if not cams:gym.step_graphics(sim);gym.render_all_camera_sensors(sim)
                m.update_door_twin_tracker(st,step,door_pos)
                images=dc.capture_dp_camera_images_from_rendered(gym,sim,st.env,st.camera_handles,st.args)
                for view in depth:
                    im=images.get(view+'_masked_depth')
                    if im is None: raise StopAttempt('missing_depth_observation')
                    depth[view].append_data(im)
            prev_xy=actual_xy.copy();prev_yaw=actual_yaw;previous_action=last_action.copy();last_action=action.copy()
            if step%50==0: event('progress',phase=st.last_phase,door_deg=float(np.rad2deg(door_pos[0])),base=[*actual_xy,actual_yaw],geometry_max_x=max_x)
            if terminal_hold.reached(2.):
                reason='measured_stable_terminal_hold';break
    except StopAttempt as e:
        reason=str(e);event('stop',reason=reason)
        try:save_stop_snapshot()
        except Exception as snapshot_error:
            stop_error='stop snapshot missing: '+repr(snapshot_error);reason+=';evidence_missing';event('error',reason=stop_error)
    except BaseException as e:
        reason='infrastructure_or_recording_error:'+repr(e);stop_error=traceback.format_exc();event('error',reason=reason);traceback.print_exc()
    finally:
        success=score.result()['success']
        if step>=0 and not (output/'stop-state.json').exists():
            try:save_stop_snapshot()
            except Exception as snapshot_error:
                stop_error='stop snapshot missing: '+repr(snapshot_error);reason+=';evidence_missing'
        log.close();contact_log.close();lock_log.close()
        for writer in depth.values(): writer.close()
        flush_inputs();policy_log.close();applied_journal.close();controller.close()
        for camera in cams.values():
            camera.expected=int(math.ceil(simulated_steps/2));camera.close()
        np.savez_compressed(output/'trajectory.npz',rows=np.asarray(samples),columns=np.array(['step','time']+['state_'+n for n in names[:0]]+['state_last_vx','state_last_yaw']+['state_'+n for n in names]+['action_vx','action_yaw']+['action_'+n for n in names]+['actual_x','actual_y','actual_yaw','door_angle','handle_angle','door_velocity','handle_velocity']))
        result=dict(success=success,reason=reason,steps=simulated_steps,attempted_step=step,
            sim_seconds=simulated_steps*dt,whole_body_pass_step=pass_step,end_panel_contact_steps=panel_contact_count,latched_angle=latched,final_hold_steps=final_count,attempts=1,error=stop_error,nominal_only=False,physical_push_review='measured state evidence; no video requested',video_frames={k:v.count for k,v in cams.items()})
        result.update(
            automatic_final_conditions_met=success,
            independent_task_acceptance='pending' if success else 'not_reached',
            end_panel_nonzero_contact_steps=panel_nonzero_contact_count,
            nonend_nonzero_record_during_measured_closing_seen=off_target_closing_contact_seen,
            success_field_meaning=score.result()['criterion'],
            latch_observed_hold_s=latch_observed_hold_s,
            final_observed_hold_s=terminal_hold.elapsed,
            lock_abstraction=lock.metadata(),simplified_score=score.result(),leg_posture_diagnostic=leg_diagnostic.metadata(),
        )
        (output/'result.json').write_text(json.dumps(serial(result),indent=2)+'\n');print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--checkpoint',required=True)
    a,remaining=p.parse_known_args();output=Path(a.output);output.mkdir(exist_ok=False)
    sys.path[:0]=[str(ROOT/'migration/wc4_act'),str(ROOT/'high-level/float_ik'),str(ROOT/'high-level/dp')]
    from overview import load,install_live
    m=load('r13_author','high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
    from fullcycle_pilot import install_frozen_trial
    install_frozen_trial(m,Path(a.manifest),0,output)
    # Guard against accidentally reaching an author motion/lock implementation.
    def forbidden(*args,**kwargs):raise RuntimeError('Forbidden expert/old-lock entry reached')
    m.trajectory_targets=forbidden;m.update_arm_ik_targets_for_env=forbidden
    m.enforce_locked_door_hinge=forbidden
    m.dc.enforce_locked_door_hinge=forbidden
    cams={}
    from newexpert_stage7.adapters import install_raw_render
    install_raw_render(m)
    original_parse=m.parse_args
    def parse():
        args=original_parse();args.show_camera_images=True
        if args.dp_policy_checkpoint or args.record_dp_dataset:raise ValueError('Custom ACT runner exclusively')
        return args
    m.parse_args=parse
    m.run_parallel_demo=lambda *args:run(m,cams,output,{'checkpoint':a.checkpoint},*args)
    sys.argv=[str(ROOT/'high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py'),*remaining]
    (output/'command.json').write_text(json.dumps(dict(argv=sys.argv,checkpoint=a.checkpoint,
        execution_horizon=25,control_hz=25,physics_hz=50,gpu={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','VIDEO2DOOR_GPU','VK_INSTANCE_LAYERS','VK_LAYER_PATH','VIDEO2DOOR_VK_DEVICE_INDEX']}),indent=2)+'\n')
    create_sim=m.base_ik.create_sim
    def claimed(*args,**kwargs):
        with (output/'scene-creation-receipt.json').open('x') as f:json.dump(dict(scenes=1,max_sim_seconds=120),f)
        return create_sim(*args,**kwargs)
    m.base_ik.create_sim=claimed
    m.main()
    result=json.loads((output/'result.json').read_text())
    if result['reason']=='controlled_stop_request':raise SystemExit(75)
    if result['error'] or any(k in result['reason'] for k in ['nonfinite','auxiliary','evidence_missing','missing_depth']):raise RuntimeError('Systemic rollout stop: '+result['reason'])


if __name__=='__main__':main()
