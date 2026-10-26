"""Live author-compatible 25Hz recording. Not a 50Hz log decimator."""
import hashlib,json
import numpy as np

PHASES=['walk','initial_hold','grasp','grasp_hold','close_gripper','rotate_handle','push_door','traverse_door','return_home','hold_home',
        'compact_unload','compact_withdraw','compact_fold','compact_bypass','compact_rendezvous','compact_seek','close','release','retract','compact_exit_turn','compact_exit_arc','restore_heading','final_hold','retimed_traverse','endpoint_hold','withdraw','fold','turn','deploy_a','deploy_b','approach']


class LiveRecording:
    def __init__(self,m,st,out,dof_names,dt):
        self.m=m;self.st=st;self.out=out;self.dt=dt
        self.saved_last=None;self.names=['vx','yaw']+['joint%d'%i for i in range(1,7)]+['jointGripper']
        st.args.dp_record_state_mode='a2w_last_command_joint_state9'
        st.args.dp_raw_root=str(out/'raw');st.args.no_keyframe_loss_weights=True
        # Pose calibration remains review metadata in physical-camera-binding;
        # do not let the converter add observation.camera_pose policy inputs.
        st.args.record_camera_pose=False;st.args.dp_task='push door, traverse, close, lock, release and return home'
        st.dp_recorder=m.dc.make_float_dp_recorder(st.args,st.door,st.index,'depth_only',PHASES,
            'ikpush',m.IKPUSH_STATE_VERSION,m.RawDoorDPRecorder,m.make_state_feature_names,
            randomization_metadata_key='ikpush_randomization',extra_metadata={
                'recording_semantics':'author record_float_dp_frame live post-physics; base command over preceding record interval',
                'full_task_recording':True,'policy_inputs':['state','front_masked_depth','wrist_masked_depth']})
        self.log=(out/'recording-clock.jsonl').open('x')

    def observe(self,gym,sim,step,dof_names,gripper_idx,door_pos,door_vel,images):
        st=self.st;m=self.m;due=step%2==0;before=st.dp_recorder.frame_count
        old=m.dc.capture_dp_camera_images_from_rendered
        # Reuse the exact already-rendered uint8 buffers; never video decode.
        if due:
            if images is None:raise RuntimeError('systemic_recording_missing_live_images')
            m.dc.capture_dp_camera_images_from_rendered=lambda *a,**k:images
        try:
            ok=m.dc.record_float_dp_frame(gym,sim,st,dof_names,gripper_idx,self.dt,
                PHASES.index(st.last_phase),door_pos,door_vel)
        finally:m.dc.capture_dp_camera_images_from_rendered=old
        if bool(ok)!=due or st.dp_recorder.frame_count!=before+int(due):
            raise RuntimeError('systemic_recording_frame_clock_mismatch')
        if not due:return
        f=st.dp_recorder.frames;s=np.asarray(f['state'][-1]);a=np.asarray(f['action'][-1])
        if s.shape!=(9,) or a.shape!=(9,) or not np.isfinite(np.r_[s,a]).all():
            raise RuntimeError('systemic_recording_invalid_joint9')
        expected=np.zeros(2) if self.saved_last is None else self.saved_last[:2]
        if not np.allclose(s[:2],expected,atol=1e-7,rtol=0):
            raise RuntimeError('systemic_recording_previous_25hz_command_mismatch')
        if not np.allclose(s[2:],np.asarray(f['replay_dof_pos'][-1])[12:19],atol=1e-7,rtol=0):
            raise RuntimeError('systemic_recording_actual_joint_alignment_mismatch')
        digests={}
        for view in ['front','wrist']:
            im=f[view+'_masked_depth'][-1]
            if im.dtype!=np.uint8 or im.shape!=(480,640):
                raise RuntimeError('systemic_recording_invalid_raw_depth')
            raw=np.asarray(images[view+'_masked_depth'])
            if raw.ndim==3:raw=raw[:,:,0]
            if not np.array_equal(im,raw):raise RuntimeError('systemic_recording_depth_buffer_changed')
            digests[view]=hashlib.sha256(im.tobytes()).hexdigest()
        self.log.write(json.dumps(dict(frame=before,source_step=step,physics_post_time_s=(step+1)*self.dt,
            prior_record_time_s=None if before==0 else (step-1)*self.dt,
            state_previous_25hz_base_command=expected.tolist(),action=a.tolist(),raw_depth_sha256=digests,
            state_joint_source='same post-physics actual arm sample',supervision='author live base-command aggregation and current joint targets'))+'\n')
        self.saved_last=a.copy()

    def close(self,automatic_success):
        self.st.dp_recorder.metadata['automatic_final_conditions_met']=bool(automatic_success)
        self.st.dp_recorder.save_episode();self.log.close()
