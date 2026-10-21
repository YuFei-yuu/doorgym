"""No-scene tests for geometry parity, shared monitor and live author clock."""
import copy,math,subprocess,tempfile,types,unittest
from pathlib import Path
import numpy as np
from fullcycle_scene import SceneFrame
from fullcycle_geometry import URDF,transform
from fullcycle_compact import CompactExpert,ClosingCurve,DEFAULT,rounded_route
from fullcycle_task_monitor import TaskMonitor

ROOT=Path(__file__).resolve().parents[2]


class PilotChecks(unittest.TestCase):
    def test_nominal_commands_equal_r7(self):
        old=types.ModuleType('frozen_r7')
        src=subprocess.check_output(['git','show','664aa7c:migration/wc4_act/fullcycle_compact.py'],cwd=ROOT,text=True)
        exec(compile(src,'frozen-r7','exec'),old.__dict__)
        robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf')
        home=np.r_[np.zeros(6),-1.1775];curve=ClosingCurve()
        for phase in ['compact_unload','compact_withdraw','compact_fold','compact_bypass','compact_rendezvous','compact_seek','close','release','retract','compact_exit_turn','compact_exit_arc','restore_heading','final_hold']:
            base=np.array([1.9,.18,math.pi]) if phase in ['compact_unload','compact_withdraw','compact_fold','compact_bypass'] else curve.at(math.pi/2)[:3]
            pair=[C(robot,DEFAULT,base,home,np.r_[.2,0.,home],math.pi/2,0.,home,math.pi,.02) for C in [old.CompactExpert,CompactExpert]]
            for c in pair:
                c.phase=phase;c.close_parameter=math.pi/2
                if phase=='compact_bypass':c.route=rounded_route(base,curve.at(math.pi/2)[:3],DEFAULT)
            a,b=[c.command(.3,base,home,math.pi/2,0.,False,False,False)[0] for c in pair]
            np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,err_msg=phase)

    def test_runtime_pitch_height_and_translation(self):
        f=SceneFrame([2.507,-.483,.02],0.,.53,math.radians(5),2.54,2.51)
        b=np.array([1.5,-.8,1.2]);np.testing.assert_allclose(f.to_reference_base(f.to_world_base(b)),b,atol=1e-12)
        np.testing.assert_allclose(f.base_transform(b),transform([1.5,-.8,.52],[0,math.radians(5),1.2]))
        self.assertEqual(f.pass_plane_x,2.51)
        np.testing.assert_array_equal(SceneFrame().world_from_reference,np.eye(4))

    def test_shared_monitor_elapsed_and_release_not_phase(self):
        m=TaskMonitor(np.zeros(6),math.pi)
        def obs(t,angle,vel,contact,end,latched=None):
            m.observe(t,True,angle,vel,latched is None,latched,contact,end,np.zeros(6),np.zeros(6),0,0,math.pi)
        obs(0.,1.,0.,False,False);obs(.02,.9,-1.,True,True)
        for k in range(25):obs(1+k*.02,0.,0.,True,True)
        self.assertFalse(m.may_latch());obs(1.52,0.,0.,True,True);self.assertTrue(m.may_latch())
        obs(2.,0.,0.,False,False,0.)
        obs(3.99,0.,0.,False,False,0.);self.assertFalse(m.accepted())
        obs(4.02,0.,0.,False,False,0.);self.assertTrue(m.accepted())

    def test_author_live_recorder_aggregates_25hz_base_command(self):
        import sys
        sys.path.insert(0,str(ROOT/'high-level/float_ik'))
        from overview import load
        from fullcycle_recording import LiveRecording
        m=load('record_clock_author','high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py')
        # Actual author state/action/recorder functions, synthetic read-only API.
        sys.argv=['test','--a2w_float_ik_config',str(ROOT/'high-level/float_ik/config/a2w_float_ik_push_door_parallel.yaml')]
        args=m.parse_args();args.sim_dt=float(np.float32(.02));args.depth_only=True
        names=['joint%d'%i for i in range(1,7)]+['jointGripper']
        door=types.SimpleNamespace(spec={'name':'wc4','path':'wc4/model.urdf'},actor_scale=1.,actor_yaw=math.pi/2,actor_position_offset=(.11,0,0),
            bounding={'min':[0,0,0]},dof_lower=np.zeros(2),dof_upper=np.ones(2)*math.pi/2,open_stage=True)
        state=np.zeros(7,dtype=[('pos','f4'),('vel','f4')]);state['pos']=.1
        gym=types.SimpleNamespace(get_actor_dof_states=lambda *a:state)
        st=types.SimpleNamespace(args=args,door=door,index=0,env=0,arm_actor=1,camera_handles={},ik_state=types.SimpleNamespace(target_quat_np=np.array([0,0,0,1.])),
            dof_positions=np.ones(7)*.2,base_start=np.array([0.,0.]),yaw_start=0.,traj={},last_target_pos=np.zeros(3),last_target_quat=None,last_gripper=.2,
            last_dp_action=None,dp_record_sim_steps=0,dp_record_warned_no_camera=False,last_phase='walk',dp_record_success=False)
        old=m.dc.current_ee_pose_from_refreshed_tensors;m.dc.current_ee_pose_from_refreshed_tensors=lambda *a:(np.zeros(3),np.array([0,0,0,1.]))
        with tempfile.TemporaryDirectory() as out:
            rec=LiveRecording(m,st,Path(out),names,args.sim_dt)
            image=np.repeat(np.arange(640,dtype=np.uint8)[None,:,None],480,axis=0);image=np.repeat(image,3,axis=2)
            images={v+'_masked_depth':image for v in ['front','wrist']}
            st.traj={'base_xy':np.array([0.,0.]),'yaw':0.};rec.observe(gym,0,0,names,6,np.zeros(2),np.zeros(2),images)
            st.traj['base_xy']=np.array([.004,0]);rec.observe(gym,0,1,names,6,np.zeros(2),np.zeros(2),None)
            st.traj['base_xy']=np.array([.010,0]);rec.observe(gym,0,2,names,6,np.zeros(2),np.zeros(2),images)
            frames=st.dp_recorder.frames
            self.assertAlmostEqual(float(frames['action'][1][0]),.25,places=6)
            self.assertAlmostEqual(float(frames['state'][1][0]),0.)
            np.testing.assert_allclose(frames['state'][1][2:],.1)
            np.testing.assert_allclose(frames['action'][1][2:],.2)
            self.assertEqual(frames['front_masked_depth'][1].shape,(480,640))
            rec.close(False)
            with np.load(Path(out)/'raw/episode_000000.npz',allow_pickle=True) as z:
                self.assertEqual(z['state'].shape,(2,9));self.assertEqual(z['front_masked_depth'].shape,(2,480,640))
        m.dc.current_ee_pose_from_refreshed_tensors=old


if __name__=='__main__':unittest.main()
