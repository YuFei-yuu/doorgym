"""Meaningful task-boundary regressions, without simulator imports."""
import copy,unittest
from criteria import score


def example():
    rows=[];last=0.
    for i in range(301):
        q=0. if i<25 else (1.5 if i<100 else max(0.,1.5-(i-99)*.075))
        rows.append(dict(step=i,t=(i+1)*.02,action=[-.1,0]+[0]*7,actual_base=[0,0,0],
            arm_q=[0]*7,arm_qd=[0]*7,base_measured_fd_speed=0.,base_measured_fd_yaw_rate=0.,
            ee_actual=[i*.001,0,0],door_q=[q,0],door_qd=[0,0],
            pre_physics=dict(door_q=[last,0]),whole_geometry_max_x=1. if i<20 else -1.,pass_plane_x=0.,
            latched=0. if i>=151 else None,unlocked=1<=i<151,door_effort=[0,0],
            end_panel_raw_contact=100<=i<=120,loaded_door_contact=100<=i<=120,
            any_door_contact=100<=i<=120,zero_lambda_door_records=1,nonend_door_records=1))
        last=q
    locks=[dict(step=0,time_s=0.,kind='initial',latched=0.,delta=0.),dict(step=151,time_s=3.02,kind='relocked',latched=0.,delta=0.)]
    return rows,locks,dict(home=[0]*7,old_expert_final_yaw=0.)


class Tests(unittest.TestCase):
    def test_allowed_reverse_wait_zero_lambda_incidental_records(self):
        s,l,r=example();out=score(s,l,r)
        self.assertTrue(out['measured_task_gates_pass'],out)
        self.assertGreater(out['diagnostics']['negative_command_steps'],0)
        self.assertGreater(out['diagnostics']['command_held_intervals'],0)
        self.assertTrue(out['requires_original_video_and_code_review'])
    def test_phase_independent(self):
        s,l,r=example();before=score(s,l,r)
        for x in s:x['phase']='arbitrary_no_expert_stage'
        self.assertEqual(before,score(s,l,r))
    def test_clean_opening_contact_gap_not_extra_gate(self):
        s,l,r=example()
        for x in s:x['any_door_contact']=True
        self.assertTrue(score(s,l,r)['measured_task_gates_pass'])
    def test_returns_fail(self):
        s,l,r=example();s[130]['whole_geometry_max_x']=1.
        self.assertIn('returned_to_original_side',score(s,l,r)['failures'])
    def test_no_end_push_not_accepted(self):
        s,l,r=example()
        for x in s:x['end_panel_raw_contact']=False
        self.assertIn('no_aligned_measured_end_push_evidence',score(s,l,r)['failures'])
    def test_angle_threshold_and_final_joint_requirement(self):
        s,l,r=example();r['home']=[.031]*7
        self.assertIn('final_joint_hold_below_2s',score(s,l,r)['failures'])
        for x in s:x['door_q'][0]=min(x['door_q'][0],1.)
        self.assertIn('opening_below_80deg',score(s,l,r)['failures'])
    def test_early_latch_snap_force_and_hidden_write(self):
        s,l,r=example();l[-1]['time_s']=2.45
        self.assertIn('close_hold_below_0p5s',score(s,l,r)['failures'])
        s,l,r=example();l[-1]['latched']=.001
        self.assertIn('latch_did_not_store_measured_angle',score(s,l,r)['failures'])
        s,l,r=example();s[125]['door_effort'][0]=.001
        self.assertIn('auxiliary_hinge_effort',score(s,l,r)['failures'])
        s,l,r=example();s[125]['pre_physics']['door_q'][0]=.001
        self.assertIn('unlogged_pre_physics_hinge_change',score(s,l,r)['failures'])
        s,l,r=example();l.append(dict(step=125,time_s=2.5,kind='initial',latched=0.,delta=0.))
        self.assertIn('hinge_write_during_unlocked_interval',score(s,l,r)['failures'])
    def test_one_contact_and_auto_flag_not_full_certification(self):
        s,l,r=example()
        for x in s:x['automatic_terminal_pass']=True
        out=score(s,l,r);self.assertNotIn('complete_task_review',out)
        self.assertTrue(out['requires_original_video_and_code_review'])


if __name__=='__main__':unittest.main(verbosity=2)
