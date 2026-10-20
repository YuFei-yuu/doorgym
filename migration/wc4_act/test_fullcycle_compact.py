"""Behavioral regressions for the fixed-90 compact design; no Gym import."""
import math
from pathlib import Path
import unittest
import numpy as np
from fullcycle_compact import Arm, ClosingCurve, CompactExpert, DEFAULT
from fullcycle_geometry import URDF, transform

ROOT=Path(__file__).resolve().parents[2]


class CompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot=URDF(ROOT/"high-level/data/asset/a2wz1/urdf/a2wz1.urdf")
        cls.curve=ClosingCurve()
        cls.home=np.r_[np.zeros(6),-1.1775]

    def controller(self,phase):
        ref=self.curve.at(math.pi/2);base=ref[:3]
        c=CompactExpert(self.robot,DEFAULT,base,self.home,np.r_[0.,0.,self.home],
                        math.pi/2,0,self.home,math.pi,.02)
        c.phase=phase
        return c,base

    def test_curve_is_nonholonomic(self):
        theta=np.linspace(math.pi/2,0,1001)
        x=np.array([self.curve.at(a) for a in theta])
        derivative=np.gradient(x[:,:2],math.pi/2-theta,axis=0)
        lateral=-np.sin(x[:,2])*derivative[:,0]+np.cos(x[:,2])*derivative[:,1]
        self.assertLess(max(abs(lateral[2:-2])),1e-5)
        self.assertTrue(np.all(x[:,6]>0))

    def test_arm_moves_right_to_left(self):
        x=np.array([self.curve.at(a) for a in np.linspace(math.pi/2,0,1001)])
        self.assertLess(x[0,4],0);self.assertGreater(x[-1,4],0)
        self.assertTrue(np.all(np.diff(x[:,4])>0))

    def test_rendezvous_ignores_measured_door_angle(self):
        c1,b=self.controller("compact_rendezvous")
        c2,_=self.controller("compact_rendezvous")
        a1,d1=c1.command(1,b,self.home,math.pi/2,0,False,False,False)
        a2,d2=c2.command(1,b,self.home,math.pi/4,0,False,False,False)
        np.testing.assert_array_equal(a1,a2)
        np.testing.assert_array_equal(d1["target_world"],d2["target_world"])

    def test_seek_does_not_wait_for_contact(self):
        c,b=self.controller("compact_seek");c.seek_offset=0
        c.command(2,b,self.home,math.pi/4,0,False,False,False)
        self.assertEqual(c.phase,"close")
        self.assertEqual(c.close_parameter,math.pi/2)

    def test_empty_closing_travel_advances(self):
        c,b=self.controller("close");c.close_parameter=math.pi/2
        action,_=c.command(2,b,self.home,math.pi/4,0,False,False,False)
        self.assertLess(c.close_parameter,math.pi/2)
        self.assertGreater(action[0],0)

    def test_stalled_panel_limits_progress(self):
        c,b=self.controller("close");c.close_parameter=math.pi/2-.1
        before=c.close_parameter
        c.command(2,b,self.home,math.pi/2,0,False,False,False)
        self.assertEqual(c.close_parameter,before)

    def test_release_has_no_reverse(self):
        c,b=self.controller("release")
        action,_=c.command(1,b,self.home,0,0,False,True,True)
        self.assertGreaterEqual(action[0],0)
        self.assertEqual(c.phase,"release")

    def test_release_waits_for_actual_separation(self):
        c,b=self.controller("release")
        c.command(3,b,self.home,0,0,False,True,True)
        self.assertEqual(c.phase,"release")
        c.command(3.02,b,self.home,0,0,False,False,True)
        self.assertEqual(c.phase,"retract")


if __name__=="__main__":unittest.main()
