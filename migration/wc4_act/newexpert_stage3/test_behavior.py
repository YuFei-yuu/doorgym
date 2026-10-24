"""Final implementation regressions: measured gates and preserved guards."""
import json,sys,math,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from fullcycle_compact import CompactExpert
from fullcycle_geometry import URDF
from newexpert_stage2.push_geometry import PadCurve
HOME=np.r_[np.zeros(6),-1.1775]
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');cls.c=json.loads(Path(__file__).with_name('geometry.json').read_text())['compact_expert'];cls.goal=PadCurve(cls.c).at(math.pi/2)[:3]
 def ctl(self,phase='compact_rendezvous'):
  b=self.goal.copy();b[:2]-=.27*np.array([math.cos(b[2]),math.sin(b[2])]);x=CompactExpert(self.robot,self.c,b,HOME,np.r_[0,0,HOME],math.pi/2,0,HOME,math.pi,.02);x.phase=phase;return x,b
 def test_stalled_base_does_not_deploy(self):
  x,b=self.ctl();a,_=x.command(100,b,HOME,math.pi/2,0,False,False,False);np.testing.assert_allclose(a[2:],HOME,atol=1e-12);self.assertEqual(x.phase,'compact_rendezvous')
 def test_base_arrived_arm_not_arrived_stays(self):
  x,b=self.ctl();x.command(100,self.goal,HOME,math.pi/2,0,False,False,False);self.assertEqual(x.phase,'compact_rendezvous')
 def test_arm_arrived_base_not_arrived_stays(self):
  x,b=self.ctl();q=np.r_[x.hover_q,0];x.command(100,b,q,math.pi/2,0,False,False,False);self.assertEqual(x.phase,'compact_rendezvous')
 def test_actual_arrival_advances(self):
  x,b=self.ctl();q=np.r_[x.hover_q,0];x.qcmd=q.copy();x.command(100,self.goal,q,math.pi/2,0,False,False,False);self.assertEqual(x.phase,'compact_seek')
 def test_nominal_reference_ignores_door_drift(self):
  x,b=self.ctl();y,_=self.ctl();a,d=x.command(2,b,HOME,math.pi/2,0,False,False,False);a2,d2=y.command(2,b,HOME,math.pi/4,0,False,False,False);np.testing.assert_array_equal(a,a2);self.assertEqual(d['target_world'],d2['target_world'])
 def test_lag_guard_unchanged(self):
  x,b=self.ctl('close');x.close_parameter=math.pi/2-.1;v=x.close_parameter;x.command(1,self.goal,np.r_[x.hover_q,0],math.pi/2,0,False,False,False);self.assertEqual(x.close_parameter,v)
 def test_release_waits_for_contact_clear(self):
  x,b=self.ctl('release');x.command(3,b,HOME,0,0,False,True,True);self.assertEqual(x.phase,'release')
if __name__=='__main__':unittest.main()
