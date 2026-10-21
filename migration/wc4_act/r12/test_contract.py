"""No scene/model construction: measured latch edges and policy-only source guard."""
import sys,ast,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r12.lock import MeasuredLock
from r11.criteria import terminal

def test_lock():
 m=MeasuredLock()
 for t in [0,.5,1.]:m.observe(t,0.,0.,False,None)
 assert not m.may_latch()
 for t in [1.1,1.7]:m.observe(t,0.,0.,True,None)
 assert not m.may_latch(),'Cannot relatch before an actual opening'
 m.observe(1.8,.1,0.,True,None)
 m.observe(2.,.001,0.,True,None);m.observe(2.49,.001,0.,True,None)
 assert not m.may_latch()
 m.observe(2.5,.001,0.,True,None);assert m.may_latch()
 m.observe(2.52,.001,.1,True,None);assert not m.may_latch()
 m.observe(3.,.001,0.,True,None);m.observe(3.5,.001,0.,True,None);assert m.may_latch()
 m.observe(3.52,.001,0.,True,.001);assert not m.may_latch()

def test_source():
 s=Path(__file__).with_name('rollout.py').read_text();tree=ast.parse(s)
 forbidden={'trajectory_targets','opening_command','CompactExpert','waypoint_command','update_arm_ik_targets_for_env','apply_dp_gripper_latch','enforce_locked_door_hinge','set_ik_target'}
 calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)]
 assert not set(calls)&forbidden
 assert 'st.ik_state.lower,st.ik_state.upper' not in s
 assert 'if not self.action_queue:' in (Path(__file__).resolve().parents[3]/'high-level/dp/door_policy_backend.py').read_text()
 assert 'dtype=torch.float64' in (Path(__file__).resolve().parents[3]/'high-level/lerobot/src/lerobot/datasets/video_utils.py').read_text()

if __name__=='__main__':
 test_lock();test_source();print('PASS: physical latch boundaries, no expert motion calls, queue replenishment and R10 preservation')
