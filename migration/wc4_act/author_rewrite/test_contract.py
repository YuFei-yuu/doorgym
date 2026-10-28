"""CPU-only regression checks for the user-requested control boundaries."""
import json,math,os,sys
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from author_rewrite.trajectory import CompleteExpert,Measured
from author_rewrite.adapter import apply_target
from author_rewrite.geometry import closing_reference


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    names=['joint%d'%i for i in range(1,7)]+['jointGripper']
    args=NS(door_x=2.52852384,door_y=.02402131,door_z_offset=.01,robot_z=.51635206,robot_pitch=.0450503,
            gripper_open=-math.pi/2,arm_ik_solver='gym_jacobian',sim_dt=.02,traverse_steps=185)
    door=NS(actor_position_offset=[.11,0,0],bounding={'min':[0,0,0]},actor_scale=1.,actor_yaw=math.pi/2)
    st=NS(args=args,door=door,home_positions=np.r_[np.zeros(6),-1.1775],base_start=np.array([3.58,.167]),
          base_stop=np.array([3.2,.16]),base_push=np.array([2.,.16]),base_traverse=np.array([1.26,.16]),
          yaw_start=3.1226,yaw_push=3.1226,yaw_traverse=3.1226,traj={},env=None,door_actor=None,ik_state=None)
    calls=[]
    def original(*a):
        assert a[1] is args
        calls.append(a[0]);return ('push_door',np.array([2.1,.21]),3.1226,np.array([2.,.2,.9]),None,-.4,None)
    m=NS(trajectory_targets=original)
    e=CompleteExpert(m,st,names)
    o=Measured(0.,np.array([1.26,.18,math.pi]),st.home_positions.copy(),np.zeros(7),np.array([1.5,.2,.9]),np.eye(3),math.pi/2,0.,True,True,0.,0.,False)
    target=e.tick(12,None,o)
    assert calls==[12] and np.array_equal(target.base,[2.1,.21,3.1226]) and target.gripper==-.4
    assert e.frame.robot_z==args.robot_z and e.frame.robot_pitch==args.robot_pitch
    # The whole prescribed closing target is invariant to actual door/contact state.
    e.enter('close',o)
    a=e.tick(0,None,replace(o,time=18.,door_angle=1.3,released=False))
    b=e.tick(0,None,replace(o,time=18.,door_angle=.3,released=True))
    assert np.array_equal(a.base,b.base) and np.array_equal(a.position,b.position) and np.array_equal(a.rotation,b.rotation)
    assert a.latch_angle is None and b.latch_angle is None
    # Cannot latch on the reference clock alone, nor without original monitor permission.
    e.closed_since=None
    z=e.tick(0,None,replace(o,time=43.,door_angle=0.,locked=False,may_latch=False))
    z=e.tick(0,None,replace(o,time=44.,door_angle=0.,locked=False,may_latch=False));assert z.latch_angle is None
    angle=.0001
    z=e.tick(0,None,replace(o,time=44.2,door_angle=angle,locked=False,may_latch=True));assert z.latch_angle==angle
    # No lateral base component: rotation in place, then a straight forward segment.
    prev=closing_reference(e.cfg,e.frame,0.)[0]
    lateral=0.
    for t in np.arange(.02,e.cfg['close_s']+.01,.02):
        B=closing_reference(e.cfg,e.frame,t)[0];d=B[:2]-prev[:2]
        lateral=max(lateral,abs(d@[-math.sin(prev[2]),math.cos(prev[2])]))
        prev=B
    assert lateral<1e-12
    print(json.dumps(dict(author_prefix_delegation=True,actual_sampled_frame=True,closing_independent_of_door_feedback=True,
        measured_monitor_latch=True,maximum_lateral_step_m=lateral,physics_steps=0)))
if __name__=='__main__':main()
