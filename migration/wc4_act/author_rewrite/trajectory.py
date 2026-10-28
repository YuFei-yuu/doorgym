"""Independent complete expert target program, around the original author prefix.

Inputs are post-physics measurements. Outputs are targets, never simulated state.
The future rollout host supplies original observations/recording and R11 geometry.
This module does not create scenes, step physics, or load a learned policy.
"""
from dataclasses import dataclass
from pathlib import Path
import json, math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from author_rewrite.geometry import DoorFrame, closing_reference, smooth, angle_delta


@dataclass
class Measured:
    time: float
    base: np.ndarray                   # world x,y,yaw
    q: np.ndarray                      # actor DOF order, including gripper
    qd: np.ndarray
    ee: np.ndarray
    ee_R: np.ndarray
    door_angle: float
    door_speed: float
    far: bool                          # measured full-body geometry, R11
    released: bool                     # contact evidence + opened gripper
    base_speed: float
    yaw_speed: float
    locked: bool                       # actual host latch state
    may_latch: bool = False             # unchanged TaskMonitor.may_latch()


@dataclass
class Target:
    phase: str
    base: np.ndarray
    position: np.ndarray = None
    rotation: np.ndarray = None
    joints: np.ndarray = None
    gripper: float = None
    latch_angle: float = None           # one request: preceding measured angle
    done: bool = False
    failed: str = None
    author_phase: str = None


class CompleteExpert:
    def __init__(self, author, st, dof_names, config=None):
        self.m, self.st = author, st
        self.cfg = config or json.loads(Path(__file__).with_name('config.json').read_text())
        self.frame = DoorFrame.runtime(st)
        self.home = np.asarray(st.home_positions).copy()
        self.arm_indices = [dof_names.index('joint%d'%i) for i in range(1,7)]
        self.gripper_index = dof_names.index('jointGripper')
        self.phase = 'author'; self.anchor = None; self.since = None
        self.closed_since = self.final_since = None
        self.fold_anchor = None; self.prefix_end = None; self.failed = None
        self.max_open = 0.; self.latched = None
        self.Bclose = closing_reference(self.cfg,self.frame,0.)[0]
        self.nav_points = [self.frame.base(x,0.)[:2] for x in self.cfg['bypass_hinge_xy']]
        self.nav_index = 0

    def next_leg(self, o):
        d=self.nav_points[self.nav_index]-o.base[:2]
        self.nav_yaw=math.atan2(d[1],d[0]) if np.linalg.norm(d)>1e-8 else o.base[2]
        if abs(angle_delta(o.base[2],self.nav_yaw))>math.pi/2:
            self.nav_yaw+=math.pi       # short reverse move avoids a 180° turn
        # Quintic peak speed is 1.875 times distance / duration.
        self.drive_s=max(1.,1.875*np.linalg.norm(d)/self.cfg['relocate_speed_m_s'])
        self.enter('move_to_start' if abs(angle_delta(o.base[2],self.nav_yaw))<1e-8 else 'turn_to_start',o)

    def enter(self, phase, o):
        self.phase, self.since, self.anchor = phase, o.time, o

    def home_target(self, phase, B):
        return Target(phase, np.asarray(B).copy(), joints=self.home.copy(), gripper=self.st.args.gripper_open)

    def joint_fold(self, o, anchor, delay=0.):
        u=smooth((o.time-anchor.time-delay)/self.cfg['fold_s'])
        q=(1-u)*anchor.q+u*self.home
        q[self.gripper_index]=self.home[self.gripper_index]
        return q

    def tick(self, step, gym, o):
        c,st,m=self.cfg,self.st,self.m
        self.max_open=max(self.max_open,abs(o.door_angle))
        if self.failed:
            return Target('failed',o.base.copy(),joints=o.q.copy(),failed=self.failed)
        if self.phase=='author':
            raw=m.trajectory_targets(step,st.args,st.door,gym,st.env,st.door_actor,st.ik_state,
                st.base_start,st.base_stop,st.base_push,st.base_traverse,st.yaw_start,st.yaw_push,st.yaw_traverse,st.traj)
            phase,xy,yaw,p,quat,grip,_=raw;B=np.r_[xy,yaw]
            R=None if quat is None else Rotation.from_quat(quat).as_matrix()
            target=Target(phase,B,p,R,gripper=grip,author_phase=phase)
            if phase=='walk':target.joints=self.home.copy()
            if phase=='traverse_door' and self.prefix_end is None:
                self.prefix_end=o.time+st.args.traverse_steps*st.args.sim_dt
                self.withdraw_anchor=o
            if self.prefix_end is not None:
                elapsed=o.time-self.withdraw_anchor.time
                # Preserve every original base sample through traverse; arm only.
                if elapsed<c['parallel_withdraw_s']:
                    direction=np.r_[self.withdraw_anchor.base[:2]-self.withdraw_anchor.ee[:2],0.]
                    direction/=max(np.linalg.norm(direction),1e-12)
                    target.position=self.withdraw_anchor.ee+smooth(elapsed/c['parallel_withdraw_s'])*c['withdraw_m']*direction
                    target.rotation=self.withdraw_anchor.ee_R
                    target.gripper=st.args.gripper_open
                else:
                    if self.fold_anchor is None:self.fold_anchor=o
                    target.joints=self.joint_fold(o,self.fold_anchor)
                    target.gripper=st.args.gripper_open
                if phase in ('return_home','hold_home'):
                    # Author base endpoint is kept; don't execute author's second fold.
                    target.phase='traverse_fold_finish'
                    if self.fold_anchor is not None and o.time-self.fold_anchor.time>=c['fold_s']:
                        if not o.far:
                            self.failed='author_traverse_did_not_clear_full_body'
                        elif self.max_open<math.radians(80):
                            self.failed='author_open_angle_below_R11'
                        else:
                            self.next_leg(o)
            return target

        # Close and subsequent phases must stay on the far side, as R11 requires.
        if not o.far:
            self.failed='full_body_returned_to_near_side'
            return Target('failed',o.base.copy(),joints=o.q.copy(),failed=self.failed)
        t=o.time-self.since;A=self.anchor
        if self.phase in ('turn_to_start','turn_for_close','restore_heading'):
            yaw=self.nav_yaw if self.phase=='turn_to_start' else self.Bclose[2] if self.phase=='turn_for_close' else st.yaw_traverse
            duration=max(c['turn_s'],1.875*abs(angle_delta(A.base[2],yaw))/.25);s=smooth(t/duration)
            B=A.base.copy();B[2]+=s*angle_delta(A.base[2],yaw)
            target=self.home_target(self.phase,B)
            if t>=duration:
                if self.phase=='turn_to_start':self.enter('move_to_start',o)
                elif self.phase=='turn_for_close':self.enter('deploy',o)
                else:self.enter('final_hold',o)
            return target
        if self.phase=='move_to_start':
            s=smooth(t/self.drive_s);B=A.base.copy();B[:2]=(1-s)*A.base[:2]+s*self.nav_points[self.nav_index]
            target=self.home_target(self.phase,B)
            if t>=self.drive_s:
                self.nav_index+=1
                if self.nav_index<len(self.nav_points):self.next_leg(o)
                else:self.enter('turn_for_close',o)
            return target
        if self.phase=='deploy':
            # Joint cues select a reachable elbow branch; the author DLS handles
            # approach/closing from the actual measured posture afterwards.
            half=c['deploy_s']/2;way=c['deploy_waypoints_q']
            q=A.q.copy();start=A.q[self.arm_indices] if t<half else np.asarray(way[0])
            end=np.asarray(way[0] if t<half else way[1])
            q[self.arm_indices]=(1-smooth((t if t<half else t-half)/half))*start+smooth((t if t<half else t-half)/half)*end
            target=Target('deploy',A.base.copy(),joints=q,gripper=st.args.gripper_open)
            if t>=c['deploy_s']:self.enter('approach',o)
            return target
        if self.phase=='approach':
            p,R,_,_,_=self.frame.contact(c,math.radians(c['reference_open_angle_deg']))
            s=smooth(t/c['approach_s'])
            target=Target('approach',A.base.copy(),(1-s)*A.ee+s*p,
                Slerp([0,1],Rotation.from_matrix([A.ee_R,R]))(s).as_matrix(),gripper=st.args.gripper_open)
            if t>=c['approach_s']:
                self.enter('close',o)
            return target
        if self.phase=='close':
            B,p,R,theta,pad=closing_reference(c,self.frame,t)
            target=Target('close',B,p,R,gripper=st.args.gripper_open)
            closed=abs(o.door_angle)<=math.radians(c['closed_tolerance_deg']) and abs(o.door_speed)<=math.radians(c['closed_speed_deg_s'])
            self.closed_since=(o.time if self.closed_since is None else self.closed_since) if closed else None
            if t>=c['close_s']:
                if self.closed_since is not None and o.time-self.closed_since>=c['closed_hold_s'] and o.may_latch:
                    self.latched=o.door_angle;target.latch_angle=o.door_angle;self.enter('release',o)
                elif t>=c['close_s']+c['close_wait_s']:
                    self.failed='physical_close_not_observed'
            return target
        if self.phase=='release':
            _,_,_,n,_=self.frame.contact(c,0.)
            target=Target('release',A.base.copy(),A.ee-smooth(t/c['release_s'])*c['withdraw_m']*n,A.ee_R,
                gripper=st.args.gripper_open)
            if t>=c['release_s'] and o.released:self.enter('fold',o)
            return target
        if self.phase=='fold':
            target=Target('fold',A.base.copy(),joints=self.joint_fold(o,A),gripper=st.args.gripper_open)
            if t>=c['fold_s']:self.enter('restore_heading',o)
            return target
        target=self.home_target('final_hold',A.base)
        ids=self.arm_indices
        okay=(o.locked and o.released and o.far and abs(o.door_angle)<=math.radians(1)
            and abs(o.door_speed)<=math.radians(1) and np.max(abs(o.q[ids]-self.home[ids]))<=.03
            and np.max(abs(o.qd[ids]))<=.05 and abs(angle_delta(o.base[2],st.yaw_traverse))<=math.radians(2)
            and abs(o.base_speed)<=.01 and abs(o.yaw_speed)<=.01)
        self.final_since=(o.time if self.final_since is None else self.final_since) if okay else None
        target.done=self.final_since is not None and o.time-self.final_since>=c['final_hold_s']
        return target
