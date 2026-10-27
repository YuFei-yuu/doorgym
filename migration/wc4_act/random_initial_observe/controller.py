"""Single nominal live-feedback controller. References are commands, never state.

Opening/retimed traversal uses the original runtime IK. Subsequent transitions
require measured tracking. Only the existing task monitor can authorize locking.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from close_inplace_stage4.common import point, pose_rotation, normal
from random_initial_observe.frame_solver import FrameSolver
from close_inplace_stage2.candidate import smooth
from close_inplace_stage6.dynamic_gate import DynamicContactGate
from random_initial_observe.reference import reference, TOTAL

WAYPOINTS=[np.array(x) for x in [
[.013347862010317076,1.0385171632217673,-.7423617644564436,.26366608012920434,.028649856633119083,-.059529263566097036],
[.21297417497412022,1.8314548255714562,-1.2052576885387072,-.19128657484004286,.07581597936310772,-.011235644380838627]]]
def angle(a,b):return (a-b+math.pi)%(2*math.pi)-math.pi

class LiveExpert:
    def __init__(self,robot,recipe,B,q,last,theta,t,home,finalyaw,dt,scene,endpoint,parking,early):
        self.scene=scene;B=scene.to_reference_base(B);endpoint=scene.to_reference_base([*endpoint,finalyaw])[:2];finalyaw-=scene.yaw_delta
        self.early=early;self.S=FrameSolver(robot,scene);self.cfg=recipe;self.dt=dt;self.home=np.array(home);self.finalyaw=finalyaw
        self.phase='retimed_traverse';self.t0=t;self.q0=np.array(q);self.cmd=np.array(last[2:]);self.route=None
        self.B0=np.array(B);self.endpoint=np.array(endpoint);self.parking=parking
        self.L=float(np.linalg.norm(B[:2]-endpoint));self.travel_direction=(endpoint-B[:2])/max(self.L,1e-12);v0=max(.001,float(last[0]));self.T=math.ceil(2*self.L/v0/dt)*dt
        if self.L<=0:raise RuntimeError('Traversal already beyond fixed endpoint')
        self.a=v0*self.T;self.prevq=np.array(q);self.previous_twist=np.zeros(2);self.bad_since=None
        self.dynamic_gate=DynamicContactGate();self.handoff=None;self.command_velocity=np.zeros(7)
        self.contact_seen=False;self.last_progress=t;self.best_angle=theta;self.follow0=None
        assert np.isfinite([scene.base_z,scene.base_pitch]).all()
    def enter(self,phase,t,q,B):
        self.phase=phase;self.t0=t;self.q0=np.array(q);self.B0=np.array(B);self.bad_since=None
        self.P0,self.R0=self.pose(B,q)
    def pose(self,B,q):
        F=self.scene.base_transform(B)@self.S.fk(q);return F[:3,3]+F[:3,:3]@self.S.pad,F[:3,:3]
    def command(self,t,B,q,theta,omega,endcontact,anycontact,latched):
        from random_initial_observe.runtime import StopAttempt
        B=self.scene.to_reference_base(B);q=np.array(q);elapsed=t-self.t0;phase=self.phase
        qd=(q-self.prevq)/self.dt;self.prevq=q.copy();v=w=0.;desired=self.cmd.copy();diag={'phase':phase,'state_source':'measured','door_angle_actual':theta}
        target=None;R=None;nextphase=None
        if phase=='retimed_traverse':
            u=min(1.,(elapsed+self.dt)/self.T);a=self.a;L=self.L
            s=a*u+(10*L-6*a)*u**3+(-15*L+8*a)*u**4+(6*L-3*a)*u**5
            xy=self.endpoint-self.travel_direction*(L-s)
            action=self.parking(xy,B,q)
            self.cmd=action[2:].copy()
            diag.update(parking_duration=self.T,endpoint=self.endpoint.tolist(),progress_m=s,early_arm=self.early.latest.copy(),early_arm_plan=self.early.plan)
            if elapsed>=self.T and np.linalg.norm(B[:2]-self.endpoint)<1e-5 and abs(action[0])<.01:
                self.enter('endpoint_hold',t,q,B)
            return action,diag
        if phase=='endpoint_hold':
            desired=self.early.command(t,q,anycontact)
            diag.update(early_arm=self.early.latest.copy(),early_arm_plan=self.early.plan)
            if elapsed>=.5 and self.early.latest['reference_complete'] and np.max(abs(q[:6]-self.home[:6]))<.03 and np.max(abs(qd[:6]))<.05 and not anycontact:nextphase='turn'
        elif phase=='withdraw':
            target=self.P0+np.array([0,.14*smooth(elapsed/3),0])
            sol,info=self.S.solve_point(B,target,None,q,80);desired[:6]=sol;diag['ik']=info
            if elapsed>=3 and np.linalg.norm(self.pose(B,q)[0]-target)<.008 and np.max(abs(qd[:6]))<.05 and not anycontact:nextphase='fold'
        elif phase in ('fold','retract'):
            duration=6. if phase=='fold' else 7.
            desired=(1-smooth(elapsed/duration))*self.q0+smooth(elapsed/duration)*self.home
            if elapsed>=duration and np.max(abs(q[:6]-self.home[:6]))<.03 and np.max(abs(qd[:6]))<.05 and not anycontact:
                nextphase='turn' if phase=='fold' else 'restore_heading'
        elif phase in ('turn','restore_heading'):
            goal=1.5*math.pi if phase=='turn' else self.finalyaw
            duration=10. if phase=='turn' else 12.
            wanted=self.B0[2]+angle(goal,self.B0[2])*smooth((elapsed+self.dt)/duration)
            w=float(np.clip(angle(wanted,B[2])/self.dt,-.6,.6));desired=self.home.copy()
            if elapsed>=duration and abs(angle(goal,B[2]))<.003 and abs(w)<.01:
                nextphase='deploy_a' if phase=='turn' else 'final_hold'
        elif phase in ('deploy_a','deploy_b'):
            index=0 if phase=='deploy_a' else 1;duration=8. if index==0 else 6.
            desired[:6]=(1-smooth(elapsed/duration))*self.q0[:6]+smooth(elapsed/duration)*WAYPOINTS[index]
            desired[-1]=self.home[-1]
            if elapsed>=duration and np.max(abs(q[:6]-WAYPOINTS[index]))<.03 and np.max(abs(qd[:6]))<.05:
                nextphase='deploy_b' if index==0 else 'approach'
        elif phase=='approach':
            s=float(np.clip(smooth(elapsed/8.8),0.,1.));goal=point(self.cfg,theta)
            target=(1-s)*self.P0+s*goal
            R=Slerp([0,1],Rotation.from_matrix(np.array([self.R0,pose_rotation(theta,13.5)])))(s).as_matrix()
        elif phase=='close':
            ref=reference(elapsed,self.handoff['theta'])
            nxt=reference(elapsed+self.dt,self.handoff['theta'])
            path_angle=ref['path_angle_rad']
            roll=13.5+2*smooth((math.pi/2-path_angle)/math.radians(10))
            nominal_target=point(self.cfg,path_angle);nominal_R=pose_rotation(path_angle,roll)
            blend=float(np.clip(smooth(elapsed/2),0.,1.))
            # Only the frozen handoff angle and clock drive the entire reference.
            transported_pad=self.P0+point(self.cfg,path_angle)-point(self.cfg,self.handoff['theta'])
            transported_R=Rotation.from_rotvec([0,0,path_angle-self.handoff['theta']]).as_matrix()@self.R0
            target=(1-blend)*transported_pad+blend*nominal_target
            R=Slerp([0,1],Rotation.from_matrix(np.array([transported_R,nominal_R])))(blend).as_matrix()
            requested_w=angle(nxt['base_yaw'],B[2])/self.dt;requested_v=0.
            w=float(np.clip(requested_w,-.2,.2))
            desired_xy=self.endpoint+np.array([nxt['follow_m'],0.])
            if nxt['follow_progress']>0:
                requested_v=float(np.dot(desired_xy-B[:2],[math.cos(B[2]),math.sin(B[2])])/self.dt)
                v=float(np.clip(requested_v,-.04,.04))
            endpoint_stopped=(ref['reference_complete'] and elapsed>=TOTAL+.5
                and np.max(abs(qd[:6]))<=.05 and np.linalg.norm(B[:2]-(self.endpoint+[.45,0]))<.008
                and abs(angle(0,B[2]))<.003 and np.max(abs(self.previous_twist))<.01)
            if latched and endpoint_stopped:nextphase='release';v=w=0.
            diag.update(handoff=self.handoff,reference=ref,reference_blend=blend,
                reference_complete=ref['reference_complete'],endpoint_stopped=bool(endpoint_stopped),
                reference_source='internal time only; measured door angle is not a target input',
                base_requested_vw=[requested_v,requested_w],base_rate_clipped=bool(abs(requested_v)>.040001 or abs(requested_w)>.200001),
                roll_deg=roll,follow_distance=float(B[0]-self.endpoint[0]))
        elif phase=='release':
            target=self.P0-np.array([.12*smooth(elapsed/4),0,0]);R=self.R0
            if elapsed>=4 and np.linalg.norm(self.pose(B,q)[0]-target)<.008 and not anycontact:nextphase='retract'
        elif phase=='final_hold':desired=self.home.copy()
        else:raise StopAttempt('unknown_live_phase:'+phase)
        if R is not None:
            sol,info=self.S.solve(B,target,R,q,80);desired[:6]=sol;diag['ik']=info
            P,actualR=self.pose(B,q);pe=float(np.linalg.norm(P-target));oe=float(Rotation.from_matrix(R@actualR.T).magnitude())
            diag.update(target_pad=target.tolist(),target_rotation=R.tolist(),measured_position_error=pe,measured_orientation_error=oe)
            if phase=='approach':
                accepted=self.dynamic_gate.observe(t,theta,omega,endcontact,pe,oe,ready=elapsed>=8.8)
                diag['dynamic_confirmation']=self.dynamic_gate.latest.copy()
                if accepted:
                    nextphase='close';self.last_progress=t;self.best_angle=theta
                    self.handoff=dict(theta=float(theta),measured_q=q.tolist(),measured_pad=P.tolist(),measured_rotation=actualR.tolist(),time_s=float(t))
            if phase=='close':
                # Preserve the previous issued joint command at the boundary as well.
                qb=float(np.clip(smooth(elapsed/2),0.,1.))
                continued_command=np.array(self.handoff['previous_joint_command'])+elapsed*np.array(self.handoff['previous_joint_velocity'])
                desired=(1-qb)*continued_command+qb*desired
            if phase=='close' and (pe>.06 or oe>.15):
                if self.bad_since is None:self.bad_since=t
                if t-self.bad_since>2:raise StopAttempt('sustained_close_pose_tracking_failure')
            else:self.bad_since=None
        # Bound command rates, using measured q as IK seed and previous command for actuator continuity.
        desired[:6]=np.clip(desired[:6],self.S.lo,self.S.hi)
        diag['requested_joint_rate_max']=float(np.max(abs(desired[:6]-self.cmd[:6]))/self.dt)
        diag['joint_rate_clipped']=bool(diag['requested_joint_rate_max']>.700001)
        old_command=self.cmd.copy()
        self.cmd+=np.clip(desired-self.cmd,-.7*self.dt,.7*self.dt)
        self.command_velocity=(self.cmd-old_command)/self.dt
        twist=np.array([v,w]);limits=np.array([.10,.3])*self.dt
        twist=self.previous_twist+np.clip(twist-self.previous_twist,-limits,limits)
        if phase=='close':diag['base_acceleration_clipped']=bool(np.max(abs(twist-np.array([v,w])))>1e-8)
        self.previous_twist=twist
        diag.update(joint_command=self.cmd.tolist(),measured_joint_speed_max=float(np.max(abs(qd[:6]))),joint_margin=float(np.min(np.r_[q[:6]-self.S.lo,self.S.hi-q[:6]])))
        # Bounded measured gates: no time-triggered skip when the requested motion was not achieved.
        limit={'endpoint_hold':3,'withdraw':10,'fold':12,'turn':15,'deploy_a':14,'deploy_b':12,'approach':16,'release':10,'retract':13,'restore_heading':18}.get(phase)
        if limit is not None and elapsed>limit:raise StopAttempt('stage_gate_timeout:'+phase)
        if nextphase:
            if nextphase=='close':
                self.handoff['previous_joint_command']=self.cmd.tolist()
                self.handoff['previous_joint_velocity']=self.command_velocity.tolist()
                diag['dynamic_handoff']=self.handoff.copy()
            self.enter(nextphase,t,q,B)
        return np.r_[twist,self.cmd],diag
