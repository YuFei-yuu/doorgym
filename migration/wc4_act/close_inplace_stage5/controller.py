"""Single nominal live-feedback controller. References are commands, never state.

Opening/retimed traversal uses the original runtime IK. Subsequent transitions
require measured tracking. Only the existing task monitor can authorize locking.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from close_inplace_stage4.common import FullPose, point, pose_rotation, normal, baseT
from close_inplace_stage2.candidate import smooth

WAYPOINTS=[np.array(x) for x in [
[.013347862010317076,1.0385171632217673,-.7423617644564436,.26366608012920434,.028649856633119083,-.059529263566097036],
[.21297417497412022,1.8314548255714562,-1.2052576885387072,-.19128657484004286,.07581597936310772,-.011235644380838627]]]
def angle(a,b):return (a-b+math.pi)%(2*math.pi)-math.pi

class LiveExpert:
    def __init__(self,robot,recipe,B,q,last,theta,t,home,finalyaw,dt,scene,endpoint,parking):
        self.S=FullPose(robot);self.cfg=recipe;self.dt=dt;self.home=np.array(home);self.finalyaw=finalyaw
        self.phase='retimed_traverse';self.t0=t;self.q0=np.array(q);self.cmd=np.array(last[2:]);self.route=None
        self.B0=np.array(B);self.endpoint=np.array(endpoint);self.parking=parking
        self.L=float(B[0]-endpoint[0]);v0=max(.001,float(last[0]));self.T=math.ceil(2*self.L/v0/dt)*dt
        if self.L<=0:raise RuntimeError('Traversal already beyond fixed endpoint')
        self.a=v0*self.T;self.prevq=np.array(q);self.previous_twist=np.zeros(2);self.bad_since=None
        self.contact_seen=False;self.last_progress=t;self.best_angle=theta;self.follow0=None
        assert abs(scene.base_z-.5)<1e-6 and abs(scene.base_pitch)<1e-6,'Nominal height/frame mismatch'
        assert np.max(abs(scene.world_from_reference-np.eye(4)))<1e-5
    def enter(self,phase,t,q,B):
        self.phase=phase;self.t0=t;self.q0=np.array(q);self.B0=np.array(B);self.bad_since=None
        self.P0,self.R0=self.pose(B,q)
    def pose(self,B,q):
        F=baseT(B)@self.S.fk(q);return F[:3,3]+F[:3,:3]@self.S.pad,F[:3,:3]
    def command(self,t,B,q,theta,omega,endcontact,anycontact,latched):
        from close_inplace_stage5.runtime import StopAttempt
        B=np.array(B);q=np.array(q);elapsed=t-self.t0;phase=self.phase
        qd=(q-self.prevq)/self.dt;self.prevq=q.copy();v=w=0.;desired=self.cmd.copy();diag={'phase':phase,'state_source':'measured','door_angle_actual':theta}
        target=None;R=None;nextphase=None
        if phase=='retimed_traverse':
            u=min(1.,(elapsed+self.dt)/self.T);a=self.a;L=self.L
            s=a*u+(10*L-6*a)*u**3+(-15*L+8*a)*u**4+(6*L-3*a)*u**5
            xy=self.endpoint.copy();xy[0]=self.endpoint[0]+L-s
            action=self.parking(xy,B,q)
            self.cmd=action[2:].copy()
            diag.update(parking_duration=self.T,endpoint=self.endpoint.tolist(),progress_m=s)
            if elapsed>=self.T and np.linalg.norm(B[:2]-self.endpoint)<1e-5 and abs(action[0])<.01:
                self.enter('endpoint_hold',t,q,B)
            return action,diag
        if phase=='endpoint_hold':
            if elapsed>=.5:nextphase='withdraw'
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
            s=smooth(elapsed/8);goal=point(self.cfg,theta)
            target=(1-s)*self.P0+s*goal
            R=Slerp([0,1],Rotation.from_matrix(np.array([self.R0,pose_rotation(theta,13.5)])))(s).as_matrix()
        elif phase=='close':
            # Advance a small contact target ahead of ACTUAL door angle. No prescribed door state.
            if theta<self.best_angle-.005:self.best_angle=theta;self.last_progress=t
            if t-self.last_progress>12 and not latched:raise StopAttempt('sustained_no_closing_progress_12s')
            roll=13.5+2*smooth((math.pi/2-theta)/math.radians(10))
            lead=math.radians(.8)*smooth(elapsed/2)
            target=point(self.cfg,max(-.005,theta-lead));R=pose_rotation(theta,roll)
            yawgoal=1.5*math.pi+math.pi/2*smooth((math.radians(70)-theta)/math.radians(35))
            w=float(np.clip(2*angle(yawgoal,B[2]),-.2,.2))
            if theta<=math.radians(35) and abs(angle(0,B[2]))<.003:
                if self.follow0 is None:self.follow0=B[:2].copy()
                progress=.45*smooth((math.radians(35)-theta)/math.radians(35))
                wanted=self.endpoint[0]+progress
                v=float(np.clip(1.5*(wanted-B[0]),0,.04))
                if B[0]+v*self.dt>self.endpoint[0]+.45:v=max(0,(self.endpoint[0]+.45-B[0])/self.dt)
            if latched:nextphase='release';v=w=0.
            diag.update(roll_deg=roll,lead_rad=lead,follow_distance=float(B[0]-self.endpoint[0]))
        elif phase=='release':
            target=self.P0-np.array([.12*smooth(elapsed/4),0,0]);R=self.R0
            if elapsed>=4 and np.linalg.norm(self.pose(B,q)[0]-target)<.008 and not anycontact:nextphase='retract'
        elif phase=='final_hold':desired=self.home.copy()
        else:raise StopAttempt('unknown_live_phase:'+phase)
        if R is not None:
            sol,info=self.S.solve(B,target,R,q,80);desired[:6]=sol;diag['ik']=info
            P,actualR=self.pose(B,q);pe=float(np.linalg.norm(P-target));oe=float(Rotation.from_matrix(R@actualR.T).magnitude())
            diag.update(target_pad=target.tolist(),target_rotation=R.tolist(),measured_position_error=pe,measured_orientation_error=oe)
            if phase=='approach' and elapsed>=8 and pe<=.008 and oe<=.01 and np.max(abs(qd[:6]))<=.05:
                nextphase='close';self.last_progress=t;self.best_angle=theta
            if phase=='close' and (pe>.06 or oe>.15):
                if self.bad_since is None:self.bad_since=t
                if t-self.bad_since>2:raise StopAttempt('sustained_close_pose_tracking_failure')
            else:self.bad_since=None
        # Bound command rates, using measured q as IK seed and previous command for actuator continuity.
        desired[:6]=np.clip(desired[:6],self.S.lo,self.S.hi)
        self.cmd+=np.clip(desired-self.cmd,-.7*self.dt,.7*self.dt)
        twist=np.array([v,w]);limits=np.array([.10,.3])*self.dt
        twist=self.previous_twist+np.clip(twist-self.previous_twist,-limits,limits)
        self.previous_twist=twist
        diag.update(joint_command=self.cmd.tolist(),measured_joint_speed_max=float(np.max(abs(qd[:6]))),joint_margin=float(np.min(np.r_[q[:6]-self.S.lo,self.S.hi-q[:6]])))
        # Bounded measured gates: no time-triggered skip when the requested motion was not achieved.
        limit={'endpoint_hold':3,'withdraw':10,'fold':12,'turn':15,'deploy_a':14,'deploy_b':12,'approach':16,'release':10,'retract':13,'restore_heading':18}.get(phase)
        if limit is not None and elapsed>limit:raise StopAttempt('stage_gate_timeout:'+phase)
        if nextphase:self.enter(nextphase,t,q,B)
        return np.r_[twist,self.cmd],diag
