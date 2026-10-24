"""R7 compact expert. Pure Joint9 commands; no Gym calls or state writes.

The approach is fixed to a nominal 90-degree open door. The nominal
closing curve satisfies the unicycle constraint while the tool moves right to
left in the body frame. Runtime IK uses the observed base and planned angle.
Measured door angle is used for settling and a progress backstop only.
"""
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fullcycle_geometry import transform, rot


def wrap(x):
    return (float(x)+math.pi) % (2*math.pi)-math.pi


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x*x*x*(10+x*(-15+6*x))


DEFAULT = dict(
    revision="compact_nominal90_v1", radius=.84, height=.90, pad_offset=.02,
    start_yaw_deg=75., end_yaw_deg=35., final_curvature_ratio=.7,
    forward_reach_start=.60, forward_reach_end=.70, lateral_reach_start=-.15,
    corridor_x=1.05, first_radius=.30, staging_distance=.22,
    max_speed=.18, max_yaw_rate=.50, linear_acceleration=.20,
    joint_speed=.70, closing_rate=.060, closing_lag_limit=.08,
    unload_seconds=.50, withdraw_seconds=2., withdraw_distance=.14,
    fold_seconds=3., deploy_seconds=3., hover_distance=.04,
    seek_speed=.025, seek_travel=.07, release_distance=.10,
    release_seconds=2., retract_seconds=3.,
)


class Arm:
    def __init__(self, robot):
        self.robot=robot
        self.names=["joint%d" % i for i in range(1,7)]
        limits=[robot.root.find("joint[@name='%s']/limit" % n) for n in self.names]
        self.lo=.75*np.array([float(e.get("lower")) for e in limits])
        self.hi=.75*np.array([float(e.get("upper")) for e in limits])
        child={j[3]:j for j in robot.joints};current="ee_gripper_link";self.chain=[]
        while current in child:
            j=child[current];self.chain.insert(0,j);current=j[2]

    def fk(self,q):
        values=dict(zip(self.names,q));T=np.eye(4)
        for name,kind,parent,child,O,axis in self.chain:
            T=T@O
            if name in values:
                M=np.eye(4);M[:3,:3]=rot(axis,values[name]);T=T@M
        return T

    def solve(self,target,seed,orientation_weight=1.,iterations=40):
        seed=np.clip(np.asarray(seed)[:6],self.lo+1e-8,self.hi-1e-8)
        def residual(q):
            F=self.fk(q)
            return np.r_[4*(F[:3,3]-target[:3,3]),
                orientation_weight*Rotation.from_matrix(target[:3,:3]@F[:3,:3].T).as_rotvec(),
                .001*(q-seed)]
        r=least_squares(residual,seed,bounds=(self.lo,self.hi),max_nfev=iterations,
                        ftol=1e-7,xtol=1e-7,gtol=1e-7)
        F=self.fk(r.x)
        return r.x,dict(position_error_m=float(np.linalg.norm(F[:3,3]-target[:3,3])),
            orientation_error_rad=float(np.linalg.norm(
                Rotation.from_matrix(target[:3,:3]@F[:3,:3].T).as_rotvec())))


class ClosingCurve:
    def __init__(self,recipe=None):
        self.c=dict(DEFAULT,**(recipe or {}))
        c=self.c
        self.s=np.linspace(-math.radians(10),math.pi/2,4001)
        S=math.pi/2;mean=(c["start_yaw_deg"]-c["end_yaw_deg"])/90
        k1=c["final_curvature_ratio"];k0=2*mean-k1
        s=self.s;theta=S-s
        psi=math.radians(c["start_yaw_deg"])-k0*s-.5*(k1-k0)*s*s/S
        d=c["forward_reach_start"]+(c["forward_reach_end"]-c["forward_reach_start"])*s/S
        k=k0+(k1-k0)*s/S;beta=theta-psi
        bp=c["radius"]*np.sin(beta)+c["pad_offset"]*np.cos(beta)+k*d
        integral=np.r_[0,np.cumsum((bp[:-1]+bp[1:])*.5*np.diff(s))]
        b=c["lateral_reach_start"]+integral-np.interp(0,s,integral)
        forward=(c["radius"]*np.cos(beta)-c["pad_offset"]*np.sin(beta)-k*b
                 -(c["forward_reach_end"]-c["forward_reach_start"])/S)
        P=self.point(theta)
        B=P-np.c_[d*np.cos(psi)-b*np.sin(psi),d*np.sin(psi)+b*np.cos(psi)]
        self.table=np.c_[B,psi,d,b,k,forward,bp]

    def point(self,theta,offset=None):
        theta=np.asarray(theta)
        e=self.c["pad_offset"] if offset is None else offset
        return np.stack([2.467-self.c["radius"]*np.sin(theta)-e*np.cos(theta),
                         -.463+self.c["radius"]*np.cos(theta)-e*np.sin(theta)],axis=-1)

    def at(self,theta):
        s=math.pi/2-float(theta)
        if s<self.s[0]-.001 or s>self.s[-1]+.06:
            raise ValueError("Door angle outside reviewed compact workspace")
        return np.array([np.interp(s,self.s,col) for col in self.table.T])

    def tool(self,theta,extra_offset=0):
        P=self.point(theta,self.c["pad_offset"]+extra_offset)
        return transform([*P,self.c["height"]],[math.pi/2,0,theta])


def rounded_route(start,goal,recipe):
    """West, CCW quarter-turn, south, compact CCW turn, staging pose.

    The arm remains folded. The endpoint is behind the current contact pose.
    This is a path reference, subsequently tracked by forward-only controls.
    """
    start=np.asarray(start,float);goal=np.asarray(goal,float)
    x=recipe["corridor_x"];R=recipe["first_radius"]
    psi=float(goal[2]%(2*math.pi))
    if not math.radians(30)<psi<math.radians(90):
        raise ValueError("Contact heading outside compact bypass family")
    staging=goal[:2]-recipe["staging_distance"]*np.array([math.cos(psi),math.sin(psi)])
    r=(staging[0]-x)/(1+math.sin(psi))
    if r<.045 or r>.35 or start[0]<x+R:
        raise ValueError("Compact route cannot fit the observed rendezvous")
    yturn=staging[1]+r*math.cos(psi)
    if yturn>=start[1]-R-.1:raise ValueError("Insufficient free-edge bypass depth")
    points=[];headings=[]
    def add(p,h):
        if not points or np.linalg.norm(np.asarray(p)-points[-1])>1e-9:
            points.append(np.asarray(p));headings.append(h)
    def line(a,b,h):
        for u in np.linspace(0,1,max(2,int(np.linalg.norm(np.asarray(b)-a)/.004)+1)):
            add((1-u)*np.asarray(a)+u*np.asarray(b),h)
    def arc(center,r,a,b):
        for h in np.linspace(a,b,max(3,int(r*(b-a)/.004)+1)):
            add(np.asarray(center)+r*np.array([math.sin(h),-math.cos(h)]),h)
    line(start[:2],[x+R,start[1]],math.pi)
    arc([x+R,start[1]-R],R,math.pi,1.5*math.pi)
    line([x,start[1]-R],[x,yturn],1.5*math.pi)
    arc([x+r,yturn],r,1.5*math.pi,2*math.pi+psi)
    return np.c_[points,headings]


def forward_pose(base,goal,vmax=.12,wmax=.4):
    """Forward-only pose acquisition; in-place rotations are permitted."""
    delta=np.asarray(goal[:2])-base[:2];distance=float(np.linalg.norm(delta))
    if distance>.008:
        error=wrap(math.atan2(delta[1],delta[0])-base[2])
        w=float(np.clip(2*error,-wmax,wmax))
        v=min(vmax,1.2*distance)*max(0,math.cos(error))
        if abs(error)>.45:v=0.
        return v,w,False
    error=wrap(goal[2]-base[2])
    return 0.,float(np.clip(2*error,-wmax,wmax)),abs(error)<.008


class CompactExpert:
    def __init__(self,robot,recipe,base,q,action,theta,t,home,final_yaw,dt,scene=None):
        from fullcycle_scene import SceneFrame
        self.scene=scene or SceneFrame()
        base=self.scene.to_reference_base(base)
        final_yaw=float(final_yaw)-self.scene.yaw_delta
        self.c=dict(DEFAULT,**recipe);self.arm=Arm(robot);self.curve=ClosingCurve(self.c)
        self.dt=dt;self.phase="compact_unload";self.since=t
        self.qcmd=np.asarray(action[2:],float).copy();self.home=np.asarray(home,float)
        self.base0=np.asarray(base,float).copy();self.q0=np.asarray(q,float).copy()
        self.v=float(action[0]);self.w=float(action[1]);self.final_yaw=float(final_yaw)
        self.E0=self.scene.base_transform(base)@self.arm.fk(q)
        self.withdraw_normal=np.array([math.cos(theta),math.sin(theta),0.])
        self.route=None;self.route_index=0;self.seek_offset=self.c["hover_distance"]
        self.close_parameter=None;self.target=None;self.ik={};self.events=[]
        self.last_t=t;self.closed=False;self.seen_closing_contact=False
        ref=self.curve.at(math.pi/2)
        goal_B=self.scene.base_transform(ref[:3])
        self.hover_q,self.hover_ik=self.arm.solve(
            np.linalg.inv(goal_B)@self.curve.tool(math.pi/2,self.c["hover_distance"]),
            [0.,1.7,-1.3,-.4,0.,math.pi/2],1.,120)

    def change(self,phase,t,q,base):
        self.events.append(dict(time_s=t,previous=self.phase,phase=phase))
        self.phase=phase;self.since=t;self.q0=np.asarray(q).copy()
        self.E0=self.scene.base_transform(base)@self.arm.fk(q)
        self.base0=np.asarray(base).copy()

    def command(self,t,base,q,theta,theta_dot,end_contact,door_contact,latched):
        c=self.c;base=self.scene.to_reference_base(base);q=np.asarray(q,float);elapsed=t-self.since
        dt=self.dt;v=w=0.;target_q=self.home.copy();self.ik={};self.target=None
        B=self.scene.base_transform(base)
        if latched and self.phase=="close":
            self.change("release",t,q,base);elapsed=0
        if self.phase=="compact_unload":
            # Keep the current arm posture while braking; release gripper to
            # the inherited open target. No yaw motion while unloading.
            target_q=np.r_[np.clip(self.q0[:6],self.arm.lo,self.arm.hi),self.home[-1]]
            if elapsed>=c["unload_seconds"] and self.v<.005:
                self.change("compact_withdraw",t,q,base)
        elif self.phase=="compact_withdraw":
            E=self.E0.copy()
            E[:3,3]+=self.withdraw_normal*c["withdraw_distance"]*smooth(elapsed/c["withdraw_seconds"])
            self.target=E
            solved,self.ik=self.arm.solve(np.linalg.inv(B)@E,self.qcmd,.15)
            target_q=np.r_[solved,self.home[-1]]
            # Completion uses actual displacement and contact, not just time.
            actual=B@self.arm.fk(q)
            separation=float(np.dot(actual[:3,3]-self.E0[:3,3],self.withdraw_normal))
            if elapsed>=c["withdraw_seconds"] and separation>=.10 and not door_contact:
                self.change("compact_fold",t,q,base)
        elif self.phase=="compact_fold":
            a=smooth(elapsed/c["fold_seconds"]);target_q=(1-a)*self.q0+a*self.home
            # Fold while continuing straight west. Turning starts only after
            # the arm has folded, so the earlier wrist contact is not swept
            # around the panel by a simultaneous base rotation.
            straight_room=base[0]-(c["corridor_x"]+c["first_radius"]+.05)
            v=min(.14,max(0.,straight_room))
            if elapsed>=c["fold_seconds"] and max(abs(q[:6]-self.home[:6]))<.05:
                goal=self.curve.at(math.pi/2)[:3]
                self.route=rounded_route(base,goal,c);self.route_index=0
                self.change("compact_bypass",t,q,base)
        elif self.phase=="compact_bypass":
            # Track the fixed nominal-90 path, no hidden side translation.
            distances=np.linalg.norm(self.route[:,:2]-base[:2],axis=1)
            lo=self.route_index;hi=min(len(self.route),lo+100)
            self.route_index=lo+int(np.argmin(distances[lo:hi]))
            look=min(len(self.route)-1,self.route_index+15)
            goal=self.route[look];delta=goal[:2]-base[:2]
            error=wrap(math.atan2(delta[1],delta[0])-base[2])
            v=c["max_speed"]*max(0,math.cos(error))
            curvature=2*math.sin(error)/max(np.linalg.norm(delta),.03)
            v=min(v,c["max_yaw_rate"]/max(abs(curvature),1e-6))
            w=v*curvature
            remaining=float(np.linalg.norm(self.route[-1,:2]-base[:2]))
            if self.route_index>len(self.route)-30:
                v,w,reached=forward_pose(base,self.route[-1],.09,c["max_yaw_rate"])
                if reached:self.change("compact_rendezvous",t,q,base)
        elif self.phase=="compact_rendezvous":
            goal=self.curve.at(math.pi/2)[:3]
            v,w,reached=forward_pose(base,goal,c.get("rendezvous_speed",.10),.4)
            # Deploy during the final straight approach. The joint endpoint
            # is computed at the fixed rendezvous, so it does not ask the arm
            # to reach the door from the more distant staging point.
            progress=elapsed/c["deploy_seconds"]
            if c.get("deploy_by_distance",False):
                # Stage-1 candidate: deploy only as the measured base advances
                # along the fixed nominal approach. No door-angle replanning.
                direction=np.array([math.cos(goal[2]),math.sin(goal[2])])
                length=float(np.dot(goal[:2]-self.base0[:2],direction))
                travelled=float(np.dot(base[:2]-self.base0[:2],direction))
                spatial=np.clip(travelled/max(length,.008),0.,1.)
                progress=min(progress,spatial)
            a=smooth(progress)
            target_q=(1-a)*self.q0+a*np.r_[self.hover_q,0.]
            self.target=self.curve.tool(math.pi/2,c["hover_distance"])
            self.ik=self.hover_ik
            if reached and self.v<.01 and max(abs(q[:6]-self.hover_q))<.04:
                self.seek_offset=c["hover_distance"]
                self.change("compact_seek",t,q,base)
        elif self.phase=="compact_seek":
            self.seek_offset=max(c["hover_distance"]-c["seek_travel"],
                                 self.seek_offset-c["seek_speed"]*dt)
            E=self.curve.tool(math.pi/2,self.seek_offset);self.target=E
            solved,self.ik=self.arm.solve(np.linalg.inv(B)@E,self.qcmd)
            target_q=np.r_[solved,0.]
            # Nominal reach ends even without contact. Early closing may
            # traverse empty space, as explicitly requested by the user.
            if end_contact or self.seek_offset<=0:
                self.close_parameter=math.pi/2
                self.change("close",t,q,base)
        elif self.phase=="close":
            # Fixed nominal progress may move before first contact. Only
            # a lagging physical door can slow progress, avoiding an advance
            # through a stalled panel. The path/endpoint never changes.
            self.seen_closing_contact |= bool(end_contact)
            rate=c["closing_rate"] if self.close_parameter>math.radians(4) else .018
            if theta-self.close_parameter>c["closing_lag_limit"]:
                rate=0.
            if self.close_parameter<=0:rate=0.
            self.close_parameter=max(0.,self.close_parameter-rate*dt)
            ref=self.curve.at(self.close_parameter)
            error=np.array([[math.cos(base[2]),math.sin(base[2])],
                            [-math.sin(base[2]),math.cos(base[2])]])@(ref[:2]-base[:2])
            v=float(np.clip(ref[6]*rate+1.2*error[0],0,c.get("closing_max_speed",.13)))
            yaw_limit=c.get("closing_max_yaw_rate",.3)
            w=float(np.clip(-ref[5]*rate+2*error[1]+1.8*wrap(ref[2]-base[2]),-yaw_limit,yaw_limit))
            E=self.curve.tool(self.close_parameter);self.target=E
            solved,self.ik=self.arm.solve(np.linalg.inv(B)@E,self.qcmd)
            target_q=np.r_[solved,0.]
            if abs(theta)<=math.radians(1) and abs(theta_dot)<=math.radians(1):
                v=w=0.
                # Do not continue increasing compression during latch hold.
                target_q=self.qcmd.copy()
        elif self.phase=="release":
            E=self.E0.copy()
            E[:3,3]-=np.array([math.cos(theta),math.sin(theta),0])*c["release_distance"]*smooth(elapsed/c["release_seconds"])
            self.target=E
            solved,self.ik=self.arm.solve(np.linalg.inv(B)@E,self.qcmd,.5)
            target_q=np.r_[solved,0.]
            if elapsed>=c["release_seconds"] and not door_contact:
                self.change("retract",t,q,base)
        elif self.phase=="retract":
            a=smooth(elapsed/c["retract_seconds"]);target_q=(1-a)*self.q0+a*self.home
            if elapsed>=c["retract_seconds"] and max(abs(q[:6]-self.home[:6]))<.04:
                self.change("compact_exit_turn",t,q,base)
        elif self.phase=="compact_exit_turn":
            # Face north first, then leave the closed slab with a short
            # forward-left arc before the final heading correction.
            err=wrap(math.pi/2-base[2]);w=float(np.clip(2*err,-.4,.4))
            if abs(err)<.008:self.change("compact_exit_arc",t,q,base)
        elif self.phase=="compact_exit_arc":
            remaining=wrap(math.pi-base[2])
            if remaining>.008:
                w=min(.35,2*remaining);v=.25*w
            else:self.change("restore_heading",t,q,base)
        elif self.phase=="restore_heading":
            err=wrap(self.final_yaw-base[2]);w=float(np.clip(2*err,-.4,.4))
            if abs(err)<.003:self.change("final_hold",t,q,base)
        elif self.phase!="final_hold":
            raise ValueError("Unknown compact phase "+self.phase)
        # Rate-limit joint commands and base velocity. Acceleration limiting
        # must not keep driving into a closed door during the latch hold.
        if self.phase in ("close","release") and abs(theta)<=math.radians(1):
            self.v=0. if v==0 else min(v,self.v+c["linear_acceleration"]*dt)
        else:
            self.v=float(np.clip(v,self.v-c["linear_acceleration"]*dt,
                                self.v+c["linear_acceleration"]*dt))
        self.v=max(0.,self.v)
        self.w=float(np.clip(w,self.w-1.2*dt,self.w+1.2*dt))
        target_q[:6]=np.clip(target_q[:6],self.arm.lo,self.arm.hi)
        self.qcmd+=np.clip(target_q-self.qcmd,-c["joint_speed"]*dt,c["joint_speed"]*dt)
        self.qcmd[:6]=np.clip(self.qcmd[:6],self.arm.lo,self.arm.hi)
        self.qcmd[-1]=np.clip(self.qcmd[-1],self.home[-1],0)
        action=np.r_[self.v,self.w,self.qcmd]
        if not np.isfinite(action).all():raise ValueError("Nonfinite compact command")
        return action,dict(phase=self.phase,elapsed_s=t-self.since,ik=self.ik,
            target_world=None if self.target is None else (self.scene.world_from_reference@self.target).tolist(),
            route_index=self.route_index,door_reference=self.close_parameter,
            seek_offset=self.seek_offset,expert_feedback_only=True)
