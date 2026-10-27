"""Measured-start withdrawal and folding concurrent with unchanged base traversal."""
import numpy as np
from close_inplace_stage4.common import FullPose,baseT
from close_inplace_stage2.candidate import smooth
class EarlyArm:
    def __init__(self,robot,B,q,last_command,previous_command,t,dt,home):
        self.S=FullPose(robot);self.t0=t;self.dt=dt;self.home=np.array(home);self.start=np.array(last_command);self.cmd=self.start.copy()
        self.initial_velocity=(self.start-np.array(previous_command))/dt
        F=baseT(B)@self.S.fk(q);P=F[:3,3]+F[:3,:3]@self.S.pad
        qw,info=self.S.solve_point(B,P+np.array([0,.10,0]),None,q,120)
        self.withdraw=np.r_[qw,self.home[-1]]
        # Analytic duration from displacement, not a parameter search.
        self.tw=max(1.4,1.875*float(np.max(abs(self.withdraw[:6]-self.start[:6])))/.60)
        self.tf=max(4.8,1.875*float(np.max(abs(self.withdraw[:6]-self.home[:6])))/.60)
        self.plan=dict(start_time_s=t,withdraw_seconds=self.tw,fold_seconds=self.tf,finish_reference_time_s=t+self.tw+self.tf,
            measured_start_q=np.array(q).tolist(),command_start=self.start.tolist(),withdraw_command=self.withdraw.tolist(),withdraw_goal_at_start_base=(P+[0,.10,0]).tolist(),ik=info,
            meaning='Joint targets retreat relative to moving base then fold; start measured, no synthetic state substitution')
        self.latest={};self.actual_home_time=None
    def command(self,t,q,contact):
        e=t-self.t0+self.dt
        if e<=self.tw:
            u=float(np.clip(e/self.tw,0,1));s=float(np.clip(smooth(u),0,1))
            # Quintic Hermite: match previous command velocity, end with zero velocity/acceleration.
            tangent=self.tw*self.initial_velocity*(u-6*u**3+8*u**4-3*u**5)
            target=(1-s)*self.start+s*self.withdraw+tangent;stage='parallel_withdraw'
        else:
            u=float(np.clip((e-self.tw)/self.tf,0,1));s=float(np.clip(smooth(u),0,1))
            target=(1-s)*self.withdraw+s*self.home;stage='parallel_fold' if u<1 else 'parallel_home'
        delta=target-self.cmd;requested=delta/self.dt
        self.cmd+=np.clip(delta,-.7*self.dt,.7*self.dt)
        if stage=='parallel_home' and np.max(abs(np.array(q)[:6]-self.home[:6]))<.03 and not contact and self.actual_home_time is None:self.actual_home_time=t
        self.latest=dict(stage=stage,elapsed_s=e,reference_complete=e>=self.tw+self.tf,actual_home_time_s=self.actual_home_time,requested_rate_max=float(np.max(abs(requested[:6]))),rate_clipped=bool(np.max(abs(requested[:6]))>.700001))
        return self.cmd.copy()
