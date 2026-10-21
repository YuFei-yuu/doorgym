"""Task acceptance/lock authorization from measured state, never expert phase.

The same monitor is the required future policy-evaluation implementation.
It observes truth for scoring/lock abstraction only, not policy inputs.
"""
import math
import numpy as np
from fullcycle_contract import ObservedHold


class TaskMonitor:
    def __init__(self,home,final_yaw,closed_angle=0.):
        self.home=np.asarray(home,float)[:6];self.final_yaw=float(final_yaw)
        self.closed_angle=float(closed_angle);self.passed=False;self.unlocked=False
        self.separated=False;self.push_seen=False;self.close_hold=ObservedHold()
        self.final_hold=ObservedHold();self.previous_angle=None;self.final_valid=False
        self.returned=False

    def observe(self,t,far,angle,speed,unlocked,latched,any_contact,end_loaded,
                q,qd,base_speed,base_yaw_rate,yaw):
        if self.passed and not far:self.returned=True
        self.passed|=bool(far);self.unlocked|=bool(unlocked)
        if self.passed and not any_contact:self.separated=True
        if (self.passed and self.separated and end_loaded and self.previous_angle is not None
                and angle<self.previous_angle):self.push_seen=True
        closed=abs(angle-self.closed_angle)<=math.radians(1) and abs(speed)<=math.radians(1)
        self.close_hold.observe(t,self.passed and self.unlocked and self.push_seen and latched is None and closed)
        err=(yaw-self.final_yaw+math.pi)%(2*math.pi)-math.pi
        self.final_valid=bool(far and not any_contact and closed and latched is not None
            and not unlocked and abs(err)<=math.radians(2) and base_speed<=.01 and base_yaw_rate<=.01
            and np.all(abs(np.asarray(q)[:6]-self.home)<=.03) and np.all(abs(np.asarray(qd)[:6])<=.05))
        self.final_hold.observe(t,self.final_valid);self.previous_angle=float(angle)

    def may_latch(self):return not self.returned and self.close_hold.reached(.5)
    def accepted(self):return not self.returned and self.push_seen and self.final_hold.reached(2.)
    def metadata(self):
        return dict(passed=self.passed,returned=self.returned,unlocked_seen=self.unlocked,separated=self.separated,
            actual_push_seen=self.push_seen,close_span_s=self.close_hold.elapsed,
            final_span_s=self.final_hold.elapsed,final_valid=self.final_valid,accepted=self.accepted())
