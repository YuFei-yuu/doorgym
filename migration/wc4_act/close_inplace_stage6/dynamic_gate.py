"""Sustained measured loaded contact + closing motion + pose tracking.

No arm-rest test here. Final/home/R11 rest requirements are unchanged.
"""
class DynamicContactGate:
    def __init__(self):
        self.start=None;self.start_angle=None;self.samples=0;self.latest={}
    def observe(self,t,theta,omega,loaded,position_error,orientation_error,ready=True):
        good=bool(ready and loaded and omega<-.002 and position_error<=.008 and orientation_error<=.01)
        if not good:self.start=None;self.start_angle=None;self.samples=0
        else:
            if self.start is None:self.start=t;self.start_angle=theta
            self.samples+=1
        span=0. if self.start is None else t-self.start
        motion=0. if self.start_angle is None else self.start_angle-theta
        accepted=bool(good and span>=.30 and motion>=.003 and self.samples>=2)
        self.latest=dict(accepted=accepted,instantaneous_valid=good,loaded_contact=bool(loaded),
            observed_span_s=span,observed_closing_rad=motion,samples=self.samples,actual_door_speed=omega,
            position_error_m=position_error,orientation_error_rad=orientation_error,
            rule='continuous loaded contact and closing speed; >=0.30s, >=0.003rad; pose 8mm/0.01rad; no arm-rest requirement')
        return accepted
