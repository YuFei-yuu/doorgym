"""Fixed time progress, analogous to author step/duration trajectory_targets.
Reference angle is a path parameter, never a written or inferred door state.
"""
import math
import numpy as np
from close_inplace_stage2.candidate import smooth
DURATIONS=(3.3,9.4,12.4)
TOTAL=sum(DURATIONS)
def progress(t,duration,ramp):
    t=float(np.clip(t,0,duration))
    integral=lambda u:2.5*u**4-3*u**5+u**6
    if t<ramp:return ramp*integral(t/ramp)/(duration-ramp)
    if t>duration-ramp:return 1-ramp*integral((duration-t)/ramp)/(duration-ramp)
    return (t-ramp/2)/(duration-ramp)
def reference(t,initial_angle):
    a,b,c=DURATIONS
    turn=progress(t-a,b,1.5)
    follow=progress(t-a-b,c,1.0)
    if t<a:theta=initial_angle+(math.radians(70)-initial_angle)*float(np.clip(smooth(t/a),0,1))
    elif t<a+b:theta=math.radians(70-35*turn)
    else:theta=math.radians(35*(1-follow))
    return dict(path_angle_rad=float(theta),base_yaw=1.5*math.pi+math.pi/2*turn,
        follow_m=.45*follow,turn_progress=turn,follow_progress=follow,
        elapsed_s=float(t),reference_complete=bool(t>=TOTAL),total_s=TOTAL)
