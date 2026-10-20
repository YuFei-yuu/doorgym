"""Pure control/evidence rules for round two; no simulator or learned policy."""
import math
import numpy as np

GEOMETRY_EPS_M = 1e-5  # numeric uncertainty band, never a penetration allowance
FORCE_ZERO_EPS = 0.0  # only an exactly zero logged force is treated as unloaded


def door_contact_review_flags(phase, raw_lambda, robot_to_door, end_to_panel):
    """Diagnostic evidence, never a contact-triggered abort decision.

    A nonzero SDK value is not a calibrated force or proof of who moved the
    door. Opening contacts must not be judged by the new closing requirement.
    """
    if not np.isfinite(raw_lambda):
        raise ValueError("Nonfinite contact record")
    nonzero = raw_lambda != 0
    return {
        "nonzero_door_contact": bool(robot_to_door and nonzero),
        "nonzero_end_panel_contact": bool(end_to_panel and nonzero),
        "off_target_closing_contact": bool(
            phase == "close" and robot_to_door and not end_to_panel and nonzero
        ),
    }


def classify_contact(force, signed_gap, expected=False, reliable=False):
    """A separating-axis gap >0 proves separation; <=0 indicates hull overlap.

    Never drop zero-force rows. In particular, zero force plus overlap stops.
    Expected operation contacts still cannot prove successful closing alone.
    """
    if not np.isfinite(force):
        return 'uncertain_stop'
    if not expected and abs(force)>FORCE_ZERO_EPS:
        return 'unexpected_loaded_stop'
    if not reliable or signed_gap is None or not np.isfinite(signed_gap):
        return 'uncertain_stop'
    if signed_gap < -GEOMETRY_EPS_M:
        return 'penetration_stop'
    if signed_gap > GEOMETRY_EPS_M:
        return 'loaded_separated' if abs(force)>FORCE_ZERO_EPS else 'proximity_separated'
    if expected:
        return 'expected_loaded_touch' if abs(force)>FORCE_ZERO_EPS else 'expected_unloaded_touch'
    return 'uncertain_boundary_stop'


def smooth_gripper(initial, home, time_s, duration_s=1.):
    t=float(np.clip(time_s/duration_s,0,1));s=t**3*(10-15*t+6*t*t)
    return (1-s)*float(initial)+s*float(home)


def timing(step,dt):
    return dict(control_step=step,command_time_s=step*dt,observation_pre_time_s=step*dt,
                measured_post_time_s=(step+1)*dt,record_25hz=(step%2==0),
                frame_25hz_index=step//2 if step%2==0 else None)


class ClosingEvidence:
    """Accumulate aligned physical contact, torque sign, work and door motion.

    No latch from an incidental contact row followed later by angle decline.
    All force evidence must belong to the same measured integration interval.
    """
    def __init__(self,dt):
        self.dt=dt;self.valid_steps=0;self.angle_with_support=0.;self.work=0.
        self.unexplained_motion=0.;self.closed_duration=0.;self.last=None

    def update(self,pre_angle,post_angle,post_speed,contacts,external_hinge_effort,
               end_displacement,unlocked,no_nonend_contact):
        delta=float(post_angle-pre_angle)
        supported=[c for c in contacts if c.get('end_panel') and c.get('geometry_reliable')
                   and c.get('force',0)>FORCE_ZERO_EPS and c.get('hinge_torque',0)<0
                   and c.get('closing_velocity',0)>0 and c.get('surface_gap',1)<=.04
                   and c.get('surface_gap',-1)>=-GEOMETRY_EPS_M]
        valid=bool(supported and external_hinge_effort==0 and unlocked and no_nonend_contact
                   and delta<0 and end_displacement>0)
        torque=sum(c['hinge_torque'] for c in supported)
        if valid:
            self.valid_steps+=1;self.angle_with_support-=delta;self.work+=torque*delta
        elif delta<-1e-5:
            self.unexplained_motion-=delta
        physically_closed=abs(post_angle)<=math.radians(1) and abs(post_speed)<=math.radians(1)
        self.closed_duration=self.closed_duration+self.dt if physically_closed else 0.
        self.last=dict(valid_support=valid,torque=torque,delta_angle=delta,closed=physically_closed)
        return self.last

    def can_latch(self,total_angle_change,far_side,unlocked):
        return bool(far_side and unlocked and self.closed_duration+1e-8>=.5
                    and self.valid_steps*self.dt>=.5 and self.work>0
                    and self.angle_with_support>=math.radians(45)
                    and self.angle_with_support>=.8*total_angle_change
                    and self.unexplained_motion<=math.radians(2))
