"""Measured hinge latch; has no robot state, phase, planner, or motion output."""
import math
from fullcycle_contract import ObservedHold


class MeasuredLock:
    def __init__(self):
        self.opened=False
        self.hold=ObservedHold()

    def observe(self,t,angle,speed,ever_unlocked,latched):
        self.opened |= bool(ever_unlocked and abs(angle)>math.radians(1))
        self.hold.observe(t,bool(ever_unlocked and self.opened and latched is None
            and abs(angle)<=math.radians(1) and abs(speed)<=math.radians(1)))

    def may_latch(self):
        return self.hold.reached(.5)

    def metadata(self):
        return dict(opened=self.opened,closed_observed_span_s=self.hold.elapsed,
            motion_output=False,uses_phase_or_expert=False)
