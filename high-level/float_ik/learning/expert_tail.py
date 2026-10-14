"""Recording duration only; never changes task events or actuator commands."""
import math


class ExpertRecordingTail:
    def __init__(self, seconds, dt):
        if seconds not in (0., 3.) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Expert tail requires 0 or 3 seconds and a positive dt')
        self.seconds, self.dt = float(seconds), float(dt)
        # Isaac Gym stores dt as float32 (50 Hz is slightly below .02).
        ratio = self.seconds / self.dt
        nearest = round(ratio)
        self.steps = int(nearest if math.isclose(ratio, nearest, abs_tol=1e-5) else math.ceil(ratio))
        self.first_pass_step = None
        self.last_step = None

    def observe(self, step, pass_done):
        if self.last_step is not None and step != self.last_step + 1:
            raise ValueError('Expert recording steps must be contiguous')
        self.last_step = int(step)
        if pass_done and self.first_pass_step is None:
            self.first_pass_step = int(step)

    @property
    def complete(self):
        return (self.first_pass_step is not None and
                self.last_step - self.first_pass_step >= self.steps)

    @property
    def recorded_seconds(self):
        return 0. if self.first_pass_step is None else (self.last_step-self.first_pass_step)*self.dt

    def require_complete(self):
        if self.seconds and not self.complete:
            raise RuntimeError('Expert post-pass recording incomplete: first_pass=%s, last_step=%s, required_tail_steps=%d' %
                               (self.first_pass_step, self.last_step, self.steps))
