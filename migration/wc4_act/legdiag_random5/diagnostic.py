"""The former sustained-leg abort condition, now events only."""
class LegDiagnostic:
    def __init__(self):
        self.active=False;self.first=None;self.events=[];self.qualified_steps=0
    def observe(self,step,t,bad_since,bad,deviations):
        qualified=bad_since is not None and t-bad_since>=.10-1e-9
        if qualified:self.qualified_steps+=1
        event=None
        if qualified and not self.active:
            event=dict(diagnostic_step=int(step),physical_post_time_s=float(t),
                first_bad_time_s=float(bad_since),bad_joints=list(bad),deviations=dict(deviations),
                meaning='diagnostic only; no interruption or control change')
            self.events.append(event)
            if self.first is None:self.first=event
        self.active=qualified
        return event
    def metadata(self):
        return dict(triggered=self.first is not None,first_event=self.first,events=self.events,
                    qualified_steps=self.qualified_steps,abort_enabled=False)
