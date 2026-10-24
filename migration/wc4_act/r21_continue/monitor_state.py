"""CPU-testable active-training staleness clock; resource checks remain independent."""
class ProgressWatch:
    def __init__(self, stale_seconds=600, failure_limit=6):
        self.stale_seconds=stale_seconds;self.failure_limit=failure_limit
        self.previous=None;self.pending_since=None;self.failures=0;self.was_paused=False
    def observe(self, now, local, remote=None, online=True, paused=False):
        if online:self.failures=0
        else:self.failures+=1
        # Normal evaluation and resume transition discard the old training clock.
        if paused or self.was_paused!=paused:self.pending_since=None
        self.was_paused=paused
        advanced=online and (self.previous is None or remote>self.previous)
        if advanced:self.previous=remote;self.pending_since=None
        if not paused and online and local>remote+100:
            if self.pending_since is None:self.pending_since=now
        else:self.pending_since=None
        age=0 if self.pending_since is None else now-self.pending_since
        return dict(stale=age>self.stale_seconds,api_failed=self.failures>=self.failure_limit,
                    active_pending_age_s=age,consecutive_failures=self.failures,paused=paused)
