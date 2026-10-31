"""Keep a completed update's original metrics before honoring a safe stop."""
class MetricBoundary:
    def __init__(self,boundary):
        self.boundary=boundary;self.logged=0
        self.original_check=boundary.checkpoint_if_requested
        boundary.checkpoint_if_requested=self.check
    def check(self):
        if self.boundary.completed>self.logged:return
        return self.original_check()
    def restored(self,step):self.logged=step
    def after_metrics(self,step):
        assert step==self.boundary.completed and not self.boundary.inflight
        self.logged=step
        return self.original_check()
