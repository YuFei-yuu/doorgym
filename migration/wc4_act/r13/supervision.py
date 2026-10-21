"""Supervisor requests a checkpointed stop, never kills an in-flight trainer."""
import time,subprocess
from pathlib import Path
from stopping import request_stop,atomic_json


def supervise(process,monitor,run,poll_s=1.,notice_after_s=120.):
    run=Path(run);requested_at=None
    while process.poll() is None:
        if monitor.poll() is not None and not (run/'training-complete.json').exists():
            request_stop(run,'monitor exited',returncode=monitor.returncode)
        if (run/'STOP_REQUEST.json').exists():
            if requested_at is None:requested_at=time.monotonic()
            if time.monotonic()-requested_at>notice_after_s and not (run/'STOP_TIMEOUT.json').exists():
                atomic_json(run/'STOP_TIMEOUT.json',dict(status='request-not-yet-acknowledged',
                    clean_stop_claim=False,forced_kill=False,lease_must_remain_held=True))
        time.sleep(poll_s)
    code=process.wait()
    if code==75 and not (run/'STOPPED.json').exists():raise RuntimeError('Stopped exit without committed state')
    return code
