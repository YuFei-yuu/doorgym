"""Cooperative stop at completed optimizer boundaries and atomic checkpoint commit.

SIGINT/SIGTERM handlers only set a request; they never unwind an in-flight step.
The file request also works when the parent inherited ignored SIGINT (nohup).
"""
import hashlib,json,os,signal,time
from pathlib import Path


def atomic_json(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(str(tmp),str(path))
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def request_stop(directory,reason,**details):
    path=Path(directory)/'STOP_REQUEST.json'
    try:
        # link publishes a complete request atomically, first request wins.
        tmp=path.with_name('.stop-request-'+str(os.getpid())+'-'+str(time.time_ns()))
        atomic_json(tmp,dict(reason=reason,at=time.time(),**details))
        try:os.link(str(tmp),str(path))
        except FileExistsError:pass
        finally:tmp.unlink()
    except OSError:
        raise RuntimeError('Could not persist stop request; no clean stop claim')


class BoundaryStop(Exception):
    pass


class StopBoundary:
    def __init__(self,directory,save_callback):
        self.directory=Path(directory);self.save_callback=save_callback
        self.completed=0;self.inflight=False;self.signal_requested=None
        self.old_handlers={};self.stopped=False
        self.journal=(self.directory/'completed-updates.jsonl').open('x',buffering=1)

    def install_signals(self):
        def handler(signum,frame):
            if self.signal_requested is None:self.signal_requested=signum
        for number in [signal.SIGINT,signal.SIGTERM]:
            self.old_handlers[number]=signal.signal(number,handler)

    def requested(self):
        return self.signal_requested is not None or (self.directory/'STOP_REQUEST.json').exists()

    def checkpoint_if_requested(self):
        if not self.requested():return
        if self.inflight:raise RuntimeError('Cannot publish a checkpoint during an optimizer update')
        if self.signal_requested is not None:
            request_stop(self.directory,'signal',signal=self.signal_requested)
        request=json.loads((self.directory/'STOP_REQUEST.json').read_text())
        receipt=self.save_callback(self.completed)
        atomic_json(self.directory/'STOPPED.json',dict(status='stopped-consistent-boundary',completed_updates=self.completed,
            uncertain_updates=0,request=request,checkpoint=receipt,automatic_resume=False))
        self.stopped=True
        raise BoundaryStop('Stopped at completed update '+str(self.completed))

    def update(self,function,*args,**kwargs):
        self.checkpoint_if_requested()
        self.inflight=True
        atomic_json(self.directory/'update-boundary.json',dict(completed=self.completed,inflight=True,next_update=self.completed+1))
        try:result=function(*args,**kwargs)
        except BaseException as exc:
            # A hard fault halfway through optimizer.step is not a clean boundary.
            atomic_json(self.directory/'UNSAFE_STOP.json',dict(completed=self.completed,inflight=True,
                possible_partial_update=True,error=repr(exc),recovery_claim=False))
            raise
        self.completed+=1;self.inflight=False
        self.journal.write(json.dumps(dict(completed=self.completed,at=time.time()))+'\n')
        self.journal.flush();os.fsync(self.journal.fileno())
        atomic_json(self.directory/'update-boundary.json',dict(completed=self.completed,inflight=False))
        self.checkpoint_if_requested()
        return result

    def close(self):
        self.journal.close()
        for number,handler in self.old_handlers.items():signal.signal(number,handler)


def commit_checkpoint(parent,step,writer):
    """Author serializer writes a private directory; only complete trees publish."""
    parent=Path(parent);parent.mkdir(parents=True,exist_ok=True)
    target=parent/('stop-%06d'%step);staging=parent/('.saving-%06d-%d'%(step,os.getpid()))
    if target.exists() or staging.exists():raise FileExistsError('Refuse checkpoint replacement')
    staging.mkdir()
    writer(staging,step)
    digests={}
    for p in sorted(staging.rglob('*')):
        if p.is_file():
            with p.open('rb') as f:
                h=hashlib.sha256()
                for block in iter(lambda:f.read(8388608),b''):h.update(block)
                os.fsync(f.fileno())
            digests[str(p.relative_to(staging))]=h.hexdigest()
    atomic_json(staging/'COMMITTED.json',dict(step=step,files=digests,atomic_publish=True))
    os.rename(str(staging),str(target))
    fd=os.open(str(parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(target),step=step,manifest='COMMITTED.json')
