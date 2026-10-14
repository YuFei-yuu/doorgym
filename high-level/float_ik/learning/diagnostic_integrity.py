"""Run ownership, frozen dependencies, and portable diagnostic delivery."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from functools import wraps


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def process_identity(pid):
    try:
        fields = Path('/proc/%d/stat' % int(pid)).read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, ValueError, IndexError, TypeError):
        return None


def assert_no_active(root):
    path = Path(root)/'active_command.json'
    if not path.exists():
        return
    record = json.loads(path.read_text())
    pid = record.get('pid')
    actual = process_identity(pid)
    if actual is not None and (record.get('proc_start_ticks') is None or actual == str(record['proc_start_ticks'])):
        raise RuntimeError('Diagnostic child is still active; no result may be finalized')
    supervisor = record.get('supervisor_pid')
    if supervisor and int(supervisor) != os.getpid() and process_identity(supervisor) is not None:
        raise RuntimeError('Prior supervisor is still active; inspect its command before resuming')


@contextmanager
def phase_lock(root, phase):
    root = Path(root)
    with (root/'diagnostics.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another diagnostic phase owns this run') from exc
        try:
            assert_no_active(root)
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_paths(repo, recorded=()):
    repo = Path(repo)
    paths = {repo/name for name in recorded}
    # Include every local helper imported by any phase, including acceptance
    # logic and the shared training/inference contract. No image/frame cache.
    for relative in ['high-level/float_ik/learning', 'high-level/dp']:
        paths.update((repo/relative).rglob('*.py'))
    paths.update([repo/'high-level/float_ik/learning/run_tools/activate_train.sh',
                  repo/'.repro/blackwell/activate.sh'])
    return sorted(p for p in paths if p.is_file())


def verify_frozen(root, repo):
    plan = json.loads((Path(root)/'plan.json').read_text())
    for name, expected in plan['source_hashes'].items():
        if sha256(Path(repo)/name) != expected:
            raise ValueError('Frozen diagnostic dependency changed: '+name)
    return plan


def frozen_phase(function):
    @wraps(function)
    def wrapped(root, *args, **kwargs):
        root = Path(root).resolve()
        with phase_lock(root, function.__name__):
            verify_frozen(root, Path(__file__).resolve().parents[3])
            return function(root, *args, **kwargs)
    return wrapped


def materialize_external_links(root):
    """Copy this delivery's external log targets, preserving link provenance."""
    root = Path(root).resolve()
    records = []
    for path in sorted(root.rglob('*')):
        if not path.is_symlink():
            continue
        target = path.resolve(strict=True)
        try:
            target.relative_to(root)
        except ValueError:
            if not target.is_file():
                raise ValueError('External directory links cannot be delivered: '+str(path))
            link = os.readlink(path)
            expected = sha256(target)
            temporary = path.with_name(path.name+'.materializing')
            with temporary.open('xb') as stream, target.open('rb') as source:
                shutil.copyfileobj(source, stream)
            if sha256(temporary) != expected:
                temporary.unlink()
                raise RuntimeError('External log changed while copying: '+str(path))
            os.replace(temporary, path)
            records.append(dict(path=str(path.relative_to(root)), original_target=link, sha256=expected))
    return records
