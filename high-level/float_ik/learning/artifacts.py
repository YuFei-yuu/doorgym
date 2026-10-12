"""Portable paths and strict JSON for current Cartesian analysis tools."""
import json
from pathlib import Path
import numpy as np


def resolve_recorded_path(root, recorded):
    root = Path(root).resolve()
    path = Path(recorded)
    if path.is_absolute() and root.name in path.parts:
        # Prefer this run's local mirror over an unrelated original server path.
        i = len(path.parts) - 1 - list(reversed(path.parts)).index(root.name)
        candidate = root.joinpath(*path.parts[i + 1:])
        if candidate.exists():
            return candidate
    candidate = path if path.is_absolute() else root / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError('Missing artifact in %s: %s' % (root, recorded))


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(finite_json(value), f, indent=2, allow_nan=False)
        f.write('\n')
