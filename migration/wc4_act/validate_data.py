#!/usr/bin/env python3
"""Read-only gates for future runs; never repair or rewrite recorded data."""
import json
from pathlib import Path
import sys
import numpy as np

C = json.loads(Path(__file__).with_name('config.json').read_text())


def validate(kind, root):
    sidecar = json.loads((root / 'door_dp_feature_names.json').read_text())
    assert sidecar['action'] == C['action_names'], sidecar
    assert sidecar.get('vision_mode') == 'depth_only', sidecar
    if kind == 'raw':
        files = sorted(root.glob('episode_*.npz'))
        assert len(files) == 50, len(files)
        frames = 0
        for path in files:
            with np.load(path, allow_pickle=False) as d:
                state, action = d['state'], d['action']
                assert state.ndim == 2 and state.shape[1] == 9, (path, state.shape)
                assert action.shape == state.shape, (path, action.shape)
                assert np.isfinite(state).all() and np.isfinite(action).all(), path
                assert int(d['fps'].item()) == 25, path
                for key in ('front_masked_depth', 'wrist_masked_depth'):
                    assert d[key].shape[0] == len(state) and d[key].shape[-2:] == (480, 640), (path, key)
                assert str(d['camera_intrinsics_mode'].item()) == 'legacy', path
                frames += len(state)
        print(json.dumps({'kind': kind, 'episodes': len(files), 'frames': frames, 'state_action_dim': 9}))
    elif kind == 'lerobot':
        info = json.loads((root / 'meta/info.json').read_text())
        assert info['total_episodes'] == 50 and info['fps'] == 25, info
        features = info['features']
        for key in ('observation.state', 'action'):
            assert features[key]['shape'] == [9], features[key]
        for view in ('front', 'wrist'):
            assert 'observation.images.' + view + '_masked_depth' in features
        assert not any(k.startswith('aux.interaction') for k in features), features
        print(json.dumps({'kind': kind, 'episodes': info['total_episodes'], 'frames': info['total_frames'], 'fps': info['fps']}))
    else:
        raise ValueError(kind)


if __name__ == '__main__':
    validate(sys.argv[1], Path(sys.argv[2]).resolve())
