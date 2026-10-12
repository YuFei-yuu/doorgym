"""NPZ container adapter; sequence indexing, statistics and normalization are upstream."""
import bisect
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'dp'))
from train_door_dp import DoorDPSequenceDataset, compute_stats, normalize_batch
from door_dp_common import DoorDPPolicyController, dp_image_inputs_from_camera_tensors


class EpisodeFrames:
    def __init__(self, episodes):
        self.episodes = episodes
        self.ends = np.cumsum([len(e['action']) for e in episodes]).tolist()

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        ep = bisect.bisect_right(self.ends, index)
        i = index - (self.ends[ep-1] if ep else 0)
        e = self.episodes[ep]
        names = ('wrist_handle_mask', 'wrist_masked_depth', 'front_handle_mask', 'front_masked_depth')
        out = {'observation.state': e['state'][i], 'action': e['action'][i], 'episode_index': ep}
        out.update({'observation.images.'+name: np.repeat(e['images'][i,j,:,:,None],3,-1) for j,name in enumerate(names)})
        return out


class NPZSequenceDataset(DoorDPSequenceDataset):
    def __init__(self, episodes, obs_horizon, pred_horizon, centers=None):
        self.dataset = EpisodeFrames(episodes)
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.length = len(self.dataset)
        self.episodes = [ep for ep,e in enumerate(episodes) for _ in range(len(e['action']))]
        valid = self._build_indices()
        if centers is not None and not set(centers).issubset(valid):
            raise ValueError('Probe centers outside upstream valid windows')
        self.indices = valid if centers is None else list(centers)
