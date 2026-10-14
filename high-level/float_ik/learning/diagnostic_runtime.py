"""Compare real Gym-runtime CPU controller against training-runtime offline evidence."""
from isaacgym import gymapi  # must precede Torch in the Gym environment
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import diffusers
from artifacts import write_new_json
from forward_diagnostics import IMAGE_KEYS, digest
from diagnostic_integrity import frozen_phase


@frozen_phase
def run(root):
    original = Path(json.loads((root/'plan.json').read_text())['original'])
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'dp'))
    from door_dp_common import DoorDPPolicyController
    torch.set_num_threads(2)
    controller = DoorDPPolicyController(original/'training/checkpoints/step_003000.pt', device='cpu')
    specs = json.loads((root/'offline/sampling_results.json').read_text())
    predicted = np.load(root/'offline/predictions.npz')['prediction']
    checks = []
    with np.load(original/'dataset/val_seed_16108/trajectory.npz') as episode:
        images = [episode[k] for k in IMAGE_KEYS]
        state = episode['state']
        for index, spec in enumerate(specs):
            if spec['step'] != 3000 or spec['seed'] != 16108 or spec['sampling_seed'] != 42 or spec['phase'] not in ['start', 'grasp', 'through']:
                continue
            controller.obs_buffer.clear(); controller.action_queue.clear()
            ids = np.maximum(np.arange(spec['center']-15, spec['center']+1), 0)
            for frame in ids:
                controller.append_observation(state[frame], *[x[frame] for x in images])
            torch.manual_seed(42); controller.sample_action_chunk()
            actual = np.asarray(list(controller.action_queue))
            checks.append(dict(phase=spec['phase'], center=spec['center'],
                               max_action_difference=float(abs(actual-predicted[index]).max())))
    write_new_json(root/'offline/gym_runtime_comparison.json', dict(checks=checks, torch_version=torch.__version__,
        diffusers_version=diffusers.__version__, cuda_initialized=torch.cuda.is_initialized(),
        device='cpu', production_controller=True, no_training=True, source_sha256=digest(Path(__file__))))
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    run(Path(parser.parse_args().root))
