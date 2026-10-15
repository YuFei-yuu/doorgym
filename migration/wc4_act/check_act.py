#!/usr/bin/env python3
"""ACT import/config and converter parser check. No dataset/model construction."""
import importlib.metadata as metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(ROOT / 'high-level/dp'))
from commands import build
import lerobot
import draccus
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    expected = ROOT / 'high-level/lerobot/src'
    assert expected in Path(lerobot.__file__).parents
    converter = load('wc4_converter', 'high-level/dp/convert_door_raw_to_lerobot.py')
    cmd = build('convert', 'preparation-command-check')['argv']
    sys.argv = cmd[1:]
    converted = converter.parse_args()
    assert converted.state_action_mode == 'joint_state9' and converted.depth_only
    cmd = build('train', 'preparation-command-check', gpu=1, train_steps=50000)['argv']
    # Resolve the ACT subclass exactly as the author's wrapped train CLI does.
    cfg = draccus.parse(TrainPipelineConfig, args=cmd[2:])
    assert isinstance(cfg.policy, ACTConfig)
    assert cfg.policy.chunk_size == 100 and cfg.policy.n_action_steps == 50
    assert not cfg.policy.use_action_loss_weight and not cfg.policy.interaction_state_conditioning
    assert cfg.wandb.mode == 'online' and cfg.wandb.enable
    exporter = load('wc4_exporter', 'high-level/dp/export_official_lerobot_act_to_door_checkpoint.py')
    sys.argv = build('export', 'preparation-command-check')['argv'][1:]
    exported = exporter.parse_args()
    assert exported.depth_only and exported.action_horizon == 25 and exported.device == 'cpu'
    print(json.dumps({'python': sys.executable, 'lerobot': lerobot.__file__,
        'act_config': inspect.getfile(ACTConfig), 'act_model': inspect.getfile(ACTPolicy),
        'versions': {n: metadata.version(n) for n in ['lerobot', 'torch', 'torchvision', 'numpy', 'datasets', 'av', 'wandb']},
        'train_config': cfg.to_dict(), 'converter_args': vars(converted),
        'exporter_args': vars(exported)}, indent=2, default=str))


if __name__ == '__main__':
    main()
