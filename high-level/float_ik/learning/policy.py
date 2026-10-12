"""Project DP model, independent raw-episode dataset, versioned checkpoint loader."""
import sys
import time
from pathlib import Path
from collections import deque
import numpy as np
import torch
from diffusers import DDPMScheduler
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'dp'))
from models.door_diffusion_policy import DoorDiffusionPolicy

from schema import SCHEMA, contract, model_config
CONFIG = model_config(SCHEMA)  # Historical import compatibility.


def scheduler():
    return DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', prediction_type='epsilon', clip_sample=False)


def images_to_torch(images, device):
    # B,T,4,H,W -> four B,T,3,H,W tensors; masks/depth preserve original encoder input.
    return [torch.as_tensor(images[:, :, i], device=device).unsqueeze(2).expand(-1, -1, 3, -1, -1) for i in range(4)]


class Policy:
    def __init__(self, checkpoint, device='cuda:0', seed=0, profile=False, inference_steps=25, schema=None):
        if inference_steps not in (25, 100):
            raise ValueError('Audited inference_steps must be 25 or 100')
        ck = torch.load(checkpoint, map_location='cpu')
        self.schema = ck['schema']
        self.action_names, self.state_names = contract(self.schema)
        self.config = model_config(self.schema)
        if (schema is not None and self.schema != schema) or ck['model_config'] != self.config or ck['action_names'] != self.action_names or ck['state_names'] != self.state_names:
            raise ValueError('Incompatible floating-base action/observation schema')
        self.device = device
        self.profile = profile
        self.last_timing = {}
        self.model = DoorDiffusionPolicy(**self.config).to(device).eval()
        self.model.load_state_dict(ck['model'])
        self.stats = {k: v.to(device) for k, v in ck['stats'].items()}
        self.scheduler = scheduler()
        self.inference_steps = inference_steps
        self.scheduler.set_timesteps(inference_steps, device='cpu')
        with torch.no_grad():
            self.time_embeddings = [self.model.timestep_encoder(t.to(device).reshape(1)) for t in self.scheduler.timesteps]
        self.history = deque(maxlen=4)
        self.queue = deque()
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.step = ck['step']

    @torch.no_grad()
    def act(self, obs):
        if np.asarray(obs['state']).shape != (len(self.state_names),):
            raise ValueError('Observation does not match checkpoint schema')
        self.history.append(obs)
        while len(self.history) < 4:
            self.history.appendleft(obs)
        if self.queue:
            return self.queue.popleft(), False
        started = time.perf_counter()
        state = torch.as_tensor(np.stack([x['state'] for x in self.history])[None], device=self.device)
        state = (state-self.stats['state_mean'])/self.stats['state_std']
        images = images_to_torch(np.stack([x['images'] for x in self.history])[None], self.device)
        cond_obs = self.model.encode_obs(state, *images)
        if self.profile:
            torch.cuda.synchronize(self.device)
        encoded = time.perf_counter()
        sample = torch.randn((1, self.config['pred_horizon'], self.config['action_dim']), device=self.device, generator=self.generator)
        for t, time_embedding in zip(self.scheduler.timesteps, self.time_embeddings):
            cond = torch.cat([cond_obs, time_embedding], -1)
            x = self.model.input_proj(sample.transpose(1, 2))
            for block in self.model.blocks:
                x = block(x, cond)
            noise = self.model.output_proj(x).transpose(1, 2)
            sample = self.scheduler.step(noise, t, sample, generator=self.generator).prev_sample
        actions = (sample[0]*self.stats['action_std']+self.stats['action_mean']).cpu().numpy()
        if self.profile:
            self.last_timing = dict(preprocess_encode_sec=encoded-started, denoise_decode_sec=time.perf_counter()-encoded)
        self.queue.extend(actions[:4])
        return self.queue.popleft(), True
