#!/usr/bin/env python3
"""Bounded DP training on successful raw episodes, with exact RNG restoration."""
import argparse
import json
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from schema import ACTION_NAMES, STATE_NAMES
from policy import CONFIG, SCHEMA, DoorDiffusionPolicy, images_to_torch, scheduler

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--output', required=True)
p.add_argument('--steps', type=int, required=True)
p.add_argument('--resume')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--seconds', type=float, default=3000)
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
(root/'checkpoints').mkdir(exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
random.seed(a.seed)
rng = np.random.RandomState(a.seed)
validation_rng = np.random.RandomState(98765)
entries = json.loads(Path(a.manifest).read_text())
data = {'train': [], 'val': []}
for item in entries:
    if item['split'] not in data or not item['success']:
        continue
    with np.load(item['episode']) as z:
        episode = {k: z[k] for k in ('state', 'images', 'action')}
    if len(episode['action']) >= 20:
        data[item['split']].append(episode)
assert len(data['train']) >= 8 and len(data['val']) >= 2, {k: len(v) for k,v in data.items()}
state = torch.from_numpy(np.concatenate([e['state'] for e in data['train']]))
action = torch.from_numpy(np.concatenate([e['action'] for e in data['train']]))
stats = dict(state_mean=state.mean(0), state_std=state.std(0).clamp_min(.01), action_mean=action.mean(0), action_std=action.std(0).clamp_min(.01))
model = DoorDiffusionPolicy(**CONFIG).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
noise_scheduler = scheduler()
step = 0
elapsed_prior = 0.
if a.resume:
    ck = torch.load(a.resume, map_location='cpu')
    assert ck['schema'] == SCHEMA and ck['manifest'] == entries
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    stats = ck['stats']
    step = ck['step']
    elapsed_prior = ck['elapsed_sec']
    rng.set_state(ck['sampler_rng'])
    random.setstate(ck['python_rng'])
    np.random.set_state(ck['numpy_rng'])
    torch.set_rng_state(ck['torch_rng'])
    torch.cuda.set_rng_state_all(ck['cuda_rng'])
stats_gpu = {k:v.cuda() for k,v in stats.items()}
start = time.monotonic()
stop_requested = False
def request_stop(signum, frame):
    global stop_requested
    stop_requested = True
signal.signal(signal.SIGTERM, request_stop)

def save():
    path = root/'checkpoints'/('step_%06d.pt' % step)
    if path.exists():
        return
    ck = dict(schema=SCHEMA, action_names=ACTION_NAMES, state_names=STATE_NAMES, model_config=CONFIG, model=model.state_dict(), optimizer=optimizer.state_dict(), stats=stats, step=step, elapsed_sec=elapsed_prior+time.monotonic()-start, manifest=entries, sampler_rng=rng.get_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), diffusion_config=dict(noise_scheduler.config), training=dict(batch_size=16, lr=1e-4, grad_clip=1., seed=a.seed, norm_std_floor=.01), action_horizon=4, inference_steps=25)
    tmp = path.with_suffix('.tmp')
    torch.save(ck, tmp)
    tmp.replace(path)
    (root/'latest.json').write_text(json.dumps(dict(checkpoint=str(path), step=step)))

def batch(split, sampler):
    states, images, actions = [], [], []
    for _ in range(16):
        e = data[split][sampler.randint(len(data[split]))]
        center = sampler.randint(0, len(e['action'])-15)
        ids = np.maximum(np.arange(center-3,center+1),0)
        states.append(e['state'][ids])
        images.append(e['images'][ids])
        actions.append(e['action'][center:center+16])
    state = torch.as_tensor(np.stack(states), device='cuda')
    action = torch.as_tensor(np.stack(actions), device='cuda')
    state = (state-stats_gpu['state_mean'])/stats_gpu['state_std']
    action = (action-stats_gpu['action_mean'])/stats_gpu['action_std']
    return state, images_to_torch(np.stack(images),'cuda'), action

save()
with (root/'train.jsonl').open('a', buffering=1) as log:
    while step < a.steps and time.monotonic()-start < a.seconds and not stop_requested:
        model.train()
        state, images, action = batch('train', rng)
        noise = torch.randn_like(action)
        times = torch.randint(0,100,(16,),device='cuda')
        noisy = noise_scheduler.add_noise(action, noise, times)
        pred = model(noisy, times, state, *images)
        loss = F.mse_loss(pred,noise)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        step += 1
        row = dict(step=step, loss=loss.item(), lr=optimizer.param_groups[0]['lr'], elapsed_sec=elapsed_prior+time.monotonic()-start, grad_norm=float(grad), max_memory_mb=torch.cuda.max_memory_allocated()/2**20)
        if step % 250 == 0 or step == a.steps:
            model.eval()
            # Fixed validation examples AND diffusion noise; no training RNG perturbation.
            with torch.random.fork_rng(devices=[0]), torch.no_grad():
                torch.manual_seed(98765)
                validation_rng.seed(98765)
                losses=[]
                for _ in range(4):
                    vs, vi, va = batch('val',validation_rng)
                    vn = torch.randn_like(va)
                    vt = torch.randint(0,100,(16,),device='cuda')
                    losses.append(F.mse_loss(model(noise_scheduler.add_noise(va,vn,vt),vt,vs,*vi),vn).item())
                row['val_loss'] = float(np.mean(losses))
        log.write(json.dumps(row)+'\n')
        if step % 100 == 0 or step == a.steps:
            print(json.dumps(row),flush=True)
        if step % 500 == 0:
            save()
    save()
print('SAVED',step,flush=True)
