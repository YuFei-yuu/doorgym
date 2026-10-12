#!/usr/bin/env python3
import argparse
import csv
import json
import time
import traceback
import signal
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from float_env import FloatEnv, SCHEMA, STATE_NAMES, ACTION_NAMES

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--mode', choices=['expert', 'policy', 'replay'], required=True)
p.add_argument('--checkpoint')
p.add_argument('--replay')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--nominal', action='store_true')
p.add_argument('--steps', type=int, default=2850)
p.add_argument('--split', default='diagnostic')
p.add_argument('--record', action='store_true')
p.add_argument('--video', action='store_true')
p.add_argument('--inference-steps', type=int, choices=[25, 100], default=25)
p.add_argument('--diagnostic-steps', type=int, default=0)
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
metadata = dict(vars(a), source_hashes=source_hashes, schema=SCHEMA, fps=60, state_names=STATE_NAMES, action_names=ACTION_NAMES, started=time.time(), status='started', success=False)
(out/'summary.json').write_text(json.dumps(metadata, indent=2))
def terminate(signum, frame):
    raise TimeoutError('Owned rollout reached its wall-time limit')
signal.signal(signal.SIGTERM, terminate)
writer = None
env = FloatEnv(expert=a.mode == 'expert', video=a.video)
states, images, depths, actions, raw_actions, metrics, phases = [], [], [], [], [], [], []
sensor_depths = []
next_states = []
execution_latencies = []
latencies, policy_latencies, decision_steps = [], [], []
early = []
try:
    obs = env.reset(a.seed, {'randomize': not a.nominal})
    metadata.update(action_lower=env.action_lower.tolist(), action_upper=env.action_upper.tolist(), config=vars(env.args), randomization=env.randomization, state_dim=len(obs['state']), initial_door_deg=float(env.gym.get_actor_dof_states(env.env, env.door_actor, 1)['pos'][0])*180/np.pi)
    (out/'config.json').write_text(json.dumps(metadata, indent=2, default=str))
    if a.mode == 'policy':
        from policy import Policy
        try:
            env.expert_action()
        except RuntimeError as exc:
            assert 'Expert function called' in str(exc)
            metadata['expert_call_guard_passed'] = True
        else:
            raise RuntimeError('Independent policy expert guard failed')
        controller = Policy(a.checkpoint, seed=a.seed, inference_steps=a.inference_steps)
        metadata['checkpoint_step'] = controller.step
        diagnostic_stats = {k: v.cpu().numpy() for k, v in controller.stats.items()}
    (out/'config.json').write_text(json.dumps(metadata, indent=2, default=str))
    if a.mode == 'replay':
        replay = np.load(a.replay)['action']
    if a.video:
        import cv2
        writer = cv2.VideoWriter(str(out/'rollout.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30., (640, 360))
        if not writer.isOpened():
            raise RuntimeError('MP4 encoder unavailable')
    for i in range(a.steps):
        before = obs
        if a.record:
            states.append(obs['state'].copy())
            images.append(obs['images'].copy())
            depths.append(obs['depth_m'].copy())
            sensor_depths.append(obs['gym_depth_m'].copy())
        start = time.perf_counter()
        decision = False
        if a.mode == 'expert':
            action = env.expert_action()
        elif a.mode == 'replay':
            action = replay[i]
        else:
            action, decision = controller.act(obs)
        latency = time.perf_counter()-start
        policy_latencies.append(latency)
        if decision:
            latencies.append(latency)
            decision_steps.append(i)
        raw_actions.append(action.copy())
        diagnostic = None
        if i < a.diagnostic_steps:
            diagnostic = {k: before[k].copy() for k in ('state', 'images', 'depth_m', 'gym_depth_m')}
            diagnostic.update(frame=i, raw_action=action.copy(), decision=decision, decision_sec=latency)
            if a.mode == 'policy':
                history_state = np.stack([x['state'] for x in controller.history])
                diagnostic.update(normalized_state=(before['state']-diagnostic_stats['state_mean'])/diagnostic_stats['state_std'], history_state=history_state, normalized_history=(history_state-diagnostic_stats['state_mean'])/diagnostic_stats['state_std'], history_images=np.stack([x['images'] for x in controller.history]), history_frames=np.asarray([x['frame'] for x in controller.history]))
        execute_started = time.perf_counter()
        obs, info, done, applied = env.step(action)
        execution_latencies.append(time.perf_counter()-execute_started)
        actions.append(applied.copy())
        if diagnostic is not None:
            diagnostic.update(next_state=obs['state'].copy(), action=applied.copy(), clipped_channels=action != applied)
            early.append(diagnostic)
        if a.record:
            next_states.append(obs['state'].copy())
        phases.append(env.phase or 'policy')
        metrics.append(info)
        if writer is not None and i % 2 == 1:
            frame = cv2.cvtColor(env.frame(), cv2.COLOR_RGB2BGR)
            label = '%s step=%s seed=%d t=%.2fs 30fps 1x' % (a.mode, metadata.get('checkpoint_step', '-'), a.seed, env.n/60)
            cv2.putText(frame, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1)
            cv2.putText(frame, out.name[:65], (8, 38), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
            cv2.putText(frame, 'door=%.1fdeg success=%s' % (info['signed_angle_deg'], info['success']), (8, 56), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
            writer.write(frame)
        if i % 600 == 0:
            print(json.dumps(dict(step=i, angle=info['signed_angle_deg'], mode=a.mode, decision_sec=latency if decision else None)), flush=True)
        if done:
            break
    metadata.update(status='completed', **metrics[-1])
    metadata['failure_type'] = 'success' if metadata['success'] else ('unreached' if not metadata['reach'] else 'not_unlocked' if not metadata['unlocked'] else 'insufficient_open' if not metadata['open80'] else 'not_passed')
except BaseException as exc:
    metadata.update(status='error', success=False, failure_type='infrastructure_or_invalid_action', error=repr(exc), traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    if writer is not None:
        writer.release()
    metadata.update(steps_completed=len(actions), elapsed_sec=time.time()-metadata['started'], phases=dict(Counter(phases)), decision_count=len(latencies), inference_p50_ms=float(np.median(latencies)*1000) if latencies else None, inference_p95_ms=float(np.percentile(latencies,95)*1000) if latencies else None, first_inference_ms=latencies[0]*1000 if latencies else None)
    if metrics:
        with (out/'metrics.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(metrics[0]))
            w.writeheader()
            w.writerows(metrics)
    if actions or raw_actions:
        # A timeout/invalid action may interrupt step after raw_action was
        # produced. Keep completed transitions aligned, and retain the pending
        # action separately rather than inventing an executed transition.
        np.savez_compressed(out/'trajectory.npz', action=np.asarray(actions).reshape(-1,10), raw_action=np.asarray(raw_actions[:len(actions)]).reshape(-1,10), rejected_or_interrupted_action=np.asarray(raw_actions[len(actions):]).reshape(-1,10), phase=np.asarray(phases), policy_call_sec=policy_latencies, decision_sec=latencies, decision_step=decision_steps, execution_observe_sec=execution_latencies)
    if a.record and actions:
        n = len(actions)
        np.savez_compressed(out/'episode.npz', state=np.asarray(states[:n]), next_state=np.asarray(next_states), images=np.asarray(images[:n]), depth_m=np.asarray(depths[:n]), gym_depth_m=np.asarray(sensor_depths[:n]), action=np.asarray(actions), frame=np.arange(n), next_frame=np.arange(1,n+1), sim_time=np.arange(n)/60.)
    if early:
        np.savez_compressed(out/'early_diagnostic.npz', **{k: np.asarray([x[k] for x in early]) for k in early[0]})
    (out/'summary.json').write_text(json.dumps(metadata, indent=2, default=str))
    print('SUMMARY ' + json.dumps(metadata, default=str), flush=True)
    env.close()
if metadata['status'] != 'completed':
    raise SystemExit(1)
