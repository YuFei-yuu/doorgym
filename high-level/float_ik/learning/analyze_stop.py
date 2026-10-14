"""Read existing small3 artifacts; write only a separate analysis directory."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

source, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
inputs = {}
def track(path):
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path
def rows(path):
    return [json.loads(line) for line in track(path).open()]
def movement(records):
    before = np.asarray([r['root_before'][:2] for r in records])
    after = np.asarray([r['base_after'][:2] for r in records])
    vx = np.asarray([r['robot_command']['vx'] for r in records])
    return dict(frames=len(records), duration_s=len(records)*.02,
                path_m=float(np.linalg.norm(after-before, axis=1).sum()),
                net_m=float(np.linalg.norm(after[-1]-before[0])),
                mean_vx=float(vx.mean()), min_vx=float(vx.min()), max_vx=float(vx.max()),
                low_speed_fraction=float((abs(vx)<.01).mean()),
                final_vx=float(vx[-1]), replans=sum(r['extra']['dp_replanned'] for r in records),
                measured_mean_xy_speed=float(np.linalg.norm(after-before,axis=1).mean()/.02))

experts=[]
for index, seed in enumerate((16040,16041,16042)):
    e=rows(source/f'reference/seed_{seed}/execution.jsonl')
    with np.load(track(source/f'data/raw/episode_{index:06d}.npz')) as raw:
        actions=raw['action']; phases=raw['subtask_index'].reshape(-1)
        done=np.asarray([r['extra']['pass_done'] for r in e])
        assert len(e)==len(actions) and np.array_equal(done, phases==12)
        assert np.max(np.abs(actions[:,0]-[r['robot_command']['vx'] for r in e]))<1e-7
        centers=np.arange(len(e)-31)
        offsets=np.zeros(32,dtype=int)
        for c in centers:
            offsets += done[c:c+32]
        experts.append(dict(seed=seed, frames=len(e), windows=len(centers),
            terminal_frames=np.flatnonzero(done).tolist(), terminal_vx=actions[done,0].tolist(),
            terminal_yaw=actions[done,1].tolist(), terminal_target_count_by_offset=offsets.tolist(),
            terminal_in_executed_prefix=int(offsets[:16].sum()), terminal_observation_centers=int(done[centers].sum()),
            total_stationary_raw_frames=int((abs(actions[:,0])<.01).sum()),
            last_three_commands=[r['robot_command'] for r in e[-3:]],
            terminal_measured_displacement_m=float(np.linalg.norm(np.asarray(e[-1]['base_after'][:2])-e[-1]['root_before'][:2]))))

evaluations=[]
for step in (250,500,1000):
    for seed in (16040,16041,16042):
        e=rows(source/f'evaluation/step_{step:06d}/seed_{seed}/execution.jsonl')
        passed=[i for i,r in enumerate(e) if r['extra']['pass_done']]
        out=dict(step=step,seed=seed,all=movement(e), first_pass_frame=passed[0] if passed else None)
        if passed:
            i=passed[0]
            out.update(first_pass_s=i*.02,post_pass=movement(e[i:]),
                post_queue_horizon=movement(e[i+16:]),last_two_seconds=movement(e[-100:]),
                first_pass_command=e[i]['robot_command'],
                last_through_distance_m=float(np.linalg.norm(np.asarray(e[-1]['base_after'][:2])-e[-1]['extra']['through_target'])))
        evaluations.append(out)

totals=dict(frames=sum(e['frames'] for e in experts),windows=sum(e['windows'] for e in experts),
    terminal_frames=sum(len(e['terminal_frames']) for e in experts),
    terminal_executed_prefix_targets=sum(e['terminal_in_executed_prefix'] for e in experts))
totals.update(terminal_raw_percent=100*totals['terminal_frames']/totals['frames'],
    terminal_sequence_target_percent=100*totals['terminal_frames']/(totals['windows']*32),
    sample_exposure_epochs=1000*64/totals['windows'])
report=dict(source=str(source),python=sys.executable,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    threads=os.environ.get('OMP_NUM_THREADS'),experts=experts,totals=totals,evaluations=evaluations,inputs_sha256=inputs)
(output/'stop_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='inputs_sha256'},indent=2))
