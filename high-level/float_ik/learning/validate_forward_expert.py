"""Predeclared 3x3 positions, two seeds each; every attempt stays in the result."""
import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE / 'run_tools'))
from launch import launch
from bounded import atomic_json
from dataset_contract import validate_episode


def main(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    with (root / 'validation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
            raise ValueError('Validation requires the committed clean working tree')
        cases = [dict(seed=14500+i, robot_x=x, robot_y=y, repeat=repeat)
                 for i, (x, y, repeat) in enumerate(itertools.product([3.9, 4.1, 4.3], [-.06, 0., .06], [0, 1]))]
        atomic_json(root / 'plan.json', dict(commit=commit, cases=cases, total=18,
                    authorization='User approved 9 positions x 2 seeds on 2026-10-13',
                    mode='expert', pass_navigation='forward', training=False,
                    limits=dict(total_seconds=7200, per_attempt_seconds=600)))
        rows = []
        source_hashes = None
        def state(phase, **extra):
            value = dict(phase=phase, commit=commit, attempted=len(rows), total=len(cases), **extra)
            atomic_json(root / 'state.json', value)
            print(json.dumps(value), flush=True)
        try:
            for case in cases:
                seed = case['seed']
                folder = root / ('seed_%d' % seed)
                state('running', current=case)
                attempt = dict(case, started=time.time(), folder=folder.name)
                with (root / 'attempts.jsonl').open('a') as log:
                    log.write(json.dumps(attempt) + '\n')
                command = ['python', '-u', str(HERE / 'wholebody_float.py'), '--headless', '--no_show_seg',
                           '--steps', '2500', '--camera_seg', '--camera_depth', '--float_video',
                           '--float_seed', str(seed), '--float_output', str(folder),
                           '--robot_x', str(case['robot_x']), '--robot_y', str(case['robot_y']),
                           '--robot_z', '.5', '--robot_yaw', str(math.pi), '--stop_distance', '.25',
                           '--push_base_vx', '.25', '--lever_step_size', '.06',
                           '--float_execution', 'source', '--float_pass_mode', 'forward',
                           '--float_push_target', 'wholebody']
                code = launch(root, 'evaluation__expert_%d' % seed, command, timeout=600, total=7200)
                summary_path = folder / 'summary.json'
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                row = dict(case, exit_code=code, opened=bool(summary.get('opened', False)),
                           pass_done=bool(summary.get('pass_done', False)),
                           success=code == 0 and bool(summary.get('success', False)),
                           frames=summary.get('frames'), max_signed_door_deg=summary.get('max_signed_door_deg'),
                           through_distance=summary.get('through_distance'),
                           first_open_frame=summary.get('first_open_frame'),
                           first_pass_frame=summary.get('first_pass_frame'),
                           invalid_image_frames=summary.get('invalid_image_frames'),
                           end_reason=summary.get('end_reason', 'process_failure'))
                if code == 0 and summary_path.exists():
                    config = json.loads((folder / 'config.json').read_text())
                    if source_hashes is None:source_hashes = config['source_hashes']
                    if source_hashes != config['source_hashes']:raise ValueError('Source drift between attempts')
                    for name, expected in source_hashes.items():
                        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
                            raise ValueError('Source changed: ' + name)
                    with np.load(folder / 'trajectory.npz') as z:
                        validate_episode(z)
                        row['initial_state'] = z['state'][0].tolist()
                        row['initial_base_pose'] = json.loads((folder / 'execution.jsonl').open().readline())['root_before'][:7]
                    events = [json.loads(x) for x in (folder / 'execution.jsonl').read_text().splitlines()]
                    roots = np.asarray([r['base_after'] for r in events])
                    q = roots[:, 3:7]
                    yaw = np.unwrap(np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]), 1-2*(q[:,1]**2+q[:,2]**2)))
                    row['pass_align_frames'] = sum(r['phase_name'] == 'pass_align' for r in events)
                    angles = -np.rad2deg([r['door_after'][0] for r in events])
                    if np.any(angles >= 80):
                        first = int(np.flatnonzero(angles >= 80)[0])
                        heading = np.array([math.cos(yaw[0]), math.sin(yaw[0])])
                        deltas = np.diff(roots[first:, :2], axis=0) @ heading
                        row['backward_travel_after80_m'] = float(np.maximum(0, -deltas).sum())
                        row['yaw_travel_after80_deg'] = float(np.rad2deg(np.abs(np.diff(yaw[first:])).sum()))
                        row['minimum_door_after80_deg'] = float(angles[first:].min())
                rows.append(row)
                atomic_json(root / 'results.json', rows)
                print(json.dumps(row), flush=True)
                if code:
                    raise RuntimeError('Process failure seed %d; attempt retained, no retry' % seed)

            state('analyzing')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            subprocess.run([sys.executable, str(HERE / 'analyze_v3.py'), *[str(root/('seed_%d'%r['seed'])) for r in rows]],
                           cwd=REPO, env=env, check=True)
            import imageio_ffmpeg
            for row in rows:
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-xerror', '-threads', '2',
                                '-i', str(root/('seed_%d'%row['seed'])/'rollout.mp4'), '-f', 'null', '-'], check=True)
            groups = []
            for x, y in itertools.product([3.9, 4.1, 4.3], [-.06, 0., .06]):
                group = [r for r in rows if r['robot_x']==x and r['robot_y']==y]
                groups.append(dict(robot_x=x, robot_y=y, attempts=len(group), opened=sum(r['opened'] for r in group),
                                   completed=sum(r['success'] for r in group), seeds=[r['seed'] for r in group]))
            aggregate = dict(attempts=len(rows), planned=18, opened=sum(r['opened'] for r in rows),
                             completed=sum(r['success'] for r in rows), groups=groups,
                             unique_actual_initial_base_poses=len({tuple(r['initial_base_pose']) for r in rows}),
                             videos_fully_decoded=len(rows), commit=commit)
            atomic_json(root / 'aggregate.json', aggregate)
            fields = list(dict.fromkeys(k for row in rows for k in row if k not in ['initial_state', 'initial_base_pose']))
            with (root / 'results.csv').open('w') as out:
                writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
                writer.writeheader();writer.writerows(rows)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
            matrix = np.array([g['completed']/g['attempts'] for g in groups]).reshape(3,3)
            axes[0].imshow(matrix, vmin=0, vmax=1, cmap='RdYlGn')
            axes[0].set(xticks=range(3), xticklabels=['-.06','0','.06'], yticks=range(3), yticklabels=['3.9','4.1','4.3'],
                        xlabel='robot_y parameter (m)', ylabel='robot_x (m)', title='Expert task completion')
            for i, g in enumerate(groups):axes[0].text(i%3, i//3, '%d/%d'%(g['completed'],g['attempts']), ha='center', va='center')
            axes[1].bar([str(r['seed']) for r in rows], [r['max_signed_door_deg'] for r in rows])
            axes[1].axhline(80, color='red', linestyle='--');axes[1].tick_params(axis='x', rotation=70)
            axes[1].set(xlabel='Predeclared seed', ylabel='Maximum signed door angle (deg)')
            fig.savefig(root/'expert_success.png', dpi=160);plt.close(fig)
            state('complete', aggregate=aggregate)
            hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob('*') if p.is_file() and p.name not in ['SHA256.json','budget.lock','validation.lock']}
            atomic_json(root / 'SHA256.json', hashes)
        except BaseException as exc:
            state('error', error=repr(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    main(parser.parse_args().root)
