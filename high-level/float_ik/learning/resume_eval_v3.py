"""Resume evaluation only; never convert data or enter a training routine."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE / 'run_tools'))
from launch import launch
from bounded import atomic_json


def main(root, total, timeout):
    root = Path(root).resolve()
    with (root / 'evaluation-resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'active_command.json').exists():
            raise RuntimeError('Inspect existing active_command before resuming')
        if (root / 'evaluation_resume.json').exists():
            raise FileExistsError('Evaluation resume already attempted')
        provenance = json.loads((root / 'training/checkpoint_provenance.json').read_text())
        checkpoint = root / 'training/frozen_step_005000.pt'
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if provenance['completed_updates'] != 5000 or digest != provenance['sha256']:
            raise ValueError('Frozen 5000-update checkpoint provenance mismatch')
        if not json.loads((root / 'checkpoint_checks.json').read_text())['passed']:
            raise ValueError('Checkpoint checks failed')
        contract = provenance['recording_contract']
        for name, expected in contract['source_hashes'].items():
            if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen control source changed: ' + name)
        old_state = json.loads((root / 'pipeline_state.json').read_text())
        atomic_json(root / 'evaluation_resume.json', dict(
            previous_state=old_state, checkpoint_sha256=digest, completed_updates=5000,
            training_forbidden=True, started=time.time(), total_seconds=total,
            per_task_timeout_seconds=timeout))

        def state(phase, **extra):
            value = dict(phase=phase, checkpoint_sha256=digest, resumed_evaluation_only=True, **extra)
            atomic_json(root / 'pipeline_state.json', value)
            print(json.dumps(value), flush=True)

        def cpu(script, args, log):
            prefix = REPO / '.repro/envs/b1z1-blackwell'
            env = dict(os.environ, PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            env.pop('PYTHONPATH', None)
            env['LD_LIBRARY_PATH'] = str(prefix / 'lib')
            with (root / 'logs' / log).open('x') as out:
                subprocess.run(['nice', '-n', '10', str(prefix / 'bin/python'), str(script),
                                *map(str, args)], cwd=REPO, env=env, stdout=out,
                               stderr=subprocess.STDOUT, check=True)

        try:
            state('evaluating', current_case='offline_retry1')
            output = root / 'evaluation/offline.json'
            if output.exists() or output.with_name('offline_predictions.npz').exists():
                raise FileExistsError('Existing offline results require inspection')
            code = launch(root, 'evaluation__offline_retry1',
                          ['python', '-u', str(HERE / 'offline_eval.py'), '--root', str(root),
                           '--checkpoint', str(checkpoint), '--output', str(output),
                           '--episodes', '4', '--windows', '32'], timeout=timeout, total=total)
            if code:
                raise RuntimeError('Offline retry failed: exit ' + str(code))

            cases = [('short_%d' % steps, 14300, steps, True) for steps in [60, 300, 900]]
            cases += [('validation_%d' % seed, seed, 2500, True) for seed in range(14300, 14303)]
            cases += [('test_%d' % seed, seed, 2500, True) for seed in range(14400, 14403)]
            cases.append(('expert_14400', 14400, 2500, False))
            for name, seed, steps, policy in cases:
                state('evaluating', current_case=name)
                folder = root / 'evaluation' / name
                if folder.exists():
                    raise FileExistsError('Refusing duplicate rollout: ' + name)
                command = ['python', '-u', str(HERE / 'wholebody_float.py'), '--headless',
                           '--no_show_seg', '--steps', str(steps), '--camera_seg', '--camera_depth',
                           '--float_seed', str(seed), '--float_output', str(folder), '--float_video']
                for key, value in contract['expert_args'].items():
                    command += ['--' + key, str(value)]
                if policy:
                    command += ['--dp_policy_checkpoint', str(checkpoint), '--dp_inference_steps',
                                '100', '--dp_action_horizon', '16']
                attempt = dict(case=name, seed=seed, steps=steps, policy=policy, started=time.time())
                with (root / 'evaluation/attempts.jsonl').open('a') as out:
                    out.write(json.dumps(attempt) + '\n')
                code = launch(root, 'evaluation__' + name, command, timeout=timeout, total=total)
                if code:
                    raise RuntimeError('Rollout process failed: %s exit %s' % (name, code))
                if not (folder / 'summary.json').exists():
                    raise RuntimeError('Rollout missing summary: ' + name)

            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
                raise ValueError('Checkpoint changed during evaluation')
            state('delivering')
            folders = [p.parent for p in (root / 'evaluation').glob('*/execution.jsonl')]
            cpu(HERE / 'analyze_v3.py', folders, 'evaluation_plots_resume.log')
            cpu(HERE / 'deliver_v3.py', ['--root', root], 'delivery_resume.log')
            state('complete')
        except BaseException as exc:
            state('error', error=repr(exc))
            traceback.print_exc()
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--total-seconds', required=True, type=float)
    p.add_argument('--timeout', type=float, default=1800)
    a = p.parse_args()
    main(a.root, a.total_seconds, a.timeout)
