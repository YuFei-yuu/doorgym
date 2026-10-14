"""Verify diagnostic media, preserve source evidence and create delivery hashes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from artifacts import write_new_json
from forward_diagnostics import digest, verify_original, REPLAY_SEEDS, MODES
from diagnostic_integrity import frozen_phase, materialize_external_links


def inspect_video(path, summary, ffmpeg, imageio_ffmpeg):
    expected = summary.get('frames', 0)//2
    try:
        if not path.exists():
            raise FileNotFoundError('missing_video')
        reader = imageio_ffmpeg.read_frames(str(path), input_params=['-threads', '2'])
        try:
            probe = next(reader)
        finally:
            reader.close()
        decoded = subprocess.check_output([ffmpeg, '-v', 'error', '-xerror', '-threads', '2', '-filter_threads', '1',
            '-i', str(path), '-threads', '2', '-progress', 'pipe:1', '-f', 'null', '-'], text=True, stderr=subprocess.PIPE)
        counts = [int(line.split('=', 1)[1]) for line in decoded.splitlines() if line.startswith('frame=')]
        if not counts or counts[-1] != expected or abs(probe['fps']-25) > 1e-6:
            raise ValueError('Video frame count differs from recorded horizon')
        return dict(expected_frames=expected, decoded_frames=counts[-1], decoded=True, error=None, **probe)
    except (OSError, ValueError, RuntimeError, StopIteration, subprocess.CalledProcessError) as exc:
        return dict(expected_frames=expected, decoded_frames=None, decoded=False, error=str(exc))


@frozen_phase
def finalize(root):
    repo = Path(__file__).resolve().parents[3]
    import imageio_ffmpeg
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    media = []
    for mode in MODES:
        for seed in REPLAY_SEEDS:
            folder = root/'replay'/mode/('seed_%d' % seed)
            path = folder/'rollout.mp4'
            summary_path = folder/'summary.json'
            try:
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            except ValueError:
                summary = {}
            media.append(dict(mode=mode, seed=seed, path=str(path.relative_to(root)),
                              **inspect_video(path, summary, ffmpeg, imageio_ffmpeg)))
    output = root/'analysis/comparison_seed_16108.mp4'
    comparison_ready = all(v['decoded'] for v in media if v['seed'] == 16108)
    if comparison_ready:
        encode_comparison(root, media, output, ffmpeg, imageio_ffmpeg, cv2, np)
    write_new_json(root/'media_verification.json', dict(videos=media, comparison=str(output.relative_to(root)) if comparison_ready else None,
        comparison_decoded=comparison_ready, errors=sum(not v['decoded'] for v in media)))
    extra_sources = {}
    for name in ['diagnostic_report.py', 'diagnostic_history.py', 'diagnostic_runtime.py', 'finalize_diagnostics.py']:
        source = Path(__file__).with_name(name)
        relative = str(source.relative_to(repo))
        extra_sources[relative] = digest(source)
        destination = root/'sources'/relative
        if not destination.exists() or digest(destination) != extra_sources[relative]:
            raise ValueError('Phase source was not frozen before execution: '+relative)
    write_new_json(root/'additional_sources.json', extra_sources)
    verify_original(root)
    acceptance = json.loads((root/'analysis/acceptance.json').read_text())
    candidate = json.loads((root/'offline/normalization_candidate.json').read_text())
    offline = json.loads((root/'offline/summary.json').read_text())
    history = json.loads((root/'offline/history_summary.json').read_text())
    preservation = json.loads((root/'preservation.json').read_text())
    passed = (acceptance['replay_attempts'] == 9 and acceptance['all_replay_horizons_complete'] and
              acceptance['recorded_successes'] == 3 and acceptance['pure_actions_successes'] == 3 and
              acceptance['initial_observations_equal'] and candidate['coverage'] == 1. and
              candidate['roundtrip_max_error'] <= 1e-5 and candidate['invalid_quaternions'] == 0 and
              offline['completed_sampling_windows'] == 405 and offline['completed_noise_windows'] == 2025 and
              offline['invalid_quaternion_count'] == 0 and preservation['passed'] and
              not offline['cpu_cuda_initialized'] and not history['cuda_initialized'] and all(v['decoded'] for v in media))
    process_text = subprocess.check_output(['ps', '-axo', 'pid,ppid,nice,etime,command'], text=True)
    active_task = [line for line in process_text.splitlines() if any('/'+name in line for name in
                   ['train_door_dp.py', 'wholebody_float.py', 'diagnostic_replay.py', 'forward_pipeline.py'])]
    phase = 'completed_no_training' if passed else 'completed_with_failed_checks'
    write_new_json(root/'completion.json', dict(phase=phase, passed=passed, no_training=True,
        objective_complete=False, holdout_test_used=False, active_training_or_simulation=active_task,
        gpu_state=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader'], text=True),
        compute_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True),
        completed=time.time(), python=sys.version, prefix=sys.prefix, nice=os.getpriority(os.PRIO_PROCESS, 0)))
    (root/'state.json').write_text(json.dumps(dict(phase=phase, no_training=True, objective_complete=False, passed=passed), indent=2))
    links = materialize_external_links(root)
    write_new_json(root/'materialized_links.json', links)
    hashes = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'delivery_sha256.json'}
    write_new_json(root/'delivery_sha256.json', hashes)
    print(json.dumps(dict(phase=phase, files=len(hashes), passed=passed)), flush=True)


def encode_comparison(root, media, output, ffmpeg, imageio_ffmpeg, cv2, np):
    readers = [imageio_ffmpeg.read_frames(str(root/'replay'/mode/'seed_16108/rollout.mp4'), input_params=['-threads', '2']) for mode in MODES]
    metadata = [next(reader) for reader in readers]
    count = next(v['expected_frames'] for v in media if v['seed'] == 16108)
    command = [ffmpeg, '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x426', '-r', '25',
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '22',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in range(count):
            panels = []
            for reader, meta, label in zip(readers, metadata, ['RECORDED EXPERT FLAGS', '10D ACTIONS ONLY', '10D ACTIONS WITH CLIPPING']):
                w, h = meta['size']
                panel = cv2.resize(np.frombuffer(next(reader), np.uint8).reshape(h, w, 3), (640, 426))
                cv2.rectangle(panel, (0, 400), (639, 425), (0, 0, 0), -1)
                cv2.putText(panel, label, (8, 418), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1)
                panels.append(panel)
            joined = np.hstack(panels)
            process.stdin.write(joined.tobytes())
            if frame in [0, count//2, count-1]:
                cv2.imwrite(str(root/'analysis'/('comparison_frame_%04d.png' % frame)), joined[:, :, ::-1])
    finally:
        for reader in readers:
            reader.close()
        process.stdin.close()
        code = process.wait()
    if code:
        raise ValueError('Comparison encoding failed')
    subprocess.run([ffmpeg, '-v', 'error', '-xerror', '-threads', '2', '-filter_threads', '1', '-i', str(output), '-threads', '2', '-f', 'null', '-'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    finalize(Path(parser.parse_args().root))
