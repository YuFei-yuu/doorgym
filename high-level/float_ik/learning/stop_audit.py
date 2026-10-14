"""Action and video audits using the original log format and bundled FFmpeg."""
from pathlib import Path
import subprocess

import numpy as np


def compare_actions(raw, trajectory, rows):
    from door_dp_common import _to_list
    raw, trajectory = np.asarray(raw), np.asarray(trajectory)
    logged = np.asarray([row['applied_action'] for row in rows], dtype=np.float64)
    shape_ok = raw.ndim == 2 and raw.shape[1:] == (10,) and raw.shape == trajectory.shape == logged.shape
    finite = shape_ok and all(np.isfinite(value).all() for value in (raw, trajectory, logged))
    exact_full_precision = bool(finite and np.array_equal(raw, trajectory))
    serialized_equal = bool(finite and np.array_equal(np.asarray(_to_list(raw)), logged))
    return dict(passed=exact_full_precision and serialized_equal,
        shape_valid=bool(shape_ok), finite=bool(finite), raw_trajectory_full_precision_exact=exact_full_precision,
        log_matches_original_serialization=serialized_equal,
        serialization='door_dp_common._to_list: float64 then np.round(precision=5)',
        max_raw_log_difference=float(np.max(np.abs(raw.astype(np.float64)-logged))) if finite else None,
        limitation='The rounded log cannot independently prove sub-rounding-bit precision; raw and trajectory are compared without rounding.')


def inspect_video(path, expected_frames, expected_fps=25.):
    import imageio_ffmpeg
    path = Path(path)
    result = dict(path=str(path), expected_frames=int(expected_frames), expected_fps=float(expected_fps),
                  passed=False, decoded=False, decoded_frames=None)
    try:
        reader = imageio_ffmpeg.read_frames(str(path), input_params=['-threads', '2'])
        try:
            metadata = next(reader)
        finally:
            reader.close()
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-xerror', '-err_detect', 'explode',
            '-threads', '2', '-filter_threads', '1', '-i', str(path), '-threads', '2', '-vsync', '0',
            '-progress', 'pipe:1', '-f', 'null', '-']
        process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        counts = [int(line.split('=', 1)[1]) for line in process.stdout.splitlines() if line.startswith('frame=')]
        count = counts[-1] if counts else None
        result.update(metadata=metadata, decoded=process.returncode == 0, decoded_frames=count,
            decoder_exit_code=process.returncode, decoder_stderr=process.stderr,
            passed=process.returncode == 0 and count == expected_frames and abs(metadata['fps']-expected_fps) <= 1e-6)
    except (OSError, ValueError, RuntimeError, StopIteration) as exc:
        result['error'] = repr(exc)
    return result
