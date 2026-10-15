#!/usr/bin/env python3
"""CPU-only synthetic codec dependency check, never touches experiment data."""
import json
import tempfile
from pathlib import Path
import av
import numpy as np
from lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec


def main():
    with tempfile.TemporaryDirectory(prefix='wc4-codec-check-') as tmp:
        path = Path(tmp) / 'synthetic.mp4'
        with av.open(str(path), 'w') as container:
            stream = container.add_stream('h264', rate=25)
            stream.width, stream.height, stream.pix_fmt = 640, 480, 'yuv420p'
            for value in (64, 128):
                frame = av.VideoFrame.from_ndarray(np.full((480, 640, 3), value, dtype=np.uint8), format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        backend = get_safe_default_codec()
        frames = decode_video_frames(path, [0.0, 0.04], tolerance_s=1e-4, backend=backend)
        assert tuple(frames.shape) == (2, 3, 480, 640), frames.shape
        assert abs(float(frames[0].mean()) - 64/255) < 0.03
        assert abs(float(frames[1].mean()) - 128/255) < 0.03
        print(json.dumps({'synthetic': True, 'fps': 25, 'encoder': 'h264', 'decoder': backend,
            'decoded_shape': list(frames.shape), 'means': [float(f.mean()) for f in frames],
            'limitation': 'Dependency check only; not a LeRobot dataset conversion.'}, indent=2))


if __name__ == '__main__':
    main()
