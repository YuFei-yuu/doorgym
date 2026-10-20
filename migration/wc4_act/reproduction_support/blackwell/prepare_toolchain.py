from pathlib import Path
import os

prefix = Path(os.environ['CONDA_PREFIX'])
root = Path('/data/zyf/video2door/.repro/blackwell/cuda-12.8')
root.mkdir(exist_ok=True)

def link(source, target):
    if target.is_symlink() or target.exists():
        assert target.resolve() == source.resolve(), (target, source)
    else:
        target.symlink_to(source)

link(prefix / 'bin', root / 'bin')
link(prefix / 'nvvm', root / 'nvvm')
link(prefix / 'targets', root / 'targets')
for name, source in [('include', prefix / 'targets/x86_64-linux/include'),
                     ('lib64', prefix / 'targets/x86_64-linux/lib')]:
    target = root / name
    target.mkdir(exist_ok=True)
    for entry in source.iterdir():
        link(entry, target / entry.name)
nvtx = prefix / 'lib/python3.8/site-packages/nvidia/nvtx'
for entry in (nvtx / 'include').iterdir():
    if not (root / 'include' / entry.name).exists():
        link(entry, root / 'include' / entry.name)
link(prefix / 'targets/x86_64-linux/lib/libnvToolsExt.so.1', root / 'lib64/libnvToolsExt.so')
print('Private CUDA 12.8 view:', root)

