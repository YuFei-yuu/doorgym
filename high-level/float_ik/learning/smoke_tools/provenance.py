#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(sys.argv[1]).resolve()
repo=Path('/data/zyf/video2door')
out=root/'provenance';out.mkdir(exist_ok=True)
runtime_code='''import json,os,sys,importlib.util,torch,numpy,diffusers,imageio_ffmpeg
print(json.dumps(dict(python=sys.version,executable=sys.executable,sys_path=sys.path,torch_version=torch.__version__,torch_path=torch.__file__,torch_cuda=torch.version.cuda,cuda_initialized=torch.cuda.is_initialized(),numpy_version=numpy.__version__,diffusers_version=diffusers.__version__,gym_path=importlib.util.find_spec('isaacgym').origin,ffmpeg=imageio_ffmpeg.get_ffmpeg_exe(),environment={k:os.environ.get(k) for k in ['CONDA_PREFIX','PYTHONPATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','VIDEO2DOOR_GPU','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TMPDIR']}),indent=2))'''
runtime=subprocess.run([sys.executable,'-u','-c',runtime_code],capture_output=True,text=True,check=True)
assert runtime.stdout.strip(),'Runtime metadata probe returned empty output'
(out/'runtime.txt').write_text(runtime.stdout+runtime.stderr)
(out/'activate.sh.txt').write_text((repo/'.repro/blackwell/activate.sh').read_text())
for name,command in [('pip-freeze.txt',[sys.executable,'-m','pip','freeze']),('conda-explicit.txt',['/data/miniconda3/bin/conda','list','--explicit']),('git-status.txt',['git','status','--short']),('git-head.txt',['git','rev-parse','HEAD']),('gpu.txt',['nvidia-smi']),('uname.txt',['uname','-a'])]:
    result=subprocess.run(command,cwd=str(repo),capture_output=True,text=True)
    (out/name).write_text(result.stdout+result.stderr)
files=[p for p in (repo/'high-level/float_ik/learning').glob('*.py') if not p.name.startswith('._')]
files += [repo/'high-level/float_ik/isaacgym_float_ik_b1z1_basearn_push_door.py',repo/'high-level/float_ik/isaacgym_visualize_b1z1_basearn.py',repo/'high-level/dp/models/door_diffusion_policy.py',repo/'high-level/door_rl/door_asset_rl_env.py',repo/'.repro/blackwell/activate.sh',repo/'high-level/data/cfg/b1z1_opendoor.yaml']
files += list((repo/'high-level/data/asset/door_set/99650089960001').rglob('*'))
files += list((repo/'low-level/resources/robots/b1z1').rglob('*'))
files += [repo/'.repro/float-push/run.sh',repo/'.repro/float-push/verify_push.py']
extra=repo/'data/asset/b1z1-col/meshes/gripper_fix_90d.obj'
if extra.exists(): files.append(extra)
hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
(out/'source_asset_sha256.json').write_text(json.dumps(hashes,indent=2))
patch=[]
for path in sorted((repo/'high-level/float_ik/learning').glob('*.py')):
    if path.name.startswith('._'):continue
    proc=subprocess.run(['git','diff','--no-index','--','/dev/null',str(path.relative_to(repo))],cwd=str(repo),capture_output=True,text=True)
    patch.append(proc.stdout)
(out/'learning.patch').write_text(''.join(patch))
subprocess.run(['tar','--exclude=._*','--exclude=__pycache__','-czf',str(out/'learning-source.tar.gz'),'-C',str(repo),'high-level/float_ik/learning'],check=True)
print('PROVENANCE',len(hashes),'files')
