import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/data/zyf/video2door-wc4-act')
ID='wc4-act-resume40k-50k-s1000-20261015';RUN=ROOT/'.repro/runs'/ID
PARENT=ROOT/'.repro/runs/wc4-act-resume30k-40k-s1000-20261015'
SOURCE=PARENT/'train/checkpoints/040000'
DATA_RUN=ROOT/'.repro/runs/wc4-joint9-skill50-s62000-20261015'
DATA_ID='local/wc4_joint9_skill_wc4-joint9-skill50-s62000-20261015';DATA=DATA_RUN/'lerobot'/DATA_ID
HEAD='5ca33bf00bf7f0bd4e0484db8cbeb48c84cb7f7f'
ACT=ROOT/'.repro/envs/door-act/bin/python'
def save(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_check():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==HEAD
 assert not subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT,text=True).strip()
def data_check():
 manifest=json.loads((DATA_RUN/'dataset-manifest.json').read_text())
 for row in manifest:
  p=DATA/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
 return manifest
class GPULease:
 def __init__(self,gpu,purpose):self.gpu=gpu;self.purpose=purpose
 def __enter__(self):
  samples=[]
  for k in range(3):
   rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
   samples.append(next([v.strip() for v in row.split(',')] for row in rows if int(row.split(',')[0])==self.gpu))
   if k<2:time.sleep(2)
  uuid=samples[0][1]
  assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
  assert all(int(s[2])<1024 and int(s[3])<10 for s in samples)
  self.file=(Path('/data/zyf/video2door/.repro/locks')/(uuid+'.lock')).open('a');fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
  save(RUN/f'gpu-lease-{self.purpose}-{time.time_ns()}.json',{'gpu':self.gpu,'uuid':uuid,'samples':samples,'purpose':self.purpose,'pid':os.getpid(),'at':time.time()});return self
 def __exit__(self,*args):self.file.close()
