import os,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
SUPPORT=ROOT/'migration/wc4_act'
RUN=ROOT/'.repro/runs/wc4-act300k-20261030'
DATA=ROOT/'.repro/runs/wc4-expert50-lerobot-20261029/lerobot/local/wc4_expert50_20261029'
REPO='local/wc4_expert50_20261029'
PARENT_FINAL=ROOT/'.repro/reviews/wc4-act200k-20261029-FINAL'
PARENT=PARENT_FINAL/'evidence/train/checkpoints/200000'
STAGES=[220000,240000,260000,280000,300000]
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def verify():
 import shutil
 assert shutil.disk_usage(ROOT).free>25*1024**3,'Capacity below 25GiB'
 assert read(DATA/'meta/info.json')['total_frames']==77356
 assert sha(DATA/'meta/stats.json')=='0903435b4403a9c5d5856e3cdaa1febd441aad4d4a8530c995a98d3434fce87a'
 for n,h in read(RUN/'source-binding.json').items():assert sha(ROOT/n)==h,n
