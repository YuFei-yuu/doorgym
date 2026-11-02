import os,json,hashlib,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];SUPPORT=ROOT/'migration/wc4_act'
RUN=ROOT/'.repro/runs/wc4-act100-120k-h15-eval16-20261102'
DATA=ROOT/'.repro/runs/wc4-expert100-lerobot-20261031/lerobot/local/wc4_expert100_20261031'
REPO='local/wc4_expert100_20261031';STAGES=[120000];FRAMES=154667
INPUT_FINAL=ROOT/'.repro/reviews/wc4-expert100-lerobot-20261031-FINAL'
OLD50_FINAL=ROOT/'.repro/reviews/wc4-act200k-20261029-FINAL'

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
 assert shutil.disk_usage(ROOT).free>40*1024**3,'Capacity below40GiB; checkpoint and preserve'
 assert read(DATA/'meta/info.json')['total_frames']==154667 and read(DATA/'meta/info.json')['total_episodes']==100
 assert sha(DATA/'meta/stats.json')=='1fdfb23cfac8591bf328490ec28c88e8e667683b34d3cb16ac0c043191531db7'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==read(RUN/'manifest.json')['start_commit']
 for n,h in read(RUN/'source-binding.json').items():assert sha(ROOT/n)==h,n
