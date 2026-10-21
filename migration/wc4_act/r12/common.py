"""This run's immutable identity and shared receipts. No environment mutation."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
ID='wc4-fulltask-act-r12-s1000-20261022'
RUN=ROOT/'.repro/runs'/ID
OLD=ROOT/'.repro/runs/wc4-fulltask-reaudit-r11-20261021'
REPO='local/wc4_fulltask_reaudit_r11_20261021'
DATA=OLD/'lerobot'/REPO
SUPPORT=ROOT/'migration/wc4_act'

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def verify():
 for name,digest in read(RUN/'source-binding.json').items():
  assert sha(ROOT/name)==digest,'Frozen source changed: '+name
 for name,digest in read(RUN/'frozen-files.json').items():
  assert sha(RUN/name)==digest,'Frozen parameters changed: '+name
 assert read(DATA/'meta/info.json')['total_frames']==159529
 assert read(DATA/'meta/info.json')['total_episodes']==72
 assert __import__('shutil').disk_usage(ROOT).free>40*1024**3,'Capacity stop'
