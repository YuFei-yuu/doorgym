"""R17 single-variable execution-horizon contract; no training/export entry."""
import json,hashlib,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];SUPPORT=ROOT/'migration/wc4_act'
RUN=ROOT/'.repro/runs/wc4-fulltask-act-r17-h5-20261023'
BASE=ROOT/'.repro/runs/wc4-fulltask-act-r15-s1000-20261022-cont001'
CHECKPOINT=BASE/'checkpoints/050000/door-checkpoint/model_latest.pt'
MODEL_SHA='76178cc3248521158535923c02ecbe8fd8085e08f47d57b8c7c3852e871293b5'
SEEDS=list(range(62200,62216))
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def verify():
 for rel,h in read(RUN/'source-binding.json').items():assert sha(ROOT/rel)==h,rel
 for rel,h in read(RUN/'frozen-files.json').items():assert sha(RUN/rel)==h,rel
 for rel,h in read(RUN/'model-binding.json')['files'].items():assert sha(rel)==h,rel
 assert shutil.disk_usage(ROOT).free>40*1024**3,'Capacity stop; preserve history'
