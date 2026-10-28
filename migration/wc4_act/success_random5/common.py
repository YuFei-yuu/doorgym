import hashlib,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE='cbcf6b965746824fdc68d6857e388bb392cd5da3'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def head():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def verify(run):
    m=read(run/'manifest.json')
    assert len(m['trials'])==5 and m['maximum_sim_seconds']==120
    assert head()==m['start_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
    assert sha(m['yaml'])==m['yaml_sha256']
    assert sha(run/'geometry.json')==m['geometry_sha256']
    return m
