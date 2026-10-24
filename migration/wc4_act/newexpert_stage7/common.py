"""Run identity and conservative systemic-stop signaling."""
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE='973c3efc2b2d53e618f68d8d60513c3f623824b4'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def stop(batch,index,stage,reason):
    try:write(batch/'SYSTEMIC_STOP.json',{'index':index,'stage':stage,'reason':reason,'time':time.time()})
    except FileExistsError:pass
def verify(batch):
    m=read(batch/'manifest.json');c=read(batch/'preparation-receipt.json')
    assert len(m['trials'])==42 and m['maximum_sim_seconds']==150
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,'Source binding mismatch: '+n
    for name in ['manifest.json','geometry.json','initial-coverage.json','reuse-index.json','trial-bindings.json','preflight.json']:
        assert sha(batch/name)==c['file_sha256'][name],'Run binding mismatch: '+name
    assert sha(m['yaml'])==m['yaml_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==m['start_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    return m
