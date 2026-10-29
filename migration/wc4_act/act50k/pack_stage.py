"""Transport completed stages only; does not import model or simulator code."""
import pathlib,json,hashlib,subprocess,sys,concurrent.futures
ROOT=pathlib.Path('/data/zyf/video2door-wc4-act');RUN=ROOT/'.repro/runs/wc4-act50k-20261029-r2'
stage=int(sys.argv[1]);assert stage in [10000,20000,30000,40000,50000]
base=RUN/'evaluation'/f'{stage:06d}';assert json.loads((base/'complete.json').read_text())['trials']==16
out=ROOT/'.repro/tmp'/f'act50k-r2-stage{stage:06d}-parts';out.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def pack(i):
 paths=[str((base/f'trial_{j:02d}').relative_to(RUN)) for j in range(2*i,2*i+2)]
 if i==0:
  paths+=['train/checkpoints/'+f'{stage:06d}','checkpoints/'+f'{stage:06d}']
  paths+=[str(p.relative_to(RUN)) for p in sorted(base.iterdir()) if p.is_file() and not p.is_symlink()]
 archive=out/f'part-{i:02d}.tar.gz';temporary=archive.with_suffix('.partial')
 command=['tar','--exclude=wandb','--exclude=__pycache__','-I','gzip -1','-cf',str(temporary),*paths]
 subprocess.run(command,cwd=RUN,check=True);temporary.rename(archive)
 return dict(part=i,path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=paths,command=command)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(pack,range(8)))
manifest=dict(stage=stage,complete=True,archives=rows)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(RUN/f'transport-parts-{stage:06d}.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(stage=stage,parts=8,bytes=sum(r['bytes'] for r in rows),manifest=str(out/'manifest.json'))),flush=True)
