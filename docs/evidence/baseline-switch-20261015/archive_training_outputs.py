"""One-off, reversible archive for the authorized 2026-10-15 baseline switch.
No deletion, simulation, model loading, or environment mutation.
"""
import argparse,hashlib,json,os,shutil,stat,subprocess,time
from pathlib import Path

ROOTS=['experiments','low-level/logs/repro-smoke',*[f'.repro/{n}' for n in
 ['logs','tmp','retained-evidence','cleanup-20261013','maintenance','commit-review-20261014',
  'code-sync','repository-sync','sync','branch16040','smallbatch30k','lr-pair']]]
SKIP={'sources','source','assets','postprocessing_revisions','source_revisions','revisions','failure_attempts'}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def inventory(repo,roots,digest):
 records=[]
 for name in roots:
  base=repo/name
  paths=[base] if base.is_file() or base.is_symlink() else sorted(base.rglob('*'))
  for p in paths:
   if p.is_symlink():records.append(dict(path=str(p.relative_to(repo)),kind='symlink',target=os.readlink(p)))
   elif p.is_file():
    s=p.stat();row=dict(path=str(p.relative_to(repo)),kind='file',bytes=s.st_size,mode=stat.S_IMODE(s.st_mode))
    if digest:row['sha256']=sha(p)
    records.append(row)
    if digest and len(records)%50000==0:print(json.dumps(dict(phase='hashing',base=str(repo),entries=len(records))),flush=True)
 return records

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--archive',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 repo=Path(a.repo).resolve();dest=Path(a.archive).resolve()
 if dest==repo or repo in dest.parents:raise ValueError('Archive must be outside repository')
 roots=[n for n in ROOTS if (repo/n).exists()]
 roots += [str(p.relative_to(repo)) for p in sorted((repo/'.repro').glob('*-driver.log'))]
 for name in roots:
  tracked=subprocess.check_output(['git','ls-files','--',name],cwd=repo,text=True)
  if tracked:raise ValueError('Refuse to move tracked files: '+name)
 running=[]
 for proc in Path('/proc').glob('[0-9]*'):
  if int(proc.name)==os.getpid():continue
  try:
   args=(proc/'cmdline').read_bytes().split(b'\0');exe=Path(args[0].decode()).name
   if 'python' in exe and any(str(repo/n).encode() in v for v in args for n in roots):running.append(int(proc.name))
  except (OSError,UnicodeError,IndexError):pass
 if running:raise RuntimeError('Processes reference selected output roots: '+str(running))
 records=inventory(repo,roots,a.execute)
 total=sum(v.get('bytes',0) for v in records)
 summary=dict(repo=str(repo),archive=str(dest),roots=roots,files=sum(v['kind']=='file' for v in records),
  symlinks=sum(v['kind']=='symlink' for v in records),bytes=total,action='move_outside_repo_no_deletion',
  preserved=['docs/','all current source/config/assets','high-level/experiments/isaacgym/b1z1_opendoor.yaml',
   '.repro/envs/','.repro/blackwell/','.repro/cache/','.repro/conda-pkgs/','.repro root scripts',
   'existing local evidence outside code/'],active_selected_processes=running)
 if not a.execute:print(json.dumps(summary,indent=2));return
 if dest.exists():raise FileExistsError(dest)
 dest.mkdir(parents=True);(dest/'before_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
 evidence=repo/'docs/evidence/baseline-switch-20261015';evidence.mkdir(parents=True,exist_ok=True)
 copied=[]
 for row in records:
  if row['kind']!='file':continue
  rel=Path(row['path'])
  if rel.suffix not in {'.py','.sh'} or set(rel.parts)&SKIP:continue
  # Preserve run-local helper scripts, not duplicated whole-repo snapshots.
  if rel.parts[0]=='experiments' and len(rel.parts)>4:continue
  out=evidence/'scripts'/rel;out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists():raise FileExistsError(out)
  shutil.copy2(repo/rel,out);assert sha(out)==row['sha256'];copied.append(dict(path=str(out.relative_to(repo)),sha256=row['sha256']))
 shutil.copy2(Path(__file__),evidence/'archive_training_outputs.py')
 moved=[]
 for name in roots:
  old=repo/name;new=dest/'tree'/name;new.parent.mkdir(parents=True,exist_ok=True)
  if old.stat().st_dev!=new.parent.stat().st_dev:raise ValueError('Same-filesystem rename required')
  old.rename(new);moved.append(dict(old=str(old),new=str(new)))
  (dest/'move_journal.json').write_text(json.dumps(moved,indent=2)+'\n')
 after=inventory(dest/'tree',roots,True)
 if records!=after:raise RuntimeError('Archive integrity mismatch; preserve both manifests for review')
 (dest/'after_manifest.json').write_text(json.dumps(after,indent=2)+'\n')
 summary.update(passed=True,at=time.time(),all_file_hashes_and_symlinks_preserved=True,
  before_manifest_sha256=sha(dest/'before_manifest.json'),after_manifest_sha256=sha(dest/'after_manifest.json'),
  source_evidence_copies=copied,moves=moved,archive_does_not_free_filesystem_space=True)
 (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (evidence/'archive_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k not in ['moves','source_evidence_copies']},indent=2))

if __name__=='__main__':main()
