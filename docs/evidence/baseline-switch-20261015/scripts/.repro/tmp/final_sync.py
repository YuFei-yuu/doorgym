import pathlib,subprocess,hashlib,json,time
repo=pathlib.Path('/data/zyf/video2door');r=repo/'experiments/float-push/20261012-ee-short-refined'
subprocess.run(['git','fetch','.repro/tmp/final.bundle','main'],cwd=repo,check=True)
commit=subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=repo,text=True).strip();assert commit.startswith('d0b7390')
old=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert old.startswith('e32bf75')
expected=json.loads((repo/'.repro/tmp/final_source_hashes.json').read_text())
changed=subprocess.check_output(['git','ls-files','-m','-o','--exclude-standard','-z'],cwd=repo).decode().split('\0');assert set(x for x in changed if x)<=set(expected)
updates={}
for name,digest in expected.items():
    data=subprocess.check_output(['git','show',commit+':'+name],cwd=repo);assert hashlib.sha256(data).hexdigest()==digest
    p=repo/name
    if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
        original=subprocess.run(['git','show',old+':'+name],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        assert original.returncode==0 and p.read_bytes()==original.stdout,'Unexpected later edit: '+name
    updates[name]=data
# All paths verified before writing; current runtime source is already identical.
for name,data in updates.items():
    p=repo/name;p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists() or p.read_bytes()!=data:p.write_bytes(data)
subprocess.run(['git','read-tree',commit],cwd=repo,check=True);subprocess.run(['git','update-ref','HEAD',commit,old],cwd=repo,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
(r/'git_delivery.json').write_text(json.dumps(dict(initial_commit=old,final_commit=commit,worktree_clean=True,verified_updated_files=len(expected)),indent=2))
# Seal only after all jobs and final Git synchronization have completed.
assert not (r/'active_command.json').exists()
prior=r/'artifact_sha256.json'
if prior.exists():(r/'artifact_sha256_initial_delivery.json').write_bytes(prior.read_bytes())
files=[p for p in r.rglob('*') if p.is_file() and p.name not in ['artifact_sha256.json','driver.lock','budget.lock']]
manifest={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(r/'artifact_sha256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(commit=commit,clean=True,sealed_artifacts=len(manifest))))
