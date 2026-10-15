import pathlib,json,hashlib,shutil,sys
repo=pathlib.Path(sys.argv[1]);dest=pathlib.Path(sys.argv[2]);server=len(sys.argv)>3
root=repo/'.repro/float-push' if server else repo.parent/'reproduction/pro5000/float-push'
m=json.loads((dest/'cleanup_manifest.json').read_text());files=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size) for f in root.rglob('*') if f.is_file() and not f.is_symlink()]
m['roots'].append(dict(root=str(root),files=files,bytes=sum(f['bytes'] for f in files)))
for f in root.rglob('summary.json'):
 target=dest/'original_expert'/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target);m['retained_sha256'][str(target.relative_to(dest))]=hashlib.sha256(target.read_bytes()).hexdigest()
(dest/'cleanup_manifest.json').write_text(json.dumps(m,indent=2));print('added_old_wrapper_bytes',sum(f['bytes'] for f in files))
