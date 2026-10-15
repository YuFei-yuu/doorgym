import subprocess,pathlib,json,hashlib
r=pathlib.Path('/data/zyf/video2door')
expected=json.loads((r/'.repro/tmp/push_source_hashes.json').read_text())
assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in expected.items())
commit=subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=r,text=True).strip()
assert commit.startswith('e32bf75')
# Only update HEAD/index after every changed/new working file matches this commit.
for p in expected:
    assert subprocess.check_output(['git','show',commit+':'+p],cwd=r)==(r/p).read_bytes(),p
subprocess.run(['git','read-tree',commit],cwd=r,check=True)
subprocess.run(['git','update-ref','HEAD',commit,'1f0f112ace6a94083273e3ff2dc59d8421542f89'],cwd=r,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=r)
print('Server clean at',commit)
