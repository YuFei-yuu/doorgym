import argparse,hashlib,json,pathlib,shutil,subprocess
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--evidence',required=True);p.add_argument('--server',action='store_true');p.add_argument('--delete',action='store_true');a=p.parse_args();repo=pathlib.Path(a.repo).resolve();dest=pathlib.Path(a.evidence).resolve()
if a.server:
    roots=[repo/'experiments/float-push'/s for s in ['20261012-195314-ee-short','20261012-ee-ik-adapter-check','20261012-ee-short-refined','20261012-upstream-float-v2','20261012-upstream-float-v2-r1','20261012-upstream-float-v2-r2']]
    roots += [repo/'.repro'/s for s in ['ee-cleanup-validation','upstream-v2-preflight']]
    refined=roots[2];v2=roots[3:6]
else:
    base=repo.parent/'reproduction/pro5000';roots=[base/s for s in ['float-push-ee-ik-check','float-push-ee-short','float-push-upstream-v2','float-push-ee-cleanup-validation']]
    refined=base/'float-push-ee-short/results';v2=[base/'float-push-upstream-v2'/s for s in ['r0','r1','r2']]
manifest=dest/'cleanup_manifest.json'
if not a.delete:
    dest.mkdir(parents=True,exist_ok=False);seeds=set();inventory=[]
    def walk(v,key=''):
        if isinstance(v,dict):
            for k,x in v.items():walk(x,k)
        elif isinstance(v,list):
            for x in v:walk(x,key)
        elif 'seed' in key.lower() and isinstance(v,int) and not isinstance(v,bool):seeds.add(v)
    for root in roots:
        files=[]
        if root.exists():
            for f in root.rglob('*'):
                if not f.is_file() or f.is_symlink():continue
                files.append(dict(path=str(f.relative_to(root)),bytes=f.stat().st_size))
                if f.suffix in ('.json','.jsonl'):
                    try:
                        for value in ([json.loads(f.read_text())] if f.suffix=='.json' else [json.loads(s) for s in f.read_text().splitlines() if s.strip()]):walk(value)
                    except (ValueError,UnicodeError):pass
        inventory.append(dict(root=str(root),files=files,bytes=sum(f['bytes'] for f in files)))
    def keep(src,target):
        if not src.is_file():return
        out=dest/target;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out)
    failure=refined/'validation/policy_seed_8500'
    for f in failure.glob('*'):
        if f.is_file():keep(f,pathlib.Path('original_failure')/f.name)
    for name in ['comparison_seed_8500.mp4','failure_audit_final.json','offline_actions.json','plan.json','manifest.json','training/train.jsonl']:
        keep(refined/name,pathlib.Path('short_experiment')/name)
    for i,root in enumerate(v2):
        for name in ['plan.json','budget.jsonl','pilot_quality.json','parent_revision.json','seed_inventory.json','source.diff']:
            keep(root/name,pathlib.Path('v2')/('r'+str(i))/name)
        for f in (root/'source').rglob('*.py'):keep(f,pathlib.Path('v2')/('r'+str(i))/'source'/f.relative_to(root/'source'))
        for name in ['summary.json','config.json','metrics.csv','trajectory.npz']:
            keep(root/'pilot/expert_nominal'/name,pathlib.Path('v2')/('r'+str(i))/'expert'/name)
        keep(root/'analysis_v2/expert_nominal_h264.mp4',pathlib.Path('v2')/('r'+str(i))/'expert/rollout_h264.mp4')
    keep(v2[-1]/'budget.jsonl','budget.jsonl')
    for f in (v2[-1]/'diagnostics_v2').glob('*'):
        if f.suffix in ('.json','.png'):keep(f,pathlib.Path('diagnostics')/f.name)
    for f in (repo/'docs').rglob('*'):
        if f.is_file():keep(f,pathlib.Path('documents_before_cleanup')/f.relative_to(repo/'docs'))
    hashes={str(f.relative_to(dest)):hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.rglob('*') if f.is_file()}
    value=dict(roots=inventory,retained_sha256=hashes,removed=False)
    manifest.write_text(json.dumps(value,indent=2));(dest/'used_seeds.json').write_text(json.dumps(dict(seeds=sorted(seeds)),indent=2))
    print(json.dumps(dict(roots=len(roots),original_bytes=sum(x['bytes'] for x in inventory),retained_files=len(hashes),retained_bytes=sum(f.stat().st_size for f in dest.rglob('*') if f.is_file()))))
else:
    m=json.loads(manifest.read_text())
    for name,h in m['retained_sha256'].items():assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==h,name
    # Re-check every inventoried path and size; refuse later files or edits.
    for item in m['roots']:
        root=pathlib.Path(item['root']);now={str(f.relative_to(root)):f.stat().st_size for f in root.rglob('*') if f.is_file() and not f.is_symlink()} if root.exists() else {}
        assert now=={f['path']:f['bytes'] for f in item['files']},'Later changes under '+str(root)
    for item in m['roots']:
        root=pathlib.Path(item['root'])
        if root.exists():shutil.rmtree(root)
    m['removed']=True;manifest.write_text(json.dumps(m,indent=2));print(json.dumps(dict(deleted_roots=len(m['roots']),deleted_bytes=sum(x['bytes'] for x in m['roots']),retained_files=len(m['retained_sha256']))))
