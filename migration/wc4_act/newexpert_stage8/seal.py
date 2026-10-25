"""Post-check CPU provenance, coverage and output seal. No data mutation."""
import csv,hashlib,importlib.metadata,json,pathlib,shutil,subprocess,sys,time
ROOT=pathlib.Path('/data/zyf/video2door-wc4-act')
RUN=ROOT/'.repro/runs/wc4-newexpert-stage8-20261025'
DATA=RUN/'lerobot/local/wc4_newexpert_stage8_20261025'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()

assert read(RUN/'CHECKS-COMPLETE.json')['passed'] and not (RUN/'STOP.json').exists()
full=read(RUN/'full-loader-result.json');stats=read(RUN/'statistics-check.json')
assert full['passed'] and full['frames']==121305 and full['episodes']==50
assert stats['passed'] and stats['numeric_statistics_exact_author_recomputation'] and stats['image_statistics_exact_author_recomputation']
assert read(RUN/'precision-guard.json')['passed']
index=read(RUN/'candidate-index.json');binding=read(RUN/'source-binding.json')
seal_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
subprocess.run(['git','merge-base','--is-ancestor',binding['actual_commit'],seal_head],cwd=ROOT,check=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
for name,h in binding['source_sha256'].items():assert sha(ROOT/name)==h
old=ROOT/'.repro/runs/wc4-newexpert-stage7-20261025'
anchors=read(RUN/'stage7-frozen-input-sha.json')
for name,expected in anchors.items():assert sha(old/name)==expected
write(RUN/'input-freeze-anchor-check.json',dict(passed=True,independent_anchor='Accepted stage-seven local FINAL/server-SHA256.json',sha256=anchors))
assert sha(old/'candidate-index.json')==index['input_candidate_sha256']
assert sha(old/'source-review-decisions.json')==index['input_review_sha256']
assert sha(old/'unified-source-index.json')==index['input_source_index_sha256']
sources=[]
for e in index['episodes']:
 p=pathlib.Path(e['raw_npz']['path']);h=sha(p)
 assert h==e['raw_npz']['sha256'] and p.stat().st_size==e['raw_npz']['bytes']
 assert pathlib.Path(e['candidate_npz']).resolve()==p
 sources.append(dict(episode_index=e['episode_index'],source_id=e['source_id'],path=str(p),sha256=h,frames=e['frames'],unchanged=True))
write(RUN/'post-check-input-integrity.json',dict(passed=True,sources=sources,source_files=50,total_frames=121305))
coverage=read(RUN/'dataset-check/timestamp-frame-coverage.json')
assert coverage['all_frames_exactly_once_per_view'] and coverage['derived_decoded_frames']==242610
metadata=read(RUN/'dataset-check/factory-loader-report.json')
assert metadata['total_frames']==121305 and len(metadata['episodes'])==50
with (RUN/'SOURCE_TABLE.csv').open('w') as f:
 fields=['episode_index','source_id','seed','frames','dataset_from_index','dataset_to_index','raw_path','raw_sha256']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for e in index['episodes']:
  w.writerow({**{k:e[k] for k in fields[:6]},'raw_path':e['raw_npz']['path'],'raw_sha256':e['raw_npz']['sha256']})
files=[dict(path=str(p.relative_to(DATA)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(DATA.rglob('*')) if p.is_file()]
write(RUN/'dataset-SHA256.json',dict(dataset_path=str(DATA),repo_id=binding['repo_id'],files=files,bytes=sum(p['bytes'] for p in files),meta_stats_sha256=sha(DATA/'meta/stats.json')))
env={'python':sys.version,'executable':sys.executable}
for name in ['torch','numpy','av','torchcodec','datasets','pyarrow','lerobot']:
 try:env[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:env[name]='not in package metadata; actual source path in factory report'
write(RUN/'environment.json',env)
resources=[json.loads(l) for l in (RUN/'resource-monitor.jsonl').read_text().splitlines()]
minimum=min(r['free_bytes'] for r in resources)
changed=subprocess.check_output(['git','diff','--name-only','4d35e3226568d8dfc12e91372e84da68eb614235','HEAD'],cwd=ROOT,text=True).splitlines()
assert all(n.startswith('migration/wc4_act/newexpert_stage8/') for n in changed)
write(RUN/'acceptance-summary.json',dict(status='FROZEN_DATA_ACCEPTANCE_COMPLETE',actual_execution_commit=binding['actual_commit'],actual_seal_commit=seal_head,episodes=50,frames=121305,state_elements=121305*9,action_elements=121305*9,numeric_source_equality=True,statistics_exact=True,views=2,decoded_frames=242610,
  phase_and_tail_windows=metadata['windows'],maximum_query_native_pts_error_s=coverage['maximum_query_native_pts_error_s'],tolerance_s=1e-4,backend='torchcodec',
  full_loader=full,dataset_sha_index=sha(RUN/'dataset-SHA256.json'),statistics_sha256=sha(DATA/'meta/stats.json'),free_minimum_bytes=minimum,free_final_bytes=shutil.disk_usage(ROOT).free,
  reserve_preserved=minimum>=40*2**30,source_unchanged=True,protected_source_diff_empty=True,changed_files=changed,collection=0,simulation=0,policy_inference=0,training_updates=0,gpu_locks_held=False,monitor='run-scoped CPU monitor exited with driver',exceptions=[],exception_scope='Data conversion and numerical acceptance commands; peripheral transport notes in COMMANDS.md',time=time.time(),
  limitations=['R11 task qualification inherited from accepted stage-seven input; no new physical execution.','H264 is lossy; no pixel-byte identity claim.','CPU reader acceptance is not a training or GPU performance result.']))
print(json.dumps(read(RUN/'acceptance-summary.json'),indent=2))
