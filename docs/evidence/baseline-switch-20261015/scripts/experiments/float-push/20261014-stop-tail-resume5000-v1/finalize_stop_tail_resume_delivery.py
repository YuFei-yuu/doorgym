"""Finalize evidence only; no training, policy, simulator, or input mutation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/data/zyf/video2door')
ROOT = REPO/'experiments/float-push/20261014-stop-tail-resume5000-v1'
sys.path.insert(0, str(REPO/'high-level/float_ik/learning'))
import collect_stop_tail as shared
from diagnostic_integrity import sha256, materialize_external_links, assert_no_active


def save(name, value):
    shared.save(ROOT/name, value)


plan = json.loads((ROOT/'plan.json').read_text())
original = json.loads((ROOT/'source_revisions/0002/parent_plan.before.json').read_text())
revision = json.loads((ROOT/'source_revisions/0002/manifest.json').read_text())
source_results = {}
for name, expected in plan['source_hashes'].items():
    active = ROOT/revision['changes'][name]['active_snapshot'] if name in revision['changes'] else ROOT/'sources'/name
    source_results[name] = dict(current=sha256(REPO/name) == expected,
                                active_snapshot=sha256(active) == expected,
                                original_snapshot=sha256(ROOT/'sources'/name) == original['source_hashes'][name])
source_passed = all(all(row.values()) for row in source_results.values())
save('final_source_verification.json', dict(passed=source_passed, files=len(source_results),
     original_snapshots_preserved=True, authorized_active_revision=2, checks=source_results, at=time.time()))
if not source_passed:
    raise RuntimeError('Final source integrity failure; stop delivery')

preservation = dict(old_run=shared.verify_inputs(plan),
                    data_and_references=shared.verify_inputs(json.loads((ROOT/'pilot/plan.json').read_text())))
save('final_preservation_verification.json', dict(passed=all(v['passed'] for v in preservation.values()), checks=preservation, at=time.time()))
if not all(v['passed'] for v in preservation.values()):
    raise RuntimeError('Final old-input integrity failure; stop delivery')

checkpoints = {}
videos = {}
for step in (2000, 3000, 5000):
    model = ROOT/'pilot/training/checkpoints'/('step_%06d.pt' % step)
    audited = json.loads((ROOT/'pilot'/('checkpoint_%06d.json' % step)).read_text())
    checkpoints[str(step)] = dict(sha256=sha256(model), matches_audit=sha256(model) == audited['sha256'],
                                  readonly=(model.stat().st_mode & 0o222) == 0, audit_passed=audited['passed'])
    for seed in (16040, 16041, 16042):
        folder = ROOT/'pilot/evaluation'/('step_%06d' % step)/('seed_%d' % seed)
        result = json.loads((folder/'video_verification.json').read_text())
        videos[str(folder.relative_to(ROOT))] = dict(passed=result['passed'] and result['decoded'],
              decoded_frames=result['decoded_frames'], expected_frames=result['expected_frames'],
              video_sha256=sha256(folder/'rollout.mp4'))
assert all(r['matches_audit'] and r['readonly'] and r['audit_passed'] for r in checkpoints.values())
assert len(videos) == 9 and all(v['passed'] and v['decoded_frames'] == v['expected_frames'] == 1250 for v in videos.values())
assert not list((ROOT/'pilot/training/checkpoints').glob('step_00[6-9]*.pt'))
save('final_artifact_verification.json', dict(passed=True, checkpoints=checkpoints, videos=videos, at=time.time()))

assert_no_active(ROOT)
assert_no_active(ROOT/'pilot')
ps = subprocess.check_output(['ps', '-eo', 'pid,ppid,args'], text=True)
owned = [line for line in ps.splitlines() if ROOT.name in line and 'finalize_stop_tail_resume_delivery.py' not in line]
save('final_process_verification.json', dict(passed=not owned, matching_processes=owned, task_gpu=1,
     gpu_snapshot=subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], text=True),
     at=time.time(), note='No other process operated; finalizer is CPU-only and excluded from process match.'))
if owned:
    raise RuntimeError('Owned process still active: '+repr(owned))

environments = {}
for label, relative in [('train', '.repro/envs/door-dp-train'), ('gym', '.repro/envs/b1z1-blackwell')]:
    python = REPO/relative/'bin/python'
    environments[label] = dict(python=str(python), version=subprocess.check_output([str(python), '--version'], text=True).strip(),
           packages=subprocess.check_output([str(python), '-m', 'pip', 'freeze', '--all'], text=True).splitlines())
save('environment_evidence.json', dict(environments=environments, at=time.time()))
save('git_identity.json', dict(server_head=shared.git('rev-parse', 'HEAD').decode().strip(),
      local_head='5dd64f2dc6142ca0f2855561d165b5d0b31e5e1c',
      starting_server_head=plan['server_head'], original_author_reference_commit=plan['source_commit'],
      status=shared.git('status', '--short', '--branch').decode(),
      no_commit_or_push=True, at=time.time()))

finished = json.loads((ROOT/'online/finished.json').read_text())
table = json.loads((ROOT/'online/final_table_verification.json').read_text())
assert finished['verified'] and finished['verified_cloud_step'] == 5000 and table['passed']
external = materialize_external_links(ROOT)
internal = []
for link in sorted(ROOT.rglob('*')):
    if link.is_symlink():
        target = link.resolve(strict=True)
        target.relative_to(ROOT)
        relative = os.path.relpath(target, link.parent)
        old = os.readlink(link)
        if old != relative:
            link.unlink()
            link.symlink_to(relative)
        internal.append(dict(path=str(link.relative_to(ROOT)), target=relative, original=old))
save('materialized_links.json', dict(external_files=external, internal_links=internal,
     note='Only new-run SDK log links changed for portable delivery; old inputs unchanged.'))

manifest = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(ROOT.rglob('*'))
            if path.is_file() and path.name not in ('delivery_sha256.json', 'SHA256SUMS')}
save('delivery_sha256.json', manifest)
manifest['delivery_sha256.json'] = sha256(ROOT/'delivery_sha256.json')
(ROOT/'SHA256SUMS').write_text(''.join('%s  %s\n' % (value, name) for name, value in sorted(manifest.items())))
print(json.dumps(dict(passed=True, source_files=len(source_results), old_files=len(plan['input_hashes']),
      data_files=len(preservation['data_and_references']['inputs']), delivery_files=len(manifest),
      SHA256SUMS_sha256=sha256(ROOT/'SHA256SUMS'), checkpoints=checkpoints, videos=len(videos))))
