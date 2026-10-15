"""Final checks/packaging only; no optimization, neural inference or simulation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main(root,repo):
    sys.path.insert(0,str(repo/'high-level/float_ik/learning'))
    from smallbatch30k import verify,NODES,OLD
    import collect_stop_tail as shared
    from diagnostic_integrity import sha256,materialize_external_links
    from stop_audit import inspect_video
    verify(root)
    plan=json.loads((root/'plan.json').read_text());freeze=json.loads((root/'training_source_freeze.json').read_text())['source_hashes']
    assert all(sha256(repo/n)==h for n,h in freeze.items())
    assert json.loads((root/'state.json').read_text())['phase']=='completed'
    native=json.loads((root/'online/full_history_verification.json').read_text())
    assert native['cloud_unique_training_rows']==3000 and native['native_metrics_match']
    tasks=json.loads((root/'final_analysis/tasks.json').read_text())
    assert [v['updates'] for v in tasks]==NODES and all(v['frames']==2500 and v['execution_ok'] for v in tasks)
    assert json.loads((root/'online/finished.json').read_text())['verified']
    stage_success=[p for p in (root/'stages').glob('*.json') if '.attempt_' not in p.name]
    attempts=[json.loads(p.read_text()) for p in (root/'stages').glob('*.attempt_*.json')]
    assert len(stage_success)==42 and len(attempts)==42 and all(x['passed'] for x in attempts)
    expected=['step_000000.pt',*['step_%06d.pt'%n for n in NODES]]
    assert sorted(p.name for p in (root/'training/checkpoints').glob('step_*.pt'))==sorted(expected)
    checks={}
    for n in NODES:
        folder=root/'pilot/evaluation'/('step_%06d'%n)/'seed_16040'
        for kind,p in [('original',folder/'rollout.mp4'),('expert_pair',root/'comparison'/('step_%06d'%n)/'expert_policy.mp4')]:
            check=inspect_video(p,1250);assert check['passed'];checks['%d_%s'%(n,kind)]=check
    checks['reference']=inspect_video(root/'reference/rollout.mp4',723);assert checks['reference']['passed']
    preservation=shared.verify_inputs(plan);assert preservation['passed']
    links=materialize_external_links(root);(root/'external_log_links.json').write_text(json.dumps(links,indent=2)+'\n')
    diff=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=repo)
    names=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=repo,text=True).splitlines()
    for name in names:
        if name.startswith(('high-level/','docs/')):
            diff+=subprocess.run(['git','diff','--binary','--no-index','/dev/null',name],cwd=repo,stdout=subprocess.PIPE).stdout
    (root/'final_uncommitted.diff').write_bytes(diff)
    # Separate this round from the previous uncommitted A/B implementation.
    previous=OLD/'sources/high-level/dp/train_door_dp.py'
    current=repo/'high-level/dp/train_door_dp.py'
    this_round=subprocess.run(['git','diff','--binary','--no-index',str(previous),str(current)],cwd=repo,stdout=subprocess.PIPE).stdout
    for name in ['high-level/dp/test_random_initialization.py','high-level/float_ik/learning/smallbatch30k.py',
                 'high-level/float_ik/learning/smallbatch30k_analysis.py','high-level/float_ik/learning/smallbatch30k_online.py',
                 'docs/FLOAT_PUSH_SMALLBATCH30K.md']:
        this_round+=subprocess.run(['git','diff','--binary','--no-index','/dev/null',name],cwd=repo,stdout=subprocess.PIPE).stdout
    (root/'revision_this_round.diff').write_bytes(this_round)
    (root/'final_git_status.txt').write_bytes(subprocess.check_output(['git','status','--short','--branch'],cwd=repo))
    for name in ['FLOAT_PUSH_SMALLBATCH30K.md','FLOAT_PUSH_TRAINING_NEXT.md']:
        shutil.copy2(repo/'docs'/name,root/name)
    processes=subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True)
    active=[line for line in processes.splitlines() if root.name in line and 'python -c' not in line and
        any(v in line for v in ['smallbatch30k.py --root','train_door_dp.py --root','wholebody_float.py --headless','smallbatch30k_online.py --root'])]
    assert not active
    (root/'final_process_verification.json').write_text(json.dumps(dict(this_run_processes=active,all_exited=True),indent=2)+'\n')
    evidence=dict(run=str(root),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        optimizer_updates=30000,batch_size=8,seed=42,training_only_stats=True,initialization='random_no_old_state',
        initialization_provenance=json.loads((root/'random_initialization_provenance.json').read_text()),
        checkpoints=json.loads((root/'final_analysis/checkpoint_hashes.json').read_text()),
        tasks=tasks,planned_attempts=6,actual_attempts=6,joint_successes=sum(v['joint'] for v in tasks),
        all42_stages_succeeded=True,all_source_frozen_hashes_match=True,input_preservation=preservation,
        videos=checks,online=native,tests=json.loads((root/'delivery_tests.json').read_text()),
        postprocessing_revision='postprocessing_revisions/0001',
        anomaly_notes=['One read-only tool approval deadline timeout, retry succeeded, training unaffected.',
            'Legacy short candidate drift field retained; qualified2-second drift isNA without full hold. Stop outcomes unchanged.'],
        no_repeat_training_or_simulation=True,no_heldout=True,no_additional_budget=True,no_commit_or_push=True,
        imitation_acceptance_threshold=None,
        conclusion='Some expert-observation physical errors improved; complete trajectory imitation and autonomous stop not achieved in this run. No claim of impossibility or generalization.')
    (root/'FINAL_EVIDENCE.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    (root/'visual_qa/visual_review.json').write_text(json.dumps(dict(reviewer='Codex visual inspection; independent final acceptance reserved',
        images=['30k_pair_keyframes.png','comparison/step_030000/true_time_full.png'],
        observations=['Policy remains at door much later than pure10D reference.','Reference end is explicitly black, not a held frame.',
            'Final motion/target tracking and commanded gripper/orientation differ clearly from reference.']),indent=2)+'\n')
    excluded={'delivery_sha256.json','SHA256SUMS'}
    hashes={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in excluded}
    (root/'delivery_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (root/'SHA256SUMS').write_text(''.join('%s  %s\n'%(h,n) for n,h in hashes.items()))
    print(json.dumps(dict(files=len(hashes),bytes=sum((root/n).stat().st_size for n in hashes),updates=30000,
        planned_attempts=6,joint_successes=evidence['joint_successes'],frozen_source=True,old_evidence_preserved=True),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repo',required=True);a=p.parse_args();main(Path(a.root).resolve(),Path(a.repo).resolve())
