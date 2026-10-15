"""Evidence finalization only: no training, policy inference or simulation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main(root,repo):
    sys.path.insert(0,str(repo/'high-level/float_ik/learning'))
    import collect_stop_tail as shared
    from diagnostic_integrity import sha256,materialize_external_links
    from stop_audit import inspect_video
    plan=json.loads((root/'plan.json').read_text())
    freeze=json.loads((root/'training_source_freeze.json').read_text())['source_hashes']
    analysis='high-level/float_ik/learning/branch16040_analysis.py'
    active_matches={n:sha256(repo/n)==h for n,h in plan['source_hashes'].items()}
    core_matches={n:sha256(repo/n)==h for n,h in freeze.items() if n!=analysis}
    assert all(active_matches.values()) and all(core_matches.values())
    preservation=shared.verify_inputs(plan);assert preservation['passed']
    assert json.loads((root/'state.json').read_text())['phase']=='completed'
    tasks=json.loads((root/'final_analysis_v2/closed_loop_tasks.json').read_text())
    assert len(tasks)==4 and all(t['complete_2500_steps'] for t in tasks)
    video_checks={}
    for t in tasks:
        key='%s_%d'%(t['branch'],t['new_updates'])
        video_checks[key]=inspect_video(t['video'],1250)
        assert video_checks[key]['passed']
    for label in ['expert_A','pure10D_B']:
        video_checks[label]=inspect_video(root/'reference'/label/'rollout.mp4',723)
        assert video_checks[label]['passed']
    for l in ['A','B']:
        steps=sorted(p.name for p in (root/l/'training/checkpoints').glob('step_*.pt'))
        assert steps==['step_000000.pt','step_000500.pt','step_001000.pt','step_002000.pt']
        assert json.loads((root/l/'online/finished.json').read_text())['verified']
    cloud=json.loads((root/'online_postprocessing_revision2/cloud_verification.json').read_text())
    assert all(v['postprocessing_step']==2000 for v in cloud.values())
    success_stages=[p.stem for p in (root/'stages').glob('*.json') if '.attempt_' not in p.name]
    attempts=[json.loads(p.read_text()) for p in (root/'stages').glob('*.attempt_*.json')]
    assert len(success_stages)==31 and all(x['passed'] for x in attempts)
    previous_links=json.loads((root/'external_log_links.json').read_text())
    links=previous_links+materialize_external_links(root)
    (root/'external_log_links.json').write_text(json.dumps(links,indent=2)+'\n')
    diff=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=repo)
    for name in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=repo,text=True).splitlines():
        if name.startswith(('high-level/','docs/')):
            diff+=subprocess.run(['git','diff','--binary','--no-index','/dev/null',name],cwd=repo,stdout=subprocess.PIPE).stdout
    (root/'final_uncommitted.diff').write_bytes(diff)
    for name in ['FLOAT_PUSH_OBSERVATION_BRANCH16040.md','FLOAT_PUSH_TRAINING_NEXT.md']:
        shutil.copy2(repo/'docs'/name,root/name)
    evidence=dict(experiment=str(root),branch_updates=2000,branches=['A','B'],
        task_attempts=4,task_joint_successes=sum(t['joint'] for t in tasks),
        tasks=tasks,checkpoint_hashes=json.loads((root/'checkpoint_hashes.json').read_text()),
        input_preservation=preservation,active_source_matches=active_matches,training_core_frozen_matches=core_matches,
        original_freeze_unchanged=True,analysis_revisions=['0001','0002'],
        video_checks=video_checks,all_v3_stage_attempts_succeeded=True,stages=len(success_stages),
        old_failures_preserved=['20261015-observation-branch16040-v1','20261015-observation-branch16040-v2'],
        no_extra_training_or_simulation=True,no_heldout_test=True,no_commit_or_push=True,
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        online_final=cloud,
        interpretation='Partial offline adaptation and B relative tracking/path improvements; full trajectory imitation and autonomous stop not achieved. Final project acceptance left to independent review.')
    (root/'FINAL_EVIDENCE.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    (root/'visual_qa/visual_review.json').write_text(json.dumps(dict(
        reviewed=['true_time_full_2000.png','four_way_2000_keyframes.png'],
        observations=['A passes early and continues far beyond the door, with large own target tracking error.',
            'B is visibly delayed relative to expert and replay; late through event does not establish stop.',
            'Ended references are black with explicit ending caption; no held frames used for metrics.'],
        reviewer='Codex visual inspection; not user or final project acceptance'),indent=2)+'\n')
    excluded={'delivery_sha256.json','SHA256SUMS'}
    hashes={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in excluded}
    (root/'delivery_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (root/'SHA256SUMS').write_text(''.join('%s  %s\n'%(h,n) for n,h in hashes.items()))
    print(json.dumps(dict(files=len(hashes),total_bytes=sum((root/n).stat().st_size for n in hashes),
        joint_successes=evidence['task_joint_successes'],frozen_core=True,preservation=True),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repo',required=True);a=p.parse_args()
    main(Path(a.root).resolve(),Path(a.repo).resolve())
