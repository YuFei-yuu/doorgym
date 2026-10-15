"""Finalize verified artifacts; never train, infer or simulate."""
import argparse,json,sys,subprocess,shutil,time
from pathlib import Path

def main(root,repo):
    sys.path[:0]=[str(repo/'high-level/float_ik/learning'),str(repo/'high-level/dp')]
    from lr_pair import OLD,SOURCE,EXPECTED,NODES,LRS
    from smallbatch30k import verify
    from diagnostic_integrity import sha256,materialize_external_links
    import collect_stop_tail as shared
    import torch
    torch.set_num_threads(2);verify(root)
    assert json.loads((root/'state.json').read_text())['phase']=='completed'
    tasks=json.loads((root/'final_analysis/tasks.json').read_text());assert len(tasks)==6 and all(v['frames']==2500 and v['execution_ok'] for v in tasks)
    inherited=json.loads((root/'inherited_upstream_source_audit.json').read_text())['unchanged_inherited_dependencies']
    for name,h in inherited.items():
        assert sha256(repo/name)==h
        target=root/'sources_inherited'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    source=torch.load(SOURCE,map_location='cpu',weights_only=False);assert sha256(SOURCE)==EXPECTED
    out=root/'final_analysis'
    (out/'inherited_statistics.json').write_text(json.dumps(dict(source_checkpoint=str(SOURCE),source_sha256=EXPECTED,recomputed=False,
        stats={k:v.tolist() for k,v in source['stats'].items()},source_statistics=json.loads((OLD/'statistics.json').read_text())),indent=2)+'\n')
    for label in LRS:
        cfg={}
        for step in NODES:
            p=root/label/'training/checkpoints'/('step_%06d.pt'%step);cfg[str(step)]=torch.load(p,map_location='cpu',weights_only=False)['config']
        (out/('config_'+label+'.json')).write_text(json.dumps(cfg,indent=2)+'\n')
    attempts=[];success=[]
    for r in [root,*[root/l for l in LRS]]:
        for p in (r/'stages').glob('*.json'):
            data=json.loads(p.read_text())
            if '.attempt_' in p.name:attempts.append(dict(path=str(p.relative_to(root)),**data))
            else:assert data['passed'];success.append(str(p.relative_to(root)))
    assert len(success)==45
    assert sum(1 for v in attempts if not v['passed'])==1
    assert all(not any((root/l/'training/checkpoints'/('step_%06d.pt'%s)).exists() for s in [41000,45000]) for l in LRS)
    for l in LRS:assert sorted(p.name for p in (root/l/'training/checkpoints').glob('step_*.pt'))==['step_032000.pt','step_035000.pt','step_040000.pt']
    preservation=shared.verify_inputs(json.loads((root/'plan.json').read_text()));assert preservation['passed']
    links=materialize_external_links(root);(root/'external_log_links.json').write_text(json.dumps(links,indent=2)+'\n')
    diff=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=repo)
    for name in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=repo,text=True).splitlines():
        if name.startswith(('high-level/','docs/')):diff+=subprocess.run(['git','diff','--binary','--no-index','/dev/null',name],cwd=repo,stdout=subprocess.PIPE).stdout
    (root/'final_uncommitted.diff').write_bytes(diff)
    own=subprocess.run(['git','diff','--binary','--no-index',str(OLD/'sources/high-level/dp/train_door_dp.py'),str(repo/'high-level/dp/train_door_dp.py')],stdout=subprocess.PIPE).stdout
    for name in ['high-level/dp/test_lr_resume.py','high-level/float_ik/learning/lr_pair.py','high-level/float_ik/learning/lr_pair_analysis.py','high-level/float_ik/learning/lr_pair_online.py','docs/FLOAT_PUSH_LR_PAIR.md']:
        own+=subprocess.run(['git','diff','--binary','--no-index','/dev/null',name],cwd=repo,stdout=subprocess.PIPE).stdout
    (root/'revision_this_round.diff').write_bytes(own)
    (root/'final_git_status.txt').write_bytes(subprocess.check_output(['git','status','--short','--branch'],cwd=repo))
    for name in ['FLOAT_PUSH_LR_PAIR.md','FLOAT_PUSH_TRAINING_NEXT.md']:shutil.copy2(repo/'docs'/name,root/name)
    lines=subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines()
    active=[s for s in lines if str(root) in s and any(v in s for v in ['lr_pair.py --root','lr_pair_online.py --root','train_door_dp.py --root','wholebody_float.py --headless']) and 'bash -c' not in s]
    assert not active
    (root/'final_process_verification.json').write_text(json.dumps(dict(all_exited=True,run_processes=active),indent=2)+'\n')
    labels=json.loads((root/'labeled_video_index.json').read_text());assert len(labels['records'])==6 and all(v['check']['passed'] for v in labels['records'])
    evidence=dict(caption_derivatives=labels,run=str(root),source_checkpoint=str(SOURCE),source_sha256=EXPECTED,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        full_state_resume=True,optimizer_reset=False,sampler_reset=False,rng_reset=False,stats_recomputed=False,only_experiment_variable='C lr1e-4 versus L lr1e-5',
        added_optimizer_updates_each=10000,cumulative_optimizer_updates_each=40000,new_rollouts=6,baseline_reused=True,no_new_baseline_simulation=True,
        checkpoint_hashes=json.loads((out/'checkpoint_hashes.json').read_text()),tasks=tasks,joint_successes=sum(v['joint'] for v in tasks),
        input_preservation=preservation,source_frozen_after_training=True,inherited_upstream_dependencies_verified=len(inherited),successful_stages=success,stage_attempts=attempts,
        online=json.loads((out/'online_verification.json').read_text()),
        engineering_errors=json.loads((root/'engineering_attempts.json').read_text()),
        monitoring_tool_anomalies=[json.loads(s) for s in (root/'monitoring_tool_anomalies.jsonl').read_text().splitlines()] if (root/'monitoring_tool_anomalies.jsonl').exists() else [],
        controlled_resume_tests=9,no_repeat_successful_training_or_simulation=True,no_heldout=True,no_additional_budget=True,
        no_commit_or_push=True,imitation_threshold=None,final_acceptance_reserved_for_independent_review=True)
    (root/'FINAL_EVIDENCE.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    # Checksum manifest is the final operation, after every report/revision has been saved.
    excluded={'delivery_sha256.json','SHA256SUMS'}
    hashes={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in excluded}
    (root/'delivery_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (root/'SHA256SUMS').write_text(''.join('%s  %s\n'%(h,n) for n,h in hashes.items()))
    print(json.dumps(dict(files=len(hashes),bytes=sum((root/n).stat().st_size for n in hashes),added_updates_each=10000,new_rollouts=6,joint_successes=evidence['joint_successes']),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repo',required=True);a=p.parse_args();main(Path(a.root).resolve(),Path(a.repo).resolve())
