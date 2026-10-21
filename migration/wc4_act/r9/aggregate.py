"""Final read-only provenance, coverage and resource summary."""
import collections, json, pathlib, shutil, subprocess
import numpy as np
from collect import ROOT, OLD, read, write, sha, verify

R=ROOT/'.repro/runs/wc4-fulltask-expand-r9-20261021'
verify(R)
attempts=[json.loads(l) for l in (R/'attempts.jsonl').open()]
index=read(R/'candidate-index.json');episodes=index['episodes'];manifest=read(R/'manifest.json')
assert len(attempts)<=64 and len(episodes)<=50
assert not read(R/'progress.json')['pending_visual']
assert len({a['seed'] for a in attempts})==len(attempts)
assert all(a['sim_seconds']<=180.000001 for a in attempts)
old=read(OLD/'candidate-index.json')['episodes']
for a,b in zip(old,episodes[:6]):
    assert a['sha256']==b['sha256']==sha(b['candidate_npz'])==sha(a['candidate_npz'])
provenance=[]
for i,e in enumerate(episodes):
    assert sha(e['candidate_npz'])==e['sha256']==sha(e['source_npz'])
    with np.load(e['candidate_npz'],allow_pickle=True) as raw:
        frames=len(raw['state']);assert raw['state'].shape==(frames,9) and raw['action'].shape==(frames,9)
    assert frames==e['frames']
    provenance.append(dict(episode=i,trial=e['trial'],source_run=e['source_run'],source_npz=e['source_npz'],candidate_npz=e['candidate_npz'],sha256=e['sha256'],frames=frames,seed=e['seed'],parameters=e['parameters']))
write(R/'unified-provenance.json',dict(episodes=provenance,retained_r8_trials=[0,1,2,3,5,6],excluded_r8_trials=[4,7]))


def coverage(trials):
    result={}
    for key in ['door_x','door_y','door_wall_x_offset','robot_x','robot_y','robot_z','robot_pitch','robot_yaw']:
        values=np.array([t['effective_sampling'][key] for t in trials])
        if len(values):result[key]=dict(min=float(values.min()),max=float(values.max()),mean=float(values.mean()),unit='rad' if key in ['robot_pitch','robot_yaw'] else 'm')
    for view in range(2):
        for key in ['position_delta','rotation_delta_rad']:
            values=np.array([t['camera_jitter'][view][key] for t in trials])
            if len(values):result[('wrist' if view==0 else 'front')+'_'+key]=dict(axis_min=values.min(0).tolist(),axis_max=values.max(0).tolist())
    return result


tried=[manifest['trials'][a['trial']] for a in attempts]
selected=[manifest['trials'][a['trial']] for a in attempts if a['status']=='candidate']
write(R/'initial-state-coverage.json',dict(frozen_distribution=coverage(manifest['trials']),attempted=coverage(tried),new_candidates=coverage(selected),
    scope='Empirical coverage only; selection includes task success and negative-command withholding, so candidate distribution may differ from attempted distribution. No range narrowing or seed replacement.'))
reviews=[read(R/('trial_%02d'%a['trial'])/'review/review.json') for a in attempts]
accepted=[r for r,a in zip(reviews,attempts) if a['status']=='candidate']
lengths=[e['frames'] for e in episodes]
assert sum(lengths[:6])==13301
stats=dict(attempts=len(attempts),scenes=sum(a['scene_started'] for a in attempts),automatic_terminal_passes=sum(a['automatic_terminal_pass'] for a in attempts),
    new_candidates=len(episodes)-6,total_candidates=len(episodes),retained_r8_frames=13301,total_candidate_frames=sum(lengths),
    status_counts=dict(collections.Counter(a['status'] for a in attempts)),unused_seeds=[s for s in manifest['seeds'] if s not in {a['seed'] for a in attempts}],
    task_failure_reasons=dict(collections.Counter(a['reason'] for a in attempts if not a['automatic_terminal_pass'])),
    candidate_frame_range=[min(lengths),max(lengths)],candidate_frame_mean=float(np.mean(lengths)),new_sim_seconds_range=[min(a['sim_seconds'] for a in attempts),max(a['sim_seconds'] for a in attempts)],
    new_candidate_min_lock_hold_s=min((r['hold']['span_s'] for r in accepted),default=None),new_candidate_min_final_common_hold_s=min((r['final_common_span_s'] for r in accepted),default=None),
    negative_command_cases=[dict(trial=a['trial'],seed=a['seed'],**a['negative_command_diagnostic']) for a in attempts if a['negative_command_diagnostic']['count']],
    critical_violations=[a for a in attempts if a.get('systemic_violations')],independent_external_acceptance='pending',no_training=True,no_policy_inference=True,
    stopped_reason='target_candidates_reached' if len(episodes)==50 else ('new_attempt_budget_exhausted' if len(attempts)==64 else 'explicit_stop_see_logs'))
assert not stats['critical_violations']
write(R/'COLLECTION_RESULT.json',stats)
write(R/'candidate-list.json',dict(episodes=provenance))
write(R/'failed-list.json',dict(attempts=[a for a in attempts if not a['automatic_terminal_pass']]))
write(R/'withheld-list.json',dict(attempts=[a for a in attempts if a['status'].startswith('withheld')],r8_excluded=[4,7]))
resources=dict(run_file_bytes=sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),raw_new_npz_bytes=sum(p.stat().st_size for p in R.glob('trial_*/raw/*.npz')),
    candidate_npz_bytes=sum(pathlib.Path(e['candidate_npz']).stat().st_size for e in episodes),candidate_frames=sum(lengths),free_server_bytes=shutil.disk_usage(ROOT).free,
    reservation=read(R/'capacity-reservation.json'),actual_new_sim_seconds=sum(a['sim_seconds'] for a in attempts),
    gpu_leases='groups/*/binding.json and gpu*-check*.txt; first trial via launch.sh',gpus=[1,2],max_concurrent_scenes=8,
    limitations=['No training or ACT inference performed. Runtime estimates do not establish convergence or robustness.','Run byte count is a snapshot before final hashing/packaging.'])
if (R/'dataset-check/factory-loader-report.json').exists():resources['factory_loader']=read(R/'dataset-check/factory-loader-report.json')
write(R/'resource-summary.json',resources)
print(json.dumps(stats,ensure_ascii=False,indent=2))
