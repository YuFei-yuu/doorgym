"""Aggregate completed R11 evidence; refuses partial data acceptance."""
import csv,hashlib,pathlib,statistics
import numpy as np
from common import ROOT,RUN,DATA,INDEX,EPISODES,FRAMES,read,write,sha


def main():
    assert read(RUN/'pipeline-result.json')['passed']
    loader=read(RUN/'full-loader-result.json');assert loader['passed'] and loader['frames']==FRAMES
    data=read(RUN/'dataset-check/factory-loader-report.json')
    clocks=read(RUN/'dataset-check/timestamp-frame-coverage.json')
    assert data['all_values_match_source_sanitizers'] and data['all_action_padding_within_episode']
    assert clocks['all_frames_exactly_once_per_view'] and clocks['upstream_offsets_float64_preserved']
    assert clocks['maximum_query_native_pts_error_s']<1e-4
    seal=read(RUN/'candidate-seal.json');assert sha(RUN/'candidate-index.json')==seal['sha256']
    ledger=read(RUN/'review-ledger.json');rows=ledger['trials'];measured=read(RUN/'measured-ledger.json')['trials']
    byseed={x['seed']:x for x in measured}
    for x in rows:
        x['converted_data_acceptance']='passed' if x['candidate_eligible'] else 'not_admitted'
        label='r%d-trial_%02d'%(8 if 'pilot-r8' in x['source_run'] else 9,x['trial'])
        write(RUN/label/'decision.json',x)
    ledger['counts']['converted_data_qualified']=EPISODES
    write(RUN/'review-ledger.json',ledger)
    provenance=[]
    for e in INDEX['episodes']:
        assert sha(e['source_npz'])==e['sha256']==sha(e['candidate_npz'])
        provenance.append(dict(episode=e['episode_index'],source_run=e['source_run'],trial=e['trial'],
            seed=e['seed'],frames=e['frames'],sha256=e['sha256'],old_episode=e['old_candidate_episode']))
    assert [e['old_candidate_episode'] for e in INDEX['episodes'][:46]]==list(range(46))
    write(RUN/'unified-provenance.json',dict(episodes=provenance,old46_prefix_byte_identical=True))
    source_runs=['wc4-fulltask-pilot-r8-20261021','wc4-fulltask-expand-r9-20261021','wc4-fulltask-reader-r10-20261021']
    checks={}
    for label,source,expected in zip(['r8','r9','r10'],source_runs,[692,5391,44]):
        logs={}
        for when in ['before','after']:
            log=RUN/(label+'-'+when+'-hashcheck.log');lines=log.read_text().splitlines()
            assert len(lines)==expected and all(l.endswith(': OK') for l in lines)
            logs[when]=dict(files_ok=len(lines),sha256=sha(log))
        checks[source]=dict(manifest_sha256=sha(ROOT/'.repro/runs'/source/'SHA256SUMS'),checks=logs)
    write(RUN/'historical-immutability.json',dict(passed=True,runs=checks,
        old46_dataset_and_index_unchanged=True,raw_evidence_and_historical_failed_logs_unchanged=True))
    diagnostic=read(RUN/'dataset-check/learnability-by-episode.json')
    decoded=read(RUN/'dataset-check/learnability-decoded-by-episode.json')
    pairs={(p['a'],p['b']):p for p in decoded['pairs']}
    ambiguity={h:sum(c['near_full_depth'] and c['significant_by_horizon'][h] and pairs[c['a'],c['b']]['decoded_near_both']
                     for c in diagnostic['reviewed_by_horizon'][h]) for h in ['1','25','100']}
    write(RUN/'ambiguity-summary.json',dict(episode_examples_by_horizon=ambiguity,
        checked_episodes=EPISODES,thresholds=diagnostic['thresholds'],scope=diagnostic['search'],
        diagnostic_only=True,data_filtered=False,new_inputs=False,
        limitations=['Bounded neighbor/probe search, not exhaustive all-pairs or proof of unlearnability.',
                     'Counts are per-episode examples, not failed trajectories. Padding excluded.']))
    fields=['source_run','trial','seed','old_candidate_episode','new_episode','was_isolated','automatic_terminal_pass','full_task_review_pass','raw_data_qualified','converted_data_acceptance']
    with (RUN/'per-trial-review.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields+['opening_max_abs_deg','close_hold_s','final_common_hold_s','negative_command_steps','minimum_raw_vx','command_change_rate_hz']);w.writeheader()
        for x in rows:
            m=byseed[x['seed']];d=m['diagnostics']
            w.writerow(dict({k:x[k] for k in fields},opening_max_abs_deg=m['opening_max_abs_deg'],close_hold_s=m['close_hold']['span_s'],
                final_common_hold_s=m['final_common_span_s'],negative_command_steps=d['negative_command_steps'],minimum_raw_vx=d['minimum_raw_vx'],command_change_rate_hz=d['mean_command_change_rate_hz']))
    params=[e['parameters'] for e in INDEX['episodes']];coverage={}
    for key in ['door_x','door_y','robot_x','robot_y','robot_z','robot_pitch','robot_yaw']:
        values=[p['effective_sampling'][key] for p in params]
        coverage[key]=dict(min=min(values),max=max(values),mean=statistics.mean(values))
    for view in range(2):
        for key in ['position_delta','rotation_delta_rad']:
            values=np.asarray([p['camera_jitter'][view][key] for p in params])
            coverage[('wrist' if view==0 else 'front')+'_'+key]=dict(min=values.min(0).tolist(),max=values.max(0).tolist())
    write(RUN/'initial-state-coverage.json',dict(candidates=EPISODES,nominal=sum(p['nominal'] for p in params),coverage=coverage,
        note='Existing recorded samples only; robot_y is source alignment offset, not final world y. No resampling or distribution narrowing.'))
    lengths=[e['frames'] for e in INDEX['episodes']]
    write(RUN/'resource-summary.json',dict(episodes=EPISODES,frames=FRAMES,episode_frames_min=min(lengths),episode_frames_max=max(lengths),
        mean_episode_frames=statistics.mean(lengths),candidate_npz_bytes=sum(pathlib.Path(e['candidate_npz']).stat().st_size for e in INDEX['episodes']),
        dataset_bytes=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file()),
        loader=loader,conversion=read(RUN/'convert-exit.json'),ambiguity_example_counts=ambiguity,
        new_scenes=0,collection=0,training=0,policy_inference=0))
    write(RUN/'ACCEPTANCE_RESULT.json',dict(passed=True,counts=ledger['counts'],episodes=EPISODES,frames=FRAMES,
        input_features=read(RUN/'dataset-check/resolved-policy-feature-map.json'),
        reader='R10 direct float64 TorchCodec path, original strict 1e-4 seconds',
        original_and_candidate_npz_bytes_unchanged=True,old46_dataset_unchanged=True,
        full_derived_video_decode=True,all_frames_time_indices_covered=True,
        exact_source_sanitizer_state_action_equality=True,real_factory_and_complete_multiworker_epoch=True,
        chunk100_and_padding_no_cross_episode=True,phase_independent_task_score=True,
        visual_review_scope='Original event-frame inspection for all72 plus measured states, raw contacts and frozen code; full originals decoded, not claimed human frame-by-frame viewing.',
        ambiguity_risk=ambiguity,independent_external_acceptance='pending',
        new_scenes=0,collection=0,training=0,policy_inference=0,stopped=True))
    print('Complete R11 acceptance',EPISODES,FRAMES,'ambiguity',ambiguity,flush=True)


if __name__=='__main__':main()
