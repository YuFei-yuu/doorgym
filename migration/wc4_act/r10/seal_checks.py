"""Read-only source comparison and aggregation after completed R10 checks."""
from common import *
import numpy as np
import pyarrow.parquet as pq


def main():
    expected_manifest = '2c763c551ce777ba61c6ee2f0801f81580b7ef459c05cd6697bb2f97c9d269b0'
    assert sha(OLD/'SHA256SUMS') == expected_manifest
    listed = {line[66:] for line in (OLD/'SHA256SUMS').read_text().splitlines()}
    actual = {str(p.relative_to(OLD)) for p in OLD.rglob('*') if p.is_file()}
    assert actual == listed | {'SHA256SUMS'}, (actual-listed, listed-actual)
    logs = {}
    for name in ['r9-before-hashcheck.log', 'r9-after-hashcheck.log']:
        lines = (RUN/name).read_text().splitlines()
        assert len(lines) == 5391 and all(x.endswith(': OK') for x in lines)
        logs[name] = dict(files_ok=len(lines), failures=0, sha256=sha(RUN/name))
    candidate_sha = sha(OLD/'candidate-index.json')
    assert candidate_sha == '5aaa93578ae5a4752b4ae8f7267344f4958bf7756ef9a106138afb9f35cccffa'
    write(RUN/'data-immutability.json', dict(passed=True, source_manifest_sha256=expected_manifest,
        source_manifest_matches_prior_local_delivery=True, candidate_index_sha256=candidate_sha,
        payload_files=5391, total_files_including_manifest=len(actual), inventory_unchanged=True,
        checks=logs, note='Original R9 acceptance/stop/error files remain unchanged; new R10 acceptance is separate.'))

    # These arrays were generated directly from every original NPZ by the
    # unchanged author sanitizer in check_dataset.py, before decoding images.
    source = np.load(RUN/'dataset-check/alias-features.npz')
    rows = pq.read_table(sorted((DATA/'data').rglob('*.parquet')),
                         columns=['index', 'observation.state', 'action']).sort_by('index')
    np.testing.assert_array_equal(np.asarray(rows['index']), np.arange(102043))
    exact = {}
    for key, source_key in [('observation.state', 'state'), ('action', 'action')]:
        converted = np.stack(rows[key].to_pylist()).astype(np.float32)
        original = source[source_key]
        assert converted.shape == original.shape == (102043, 9)
        np.testing.assert_array_equal(converted, original)
        exact[key] = dict(shape=list(converted.shape), dtype=str(converted.dtype),
                          exact_equal=True, maximum_absolute_error=float(np.max(abs(converted-original))))
    write(RUN/'exact-sanitized-values.json', dict(passed=True, values=exact,
        source='Every original candidate NPZ, same author sanitizer; alias-features.npz produced by check_dataset.py.'))

    raw = read(RUN/'dataset-check/learnability-by-episode.json')
    decoded = read(RUN/'dataset-check/learnability-decoded-by-episode.json')
    pairs = {(p['a'],p['b']):p for p in decoded['pairs']}
    counts = {}
    for horizon in ['1', '25', '100']:
        candidates = raw['reviewed_by_horizon'][horizon]
        good = [c for c in candidates if c['near_full_depth'] and
                c['significant_by_horizon'][horizon] and pairs[c['a'],c['b']]['decoded_near_both']]
        counts[horizon] = dict(representatives_checked=len(candidates),
            original_and_training_decoded_near_significant=len(good),
            retained_r8_episodes=sum(c['episode_a']<6 for c in good),
            new_r9_episodes=sum(c['episode_a']>=6 for c in good))
    write(RUN/'ambiguity-summary.json', dict(horizon_counts=counts,
        decoded_unique_pairs=decoded['checked_pairs'],
        decoded_near_unique_pairs=sum(p['decoded_near_both'] for p in pairs.values()),
        scope=raw['search'], thresholds=raw['thresholds'],
        limitations=['Bounded neighbor and probe search, not exhaustive all-pairs.',
                     'Near observations are not identical; these examples do not prove unlearnability.',
                     'Zero current-action representatives does not prove absence of ambiguity.',
                     'Counts represent per-episode examples, not failed trajectories or all ambiguous pairs.',
                     'Padding excluded from action comparisons; diagnostic does not filter data.']))
    print('Immutable inventory, exact sanitized values and ambiguity aggregation passed', flush=True)


if __name__ == '__main__':
    main()
