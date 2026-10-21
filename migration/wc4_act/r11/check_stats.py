"""Recompute numerical statistics with the unchanged author statistics code."""
from common import *
import numpy as np
from lerobot.datasets.compute_stats import get_feature_stats,aggregate_stats


def main():
    z=np.load(RUN/'dataset-check/alias-features.npz')
    stats=None
    for e in range(EPISODES):
        take=z['episode']==e
        ep={name:get_feature_stats(z[source][take],axis=0,keepdims=False)
            for name,source in [('observation.state','state'),('action','action')]}
        stats=aggregate_stats([stats,ep]) if stats is not None else ep
    stored=read(DATA/'meta/stats.json');errors={}
    for key in stats:
        errors[key]={}
        for field,value in stats[key].items():
            actual=np.asarray(stored[key][field]);np.testing.assert_array_equal(actual,value)
            errors[key][field]=float(np.max(abs(actual-value)))
        assert stored[key]['count']==[FRAMES]
    images={}
    for key in ['observation.images.front_masked_depth','observation.images.wrist_masked_depth']:
        images[key]=stored[key]
        for field,value in stored[key].items():assert np.isfinite(np.asarray(value)).all()
    write(RUN/'statistics-check.json',dict(passed=True,episodes=EPISODES,frames=FRAMES,
        numeric_statistics_exact_author_recomputation=True,maximum_absolute_errors=errors,
        image_statistics=images,meta_stats_sha256=sha(DATA/'meta/stats.json'),
        method='Same get_feature_stats and sequential aggregate_stats as author converter, on every sanitized source state/action. Histogram quantiles retain author approximation; no new normalization method.'))
    print('All numerical metadata statistics exactly match author recomputation',flush=True)


if __name__=='__main__':main()
