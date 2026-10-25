"""Recompute numerical statistics with the unchanged author statistics code."""
from common import *
import numpy as np
from lerobot.datasets.compute_stats import get_feature_stats,aggregate_stats,sample_indices,auto_downsample_height_width


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
    image_stats=None
    for item in INDEX['episodes']:
        with np.load(item['candidate_npz'],allow_pickle=True) as raw:
            ep={}
            for view in ['front','wrist']:
                data=raw[view+'_masked_depth']; ids=sample_indices(len(data))
                # Same author sampling/downsampling as PNG-based compute_episode_stats. PNG is lossless.
                sampled=np.stack([auto_downsample_height_width(np.repeat(data[i][None],3,axis=0)) for i in ids])
                values=get_feature_stats(sampled,axis=(0,2,3),keepdims=True)
                ep['observation.images.'+view+'_masked_depth']={k:v if k=='count' else np.squeeze(v/255.,axis=0) for k,v in values.items()}
            image_stats=aggregate_stats([image_stats,ep]) if image_stats is not None else ep
        print('image stats',item['source_id'],flush=True)
    for key in image_stats:
        for field,value in image_stats[key].items():np.testing.assert_array_equal(np.asarray(stored[key][field]),value)
    images={}
    for key in ['observation.images.front_masked_depth','observation.images.wrist_masked_depth']:
        images[key]=stored[key]
        for field,value in stored[key].items():assert np.isfinite(np.asarray(value)).all()
    write(RUN/'statistics-check.json',dict(passed=True,episodes=EPISODES,frames=FRAMES,
        numeric_statistics_exact_author_recomputation=True,image_statistics_exact_author_recomputation=True,use_imagenet_stats=False,image_scaling="uint8 /255 to float32 in actual reader",state_action_preprocess="identity after unchanged sanitizer",maximum_absolute_errors=errors,
        image_statistics=images,meta_stats_sha256=sha(DATA/'meta/stats.json'),
        method='Same get_feature_stats and sequential aggregate_stats as author converter, on every sanitized source state/action. Histogram quantiles retain author approximation; no new normalization method.'))
    print('All numerical metadata statistics exactly match author recomputation',flush=True)


if __name__=='__main__':main()
