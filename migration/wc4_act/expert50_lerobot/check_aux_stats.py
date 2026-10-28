"""Check remaining author scalar-feature statistics without decoding more images."""
import sys,json
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
ROOT=Path('/data/zyf/video2door-wc4-act');RUN=ROOT/'.repro/runs/wc4-expert50-lerobot-20261029';DATA=RUN/'lerobot/local/wc4_expert50_20261029'
from lerobot.datasets.compute_stats import get_feature_stats,aggregate_stats
index=json.loads((RUN/'candidate-index.json').read_text());stored=json.loads((DATA/'meta/stats.json').read_text())
keys=['frame_index','subtask_index','index','episode_index','task_index','loss.action_weight','timestamp']
table=pq.read_table(sorted((DATA/'data').rglob('*.parquet'))[0]).to_pydict()
aggregate=None
for e,item in enumerate(index['episodes']):
 t=item['frames'];start=item['dataset_from_index'];end=item['dataset_to_index']
 with np.load(item['source_npz'],allow_pickle=True) as z:phase=np.asarray(z['subtask_index'],dtype=np.int64).reshape(t,1)
 arrays=dict(frame_index=np.arange(t),subtask_index=phase,index=np.arange(start,end),episode_index=np.full(t,e),
  task_index=np.asarray(table['task_index'][start:end]),timestamp=np.asarray([i/25 for i in range(t)]))
 arrays['loss.action_weight']=np.asarray(table['loss.action_weight'][start:end],dtype=np.float32).reshape(t,1)
 stats={k:get_feature_stats(v,axis=0,keepdims=v.ndim==1) for k,v in arrays.items()}
 aggregate=aggregate_stats([aggregate,stats]) if aggregate is not None else stats
errors={}
for key in keys:
 errors[key]={}
 for field,value in aggregate[key].items():
  actual=np.asarray(stored[key][field]);np.testing.assert_array_equal(actual,value)
  errors[key][field]=float(np.max(abs(actual-value)))
assert set(stored)==set(keys)|{'observation.state','action','observation.images.front_masked_depth','observation.images.wrist_masked_depth'}
(RUN/'aux-statistics-check.json').write_text(json.dumps(dict(passed=True,remaining_numeric_keys=keys,all_fields_exact=True,maximum_absolute_errors=errors,method='Original get_feature_stats and sequential aggregate_stats; source phase/frame/episode/global counters and exact author timestamps; stored task IDs/action-loss weights; no image reread or model'),indent=2)+'\n')
print('All remaining seven scalar-feature statistics exactly verified')
