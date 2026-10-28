"""Actual training factory/DataLoader checks. No ACT model or optimizer."""
import importlib.util,json,os,pathlib,sys,time,resource
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch,av
from torch.utils.data import DataLoader,Subset
from common import ROOT, OLD as R, RUN, DATA, config, write
sys.path.insert(0,str(ROOT/'high-level/dp'))
import lerobot
from door_dp_common import sanitize_door_dp_action,sanitize_door_dp_state,DEFAULT_NEAR_ZERO_RATE_EPS
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.datasets.factory import make_dataset


def arr(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().numpy()
 return np.asarray(x)


def main():
 out=RUN/'dataset-check';out.mkdir(exist_ok=False);default_dtype=torch.get_default_dtype();assert default_dtype==torch.float32
 index=json.loads((R/'candidate-index.json').read_text());repo_id='local/wc4_expert50_20261029';path=R/'lerobot'/repo_id
 assert len(index['episodes'])==50 and sum(x['frames'] for x in index['episodes'])==77356
 assert ROOT/'high-level/lerobot/src' in pathlib.Path(lerobot.__file__).parents
 cfg=config()
 start=time.perf_counter();ds=make_dataset(cfg);factory_seconds=time.perf_counter()-start
 assert ds.video_backend=='torchcodec' and ds.tolerance_s==1e-4
 write(out/'resolved-reader.json',dict(backend=ds.video_backend,tolerance_s=ds.tolerance_s,comparison='strict < unchanged',default_dtype=str(torch.get_default_dtype())))
 (out/'factory-config.json').write_text(json.dumps(cfg.to_dict(),indent=2,default=str)+'\n')
 keys=list(ds.features);visual=[k for k in keys if k.startswith('observation.images.')]
 assert len(visual)==2 and 'observation.state' in keys
 assert set(k for k in keys if k.startswith('observation.'))==set(visual+['observation.state'])
 assert not any(k.startswith(('observation.camera_pose','observation.phase','observation.door','observation.contact')) for k in keys)
 print('Factory loaded',len(ds),'frames',visual,flush=True)
 episodes=arr(ds.hf_dataset['episode_index']).reshape(-1);obs=arr(ds.hf_dataset['observation.state']);acts=arr(ds.hf_dataset['action'])
 if obs.ndim==1:obs=np.stack(obs)
 if acts.ndim==1:acts=np.stack(acts)
 windows=[];metadata=[];starts=[];staging=[];features=[]
 for e,item in enumerate(index['episodes']):
  loc=np.flatnonzero(episodes==e);assert len(loc)>0 and np.all(np.diff(loc)==1);base=int(loc[0]);starts.append(base)
  raw=pathlib.Path(item['candidate_npz'])
  with np.load(raw,allow_pickle=True) as z:
   T=len(z['state']);assert len(loc)==T==item['frames']
   state_names=[str(k) for k in z['state_feature_names']];action_names=[str(k) for k in z['action_names']]
   sidecar=json.loads((path/'door_dp_feature_names.json').read_text())
   assert sidecar['state']==state_names and sidecar['action']==action_names
   assert ds.features['observation.state']['names']==state_names and ds.features['action']['names']==action_names
   assert sidecar['state_action_mode']=='joint_state9' and sidecar['action_frame']=='joint_command'
   assert sidecar['state_preprocess']['mode']==sidecar['action_preprocess']['mode']=='identity'
   assert sidecar['fps']==25 and sidecar['image_storage']=='video' and sidecar['video_codec']=='h264'
   st=sanitize_door_dp_state(z['state'].astype(np.float32),state_names=state_names,eps=DEFAULT_NEAR_ZERO_RATE_EPS)
   ac=sanitize_door_dp_action(z['action'].astype(np.float32),action_names=action_names,eps=DEFAULT_NEAR_ZERO_RATE_EPS)
   np.testing.assert_array_equal(obs[base:base+T],st);np.testing.assert_array_equal(acts[base:base+T],ac)
   np.testing.assert_array_equal(arr(ds.hf_dataset.select(range(base,base+T))['frame_index']).reshape(-1),np.arange(T))
   np.testing.assert_array_equal(arr(ds.hf_dataset.select(range(base,base+T))['index']).reshape(-1),np.arange(base,base+T))
   timestamps=arr(ds.hf_dataset.select(range(base,base+T))['timestamp'])
   np.testing.assert_allclose(timestamps,np.arange(T)/25,atol=1e-4,rtol=0)
   phase=z['subtask_index'].reshape(-1);changes=np.flatnonzero(np.diff(phase)!=0)+1
   positions=set([0,1,max(0,T-101),max(0,T-100),max(0,T-2),T-1])
   for k in changes:
    positions.update([int(k)-1,int(k),min(T-1,int(k)+1)])
   closing=np.flatnonzero(phase==list(z['phase_names']).index('close'))
   if len(closing):positions.add(int(closing[len(closing)//2]))
   image_errors={};rawviews={v:z[v+'_masked_depth'] for v in ['front','wrist']}
   for view in ['front','wrist']:
    rawimages=rawviews[view]
    kname=next(k for k in visual if view in k)
    maes=[];cross_maes=[]
    # Explicit beginning, all phase edges, closing, and final frame.
    for local in sorted(positions):
     sample=ds[base+local];im=arr(sample[kname]);assert im.shape==(3,480,640) and np.isfinite(im).all()
     assert im.min()>=0 and im.max()<=1
     maes.append(float(np.mean(np.abs(im-rawimages[local][None]/255.))))
     other=rawviews['wrist' if view=='front' else 'front'][local]
     cross_maes.append(float(np.mean(np.abs(im-other[None]/255.))))
    image_errors[view]=dict(max_mae=max(maes),mean_mae=float(np.mean(maes)),opposite_view_mean_mae=float(np.mean(cross_maes)),samples=len(maes),lossy_encoding=True,no_pixel_equality_requirement=True)
   for local in sorted(positions):
    sample=ds[base+local];action=arr(sample['action']);pad=arr(sample['action_is_pad']).astype(bool)
    assert action.shape==(100,9) and pad.shape==(100,)
    take=np.minimum(local+np.arange(100),T-1);expected_pad=local+np.arange(100)>=T
    np.testing.assert_array_equal(pad,expected_pad);np.testing.assert_array_equal(action,ac[take])
    np.testing.assert_array_equal(arr(sample['observation.state']),st[local])
    assert int(arr(sample['episode_index']))==e;windows.append(base+local)
   metadata.append(dict(episode=e,source_id=item['source_id'],frames=T,start=base,end=base+T,checked_windows=len(positions),image_errors=image_errors))
   features.append(dict(state=st,action=ac,episode=np.full(T,e,np.int32),local_frame=np.arange(T,dtype=np.int32),phase=phase.astype(np.int32)))
  print('episode',e,'all numeric values and phase windows passed',len(positions),flush=True)
 write(out/'checked-window-indices.json',dict(global_indices=windows,semantics='phase labels locate audits only, never task scoring or policy inputs'))
 print('All values, phase windows and padding verified',len(windows),flush=True)
 from lerobot.datasets.video_utils import _default_decoder_cache
 _default_decoder_cache.clear()  # Workers open their own actual decoders.
 # Same DataLoader class/settings as train; Subset selects audited windows.
 load_start=time.perf_counter();batches=0;loaded=0
 loader=DataLoader(Subset(ds,windows),num_workers=3,batch_size=16,shuffle=False,pin_memory=False,drop_last=False,prefetch_factor=2)
 for batch in loader:
  assert tuple(batch['action'].shape[1:])==(100,9)
  assert tuple(batch['action_is_pad'].shape[1:])==(100,)
  for k in visual:assert tuple(batch[k].shape[1:])==(3,480,640)
  loaded+=len(batch['action']);batches+=1
 loader_seconds=time.perf_counter()-load_start
 # Sequential full decode and complete timestamp/frame-index coverage.
 import pyarrow.parquet as pq
 tables=[pq.read_table(p).to_pylist() for p in sorted((path/'meta/episodes').rglob('*.parquet'))]
 stored={int(row['episode_index']):row for table in tables for row in table}
 for p in (path/'meta/episodes').rglob('*.parquet'):
  schema=pq.read_schema(p)
  for key in visual:assert str(schema.field('videos/'+key+'/from_timestamp').type)=='double'
 videos=[];native_pts={};native_coverage={};clock_arrays={}
 for file_index,p in enumerate(sorted(path.rglob('*.mp4'))):
  pts=[]
  with av.open(str(p)) as f:
   stream=f.streams.video[0];fps=float(stream.average_rate);codec=stream.codec_context.name
   assert fps==25
   for n,frame in enumerate(f.decode(video=0)):
    assert frame.pts is not None
    timestamp=float(frame.pts*frame.time_base);assert round(timestamp*fps)==n
    assert abs(timestamp-n/fps)<cfg.tolerance_s
    pts.append(timestamp)
  pts=np.asarray(pts,dtype=np.float64);assert np.all(np.diff(pts)>0)
  native_pts[str(p)]=pts;native_coverage[str(p)]=np.zeros(len(pts),np.uint8)
  clock_arrays['video%d_pts'%file_index]=pts
  clock_arrays['video%d_frame_indices'%file_index]=np.arange(len(pts),dtype=np.int64)
  videos.append(dict(path=str(p),frames=len(pts),fps=fps,codec=codec,fully_decoded=True,bytes=p.stat().st_size))
  print('Full derived video decoded',p,len(pts),flush=True)
 mapping=[];all_timestamps=arr(ds.hf_dataset['timestamp']).reshape(-1)
 for e,item in enumerate(index['episodes']):
  ids=np.flatnonzero(episodes==e);local_ts=all_timestamps[ids];ep=ds.meta.episodes[e]
  for key in visual:
   field='videos/'+key+'/from_timestamp';offset=ep[field];original=stored[e][field]
   assert isinstance(offset,(float,np.float64)), ('upstream offset lost precision',type(offset))
   assert offset==original
   first_frame=round(offset*25)
   # Reject earlier float32 loss in a CFR episode offset; this bound only
   # is a frame-grid representation check, not a relaxed read tolerance.
   grid_error=abs(offset-first_frame/25)
   assert grid_error<=64*max(abs(np.spacing(np.float64(offset))),np.finfo(np.float64).eps)
   query=np.asarray([offset+float(t) for t in local_ts],dtype=np.float64)
   selected=np.asarray([round(t*25) for t in query],dtype=np.int64)
   expected=first_frame+np.arange(len(ids));np.testing.assert_array_equal(selected,expected)
   video=str(path/ds.meta.get_video_file_path(e,key));pts=native_pts[video]
   assert selected.min()>=0 and selected.max()<len(pts)
   errors=abs(query-pts[selected]);assert np.all(errors<cfg.tolerance_s)
   np.add.at(native_coverage[video],selected,1)
   mapping.append(dict(episode=e,key=key,offset_type=type(offset).__name__,stored_offset_type='Parquet float64',offset=offset,offset_frame_grid_error_s=grid_error,
       frames=len(ids),first_video_frame=int(selected[0]),last_video_frame=int(selected[-1]),maximum_query_native_pts_error_s=float(errors.max()),all_indices_correct=True))
 for key in visual:assert sum(v['frames'] for v in videos if key in v['path'])==len(ds)
 assert all(np.all(count==1) for count in native_coverage.values()), 'missing/duplicate/cross-episode video indices'
 np.savez_compressed(out/'full-video-clocks.npz',**clock_arrays)
 write(out/'timestamp-frame-coverage.json',dict(total_dataset_frames=len(ds),views=2,derived_decoded_frames=sum(v['frames'] for v in videos),all_frames_exactly_once_per_view=True,
     upstream_offsets_float64_preserved=True,maximum_query_native_pts_error_s=max(m['maximum_query_native_pts_error_s'] for m in mapping),tolerance_s=cfg.tolerance_s,comparison='strict <',videos=videos,episode_mappings=mapping))
 from lerobot.datasets.utils import dataset_to_policy_features
 from lerobot.configs.types import FeatureType
 parsed=dataset_to_policy_features(ds.features);outputs={k:v for k,v in parsed.items() if v.type is FeatureType.ACTION};inputs={k:v for k,v in parsed.items() if k not in outputs}
 assert set(inputs)==set(visual+['observation.state']) and set(outputs)=={'action'}
 write(out/'resolved-policy-feature-map.json',dict(inputs={k:dict(type=v.type.value,shape=list(v.shape)) for k,v in inputs.items()},outputs={k:dict(type=v.type.value,shape=list(v.shape)) for k,v in outputs.items()},no_policy_constructed=True))
 assert torch.get_default_dtype()==default_dtype
 arrays={k:np.concatenate([f[k] for f in features]) for k in features[0]}
 np.savez_compressed(out/'alias-features.npz',**arrays)
 report=dict(factory='lerobot.datasets.factory.make_dataset',lerobot_path=lerobot.__file__,no_model_or_optimizer_created=True,episodes=metadata,total_frames=len(ds),features=ds.features,
  chunk_size=100,all_values_match_source_sanitizers=True,all_action_padding_within_episode=True,all_image_shapes_pass=True,windows=len(windows),batches=batches,
  factory_seconds=factory_seconds,loader_seconds=loader_seconds,loader_windows_per_second=loaded/loader_seconds,loader_batch_size=16,loader_workers=3,
  peak_main_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  videos=videos,dataset_bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),candidate_npz_bytes=sum(pathlib.Path(i['candidate_npz']).stat().st_size for i in index['episodes']))
 (out/'factory-loader-report.json').write_text(json.dumps(report,indent=2,default=str)+'\n');print(json.dumps(report,indent=2,default=str),flush=True)


if __name__=='__main__':main()
