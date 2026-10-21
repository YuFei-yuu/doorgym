"""Actual training factory/DataLoader checks. No ACT model or optimizer."""
import importlib.util,json,os,pathlib,sys,time,resource
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch,av
from torch.utils.data import DataLoader,Subset
ROOT=pathlib.Path('/data/zyf/video2door-wc4-act');R=ROOT/'.repro/runs/wc4-fulltask-pilot-r8-20261021'
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
 out=R/'dataset-check';out.mkdir(exist_ok=False)
 index=json.loads((R/'candidate-index.json').read_text());repo_id='local/wc4_fulltask_pilot_r8_20261021';path=R/'lerobot'/repo_id
 assert ROOT/'high-level/lerobot/src' in pathlib.Path(lerobot.__file__).parents
 cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id=repo_id,root=str(path),use_imagenet_stats=False),
     policy=ACTConfig(device='cpu',chunk_size=100,n_action_steps=50,
        use_action_loss_weight=False,plucker_conditioning=False,interaction_state_conditioning=False,
        camera_input_gating=False,handle_latent_aux=False,point_cloud_conditioning=False,end_signal_prediction=False),
     batch_size=16,num_workers=3)
 start=time.perf_counter();ds=make_dataset(cfg);factory_seconds=time.perf_counter()-start
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
   T=len(z['state']);assert len(loc)==T
   state_names=[str(k) for k in z['state_feature_names']];action_names=[str(k) for k in z['action_names']]
   st=sanitize_door_dp_state(z['state'].astype(np.float32),state_names=state_names,eps=DEFAULT_NEAR_ZERO_RATE_EPS)
   ac=sanitize_door_dp_action(z['action'].astype(np.float32),action_names=action_names,eps=DEFAULT_NEAR_ZERO_RATE_EPS)
   np.testing.assert_allclose(obs[base:base+T],st,atol=1e-7,rtol=0);np.testing.assert_allclose(acts[base:base+T],ac,atol=1e-7,rtol=0)
   timestamps=arr(ds.hf_dataset.select(range(base,base+T))['timestamp'])
   np.testing.assert_allclose(timestamps,np.arange(T)/25,atol=1e-4,rtol=0)
   phase=z['subtask_index'].reshape(-1);changes=np.flatnonzero(np.diff(phase)!=0)+1
   positions=set([0,1,max(0,T-101),max(0,T-100),max(0,T-2),T-1])
   for k in changes:
    positions.update([int(k)-1,int(k),min(T-1,int(k)+1)])
   closing=np.flatnonzero(phase==list(z['phase_names']).index('close'))
   if len(closing):positions.add(int(closing[len(closing)//2]))
   image_errors={}
   for view in ['front','wrist']:
    rawimages=z[view+'_masked_depth'];emb=rawimages[:,::20,::20].copy()
    if view=='front':compact=emb.reshape(T,-1)
    else:compact=np.concatenate([compact,emb.reshape(T,-1)],axis=1)
    kname=next(k for k in visual if view in k)
    maes=[]
    # Explicit beginning, all phase edges, closing, and final frame.
    for local in sorted(positions):
     sample=ds[base+local];im=arr(sample[kname]);assert im.shape==(3,480,640) and np.isfinite(im).all()
     assert im.min()>=0 and im.max()<=1
     maes.append(float(np.mean(np.abs(im-rawimages[local][None]/255.))))
    image_errors[view]=dict(max_mae=max(maes),mean_mae=float(np.mean(maes)),samples=len(maes))
   for local in sorted(positions):
    sample=ds[base+local];action=arr(sample['action']);pad=arr(sample['action_is_pad']).astype(bool)
    assert action.shape==(100,9) and pad.shape==(100,)
    take=np.minimum(local+np.arange(100),T-1);expected_pad=local+np.arange(100)>=T
    np.testing.assert_array_equal(pad,expected_pad);np.testing.assert_allclose(action,ac[take],atol=1e-7,rtol=0)
    np.testing.assert_allclose(arr(sample['observation.state']),st[local],atol=1e-7,rtol=0)
    assert int(arr(sample['episode_index']))==e;windows.append(base+local)
   metadata.append(dict(episode=e,trial=item['trial'],frames=T,start=base,end=base+T,checked_windows=len(positions),image_errors=image_errors))
   features.append(dict(state=st,action=ac,image=compact,episode=np.full(T,e,np.int32),local_frame=np.arange(T,dtype=np.int32),phase=phase.astype(np.int32)))
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
 videos=[]
 for p in path.rglob('*.mp4'):
  with av.open(str(p)) as f:
   st=f.streams.video[0];fps=float(st.average_rate);n=sum(1 for _ in f.decode(video=0));codec=st.codec_context.name
  assert fps==25;videos.append(dict(path=str(p),frames=n,fps=fps,codec=codec,fully_decoded=True,bytes=p.stat().st_size))
 for k in visual:assert sum(x['frames'] for x in videos if k in x['path'])==len(ds)
 arrays={k:np.concatenate([f[k] for f in features]) for k in features[0]}
 np.savez_compressed(out/'alias-features.npz',**arrays)
 report=dict(factory='lerobot.datasets.factory.make_dataset',lerobot_path=lerobot.__file__,no_model_or_optimizer_created=True,episodes=metadata,total_frames=len(ds),features=ds.features,
  chunk_size=100,all_values_match_source_sanitizers=True,all_action_padding_within_episode=True,all_image_shapes_pass=True,windows=len(windows),batches=batches,
  factory_seconds=factory_seconds,loader_seconds=loader_seconds,loader_windows_per_second=loaded/loader_seconds,loader_batch_size=16,loader_workers=3,
  peak_main_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  videos=videos,dataset_bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),candidate_npz_bytes=sum(pathlib.Path(i['candidate_npz']).stat().st_size for i in index['episodes']))
 (out/'factory-loader-report.json').write_text(json.dumps(report,indent=2,default=str)+'\n');print(json.dumps(report,indent=2,default=str),flush=True)


if __name__=='__main__':main()
