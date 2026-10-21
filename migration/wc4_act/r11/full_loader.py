"""One complete training-style multiworker traversal; no policy/model."""
from common import *
import time,resource
import numpy as np,torch
from torch.utils.data import DataLoader
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.video_utils import _default_decoder_cache


def rss(pid):
 try:
  line=next(l for l in pathlib.Path('/proc/%d/status'%pid).read_text().splitlines() if l.startswith('VmRSS:'))
  return int(line.split()[1])
 except (FileNotFoundError,StopIteration):return 0


def main():
 cfg=config();ds=make_dataset(cfg);assert len(ds)==FRAMES and torch.get_default_dtype()==torch.float32
 state=np.asarray(ds.hf_dataset['observation.state']);action=np.asarray(ds.hf_dataset['action'])
 if state.ndim==1:state=np.stack(state)
 if action.ndim==1:action=np.stack(action)
 ep=np.asarray(ds.hf_dataset['episode_index']).reshape(-1)
 ends=np.empty(len(ds),np.int64)
 for e in range(EPISODES):ids=np.flatnonzero(ep==e);ends[ids]=ids[-1]
 visual=[k for k in ds.features if k.startswith('observation.images.')]
 assert len(visual)==2
 _default_decoder_cache.clear()
 generator=torch.Generator().manual_seed(1000)
 loader=DataLoader(ds,batch_size=cfg.batch_size,num_workers=cfg.num_workers,shuffle=True,pin_memory=False,drop_last=False,prefetch_factor=2,generator=generator)
 write(RUN/'full-loader-settings.json',dict(batch_size=16,num_workers=3,shuffle=True,pin_memory=False,device='cpu',drop_last=False,prefetch_factor=2,generator_seed=1000,
  chunk100=True,implementation='torch.utils.data.DataLoader on real make_dataset output; same nonstreaming ACT loader settings; CPU means pin_memory=False',cfg=cfg.to_dict()))
 seen=np.zeros(len(ds),np.uint8);batches=0;count=0;peak_parent=0;peak_workers=0;peak_total=0
 start=time.perf_counter();iterator=iter(loader)
 workers=[w.pid for w in iterator._workers]
 for batch in iterator:
  ids=batch['index'].numpy().reshape(-1).astype(np.int64);assert np.all(seen[ids]==0) and len(np.unique(ids))==len(ids)
  np.add.at(seen,ids,1);n=len(ids);assert n<=16
  assert tuple(batch['action'].shape)==(n,100,9) and batch['action'].dtype==torch.float32
  assert tuple(batch['observation.state'].shape)==(n,9) and batch['observation.state'].dtype==torch.float32
  np.testing.assert_array_equal(batch['episode_index'].numpy().reshape(-1),ep[ids])
  future=ids[:,None]+np.arange(100);pad=future>ends[ids,None];take=np.minimum(future,ends[ids,None])
  np.testing.assert_array_equal(batch['action_is_pad'].numpy(),pad)
  np.testing.assert_array_equal(batch['action'].numpy(),action[take])
  np.testing.assert_array_equal(batch['observation.state'].numpy(),state[ids])
  for key in visual:
   im=batch[key];assert im.shape==(n,3,480,640) and im.dtype==torch.float32
   assert torch.isfinite(im).all() and im.min()>=0 and im.max()<=1
  if batches%16==0:
   parent=rss(os.getpid());children=sum(rss(p) for p in workers)
   peak_parent=max(peak_parent,parent);peak_workers=max(peak_workers,children);peak_total=max(peak_total,parent+children)
  count+=n;batches+=1
  if batches%128==0:print('full loader',count,'/',len(ds),'%.2f samples/s'%(count/(time.perf_counter()-start)),flush=True)
 elapsed=time.perf_counter()-start
 assert count==len(ds) and np.all(seen==1) and torch.get_default_dtype()==torch.float32
 np.save(RUN/'full-loader-frame-coverage.npy',seen)
 write(RUN/'full-loader-result.json',dict(passed=True,frames=count,episodes=EPISODES,batches=batches,last_batch_size=n,elapsed_s=elapsed,samples_per_second=count/elapsed,
  each_index_exactly_once=True,shuffled_order=True,all_action_chunks_and_padding_within_episode=True,all_state_action_values_match=True,all_images_float32_finite=True,
  peak_sampled_parent_rss_kib=peak_parent,peak_sampled_workers_rss_sum_kib=peak_workers,peak_sampled_total_rss_kib=peak_total,main_ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  memory_sampling_every_batches=16,num_workers=3,batch_size=16,prefetch_factor=2,image_dtype='torch.float32',default_dtype='torch.float32',errors=[],model_constructed=False,training=False,policy_inference=False,
  limitations=['CPU data-reader traversal, not a GPU training benchmark.','RSS sampled every16 batches includes shared pages; not proportional set size or guaranteed instantaneous peak.']))
 print('Complete pass',count,batches,elapsed,flush=True)


if __name__=='__main__':main()
