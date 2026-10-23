"""Frozen exact 8 uniform +2 per event class/batch; no dataset or loss changes."""
import pathlib,os,json,hashlib,time
import numpy as np
SCHEDULE=None

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze_schedule(run):
 w=json.loads((run/'windows.json').read_text());assert digest(run/'windows.json')==json.loads((run/'WINDOWS-FROZEN.json').read_text())['sha256']
 rng=np.random.default_rng(1000);indices=np.empty((25000,16),dtype=np.int64);groups=np.empty_like(indices)
 for u in range(25000):
  ids=list(rng.integers(0,159529,8));g=[0]*8
  for j,pool in enumerate(w['pools']):
   ids.extend(rng.choice(pool,2));g.extend([j+1]*2)
  order=rng.permutation(16);indices[u]=np.asarray(ids)[order];groups[u]=np.asarray(g)[order]
 np.savez_compressed(run/'sampling-schedule.npz',indices=indices,groups=groups)
 (run/'SAMPLING-FROZEN.json').write_text(json.dumps(dict(at=time.time(),algorithm='numpy.default_rng(1000); each batch8 uniform with replacement plus2 draws with replacement per event-class pool;shuffle16 within batch',updates=25000,batch16=True,total_samples=400000,group_counts=np.bincount(groups.ravel(),minlength=5).tolist(),schedule_sha256=digest(run/'sampling-schedule.npz'),window_sha256=digest(run/'windows.json'),unique_sampled_frames=int(len(np.unique(indices))),episode_padding_unchanged=True,chunk100_unchanged=True,loss_weighting=False,global_model_rng_untouched=True,per_frame_counts=np.bincount(indices.ravel(),minlength=159529).tolist()),indent=2)+'\n')

def schedule():
 global SCHEDULE
 if SCHEDULE is None:
  run=pathlib.Path(os.environ['WC4_EXECUTION_RUN']);f=run/'sampling-schedule.npz'
  assert digest(f)==json.loads((run/'SAMPLING-FROZEN.json').read_text())['schedule_sha256']
  with np.load(f) as a:SCHEDULE=(a['indices'].copy(),a['groups'].copy())
 return SCHEDULE

def audit_batch(batch,ordinal):
 indices,groups=schedule();assert 0<=ordinal<25000
 actual=batch['index'].detach().cpu().numpy().reshape(-1).astype(np.int64)
 assert np.array_equal(actual,indices[ordinal]),('Batch differs from frozen schedule',ordinal,actual.tolist())
 return dict(batch_ordinal=ordinal,indices=actual.tolist(),groups=groups[ordinal].tolist(),group_counts=np.bincount(groups[ordinal],minlength=5).tolist())

def install():
 import torch
 Original=torch.utils.data.DataLoader
 class ScheduledDataLoader(Original):
  def __init__(self,dataset,*args,**kwargs):
   assert len(dataset)==159529 and kwargs.get('batch_size')==16 and not kwargs.get('drop_last')
   indices,_=schedule();kwargs['shuffle']=False;kwargs['sampler']=indices.ravel().tolist()
   super().__init__(dataset,*args,**kwargs)
 torch.utils.data.DataLoader=ScheduledDataLoader
