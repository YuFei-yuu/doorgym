"""R20 distribution; reconstruct/verify its25k prefix, then continue PCG64 for125k fresh batches."""
import pathlib,os,json,hashlib,time
import numpy as np
SCHEDULE=None;NEW_UPDATES=125000;START=75000
ROOT=pathlib.Path(__file__).resolve().parents[3];OLD=ROOT/'.repro/runs/wc4-fulltask-act-r20-bypass-s1000-20261024'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def draw(rng,pools):
 ids=list(rng.integers(0,159529,8));g=[0]*8
 for j,pool in enumerate(pools):ids.extend(rng.choice(pool,2));g.extend([j+1]*2)
 order=rng.permutation(16);return np.asarray(ids,dtype=np.int64)[order],np.asarray(g,dtype=np.int64)[order]
def schedule():
 global SCHEDULE
 if SCHEDULE is None:
  r=pathlib.Path(os.environ['WC4_EXECUTION_RUN']);p=r/'sampling-schedule.npz';assert digest(p)==json.loads((r/'SAMPLING-FROZEN.json').read_text())['schedule_sha256']
  with np.load(p) as a:SCHEDULE=a['indices'].copy(),a['groups'].copy()
 return SCHEDULE
def audit_batch(batch,ordinal):
 ii,gg=schedule();assert 0<=ordinal<NEW_UPDATES;actual=batch['index'].detach().cpu().numpy().reshape(-1).astype(np.int64);assert np.array_equal(actual,ii[ordinal]),('Frozen sample mismatch',ordinal)
 return dict(batch_ordinal=ordinal,R20_seeded_stream_batch_ordinal=25000+ordinal,indices=actual.tolist(),groups=gg[ordinal].tolist(),group_counts=np.bincount(gg[ordinal],minlength=5).tolist())
def install():
 import torch
 Original=torch.utils.data.DataLoader
 class ScheduledDataLoader(Original):
  def __init__(self,dataset,*args,**kwargs):
   assert len(dataset)==159529 and kwargs.get('batch_size')==16 and not kwargs.get('drop_last');ii,_=schedule();kwargs['shuffle']=False;kwargs['sampler']=ii[75155:].ravel().tolist();super().__init__(dataset,*args,**kwargs)
 torch.utils.data.DataLoader=ScheduledDataLoader
