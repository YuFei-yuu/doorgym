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
def freeze_schedule(run):
 w=json.loads((run/'windows.json').read_text());assert digest(run/'windows.json')=='d0cb01abcde64acc13cf870acf547d84cc7fa8429e257edfe6fc1a365c9ebedb'
 pools=[np.asarray(p,dtype=np.int64) for p in w['pools']];rng=np.random.default_rng(1000)
 with np.load(OLD/'sampling-schedule.npz') as z:prior_i=z['indices'];prior_g=z['groups']
 assert digest(OLD/'sampling-schedule.npz')=='6dac28a9ca89a19c7660af926d7f0db6d2d5cbc541189a08f217f24dd530bdc4'
 for k in range(25000):
  ii,gg=draw(rng,pools);assert np.array_equal(ii,prior_i[k]) and np.array_equal(gg,prior_g[k]),('R20 RNG prefix mismatch',k)
 states={'after_R20_075000':rng.bit_generator.state};indices=np.empty((NEW_UPDATES,16),np.int64);groups=np.empty_like(indices)
 for k in range(NEW_UPDATES):
  indices[k],groups[k]=draw(rng,pools)
  if k+1 in [25000,75000,125000]:states[f'after_{START+k+1:06d}']=rng.bit_generator.state
 np.savez_compressed(run/'sampling-schedule.npz',indices=indices,groups=groups)
 equal=[bool(np.array_equal(indices[k:k+25000],prior_i)) for k in range(0,NEW_UPDATES,25000)];assert not any(equal)
 write(run/'SAMPLING-RNG-CONTINUATION.json',dict(seed=1000,numpy_version=np.__version__,bit_generator=type(rng.bit_generator).__name__,prefix_batches_verified_exact=25000,prefix_source_sha256=digest(OLD/'sampling-schedule.npz'),new_batches=NEW_UPDATES,new_plan_25k_blocks_equal_old_plan=equal,states=states,rule='Reconstruct seed1000 algorithm and verify every originalR20 row/group; continue resulting PCG64 state, never loop the old list. This is sampling-plan RNG only; author DataLoader/prefetch cursor not serialized.'))
 write(run/'SAMPLING-FROZEN.json',dict(at=time.time(),algorithm='Same R20 numpy choice with replacement; each batch8 globaluniform+2 per four frozen pools, then permutation16; continue verified25k prefix',updates=NEW_UPDATES,start_update=START,end_update=200000,batch_size=16,total_samples=2000000,group_counts=np.bincount(groups.ravel(),minlength=5).tolist(),schedule_sha256=digest(run/'sampling-schedule.npz'),window_sha256=digest(run/'windows.json'),rng_continuation_sha256=digest(run/'SAMPLING-RNG-CONTINUATION.json'),unique_sampled_frames=int(len(np.unique(indices))),episode_padding_unchanged=True,chunk100_unchanged=True,loss_weighting=False,per_frame_counts=np.bincount(indices.ravel(),minlength=159529).tolist()))
 segments=w['segments'];ends=np.array([e['episode_from_index']+e['episode_frames'] for e in segments]);ep=np.searchsorted(ends,indices,side='right');valid=np.minimum(100,ends[ep]-indices);assert (valid[groups==2]==100).all()
 per_ep=[]
 for e in segments:
  off=e['episode_from_index'];n=e['episode_frames'];mask=(indices>=off)&(indices<off+n);rec=dict(episode=e['episode'],frames=n,bypass_support=e['frames'],groups={})
  for g in range(5):
   a=indices[mask&(groups==g)]-off;v=np.minimum(100,n-a);delta=np.zeros(n+1,np.int64);np.add.at(delta,a,1);np.add.at(delta,a+v,-1);covered=np.cumsum(delta[:-1])>0
   rec['groups'][str(g)]=dict(samples=len(a),unique_anchors=int(len(np.unique(a))),valid_future_tokens=int(v.sum()),padding_tokens=int(100*len(a)-v.sum()),bypass_frames_supervised=int(covered[e['lo']:e['hi_inclusive']+1].sum()))
  per_ep.append(rec)
 stages={}
 for step,a,b in [(100000,0,25000),(150000,25000,75000),(200000,75000,125000)]:
  stages[str(step)]=dict(new_updates_in_interval=b-a,cumulative_R21_updates=b,samples_in_interval=(b-a)*16,group_counts=np.bincount(groups[a:b].ravel(),minlength=5).tolist(),valid_future_tokens=int(valid[a:b].sum()),padding_tokens=int((b-a)*1600-valid[a:b].sum()))
 write(run/'future-supervision-coverage.json',dict(episodes=72,frames=159529,complete_bypass_support_all72=True,bypass_support_frames=31114,valid_future_tokens=int(valid.sum()),padding_tokens=int(valid.size*100-valid.sum()),per_group_valid_future_tokens={str(g):int(valid[groups==g].sum()) for g in range(5)},class2_all100_valid=True,per_episode=per_ep,stages=stages,scope='Frozen new-plan accounting; actual training requires completed batch index equality. No stage/window truncation, no episode stitching.'))
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
   assert len(dataset)==159529 and kwargs.get('batch_size')==16 and not kwargs.get('drop_last');ii,_=schedule();kwargs['shuffle']=False;kwargs['sampler']=ii.ravel().tolist();super().__init__(dataset,*args,**kwargs)
 torch.utils.data.DataLoader=ScheduledDataLoader
