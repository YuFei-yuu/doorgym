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
 (run/'SAMPLING-FROZEN.json').write_text(json.dumps(dict(at=time.time(),algorithm='numpy.default_rng(1000); each batch8 uniform with replacement plus2 draws with replacement per event-class pool;shuffle16 within batch',updates=25000,batch16=True,total_samples=400000,group_counts=np.bincount(groups.ravel(),minlength=5).tolist(),schedule_sha256=digest(run/'sampling-schedule.npz'),window_sha256=digest(run/'windows.json'),unique_sampled_frames=int(len(np.unique(indices))),episode_padding_unchanged=True,chunk100_unchanged=True,loss_weighting=False,global_model_rng_untouched=True,only_changed_support_class=2,second_class='complete compact_bypass over all72 episodes',per_frame_counts=np.bincount(indices.ravel(),minlength=159529).tolist()),indent=2)+'\n')

 future_coverage(run,indices,groups)

def future_coverage(run,indices,groups):
 w=json.loads((run/'windows.json').read_text());segments=w['segments'];ends=np.array([x['episode_from_index']+x['episode_frames'] for x in segments]);starts=np.r_[0,ends[:-1]]
 valid=np.minimum(100,ends[np.searchsorted(ends,indices,side='right')]-indices)
 assert (valid>0).all() and (valid[groups==2]==100).all()
 per_episode=[]
 for e in segments:
  ep=e['episode'];off=e['episode_from_index'];n=e['episode_frames'];mask=(indices>=off)&(indices<off+n)
  record=dict(episode=ep,seed=e['seed'],bypass_support_frames=e['frames'],groups={})
  for g in range(5):
   local=indices[mask&(groups==g)]-off;v=np.minimum(100,n-local);delta=np.zeros(n+1,dtype=np.int64);np.add.at(delta,local,1);np.add.at(delta,local+v,-1);covered=np.cumsum(delta[:-1])>0
   record['groups'][str(g)]=dict(samples=int(len(local)),unique_anchor_frames=int(len(np.unique(local))),valid_future_action_tokens=int(v.sum()),padding_tokens=int(100*len(local)-v.sum()),unique_episode_frames_supervised=int(covered.sum()),bypass_frames_supervised=int(covered[e['lo']:e['hi_inclusive']+1].sum()))
  assert record['groups']['2']['samples']>0;per_episode.append(record)
 old=pathlib.Path(__file__).resolve().parents[3]/'.repro/runs/wc4-fulltask-act-r19-windows-s1000-20261023/FINAL/sampling/sampling-schedule.npz'
 with np.load(old) as z:old_i=z['indices'];old_g=z['groups']
 coupling={str(g):dict(equal_indices=int(np.sum(indices[groups==g]==old_i[old_g==g])),total=int(np.sum(groups==g))) for g in range(5)}
 result=dict(episodes=72,frames=159529,complete_bypass_support_all72=True,all_actual_class2_samples_have100_valid_future_actions=True,supervision_chunk=100,stage_boundary_truncation=False,cross_episode_stitching=False,original_episode_padding_only=True,nominal_future_action_tokens=int(indices.size*100),valid_future_action_tokens=int(valid.sum()),padding_tokens=int(indices.size*100-valid.sum()),per_group_valid_future_tokens={str(g):int(valid[groups==g].sum()) for g in range(5)},actual_class2_supervised_bypass_frames=sum(e['groups']['2']['bypass_frames_supervised'] for e in per_episode),total_bypass_support_frames=len(w['pools'][1]),per_episode=per_episode,R19_rng_schedule_coupling=coupling,R19_group_position_matches=int(np.sum(groups==old_g)),note='Anchor coverage and future-supervision coverage are separate; actual category2 may miss early anchors even though future intervals cover most/all phase frames. No action sequence truncated by stage/window boundary.')
 (run/'future-supervision-coverage.json').write_text(json.dumps(result,indent=2)+'\n')

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
