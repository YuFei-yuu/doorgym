"""R20: reuse R19 point pools exactly, replace ONLY class2 by all compact_bypass frames."""
import pathlib,json,hashlib,copy,time,shutil
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3];R11=ROOT/'.repro/runs/wc4-fulltask-reaudit-r11-20261021'
OLD=ROOT/'.repro/runs/wc4-fulltask-act-r19-windows-s1000-20261023/FINAL'
RUN=ROOT/'.repro/runs/wc4-fulltask-act-r20-bypass-s1000-20261024'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 assert shutil.disk_usage(ROOT).free>=80*1024**3,'R20 start capacity below80GiB; preserve history and stop'
 assert not RUN.exists();prior=read(OLD/'sampling/windows.json');assert sha(OLD/'sampling/windows.json')=='9f447e277e02404855e643cb0d704b30d49cfb651360e429d7af8dbf38ba9b8d'
 assert sha(OLD/'sampling/windows.json')==read(OLD/'sampling/WINDOWS-FROZEN.json')['sha256']
 prov=read(R11/'unified-provenance.json')['episodes'];assert len(prov)==72
 labels=prior['labels'].copy();labels[1]='full_compact_bypass';events=[];segments=[];offset=0;pools=copy.deepcopy(prior['pools']);pools[1]=[];overlap=[]
 for ep,e in enumerate(prov):
  p=R11/'candidates'/f'episode_{ep:06d}.npz';assert sha(p)==e['sha256']==prior['sources'][str(p)]
  old=[x for x in prior['events'] if x['episode']==ep];assert len(old)==4
  with np.load(p,allow_pickle=True) as a:
   phase=a['subtask_index'].reshape(-1);names=list(a['phase_names']);n=len(phase);assert n==e['frames']
   ids=np.flatnonzero(phase==names.index('compact_bypass'));assert len(ids)>0 and np.all(np.diff(ids)==1)
   lo,hi=int(ids[0]),int(ids[-1]);assert lo==old[0]['frame'] and hi+1==old[1]['frame']
   assert names[int(phase[lo-1])]=='compact_fold' and names[int(phase[hi+1])]=='compact_rendezvous'
   # Expand anchor support only; future actions still come from the original full episode.
   indices=(ids+offset).tolist();pools[1].extend(indices)
   second=dict(episode=ep,seed=e['seed'],label=labels[1],frame=lo,index=offset+lo,time_s=old[0]['time_s'],source_step=old[0]['source_step'],phase='compact_bypass',lo=lo,hi_inclusive=hi,frames=len(indices),clipped=False,indices=indices,measured_anchor=old[0]['measured_anchor'],scope='every recorded compact_bypass frame, including both stage endpoints; segment start is not a point-window center',original_R19_second_anchor=old[1]['frame'],valid_future_min=min(100,n-hi),valid_future_max=min(100,n-lo),source_npz=str(p),source_sha256=e['sha256'])
   assert second['valid_future_min']==second['valid_future_max']==100
   new=[copy.deepcopy(old[0]),second,copy.deepcopy(old[2]),copy.deepcopy(old[3])];events.extend(new)
   sets=[set(x['indices']) for x in new];overlap.append(dict(episode=ep,pairwise={labels[i]+' / '+labels[j]:len(sets[i]&sets[j]) for i in range(4) for j in range(i+1,4)}))
   segments.append(dict(episode=ep,seed=e['seed'],episode_from_index=offset,episode_frames=n,lo=lo,hi_inclusive=hi,frames=len(ids),all_phase_frames_included=True,phase_boundary_does_not_truncate_chunk=True,valid_future_actions_per_anchor=100))
  offset+=n;print('full-bypass',ep,lo,hi,len(ids),flush=True)
 assert offset==159529
 for j in [0,2,3]:assert pools[j]==prior['pools'][j]
 out=dict(definitions={labels[0]:prior['definitions'][prior['labels'][0]],labels[1]:'all recorded frames whose original expert subtask_index is compact_bypass; first through last inclusive in every episode',labels[2]:prior['definitions'][prior['labels'][2]],labels[3]:prior['definitions'][prior['labels'][3]]},window='R19 class1/3/4 exact same +/-25-frame inclusive windows; class2 full contiguous compact_bypass. No stage-boundary truncation of chunk100; original episode padding only.',mixture=prior['mixture'],labels=labels,events=events,pools=pools,pool_sizes=list(map(len,pools)),overlaps=overlap,clipped_windows=0,unique_window_frames=len(set(sum(pools,[]))),sources=prior['sources'],episodes=72,frames=offset,frozen_at=time.time(),new_model_results_seen=False,only_changed_support_class=2,R19_windows_sha256=sha(OLD/'sampling/windows.json'),unchanged_classes=[1,3,4],segments=segments)
 RUN.mkdir();(RUN/'windows.json').write_text(json.dumps(out,indent=2)+'\n')
 (RUN/'WINDOWS-FROZEN.json').write_text(json.dumps(dict(sha256=sha(RUN/'windows.json'),frozen_at=time.time(),training_updates=0,model_forwards=0,scenes=0,capacity_bytes=shutil.disk_usage(ROOT).free,minimum_start_capacity_bytes=80*1024**3),indent=2)+'\n')
 from route_reference import freeze_route
 freeze_route(RUN,ROOT)
 print('SUMMARY',json.dumps({k:out[k] for k in ['pool_sizes','unique_window_frames','unchanged_classes']}),flush=True)
if __name__=='__main__':main()
