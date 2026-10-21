"""Observation/action near-duplicate diagnostic; no new policy features."""
import json,pathlib
import numpy as np
from scipy.spatial import cKDTree
R=pathlib.Path('/data/zyf/video2door-wc4-act/.repro/runs/wc4-fulltask-pilot-r8-20261021')
F=R/'dataset-check';z=np.load(F/'alias-features.npz')
state=z['state'];action=z['action'];images=z['image'];episode=z['episode'];local=z['local_frame'];phase=z['phase']
scale=np.array([.01,.01]+[.03]*7);X=state/scale
action_scale=np.array([.10,.15]+[.15]*7)
ends=np.empty(len(state),dtype=np.int64)
for ep in np.unique(episode):ends[episode==ep]=np.flatnonzero(episode==ep)[-1]
future_offsets=np.array([0,5,25,50,99])
future=action[np.minimum(np.arange(len(state))[:,None]+future_offsets,ends[:,None])]
pairs=set()
def query_group(target,source,k):
 if len(target)==0 or len(source)==0:return
 tree=cKDTree(X[target]);d,j=tree.query(X[source],k=min(k,len(target)),p=np.inf)
 d=np.asarray(d).reshape(len(source),-1);j=np.asarray(j).reshape(len(source),-1)
 a=np.broadcast_to(source[:,None],j.shape);b=target[j];keep=d<=1.
 keep&=(episode[a]!=episode[b])|(abs(local[a]-local[b])>=125)
 keep&=np.max(abs(future[a]-future[b])/action_scale,axis=(2,3))>=1.
 for u,v in zip(a[keep],b[keep]):pairs.add(tuple(sorted((int(u),int(v)))))
for ph in np.unique(phase):query_group(np.flatnonzero(phase==ph),np.flatnonzero(phase!=ph),4)
for ep in np.unique(episode):query_group(np.flatnonzero(episode==ep),np.flatnonzero(episode!=ep),8)
for ep in np.unique(episode):
 ids=np.flatnonzero(episode==ep)
 query_group(ids,ids,512)  # Same-phase bypass legs, excluding nearby times.
candidates=[];half=images.shape[1]//2
for a,b in pairs:
 d=(images[a].astype(float)-images[b])/255
 rmse=[float(np.sqrt(np.mean(d[:half]**2))),float(np.sqrt(np.mean(d[half:]**2)))]
 if max(rmse)<=.02:
  candidates.append(dict(a=a,b=b,episode_a=int(episode[a]),episode_b=int(episode[b]),frame_a=int(local[a]),frame_b=int(local[b]),
    phase_a=int(phase[a]),phase_b=int(phase[b]),state_abs_difference=abs(state[a]-state[b]).tolist(),action_abs_difference=abs(action[a]-action[b]).tolist(),
    future_probe_max_abs_difference=np.max(abs(future[a]-future[b]),axis=0).tolist(),
    downsampled_depth_rmse=rmse,ranking=max(rmse)))
candidates.sort(key=lambda c:c['ranking']);top=candidates[:12]
index=json.loads((R/'candidate-index.json').read_text())['episodes']
needed={}
for c in top:
 for suffix in ['a','b']:needed.setdefault(c['episode_'+suffix],set()).add(c['frame_'+suffix])
pixels={}
for ep,ids in needed.items():
 with np.load(index[ep]['candidate_npz'],allow_pickle=True) as raw:
  for view in ['front','wrist']:
   im=raw[view+'_masked_depth']
   for i in ids:pixels[(ep,i,view)]=im[i].copy()
confirmed=[]
for c in top:
 full=[]
 for view in ['front','wrist']:
  a=pixels[(c['episode_a'],c['frame_a'],view)];b=pixels[(c['episode_b'],c['frame_b'],view)]
  full.append(float(np.sqrt(np.mean(((a.astype(np.float32)-b)/255)**2))))
 c['full_depth_rmse']=full
 aa=np.minimum(c['a']+np.arange(100),ends[c['a']]);bb=np.minimum(c['b']+np.arange(100),ends[c['b']])
 c['chunk100_max_abs_difference']=np.max(abs(action[aa]-action[bb]),axis=0).tolist()
 c['immediate_action_significant']=bool(np.max(abs(action[c['a']]-action[c['b']])/action_scale)>=1)
 if max(full)<=.02:confirmed.append(c)
report=dict(kind='diagnostic_near_duplicates_not_a_proof_of_unlearnability',frames=len(state),candidate_state_pairs=len(pairs),coarse_near_pairs=len(candidates),full_resolution_confirmed_in_top12=len(confirmed),
 thresholds=dict(previous_base_command_abs=[.01,.01],joint_state_abs_rad=.03,dual_depth_rmse=.02,action_significant=[.10,.15]+[.15]*7,min_same_episode_time_gap_s=5.,future_probe_offsets_frames=future_offsets.tolist(),full_chunk_confirm_frames=100),
 scope='k4 per different phase, k8 per different episode, k512 within each episode with >=5s time gap, including same-phase bypass legs. Not exhaustive and not a robustness claim.',
 confirmed_pairs=confirmed,reviewed_pairs=top,new_policy_inputs=[],model_or_network_changes=False,
 conclusion='Potential input/action ambiguity found; report before deciding training or added history/labels.' if confirmed else 'No qualifying ambiguity in this bounded diagnostic; absence is not established.')
(F/'learnability-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
