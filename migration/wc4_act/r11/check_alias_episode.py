"""Bounded raw-observation ambiguity search, separated by action horizon.

Read-only CPU analysis. Stages are used for audit search groups, never inputs.
"""
import json, pathlib, time
import numpy as np
from scipy.spatial import cKDTree

R=pathlib.Path('/data/zyf/video2door-wc4-act/.repro/runs/wc4-fulltask-reaudit-r11-20261021')
from common import RUN
F=RUN/'dataset-check';z=np.load(F/'alias-features.npz')
state=z['state'];action=z['action'];images=z['image'];episode=z['episode'];local=z['local_frame'];phase=z['phase']
scale=np.array([.01,.01]+[.03]*7);action_scale=np.array([.10,.15]+[.15]*7)
X=state/scale;ends=np.empty(len(state),dtype=np.int64)
for ep in np.unique(episode):ends[episode==ep]=np.flatnonzero(episode==ep)[-1]
offsets=np.array([0,5,10,24,25,50,99]);future=action[np.minimum(np.arange(len(state))[:,None]+offsets,ends[:,None])]
half=images.shape[1]//2
small=np.concatenate([images[:,:half:4],images[:,half::4]],axis=1).astype(np.float32)/255
split=small.shape[1]//2
top={h:[] for h in [1,25,100]};episode_best={h:{} for h in [1,25,100]};stats=dict(query_records=0,probe_action_different_records=0,coarse_image_near_records=0)
start=time.perf_counter()


def consider(a,b):
    if not len(a):return
    # These are query records and may duplicate pairs across search groups.
    stats['query_records']+=len(a)
    for begin in range(0,len(a),1024):
        aa=a[begin:begin+1024];bb=b[begin:begin+1024]
        probe=np.max(abs(future[aa]-future[bb])/action_scale,axis=2)
        valid_probe=offsets[None,:]<np.minimum(ends[aa]-aa+1,ends[bb]-bb+1)[:,None]
        diff=np.max(np.where(valid_probe,probe,0.),axis=1);take=diff>=1
        aa=aa[take];bb=bb[take];stats['probe_action_different_records']+=len(aa)
        if not len(aa):continue
        delta=small[aa]-small[bb]
        distances=np.stack([np.sqrt(np.mean(delta[:,:split]**2,axis=1)),np.sqrt(np.mean(delta[:,split:]**2,axis=1))],axis=1)
        take=np.max(distances,axis=1)<=.02;aa=aa[take];bb=bb[take];distances=distances[take]
        stats['coarse_image_near_records']+=len(aa)
        if not len(aa):continue
        ai=np.minimum(aa[:,None]+np.arange(100),ends[aa,None]);bi=np.minimum(bb[:,None]+np.arange(100),ends[bb,None])
        d=abs(action[ai]-action[bi])
        common_valid=np.arange(100)[None,:]<np.minimum(ends[aa]-aa+1,ends[bb]-bb+1)[:,None]
        d=np.where(common_valid[:,:,None],d,0.)
        for h in [1,25,100]:
            maxima=np.max(d[:,:h],axis=1);significant=np.max(maxima/action_scale,axis=1)>=1
            for j in np.flatnonzero(significant):
                u,v=sorted((int(aa[j]),int(bb[j])));rank=float(np.max(distances[j]));key=(u,v)
                c=dict(key=key,a=u,b=v,ranking=rank,coarse_depth_rmse=distances[j].tolist())
                ep=int(episode[u])
                old=episode_best[h].get(ep)
                if old is None or (rank,u,v)<(old['ranking'],old['a'],old['b']):episode_best[h][ep]=c.copy()
                top[h].append(c);top[h].sort(key=lambda x:(x['ranking'],x['a'],x['b']));del top[h][12:]


def query(target,source,k):
    if not len(target) or not len(source):return
    tree=cKDTree(X[target]);kk=min(k,len(target))
    # Limit temporary buffers while retaining all source states in each group.
    for begin in range(0,len(source),1024):
        src=source[begin:begin+1024];distance,neighbor=tree.query(X[src],k=kk,p=np.inf)
        distance=np.asarray(distance).reshape(len(src),kk);neighbor=np.asarray(neighbor).reshape(len(src),kk)
        aa=np.broadcast_to(src[:,None],neighbor.shape);bb=target[neighbor]
        keep=(distance<=1)&(aa<bb)&((episode[aa]!=episode[bb])|(abs(local[aa]-local[bb])>=125))
        consider(aa[keep],bb[keep])


for ep in np.unique(episode):
    ids=np.flatnonzero(episode==ep);query(ids,ids,512)
    print('same-episode',int(ep),stats,flush=True)

# R8's predeclared nominal pair remains a longitudinal diagnostic anchor.
anchor=(int(np.flatnonzero(episode==0)[0])+703,int(np.flatnonzero(episode==0)[0])+832)
unique={tuple(c['key']) for group in episode_best.values() for c in group.values()};unique.add(anchor)
index=json.loads((R/'candidate-index.json').read_text())['episodes'];needed={}
for a,b in unique:
    for i in [a,b]:needed.setdefault(int(episode[i]),set()).add(int(local[i]))
pixels={}
for ep,ids in needed.items():
    with np.load(index[ep]['candidate_npz'],allow_pickle=True) as raw:
        for view in ['front','wrist']:
            im=raw[view+'_masked_depth']
            for i in ids:pixels[(ep,i,view)]=im[i].copy()
confirmed={}
for a,b in sorted(unique):
    full=[]
    for view in ['front','wrist']:
        p=pixels[(int(episode[a]),int(local[a]),view)];q=pixels[(int(episode[b]),int(local[b]),view)]
        full.append(float(np.sqrt(np.mean(((p.astype(np.float32)-q)/255)**2))))
    ai=np.minimum(a+np.arange(100),ends[a]);bi=np.minimum(b+np.arange(100),ends[b]);d=abs(action[ai]-action[bi])
    common_valid=min(100,int(ends[a]-a+1),int(ends[b]-b+1));d[common_valid:]=0.
    maxima={str(h):np.max(d[:h],axis=0).tolist() for h in [1,25,100]}
    confirmed[(a,b)]=dict(a=a,b=b,episode_a=int(episode[a]),episode_b=int(episode[b]),frame_a=int(local[a]),frame_b=int(local[b]),phase_a=int(phase[a]),phase_b=int(phase[b]),
        state_abs_difference=abs(state[a]-state[b]).tolist(),full_depth_rmse=full,near_full_depth=max(full)<=.02,
        common_valid_future_frames=common_valid,maximum_action_abs_difference_by_horizon=maxima,significant_by_horizon={str(h):bool(np.max(np.asarray(maxima[str(h)])/action_scale)>=1) for h in [1,25,100]})
groups={str(h):[confirmed[tuple(c['key'])] for ep,c in sorted(episode_best[h].items())] for h in [1,25,100]}
report=dict(kind='bounded_ambiguity_diagnostic_not_proof_of_unlearnability',frames=len(state),episodes=len(index),elapsed_s=time.perf_counter()-start,query_statistics=stats,
    horizons=dict(current_action=[0],first25=list(range(25)),chunk100=list(range(100))),
    thresholds=dict(state_abs=[.01,.01]+[.03]*7,dual_depth_normalized_rmse=.02,action_significant=action_scale.tolist()),
    search='Within every one of the admitted episodes, all source states, k512 same-episode neighbors with >=5s gap. Probe prefilter0,5,10,24,25,50,99; full shared-valid horizon comparison. Retain nearest qualifying pair per episode per horizon; full original depth confirmation. Bounded/nonexhaustive; query-record counts may repeat.',
    reviewed_by_horizon=groups,full_resolution_confirmed_significant_counts={str(h):sum(c['near_full_depth'] and c['significant_by_horizon'][str(h)] for c in groups[str(h)]) for h in [1,25,100]},
    r8_anchor=confirmed[anchor],new_policy_inputs=[],network_changes=False,training_performed=False,
    limitations=['Near inputs are not mathematically identical; no inference or trainability claim.','Current, first25 and chunk100 must be interpreted separately; padded future slots are excluded from all action-difference tests.','No difficult frames were removed and no stage/history inputs were added.'])
(F/'learnability-by-episode.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
